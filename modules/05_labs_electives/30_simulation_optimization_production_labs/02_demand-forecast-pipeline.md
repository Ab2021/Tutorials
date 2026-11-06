# Demand Forecast Pipeline

> Module 30 — Simulation, Optimization, and Real-Time Production Labs
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: basic Python, CSV, time-series forecasting and chronological evaluation; [environment lab](01_environment-repo.md).
> Version 2026.5

## 1. Why this topic exists

A replenishment planner needs an estimate of future customer demand before deciding what to order.
An operations team needs to know whether that estimate comes from a repeatable process rather than a spreadsheet copied from an unknown week.
A forecasting model alone does not solve either problem.
The pipeline must establish which observations were available, check their meaning, compare candidates with a simple benchmark, and publish a traceable result.

Suppose a distributor ships a product weekly.
The purchasing lead must place an order before the next week begins.
Late data, duplicated files, and stockouts can all change the interpretation of observed shipments.
A forecast that looks accurate after the fact may have used corrections unavailable when the decision was made.
A low error measured on a convenient slice can hide poor performance during promotions or for small-volume products.
An unpublished forecast has no operational owner; an unversioned published number cannot be audited.

The business outcome is a better-informed inventory decision, not an assertion that a model automatically reduces inventory or increases service.
The forecast is an estimate of unconstrained demand under stated assumptions.
A plan is an action proposed in response to that estimate.
A target is a desired performance level.
A commitment is an agreed obligation.
These are distinct even when a system stores them in adjacent columns.

This lab creates one weekly forecast for one synthetic product and one synthetic location.
It intentionally limits the models to transparent baselines.
Its output is a local file, not a production deployment or a real business result.
You will inspect an ingestion contract, reproduce chronological backtests, compare an error benchmark, and publish an artifact with a decision trail.
A later inventory policy can consume a forecast, but inventory decisions also require lead time, costs, and a service objective.

## 2. Learning outcomes

After completing this topic you can:

1. Define the grain, timing, and meaning of a demand observation; exercise 1 checks the contract.
2. Validate dates, keys, values, cadence, and data availability; exercise 2 diagnoses invalid inputs.
3. Construct expanding-window, one-step chronological backtests; exercise 3 reconstructs folds.
4. Compute mean absolute error, signed bias, and weighted absolute percentage error; exercise 4 checks arithmetic.
5. Select a model using a declared metric and deterministic tie rule; exercise 5 challenges the selection.
6. Publish a versioned local forecast with provenance and distinguish it from a plan; exercise 6 reviews the artifact.
7. Translate model errors into a decision-oriented cost scenario without claiming realized savings; exercise 7 evaluates an asymmetric loss.
8. Design production controls for leakage, revisions, and monitoring; exercise 8 supplies a review rubric.

## 3. The core mental model

```text
source events / approved snapshot
              |
              v
 contract: grain + availability + target definition
              |
              v
 validation -----> quarantine / stop with diagnostics
              |
              v
 ordered historical observations
              |
              v
 rolling origins: train through cutoff -> predict next week -> score later
              |
              v
 baseline <---- same folds and scoring ----> candidate
              |                                |
              +------------ select ------------+
                             |
                             v
                 refit using permitted history
                             |
                             v
          publish forecast + lineage + selection evidence
                             |
                             v
         planner review -> plan / commitment -> outcomes
                             |
                             v
            actuals maturation + monitoring + revision
```

Each arrow transfers a contract, not merely a file.
At ingestion the contract says which facts are present and when they could have been known.
At evaluation it says which dates may influence each prediction.
At publication it says what the number represents and which run produced it.
The final arrow compares forecasts with matured outcomes, not with a convenient future revision silently substituted for the historical truth.

Chronology is a causal boundary.
At each origin, train only on rows available by that origin.
Predict the future bucket, then compare with the observation for that bucket.
Repeated origins expose the candidates to several conditions, although four origins do not establish stable performance.

A benchmark is a decision alternative.
Here the latest observed value predicts the next week.
A candidate uses the average of the last three observations.
Both use exactly the same four origins and have no fitted parameters other than the historical data they read.
The sandbox chooses the lowest mean absolute error (MAE); ties go to the naive benchmark.
This selection rule is chosen before evaluating the fixture.

The local publication is a handoff, not a durable registry.
Its data, model name, evaluation, issue date, target date, and scope are inspectable.
A real system also needs access controls, monitoring, a durable audit trail, reproducible code and dependencies, approval, and recovery.

## 4. Definitions and vocabulary

| Term | Definition and units | Common misuse |
|---|---|---|
| Demand | Units customers would request in a weekly bucket | Assuming fulfilled shipments always equal unconstrained demand |
| Observation | One recorded value for a specific product, location and week, in units | Treating the value as available before it was recorded |
| Grain | Attributes uniquely identifying a measurement | Joining data at a different grain without aggregation |
| Bucket | Interval over which units are summed | Confusing the Monday start date with a daily measurement |
| Weekly start | Monday identifying a seven-day period | Using local and UTC dates interchangeably |
| Snapshot | Frozen set of values and their as-of time | Replacing a historical snapshot with revised current values |
| Availability time | Earliest time an observation can be used | Assuming event time equals processing time |
| Origin | Cutoff at which a prediction is issued | Using actuals from the target week |
| Horizon | Number of buckets between origin and predicted bucket | Treating one-week and four-week errors as comparable |
| Training window | Observations available to the candidate at an origin | Including the target observation |
| Expanding window | Training history that grows as origins advance | Forgetting that old history can become unrepresentative |
| Backtest | Historical replay of predictions and actuals | Randomly splitting a time series |
| Fold | One origin, its eligible training history and target | Counting correlated folds as independent experiments |
| Leakage | Information not available at issue time enters a prediction or choice | Treating a small amount of future data as harmless |
| Naive forecast | Next week's predicted units equal the latest available week's units | Assuming it is always inferior |
| Moving average | Arithmetic mean of a specified number of most recent observations | Averaging across an unknown window |
| Actual | Observed target used for evaluation, in units per week | Equating a censored sales record with true demand |
| Error | Actual minus forecast, in units | Switching sign convention halfway through reporting |
| Absolute error | Magnitude of error, in units | Assuming absolute error reveals overforecast direction |
| MAE | Sum of absolute errors divided by number of forecasts, in units | Averaging incomparable horizons without stating weights |
| WAPE | Sum of absolute errors divided by sum of absolute actuals, dimensionless | Dividing by zero or calling it average percentage error |
| Bias | Mean signed error, in units; positive here means underforecast | Mistaking cancellation for low total error |
| Model selection | Choosing a candidate using a declared criterion | Calling the selection score an unbiased final performance estimate |
| Holdout | Later, untouched period used for final assessment | Repeatedly tuning against the same holdout |
| Forecast issue date | Date on which forecast is available for use | Confusing it with the target week's start |
| Publication | Making a forecast and its metadata available to a consumer | Treating a temporary local file as a governed service |
| Provenance | Evidence linking output to input and process | Recording only a model display name |
| WAPE denominator | Sum of absolute actual units for the scored slice | Dividing by a net total that can cancel |
| Baseline | Reference method evaluated on the same observations | Comparing against a stale or differently scoped benchmark |
| Censoring | Recorded sales are limited by available stock | Treating missing sales during a stockout as no demand |
| Service level | A defined fulfillment or availability measure | Assuming all service definitions are equivalent |
| Forecast override | Human or system change to a model forecast | Replacing the original without reason or audit trail |

## 5. The mechanics, step by step

### 5.1 Frame the decision

1. Identify the decision, owner, and cutoff.
   In this lab the planner considers a one-week-ahead estimate after the last historical weekly observation is available.
   The sandbox assumes the weekly actual is available at the beginning of the following Monday.
   An actual business must check this timing with source-system evidence.
2. Name the target and unit.
   Here the target is weekly demand in nonnegative units for one product at one location.
   No returns, promotions, stockouts, or late corrections are represented.
3. Specify the horizon and calendar.
   Weeks start on Monday, represent seven calendar days, and use calendar dates rather than timestamps.
   The target starts exactly seven days after the final historical week.
4. Separate the decision from the prediction.
   A forecast of 30 units is not an order for 30 units.
   The order depends on on-hand stock, on-order quantity, lead time, constraints, and policy.
5. Decide the publication and review schedule before examining results.
   The local exercise writes one forecast when the script runs.
   A production scheduler must distinguish successful execution from a reviewed and accepted publication.

### 5.2 Define the input contract

6. Declare a schema.
   The fixture uses `week_start`, `sku`, `location`, and `units`.
   Each row is one SKU-location-week and each value refers to the same weekly measure.
7. Declare identifiers.
   The lab requires `SKU-1` and `DC-1` on every row.
   This narrow contract prevents a second series from being accidentally concatenated into the first.
8. Define uniqueness.
   A row key consists of the product, location, and week start.
   Duplicate keys stop the run even if their values agree.
9. Define completeness.
   The weeks must be consecutive Mondays.
   Missing weeks stop the run rather than being silently interpreted as zero demand.
10. Define value constraints.
    Values must parse as finite nonnegative numbers.
    Negative returns require a separate representation or an explicitly different target definition.
11. Define availability.
    The fixture is a frozen historical snapshot and the script checks that every completed week precedes the issue date.
    It cannot verify when a real source first made each row available.
12. Define the unit and aggregation rules.
    The fixture is already aggregated in units per week.
    Real inputs need explicit handling for pack sizes, duplicate events, cancellations, and source corrections before aggregation.

### 5.3 Ingest without losing evidence

13. Save the exact source bytes.
    The example hashes the fixture bytes using SHA-256 and carries the digest into the published metadata.
    A checksum detects a byte-level difference; it does not prove source truth or authorize use.
14. Parse with a standard CSV reader.
    Verify column names before consuming rows.
    Do not infer that an empty field is zero.
15. Preserve the raw snapshot separately from cleaned records.
    The exercise creates a small local fixture and reads it unchanged.
    A governed implementation would retain an immutable source snapshot and processing log subject to retention rules.
16. Sort by date only after checking row keys.
    Sorting solves ordering, not duplication or missing data.
17. Fail the run on invalid input.
    Diagnostics should identify the violated rule without publishing a partial replacement.
    Route a real rejected batch to an owned remediation process.

### 5.4 Validate temporal integrity

18. Check ISO dates and Monday starts.
    A valid date in a different cadence can yield plausible but incorrect backtests.
19. Verify that dates increase in seven-day steps after sorting.
    A gap is not equivalent to a zero, because an absent record may indicate pipeline failure.
20. Check that each week has finished before the issue date.
    For a Monday start, the next Monday is the first possible date after the full seven-day period.
21. Require enough rows for the longest lookback and all requested origins.
    This example needs three historical values at the earliest fold and four distinct future actuals.
22. For production, store event time, recorded time, and revision time.
    A row dated in the past may still have arrived after an origin.
    The synthetic file has no recorded-time field, so its as-of validity is an explicit assumption rather than a proven property.
23. Decide how to handle censoring and known anomalies.
    Investigate a stockout, substitution, or data outage before using the observation as an unconstrained-demand label.
    Never quietly erase difficult observations just because they worsen accuracy.

### 5.5 Set up a fair chronological comparison

24. Freeze candidates and scoring rule before comparing results.
    The candidates are latest value and three-observation mean.
    All folds use one-step-ahead forecasts and identical actuals.
25. Select four successive origins.
    At each fold the training prefix ends immediately before its target week.
    Earlier folds cannot read target-week data or later folds.
26. Use an expanding prefix.
    The naive forecast reads its last value.
    The moving average reads its last three values.
    Older rows remain available but neither candidate needs them to calculate a prediction.
27. Score only the predicted target.
    Store origin, target date, actual, each forecast, and each signed error.
    Those rows explain aggregate metrics and expose a single dominant miss.
28. Compute each model's MAE and bias.
    Compute WAPE only when the sum of absolute actual units is positive.
    If that denominator is zero, report WAPE as undefined rather than zero or infinity.
29. Choose the lower MAE.
    If scores are equal, choose naive, which avoids claiming additional complexity is beneficial when evidence is tied.
    Record all scores, not just the winning score.
30. Recognize selection optimism.
    The four backtest folds choose the model, so they are not an independent estimate of its final generalization.
    A real deployment should reserve a later untouched period or use nested evaluation and periodically reassess.

### 5.6 Publish an interpretable artifact

31. Recalculate the selected method from all eligible history.
    The target is the week starting on the issue date.
    The selected naive model emits the latest actual, not the last backtest prediction.
32. Record the issue date and target date separately even when the date values match.
    They describe different roles.
33. Include series identifiers, horizon, forecast units, selected model, scores, and input digest.
    The exercise stores them in one JSON object with a schema version.
34. Write a temporary file and replace the target only after successful validation and calculation.
    This protects the previous file from ordinary pre-publication exceptions on the same local filesystem.
    It is not a distributed transaction or a guarantee of persistence after a power failure.
35. Tell the consumer what the file does not contain.
    It contains no prediction interval, plan, order quantity, override, approval, or actual realization.
36. Separate monitoring from selection.
    Compare published forecasts to matured actuals by issue date and horizon.
    Track missing publications, data freshness, aggregate bias, absolute error, and business outcomes.
37. Establish ownership and correction procedures.
    Record who may override, why, when, and which version a consumer used.
    A revised forecast should not erase the originally issued forecast.

### 5.7 Translate forecast evidence into a decision

38. Define the decision cost model with the operational owner.
    Underforecast and overforecast may have different consequences.
    Unit margin, shortage exposure, inventory holding cost, substitution, and capacity constraints are distinct quantities.
39. Reconstruct the inventory state at each historical origin if simulating orders.
    Demand errors alone are not realized shortage or inventory costs.
40. Compare a model-based policy and a baseline policy on the same demand and assumptions.
    Report both accuracy and modeled business impact, with a sensitivity analysis.
41. Review slice performance.
    Aggregate scores can hide systematic misses for a location, a product class, a high-demand week, or a particular lead-time horizon.
42. Keep model output and human override separate in the audit trail.
    A lower error after overrides does not by itself identify whether the model or the human caused the improvement.

### 5.8 Operational review cards

Use these cards to review the interfaces around the simple runnable pipeline.
Each card identifies an observable situation, the risk, an action, the evidence to inspect, and a limitation.
They are design guidance, not claims that the sandbox implements every control.

**1. Source owner.**
Situation: Two extracts disagree for the same completed week.
Risk: Choosing whichever has the larger value can double count a correction.
Action: Designate one system of record and preserve both original extracts.
Evidence: Reconcile by key, source timestamp, and documented revision number.
Limit: Do not publish until the discrepancy has an accountable resolution.

**2. File arrival.**
Situation: A scheduled file has not arrived at the expected cutoff.
Risk: An old file can look syntactically valid while representing stale demand.
Action: Compare source completion and ingestion times with the service-level objective.
Evidence: Expose the missing delivery to the data owner and block normal publication.
Limit: A fallback forecast needs separate approval and explicit stale-data labeling.

**3. Partial delivery.**
Situation: The source file arrives but covers only some locations.
Risk: Row-level checks can pass even when an entire series is absent.
Action: Compare expected series coverage against a versioned assortment roster.
Evidence: Report missing and unexpected product-location combinations.
Limit: The sandbox checks only its single expected series, not a multi-series roster.

**4. Duplicate batch.**
Situation: The transport retries an already ingested source file.
Risk: Appending it again duplicates all weekly values.
Action: Identify deliveries with stable batch IDs and content digests.
Evidence: Check the prior ingestion log before accepting a retry.
Limit: A content digest alone cannot determine whether a legitimate correction occurred.

**5. Reordered rows.**
Situation: A valid source lists weeks in reverse chronological order.
Risk: Using file order as time order changes training and target alignment.
Action: Parse dates and sort only after checking keys.
Evidence: Inspect first and last dates plus the count and cadence.
Limit: Sorting must not hide duplicate or missing observations.

**6. Mixed series.**
Situation: A second location appears inside the fixture.
Risk: A single-series moving average could mix unrelated demand histories.
Action: Validate product and location against the expected series.
Evidence: Reject the batch or explicitly partition by the complete key.
Limit: The example rejects the extra series instead of generalizing automatically.

**7. Duplicate value.**
Situation: Two identical rows share the same product, location, and date.
Risk: Equal quantities do not establish that only one underlying event exists.
Action: Treat the repeated aggregate key as a contract violation.
Evidence: Trace the duplicate back to the source batch and aggregation step.
Limit: A documented event-level deduplication rule belongs upstream of weekly aggregation.

**8. Malformed schema.**
Situation: A producer renames the units column without notifying consumers.
Risk: Positional parsing might interpret a different field as demand.
Action: Validate exact column names and their approved version.
Evidence: Save the rejected header and producer version for diagnosis.
Limit: Negotiate schema evolution rather than guessing a replacement column.

**9. Extra column.**
Situation: A source adds a promotion flag to the weekly file.
Risk: Silent acceptance may hide a breaking contract or an unused feature.
Action: Apply the agreed schema-evolution policy before deployment.
Evidence: Document whether the field is optional and when it is available.
Limit: This lab rejects all unexpected columns and does not use external features.

**10. Empty field.**
Situation: A weekly quantity cell contains no value.
Risk: Replacing an empty string by zero can conceal a missing measurement.
Action: Reject the value and request the upstream reason.
Evidence: Distinguish measured zero, unavailable data, and not-applicable records.
Limit: Only an approved imputation policy may change the observation.

**11. Invalid numeric value.**
Situation: A row contains a nonfinite number such as NaN or infinity.
Risk: The value can propagate into scores and produce unusable JSON.
Action: Require finite numeric quantities before any model computation.
Evidence: Check rejected-row diagnostics and repair the upstream calculation.
Limit: Do not turn an invalid number into a plausible forecast with a default.

**12. Negative quantity.**
Situation: A weekly record has negative net sales after returns.
Risk: Gross demand and returns have different operational meanings.
Action: Separate return events or redefine and approve the target.
Evidence: Reconcile gross sales, return units, and net values by period.
Limit: The sandbox deliberately rejects negative units.

**13. Unit conversion.**
Situation: A supplier changes its feed from individual units to cases.
Risk: An unannounced pack-size change creates an artificial level shift.
Action: Map each effective-dated source unit to the canonical forecast unit.
Evidence: Reconcile converted totals against source invoices or master data.
Limit: A single constant conversion is wrong when pack sizes change over time.

**14. Calendar convention.**
Situation: A source uses Sunday-start weeks while planning uses Monday-start weeks.
Risk: Labels with similar dates refer to different seven-day intervals.
Action: Reaggregate daily events onto the approved business calendar.
Evidence: Compare bucket boundaries and total units across the conversion.
Limit: A weekly aggregate cannot always be exactly reallocated without daily detail.

**15. Timezone boundary.**
Situation: Transactions near midnight cross a business-date boundary.
Risk: UTC and local aggregation can place the same event in different weeks.
Action: Define source and destination timezones before daily aggregation.
Evidence: Check sample events around midnight and daylight-saving transitions.
Limit: The fixture contains date labels only and cannot demonstrate this conversion.

**16. Missing period.**
Situation: A series jumps directly from one Monday to a date 14 days later.
Risk: An omitted week can be a failed feed or an unreported zero.
Action: Stop and compare the missing period with the source system.
Evidence: Record the reason and obtain an approved corrected snapshot.
Limit: Do not insert zero merely to satisfy a continuity check.

**17. True zero.**
Situation: A verified active product had zero customer demand for a week.
Risk: Dropping its zero row makes the average systematically too high.
Action: Retain the explicit zero and its matching weekly key.
Evidence: Use source coverage and product activity records to corroborate it.
Limit: Sparse and intermittent demand may need models beyond these baselines.

**18. Stockout week.**
Situation: The product was unavailable for several days of a target week.
Risk: Recorded sales can underestimate the demand customers would have placed.
Action: Mark the week as potentially censored and involve the inventory owner.
Evidence: Compare availability, lost-sales signals, and comparable locations.
Limit: The true unconstrained quantity is generally not directly observed.

**19. Returns processing.**
Situation: Returned goods are posted weeks after the original purchase.
Risk: Subtracting returns from current demand moves an earlier event into a later period.
Action: Define returns as a separate series unless net demand is the agreed target.
Evidence: Reconcile transaction dates, return dates, and policy treatment.
Limit: Forecast and financial reporting may legitimately use different measures.

**20. Promotion event.**
Situation: A large promotion occurs in one historical week.
Risk: Deleting the spike treats a genuine demand driver as an error.
Action: Check whether the event was real and whether it was known at the origin.
Evidence: Retain source evidence and label the event for slice analysis.
Limit: Do not use an event calendar that was finalized after the forecast date.

**21. Product launch.**
Situation: A new product has fewer than seven weeks of history.
Risk: Four evaluation folds and a three-week average cannot be constructed fairly.
Action: Fail the stated minimum-history gate rather than inventing observations.
Evidence: Record launch date and approved analog-series evidence.
Limit: An analog or pooled model requires its own leakage controls and evaluation.

**22. Product retirement.**
Situation: A product is discontinued before the next forecast target.
Risk: A historical baseline may continue predicting sales after the product is inactive.
Action: Consult effective-dated lifecycle status available at issue time.
Evidence: Document the discontinuation decision and its effective week.
Limit: The lab has no lifecycle feed and assumes an active series.

**23. Stale final week.**
Situation: The most recent historical week ended before the immediately preceding week.
Risk: The desired issue-date target no longer matches a one-step forecast.
Action: Reject the run under the sandbox freshness contract.
Evidence: Determine whether the data are late or the series was inactive.
Limit: A stale-data fallback must state its actual horizon and assumptions.

**24. Incomplete week.**
Situation: A weekly quantity is recorded before the seven-day bucket ends.
Risk: Comparing a partial bucket with completed weeks biases the forecast downward.
Action: Require bucket end no later than the issue date.
Evidence: Check source completion status as well as the calendar boundary.
Limit: The lab checks dates but cannot prove the source has finished processing.

**25. Late event.**
Situation: An event dated February arrives in April.
Risk: Event-date filtering alone places it into February training at March origins.
Action: Apply a recorded-time cutoff during each historical replay.
Evidence: Keep event time, first-seen time, and each correction version.
Limit: The static fixture cannot establish these timestamps retrospectively.

**26. Revised truth.**
Situation: A closed week is corrected after forecasts were issued.
Risk: Replacing prior snapshots can alter historical training and scoring simultaneously.
Action: Define separate as-of training snapshots and a matured evaluation truth.
Evidence: Retain original and revised values with revision timestamps.
Limit: A changed truth version should produce a separately labeled score.

**27. Fold boundary.**
Situation: A developer accidentally includes the target row in the training prefix.
Risk: The prediction has access to the answer it is meant to forecast.
Action: Use an exclusive slice ending at the target index.
Evidence: Check the first fold manually against its last eligible three values.
Limit: A passing aggregate MAE alone does not prove every fold is leakage-free.

**28. Origin definition.**
Situation: A forecast is issued Monday afternoon but actuals arrive Tuesday.
Risk: A weekly label does not establish that the previous observation was available.
Action: Define an explicit issue timestamp and source latency service level.
Evidence: Compare per-row recorded time to each historical issue timestamp.
Limit: The sandbox assumes Monday availability and checks dates only.

**29. Different horizons.**
Situation: A planner orders for four weeks but evaluates only next week.
Risk: One-step accuracy may not represent the uncertainty over lead time.
Action: Build forecasts and evaluate at the actual decision horizons.
Evidence: Compare horizon-specific errors and cumulative lead-time demand.
Limit: Do not use a one-week result as evidence for a four-week policy.

**30. Unequal folds.**
Situation: A candidate is scored on easier weeks than the baseline.
Risk: Score differences mix model behavior with differences in test population.
Action: Construct the common origin and target set before running either candidate.
Evidence: Compare target identifiers and actuals row by row.
Limit: A model unable to forecast a fold needs a declared missing-prediction policy.

**31. Candidate expansion.**
Situation: A team tries many window lengths after seeing the initial results.
Risk: The smallest observed selection error may reflect chance.
Action: Keep a log of all attempted candidates and reserve independent evaluation.
Evidence: Document the search space and dates before opening the holdout.
Limit: Four folds cannot support a broad model search.

**32. Metric direction.**
Situation: One report uses actual minus forecast and another uses the opposite sign.
Risk: An underforecast can be labeled as an overforecast.
Action: Specify the error formula in every metric contract.
Evidence: Check the sign on a known example such as actual 27 and forecast 26.
Limit: Absolute error is unchanged by sign, so MAE cannot detect the inconsistency.

**33. Zero actuals.**
Situation: All actual units in a reporting slice are zero.
Risk: WAPE has a zero denominator even if forecasts are nonzero.
Action: Return an undefined percentage and show absolute error in units.
Evidence: Include the denominator and count with every score.
Limit: Adding a tiny denominator invents an arbitrary and unstable percentage.

**34. Bias cancellation.**
Situation: Some weeks are overforecast and others underforecast.
Risk: Signed errors can cancel despite large individual errors.
Action: Review bias alongside MAE and the distribution of errors.
Evidence: Inspect fold-level signed and absolute values.
Limit: Low bias is not evidence that every decision was accurate.

**35. Segment masking.**
Situation: A high-volume distribution center dominates a network score.
Risk: Poor performance in a small but critical product group disappears in aggregate.
Action: Report agreed segments alongside the overall measure.
Evidence: Show sample counts, total demand, absolute error, and bias per slice.
Limit: Avoid exposing sensitive small cells without appropriate access controls.

**36. Tie handling.**
Situation: Both candidates have exactly the same selection MAE.
Risk: An unspecified tie can make publication change across environments.
Action: Use the documented candidate order to select naive deterministically.
Evidence: Test the tie using a constant-demand fixture.
Limit: Near ties still require careful interpretation and more evaluation data.

**37. Selection reuse.**
Situation: The winning method is reported using the same folds that chose it.
Risk: The selection procedure favors chance performance in those folds.
Action: Reserve a later period or use nested chronological evaluation.
Evidence: Record which dates were used for selection and final assessment.
Limit: A final holdout stops being independent after repeated tuning against it.

**38. Model refit.**
Situation: The selected model is copied from the final backtest fold without updating history.
Risk: The last backtest predicted an already completed week.
Action: Recompute the winner with all eligible observations at issue time.
Evidence: Verify that the publication target follows the last history bucket.
Limit: The selected method is not a guarantee of accuracy in the next week.

**39. Rounding.**
Situation: A moving-average candidate predicts fractional units.
Risk: Rounding each fold early changes computed errors and potentially selection.
Action: Keep full precision for scoring and document operational rounding separately.
Evidence: Compare raw forecast and rounded order quantities in different fields.
Limit: Some ordering constraints may require integer decisions, not integer forecasts.

**40. Output overwrite.**
Situation: A second local run replaces the first JSON file.
Risk: The original issued forecast is no longer available for audit.
Action: Write immutable run-specific artifacts and promote a pointer in production.
Evidence: Store run ID, issue timestamp, source digest, and consumer acknowledgments.
Limit: The sandbox intentionally uses one local output path for simplicity.

**41. Concurrent runs.**
Situation: Two processes stage output using the same temporary filename.
Risk: One writer can replace or remove the other writer’s temporary file.
Action: Use unique temporary names and enforce an atomic promotion protocol.
Evidence: Log run identifiers and detect overlapping schedules.
Limit: The sample publisher is designed only for one local writer.

**42. Rollback.**
Situation: A newly published version is found to use an incorrect input snapshot.
Risk: Consumers may already have read or acted on it.
Action: Stop promotion, identify affected consumers, and issue a versioned correction.
Evidence: Retain both original and corrected artifacts plus the incident timeline.
Limit: Replacing a file silently does not undo decisions already made.

**43. Human adjustment.**
Situation: A planner changes the forecast after receiving a customer signal.
Risk: The final number cannot be attributed to either source without separate versions.
Action: Record the original, adjusted value, rationale, actor, and timestamp.
Evidence: Compare model and adjusted forecasts using the same matured actuals.
Limit: An observational comparison alone does not establish causal value of overrides.

**44. Inventory interpretation.**
Situation: A 30-unit forecast is forwarded directly as a 30-unit purchase order.
Risk: Existing stock and open orders are ignored.
Action: Pass the forecast into an explicit inventory policy rather than equating it with an order.
Evidence: Record on-hand, on-order, lead time, constraints, and policy parameters.
Limit: The lab does not implement that inventory policy.

**45. Cost proxy.**
Situation: An analyst multiplies absolute errors by a single arbitrary dollar rate.
Risk: Asymmetric stockout and excess outcomes are collapsed into one unsupported number.
Action: Elicit cost assumptions and distinguish proxy loss from simulated policy cost.
Evidence: Perform sensitivity analysis across plausible underage and overage rates.
Limit: Even a detailed proxy is not an observed business saving.

**46. Monitoring maturity.**
Situation: A forecast is scored before late sales records have arrived.
Risk: Preliminary actuals may make healthy predictions appear inaccurate.
Action: Specify a data-maturation delay and retain actual-version metadata.
Evidence: Compare preliminary and finalized scores on the same issued forecasts.
Limit: Late corrections should not rewrite the forecast originally used.

**47. Access and privacy.**
Situation: A small location-product slice can reveal a customer relationship.
Risk: Forecast artifacts may expose commercially sensitive or personal information.
Action: Apply least-privilege access, retention, and a privacy review.
Evidence: Record who can publish, consume, correct, and delete artifacts.
Limit: Synthetic data in this lab carry no inference about production legal compliance.

**48. Recovery drill.**
Situation: The artifact store or orchestration service is unavailable.
Risk: A forecast may be lost or published without its evidence.
Action: Back up source snapshots and versioned artifacts and rehearse restoration.
Evidence: Record restore point, recovery duration, and validation of restored versions.
Limit: A local file replacement is not a backup or a disaster-recovery plan.

## 6. Worked example

The fixture describes one synthetic series in units per week.
All dates are Mondays in 2025.
The data have no missing weeks and no late revisions by assumption.

| Week start | Units | Use in selection |
|---|---:|---|
| Jan 06 | 20 | Earliest history |
| Jan 13 | 22 | Earliest history |
| Jan 20 | 21 | Earliest history |
| Jan 27 | 23 | Earliest history |
| Feb 03 | 24 | Earliest history |
| Feb 10 | 25 | Earliest history |
| Feb 17 | 24 | Earliest history |
| Feb 24 | 26 | Earliest history |
| Mar 03 | 27 | Fold 1 actual |
| Mar 10 | 29 | Fold 2 actual |
| Mar 17 | 28 | Fold 3 actual |
| Mar 24 | 30 | Fold 4 actual |

The four origins occur at the start of the respective target weeks.
The preceding week's final value is assumed to be available at each origin.
The one-week horizon means a target is the next bucket after the latest eligible week.
The three-week moving average uses only the last three observations preceding each target.

| Target | Three prior units | Actual units | Naive units | Moving-average units |
|---|---|---:|---:|---:|
| Mar 03 | 25, 24, 26 | 27 | 26 | 25.000000 |
| Mar 10 | 24, 26, 27 | 29 | 27 | 25.666667 |
| Mar 17 | 26, 27, 29 | 28 | 29 | 27.333333 |
| Mar 24 | 27, 29, 28 | 30 | 28 | 28.000000 |

The first moving average is `(25 + 24 + 26) / 3 = 25` units.
The second is `(24 + 26 + 27) / 3 = 25.666667` units.
The third is `(26 + 27 + 29) / 3 = 27.333333` units.
The fourth is `(27 + 29 + 28) / 3 = 28` units.
The exact calculation is performed without rounding intermediate values.

Using error = actual minus forecast, naive errors are `1, 2, -1, 2` units.
Their absolute values sum to `1 + 2 + 1 + 2 = 6` units.
Naive MAE is `6 / 4 = 1.5` units per forecast.
Naive bias is `(1 + 2 - 1 + 2) / 4 = 1` unit per forecast.
Actuals sum to `27 + 29 + 28 + 30 = 114` units.
Naive WAPE is `6 / 114 = 0.0526316`, or approximately `5.263%`.
The positive bias means these four predictions underforecast by one unit on average.

The moving-average errors are `2, 10/3, 2/3, 2` units.
Their absolute values sum to `8` units.
Moving-average MAE is `8 / 4 = 2` units per forecast.
Moving-average bias is `8 / 4 = 2` units per forecast.
Moving-average WAPE is `8 / 114`, or approximately `7.018%`.
The naive method wins the predeclared MAE comparison: `1.5 < 2`.
This is a selection result on four synthetic folds, not evidence of superiority in production.

At the issue date of March 31, the latest completed week begins March 24 and has 30 units.
The selected method publishes 30 units for the week starting March 31.
The hypothetical three-week moving average at that issue date would be `(29 + 28 + 30) / 3 = 29` units.
Neither value is a purchase order or a promise to a customer.
The next actual is intentionally unknown and is not in the input fixture.

For a business-oriented illustration, suppose each underforecast unit incurs an assumed USD 4 exposure and each overforecast unit incurs an assumed USD 1 exposure.
This is an error-loss proxy, not a reconstructed inventory simulation.
For naive, underforecast units are `1 + 2 + 2 = 5` and overforecast units are `1`.
The proxy loss is `5 × USD 4 + 1 × USD 1 = USD 21` across four folds.
The moving average underforecasts by 8 units and never overforecasts on these folds.
Its proxy loss is `8 × USD 4 = USD 32` across four folds.
These values do not prove realized savings: they ignore inventory on hand, lead times, substitution, capacity, and correlated demand.
The forecast-selection rule remains MAE; the asymmetric proxy is a separate diagnostic.
A different cost ratio or longer decision horizon may produce a different preference.

## 7. Code, query, or calculation

The lab requires Python 3.10 or later and its standard library only.
No network access, external package, database, credentials, or container is required.
Run in an empty writable sandbox directory; files named below are created or replaced there.
The fixture is synthetic, one series, and has no availability timestamps or true stockout flags.
The fixed issue date and data are for reproducibility, not a live clock.

Create `forecast_lab.py` with this complete file:

```python
import csv
import hashlib
import io
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path

ISSUE = date(2025, 3, 31)
INPUT = Path("demand_fixture.csv")
OUTPUT = Path("forecast.json")
FIXTURE = """week_start,sku,location,units
2025-01-06,SKU-1,DC-1,20
2025-01-13,SKU-1,DC-1,22
2025-01-20,SKU-1,DC-1,21
2025-01-27,SKU-1,DC-1,23
2025-02-03,SKU-1,DC-1,24
2025-02-10,SKU-1,DC-1,25
2025-02-17,SKU-1,DC-1,24
2025-02-24,SKU-1,DC-1,26
2025-03-03,SKU-1,DC-1,27
2025-03-10,SKU-1,DC-1,29
2025-03-17,SKU-1,DC-1,28
2025-03-24,SKU-1,DC-1,30
"""


def load(raw, issue):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    expected = ["week_start", "sku", "location", "units"]
    if reader.fieldnames != expected:
        raise ValueError("unexpected CSV columns")
    rows, seen = [], set()
    for row in reader:
        if None in row or any(row[k] is None for k in expected):
            raise ValueError("malformed CSV row")
        if row["sku"] != "SKU-1" or row["location"] != "DC-1":
            raise ValueError("unexpected series")
        try:
            day = date.fromisoformat(row["week_start"])
            value = float(row["units"])
        except ValueError as exc:
            raise ValueError("invalid date or units") from exc
        if day.weekday() != 0 or day + timedelta(days=7) > issue:
            raise ValueError("week must be complete Monday bucket")
        if not math.isfinite(value) or value < 0:
            raise ValueError("units must be finite and nonnegative")
        if day in seen:
            raise ValueError("duplicate series-week")
        seen.add(day)
        rows.append((day, value))
    rows.sort()
    if len(rows) < 7:
        raise ValueError("need at least seven weeks")
    if any(b[0] - a[0] != timedelta(days=7)
           for a, b in zip(rows, rows[1:])):
        raise ValueError("missing weekly bucket")
    if rows[-1][0] + timedelta(days=7) != issue:
        raise ValueError("latest completed week is stale")
    return rows


def predict(name, history):
    if name == "naive":
        return history[-1]
    if name == "mean3":
        return sum(history[-3:]) / 3
    raise ValueError("unknown model")


def scores(pairs):
    errors = [actual - forecast for actual, forecast in pairs]
    denominator = sum(abs(actual) for actual, _ in pairs)
    return {
        "mae": sum(abs(e) for e in errors) / len(errors),
        "bias": sum(errors) / len(errors),
        "wape": (sum(abs(e) for e in errors) / denominator
                 if denominator else None),
    }


def build(raw):
    rows = load(raw, ISSUE)
    models = ("naive", "mean3")
    pairs = {model: [] for model in models}
    folds = []
    for index in range(len(rows) - 4, len(rows)):
        history = [value for _, value in rows[:index]]
        target, actual = rows[index]
        predictions = {model: predict(model, history) for model in models}
        for model in models:
            pairs[model].append((actual, predictions[model]))
        folds.append({"target_week": target.isoformat(),
                      "actual_units": actual,
                      "predictions": predictions})
    metrics = {model: scores(pairs[model]) for model in models}
    winner = min(models, key=lambda m: (metrics[m]["mae"], models.index(m)))
    target = rows[-1][0] + timedelta(days=7)
    return {
        "schema_version": 1,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "issue_date": ISSUE.isoformat(),
        "target_week_start": target.isoformat(),
        "horizon_weeks": 1,
        "sku": "SKU-1",
        "location": "DC-1",
        "model": winner,
        "forecast_units": predict(winner, [value for _, value in rows]),
        "selection_metric": "mae",
        "metrics": metrics,
        "folds": folds,
    }


def publish(result):
    temp = OUTPUT.with_suffix(".json.tmp")
    try:
        temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        os.replace(temp, OUTPUT)
    finally:
        if temp.exists():
            temp.unlink()


def main():
    if not INPUT.exists():
        INPUT.write_text(FIXTURE, encoding="utf-8", newline="")
    result = build(INPUT.read_bytes())
    publish(result)
    print("selected:", result["model"])
    print("naive MAE:", result["metrics"]["naive"]["mae"])
    print("mean3 MAE:", result["metrics"]["mean3"]["mae"])
    print("target:", result["target_week_start"])
    print("forecast units:", result["forecast_units"])
    print("published:", OUTPUT)


if __name__ == "__main__":
    main()
```

Run it from the sandbox directory:

```text
python forecast_lab.py
```

Expected console output:

```text
selected: naive
naive MAE: 1.5
mean3 MAE: 2.0
target: 2025-03-31
forecast units: 30.0
published: forecast.json
```

Open `demand_fixture.csv` and `forecast.json` in a text editor.
The JSON records all four folds and both score sets, not only the winner.
The SHA-256 digest is determined by the exact CSV bytes, including line endings.
Existing input is not overwritten; delete or rename it only within your own sandbox to regenerate the fixture.
A rerun replaces `forecast.json`, so preserve a copy if you need to compare runs.
The fixed temporary filename and single output path are not safe for concurrent writers.

### Acceptance checks

Create `test_forecast_lab.py` in the same directory:

```python
import json
import math
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import forecast_lab as lab


class PipelineChecks(unittest.TestCase):
    def setUp(self):
        self.raw = lab.FIXTURE.encode("utf-8")

    def test_expected_selection_and_metrics(self):
        result = lab.build(self.raw)
        self.assertEqual(result["model"], "naive")
        self.assertEqual(result["forecast_units"], 30)
        self.assertEqual(result["target_week_start"], "2025-03-31")
        self.assertEqual(result["metrics"]["naive"]["mae"], 1.5)
        self.assertEqual(result["metrics"]["mean3"]["mae"], 2)
        self.assertTrue(math.isclose(result["metrics"]["naive"]["wape"], 6 / 114))
        self.assertEqual(result["metrics"]["naive"]["bias"], 1)

    def test_fold_does_not_use_target(self):
        result = lab.build(self.raw)
        self.assertEqual(result["folds"][0]["predictions"]["naive"], 26)
        self.assertEqual(result["folds"][0]["predictions"]["mean3"], 25)
        self.assertEqual(result["folds"][0]["actual_units"], 27)

    def test_bad_values_and_duplicate_fail(self):
        for bad in ("-1", "nan", "inf"):
            altered = lab.FIXTURE.replace(",DC-1,20\n", f",DC-1,{bad}\n", 1)
            with self.subTest(value=bad), self.assertRaises(ValueError):
                lab.build(altered.encode())
        with self.assertRaises(ValueError):
            lab.build(self.raw + b"2025-03-24,SKU-1,DC-1,30\n")

    def test_gap_and_incomplete_week_fail(self):
        missing = lab.FIXTURE.replace("2025-02-03,SKU-1,DC-1,24\n", "")
        with self.assertRaises(ValueError):
            lab.build(missing.encode())
        with self.assertRaises(ValueError):
            lab.load(self.raw, date(2025, 3, 30))

    def test_zero_denominator_is_undefined(self):
        self.assertIsNone(lab.scores([(0, 0), (0, 1)])["wape"])

    def test_tie_prefers_naive(self):
        flat = lab.FIXTURE
        for number in (20, 22, 21, 23, 24, 25, 26, 27, 29, 28, 30):
            flat = flat.replace(f",DC-1,{number}\n", ",DC-1,24\n")
        self.assertEqual(lab.build(flat.encode())["model"], "naive")

    def test_publication_and_failed_validation(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "forecast.json"
            with patch.object(lab, "OUTPUT", output):
                lab.publish(lab.build(self.raw))
                original = output.read_bytes()
                self.assertEqual(json.loads(original)["model"], "naive")
                with self.assertRaises(ValueError):
                    lab.publish(lab.build(b"bad\n"))
                self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
```

Run `python -m unittest -v test_forecast_lab`.
Expected result: seven tests pass and the runner reports `OK`.
These checks exercise the example's contract; they do not establish that a production input is complete or point-in-time correct.
If a check fails, inspect the failing assertion and input before changing either expected values or validation rules.

## 8. Data, standards, and regulatory context

The minimum data contract describes the entity, grain, target, unit, calendar, expected cadence, availability, and ownership.
Its tests should reject invalid input before publication, while preserving enough evidence for the source owner to investigate.
A production interface should document schema versions and how a consumer learns about breaking changes.

GS1 identification standards provide a common vocabulary for trade items and locations.
A Global Trade Item Number (GTIN) and Global Location Number (GLN) can identify entities in partner exchanges where those identifiers are appropriate.
The synthetic `SKU-1` and `DC-1` are local labels, not claimed GS1 identifiers.
GS1 identifiers do not decide whether sales represent demand, or whether a forecast is statistically valid.

Electronic data interchange (EDI) messages can transport orders or inventory information between organizations.
The message standard and trading-partner implementation guide constrain field meaning and syntax for a particular exchange.
They do not guarantee timely arrival, truthful measurement, or aligned weekly aggregation.
This lab neither implements an EDI transaction nor assumes one specific partner's message contract.

Data protection rules depend on jurisdiction and the data actually processed.
A product-location series may become sensitive when linked to individuals, customer accounts, or small cells.
The EU General Data Protection Regulation (GDPR) applies to personal data within its scope; its principles include data minimization and accuracy.
It does not prescribe a forecast model or make every aggregate demand record personal data.
Seek qualified advice about applicability, retention, cross-border transfers, and individual rights.
The fixture contains no personal information and is not a legal compliance assessment.

Maintain a data dictionary and document unit conversions.
Record whether the date denotes start or end of week, local business calendar or UTC, and how daylight-saving changes affect timestamp aggregation.
Date-only weekly labels do not require a timezone in this exercise, but source events usually do.
Record returns and substitutions separately when they change the target interpretation.
Do not compare cases, packs, kilograms, and individual units without a governed conversion.

For point-in-time replay, retain when a record was known as well as the event date.
A later correction may be legitimate for final evaluation but is not a feature that an earlier forecaster could have used.
A revised truth policy needs a maturation window and versioned actuals.
This lab uses a fixed synthetic snapshot and assumes all observations were available at every relevant origin.
Its checksum records bytes, not a guarantee of event-time completeness or an authorization to process the data.

## 9. Industry practice and case evidence

The M5 forecasting competition used hierarchical retail unit sales from Walmart and evaluated many series and horizons.
Its published paper documents the competition setup and results; it is not evidence that this small lab matches Walmart's operational production pipeline.
The case motivates multiple-series evaluation and appropriate aggregation, not a claim that the winner of this four-fold example would work at scale.

The M4 competition compared forecasting methods across a large collection of time series.
Its findings support taking simple benchmarks seriously and evaluating methods across multiple series.
The competition was not this lab's data, business, horizon, or cost structure.
A competition score cannot substitute for a retailer's own point-in-time replay and decision evaluation.

Forecasting: Principles and Practice describes time-series cross-validation as repeated forecasting from historical origins.
That general procedure underlies this lab's chronological folds.
Our implementation uses one-step forecasts and a small expanding window, not a full library implementation or a claim of statistical significance.

In practice, an organization may maintain several forecasts: a statistical baseline, a demand-planning override, and a consensus plan.
Those versions should be linked but not conflated.
A planner may have information about a customer event that a simple baseline does not; the audit record should capture the reason and timing.
A measured improvement after an override needs an appropriate comparison, and should not be attributed automatically to an individual or model.

A common operational pattern is to gate publication on data checks and preserve historical issued versions.
This is a recommended design pattern, not a documented claim about any named company.
The lab's local `os.replace` illustrates a single-process file handoff only.
A distributed deployment additionally needs concurrency controls, access policy, monitoring, an immutable version history, and tested recovery.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Duplicate rows | Repeated product-location-week | Source replay counted twice | Inflated demand | Enforce unique keys and investigate source |
| Missing week | Seven-day cadence breaks | Failed extract interpreted as no activity | Biased forecast | Stop; recover or explicitly adjudicate gap |
| Premature actual | Week incomplete at issue | Partial week enters history | Artificial demand drop | Enforce bucket completion and availability |
| Late correction leakage | Historic replay changes after refresh | Revised values were not known then | Optimistic selection | Retain as-of snapshots and recorded times |
| Stockout censoring | Sales fall while availability is zero | Sales are capped by inventory | Demand understated | Flag availability and model or review censoring |
| Random split | High validation score but poor forward result | Training includes future patterns | Misleading model choice | Use chronological folds |
| Inconsistent folds | Different candidate target dates | Unequal comparison | Apparent improvement is not causal | Freeze identical origin and horizon set |
| Changed metric | Winner changes after score inspection | Selection criterion adapted to outcome | Selection bias | Declare metric and tie policy first |
| Zero WAPE denominator | All actuals zero in slice | Division by zero | Undefined or misleading percentage | Report undefined and use unit metrics |
| Signed-error cancellation | Near-zero bias with large misses | Positive and negative errors offset | False confidence | Report absolute error and distribution too |
| Short evaluation | Only a few folds | Unrepresentative regime | Unstable selection | Expand history and retain final holdout |
| Stale publication | Latest week not present | Upstream job delayed | Forecast reflects old information | Freshness gate and owned alert |
| Unversioned overwrite | Same output path reused | Previous forecast lost | Audit and replay impossible | Immutable versions and controlled promotion |
| Plan mislabeled forecast | Override not identified | Desired quantity replaces prediction | Distorted error metrics | Store forecast, override, plan separately |
| Concurrent writers | Temporary file collisions | Multiple runs use same paths | Mixed or failed publication | Unique staging paths and lock/transaction |
| Wrong unit | Sudden factor-of-case-size change | Units and cases mixed | Large order error | Govern conversions and unit checks |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| One product and location | Code holds one series in memory | Multiple identifiers appear | Reject other series; partition explicitly in an extension |
| Weekly Monday buckets | Defines cadence and target | Different fiscal calendar | Reconcile source calendar with approved calendar |
| Values are demand | Forecast target is unconstrained demand | Lost sales, stockouts, or returns | Join availability and assess censoring |
| Nonnegative units | Fixture represents gross demand | Net returns exceed shipments | Define separate return measure or change contract |
| Values available by issue | Historical model can use them | Delayed ingestion or revisions | Store recorded time and replay as-of |
| Full latest week available | Next-week issue occurs promptly | Reporting latency beyond Monday | Delay issue or model partial observations |
| Last four folds informative | Local selection can compare models | Regime changes or seasonal gaps | Extend rolling evaluation and reserve holdout |
| One-week horizon matters | Backtest matches decision | Longer replenishment lead time | Score relevant lead-time horizons |
| Equal fold weight | MAE treats four weeks equally | Promotions or segments matter differently | Define decision weights before selection |
| Fixture needs no privacy controls | Synthetic identifiers and values | Customer-level production records | Apply access, retention and privacy review |
| Local replacement suffices | Single process writes local file | Concurrent runs or distributed storage | Use versioned transactional publication |
| USD cost proxy is illustrative | Demonstrates asymmetric errors | No actual per-unit cost model | Obtain operational costs and simulate policy |

The sandbox does not train a machine-learning model, estimate uncertainty, handle multiple series, or connect to a real source.
It cannot prove that an upstream observation was available when a historical prediction was issued.
It does not reserve an independent holdout, so the displayed selection score should not be represented as final expected accuracy.
No production outcome, regulatory approval, or business saving is implied.

## 12. Exercises

### Exercise 1 — Contract (15 minutes)

Deliverable: a one-page demand data contract for a weekly store-product series.
Specify grain, measurement, unit, calendar, key, availability, and missing-week behavior.
Acceptance: every attribute is explicit; the document distinguishes zero activity from missing data.
Relates to learning outcome 1.

### Exercise 2 — Validation triage (15 minutes)

A row is duplicated; another contains `NaN`; a third arrives two weeks late with an old event date.
Deliverable: identify the code's response and the extra evidence needed for each case.
Acceptance: duplicate and nonfinite values are rejected; late-arrival leakage is identified as outside the fixture's validation capability.
Relates to outcomes 2 and 8.

### Exercise 3 — Reconstruct folds (20 minutes)

Deliverable: list the training endpoint, target, naive prediction, and moving-average prediction for each of the four folds.
Acceptance: no fold uses its target as input; all predictions match section 6.
Relates to outcome 3.

### Exercise 4 — Metrics (20 minutes)

Deliverable: calculate naive MAE, signed bias, and WAPE from the fold table.
Acceptance: show denominators, sign convention, and units; explain what happens to WAPE if every actual is zero.
Relates to outcome 4.

### Exercise 5 — Select and critique (20 minutes)

Deliverable: identify the winning model and explain why its selected score is not an independent final test.
Acceptance: use MAE and the declared tie rule, and propose a later holdout.
Relates to outcome 5.

### Exercise 6 — Publication review (20 minutes)

Deliverable: inspect the generated JSON and write a consumer handoff specifying its target, units, issue date, model, provenance, and limitations.
Acceptance: forecast is not called an order; identify local overwrite and missing production controls.
Relates to outcome 6.

### Exercise 7 — Business proxy (20 minutes)

Deliverable: compute the illustrative asymmetric loss for each model with underforecast at USD 4 per unit and overforecast at USD 1 per unit.
Acceptance: show the counts of underforecast and overforecast units, costs, and why they are not realized inventory savings.
Relates to outcome 7.

### Exercise 8 — Production design (30 minutes)

Deliverable: diagram the controls needed to replace the sandbox with a governed multi-series pipeline.
Acceptance: include point-in-time data, nonoverlapping selection and final evaluation, publication versioning, monitoring, ownership, and a rollback strategy.
Do not claim any real system meets your proposed controls without evidence.
Relates to outcomes 2, 6, and 8.

## 13. Self-check answers

1. A suitable key is product, location, and week start.
   Units are nonnegative units per week.
   Record when the week closed and when the observation arrived.
   A missing row is a data-quality question, not automatically zero.
2. Duplicate keys and nonfinite values raise `ValueError`.
   The code does not know when a late record arrived.
   An ingestion timestamp and versioned as-of snapshot are needed to verify historical availability.
3. Training endpoints are February 24, March 3, March 10, and March 17.
   Target dates are March 3, March 10, March 17, and March 24.
   Naive predictions are 26, 27, 29, and 28 units.
   Moving-average predictions are 25, 25.666667, 27.333333, and 28 units.
4. Absolute naive errors sum to 6 units across four forecasts, giving MAE 1.5 units.
   Signed errors sum to 4 units, giving bias +1 unit.
   WAPE is `6 / 114 = 5.263%` approximately.
   When all actuals are zero, its denominator is zero and the percentage is undefined.
5. Naive wins because 1.5 is lower than 2 units.
   If scores tie, the ordered candidate list gives naive preference.
   The same folds selected the winner, so use later untouched data or nested evaluation for a separate estimate.
6. The artifact predicts 30 units for the week starting March 31, issued March 31.
   It records the input byte digest and both candidates' fold evidence.
   It is a forecast, not a purchase order; local overwrite is not immutable versioning.
7. Naive underforecast is 5 units and overforecast is 1 unit, producing USD 21 proxy loss.
   Mean3 underforecast is 8 units and overforecast is zero, producing USD 32 proxy loss.
   Neither is realized cost because the calculation omits actual inventory decisions and operational constraints.
8. A strong design gives each prediction a unique issue and target, captures immutable as-of source versions, and tests source freshness.
   It freezes candidates and folds, preserves a later holdout, versions publication, and defines rollback and accountable owners.
   It separates service, demand error, data health, and business outcome monitoring.

## 14. Interview questions

1. How would you define weekly demand?
   Strong: states unit, series grain, calendar and censoring.
   Weak: assumes every sales record is unconstrained demand.
2. Why are random train/test splits inappropriate here?
   Strong: explains the information boundary and changing time regimes.
   Weak: says only that time series are special.
3. What does a historical as-of join protect against?
   Strong: prevents later arrivals and corrections entering earlier predictions.
   Weak: sorts current rows by event date and assumes that is sufficient.
4. Why retain a naive model?
   Strong: treats it as a fair and meaningful decision alternative on identical folds.
   Weak: compares a new model to no forecast.
5. What does positive bias mean in this topic?
   Strong: actual minus forecast is positive, so average underforecast.
   Weak: ignores the stated sign convention.
6. When is WAPE undefined?
   Strong: when total absolute actuals are zero; report unit-based metrics too.
   Weak: silently assigns zero error.
7. Why is the winning backtest score optimistic?
   Strong: model selection reused those observations and needs separate assessment.
   Weak: assumes chronological ordering alone eliminates selection bias.
8. How do you evaluate business value?
   Strong: compares policies with historical inventory state, cost assumptions and sensitivity.
   Weak: converts a percentage error reduction directly into guaranteed savings.
9. What makes a forecast auditable?
   Strong: identifies input snapshot, code/model version, issue, target, selection and overrides.
   Weak: retains only the latest output number.
10. How would you prevent bad data replacing a good forecast?
    Strong: validation gates, staging, controlled promotion, alerts and rollback.
    Weak: catches every exception and silently publishes zeros.

## 15. Key terms

- [Actual](#4-definitions-and-vocabulary): observed value used for scoring.
- [Availability time](#4-definitions-and-vocabulary): time a record became usable.
- [Backtest](#4-definitions-and-vocabulary): historical prediction replay.
- [Baseline](#4-definitions-and-vocabulary): reference forecast for comparison.
- [Bias](#4-definitions-and-vocabulary): mean signed error.
- [Bucket](#4-definitions-and-vocabulary): interval of aggregation.
- [Censoring](#4-definitions-and-vocabulary): observation constrained by availability.
- [Demand](#4-definitions-and-vocabulary): desired product units.
- [Error](#4-definitions-and-vocabulary): actual minus forecast here.
- [Expanding window](#4-definitions-and-vocabulary): increasing history across origins.
- [Fold](#4-definitions-and-vocabulary): one prediction and evaluation origin.
- [Forecast override](#4-definitions-and-vocabulary): traceable change from model output.
- [Grain](#4-definitions-and-vocabulary): unique measurement level.
- [Holdout](#4-definitions-and-vocabulary): later untouched assessment period.
- [Horizon](#4-definitions-and-vocabulary): distance to target bucket.
- [Leakage](#4-definitions-and-vocabulary): unavailable information used early.
- [MAE](#4-definitions-and-vocabulary): mean absolute error in units.
- [Model selection](#4-definitions-and-vocabulary): choice using declared criterion.
- [Moving average](#4-definitions-and-vocabulary): mean of recent values.
- [Naive forecast](#4-definitions-and-vocabulary): last value carried forward.
- [Origin](#4-definitions-and-vocabulary): forecast issue cutoff.
- [Provenance](#4-definitions-and-vocabulary): evidence of source and process.
- [Publication](#4-definitions-and-vocabulary): handoff with metadata.
- [Snapshot](#4-definitions-and-vocabulary): frozen source state.
- [WAPE](#4-definitions-and-vocabulary): aggregate absolute error divided by absolute actuals.

## 16. Sources

- Hyndman, Rob J., and George Athanasopoulos. [Forecasting: Principles and Practice, 3rd edition](https://otexts.com/fpp3/tscv.html), section on time-series cross-validation.
  Explains rolling-origin evaluation; accessed for methodological context, not evidence of this lab's performance.
- Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos. [The M4 Competition: Results, findings, conclusion and way forward](https://doi.org/10.1016/j.ijforecast.2018.06.001), International Journal of Forecasting, 2018.
  Provides comparative forecasting evidence from a different dataset.
- Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos. [The M5 accuracy competition: Results, findings, and conclusions](https://doi.org/10.1016/j.ijforecast.2021.11.013), International Journal of Forecasting, 2022.
  Documents the retail forecasting competition, not this lab's implementation.
- GS1. [GS1 identification keys](https://www.gs1.org/standards/id-keys).
  Describes identifiers and their intended uses.
- European Parliament and Council. [Regulation (EU) 2016/679, Article 5](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
  Primary text for personal-data processing principles; applicability depends on the facts.
- Python Software Foundation. [CSV File Reading and Writing](https://docs.python.org/3/library/csv.html).
  Documents the standard-library CSV parser used in the exercise.
- Python Software Foundation. [`os.replace`](https://docs.python.org/3/library/os.html#os.replace).
  Documents file replacement behavior and filesystem limitations.
- Python Software Foundation. [`unittest`](https://docs.python.org/3/library/unittest.html).
  Documents the test runner and assertion interface.

## 17. Links

- [Module 30 topic index](_index.md).
- [Module 30 survey and original lab](../../../course/30_simulation_optimization_production_labs.md).
- [Environment and repository lab](01_environment-repo.md).
- [Inventory policy simulation lab](03_inventory-policy-simulation.md).
- [Model service and container lab](08_model-service-container.md).

Sibling topic links are indexed as planned until their files are created.
The standalone code and synthetic fixture above do not depend on a sibling lab being complete.
