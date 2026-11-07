---
case_id: "CASE-01-10-002"
case_title: "Ride-hailing demand forecasting for operational planning"
case_version: "0.1"
created: "2026-11-07"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Ride-hailing marketplace demand"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["data-leakage", "model-monitoring"]
evidence_level: "not-measured"
source_basis: "published and illustrative"
as_of_date: "2026-11-07"
owner_role: "Marketplace forecasting lead (proposed)"
source_index: "INDEX.md"
review_trigger: "Change to target definition, data availability, intervention policy, or primary source interpretation"
---

# Ride-hailing demand forecasting for operational planning

## 1. The One-Line Framing

Given historically available request counts by geographic zone and hour, propose an hourly forecast of the next hour's requests for a hypothetical marketplace operations team, against a previous-week same-hour baseline, subject to point-in-time data correctness and explicit uncertainty.
The primary success condition is a pre-specified reduction in held-out absolute request-count error without unacceptable operational or coverage regressions.
Forecasting is not dispatch: the forecast informs a separately governed human decision about whether and where to offer optional driver guidance.
The primary pattern is `evaluation-mismatch`; `data-leakage` and `model-monitoring` are secondary.
**Evidence declaration:** Uber's historical article is published evidence of general forecasting practices and marketplace context, not evidence of the proposed design or its results.
All data, numbers, policy choices, thresholds, and architecture below are illustrative unless explicitly attributed to a source.
No proprietary source data, model configuration, operational deployment, or measured intervention effect was supplied.
The source does not establish a demand forecast accuracy figure or ETA configuration; ETA remains a separate planned case.
The important tradeoff is that better error on recorded requests need not improve driver or rider outcomes when actions change the marketplace.

## 2. Context & Background

[Smyl and Bell's 2018 article](https://www.uber.com/us/en/blog/forecasting-introduction/) describes geographically and temporally granular marketplace forecasts of supply and demand and states that they can direct drivers toward high-demand areas.
It discusses physical distance and road throughput as reasons geographic structure matters, and presents illustrative historical trip-series seasonality.
Those statements are historical published claims, not a description of the present system or evidence of a causal gain in trips or earnings.
For the hypothetical workflow, a planner sees next-hour request forecasts and uncertainty by zone before choosing whether to issue optional guidance.
Riders originate requests; drivers choose whether and where to work; operations sets policy; data engineering owns event pipelines; forecasting owns model evaluation; platform engineering owns the serving interface.
The safety and policy owner may veto guidance when the operational implications are unclear.
The illustrative baseline is the last comparable hour one week earlier, not a claim about an existing Uber baseline.
Historical population, geographic coverage, request volumes, rollout dates, and prior attempts are unknown.
No legal applicability, current organizational structure, or binding company policy is inferred from the article.
UTC is used for storage in the proposed exercise; local calendar features require a versioned time-zone mapping.
The target is recorded requests, not all latent rider intent, completed trips, driver supply, or an ETA.

## 3. The Business Question

**Illustrative stakeholder question:** Should a marketplace planning team replace a weekly seasonal-naive demand forecast with a candidate forecast for next-hour zone-level planning?
The decision owner is the hypothetical operations lead, with forecasting and platform leads as technical reviewers.
The decision would occur before the forecast is used for a guidance cycle; no actual deadline or approval exists.
Actions are to retain the baseline, conduct a shadow evaluation, run a carefully governed experiment, or defer for insufficient evidence.
Underforecasting can cause a planner to miss demand; overforecasting can send drivers toward an area with too little work.
The primary proxy is request-count absolute error; rider experience and driver outcomes require separate guardrails and, for causal impact, an appropriate experiment.
The scope is one-hour-ahead requests in a defined zone set, not individual route or arrival-time prediction.
No action should be attributed to Uber from this exercise.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Status and verification | Consequence and fallback | Owner / trigger |
|---|---|---|---|---|
| A-01 | Request event time and ingestion time are both retained | Unverified; audit event histories and replay cutoffs | Without them, backtest leakage is unbounded; defer model comparison | Data lead / ingestion change |
| A-02 | A stable request identity supports deduplication | Unverified; inspect retries and revisions | Duplicates inflate labels; quarantine affected partitions | Data lead / producer change |
| A-03 | Request counts are useful for the operational decision | Unverified; interview planners and compare policy outcomes | Counts may conceal unmet intent; retain manual review | Operations / policy change |
| A-04 | A historical zone map can be reconstructed | Unverified; audit boundary versions | Spatial joins may silently rewrite history; aggregate to stable geography | Geospatial lead / remapping |
| A-05 | Labels mature within a documented delay | Unverified; measure revisions against event age | Premature labels bias scores; delay scoring | Data lead / latency shift |
| C-01 | No actual service deadline is known | Scenario constraint; agree with consumers before deployment | Reject stale outputs and use approved fallback | Platform lead / contract change |
| C-02 | Access and retention must be approved | No applicable policy was supplied; privacy review required | Block access until approved, minimize retained fields | Privacy owner / data-use change |
| C-03 | No real budget or capacity figures are available | Unverified; benchmark and obtain approval | Avoid claiming production feasibility | Platform lead / workload change |

The three assumptions most likely to reverse the design are historical availability, target usefulness, and the ability to evaluate a policy without substantial cross-zone interference.
A privacy or safety requirement may impose additional restrictions, but its legal basis is not established here.

## 5. Data, Tools & Method

### 5a. Data Specification

A training example is `(zone_id, forecast_origin, horizon=1 hour)` with the target counting unique accepted request events in the half-open interval `[origin, origin + 1 hour)`.
Here “accepted” means accepted by the data ingestion contract, not accepted by a driver; the event definition must be agreed with the data owner.
The example scenario uses a fixed zone map and hourly UTC buckets; both are proposals.
The source supplies no accessible row-level dataset, volume, actual period, or request schema for this case.
Include zero counts only when the event pipeline confirms complete coverage; distinguish zero activity from an absent or incomplete partition.
Exclude test traffic under an auditable rule known before scoring; preserve counts of exclusions by reason.
A request's event timestamp determines its target bucket; its ingestion timestamp determines when it can be used as a feature.
A corrected event may revise a historical label but cannot be retroactively available as a feature in a historical forecast.
Aggregate requests once per stable event ID and resolve updates deterministically by version and ingestion time.
Join zone assignments to the map version effective at event time and available at the forecast origin, not today's map applied silently to the past.
Candidate inputs include past mature request counts, calendar features, and published external forecasts only if their issue time precedes the origin.
Historical realized weather, final trip completion, future price decisions, and labels not yet mature are not valid origin-time inputs.
Use rolling chronological origins for tuning and a later untouched chronological test interval for final comparison.
Adjacent origins and adjacent zones are dependent, so raw row count is not an independent sample size.

#### Proposed event contract and decision-time examples

The following fields and rules describe a proposed implementation, not a recovered Uber schema.
Each field has a distinct function in determining whether a record can influence a forecast.
`event_id` identifies the underlying request event, not a delivery attempt.
`event_version` orders legitimate corrections from the event producer.
`event_time_utc` assigns the event to a target interval.
`available_time_utc` records when this version became queryable by the forecaster.
`received_time_utc` can diagnose transport delay but need not equal query availability.
`origin_utc` is the cutoff at which the forecast is issued.
`target_start_utc` equals the origin in this one-hour-ahead design.
`target_end_utc` is exclusive and equals one hour after the origin.
`map_version` identifies the geographic partition used for the output.
`zone_id` identifies one member of that map version, not a permanent geographic fact.
`source_schema_version` identifies the producer contract in force for the record.
`is_test` identifies traffic excluded under a predeclared operational rule.
`correction_type` distinguishes replacement, deletion, and ordinary delivery.
`completeness_state` records whether all required producers have reported for a partition.
`lineage_id` connects an aggregate back to its versioned inputs without exposing unnecessary personal details.
An as-of reconstruction chooses only versions with `available_time_utc <= origin_utc`.
When versions arrive out of order, producer sequence and the correction contract govern which version supersedes another.
An event correction that arrives after the origin can change the final label but cannot change that origin's inputs.
An event with a future event time cannot be a historical request-count feature, even if a malformed producer sends it early.
A scheduled calendar entry can concern a future event because its announcement may already be known.
The decisive distinction is availability at the origin, not whether a feature describes the future.
For externally published forecasts, retain the issue time and the exact forecast vintage.
The later revised forecast must not replace the forecast that was actually available to the original decision.
For a manual event calendar, record when an operator entered or amended the event.
The final verified event calendar may be useful for retrospective stratification but not as an unqualified model input.
When an event moves across a zone boundary after correction, the as-of record and the mature label may legitimately use different information.
The reconciliation report should account for both the removed contribution and the added contribution.
Counts cannot be compared across zone-map versions without either a stable crosswalk or a common coarser partition.
A many-to-many crosswalk can allocate counts only under explicit assumptions; it is not an exact replacement for the original events.
If a map change changes the population of a zone, an apparent model drift may actually be a definition change.
Keep the original map version with every forecast and score.
One hourly example uses a forecast origin at 12:00 UTC and a target window of `[12:00, 13:00)` UTC.
Suppose request A occurred at 11:58 and became queryable at 11:59.
Request A may contribute to the latest historical count if the relevant bucket is complete.
Suppose request B occurred at 11:59 but became queryable at 12:03.
Request B cannot be used at the 12:00 origin, even though it occurred before that origin.
Suppose a correction to request A became queryable at 12:05.
The 12:00 feature must use the version available at 12:00, not the 12:05 correction.
Suppose request C occurred at exactly 12:00.
Request C belongs to the target window, not the historical bucket ending at 12:00.
Suppose request D occurred at exactly 13:00.
Request D belongs to the next target window, not `[12:00, 13:00)`.
Suppose a weather forecast for the target hour was issued at 11:30 and available at 11:35.
That weather vintage is eligible at 12:00, subject to the agreed access and lineage contract.
Suppose the same provider issues an improved weather forecast at 12:15.
That later weather vintage is not eligible for the 12:00 forecast.
Suppose a venue announces an event at 10:00 but the announcement is ingested at 12:10.
The forecast system cannot use it at 12:00 merely because the announcement was publicly available earlier.
The relevant system availability is when the authorized feature pipeline could have supplied the information.
Suppose the venue cancels the event at 12:30.
The cancellation may explain the later outcome but is not a valid 12:00 input.
These times are constructed to expose leakage, not to characterize any real service.
An audit should retain both examples that fail and examples that pass the cutoff predicate.
A sample with only accepted rows cannot expose whether late rows were mistakenly filtered.
The data owner should approve the handling of records whose timestamp is missing or impossible.
Quarantine is preferable to silently substituting processing time for event time.
Quarantine counts should appear in completeness reporting rather than disappearing from the denominator.
An unavailable source should produce an explicit missingness state, not an invented count of zero.
The training code must consume the same missingness representation as the serving code.
Feature code deployed with different defaults can create training-serving skew even when source data are identical.
The event store needs a stable ordering rule when two updates have the same available timestamp.
The ordering rule should be deterministic under replay and independent of distributed worker arrival order.
An immutable audit record should preserve the discarded duplicate's reason without exposing unnecessary identifiers.
When a request is cancelled, the target definition must specify whether the original request remains counted.
That choice must be decided before comparing model scores because it changes both labels and operational meaning.
The proposed target counts an eligible submitted request rather than a completed ride.
This target does not measure would-be riders who never submit a request.
It also does not measure whether a driver was available to fulfill each request.
A count of completions would mix demand with matching, cancellation, and supply availability.
Using completions as demand would therefore answer a different question.
An exposure metric such as app opens could help investigate latent intent but would require its own consent, eligibility, and interpretation.
Pricing or service availability may affect whether an open becomes a request.
Consequently, even a perfectly measured request count remains conditional on the marketplace policy that generated it.
Before using a new source, document who owns its availability guarantee.
Document whether backfills replace or append prior records.
Document whether a deletion propagates to historical aggregates and derived artifacts.
Document how long a replayable snapshot may lawfully and contractually be retained.
Document whether a service account can read the data across all intended regions.
Document which regions cannot be joined because access approval differs.
These are preconditions to an honest replay, not evidence that such approvals exist.

### 5b. Tools & Stack

No actual code or platform is identified by the source for this hypothetical case.
A proposed stack consists of an append-only event store, a versioned aggregate store, a batch training runner, a model registry, and an authenticated forecast API.
The interface and controls matter more than a particular vendor or language; no platform deployment is asserted.
Operators need a dashboard of forecast freshness, data completeness, error, coverage, and fallback activity.
Versions, compute sizing, and library dependencies must be selected and recorded during a real implementation.

### 5c. Method Summary

| Candidate | Evidence or status | Strength | Limitation | Revisit condition |
|---|---|---|---|---|
| Weekly seasonal naive | Proposed baseline; the source discusses seasonal naive generally | Transparent and inexpensive | Holidays and regime changes break last week's correspondence | Calendar or policy changes |
| Recent-value naive | Considered, not tested | Reacts quickly to persistent shifts | Can miss weekly structure | Long seasonal history unavailable |
| Statistical seasonal model | Considered, not tested | Interpretable trends and seasonality | May miss spatial interactions | Stable seasonal histories |
| Global regression or boosted trees | Considered, not tested | Shares information across zones and supports external features | Requires leakage-safe features and tuning | Sufficient trustworthy training examples |
| Human forecast | Considered, not tested | Can incorporate known events | Inconsistent and costly; must record origin-time judgment | Rare disruptions or new zones |

For a candidate trained to minimize absolute error, the conditional median is the point target; interval endpoints require separately validated quantile estimates or calibration.
Choose any hyperparameters within rolling training/validation folds only; the actual optimizer, search space, stopping rule, and model are not selected.
Keep baseline and candidate on identical origins, zone sets, target definitions, and label maturity rules.
The [2018 article](https://www.uber.com/us/en/blog/forecasting-introduction/) describes chronological sliding and expanding backtests, seasonal-naive comparisons, prediction intervals, and Omphalos as a parallel, language-extensible backtesting framework; it does not establish that the proposed model or pipeline used Omphalos.

### 5d. Data provenance, quality, exclusions and leakage controls

The public article is the only company-specific evidence used; it does not expose the raw marketplace records or run artifacts required to reproduce a model.
For a future real dataset, record producer, access approval, extraction time, snapshot ID, schema, retention limit, and transformation revision.
At each forecast origin, reconstruct only events and external forecasts whose availability timestamps are no later than the origin.
Fit imputation, scaling, and learned encoders separately within each training fold.
Track pipeline freshness and missing partitions independently from the demand values they contain.
Reconcile raw event counts, deduplicated counts, excluded events, and zone aggregates, with explicit tolerances approved by the data owner.
Investigate abrupt count changes against ingestion delays, schema revisions, map changes, and real marketplace changes before treating them as model drift.
Avoid storing unnecessary rider or driver identifiers in aggregate forecast artifacts.

### 5e. Reproducibility and method-selection ledger

| ID | Artifact | Date and access | Supports | Limitation |
|---|---|---|---|---|
| SRC-01 | [Smyl and Bell, *Forecasting at Uber: An Introduction*](https://www.uber.com/us/en/blog/forecasting-introduction/) | 2018-09-06; public; read 2026-11-07 | Historical marketplace context, backtesting discussion, baseline, intervals, Omphalos | No case-specific causal impact, accuracy figure, configuration, or ETA details |
| SRC-02 | [Forecasting: Principles and Practice, time-series cross-validation](https://otexts.com/fpp3/tscv.html) | Public method reference | Rolling-origin reasoning | Not evidence about Uber's implementation |
| CALC-01 | Section 6 constructed example | 2026-11-07; inline | Arithmetic only | Not an empirical backtest |
| RUN-01 | No actual run | Not available | Nothing | No code, data snapshot, dependencies, seed, or artifacts exist |

A future run manifest must pin origin schedule, zone-map version, event snapshot, code revision, dependencies, feature and label definitions, model settings, and output artifact IDs.
A seed alone would not guarantee deterministic results across hardware or library versions.

#### Claim ledger and evidence boundaries

| ID | Claim and scope | Evidence | Supports | Does not establish |
|---|---|---|---|---|
| CLM-01 | The 2018 article describes fine-grained spatiotemporal marketplace forecasts | SRC-01, marketplace forecasting | Historical motivation for the topic | Current deployment, scale, or measured benefit |
| CLM-02 | The article describes chronological sliding and expanding comparisons | SRC-01, comparing forecasting methods | Evaluation alternatives | A specific backtest for this decision |
| CLM-03 | The article explains seasonal-naive comparison | SRC-01, comparing forecasting methods | A meaningful proposed baseline | This baseline's actual error |
| CLM-04 | The article discusses prediction intervals and their decision relevance | SRC-01, importance of uncertainty | Reporting uncertainty as well as point forecasts | Calibration of the constructed intervals |
| CLM-05 | The article identifies Omphalos as a parallel, language-extensible backtesting framework | SRC-01, comparing forecasting methods | Historical provenance of that framework | Its use in the proposed design or a particular ride-hailing model |
| CLM-06 | No real model replacement is supported by the available evidence | Sections 6, 7, 9, and 11; not-measured | Defer adoption pending evidence | A real-world finding that the candidate failed |
| CLM-07 | CALC-01 has MAEs of 3.25 and 2 on constructed values | Section 6; illustrative | Auditable arithmetic | Real-world accuracy or an intervention effect |
| CLM-08 | The proposed capacity arithmetic depends on a service-rate assumption | Section 8b; illustrative | Sensitivity of the decision to units and assumptions | Actual driver capacity or dispatch requirements |

The review date for SRC-01 is the case as-of date, 2026-11-07.
The publication date is 2018-09-06; it is not treated as a deployment date.
The article's authors describe several forecasting use cases in one general introduction.
This case uses the marketplace discussion as context and does not merge its other use cases into the proposed target.
The article describes a figure with historical trip counts and observed seasonality.
Those trip counts are not the hypothetical request counts used in CALC-01.
The article's reference to geographic distance motivates attention to space.
It does not specify this case's fictional four-zone layout.
The article's discussion of weather and events motivates candidate exogenous information.
It does not establish that the proposed model uses any particular external provider.
The article describes a framework for comparing forecasting methodologies.
It does not identify the code version, configuration, or dataset of this exercise.
The article describes interval forecasts in general terms.
It does not supply the four numerical intervals in the worked example.
The article does not report a measured effect for this proposed optional-guidance policy.
The article's stated marketplace motivation is not a causal estimate.
The absence of a causal estimate in this case does not prove that no such estimate exists elsewhere.
Another publication could support an additional claim only after its scope is checked.
An unrelated paper about arrival times would not establish this case's demand model configuration.
An example of a model family in a general forecasting overview would not establish its deployment here.
An internal system name alone would not establish which version was used in a particular market.
A source's description of a platform is not a benchmark for this hypothetical workload.
A product claim does not by itself define an experimental control group.
A statement about a forecast's usefulness does not reveal its exact MAE.
A reported relative improvement elsewhere cannot be assigned to these constructed zone-hours.
A time-series figure does not provide the raw event timestamps needed for point-in-time replay.
A screenshot of an aggregate series does not establish request-level deduplication rules.
A historical description of an organization does not establish its present ownership structure.
An article's byline does not make the authors the owners of this proposed pipeline.
The named owner roles in this document are illustrative responsibilities.
Any real implementation must identify actual accountable people or teams.
No private material was accessed to complete the design exercise.
No confidential schema or proprietary training pipeline is reconstructed.
No actual market, region, or service-level objective is asserted for the hypothetical scenario.
No operational target is inferred from the fact that a forecast is hourly.
No causal conclusion is inferred from the arithmetic comparison.
No confidence interval is inferred from four constructed observations.
No measured incident is inferred from a hypothetical diagnostic drill.
No completed validation is inferred from a checklist of proposed validation steps.
No pilot approval is inferred from the presence of a rollout playbook.
No current product behavior is inferred from a 2018 article.
The retail case is a comparison of problem structures, not corroboration of Uber-specific claims.
The time-series cross-validation reference supports a general method, not a claim of deployment.
Future reviewers should revisit CLM-01 through CLM-05 if the source changes or becomes inaccessible.
Future reviewers should revisit CLM-06 if real access and a valid comparative evaluation become available.
Future reviewers should retain the constructed examples as illustrations even if a real study is later added.
Any real study should receive a new run identifier and a separate result row.
Its artifact location and access restriction should be documented without exposing private data.
Its result should state the actual population, comparator, period, metric, and uncertainty.
Its impact claim should cite the intervention record rather than the forecast score alone.
Its known negative results should remain visible even if an aggregate metric improves.
An edited outcome should not silently replace an earlier score from a different label snapshot.
The final case card must continue to distinguish publication state from implementation stage.

#### Proposed handoff evidence for an independent reviewer

The handoff should contain the written target definition and the version used in every run.
It should contain the policy for identifying duplicate events.
It should contain the rule for selecting a correction when multiple versions exist.
It should contain a description of event-time and availability-time semantics.
It should contain the historical zone map and its effective interval.
It should contain a reconciliation of source counts to model-ready aggregates.
It should contain counts and reasons for quarantined records.
It should contain a record of source outages during each evaluation window.
It should contain the as-of query or equivalent reconstruction specification.
It should contain the external-feature issue-time rule.
It should contain the origin schedule and the forecast horizon.
It should contain the chronological boundaries of training, tuning, and final testing.
It should contain the baseline definition and its missing-history fallback.
It should contain the candidate's objective and model-selection record.
It should contain the actual model artifact hash and compatible feature contract.
It should contain any calibration artifact and the data used to fit it.
It should contain the prediction files before joining mature labels.
It should contain the label snapshot and the associated maturity policy.
It should contain the paired scoring query and its denominator.
It should contain a separate count of failed or missing predictions.
It should contain uncertainty assumptions and the resampling unit if one was used.
It should contain slice definitions chosen before final evaluation.
It should contain exploratory analyses labeled as exploratory.
It should contain service-reliability and freshness results measured at the consumer boundary.
It should contain the approved action thresholds and guardrail owners.
It should contain any reasons for deviating from the protocol.
It should contain a record of changes made after inspecting earlier results.
It should contain any known security or privacy restrictions on reproducing the run.
It should contain the approval decision and unresolved objections.
It should not contain unneeded rider or driver identifiers.
If an artifact cannot be shared, the reviewer should know which conclusion depends on it.
If a metric cannot be reproduced, the decision record should say so rather than cite a missing run.
If a source was corrected later, retain the original and revised score identities.
If an independent reviewer disagrees with an exclusion, preserve both interpretations.
If a decision threshold was selected after results were seen, describe the result as exploratory.
If the training window is changed, record whether the original final test has been consumed.
If a model is retrained after approval, treat its new artifact as a distinct release.
If the target changes from submitted requests to completed trips, define a new comparison.
If the action changes from optional guidance to automatic dispatch, review the decision contract again.
If a new region has different data availability, establish a separate eligibility audit.
If label maturity changes, recompute comparable baseline and candidate scores together.
If an external data provider revises its historical archive, preserve the original forecast vintages.
If source access expires, disclose which parts of the prior evaluation are no longer reproducible.
If a numerical example is updated, independently recalculate every derived quantity.
If a published claim is updated, check the original source rather than copying a secondary summary.
If a source no longer supports a claim, narrow or remove that claim instead of presenting it as fact.

## 6. Analysis Walkthrough

The supported decision is to defer replacement until a valid comparative evaluation exists; SRC-01 supports the general evaluation design, not a measured local gain.
The following example is constructed solely to check metric mechanics; it is not Uber data or a completed test.
For four hypothetical zone-hour observations, actual request counts are `[10, 20, 30, 40]`.
The seasonal-naive forecasts are `[8, 24, 25, 42]` and the candidate forecasts are `[9, 18, 32, 37]`.
Baseline absolute errors are `[2, 4, 5, 2]`, totaling 13 requests.
Candidate absolute errors are `[1, 2, 2, 3]`, totaling 8 requests.
Baseline mean absolute error (MAE) is `13 / 4 = 3.25` requests per zone-hour.
Candidate MAE is `8 / 4 = 2` requests per zone-hour.
The absolute difference is `1.25` requests per zone-hour and the relative reduction is `5 / 13`, approximately `38.46%`.
These four constructed observations support only the arithmetic, not statistical significance, deployment, or impact.

| Step | Operation | Output | Interpretation and status |
|---|---|---|---|
| 1 | Define origin and target before joining data | One future hour per zone | Prevents an ambiguous business label; proposed |
| 2 | Reconstruct availability at origin | Eligible historical events only | Guards against future information; not executed |
| 3 | Run seasonal-naive and candidate on same origins | Paired errors | Needed for a fair comparison; not executed |
| 4 | Compute CALC-01 | 3.25 versus 2 MAE | Constructed arithmetic only |
| 5 | Score by time, zone, and demand regime | No real output | Detect aggregate masking; planned |
| 6 | Evaluate intervals and policy separately | No real output | Error improvement is not causal impact; planned |

Sensitivity example: if the fourth candidate forecast had been 46 instead of 37, its fourth absolute error would be 6 and its total error would be 11.
The resulting candidate MAE would be `11 / 4 = 2.75`, versus baseline `3.25`; the illustrative relative reduction falls to `2 / 13`, approximately `15.38%`.
One changed observation thus materially changes this tiny example; it does not estimate real-world uncertainty.
A plausible competing explanation for apparent offline improvement is leakage from later-arriving request records, not a genuinely better forecast.
No approach was actually tested and rejected; model selection remains open.

#### Constructed point-in-time run and comparison

This trace extends CALC-01 without purporting to be a run on real records.
Use four fictional zones named North, South, East, and West.
All four forecasts share the same constructed 12:00 UTC origin and one-hour horizon.
The label window is `[12:00, 13:00)` UTC for each zone.
The weekly-naive values refer to the corresponding hour seven days earlier, using data available by the current origin.
The candidate values are supplied directly; no training process or model family is implied.
The counts in the table are assumed mature solely for the arithmetic exercise.

| Zone | Actual | Previous week | Candidate | Baseline error | Candidate error | Paired improvement |
|---|---:|---:|---:|---:|---:|---:|
| North | 10 | 8 | 9 | 2 | 1 | 1 |
| South | 20 | 24 | 18 | 4 | 2 | 2 |
| East | 30 | 25 | 32 | 5 | 2 | 3 |
| West | 40 | 42 | 37 | 2 | 3 | -1 |
| Total | 100 | 99 | 96 | 13 | 8 | 5 |

The paired improvement is baseline absolute error minus candidate absolute error.
Positive values favor the candidate; a negative value favors the baseline.
West is worse under the candidate even though total candidate error is lower.
The last row sums counts and absolute errors; it is not an additional observation.
The sum of actual requests is 100 in this fictional example.
The sum of baseline forecasts is 99 and the sum of candidate forecasts is 96.
The baseline aggregate forecast misses the aggregate actual by one request.
The candidate aggregate forecast misses the aggregate actual by four requests.
The candidate wins zone-level MAE while losing this particular aggregate-total comparison.
Opposite-signed zone errors can cancel in an aggregate, concealing local shortages.
A consumer acting only on the city total might care more about the aggregate metric.
The business owner must define the action before declaring either metric decisive.
For North, baseline and candidate errors are `|10 - 8| = 2` and `|10 - 9| = 1`.
For South, baseline and candidate errors are `|20 - 24| = 4` and `|20 - 18| = 2`.
For East, baseline and candidate errors are `|30 - 25| = 5` and `|30 - 32| = 2`.
For West, baseline and candidate errors are `|40 - 42| = 2` and `|40 - 37| = 3`.
The average signed forecast error, forecast minus actual, is `-1 / 4 = -0.25` for the baseline.
The average signed forecast error is `-4 / 4 = -1` for the candidate.
These signs indicate underprediction in this constructed set, not measured systematic bias.
The four paired improvements are `[1, 2, 3, -1]`, averaging `5 / 4 = 1.25`.
Their range is from -1 to 3 requests per zone-hour.
The range is descriptive and is not a confidence interval.
No independence assumption is justified for four zones observed in the same hour.
A citywide event could shift all four observations together.
A blocked analysis requires multiple defensible blocks and a prespecified resampling unit.
Even a bootstrap interval misleads if its resampling unit ignores shared shocks or driver movement.
These points cannot estimate a reliable uncertainty interval for real deployment.
A genuine comparison should show distinct days, weeks, zones, and origins as well as row count.
It should report excluded periods and whether exclusions differ by model.
It should score both forecasts against the same mature-label snapshot.
It should report paired differences, not only a ratio of aggregate scores.
It should separate a prespecified decision threshold from one chosen after seeing results.
It should show whether a few high-demand periods dominate the total improvement.
It should identify negative slices even if the overall difference is positive.
It should reserve the final test window from feature selection and repeated tuning.
It should remain inconclusive if uncertainty spans practically important outcomes.

#### Constructed uncertainty example

Consider the same four actual counts and hypothetical 80% prediction intervals.
Use intervals `[7, 12]`, `[18, 23]`, `[25, 34]`, and `[42, 48]`, respectively.
The actual counts 10, 20, and 30 are inside their corresponding intervals.
The actual count 40 is below the fourth interval's lower endpoint of 42.
Empirical coverage in this tiny constructed set is `3 / 4 = 75%`.
The four interval widths are 5, 5, 9, and 6 requests.
The average width is `(5 + 5 + 9 + 6) / 4 = 6.25` requests.
This 75% is not a calibration estimate for a real population.
It is not evidence that an intended 80% interval is miscalibrated.
Four dependent examples cannot resolve whether a nominal coverage difference is meaningful.
The fourth interval demonstrates a lower-tail miss, distinct from an upper-tail miss.
Separate lower-tail and upper-tail miss rates when operational costs are asymmetric.
An interval can be calibrated overall but under-cover in sparse zones or special events.
Report subgroup denominators and uncertainty because small slices vary by chance.
Widening every interval may increase coverage while making guidance less actionable.
Narrow intervals may cause operators to underestimate capacity uncertainty.
If endpoints cross, reject the output before calculating coverage.
If a count forecast is negative, distinguish a raw output from a published feasible count.
Clipping changes the predictive distribution and belongs in held-out calibration checks.
A lower endpoint of zero can be reasonable for a sparse zone.
Consumer semantics must state whether noninteger endpoints are allowed.
The forecast horizon must accompany an interval because uncertainty can vary with horizon.
The interval is for a future request count, not a confidence interval for average model error.
Neither interval automatically captures an unanticipated policy change.
Guidance can change the data-generating process and invalidate old-policy calibration.
Document label delay so immature outcomes do not create spurious misses.
Compare coverage only across predictions with matching eligibility and maturity criteria.
Keep calibration data separate from the final test used for unbiased assessment.
Version any recalibration artifact alongside the model.
For sparse zones, test an appropriately defined pooled group.
Pooling adds data but can hide geographic heterogeneity.
An operator should see a low-support indicator when local evidence is weak.
An interval is an input to a decision, not an instruction to move drivers.

#### Proposed historical replay sequence

First, choose an origin from a predeclared chronological schedule.
Second, resolve the map effective and available at that origin.
Third, select eligible event versions as of that origin.
Fourth, check whether required historical buckets were complete at that origin.
Fifth, retrieve the external forecast vintage available at that origin.
Sixth, calculate features using the fold's transformation revision.
Seventh, generate the weekly baseline from eligible historical values.
Eighth, generate the candidate forecast using the fold's training artifact.
Ninth, write both predictions before reading the mature target label.
Tenth, apply a label-maturity rule equally to both forecasts.
Eleventh, join the mature label to the original prediction key.
Twelfth, calculate paired metrics and record excluded or unmatched observations.
Thirteenth, aggregate by predeclared operational slices.
Fourteenth, retain artifacts sufficient to reconstruct the prediction.
Fifteenth, send a review report without silently promoting the model.
If weekly history is missing, do not replace it with the candidate value.
Apply an agreed alternate baseline or mark the shared comparison ineligible.
If the candidate is missing while the baseline is valid, record candidate failure separately.
Scoring only successful candidate predictions hides a reliability cost.
An availability-adjusted report should count missing candidate outputs against the intended population.
The utility of abstention depends on the cost and quality of the fallback.
For an initial model study, report conditional accuracy and availability separately.
For an operational decision, evaluate the combined forecast-and-fallback policy.
Do not use a mature correction as evidence that the original decision maker could have seen it.
Do not overwrite old score artifacts when a label revision arrives.
Publish a new score version and record why its denominator or outcome changed.
An auditor should be able to reconcile the original score with the later correction.

## 7. The Finding

Do not replace the baseline on the available evidence; first run a point-in-time chronological evaluation and validate the downstream decision separately.
The source supports the importance of chronological comparisons and uncertainty but reports no case-specific accuracy or causal impact for this exercise.
CALC-01 shows how to compute a comparison on fabricated inputs, not that a real candidate improved performance.
The next justified action is to secure appropriate data access and agree on labels, ownership, and a falsifiable evaluation plan.

## 8. System/Solution Design

Everything in this section is proposed; no system implementation, vendor, or configuration is attributed to Uber.

### 8a. Offline / Training / Batch Path

```text
Request events + versioned zone map + available external forecasts
  -> schema / completeness / identity checks -> quarantine invalid partitions
  -> point-in-time hourly aggregates -> rolling training/validation folds
  -> baseline and candidate scoring -> interval and quality gates
  -> versioned model + feature contract + evaluation manifest -> approved registry
                                                -> rejected candidate archive
```

Data engineering owns ingestion and reconciliation; forecasting owns training and model selection; platform owns artifacts and deployment compatibility.
The batch runner produces immutable artifacts and records the cutoff for each fold.
A failed data-quality gate blocks candidate promotion; the baseline may be used only if its own source history passes an agreed freshness check.
A registry entry includes model version, zone-map version, target schema, training cutoff, and supported input schema.
No concrete runtime, hardware, batch schedule, or checkpoint strategy is established by the source.

### 8b. Online / Serving / Real-Time Path

```text
Authorized planner request (origin, zone set, horizon)
  -> authenticate and validate contract -> retrieve approved model and as-of features
  -> check freshness / completeness -> forecast point and interval
  -> policy guard / human review -> response with provenance and expiry
  -> immutable decision log -> mature outcomes -> monitoring and later evaluation
              | stale data, timeout, unavailable model
              v
  approved seasonal-naive fallback if valid; otherwise abstain and alert owner
```

The consumer receives zone, target interval, origin, forecast, interval level and endpoints, model version, feature cutoff, quality status, and expiry.
Requests have an idempotency key so retries do not create multiple logged decisions.
A bounded timeout and retry budget must be set by the service owner from a real consumer contract; no deadline is asserted here.
Freshness is evaluated at the point of use, not merely when a cache was populated.
A cold-start zone lacking sufficient history is flagged; an approved coarser aggregate or abstention is preferable to inventing certainty.
Under peak overload, shed optional requests and serve valid precomputed results only within their expiry, otherwise abstain.
The system records advice and human override separately from driver behavior so later analysis can distinguish eligibility, exposure, and action.

#### Proposed guidance and capacity boundary

The forecast service does not assign a driver or guarantee a completed trip.
An operations tool could present a demand forecast alongside separately sourced supply information.
That combination is a proposed decision aid, not a system described in SRC-01.
The owner must define whether supply means online drivers, idle drivers, available vehicle-hours, or some other measure.
Those measures are not interchangeable because a driver may be online but occupied.
Drivers may also be unable or unwilling to travel to a suggested zone.
The point forecast counts requests, whereas available driver-hours measure capacity in a different unit.
Comparing the two without a service-rate assumption would mix incompatible quantities.
Travel time, trip duration, acceptance, cancellations, and dispatch rules can all affect realized capacity.
Those quantities are not specified in the source for this use.
A capacity model should therefore make its assumptions visible rather than turning requests directly into driver assignments.
An optional guidance recommendation is not a dispatch command.
The driver's decision to accept or disregard guidance must remain observable only where access and consent allow.
The planner must know whether guidance has already been issued for the same window.
Repeated advice can create duplicate exposure even if the forecast request itself is idempotent.
The decision log should identify the policy version and the target zone set.
The log should distinguish generated advice, approved advice, delivered advice, and acknowledged advice.
Failure at one stage must not be recorded as success at a later stage.
If no driver receives advice, the forecast may still be evaluated for accuracy but cannot have the proposed exposure effect.
If advice is delivered late, classify the delay relative to the decision window.
Late exposure might move drivers after the relevant demand has passed.
If a recommendation is revoked, the revocation should be linked to its original decision.
A later audit must not infer that a driver complied merely because the driver entered the recommended zone.
The driver may have independently planned that movement.
Likewise, a driver who stayed put may never have received the recommendation.
The proposed interface should display confidence and data freshness next to the forecast.
It should indicate when an interval is unavailable rather than showing a fabricated default range.
It should provide the weekly-naive comparator so the planner can understand the incremental model output.
The operator should be able to suppress guidance for a zone affected by a road closure.
A suppression decision should have a reason code and a responsible role.
The interface should not suggest that a large point forecast implies a guaranteed number of profitable trips.
The interface should not show individual rider locations as necessary for a zone-level decision.
Geographic aggregation alone does not eliminate all privacy risk in very sparse areas.
Minimum display thresholds, retention, and access controls require policy review rather than invented legal claims.
For a small market, a human can sometimes incorporate local context more cheaply than an automated model.
For a large market, human review capacity can become the limiting resource.
The choice between review and automation must account for error severity and available staffing.
If operators cannot review every zone, the selection rule for review itself needs evaluation.
A review queue based only on high predicted demand may neglect low-volume zones with high uncertainty.
A queue based only on uncertainty may overwhelm staff with cases where no action is possible.
The queue should surface actionable deviations under a policy approved by operations.
The system should expose whether the suggestion came from a candidate or a fallback.
An operator should not have to infer fallback status from a numerical forecast.
The consumer contract must define what to do if only some zones have valid forecasts.
One option is to suppress the entire guidance batch until all required zones are valid.
Another is to proceed with valid zones and visibly exclude incomplete zones.
The second choice risks shifting attention away from excluded areas.
The owner should choose between those options based on decision consequences, not implementation convenience.
The same issue arises if an interval is valid for some zones but missing for others.
Any cross-zone ranking should identify the comparable set and excluded zones.
A ranking can change when a zone is missing even if every remaining forecast stays fixed.
Therefore, a missing high-demand zone cannot be treated as simply absent from the city.
An approved manual override should survive a routine model refresh for its declared duration.
An emergency stop should override both model output and previous manual approval.
Restoring service should not automatically restore an override whose context has expired.
Operators need a record of which model and map version they actually saw.
A dashboard refresh that replaces a forecast without a visible version change creates audit ambiguity.
Human review is a control only when the reviewer has sufficient time and relevant information.
Review throughput and override burden are separate metrics from forecast accuracy.
An override may signal a model problem, local knowledge, or a misunderstanding of the interface.
Investigate these competing explanations before using override rate as a performance score.
The tool should record a correction without forcing the reviewer to disclose sensitive information in free text.
Use predefined reason categories plus a restricted escalation path where details are necessary.
The forecasting team should periodically review override categories with operations.
The operations team should be able to dispute a forecast without needing access to raw individual events.
If guidance changes supply, future request counts may also change through availability or price.
That feedback can alter the target distribution used for subsequent training.
Training data should retain the policy version active when the label was generated.
Policy version alone does not solve confounding because the policy can respond to expected demand.
An operational rollout that coincides with a holiday cannot attribute a demand change to the model.
The organization would need an experiment or defensible quasi-experimental design.
Such a design is not present in this case.

#### Hypothetical capacity arithmetic

Suppose one zone has a forecast of 60 requests in the next hour.
Suppose an operations analyst assumes one available driver can serve at most two requests in that hour.
The arithmetic lower bound under that simplifying assumption is `60 / 2 = 30` available drivers.
The 30 is not an estimated staffing requirement without validating the service-rate assumption.
It ignores repositioning time, pickup distance, trip duration, acceptance, cancellations, and uneven arrival times.
It also assumes every available driver remains available for the entire hour.
Suppose instead the assumed service rate is 1.5 requests per available driver-hour.
The corresponding arithmetic is `60 / 1.5 = 40` driver-hours.
That ten-unit difference illustrates sensitivity to the capacity assumption.
The first quantity is phrased as drivers only because full-hour availability was assumed.
The second is driver-hours and should not be directly interpreted as a driver count without a schedule.
If each driver were available for only half an hour, 40 driver-hours would require 80 such half-hour availabilities.
This does not imply 80 distinct people if one person can provide multiple non-overlapping availabilities.
A system that cannot estimate service time should not claim a precise supply requirement.
An uncertainty interval for demand would produce a range of capacity calculations under fixed service assumptions.
For example, with a demand range of 45 to 75 and service rate of 1.5, the arithmetic range is 30 to 50 driver-hours.
That range ignores uncertainty in the service rate itself.
Correlations between demand and service time can make independent adjustments misleading.
High demand might coincide with congestion or a large event that changes trip lengths.
Road closures can reduce access to the very zone where demand is forecast.
Driver location at the start of the horizon affects how much of the hour remains for service.
A nearby available driver and a distant available driver are not equivalent units of immediately usable capacity.
Any actual repositioning estimate requires a routing and travel-time model outside this case.
The source does not provide such a model or an ETA configuration.
A planner should also consider the cost of drawing drivers away from adjacent zones.
Treating each zone as an independent capacity pool can double-count the same driver.
The same driver cannot provide a full hour of simultaneous supply in two zones.
Forecasting demand in both zones does not by itself allocate scarce supply between them.
An optimization model would need feasibility, fairness, safety, and business constraints approved by the relevant owners.
No such optimizer is selected or evaluated in this exercise.

### 8c. Key Design Decisions

| ID | Proposed choice | Alternative | Evidence and boundary | Trigger |
|---|---|---|---|---|
| DEC-01 | Compare against weekly naive | Recent-value or manual rule | Source supports seasonal naive generally; effectiveness here untested | New market or unstable weekly pattern |
| DEC-02 | Version point-in-time aggregates | Recompute from latest corrected tables | Leakage risk favors as-of reconstruction; storage cost unknown | Ingestion or retention change |
| DEC-03 | Human-approved guidance | Automated action | Decision consequences and interference unmeasured | Evidence and policy approval |
| DEC-04 | Return intervals and provenance | Point forecast only | Source discusses the decision relevance of prediction intervals | Calibration or consumer contract change |
| DEC-05 | Abstain when no valid fallback exists | Serve stale result | Avoids presenting stale information as fresh; operational cost unknown | Agreed service-level tradeoff |

### 8d. Architecture artifacts and operational readiness

A request schema must reject invalid zone IDs, unsupported horizons, incompatible model/map versions, and unauthorized callers with distinct error codes.
An event or artifact key must include immutable source identity and version; replay should be idempotent and auditable.
The data owner must define retention, deletion, backup, restore, and reconciliation rules before storing real data.
An illustrative user-facing service-level indicator is the fraction of eligible planner requests receiving a valid, fresh forecast or explicitly identified fallback before the agreed deadline.
Its target and rolling measurement window are not specified; they require an owner and actual operational agreement.
Alert separately on stale inputs, quality-gate failures, fallback rate, request errors, forecast bias, and delayed labels.
A release should first run in shadow, then use a reversible limited rollout only after an approved evaluation and risk review.
Rollback must restore a compatible model, feature schema, and zone-map version together; model rollback alone may not repair incompatible data.
Service identities should have least-privilege access; secrets should not be embedded in manifests or logs.
Cost analysis must include ingestion, storage, training, serving, monitoring, retries, and review labor; prices and actual utilization are unknown.

**Illustrative sizing, not a measurement:** 100 zones, 24 hourly forecasts per day, and 30 days yield `100 × 24 × 30 = 72,000` forecast records.
At an assumed 1,000 serialized bytes per record, the raw payload is `72,000,000` bytes, or 72 MB in decimal units, before indexes, replication, logs, and compression.
At ten times the zones with the same assumptions, raw payload becomes 720 MB; compute, concurrency, network bursts, and storage overhead cannot be inferred from this calculation.
Benchmark end-to-end latency under realistic burst and failure conditions; do not add component p99 values and call their sum an end-to-end p99.
The platform lead owns recovery; operations can stop the guidance workflow; forecasting investigates model-quality regressions.

#### Proposed rollout and recovery playbook

This playbook is a design exercise and does not report an actual deployment.
The first gate is an agreed target definition signed off by the operations and data owners.
The second gate is an access review for every data source and downstream consumer.
The third gate is a reproducible replay showing origin-time feature availability.
The fourth gate is a chronological comparison against the predeclared baseline.
The fifth gate is an independent review of intervals and operational slices.
The sixth gate is a service readiness review covering capacity, timeout, fallback, and rollback.
The seventh gate is a privacy and security review of interfaces, storage, and logs.
The eighth gate is an operations review of the guidance workflow and human override.
The ninth gate is an explicit decision about whether shadow deployment is appropriate.
Passing an offline metric alone does not pass all the other gates.
In shadow mode, the candidate receives inputs but cannot influence driver-facing guidance.
The existing operational decision remains the control path during shadow mode.
The shadow output should be logged with the same origin and version fields as a real output.
Shadow mode can reveal serving skew and missed deadlines under actual traffic.
Shadow mode cannot by itself measure the behavioral effect of advice that nobody sees.
Compare shadow features to a separately reconstructed as-of record for sampled origins.
Compare shadow predictions to offline replay using the same model bundle and inputs.
Investigate discrepancies before attributing them to numerical nondeterminism.
Check whether the online service used the same timezone and map version as training.
Check whether the online service applied the same missing-value treatment.
Check whether the service loaded the correct model and calibration artifacts together.
Check whether a cache returned a forecast from a previous origin.
Record failed shadow requests as failures rather than omitting them from accuracy analysis.
Set a minimum observation window based on relevant seasonal and workload variation.
The actual duration must be chosen by the owners; none is asserted here.
Before a canary, define eligible geography and excluded populations.
Ensure the interface can distinguish canary from ordinary outputs in audit records.
Keep model promotion separate from policy activation.
An approved model artifact should not automatically authorize a change to driver guidance.
Choose a canary assignment scheme that does not silently invalidate the intended comparison.
When treatment can spill across boundaries, a geographically isolated canary may still affect nearby controls.
Record those boundaries and consider buffer zones in the study design.
Buffer zones consume sample and may not prevent all spillovers.
If exposure cannot be measured, do not infer compliance from assignment alone.
If a safety concern appears, operations may suspend guidance before the statistical study finishes.
The predetermined stopping rule should distinguish a safety stop from an efficacy conclusion.
The experiment owner should record every early stop and its rationale.
If a canary is paused, preserve the current artifact and exposure logs for later analysis.
Do not continue a treatment merely to obtain statistical power after an urgent safety signal.
If a canary is inconclusive, retaining the baseline is a valid decision.
If it passes, review the practical magnitude and subgroup tradeoffs before expansion.
The decision record should name the owner, date, evidence, scope, and unresolved risks.
Any expansion changes the interference and load profile and therefore needs its own review.
An apparently safe low-volume canary does not establish peak-load behavior.
An apparently accurate urban canary does not establish performance in a sparse region.
The proposed rollback unit is the model, calibration artifact, feature contract, and zone map bundle.
Store the previous approved bundle until the recovery policy permits its removal.
Before rollback, verify that its required feature inputs remain available.
An older model may fail after a producer removes an older schema.
If no compatible old bundle exists, abstention may be safer than a broken rollback.
Recovery should restore forecast validity, not just return an HTTP success response.
Reconciliation should check that skipped or duplicated decisions are identified after recovery.
The platform owner should maintain a recovery contact and escalation route.
The forecasting owner should assess whether post-recovery predictions are still calibrated.
The data owner should validate that any replay preserved as-of semantics.
The operations owner should decide when guidance can resume.
The security owner should assess whether an incident involved unauthorized access.
No one role should silently certify all of these independent concerns.

#### Worked operational scenarios

Scenario A begins with an event source that stops reporting while the service stays reachable.
The freshness check identifies the last complete event bucket.
The API returns an explicit degraded state rather than treating missing records as zero.
If the approved weekly baseline remains valid and within its own freshness policy, the service may return it.
If it depends on the same unavailable source and cannot be verified, the service abstains.
The dashboard differentiates source outage from model computation failure.
The data owner investigates upstream delivery while the platform owner monitors consumer errors.
Recovery requires a completeness check, not merely a resumed heartbeat.
The pipeline must reconcile late records before publishing revised labels.
The audit log should preserve which decisions were made during degraded operation.
Scenario B begins with the model registry becoming unavailable.
The service may use an already loaded approved bundle if its validity and expiry remain satisfied.
It must not load an unverified local file merely because the registry is unavailable.
If the cached bundle expires and no approved fallback exists, the service abstains.
The platform owner diagnoses registry access and verifies artifact integrity after restoration.
The operations owner is informed that a previously valid forecast may not cover the next origin.
Scenario C begins with a new map version activated before the model supports it.
The interface rejects the incompatible bundle rather than joining zone IDs by name alone.
The map owner and forecasting owner determine whether an approved compatible version exists.
If safe, the platform reverts the complete compatible bundle.
If not, the consumer sees an explicit unavailable state.
Historical reports retain the original map identifier and are not rewritten in place.
Scenario D begins with an external forecast provider sending a revised file without its issue time.
The feature pipeline cannot verify whether the file was available at the origin.
The affected feature is quarantined rather than retroactively inserted into historical records.
An approved missing-feature path may operate if it was trained and evaluated for that condition.
Otherwise, the candidate abstains or falls back.
The provider and data owner must repair the vintage contract before promotion resumes.
Scenario E begins with a sudden tenfold increase in forecast requests.
The service enforces an agreed concurrency limit to protect core traffic.
Optional refreshes can be shed before critical planner requests if the contract permits.
Retries require backoff and a bounded budget so they do not amplify overload.
Expired cached outputs remain expired even when the service is overloaded.
The response distinguishes overload from invalid input and authorization failure.
The operator sees whether the latest valid forecast remains usable for the current origin.
Capacity owners inspect queue depth, rejected calls, worker saturation, and backend latency.
The example does not claim that any real service has been load tested at this level.
Scenario F begins with a valid forecast but a failure delivering optional guidance.
The audit record marks forecast success and guidance-delivery failure separately.
Retrying delivery must use an idempotency key tied to the same advice version.
If delivery occurs after its useful window, the consumer should reject it as expired.
The experiment analysis must not classify failed delivery as confirmed exposure.
The operations owner can suspend the channel while the messaging owner investigates.
Scenario G begins when an operator reports unusually frequent overrides.
First check whether the interface displays the current origin and correct local time.
Next compare overrides to source freshness and interval width.
Then ask whether local events were known to the operator but unavailable to the model.
Finally examine whether overrides improved retrospective error without treating them as randomized evidence.
A successful override on one event does not prove that all manual interventions are beneficial.
The review should record disagreement and preserve the original forecast.
Scenario H begins when a deletion request affects source records retained in an aggregate.
The data owner consults the applicable retention and deletion policy.
The system identifies affected derived artifacts through lineage.
Any recomputation or deletion must follow the approved policy and preserve required audit records.
The service may need to retrain or invalidate artifacts if the policy requires it.
No legal deadline or jurisdiction-specific outcome is asserted here.

## 9. Evaluation & Validation

No evaluation has been executed; the following is a proposed protocol, not an observed result.
Primary metric: `MAE = sum(|actual - forecast|) / N`, in requests per zone-hour, on the same eligible zone-hour population for baseline and candidate.
The denominator N is the number of eligible, mature, complete zone-hours, not the number of riders or independent experiments.
Specify chronological training, tuning, and final test boundaries before analysis; the actual calendar dates and sample size are unknown.
Use both sliding and expanding histories as sensitivity analyses where sufficient data exist, fitting preprocessing inside each fold.
Select a practical improvement threshold and guardrail limits with the decision owner before inspecting final test outcomes; no threshold has been approved.
Report paired error differences by zone, horizon, demand level, weekday, and disruption regime as well as overall MAE.
If weighting by demand or operational cost is desired, define weights independently of the candidate and report the unweighted result too.
For interval calibration, calculate the fraction of mature actual counts between the stated lower and upper bounds, inclusive, and report interval width and subgroup coverage.
Coverage alone is insufficient because arbitrarily wide intervals can cover most outcomes without being useful.
Dependence across adjacent time and space makes an independent-row confidence interval inappropriate; define a defensible block or cluster resampling scheme after inspecting the design.
Require data-completeness, privacy, service-reliability, latency, and operational guardrails; thresholds remain unapproved.
Backtests evaluate forecasts under the historical policy, not what would have happened under a new driver-guidance policy.
A proposed experiment must predefine eligibility, assignment unit, exposure, sample-ratio checks, duration, stopping rules, and primary and guardrail outcomes.
Cross-zone movement can contaminate controls; consider larger geographic clusters or time-based designs and estimate spillovers before claiming a causal effect.
Neither independence nor absence of interference should be assumed merely because individual requests can be randomized.
No experiment artifact, power calculation, data access, pass/fail decision, or deployment approval exists.

#### Proposed comparison protocol: decisions before analysis

The following checklist turns the evaluation proposal into specific decisions to be made before accessing a final test.
Name the geographic area and define its zone-map version.
Name the decision owner and the operations representative who will consume the result.
Define the UTC origin schedule and the corresponding local calendar interpretation.
Define the target horizon and the treatment of events exactly on interval boundaries.
Define the population of valid request events, including cancellations and test traffic.
Define which source quality conditions make a zone-hour eligible.
Define a separate category for unknown labels rather than silently excluding them.
Define the label maturity cutoff and treatment of late corrections.
Define the baseline when the weekly comparison bucket is complete.
Define what happens when that historical bucket is absent or revised.
Define whether a secondary recent-value baseline is required.
Define the candidate's input contract and the allowed source vintages.
Define whether the candidate produces a median, mean, or another point functional.
If MAE is primary, compare forecasts that are appropriate for absolute-error loss.
Define the exact data transformations that are fitted within each fold.
Define how model hyperparameters are selected without reading the final test.
Define the rolling-origin schedule independently of the observed outcomes.
Define whether training windows slide, expand, or use both for sensitivity.
Define how much historical data each baseline and candidate can use.
If histories differ, explain why the comparison remains operationally meaningful.
Define a minimum history condition for a new zone.
Define whether a parent-zone fallback is allowed and how it is scored.
Define the primary metric's numerator, denominator, and units.
Define the practical margin that would justify the additional operational complexity.
Define a policy for an exact tie or an improvement smaller than that margin.
Define a policy for a metric improvement paired with unacceptable coverage.
Define a policy for a metric improvement paired with an availability regression.
Define a policy for a metric improvement concentrated in one high-volume zone.
Define a policy for a model that improves averages while harming a protected operational slice.
No legal protected-class conclusion is implied by the phrase operational slice.
Define the relevant slices before inspecting the final test.
Report exploratory slices separately so post-selection is visible.
Define the minimum slice size needed for a useful descriptive report.
Do not hide smaller slices merely because the confidence interval is wide.
Instead mark them as underpowered and avoid a definitive subgroup conclusion.
Define how correlated observations will be grouped for uncertainty estimation.
Check whether common shocks make a single zone-day block too small.
Check whether temporal dependence requires blocks longer than a day.
Check whether cluster resampling has enough independent clusters to be credible.
If no credible uncertainty estimate is possible, report the limitation and defer a strong claim.
Define the expected treatment of missing candidate forecasts.
Define the availability denominator from all eligible forecast opportunities.
Define an operational evaluation of the candidate plus fallback, not just successful model calls.
Define how to report latency without excluding timeouts.
Define the freshness deadline measured from the consumer's perspective.
Define how different label snapshots affect the score version.
Define the owner of the reproducibility manifest and artifact permissions.
Define who can inspect the final test and who can authorize model changes afterward.
If the final test influences another model iteration, reserve a new future test.
Define who adjudicates disagreements over data exclusions.
Define the date and event that trigger a re-evaluation.

#### Intervention design and interference

An intervention study would answer a different question from a forecast backtest.
The intervention is a change in optional driver guidance, not merely a different forecast number.
The estimand must specify the population, treatment assignment, comparison, and time window.
For an intention-to-treat estimand, compare outcomes by assigned policy regardless of subsequent compliance.
That estimand measures the policy as offered, subject to the design's assumptions.
A per-protocol comparison among compliers can be biased because compliance is not generally random.
The guidance-delivery log is needed to distinguish assignment from exposure.
The analysis should preserve drivers and zone-hours that were eligible but never exposed.
The analysis should report whether delivery failures differ by assignment.
If drivers move across treatment and control zones, one driver's response can change another zone's supply.
That movement violates a simplistic assumption that each zone's outcome depends only on its own assignment.
Larger geographic clusters may reduce some spillovers but also reduce the number of independent units.
Fewer clusters can make uncertainty estimates unstable even when many trips are observed.
Buffer zones may reduce cross-boundary interaction but change the eligible population.
The size of a useful buffer depends on movement patterns and the decision horizon.
No movement data are available here to select a defensible buffer size.
Alternating treatment by time can reduce geographic contamination but introduces time-varying confounding.
Day-of-week patterns, weather, events, and policy changes may coincide with treatment periods.
Randomized switchback periods can help when the design accounts for serial carryover.
Carryover matters if drivers remain in a zone after the assigned period ends.
A washout period can reduce carryover but also reduces the usable sample.
The required washout length would need evidence about movement and persistence.
Neither a switchback nor a cluster design should be presented as automatically unbiased.
Randomization should occur before exposure and follow a documented allocation mechanism.
The experiment should record the randomization unit and the assignment timestamp.
Check assignment balance at the randomization unit, not only at the event level.
Investigate a sample-ratio mismatch before interpreting treatment effects.
Check whether eligibility changed after assignment in a treatment-dependent way.
Changing eligibility after seeing treatment can create selection bias.
Record the denominator for rider and driver guardrails separately.
A driver outcome cannot use request counts as its denominator without a clear estimand.
A rider outcome cannot use only completed trips if treatment changes completion probability.
Report the number of assigned clusters or periods alongside the number of requests.
Power analysis must use assumptions about variance, clustering, interference, and minimum effect.
Those assumptions are unknown here and no power claim is made.
Predefine a maximum duration or an approved sequential analysis procedure.
Repeatedly checking a fixed-horizon test and stopping when favorable can inflate false positives.
Emergency safety stopping remains distinct from informal efficacy peeking.
An increase in requests may reflect a pricing or supply change rather than increased underlying intent.
If pricing changes concurrently, the experiment should measure and account for the policy interaction.
An exposure effect estimated under one policy may not transport to a different pricing regime.
Monitor whether drivers repeatedly cross experiment boundaries.
Monitor whether a cluster's outcomes respond to treatment in adjacent clusters.
The latter check can reveal interference but is not by itself a complete correction.
If interference is too severe to identify the intended estimand, redesign the experiment or narrow the claim.
Do not reinterpret an inconclusive test as proof of no harm.
Do not interpret a favorable operational proxy as proof of increased driver earnings.
Earnings require their own definition, denominator, observation window, and approved access.
Driver-hours spent repositioning are a potential cost even if completed trips increase.
Rider wait and cancellation metrics may move differently from driver outcomes.
The experiment owner should review tradeoffs with the operations and policy owners.
An effect estimate should include uncertainty and the scope to which it applies.
An estimate from a pilot geography is not automatically representative of all markets.
If a rollout alters the distribution of demand, reassess both forecast accuracy and intervention effects.
Use policy-version labels to identify when historical training data were generated under different guidance.
Keep the model version separate from the guidance-policy version in every evaluation record.
A model can remain unchanged while the operational policy changes.
A policy can remain unchanged while the forecast model changes.
Changing both simultaneously complicates attribution to either component.
Where feasible, isolate the changes or design a factorial comparison with adequate power.
If isolation is infeasible, report the combined intervention rather than claiming a model-only effect.

## 10. Action Taken

No action taken as of 2026-11-07 for this illustrative case.
The recommendation is to agree the target and access rules, then run a reproducible shadow comparison before considering any policy experiment.
The hypothetical operations owner would approve a pilot; data, forecasting, platform, and privacy owners would sign off their own controls.
There is no release date, actual deployment, recorded approval, or handoff artifact.
A future stop rule should include invalid or stale input, safety concerns, and pre-agreed guardrail regression; operations must be able to suspend guidance.
ETA prediction and automated driver dispatch are excluded and remain separate decisions.

## 11. Outcome & Results

**Mixed evidence declaration:** Published historical claims and constructed illustrative calculations are separate; figures in the calculation are not measurements of any named organization.

| Metric | Baseline | Candidate / after | Change and uncertainty | Evidence and limitation |
|---|---|---|---|---|
| Real held-out MAE | Not measured | Not measured | Unknown | No dataset or run exists |
| Real policy effect | Not measured | Not measured | Unknown | No controlled intervention exists |
| Interval coverage | Not measured | Not measured | Unknown | No forecast intervals or mature labels exist |
| Service reliability and cost | Not measured | Not measured | Unknown | No deployed system exists |
| CALC-01 illustrative MAE | 3.25 requests per zone-hour | 2 requests per zone-hour | 1.25 lower; no empirical uncertainty | Four constructed examples, not an outcome |

Outcome not yet measured; the proposed forecasting lead would score mature labels over a pre-agreed chronological test window and return results to the operations owner for a go/no-go decision.
No duration, sample size, adoption, cost, or realized economic gain can be reported.

## 12. Failure Modes & What Went Wrong

No incident is documented; each row is a hypothetical failure and a proposed response.

| ID | Trigger and symptom | Discriminating check | Containment and prevention | Owner / verification |
|---|---|---|---|---|
| F-01 | Candidate suddenly looks implausibly accurate | Compare feature availability and event ingestion times with origins | Block promotion; repair as-of joins | Data lead; replay late-event fixture |
| F-02 | Errors increase in all zones | Compare raw ingestion delay and missing partitions to genuine request counts | Quarantine bad partitions, use valid fallback or abstain | Data lead; reconciliation test |
| F-03 | Errors increase only near map boundaries | Compare map version and spatial join cardinality | Revert compatible map/model bundle | Geospatial lead; historical boundary replay |
| F-04 | Aggregate coverage appears healthy but sparse zones under-cover | Stratify coverage and widths by volume and region | Flag affected segments, review uncertainty method | Forecast lead; held-out subgroup check |
| F-05 | Good offline error coincides with poor driver outcomes | Compare exposure and behavior, inspect interference and policy changes | Suspend guidance pending experiment | Operations lead; causal design review |
| F-06 | Fallback produces expired predictions | Inspect source age and fallback provenance | Abstain instead of silently serving stale data | Platform lead; failure-injection test |
| F-07 | Planner overrides rise | Review explanations, timestamps, and event coverage with users | Enable manual control and investigate mismatch | Operations lead; blinded review of cases |
| F-08 | Labels revise after scoring | Compare error by maturity lag and data revision | Delay scoring and rerun affected reports | Data lead; maturity audit |

No failed mitigation or alarm failure is claimed; those require an actual incident or replay.

#### Diagnostic drills for hypothetical incidents

The drills below are proposed investigation exercises, not incidents reported by the source.
Each starts from an ambiguous symptom and distinguishes at least two plausible mechanisms.
They are intended to prevent a model change from being used as a substitute for diagnosis.

**Drill A: forecast errors improve immediately after a pipeline migration.**
One hypothesis is that the new pipeline eliminated stale features.
Another is that the new backtest used corrections unavailable at each historical origin.
Compare the availability timestamps used by the old and new feature jobs.
Check whether the evaluation origins and target labels changed at the same time.
Reconstruct a sample using a frozen origin-time snapshot.
Look for records whose event time is before the origin but whose ingestion time is after it.
If those records enter the feature set, the apparent improvement is not a valid comparison.
If they do not, check whether population exclusions changed.
The data lead should block promotion until the comparison uses a common eligible set.
The prevention check is a late-arrival fixture that fails when post-origin data enter features.
Passing that fixture alone does not establish that every feature is leakage-free.

**Drill B: both baseline and candidate errors rise in the same hour.**
One hypothesis is a real marketplace shock affecting all forecasters.
Another is a data loss or duplicate-ingestion incident affecting the observed label.
Inspect source completeness, ingestion lag, and deduplication rates.
Compare preliminary counts with later mature revisions.
Inspect whether the error is concentrated in a single producer or geography.
Ask whether a major public event was announced before the origin and available to the system.
An event learned only afterward may explain error but cannot retroactively validate a leaked feature.
If the label itself is incomplete, delay scoring rather than retraining against a bad target.
If the shock is real, check whether intervals widened or missed in the same direction.
Operations should retain manual control while the data owner establishes input validity.
The prevention check is an alert on completeness distinct from a model-error alert.

**Drill C: only the candidate deteriorates while the baseline remains stable.**
One hypothesis is a candidate model regression.
Another is a training-serving transformation mismatch.
Compare online and replayed feature vectors for identical origin keys.
Verify the loaded model checksum and the calibration artifact version.
Inspect whether a feature default changed at deployment.
Check whether categorical values previously seen in training became unknown at serving.
If feature values disagree, restore the compatible transformation bundle before retraining.
If features agree, inspect recent data distribution and segment-specific residuals.
Rollback requires a compatible model and schema, not just a model version number.
The forecasting lead owns model investigation; the platform lead owns artifact delivery.
The prevention check is a replay of selected production inputs through the release candidate.

**Drill D: a citywide total looks accurate while individual zones are badly wrong.**
One hypothesis is error cancellation between overpredicted and underpredicted zones.
Another is a zone-map join that allocates requests to the wrong boundaries.
Compare the sum of signed errors with the sum of absolute errors.
Check whether neighboring zones show opposite-signed changes at the map transition.
Reconcile each aggregate to its historical map version.
If the geography changed, avoid comparing zone IDs as if their shapes were fixed.
If the geography is stable, inspect spatial features and local event coverage.
Operations should avoid ranking zones by a suspect local forecast.
The geospatial lead should validate the boundary mapping against raw event locations where authorized.
The prevention check is a versioned boundary-crossing fixture with expected allocations.

**Drill E: interval coverage drops while point MAE is stable.**
One hypothesis is that residual variance increased without changing the central forecast much.
Another is that the interval-calibration artifact was not deployed with the point model.
Check the versions of both artifacts in each response.
Compare interval width and tail misses by zone and hour.
Check whether post-processing clipped one endpoint unexpectedly.
Verify that label maturity and population filters match the previous coverage report.
If the artifact pair is incompatible, roll back the complete bundle.
If variance shifted, evaluate recalibration on a separate recent window.
Do not recalibrate on the untouched final test and then report that same test as independent.
The forecasting lead should quantify coverage uncertainty before declaring a stable recovery.

**Drill F: reported coverage improves dramatically after widening intervals.**
One hypothesis is genuine correction of previously narrow intervals.
Another is that the intervals became too broad to support any useful decision.
Report average and slice-specific widths alongside coverage.
Compare lower-tail and upper-tail misses separately.
Ask the operator whether the widened range changes the recommended action.
If every zone now requires the same action, the interval may have little decision value.
An apparently high coverage percentage is not sufficient to authorize rollout.
Check whether low-volume zones dominate the count of covered observations.
Review width-weighted or decision-specific diagnostics only after defining their purpose.
The operations owner and forecasting lead should agree on useful uncertainty reporting.

**Drill G: a canary shows more completed trips but driver outcomes worsen.**
One hypothesis is that drivers spent more time repositioning without adequate compensation.
Another is a shift in the composition of drivers observed in treatment and control.
Check assignment, delivery, exposure, and eligibility denominators.
Inspect driver-hours, repositioning time, and outcome distributions if approved data exist.
Check whether treatment affects who remains active and therefore appears in the outcome sample.
Review spillovers into adjacent control zones.
Do not infer earnings from completed-trip count.
If a predeclared guardrail is breached, the operations owner can suspend guidance.
The experiment owner should preserve the assignment record and explain any early stop.
The result would remain specific to the tested policy, population, and period.

**Drill H: errors appear only around daylight-saving transitions.**
One hypothesis is an incorrect conversion between UTC and local time.
Another is a valid behavioral shift around a local clock change.
Inspect the stored UTC origin and its mapped local offset.
Check repeated local hours when clocks move backward.
Check nonexistent local hours when clocks move forward.
Verify that the baseline selects a comparable historical hour under a documented rule.
Do not use an ambiguous local timestamp as a unique forecast key.
The data lead should reproduce the conversion with the map and timezone version used at the origin.
An origin key based on UTC remains unique even when a local hour repeats.
The prevention check includes both repeated-hour and missing-hour calendar fixtures.

**Drill I: a new zone has consistently empty historical features.**
One hypothesis is that the zone is genuinely new.
Another is that a map-version join silently dropped records.
Inspect the zone creation date and the lineage of the map crosswalk.
Check whether nearby historical events can be assigned to the new geography under an approved rule.
Do not claim exact reconstruction from a coarse many-to-many crosswalk.
If the zone has no reliable history, mark it as low support.
A parent-region fallback may help if its limitations and weighting are evaluated.
The fallback must not duplicate the same regional demand across multiple child zones.
The operator should see that the output is a fallback rather than a locally trained forecast.
The geospatial and forecasting owners should approve the cold-start policy together.

**Drill J: repeated retries create duplicate guidance.**
One hypothesis is that the consumer retried after losing an acknowledgement.
Another is that the producer generated multiple advice IDs for the same policy decision.
Trace the request ID, advice version, delivery attempt, and acknowledgement.
Check whether the idempotency key is stable across retries.
Distinguish an intentional revised recommendation from a duplicate transmission.
If a duplicate reached users, suspend retries until the deduplication rule is verified.
Record whether any duplicate was delivered after the target window began.
The platform owner repairs delivery semantics; operations owns user-facing communication.
The prevention check injects an acknowledgement loss and verifies a single logical delivery.
The test must also verify that a legitimate revised advice version can still be delivered.

**Drill K: a model version is rolled back but errors remain elevated.**
One hypothesis is that the underlying data regime changed.
Another is that the old model is still receiving new incompatible features.
Check the full bundle rather than the model ID alone.
Inspect feature schema, map version, preprocessing, and calibration version.
Verify that cached outputs were invalidated where the rollback contract requires it.
Check that the supposed rollback artifact was actually loaded by all serving instances.
If only some instances changed, compare error and version by instance.
If all components are compatible, investigate whether the baseline also deteriorated.
A persistent citywide shock may affect both old and new models.
The recovery decision should consider data validity and operational impact, not version status alone.

## 13. Transferable Lesson

The `evaluation-mismatch` pattern is to separate a forecast's statistical fit from the quality and causal consequences of the decision it informs.
Its hypothesized mechanism is that the data-generation process changes when participants respond to guidance, so historical accuracy alone cannot identify the policy's effect.
That mechanism is a design concern here, not a demonstrated marketplace effect.
The approach requires a clearly defined decision, historical as-of data, a credible baseline, mature labels, and a way to govern or measure actions.
Avoid assuming that a model is useful when the target is heavily censored or the policy cannot be compared without serious interference.
A portability test is to ask whether the new domain can reconstruct what was known at the forecast origin and whether its action changes the future target.
The comparable [retail replenishment case](01_Retail_CPG_Demand_Forecasting_At_Scale.md) has a related evaluation mismatch, but inventory stockouts and supplier lead times differ from mobile supply and cross-zone spillovers.

## 14. Connections & Cross-References

### 14a. Related cases, theory, and cheat sheets

| Relationship | Target | Shared mechanism and boundary |
|---|---|---|
| contrasts-with | [Retail CPG demand forecasting](01_Retail_CPG_Demand_Forecasting_At_Scale.md) | Both require decision-aware demand evaluation; inventory and mobile supply have different constraints |
| uses-method | [Time-series cross-validation](https://otexts.com/fpp3/tscv.html) | Training precedes each test origin; a policy effect needs more than predictive evaluation |
| historical-source | [Uber forecasting introduction](https://www.uber.com/us/en/blog/forecasting-introduction/) | Explains baselines, backtesting, and intervals; does not document this proposed implementation |

### 14b. What this case leaves open

| Question | Missing evidence | Resolution | Priority / owner |
|---|---|---|---|
| Are historical origin-time features reconstructable? | No event logs or availability history | Audit snapshots and late-arrival distributions | Before backtest / data lead |
| Does request count represent the decision need? | No access to latent demand or operational utility | Validate target and distinguish observed from unobserved intent | Before model selection / operations |
| Does guidance improve outcomes? | No intervention or causal evidence | Design a spillover-aware experiment | Before rollout / experimentation lead |
| Are intervals useful by zone and time? | No interval forecasts or mature labels | Evaluate coverage and width across relevant slices | Before adoption / forecasting lead |

### 14c. Intense interview pack — questions with checked answers

Scoring for every question: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended tradeoff. Follow-ups invite alternative assumptions, not invented experience.

**Q01 / framing:** What is the decision, and why is ETA excluded?
Strong answer: choose whether a next-hour zone demand forecast should inform optional planning, compared with seasonal naive; ETA is a different target and decision.
Clarify zone, origin, action owner, and baseline; acceptable alternative is deferral when the action is undefined.
Follow-up: what if the only consumer needs an individual arrival time? Red flag: treating request counts as ETA. Support: Sections 1–3.

**Q02 / business:** Why can lower MAE fail to improve marketplace outcomes?
Strong answer: guidance changes driver locations, and error weights may not represent rider or driver costs; test outcomes with appropriate guardrails.
Clarify the action and eligible population; a decision-weighted metric may help but is not itself a causal effect.
Follow-up: how would the answer change if forecasts were never acted on? Red flag: equating error reduction with earnings. Support: Sections 3 and 9.

**Q03 / data:** Which timestamps are needed for an as-of join?
Strong answer: event time locates the event and ingestion or availability time determines what was knowable at the origin; retain revisions and map versions.
Clarify the delay and update semantics; an immutable historical snapshot can be an acceptable alternative.
Follow-up: what happens to a late event used in a final label? Red flag: joining latest corrected tables without cutoff. Support: Section 5.

**Q04 / label:** Why distinguish zero requests from missing records?
Strong answer: an empty but complete partition can represent zero, whereas missing telemetry is unknown; mixing them biases both training and evaluation.
Clarify completeness evidence; fallback or abstention is defensible when it is absent.
Follow-up: what if a whole zone becomes unmapped? Red flag: automatically filling every missing bucket with zero. Support: Sections 5 and 12.

**Q05 / method:** Why compare a seasonal baseline on the same origins?
Strong answer: matched origins, geography, label maturity, and exclusions prevent workload differences from masquerading as model quality.
Clarify weekly stability; recent-value naive can be more useful when weekly history is unavailable.
Follow-up: how should a holiday be evaluated? Red flag: comparing scores from different periods. Support: Sections 5–6.

**Q06 / arithmetic:** With baseline absolute errors 2, 4, 5, 2 and candidate errors 1, 2, 2, 3, calculate MAE and relative reduction.
Strong answer: baseline 13/4 = 3.25, candidate 8/4 = 2, difference 1.25, relative reduction 5/13 ≈ 38.46%.
Clarify these are constructed values; report totals instead when equal sample sizes are known, but retain units.
Follow-up: what if the fourth candidate error becomes 6? Red flag: calling the constructed reduction a production result. Support: CALC-01.

**Q07 / sizing:** How many records and raw bytes result from 100 zones, 24 hourly outputs per day, 30 days, and 1,000 bytes each?
Strong answer: 72,000 records and 72,000,000 bytes, or 72 decimal MB, excluding replication and overhead.
Clarify serialization and retention; a measured benchmark is better than a sizing assumption.
Follow-up: what changes at ten times the zones? Red flag: asserting this calculation predicts latency. Support: Section 8d.

**Q08 / intervals:** What does 90% interval coverage mean?
Strong answer: over the evaluated population, 90% of observed targets lie within the specified bounds; report width and subgroup coverage too.
Clarify the window and maturity; a quantile-based interval or a calibrated residual interval may be reasonable if evaluated honestly.
Follow-up: can an arbitrarily wide interval achieve good coverage? Red flag: claiming 90% coverage guarantees each individual outcome. Support: Section 9.

**Q09 / production:** What should happen if the feature store is stale?
Strong answer: use an independently valid approved fallback if available, otherwise abstain and alert; return provenance and expiry.
Clarify the consumer deadline and fallback freshness requirement; serving a still-valid precomputed forecast is another possible design.
Follow-up: what if the fallback uses the same stale dependency? Red flag: silently returning expired values. Support: Section 8b.

**Q10 / rollout:** How do you roll back a model after a zone-map migration?
Strong answer: restore a mutually compatible model, map, and feature schema bundle, or abstain if the old bundle cannot safely operate.
Clarify compatibility and data retention; a forward fix may be necessary if restoration is impossible.
Follow-up: who can stop guidance? Red flag: assuming model weights alone determine correctness. Support: Section 8d.

**Q11 / evaluation:** How should chronological evaluation be structured?
Strong answer: train only on data available before each origin, tune on rolling validation folds, and reserve a later untouched test window.
Clarify sliding versus expanding history; both are reasonable under different stationarity and data-volume assumptions.
Follow-up: why is a random row split suspect? Red flag: fitting transformations on all data before splitting. Support: Sections 5 and 9.

**Q12 / causal inference:** Why may randomized zones still interfere?
Strong answer: drivers can move between zones and change both treatment and control outcomes; cluster design and spillover analysis may reduce or expose contamination.
Clarify assignment, eligibility, exposure, and mobility; time-based designs have their own temporal confounding risks.
Follow-up: what if zones cannot be made independent? Red flag: claiming any randomization automatically eliminates interference. Support: Section 9.

**Q13 / debugging:** All zones show a sudden error increase; what do you investigate first?
Strong answer: distinguish data freshness, schema and label failures from a real demand shift using raw counts, completeness, ingestion lag, and baseline behavior.
Clarify onset and affected versions; a genuine shift remains plausible after pipeline checks.
Follow-up: what if only one map boundary is affected? Red flag: immediately retraining without diagnosis. Support: Section 12.

**Q14 / feedback:** Why log both advice and actual behavior?
Strong answer: eligibility, exposure, compliance, and outcomes differ; without them it is difficult to analyze the downstream decision or diagnose feedback.
Clarify privacy and retention rules; aggregate logs can be preferable if they support the analysis.
Follow-up: what if most drivers never see the guidance? Red flag: treating all eligible drivers as exposed. Support: Sections 8–9.

**Q15 / transfer:** What changes when transferring this approach to retail replenishment?
Strong answer: retain point-in-time evaluation and decision-aware metrics, but model inventory position, supplier lead time, and stockout censoring rather than mobile driver movement.
Clarify decision horizon and operational cost; a simpler rule may be sufficient with limited data.
Follow-up: what if stockout data are missing? Red flag: assuming observed sales equal unconstrained demand. Support: Section 13.

**Q16 / evidence:** What did the historical article establish about this proposed model's impact?
Strong answer: nothing about its accuracy, configuration, ETA, or causal impact; it supplies general historical forecasting context and methods.
Clarify publication date and scope; other primary artifacts could support additional claims if independently reviewed.
Follow-up: does mentioning Omphalos establish this design used it? Red flag: inventing an operational percentage. Support: SRC-01.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Ride-hailing marketplace planning |
| Problem type | Next-hour zone-level observed request forecasting |
| Method | Proposed chronological comparison against weekly seasonal naive |
| Primary result | Not measured; CALC-01 is illustrative arithmetic only |
| Key lesson | Forecast error and causal policy value are distinct |
| Tags | evaluation-mismatch; data-leakage; model-monitoring |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-01-10-002 / 0.1 / 2026-11-07 |
| Evidence level | not-measured; historical source claims and illustrative arithmetic are separately labeled |
| Implementation stage | proposed |
| Baseline / primary metric | Weekly same-hour naive / held-out request-count MAE |
| Key limitation | No real dataset, evaluation, configuration, or policy outcome |
| Review trigger / owner | Target, availability, policy, or source interpretation change / forecasting lead |
