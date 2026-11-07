---
case_id: "CASE-01-10-003"
case_title: "Day-ahead energy load forecasting for utility planning"
case_version: "1.0"
created: "2026-11-07"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Electric utility operations"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["data-leakage", "model-monitoring"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-07"
owner_role: "Utility load forecasting lead"
source_index: "INDEX.md"
review_trigger: "Changes to target boundary, weather availability, settlement revisions, or planning deadline"
---
# Case Study: Day-Ahead Energy Load Forecasting

## 1. The One-Line Framing

Given system-boundary load history, calendar information, and weather forecasts actually available at a fixed cutoff, produce next-day hourly load quantiles for a utility planning desk, compared with a seasonal-naive forecast, without granting the model authority to dispatch equipment.
- The decision is whether to admit a candidate forecast into a human-reviewed day-ahead planning workflow.
- The primary planned success measure is mean absolute error in megawatts over all eligible next-day hours, relative to the same-origin baseline.
- Peak risk, interval calibration, deadline reliability, and operator safety are independent guardrails, not substitutes for that measure.
- The forecast is advisory: operators retain responsibility for commitments, reserves, outages, operating limits, and all market submissions.
- **Evidence:** This is a hypothetical utility, proposed system, constructed examples, and unmeasured outcomes; no utility deployment, incident, or improvement is claimed.
- The unavailable inventory source `DS_Domain/Case_Studies_Production_ML_Systems/TimeSeries_Forecasting/04_Energy_Load_Forecasting.md` supplies a topic title only, not technical content or evidence.
- A single point estimate is too narrow because operational exposure depends on the distribution, hour, peak, forecast vintage, and available decision time.

## 2. Context & Background

The hypothetical utility's planning desk prepares an hourly demand view before a next-day procurement and commitment meeting.
The desk compares the forecast with available supply, contractual arrangements, network constraints, scheduled outages, and reserve requirements.
The proposed tool supplies demand information only; the downstream engineering and market systems remain authoritative for actual actions.
The synthetic service territory has one aggregate balancing boundary and a local civil timezone with daylight-saving changes.
Its intended target is the average megawatt flow into that boundary, net of behind-the-meter generation as reflected in the meter.
This target is not gross customer demand and does not count generation behind the meter separately.
The net boundary can be negative if exports exceed imports; no unconditional nonnegative clipping is justified.
The notional historical process is a human-reviewed same-weekday and same-hour forecast.
No actual baseline accuracy, available data volume, regulatory jurisdiction, or project history is known.
The forecast lead owns methodology; an operations planner owns use of each forecast.
The data engineer owns availability-time records and source reconciliation.
The weather-data owner confirms archive permissions and vintage completeness.
The reliability engineer owns monitoring, recovery, and access controls.
A risk reviewer decides whether an offline candidate may enter shadow operation.
The planning meeting and 10:00 local cutoff are scenario design choices, not descriptions of a named operator.
Actual market deadlines and applicable rules must be established from the relevant operator's primary documents before deployment.

## 3. The Business Question

**Illustrative stakeholder paraphrase:** Can the planning desk receive a defensible next-day hourly demand range before its meeting, including credible peak risk, without weakening existing operational controls?
- Decision owner: operations planning manager, with risk and system-operations approval.
- Trigger: a proposed replacement of a manual planning input; no actual triggering event is asserted.
- Decision deadline: 10:15 local each operating day; data freeze at 10:00.
- Options: retain the baseline; run the candidate in shadow; or permit advisory use after gates are met.
- Unit of decision: one forecast issuance for one balancing boundary and its next local operating day.
- Primary population: every eligible hour in the day-ahead horizon, including 23- and 25-hour local days.
- Objective: lower paired hourly absolute error on a locked future period, without violating peak, uncertainty, or reliability gates.
- Cost of underforecast: potential expensive balancing, operational stress, or inadequate preparation, depending on the actual procurement and reserve process.
- Cost of overforecast: possible excess commitments or procurement and unnecessary expense; actual tariffs and resource costs are unknown.
- Safety guardrail: the tool cannot issue dispatch, reserve, breaker, or market commands.
- Scope excludes feeder dispatch, unit commitment optimization, demand response control, individual-customer forecasting, and causal attribution of savings.
- A forecast score is a proxy, not a realized business outcome: a downstream simulator and prospective operational evaluation are needed to value decisions.
- If the data archive cannot reconstruct information at issuance, defer selection rather than claim a valid historical improvement.

## 4. Assumptions & Constraints

All entries below are unverified scenario assumptions or proposed constraints, not measured capabilities.
| ID | Assumption or constraint | Type | Check and owner | Failure consequence and fallback |
|---|---|---|---|---|
| A1 | Load measurements have event and ingestion timestamps | Data, invalidating | Data engineer audits immutable event history | If missing, use conservatively lagged snapshots or abandon vintage-valid backtesting. |
| A2 | Weather issue and actual receipt times are archived | Data, invalidating | Weather owner checks issue manifests and late-arrival logs | If absent, do not substitute realized weather; evaluate an explicitly weather-free model. |
| A3 | Meter boundary is stable or versioned | Data, invalidating | Operations reconciles mapping against settlement records | Boundary change requires consistent reaggregation or separate regimes. |
| A4 | Mature labels can be identified without contaminating features | Data | Metering owner documents revision schedule | Report provisional and final scores separately and exclude immature labels from final scoring. |
| A5 | Adequate historical extreme-weather periods exist | Statistical | Analyst counts independent events by season | Report low effective sample and avoid confident tail claims. |
| A6 | Calendar, holiday and timezone versions can be reconstructed | Data | Data engineer checks historical calendar and timezone database | Recreate original mappings or exclude unresolvable origins. |
| A7 | The desk can review distributions by deadline | Workflow | Operations observes a timed rehearsal | Use the existing authorized manual process if review is unavailable. |
| A8 | Historical operator overrides are recorded | Workflow | Planning lead audits decision log | Compare forecast quality only; do not infer policy impact. |
| C1 | Forecast has no write permission to operational controls | Safety | Security reviews identities, ACLs and network routes | Block release on unexpected command path. |
| C2 | Output must arrive by 10:15 local | Scenario service requirement | Reliability team measures time-to-publish | Escalate and use a clearly marked manual fallback. |
| C3 | Customer-level information is excluded | Privacy design choice | Data owner inspects fields and roles | Redesign aggregation and approval if granular data is required. |
| C4 | Expenditure requires planning approval | Business | Finance establishes actual limits | No invented price or fixed vendor contract is assumed. |
| C5 | Applicable grid and market requirements are verified locally | Governance | Legal and operations identify jurisdiction and sources | No automation until responsible authorities approve the workflow. |
Critical invalidators are A1, A2, and A3: they respectively threaten historical causality of inputs, weather realism, and target comparability.
A substitute dataset may restore feasibility only if its availability and semantic contracts are independently demonstrated.

## 5. Data, Tools & Method

### 5a. Data Specification

The dataset grain is `(boundary_id, forecast_origin_utc, target_interval_start_utc)`.
A row describes one target interval at one forecast origin; the same realized hour can occur in many forecast origins.
The interval is left-closed and right-open, with average power in megawatts over the interval.
Energy in megawatt-hours equals average megawatts times interval duration in hours, not simply the raw measurement when duration differs.
All joins and unique keys use UTC instants; the local date, hour, UTC offset and fold indicator are retained for operations.
The desired next-day set is all intervals belonging to the next local civil day; it can contain 23, 24 or 25 hourly intervals.
For an illustrative 10:00 freeze, a load value with event end 09:00 and receipt 09:12 is eligible.
A load value ending 09:00 but first received at 10:02 is not eligible, even if its event timestamp precedes 10:00.
A revision received after 10:00 cannot retroactively become an input for that origin.
A finalized settlement label may be used for later scoring, but never as an earlier input unless it was already available then.
Keep raw meter readings, change records, receipt timestamps, quality flags, and boundary mapping versions.
Each label has an initial value, revision sequence, as-of timestamp, finality flag, and source identifier.
The training outcome is the final eligible label only after a declared maturity lag; ambiguous or disputed labels are retained with an exclusion reason.
Missing or censored hours are counted by season, weather severity, and cause, rather than silently dropped.
If extreme events disproportionately lack labels, reported peak performance is biased toward easier cases.
Weather records have provider, model, cycle issue time, grid/station, valid interval, variable, unit, ensemble member, receipt time, and revision ID.
The selected weather forecast must have both issue time and internal receipt time no later than the cutoff and a valid time matching the target hour.
The most recent eligible cycle is selected independently for each source and forecast origin, subject to a documented completeness rule.
Use a deterministic tie-break on provider priority and revision ID after receipt-time ordering; save the chosen identifiers.
A cycle issued before the cutoff but ingested later is ineligible at that cutoff.
Realized weather is useful for retrospective diagnostics but is prohibited as a feature for the original decision.
A later reanalysis product is also prohibited as a historical stand-in for an unavailable operational forecast.
Calendar features include local weekday, official holiday version, target hour, UTC offset, and daylight-saving fold.
Potential inputs include eligible lagged load, trailing statistics constructed from available observations, and forecast temperature and humidity.
Temperature units and spatial weighting are fixed in the model contract; a weather station relocation or grid change requires review.
The population is all target intervals in the configured balancing boundary on evaluation days.
A historical evaluation window must be chosen after archive coverage is established; no real dates or row counts are asserted here.

### 5b. Tools & Stack

A proposed Python-based batch pipeline reads immutable object-store snapshots and a versioned table of event and availability times.
A relational feature-materialization job performs as-of selection and writes an auditable feature snapshot.
A tree-based quantile regression candidate and a seasonal-naive comparator run in the same environment.
A scheduler, artifact registry, append-only issuance log, and access-controlled planner interface complete the proposed stack.
Exact package versions, hardware, provider contracts, and deployment location remain unselected and must be pinned before any execution.
No cloud vendor cost or deployed software is claimed.

### 5c. Method Summary

The proposed sequence is eligible source versions → point-in-time features → candidate conditional quantiles → consistency checks → advisory forecast and quality flags.
The baseline uses the load at the same local weekday and hour from the prior week, subject to as-of availability and offset-aware matching.
On daylight-saving anomalies, the baseline must explicitly select a comparable fold or use an approved nearby-hour rule rather than silently duplicating a value.
A weather-free seasonal model is a useful fallback and comparator when forecast vintages are unavailable.
A weather-conditioned quantile tree is a candidate because nonlinear interactions and mixed calendar signals are plausible, not because a test has shown it wins.
A linear temperature-and-calendar regression is simpler to inspect and may win with scarce data.
A neural sequence model is an alternative only if added complexity survives identical temporal tests and operational constraints.
All alternatives are considered, not tested; selection is conditional on locked evaluation and operations approval.
Separate quantile models may cross; a documented monotone rearrangement may restore ordering but must be evaluated after postprocessing.
A point median is useful for absolute-error scoring, whereas a different quantile may suit an asymmetric decision cost.
Marginal hourly quantiles do not specify the joint distribution of the daily peak.

### 5d. Provenance, Quality, Exclusions & Leakage

Source IDs and precise supported scope:
- S1: NOAA National Centers for Environmental Information, [Global Forecast System](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast), agency product description, accessed 2026-11-07; supports existence of archived numerical weather model products, not availability of the hypothetical utility's historical receipts or particular prices.
- S2: Gneiting, Balabdaoui and Raftery, [Probabilistic forecasts, calibration and sharpness](https://academic.oup.com/jrsssb/article-abstract/69/2/243/7109375), *Journal of the Royal Statistical Society B* 69(2), 2007, accessed 2026-11-07; supports evaluating probabilistic calibration and sharpness together, not a guarantee for this proposed model.
- S3: Hyndman and Athanasopoulos, [Time series cross-validation](https://otexts.com/fpp3/tscv.html), *Forecasting: Principles and Practice*, third edition, accessed 2026-11-07; supports rolling-origin evaluation, not evidence of any performance here.
- S4: Gneiting and Raftery, [Strictly proper scoring rules, prediction, and estimation](https://doi.org/10.1198/016214506000001437), *Journal of the American Statistical Association*, 2007; supports proper scoring-rule methodology, not the proposed thresholds.
- S5: Inventory title in `Train/Templates/directory.txt`, source body unavailable; establishes topic provenance only.
The model must never infer receipt time from issue time or from the date an archive was downloaded.
When historical receipt records are missing, a provider's publication schedule is not proof of local ingestion.
Backfilled rows retain original knowledge time if it is known; otherwise they are not eligible as original-time features.
Validate meter duplicates and unit conversions against independently maintained aggregate totals.
Reconcile the sum of underlying data with the modeled boundary only where the respective metering definitions agree.
Quarantine malformed records and record rejected counts and affected hours in the snapshot manifest.
A physical outage or demand response event can change observed net load; do not automatically label its measurements erroneous.
Do not drop inconvenient extremes merely because the candidate performs poorly.
Separate a data-quality correction supported by source evidence from a model-driven outlier removal.
Apply access restrictions, retention approval, and audit logging to meter and planning information.

### 5e. Reproducibility & Method-Selection Ledger

An issuance manifest records source snapshots, issue and receipt timestamps, cutoff, timezone database, feature code revision, model artifact, configuration, and output hash.
A training manifest additionally records training origin range, label maturity rule, split definitions, seed, environment lockfile, and tuning decisions.
An evaluation manifest records the locked origins, weather-vintage selection, exclusions, per-hour losses, aggregation weights, and final-label version.
No such artifacts currently exist for this hypothetical exercise.
| Candidate | Status | Reason to consider | Main limitation |
|---|---|---|---|
| Same-weekday/hour prior week | Proposed comparator | Transparent and cheap | Holidays, weather shifts and calendar transitions can be difficult. |
| Linear weather/calendar model | Considered | Interpretable and lightweight | Fixed functional form may underfit thresholds and interactions. |
| Quantile boosted trees | Proposed candidate | Flexible nonlinear conditional quantiles | Needs enough representative data and explicit crossing checks. |
| Neural sequence model | Considered | Can learn long interactions | More operational complexity and evidence needed. |
| Weather-free seasonal model | Proposed fallback | Independent of weather-provider failure | Misses useful forecast weather information. |
A failed weather ingest must not silently masquerade as a valid full-feature prediction.
Persist the actual fallback model identity and input state alongside the issued forecast.

## 6. Analysis Walkthrough

### Step 1 — Reconstruct a forecast origin

Consider an illustrative 10:00 local origin, represented internally as an unambiguous UTC instant.
Weather cycle A is issued at 06:00 and received at 06:20; its valid hour is the following day at 17:00.
Cycle B is issued at 09:30 and received at 10:05 for the same valid hour.
Cycle A is eligible; cycle B is not eligible despite its earlier issuance.
If cycle A is incomplete for a required variable, the fallback rule must either select an earlier complete cycle or explicitly mark a mixed-vintage bundle.
The bundle chosen and the exact rule belong in the manifest; hindsight may not replace the selection.
This is a constructed trace, not an observed ingest event.

### Step 2 — Construct a point forecast and score

For four illustrative target hours, actual load is `[100, 110, 120, 130]` MW.
The candidate median is `[105, 106, 118, 137]` MW.
Absolute errors are `[5, 4, 2, 7]` MW, which sum to 18 MW.
The prior-week baseline is `[98, 102, 125, 120]` MW, with absolute errors `[2, 8, 5, 10]` MW, summing to 25 MW.
Validity: these are equal-duration hours in the same population; the example is not a measured backtest.
```text
FORMULA [M1]: MAE = sum_i |y_i - forecast_i| / n
WHERE: y_i, forecast_i are average interval load [MW]; n is eligible intervals.
CONSTRAINT: compare the same origin-target pairs and label version.
NUMERICAL EXAMPLE: candidate 18/4 = 4.5 MW; baseline 25/4 = 6.25 MW.
INTERPRETATION: candidate error is 1.75 MW lower in this constructed sample.
```
Independent arithmetic check: `4 × 4.5 = 18` and `4 × 6.25 = 25`; mean difference is `(25 − 18)/4 = 1.75` MW.
The candidate's illustrative relative MAE reduction is `1.75 / 6.25 = 0.28`, or 28%; this is not a population estimate.
In the last hour the candidate is 7 MW high while the baseline is 10 MW low, so symmetric absolute error hides the direction of operational exposure.

### Step 3 — Evaluate a quantile

For observed `y = 130 MW`, a proposed 0.9 quantile `q = 125 MW` understates the outcome by 5 MW.
The pinball loss weights underprediction at the quantile level and overprediction at its complement.
```text
FORMULA [M2]: L_tau(y,q) = tau*(y-q) if y >= q; (1-tau)*(q-y) otherwise.
WHERE: tau is dimensionless in (0,1); y and q are in MW; loss is in MW.
CONSTRAINT: q predicts the same conditional target and horizon as y.
NUMERICAL EXAMPLE: tau=0.9, y=130, q=125; loss=0.9*5=4.5 MW.
INTERPRETATION: the same-size overprediction costs 0.1*5=0.5 MW.
```
Independent check: branch weights add to one; a 5 MW miss gives 4.5 MW or 0.5 MW under the respective branch.
At `tau=0.5`, both weights equal 0.5, so pinball loss is half the absolute error.
Quantile choice is not automatically an operational reserve rule; it depends on verified decision costs and constraints.

### Step 4 — Show a peak pitfall

For the same actual hours, the realized daily maximum is 130 MW and occurs in the fourth hour.
The candidate point maximum is 137 MW at the fourth hour, giving peak magnitude error +7 MW and zero hour timing error in this toy example.
The baseline maximum is 125 MW at the third hour, giving peak magnitude error −5 MW and a one-hour timing error.
The candidate has worse peak magnitude error despite better overall MAE; this arithmetic does not establish which is economically preferable.
For a distribution of daily maxima, generate coherent joint hourly trajectories and take the maximum of each trajectory.
The maximum of separate hourly 90th percentiles is generally not the 90th percentile of the daily maximum.
For example, two independent hours each equal 1 with probability 0.08 and 0 otherwise.
Each hourly 90th percentile is 0, yet the daily maximum is 0 with probability `0.92 × 0.92 = 0.8464`, so its 90th percentile is 1.
The independent-hour assumption is solely for this constructed counterexample; actual hourly dependence must be estimated and validated.

### Step 5 — Sensitivity and decision consequence

Suppose the candidate's fourth-hour forecast is 145 rather than 137 MW, with all other values unchanged.
Its absolute error sum becomes `5 + 4 + 2 + 15 = 26 MW`, giving `26/4 = 6.5 MW`.
The baseline remains 6.25 MW, so the candidate now loses the toy MAE comparison by 0.25 MW.
Check: `4 × 6.5 = 26` and `6.5 − 6.25 = 0.25`.
This alternative is a hand calculation, not a rerun on real data; the apparent preference is sensitive to one hour in a tiny sample.
A real decision requires enough independent days, complete seasons, paired uncertainty intervals, and predeclared gates.

### Step 6 — Illustrative operating economics

Assume a simplified one-hour linear shortage charge of $200 per MWh and excess charge of $50 per MWh, excluding all other market and reliability effects.
If a planner commits 120 MW and realized load is 130 MW for the hour, the shortfall is 10 MWh and the illustrative charge is `$200 × 10 = $2,000`.
If the planner commits 140 MW, the excess is 10 MWh and the illustrative charge is `$50 × 10 = $500`.
Check: `10 MW × 1 h = 10 MWh`; multiplying by dollars per MWh gives dollars.
Under the highly simplified one-period, unconstrained, linear asymmetric loss, the optimal quantile is `200/(200+50) = 0.8`.
That follows because increasing commitment by one unit reduces expected shortage cost with probability `P(Y>q)` and increases expected excess cost with probability `P(Y<q)`.
At the interior optimum the two marginal expected costs balance; with a continuous distribution this gives `F(q)=0.8`.
If shortage cost were instead $100/MWh with excess unchanged, the fraction becomes `100/150 = 2/3`, not 0.8.
Actual constraints, non-linear costs, uncertain prices, intertemporal coupling, and reserve requirements invalidate the simple prescription.
No real price source or realized savings is represented by these teaching numbers.

### Step 7 — Point-in-time source reconciliation

The following worked traces are hypothetical diagnostic exercises, not records of incidents or executed tests.
Each trace exposes a distinct decision about what the planner could know and what the scoring process may later learn.

#### Trace A: revised meter value

An hourly reading initially says 90 MW and is received at 08:20.
A correction to 94 MW arrives at 13:00 on the same day.
At a 10:00 origin, the eligible feature value is 90 MW.
The 94 MW correction may be used at a later origin if that later origin is after receipt.
If 94 MW is the mature adjudicated outcome, the evaluation can score against 94 MW after finalization.
Using 94 MW in the 10:00 feature creates a hindsight feature even though it is a valid final label.
The feature store therefore needs versioned knowledge time, not just the timestamp of the physical interval.
An append-only revision sequence permits replay of both the original feature and the final outcome.
If the initial receipt time was not logged, a reconstructed 10:00 value is uncertain.
The safe response is to identify that uncertainty in cohort eligibility, not to assume the first stored revision was originally available.
An analyst should compare the number of uncertain revisions across hot and mild days before excluding any of them.
An exclusion concentrated on difficult days changes the population to which the score applies.
The audit deliverable is an origin-to-revision selection table, not a visually plausible line chart.

#### Trace B: two suppliers with asynchronous delivery

Provider A publishes a cycle at 06:00 and the utility receives it at 06:12.
Provider B publishes a cycle at 07:00 but its transfer is delayed until 10:10.
At 10:00, choosing B merely because its issue timestamp is newer is invalid.
Choosing A is valid only if its required variables and horizon are complete and its permitted age has not expired.
The completeness check must use the files actually received by 10:00, not the files later discovered in an archive.
If no permitted cycle is complete, the model should receive a failed-input status rather than a partially populated bundle.
If a model is specifically trained to consume missing weather features, that alternative still requires separate validation and explicit missingness markers.
If different variables come from different cycles, the feature contract must define which combinations were validated.
The retained metadata includes provider, issue time, receipt time, valid time, variable, units, and file revision.
This permits an operator to explain why an older issue was used despite a newer issue being visible later.
Retrospective diagnostics can compare the two cycles after the fact but cannot replace the originally selected one.
No provider availability rate or licensing term follows from this example.

#### Trace C: weather-valid time and horizon

Suppose the target is 17:00–18:00 local on the next operating day.
An hourly weather value labeled 17:00 might describe the start of that hour or an instantaneous observation.
The weather provider's field definition must settle this before the feature is joined.
A temperature at 17:00 UTC is not automatically the same instant as 17:00 local.
Convert the target interval to UTC with the applicable timezone rules before joining.
Keep the original valid-time label so a later auditor can reconstruct the conversion.
If the weather field is a six-hour average, treating it as an hourly observation changes its meaning.
An aggregation or interpolation rule must be trained and evaluated consistently with the available forecast field.
The target lead time is measured from issuance to target, not from the latest meter event.
Two target hours in the same day therefore have different lead times even though they share one issuance.
Compare errors by lead-time bucket to see whether a model only helps near the start of the horizon.
When a clock change occurs between issuance and target, calculate elapsed duration from UTC instants.
This distinction prevents a local-clock subtraction from assigning the wrong forecast lead.

#### Trace D: boundary mapping as of issuance

Suppose a new interconnection is added to a balancing boundary after the historical forecast date.
A current asset map might include that interconnection for all past hours.
Rebuilding historical input with the current map would describe a different population from the historical desk's decision.
The map must have an effective interval and an availability timestamp.
The effective interval says when a mapping applies physically or contractually.
The availability timestamp says when the forecasting system could know that mapping.
An effective date earlier than the publication date is possible for a backdated correction.
The original forecast replay must obey knowledge time, while an adjudicated target definition may use an agreed corrected map.
If the evaluation target is restated, the baseline and candidate must both be scored against that same restated target.
A map revision that substantially changes target semantics warrants a new evaluation cohort or a documented bridging method.
Merely adding a map-version column is insufficient if joins ignore its valid interval.
The responsible operations owner must approve the accounting boundary before model comparisons begin.

#### Trace E: missing hourly meter data

Imagine a one-hour telemetry gap before the forecast cutoff.
The absence could reflect a communications outage, meter maintenance, an invalid reading, or a real zero.
Represent missingness separately from a value of zero.
An as-of interpolation may use only observations available at the origin.
An interpolation that reaches into the next day leaks information even if it produces a smoother history.
The gap length and imputation method should be features or quality flags if the model was trained that way.
Do not fill target labels with model predictions and then report accuracy against those predictions.
For evaluation, distinguish missing features from missing mature outcomes.
If missing outcomes correlate with severe storms, exclude-and-average is likely to overstate coverage of the intended operational population.
Report the excluded origin count and its distribution across weather conditions and event severity.
If the source gap exceeds the fallback model's validated tolerance, request human review instead of extrapolating without warning.
The source owner must separately repair data where authoritative records allow it.

#### Trace F: mixed units in the source

One source reports average power in kilowatts, while the target contract requires megawatts.
One megawatt equals 1,000 kilowatts, so 125,000 kW is 125 MW.
An hourly reading of 125 MW over a full hour represents 125 MWh, assuming the average is over that entire hour.
The same 125 MW averaged over a quarter hour represents 31.25 MWh, not 125 MWh.
Check: `125 × 0.25 = 31.25`; converting 31.25 MWh back over 0.25 hours gives 125 MW.
Units must be attached to both field names and schema metadata rather than inferred from magnitude.
A sudden thousandfold jump could be a unit change, but it could also indicate a malformed aggregation.
The diagnostic compares source schema versions and independently reconciled totals before rejecting data.
Never fit a model to a mixed-unit history and hope that scaling in training resolves the semantic error.
If a unit change is legitimate, convert both training and inference records through the same versioned transformation.
Do not assign a cost to MW directly when the settlement charge is denominated in dollars per MWh.
Duration must enter the conversion, especially when the interval granularity changes.

#### Trace G: four-hour interval coverage

Suppose four actual hourly values are 100, 110, 120 and 130 MW.
Suppose proposed lower bounds are 95, 105, 115 and 125 MW.
Suppose proposed upper bounds are 105, 115, 125 and 128 MW.
The first three observations fall inside the closed intervals and the fourth falls outside.
Empirical coverage is `3/4 = 75%` for this constructed sample.
The widths are 10, 10, 10 and 3 MW, respectively.
Their mean width is `33/4 = 8.25 MW`.
Check: `8.25 × 4 = 33`, and `0.75 × 4 = 3` covered hours.
The final interval is narrow and misses the largest value, which illustrates why coverage should be sliced by peak condition.
Four hours are far too few to establish calibration at any nominal level.
The interval endpoints should be scored after any crossing correction.
The choice of closed versus open endpoint matters only for ties but must be consistent.
The computed mean width does not indicate whether the intervals are economically useful.

#### Trace H: threshold-exceedance verification

For ten hypothetical operating days, suppose a 140 MW threshold is exceeded on two days.
Suppose the alert system flags three days, including one true exceedance and two non-events.
There is one true positive, one false negative, two false positives, and six true negatives.
The event recall is `1/2 = 50%`; the alert precision is `1/3`, approximately 33.3%.
The false-alarm rate among non-events is `2/8 = 25%`.
Check: the four cells sum to `1 + 1 + 2 + 6 = 10`.
The overall correct classification rate is `7/10 = 70%`, but it hides half of the actual exceedances.
If the threshold was chosen after viewing these same ten days, the evaluation is selection-biased.
The alert threshold and probability cutoff should be fixed before the final sample is examined.
Precision depends strongly on the prevalence of threshold events and may change across seasons.
Reporting only recall conceals the operational workload from false alerts.
Reporting only precision conceals missed dangerous days.
Neither measure shows how well predicted probabilities are calibrated.

#### Trace I: an operational deadline breach

Assume the forecast is scheduled to publish by 10:15 local.
A weather cycle arrives at 10:01, one minute after the 10:00 feature freeze.
The inference run based on the earlier eligible snapshot completes at 10:11.
That output can be on time if all validations and publication finish before 10:15.
The 10:01 data is excluded from the initial version regardless of its usefulness.
If permitted by operating policy, a separate revision can use the new cycle and clearly show a later issue time.
The revision must not overwrite the original 10:11 issuance or inherit its timestamp.
If publication occurs at 10:16, the initial issuance is late even if the underlying model finished at 10:11.
An availability denominator includes the scheduled issuance whether it was late, failed, or used fallback.
For example, 29 on-time usable issuances out of 30 scheduled gives `29/30 = 96.67%` rounded.
This arithmetic says nothing about the proposed system's actual availability.
The reliability team must measure end-to-end publication, not merely model inference time.

#### Trace J: source outage and fallback cohort

Suppose a weather provider is unavailable on particularly severe storm days.
Scoring only candidate runs with successful weather access removes exactly the difficult days where fallback matters most.
Score the end-to-end policy on every eligible scheduled day, including weather-free fallback and explicit no-forecast outcomes.
Also score the weather-conditioned candidate on the subset where it ran, but label that subset clearly.
The two reported populations answer different questions and cannot be compared as though they were identical.
A no-forecast day is an availability failure, not an hour with a perfect zero error.
For a paired model comparison, use matched forecast outputs and separately report excluded or unavailable pairs.
If fallback outputs are systematically worse during storms, the operations desk needs a visible uncertainty flag.
An overall MAE improvement that excludes the outages cannot establish end-to-end improvement.
Review fallback frequency by weather event, provider, source age and forecast horizon.
The fallback choice and acceptance threshold must be fixed before the final holdout.
Actual outage probability and fallback performance remain unknown in this case.

#### Trace K: rolling-origin training eligibility

Assume a historical daily forecast origin occurs at 10:00 on Monday.
The target day is Tuesday and its labels are not finalized until Thursday.
A model retrained on Wednesday cannot use those Tuesday finalized labels if they are not yet available.
A model retrained on Friday may use the final labels if its training freeze is after finalization.
Using event date alone would incorrectly admit the Tuesday labels into Wednesday's model.
The training materializer must constrain both the label's event interval and its finalization or allowed-availability timestamp.
If provisional labels are permitted for training, that policy must be explicit and applied in every fold.
The final evaluation can still use mature labels, provided that finality affects scoring rather than historical feature construction.
An expanding-window evaluation adds data over time; a sliding-window evaluation also drops older data.
Both strategies are viable only if every fold follows the same eligibility and maturity rules.
The decision between them depends on stability, drift, data volume, and observed validation evidence.
It is not legitimate to choose the window after inspecting the locked final holdout.

#### Trace L: candidate ranking instability

Suppose a development fold gives baseline MAE 10 MW and candidate MAE 9 MW.
A second fold gives baseline MAE 12 MW and candidate MAE 14 MW.
If the folds have equal eligible-hour counts, the mean baseline MAE is 11 MW.
Under the same equal weighting, the mean candidate MAE is 11.5 MW.
Check: `(10 + 12)/2 = 11` and `(9 + 14)/2 = 11.5`.
The candidate improves the first fold by 1 MW but worsens the second by 2 MW.
If fold sizes differ, use the intended weighting rather than averaging fold means blindly.
Investigate whether the difference reflects weather regime, an outage, target changes, or random sampling variability.
Do not label either fold representative of all seasons from two numbers alone.
The final gate should be established before model tuning and should include a paired uncertainty assessment.
Report each fold and the pooled estimand so a strong fold cannot hide an operationally important failure.
These numbers are constructed and are not results of any model run.

#### Trace M: mean forecast bias

Suppose four residuals defined as forecast minus actual are `+5, −4, −2, +7` MW.
Their sum is 6 MW and their mean signed bias is `6/4 = +1.5 MW`.
Check: positive residual means overforecast under this sign convention.
The same residuals have MAE `18/4 = 4.5 MW`, as in the earlier example.
Signed errors can cancel, so a small average bias does not imply small individual misses.
Conversely, the absolute error hides whether a particular regime is systematically overforecast.
Report signed bias separately for high-load hours, low-load hours, seasons and relevant lead times.
Always state the residual sign convention because other teams may use actual minus forecast.
Use paired differences and consistent labels when comparing the model with the baseline.
Negative load values remain valid if the target is net boundary import and exports are possible.
The pattern in this toy sample cannot establish whether the proposed model is biased in practice.

#### Trace N: uneven hourly aggregation

Suppose a 15-minute source has four average-power readings for an ordinary hour.
For equal-duration complete intervals of 100, 104, 108 and 112 MW, the hourly mean is 106 MW.
The corresponding energy is `106 MW × 1 h = 106 MWh`.
Equivalently, sum the four interval energies: `(100 + 104 + 108 + 112) × 0.25 = 106 MWh`.
The two routes agree because each interval is one quarter hour and all four are present.
If the last interval is missing, treating the three available readings as the complete hour can bias the result.
Do not count a missing quarter hour as zero without physical evidence.
If an authoritative estimate is substituted, retain the estimate flag and its source.
For unequal durations, weight each average-power observation by its duration before computing hourly energy.
Reconcile the aggregate with a trusted settlement or meter total when definitions agree.
The record should distinguish an estimated hour from a measured complete hour.
Data-quality thresholds must be fixed before seeing forecast losses.

#### Trace O: unequal decision costs

Consider two constructed one-hour forecasts: A is 10 MW too low and B is 10 MW too high.
Both have 10 MW absolute error for a one-hour average-power target.
At the illustrative $200/MWh shortage rate, A incurs a simplified charge of $2,000.
At the illustrative $50/MWh excess rate, B incurs a simplified charge of $500.
Check: each error corresponds to 10 MWh over one hour, so the charges follow from the stated rates.
The 1,500-dollar difference arises from the assumed asymmetric charges, not from a measured market.
If actual prices are negative or nonlinear, this simple calculation may be inappropriate.
If available resources have ramp limits or minimum run times, a one-hour decision cannot capture all consequences.
An hourly cost replay must account for the actual feasible actions and the information available to the operator.
Do not infer that an upper quantile is always optimal when penalty schedules change by hour.
Show results under multiple documented cost scenarios when the real distribution of costs is uncertain.
The operational owner decides whether the replay is credible enough to inform planning.

#### Trace P: quantile crossing and repair

Suppose independently fitted outputs for one hour are q10 = 110, q50 = 105, and q90 = 120 MW.
The ordering is invalid because the median is below the lower quantile.
One candidate repair sorts the three values into 105, 110 and 120 MW.
This enforces order but changes which numerical value represents each quantile.
The repair may improve or worsen pinball loss and calibration; those effects must be measured.
Another candidate is a model parameterization that enforces noncrossing outputs during estimation.
The choice should be made using development data and fixed before final evaluation.
After any repair, compute coverage and scoring metrics on the published outputs, not the pre-repair outputs.
Log the frequency and magnitude of crossing before correction as a diagnostic.
A sudden increase could indicate feature corruption, model drift, or a code regression.
The postprocessor must not silently turn nonfinite values into valid-looking quantiles.
If correction cannot produce a valid vector, reject that issuance and invoke the approved fallback.

#### Trace Q: label finality versus operational monitoring

Suppose the operational meter feed posts an initial hourly value the next morning.
Suppose the settlement process may revise it several days later.
An immediate dashboard can compare forecasts with provisional values if it labels the scores provisional.
An adjudicated historical evaluation should use the declared mature or final label policy.
The difference between provisional and mature scores is itself a useful data-quality diagnostic.
A forecast can appear to degrade merely because the label revision process changed.
Conversely, a model can appear stable on provisional labels while final-label errors worsen.
When a label is corrected, recompute affected dashboard scores as a new revision rather than overwriting the audit trail.
Do not retroactively alter the forecast issuance or its original input snapshot.
Report the proportion of labels that remain immature at every monitoring cutoff.
Escalate when finalization delay grows beyond the training policy's assumption.
The metering owner, not the model, decides which disputed observation is authoritative.

#### Trace R: manual overrides

Suppose a planner increases the evening peak estimate because of a newly announced industrial restart.
The override may contain information that the model never received before its cutoff.
It may also be an overreaction; the direction cannot be decided from the override alone.
Record the original forecast, edited quantity, editor role, time, reason and information available.
Score the unmodified forecast and the final planner view separately against the same mature label.
The difference is descriptive and is not automatically the causal effect of the override.
Planners may override only the most difficult days, creating selection bias.
A comparison limited to overridden days should include the difficulty of those days and the unchanged baseline.
An override can reduce forecast error while increasing operational cost if its timing or shape is wrong.
Some information may arrive after the model cutoff but before a revised planning decision.
That information must be attached to the revision timestamp, not backdated into the initial issuance.
An operator retains accountability for the final decision even if the model suggests a different value.

#### Trace S: reproducible fold selection

Assume the development period covers multiple seasons and the final period is reserved before tuning.
Construct each origin with its original weather-vintage selection and available meter revisions.
Train the first fold using only data preceding its origin and the defined label maturity lag.
Evaluate the resulting model on the subsequent target intervals in that fold.
Advance the origin and repeat, allowing only newly eligible observations into the next fit.
If hyperparameters are chosen using the pooled fold results, keep the final holdout untouched.
If a calibration layer is fitted, its training data must also precede its validation targets.
A final model may be retrained on all eligible development data after selection is frozen.
The final evaluation still requires a simulation of the permitted production retraining schedule.
Report whether the model is refit daily, weekly, or held fixed, because those are different procedures.
A model trained once on all final-period labels is not a legitimate final-period forecast.
Save fold-level manifests to allow independent reconstruction of every eligibility decision.

#### Trace T: episode-level uncertainty

Suppose 240 hourly forecasts come from ten consecutive 24-hour operating days.
Treating those 240 hours as independent would ignore shared weather systems and daily conditions.
An episode can affect several neighboring days, so even ten day blocks may not be independent.
Choose a resampling unit long enough to preserve the dependence relevant to the estimand.
For each sampled block, preserve the candidate and baseline outcomes together.
Compute the paired loss difference on each resample, not separate unrelated resamples for each model.
This preserves the pairing that controls shared hourly difficulty.
With only ten days, uncertainty remains large even if the bootstrap produces a numerical interval.
The resampling procedure cannot create unobserved heat waves or previously unseen grid regimes.
Inspect individual event contributions and report the number of independent weather episodes.
If the historical record contains only one extreme event, no narrow interval can make its generalizability certain.
Any block length and interval method should be justified before reviewing the final results.

#### Trace U: planned model rollback

A newly registered candidate produces nonfinite outputs for several target hours during a rehearsal.
The release gate rejects that candidate before the planner sees a complete-looking forecast.
The system selects the last approved compatible fallback artifact rather than an arbitrary older file.
Compatibility includes target definition, feature schema, boundary map, and calendar conventions.
The issuance manifest records both the failed candidate and the fallback actually used.
If the fallback also fails validation, the interface displays an unavailable state and alerts the desk.
Restarting the pipeline must not create conflicting initial issuances for the same boundary and operating day.
A later corrected issue receives a new revision identifier and an explicit timestamp.
Rollback cannot repair an upstream corrupted meter feed shared by both candidate and fallback.
That common-mode risk requires independent source-quality checks and a manual contingency.
The rehearsal needs synthetic fault injection and an independently checked publication log.
No rehearsal or rollback has actually been performed in this illustrative case.

#### Trace V: extreme weather and covariate support

Suppose the hottest weather forecast in historical development data is 40 degrees on the provider's specified scale.
An eligible day-ahead forecast predicts a higher temperature outside that development range.
A tree model may return a value based on terminal regions rather than extrapolating the way a linear model would.
A linear model may extrapolate far beyond credible physical or operational behavior.
Neither behavior is automatically safe merely because it is mathematically well defined.
Display a support warning and compare with a domain-approved stress scenario.
Evaluate historical near-boundary periods without pretending they establish accuracy beyond observed conditions.
An uncertainty interval calibrated only on mild periods may substantially understate extreme risk.
If the data record contains too few independent hot events, say that tail calibration is unresolved.
The operator must retain authority to incorporate credible external information.
Recalibration must use only labels available at its own training cutoff.
A source change or sensor relocation can imitate an extreme, so inspect units and version history first.

#### Trace W: net load and behind-the-meter generation

Suppose customer consumption is 150 MW and behind-the-meter generation is 30 MW.
Ignoring other flows for this simplified example, the boundary net load is 120 MW.
If generation rises to 50 MW while consumption stays 150 MW, net load becomes 100 MW.
The change is a 20 MW reduction in net load, not proof that customer demand fell.
Check: `150 − 30 = 120` and `150 − 50 = 100`.
The boundary measurement may incorporate losses and flows not represented by this simple identity.
Do not add a separate behind-the-meter generation estimate to net load unless the intended target is explicitly gross demand.
Cloud forecasts can affect net load through embedded solar output as well as through weather-sensitive consumption.
An external generation estimate must obey the same availability and version rules as every other feature.
The data owner should document whether battery charging and exports are included in the boundary measurement.
A structural change in embedded generation can invalidate a historical same-hour baseline.
The appropriate response is to assess the changed target relationship, not to relabel net load as customer consumption.

#### Trace X: daily maximum timing

Consider hourly actual load `[100, 130, 130, 110]` MW.
There are two tied maximum hours, the second and third.
The evaluation policy must specify whether the earliest maximum, latest maximum, or set of all maxima defines timing.
Suppose the policy uses the earliest; the actual peak time is the second hour.
If the predicted peak occurs in the third hour, the stated timing error is one elapsed hour.
Under a set-based rule the prediction may instead count as matching an actual peak hour.
Neither convention is intrinsically universal; the choice should reflect the operational question.
Fix the convention before seeing model results and report its effect on tied days.
For a daylight-saving transition, elapsed time must be calculated from distinct UTC instants.
A one-hour local-clock difference can otherwise be ambiguous.
Magnitude error and timing error must remain separate metrics because a model can be right on one and wrong on the other.
A peak threshold event is also different from exact peak-time prediction.

#### Trace Y: as-of forecast correction

Suppose an initial issuance is delivered at 10:12 and a weather update arrives at 10:20.
The operator may decide that an updated forecast is useful before a later planning milestone.
If the workflow authorizes it, issue revision 2 using an explicit new cutoff after 10:20.
Retain the 10:12 forecast exactly as it appeared to the original decision.
Record the new weather cycle, input snapshot, model version and reason for revision.
Do not compare revision 2 against a baseline frozen at 10:00 without stating the different information advantage.
For model comparison at the revised cutoff, reconstruct the baseline using the same revised information policy.
The planner should see whether a displayed value is the initial or revised issuance.
A revision timestamp is not a substitute for a source receipt timestamp.
If the new cycle does not cover the full target horizon, the revised output still needs the approved completeness policy.
Multiple revisions can improve a later decision without proving that the original day-ahead forecast was more accurate.
The record must support both initial-decision and revised-decision evaluations.

#### Trace Z: finite capacity budget

Suppose feature construction takes 4 minutes, model scoring 3 minutes, validation 2 minutes and publication 3 minutes.
The serial critical-path time is `4 + 3 + 2 + 3 = 12` minutes, excluding snapshot acquisition and queueing.
If acquisition takes 2 minutes and begins at the 10:00 cutoff, completion is at 10:14 under these assumptions.
That leaves one minute before a 10:15 deadline, insufficient for a guaranteed 3-minute retry.
Check: `2 + 12 + 3 = 17` minutes if a full 3-minute retry is needed.
The retry could therefore finish at 10:17 rather than the deadline.
Parallelizing independent tasks might help, but only measured dependencies and capacity can establish feasibility.
An earlier data freeze changes the information available and must be reflected in the backtest.
Precomputing features is safe only if their time-dependent components are refreshed at the correct cutoff.
The system's tail latency, not just the average stage time, determines deadline reliability.
These numbers are an illustrative critical-path calculation and do not claim measured run times.
An outage can cause multiple retries or a queue backlog, neither represented by the simple sum.

### Step 8 — Further decision and production exercises

The following scenarios extend the same proposed decision contract; all quantities and events are invented for instruction.

#### Exercise AA: a holiday-calendar revision

An origin occurs before a newly declared one-off holiday is publicly announced.
A retrospective calendar table created later marks the target day as a holiday.
Using the later flag at the earlier origin would reveal information the planner did not possess.
The calendar table therefore needs announcement availability as well as the date to which the holiday applies.
Ordinary recurring holidays may be known years in advance, but exceptions need their actual announcement time.
If the holiday is announced before a later revised issuance, it may be included in that revision.
The original and revised issuances must retain different calendar snapshots.
The baseline should be subject to the same information boundary as the candidate.
An operator's knowledge of an unannounced closure is not automatically available to the model.
It can be recorded as a timestamped manual override or as an approved input with a known receipt time.
The evaluation should distinguish holidays known at cutoff from unexpected closures.
If there are very few examples, a narrow error estimate for those days is not credible.

#### Exercise AB: same-weekday baseline selection

Suppose the target is a Monday evening hour and the prior Monday was a holiday.
A simple seven-day lag may be an atypical comparator.
The baseline must still use a predeclared rule rather than choosing the best historic day after seeing actual load.
An alternative rule could use the median of the previous four eligible nonholiday Mondays at the same local hour.
That alternative needs a clear eligibility rule when one or more historical inputs are missing.
It also needs an as-of calendar, a daylight-saving fold rule, and a stable meter boundary.
The original single-week baseline should remain visible if it represents the current workflow.
If the alternative is adopted as a challenger, it must be scored on the same origin-target pairs as the model.
Predeclaring a stronger baseline makes the performance claim more informative.
A more elaborate baseline can still fail during abrupt weather or demand changes.
The best comparator is the one that represents an available decision alternative, not simply the easiest model to beat.
No baseline selection here is supported by an executed comparison.

#### Exercise AC: metric weighting

Suppose two evaluation days contain 24 and 25 hourly intervals, respectively.
An hour-weighted MAE sums all hourly absolute errors and divides by 49.
A day-weighted MAE first averages within each day and then averages the two daily means.
These are different estimands because the 25-hour day receives slightly more weight in the first.
If each hour's operational exposure is equivalent, hourly weighting can be appropriate.
If each day represents one equally weighted planning decision, day weighting can be a useful companion.
Neither choice is automatically correct for a cost-weighted business objective.
Suppose the first day has MAE 2 MW and the second has MAE 4 MW.
The hour-weighted result is `(24×2 + 25×4)/49 = 148/49 ≈ 3.0204 MW`.
The equally day-weighted result is `(2 + 4)/2 = 3 MW`.
Check: 48 plus 100 is 148, so the first result lies between the two day means.
Fix the weighting before final evaluation and report both if they serve distinct decisions.

#### Exercise AD: percentage error near zero

Assume a net-load target can be close to zero because embedded generation offsets demand.
If actual load is 1 MW and forecast load is 3 MW, absolute error is 2 MW.
A percentage error using the actual as denominator is `2/1 = 200%`.
If actual load is 0 MW, the same percentage calculation divides by zero and is undefined.
If the target may be negative, percentage interpretations can become even less intuitive.
Avoid silently adding an arbitrary denominator constant and calling the result an ordinary percentage error.
MAE in MW remains defined at zero and has a direct power-unit interpretation.
An energy-weighted measure may be useful for another question but needs its own denominator and zero-handling rule.
Report whether the evaluation emphasizes small loads, high loads, or energy volume.
Any normalized metric requires a clearly defined reference scale calculated without future leakage.
Use the same normalization for baseline and candidate over the same population.
The metric choice should follow the decision rather than convenience.

#### Exercise AE: aggregate versus subgroup performance

Suppose 90 ordinary hours improve by 2 MW each relative to baseline.
Suppose 10 high-risk hours deteriorate by 10 MW each.
The aggregate paired improvement is `(90×2 − 10×10)/100 = 0.8 MW` per hour.
Check: 180 MW of summed ordinary-hour improvement minus 100 MW deterioration is 80 MW.
The candidate therefore improves the aggregate despite worsening every high-risk hour in this constructed example.
If high-risk hours drive operational decisions, a peak guardrail can reject the candidate.
Define the subgroup using information or labels appropriate to the evaluation question.
If using realized peaks to define a retrospective slice, do not pretend the realized peak was known at issuance.
An operational alert rule needs a threshold based on information available at the time.
Report the number of distinct days and weather events represented by the subgroup.
An apparently large hourly sample can still contain only one underlying extreme event.
The outcome of this toy arithmetic does not predict the result of an actual utility evaluation.

#### Exercise AF: scenario dependence

Suppose two hourly loads independently take 0 or 1 MW, each with probability 0.5.
The daily peak is zero only when both hours are zero, with probability 0.25.
The peak is therefore 1 MW with probability 0.75.
Now suppose the two hours are perfectly positively dependent while preserving the same marginal distributions.
Both are zero together with probability 0.5 and one together with probability 0.5.
The peak is then 1 MW with probability 0.5 rather than 0.75.
The hourly marginals are identical in both constructions.
The daily peak probabilities differ because their temporal dependence differs.
Check the four independent combinations each have probability 0.25, three of which include at least one 1.
Sampling hourly marginals independently is therefore an assumption, not a consequence of knowing the marginals.
Scenario construction should preserve and validate relevant temporal dependence.
There is no evidence here that either synthetic dependence structure describes an actual grid.

#### Exercise AG: empirical quantile coverage

Suppose a proposed 90th-percentile forecast is exceeded in 14 of 100 eligible hours.
The empirical exceedance rate is `14/100 = 14%`.
The empirical fraction at or below the forecast is 86% under a consistent treatment of ties.
The difference from the nominal 10% exceedance rate is four percentage points.
That deviation alone does not identify a cause or establish statistical significance.
The hours may be dependent, and 100 hourly observations may represent far fewer independent days.
Inspect whether the excess exceedances occur in a single weather episode or across many episodes.
Check for late label corrections and feature-vintage errors before changing the model.
Also report the magnitude of exceedances; a 1 MW miss and a 50 MW miss have different consequences.
Pinball loss reflects both direction and size, but still does not replace decision-cost analysis.
If recalibration is proposed, fit it on an earlier eligible calibration period.
Reserve later data for assessing whether that calibration generalizes.

#### Exercise AH: train-serving skew

Suppose historical training uses a 24-hour rolling load mean that includes the current target hour.
At the 10:00 production origin, tomorrow's target-hour observation cannot be known.
The same feature definition is therefore unavailable at inference and constitutes leakage.
Replacing the missing future value with zero in production would create a different feature distribution.
A valid alternative is a lagged rolling mean ending at the most recent eligible interval before cutoff.
Define the window endpoints explicitly and require each component's receipt time to meet the origin cutoff.
Apply the same window definition to every backtest origin and every live issuance.
Persist a sample of feature vectors and compare materialized training and serving values for the same snapshot.
Investigate mismatches caused by timezone conversion, imputation or source revision order.
Numerical equality on one ordinary day is not sufficient to cover clock changes or delayed data.
Test edge cases with intentionally late meter values and repeated local hours.
The training-to-serving contract should fail closed if an incompatible schema is detected.

#### Exercise AI: access-control boundary

Assume the forecasting job needs read access to aggregate meter and weather inputs.
It needs write access to a forecast artifact store and an issuance log.
Neither requirement implies permission to write to a market bidding interface.
Neither requirement implies permission to issue commands to a supervisory control system.
Use separate identities for feature generation, model registration, advisory publication and operational control.
The planner interface should read approved forecast artifacts rather than execute arbitrary model code.
An approval record should identify the artifact hash and the target boundary to which approval applies.
If a model artifact is replaced under an existing mutable name, the approval may no longer match the bytes.
Signed or otherwise integrity-verified immutable artifacts reduce this ambiguity.
Access logs should record denied attempts as well as successful publication actions.
Credential rotation and emergency revocation procedures need operational ownership.
An access-control design is proposed here; no security review has been performed.

#### Exercise AJ: duplicate jobs and revisions

Suppose two scheduler instances start the same boundary-day job after a retry.
Both may compute an identical forecast, but simultaneous writes can still produce conflicting records.
An idempotency key can be based on the boundary, operating day, issuance type and revision number.
The durable state transition should allow only one successful initial publication for a key.
The second writer should read the existing result and verify its identity rather than overwrite it.
If the two runs used different input snapshots, the difference must trigger investigation.
A later permissible forecast is a new revision, not a second initial publication.
If a crash occurs after writing the payload but before updating the issuance record, recovery must reconcile those states.
A transactional publication mechanism or explicitly documented reconciliation protocol is needed.
Do not assume a scheduler's retry semantics guarantee exactly-once external publication.
The observer should distinguish compute attempts from authoritative forecast issuances.
A rehearsal should include crashes before write, after payload write and after manifest commit.

#### Exercise AK: choice of model refresh frequency

Daily retraining can incorporate newly matured observations quickly.
It also increases computation, validation load, operational complexity and exposure to recent anomalous data.
Monthly retraining is simpler but can lag a rapid change in consumption or embedded generation.
A fixed model with periodic recalibration is a third candidate, not a universal compromise.
Each candidate schedule must be replayed with labels eligible at the actual historical refresh time.
Its evaluation includes failed training runs and the behavior of the last approved artifact.
The model refresh deadline must precede inference and allow artifact validation.
A retraining job that uses today's evening outcome before today's morning issuance leaks future information.
An automatic trigger based on recent error must account for provisional labels and normal seasonal variation.
Change approval and rollback requirements apply to automatically trained artifacts as well as manually trained ones.
The best cadence depends on observed drift, available data, operating costs and governance capacity.
No cadence has been experimentally selected for the hypothetical utility.

#### Exercise AL: archive sizing

Assume an illustrative archive receives 40 MB of weather data per day for one chosen region and set of variables.
At 365 days, raw annual volume is `40 × 365 = 14,600 MB`, using decimal megabytes.
Three independently stored copies require `14,600 × 3 = 43,800 MB` before metadata and indexes.
Check: `40 × 300 = 12,000` and `40 × 65 = 2,600`; their sum is 14,600.
The three-copy estimate is a storage assumption, not a claim about any vendor's replication.
Compression, schema overhead, object metadata, checkpoints and version retention can change billed storage.
Ingesting additional ensemble members or more frequent cycles increases volume.
Retaining historical revisions increases volume further but is important for vintage reconstruction.
Network egress, API requests and retrieval operations require separate prices and counts.
The hypothetical 40 MB input is not a NOAA product size or a quote from a provider.
Obtain dated rate cards and contractual terms before estimating a real budget.
Add a sensitivity table for retention period, number of regions, compression and replication factor.

#### Exercise AM: uncertain cost assumptions

Suppose a hypothetical annual service budget contains $8,000 compute, $4,000 storage and $12,000 labor.
The total is `$8,000 + $4,000 + $12,000 = $24,000` per year.
If weather licensing adds $10,000, the total becomes $34,000 before any unpriced costs.
Check: 24 plus 10 equals 34 when values are expressed in thousands of dollars.
If on-call and security review are omitted, the calculation is not a full cost of ownership.
Labor estimates should include monitoring, failed runs, investigation and periodic model review.
Historical backfill and initial engineering are one-time costs, not automatically annual recurring costs.
Forecasting cost cannot be compared with avoided balancing cost without a credible decision-effect estimate.
A change in electricity market prices could alter both potential benefit and risk.
The assumed dollar figures are fabricated teaching inputs, not vendor rates or utility expenses.
Finance must replace them with current dated prices and authorized labor assumptions.
No positive return on investment is established by summing the costs.

#### Exercise AN: distribution shift and adaptation

Suppose the feature distribution changes after a large industrial customer connects.
That change may be genuine demand growth, a new boundary definition, or a data ingestion defect.
Compare meter mapping, customer commissioning records and raw measurement continuity.
A change detector can signal a difference but cannot by itself identify its cause.
If the boundary changed, consider target reconstruction before treating the event as ordinary covariate drift.
If the boundary did not change, assess whether the new operating regime is represented in training data.
A short post-change sample may be too small for reliable tail calibration.
Keep the existing approved fallback available while investigating.
An operator may need to supply separately documented information about the connection.
Retraining before labels mature can incorporate errors that later disappear through settlement correction.
Retrospective evaluation should include a prospective-style transition period if historical records permit.
Do not claim that a generic drift threshold is a substitute for engineering review.

#### Exercise AO: competing probabilistic models

Suppose model A has narrower 80% intervals than model B.
Suppose A covers only 60% of outcomes and B covers 79% on a comparable evaluation set.
Model A is sharper in this sample but is undercovered relative to its nominal interval.
Model B is closer to nominal coverage but may still be too wide to support a useful decision.
Compare both with interval width, pinball or another proper score, and conditional diagnostics.
Use identical target hours, mature labels, and postprocessing for the comparison.
Examine whether B's apparent aggregate calibration masks overcoverage in mild weather and undercoverage in heat.
Report uncertainty around coverage using a dependence-aware procedure.
If sample sizes are small, a one-percentage-point difference does not establish superiority.
Check whether either model produces valid ordered quantiles for every published hour.
The decision owner may prefer a calibrated model whose uncertainty is operationally interpretable.
These values are illustrative examples, not results from an actual pair of models.

#### Exercise AP: regional aggregation and coincident peaks

Suppose area A has load `[100, 80]` MW in two consecutive hours.
Suppose area B has load `[50, 90]` MW in those same hours.
Area A peaks at 100 MW in the first hour, while area B peaks at 90 MW in the second.
The sum of their individual peaks is `100 + 90 = 190 MW`.
The aggregate hourly loads are `[150, 170]` MW, so the coincident aggregate peak is 170 MW.
Check: `100 + 50 = 150` and `80 + 90 = 170`.
Adding separate noncoincident peaks overstates the coincident maximum in this example.
For an actual system, aggregation must align UTC intervals, measurement boundaries and units.
An area-to-system mapping must also avoid double-counting overlapping meters.
A regional model can be evaluated against regional operational decisions, while the aggregate model serves the defined system boundary.
If forecasts are reconciled hierarchically, assess both aggregate and regional errors after reconciliation.
Do not assume that coherent point totals make independently produced probabilistic trajectories coherent.

#### Exercise AQ: balancing energy versus average-power forecast

Suppose realized average load for a one-hour interval is 125 MW.
Suppose the hypothetical scheduled quantity is 120 MW for that interval.
The net deviation is `125 − 120 = 5 MW` in average power.
Over one full hour the corresponding energy deviation is 5 MWh.
For a 15-minute interval with the same average deviation, it is `5 × 0.25 = 1.25 MWh`.
Actual settlement may use a different interval, netting rule, price schedule, or loss adjustment.
Those rules require an authoritative jurisdiction-specific source and cannot be inferred from this exercise.
The forecast is not necessarily equal to the schedule because the planner considers supply, reserves and contractual constraints.
Consequently, forecast error and settlement deviation are different quantities.
Any cost replay must document how the forecast changes a feasible scheduled quantity.
The value of that change depends on the realized price and counterfactual action.
No market settlement has been calculated or attributed to this proposed system.

#### Exercise AR: event-time versus ingestion-time watermark

A batch process wants to decide when all load observations through 09:00 are available.
An event-time watermark may express a belief about lateness, not an absolute guarantee of completeness.
One sensor may send an 08:45 observation at 09:50 while another sends it at 10:10.
At the 10:00 origin, the second observation is not eligible regardless of its event timestamp.
A completeness estimate should identify expected sources, received sources, and unknown or failed sources.
If expected source membership changes, the completeness denominator must use the appropriate boundary version.
The issuance can proceed only under a preapproved tolerance for missing inputs or a validated fallback.
Record the actual completeness fraction and the specific missing source identities.
Do not mark a late record as on time because its payload carries an old timestamp.
When the record finally arrives, append it with its actual receipt time and maintain the prior snapshot.
The fallback escalation clock should refer to the operational deadline, not merely a watermark.
Historical replay must recreate the same cutoff decision from the preserved receipt history.

#### Exercise AS: evaluating forecast revisions

Suppose an initial forecast has MAE 8 MW and a revised forecast has MAE 6 MW over matched target hours.
The numerical improvement is 2 MW on that illustrative matched set.
The revision may have used a newer weather forecast and therefore had more information.
It would be misleading to attribute the entire improvement to a different forecasting algorithm.
Compare two algorithms at the same cutoff and with the same information set to isolate model differences.
Compare initial and revised workflows separately to understand the value of later information and extra review.
Ensure that the revised forecast still arrives before the decision it is intended to inform.
An accurate revision arriving after the relevant commitment deadline may have no value for that commitment.
It could still inform a later operational decision, which needs its own owner and deadline.
Track the number and timing of revisions and the human review burden.
Repeated revisions may create confusion if users cannot identify the currently authoritative version.
These constructed scores are not measured and do not estimate actual value of information.

#### Exercise AT: replaying a proposed decision

Constructing a decision replay requires archived forecasts, decision constraints, actions and outcome prices.
Each replayed action must use only information available at its own decision time.
The candidate and baseline policies should share the same feasible resource and network constraints.
If a constraint was not recorded, describe the assumption and its potential effect on the result.
The replay cannot change actual generator outages or weather that occurred after the historical decision.
It also cannot assume that market prices would remain unchanged under a materially different action.
If the modeled action is large enough to affect prices, a price-taking replay may be invalid.
Compare operationally meaningful cost components rather than multiplying every forecast error by one price.
Report uncertainty in price, resource availability, constraints and policy behavior separately.
An infeasible schedule must be rejected rather than counted as a cheap successful action.
Have operations review a sample of replayed decisions for realism.
This case contains a replay plan only; no actual counterfactual or economic benefit has been established.

#### Exercise AU: lifecycle of an anomalous measurement

A validation rule detects a load value outside the range observed in the previous year.
Such a rule identifies a candidate anomaly, not proof that the measurement is false.
Compare the source meter, upstream aggregation, units and independent telemetry.
Check whether an unusual weather event, outage, export, or large customer transition could explain it.
Preserve the original value and all evidence used to decide whether it should be corrected.
If a source owner confirms an error, append the corrected value with its own receipt time.
If the value is genuine, retain it and examine whether the model's training support includes comparable cases.
Do not remove genuine extremes from the holdout solely to improve a score.
If a measurement remains disputed, score a sensitivity range and report its effect on candidate ranking.
The audit trail should separate automatic flags, human decisions and source corrections.
Anomaly thresholds must be chosen before seeing the final evaluation's model residuals.
The remediation owner should document how recurrence is detected and how the fix is verified.

#### Exercise AV: comparison under a weather provider change

A provider changes its forecast model and alters the distribution of temperature errors.
The new product may have the same field names and units but different error characteristics.
An existing demand model trained on the old product may no longer be calibrated.
Archive the provider product version and effective dates with each weather forecast.
Backtest separately by product regime where sufficient archived vintages exist.
If historical new-product reforecasts exist, establish when they were produced and do not treat them as original-time forecasts.
They can support a separately labeled counterfactual experiment but not a claim of historical operational performance.
During transition, monitor forecast residuals, coverage, and weather error by lead time.
If a parallel feed is available, run a timestamped shadow comparison without silently switching inputs.
A provider switch also affects licensing, data access, fallback paths and support responsibilities.
Review selection and release criteria with the weather owner and operations desk.
No specific provider migration or performance change is asserted.

#### Exercise AW: handling partial boundary outages

Suppose one substation meter stops reporting during a storm.
The system aggregate may decrease even though underlying customer demand did not decrease proportionally.
It is also possible that the outage genuinely interrupts load, in which case lower net flow is real.
Use independent operations records and meter reconciliation to distinguish telemetry loss from physical interruption.
The modeled target must specify whether it represents observed net flow or a counterfactual served demand.
Observed flow can legitimately reflect an outage; counterfactual demand is a different and harder target.
If outage flags are only published after the issuance, they cannot be used as original-time features.
They may still explain realized error in a later diagnostic report.
Estimate missing telemetry only under an approved method and identify estimated intervals in the label.
Evaluate sensitivity to excluding disputed intervals and to using plausible alternative labels.
The planner must know when the target itself is uncertain, not only when the model is uncertain.
No outage event or impact is claimed for the hypothetical utility.

## 7. The Finding

- CLM-01: the hypothetical four-hour arithmetic favors the candidate on MAE but not on peak magnitude error; evidence is illustrative, not measured.
- CLM-02: a later-arriving weather cycle is not eligible at the original decision cutoff; this is a defined point-in-time contract, not a utility observation.
- CLM-03: independent hourly quantiles do not generally determine a peak quantile; Section 6 gives a complete counterexample.
- No candidate has been trained, evaluated, deployed, or demonstrated to improve an operational decision.
- The appropriate current decision is to retain the existing planning process and seek the missing archive, label, and workflow evidence.
- The unresolved link is from retrospective forecast scores to actual procurement or reliability outcomes.

## 8. System/Solution Design

### Proposed offline path

```text
Meter changes + weather vintages + calendar versions
    -> immutable time-stamped archive -> point-in-time materializer
    -> rolling-origin training/tuning -> locked future evaluation
    -> signed model + input contract + reproducibility manifest
```

The archive and materializer are distinct from the model registry: changing one cannot silently rewrite an issued forecast.
The evaluation process must apply the same eligible-input rules as the inference process.
The operations manager, forecast lead, security reviewer, and reliability owner must all sign release gates.
The registry stores both a candidate and approved fallback, with immutable model IDs and compatibility metadata.

### Proposed daily issuance path

```text
10:00 cutoff -> eligible snapshot -> validation -> forecast or fallback
             -> quantile/shape checks -> signed issuance record
             -> planner advisory display -> human planning decision
                                      -> alerts and audit log
```

Input validation checks expected variables, source freshness, units, boundary version, data coverage, and target-hour count.
A successful run publishes exactly one immutable initial issuance per boundary and operating day, with explicit origin and model version.
A corrected issuance is a new revision with a reason, timestamp, and pointer to the previous version; it never overwrites the original record.
The display exposes source freshness, uncertainty, forecast version, eligible horizon, and whether a fallback was used.
A planner can annotate a proposed override; the original forecast, editor, reason, and timestamp remain visible.
No API credentials in the forecast system permit dispatch, switching, market bidding, or reserve changes.
An operator must use the separate approved system to make any consequential decision.

### Failure, recovery, and readiness

If a weather cycle misses the cutoff, select the newest earlier complete eligible cycle only if its age is within the approved policy.
If no valid cycle is available, issue the separately validated weather-free fallback or escalate to the manual process.
If the target boundary map fails validation, do not publish a misleading forecast for a different population.
If a model artifact is missing, verify the signed approved fallback and record the reason for failover.
If both automated paths fail or miss the deadline, notify the on-call owner and planning desk; do not present stale output as current.
A late-arriving source can inform a separately timestamped revised forecast only if operational policy permits it.
On restart, recover the issuance state from a durable manifest and use idempotency keys to avoid duplicate conflicting outputs.
For a storage outage, preserve the authoritative manual process and reconcile any temporary records after recovery.
The desk must be able to distinguish an unavailable forecast from a low or zero demand estimate.
Validate recovery with injected failures before production; this is a planned exercise, not a passed test.

### Sizing and cost

Illustrative workload assumption: one boundary, one issuance per day, 24 typical target hours, and three quantiles per hour.
That produces `1 × 24 × 3 = 72` scalar quantile values per ordinary-day issuance before metadata.
At eight bytes per value the raw numeric payload is `72 × 8 = 576` bytes, excluding keys, serialization, indexes, storage replication and logs.
On a 25-hour day the same count is `25 × 3 = 75` values, or 600 raw bytes; there is no claim about actual wire size.
Assume a 15-minute end-to-end deadline and illustrative budgets of 3 minutes for snapshot, 4 for features, 3 for scoring, 2 for checks and 3 for publication.
The budgets sum to 15 minutes; they leave no allowance for a retry, so a production design needs a shorter measured critical path or earlier cutoff.
Cost accounting must include weather licensing, historical archive, compute, storage/replicas, data transfer, failed runs, review labor, on-call, security and change management.
Dollar rates and actual resource use are unknown; any budget requires dated vendor and labor inputs and a sensitivity analysis.
Do not confuse this raw-payload calculation with model training memory, archive size, or an availability guarantee.

## 9. Evaluation & Validation

### Temporal protocol

Choose a final contiguous holdout containing operationally relevant seasons before tuning, subject to sufficient archive coverage.
Within the earlier development period, use rolling origins: train only on information and mature labels available before each simulated origin.
Freeze all feature construction, preprocessing, model selection, calibration and thresholds before examining final-holdout outcomes.
Recreate the baseline and candidate at identical origins with identical cutoff and target definitions.
Each fold must re-fit transformations and model parameters using only the eligible training data for that fold.
Forecasts overlapping the evaluation interval do not entitle the training process to use labels that were not yet mature at the training cutoff.
Keep a horizon-specific eligibility and missingness report; avoid scoring only the successful model runs.
Score operational fallback outputs as part of the end-to-end process, while separately reporting candidate-only scores.
When the baseline is also unavailable, record the missing pair and report its characteristics rather than silently excluding it.
Predeclare handling of meter corrections, exceptional operating days, boundary changes, and daylight-saving folds.
Use the same mature label snapshot for both models; keep provisional-label scores as a separate operational monitoring view.

### Metrics and proposed gates

Primary metric: paired hourly MAE [MW], summed over all eligible target hours and divided by their count, separately for each system.
Comparator: same-origin seasonal-naive baseline, evaluated on the exact same target hours and finalized label version.
Population: one defined boundary and all next-local-day hours in the locked holdout; report coverage and exclusions.
Window: the predeclared locked future period, not yet chosen; desired direction is lower MAE.
Proposed selection rule: a paired day-block confidence interval for baseline-minus-candidate MAE must have a lower bound above zero.
That rule is a proposed gate, not a universal significance standard or a claim of measured improvement.
Also require no predeclared material degradation in signed bias, daily peak magnitude error, peak timing, and tail performance.
For each quantile level report mean pinball loss, quantile exceedance rate, and sample count by horizon and weather regime.
For the nominal 80% central interval, report inclusion count divided by scored hours and mean interval width in MW.
An 80% coverage claim alone can be gamed by excessively wide intervals; width must accompany it [S2].
For coherent scenarios report peak distribution scores and calibration of daily maximum and threshold exceedances.
Compare day-level peak error using one maximum per operating day; define ties deterministically and report timing in actual elapsed hours.
For peak threshold events, choose thresholds before final evaluation and report missed-event and false-alarm counts with denominators.
A threshold selected from the entire future dataset leaks future distributional information.
Report operational availability as on-time usable issuances divided by scheduled issuances, including fallbacks and missed days.
Break out fallback use, no-forecast days, corrupted outputs, late results and human escalation separately.

### Uncertainty, bias and prospective validation

Use paired blocks by day or longer weather episodes, preserving within-day dependence; do not bootstrap correlated hours as if independent.
A block bootstrap confidence interval is conditional on the sampled historical regimes; it does not guarantee performance in unprecedented extremes.
Report extreme-event counts, missing events, and uncertainty rather than a precise tail percentage based on a handful of days.
Compare load-level, season, day type, horizon, weather-vintage age, temperature regime and boundary-version slices.
A candidate can improve aggregate MAE while degrading hot-weather or evening-peak outcomes.
Calibration, sharpness, and peak metrics must be checked on the final postprocessed output, not raw model values.
Run a shadow period with timestamped baseline and candidate, but no operational influence, after retrospective gates are met.
A prospective human-use or controlled policy study needs approved design, plausible counterfactual, interference analysis, and independent safety oversight.
Do not attribute lower balancing costs to the forecast without accounting for weather, generation availability, price, operator actions and market rules.
All thresholds, confidence-interval methods and target event definitions require approval before examining final-holdout results.
No evaluation, bootstrap, shadow period, load test, or policy experiment has been executed for this case.

## 10. Action Taken

The only action represented here is drafting a proposed design and its evaluation plan.
No utility has adopted the forecast, changed its operating procedure, or authorized a trial as a result of this document.
The recommended next decision is to audit archive completeness and planning workflows before assembling a point-in-time retrospective dataset.
If the audit fails, retain the baseline and investigate a weather-free model or a prospective archive rather than fabricate weather vintages.
If offline gates pass, request an approved shadow evaluation with the same deadline, complete run logging, and no control permissions.
Even after a positive shadow evaluation, operational use requires a separate safety and governance decision.

## 11. Outcome & Results

| Claim | Evidence and status | Interpretation |
|---|---|---|
| Four-hour MAE comparison | Illustrative arithmetic, Section 6 | Not a utility backtest or performance estimate. |
| Weather-vintage exclusion | Constructed trace, Section 6 | A correct eligibility rule, not an observed provider failure. |
| Peak-quantile counterexample | Constructed probability model, Section 6 | Shows why marginal quantiles do not determine a joint maximum. |
| System performance | Not measured | No availability, recovery, latency, accuracy, or improvement results exist. |
| Business impact | Not measured | No attributable savings, reliability improvement, or customer effect exists. |
The source material contains no accessible utility case body, so no historical outcomes can be inferred.

## 12. Failure Modes & What Went Wrong

No real incident is described. Each row is a hypothetical failure, diagnosis, containment, and proposed prevention.
| ID | Symptom and competing causes | Discriminating check | Containment and preventive owner/test |
|---|---|---|---|
| F1 | Backtest is unusually accurate: realized weather leakage or genuinely easy weather | Compare selected issue and receipt IDs to origin, then stratify by weather | Reject ineligible features; data engineer tests late-arrival fixtures. |
| F2 | Forecast is systematically low: meter boundary change or model bias | Compare boundary maps and raw aggregates around change date | Hold issuance for map mismatch; metering owner tests historical remapping. |
| F3 | Missing latest weather: provider delay or local ingestion fault | Compare provider manifest, transport receipt, and local queue times | Use approved older cycle or fallback; weather owner injects both failure types. |
| F4 | Interval coverage deteriorates: narrow distribution or label revision | Compare mature versus provisional labels and interval width | Flag uncertainty; forecast lead recalibrates only on eligible past data and tests held-out coverage. |
| F5 | Daily peak is missed despite adequate hourly coverage: temporal dependence or marginal bias | Evaluate scenario peak calibration and per-hour exceedances separately | Avoid unsupported peak probability; model owner tests coherent scenarios. |
| F6 | Double issuance after retry: duplicate job or conflicting revision | Inspect idempotency key and immutable event sequence | Stop automatic republication; reliability owner tests restart and replay. |
| F7 | One hour missing on autumn clock change: key collision or source gap | Compare UTC instants and local fold flags | Escalate missing horizon; engineer tests 23/25-hour calendars. |
| F8 | Huge value: unit conversion or real extreme | Compare independent meter, unit metadata, and physical boundary limits | Quarantine only confirmed corruption; operations adjudicates genuine extremes. |
| F9 | Planner uses stale forecast: expired display cache or delayed issue | Compare displayed issuance timestamp with durable issuance record | Show explicit stale state; UI owner tests fail-closed expiration. |
| F10 | Override consistently increases error: changed information or human bias | Compare paired overrides, reason codes, timing, and outcomes | Review with operations without assuming overrides are causal; planner audits samples. |
| F11 | Apparently cheap solution ignores licensing and support | Reconcile invoice categories, labor and failed-run cost | Pause budget approval; finance validates full-cost scenarios. |
| F12 | A quantile ordering violation appears: separately fitted models or a postprocess bug | Inspect raw versus final quantile vectors and version | Reject invalid payload; model owner tests monotonicity after all transformations. |
| F13 | Boundary export appears negative: legitimate net export or sign error | Reconcile directional meter convention and independent telemetry | Preserve valid negative data; metering owner tests sign conventions. |
| F14 | Model score improves but operating cost rises: price asymmetry or dispatch coupling | Compare forecast slices and a separately approved decision replay | Retain manual control; operations and finance specify valid cost model. |
| F15 | Archived weather appears complete but only includes later backfills | Compare original local receipt logs with provider issue times | Exclude unknowable vintages; archive owner validates immutable ingestion records. |
Prevention checks are plans; none has been run and none proves that a real system would be safe.

## 13. Transferable Lesson

The primary pattern is `evaluation-mismatch`: the error measure that is easiest to optimize may not represent the action's risk.
A defensible forecast contract links the target, information cutoff, evaluation population, and human decision before comparing models.
The mechanism is a design hypothesis informed by the constructed examples, not a demonstrated operational improvement.
It requires reconstructable as-of data, a stable target boundary, mature labels, and a reviewer who can act within the deadline.
Do not transfer the design unchanged to real-time dispatch, feeder protection, or a territory without reliable vintage records.
For inventory forecasting, the corresponding boundary is sell-through versus unconstrained demand; stockouts can hide true demand.
For utility planning, observed net import can change because of behind-the-meter generation and boundary definitions.
The portability test is whether every selected feature and transformation could really have been known when the operational decision was made.

## 14. Connections & Cross-References

- [Retail CPG demand forecasting](01_Retail_CPG_Demand_Forecasting_At_Scale.md) shares point-in-time demand estimation but differs in its inventory target and decision consequences.
- [Ride-hailing demand forecasting](02_Ride_Hailing_Demand_Forecasting.md) also distinguishes predictive quality from a downstream human or marketplace action.
- [Rolling-origin evaluation](https://otexts.com/fpp3/tscv.html) [S3] provides the temporal split principle; receipt-time reconstruction remains an additional system requirement.
- [Probabilistic calibration and sharpness](https://academic.oup.com/jrsssb/article-abstract/69/2/243/7109375) [S2] supports reporting reliability together with concentration.
- Open question: Can original local weather receipts be reconstructed? Resolve with immutable ingest logs and provider manifests.
- Open question: How stable is the metered population? Resolve with historical boundary mapping and operational reconciliation.
- Open question: Are extreme-period uncertainty and peak scenarios useful? Resolve with out-of-time event analysis and a prospective shadow study.
- Open question: Does advisory use improve a safe operational decision? Resolve with an approved prospective evaluation and explicit counterfactual.

### Interview pack

Each answer describes the proposed case, not personal implementation experience. Scoring is 0 for a critical error, 1 for partial reasoning, and 2 for the complete mechanism and boundary.

#### Q1 — Frame the decision
- Question: What exactly is forecast and who acts?
- Answer: Next-local-day hourly average net boundary load in MW, with uncertainty, is provided to a planning desk; humans retain all operational authority.
- Reasoning: A boundary-specific advisory forecast is distinct from a commitment decision or feeder-level control.
- Follow-up: What changes if management requests automatic dispatch? A separate decision, safety case, controls and evaluation are required.
- Acceptable alternative: A different target is valid if its boundary, unit, cutoff, and owner are explicitly redefined.
- Red flag: Calling the forecast a dispatch command.
- Score: 2 requires target, horizon, owner and control boundary; 1 omits one; 0 grants model control without safeguards.
- Support: Sections 1–3 and 8.

#### Q2 — Diagnose a weather vintage
- Question: A 09:30 weather cycle arrives at 10:05; the feature cutoff is 10:00. Can it be used?
- Answer: No, because local receipt occurs after cutoff even though issue time is earlier.
- Reasoning: An operator cannot have used a forecast that had not arrived.
- Follow-up: If a validated earlier cycle exists, select it under the predeclared completeness and age rules.
- Acceptable alternative: An independently validated weather-free fallback, explicitly flagged.
- Red flag: Use the latest cycle available today for every historical origin.
- Score: 2 states both time tests and fallback; 1 identifies lateness only; 0 admits the late cycle.
- Support: Sections 5a and 6, S1's archive context.

#### Q3 — Compute paired MAE
- Question: What are candidate and baseline MAE in the four-hour example?
- Answer: Candidate 18/4 = 4.5 MW; baseline 25/4 = 6.25 MW.
- Reasoning: Absolute errors are paired over identical hours and averaged after summation.
- Follow-up: The 1.75 MW improvement corresponds to 28% of the baseline, but only in the constructed sample.
- Acceptable alternative: Report only the absolute difference if avoiding a small-sample percentage.
- Red flag: Report 28% as an achieved utility improvement.
- Score: 2 gives both numbers and limits; 1 gets one number; 0 compares different populations.
- Support: Section 6, M1.

#### Q4 — Explain quantile loss
- Question: What is 0.9 pinball loss when actual is 130 MW and forecast is 125 MW?
- Answer: `0.9 × 5 = 4.5 MW` because the actual exceeds the forecast.
- Reasoning: Quantile loss asymmetrically penalizes misses according to the specified quantile.
- Follow-up: A forecast of 135 MW yields `0.1 × 5 = 0.5 MW`.
- Acceptable alternative: State the equivalent max-of-two-linear-terms definition with consistent sign convention.
- Red flag: Treat 4.5 as a probability or compare a 90th percentile with a median MAE without noting the objective.
- Score: 2 calculates both directions; 1 calculates one; 0 reverses weights.
- Support: Section 6, M2; S4 for scoring-rule context.

#### Q5 — Diagnose apparent peak accuracy
- Question: Can the maximum hourly 90th percentile be treated as the 90th percentile of the daily maximum?
- Answer: Not generally; peak risk depends on the joint temporal distribution.
- Reasoning: In the two-hour example each hour's 90th percentile is zero but the maximum's is one.
- Follow-up: Produce coherent trajectories and validate the distribution of their maxima on held-out days.
- Acceptable alternative: Use a direct peak-distribution model if it is properly calibrated and separately validated.
- Red flag: Assume quantile operators commute with a maximum.
- Score: 2 explains dependence and gives a validation approach; 1 notes only dependence; 0 equates the two quantities.
- Support: Section 6 step 4.

#### Q6 — Design a temporal split
- Question: Why not randomly split hourly rows?
- Answer: It can leak future regimes, correlated adjacent targets, and transformations fitted on later data.
- Reasoning: The intended decision sees only past inputs, so rolling origins and an untouched later period better reproduce the information boundary.
- Follow-up: Refit transformations and ensure labels had matured at each simulated training cutoff.
- Acceptable alternative: A fixed chronological train/validation/test design when data is sufficient and all as-of constraints are retained.
- Red flag: Randomly split rows from the same forecast day across train and test.
- Score: 2 includes label maturity and as-of features; 1 mentions chronology only; 0 defends a leaky split.
- Support: Section 9; S3.

#### Q7 — Handle daylight saving time
- Question: What happens when the next local operating day has 25 hours?
- Answer: Publish 25 distinct UTC intervals, preserving local offset and fold for the repeated hour.
- Reasoning: Local clock labels alone are not unique keys.
- Follow-up: On a 23-hour day do not manufacture the missing local hour.
- Acceptable alternative: A market-defined interval convention if the actual operator's rules specify it and the system is designed accordingly.
- Red flag: Deduplicate on local date and hour.
- Score: 2 covers UTC identity and both clock transitions; 1 covers only one; 0 loses or duplicates an interval.
- Support: Sections 5a and 12, F7.

#### Q8 — Separate forecast and decision economics
- Question: Under a one-hour shortage cost of $200/MWh and excess cost of $50/MWh, which quantile minimizes the simplified expected loss?
- Answer: The 0.8 quantile, since `200/(200+50)=0.8` under the stated assumptions.
- Reasoning: Balance the expected marginal shortage and excess costs at an interior solution.
- Follow-up: Binding ramp, reserves, uncertain prices, and multi-hour resource constraints require a richer decision model.
- Acceptable alternative: A direct expected-cost optimization with a validated scenario distribution and actual constraints.
- Red flag: Say 0.8 is an actual grid reserve requirement.
- Score: 2 derives the fraction and caveats; 1 gives the fraction; 0 uses the excess-cost fraction.
- Support: Section 6 step 6.

#### Q9 — Resolve an ambiguous underforecast
- Question: A new month is systematically underforecast; how do you diagnose it?
- Answer: Compare target boundary versions, meter totals, weather forecast errors, label revisions, and load-regime changes before naming a cause.
- Reasoning: A changed boundary and a model weather bias can produce the same signed residual.
- Follow-up: If the boundary changed, reconcile historical labels and features before retraining.
- Acceptable alternative: A defensible ordered diagnostic based on available operational evidence.
- Red flag: Immediately retrain and declare distribution shift without checking data semantics.
- Score: 2 provides competing hypotheses, discriminating checks, and containment; 1 provides only one hypothesis; 0 asserts a cause without evidence.
- Support: Sections 5a and 12, F2.

#### Q10 — Assess uncertainty
- Question: Is 80% interval coverage sufficient by itself?
- Answer: No; report interval width, sample size, horizon and condition slices, and a proper score.
- Reasoning: Very wide intervals can cover frequently while providing little planning value.
- Follow-up: Use day or weather-event blocks for uncertainty intervals rather than assuming hourly independence.
- Acceptable alternative: Other proper distribution scores with transparent calibration diagnostics.
- Red flag: Claim safety from aggregate coverage alone.
- Score: 2 includes width and dependence; 1 mentions width; 0 treats coverage as sufficient.
- Support: Section 9 and S2.

#### Q11 — Recover from a failed run
- Question: The weather ingest misses the cutoff and the candidate cannot run. What should the planner see?
- Answer: An explicitly marked, validated fallback forecast if available, otherwise a no-forecast alert and the manual process.
- Reasoning: A stale or silently incomplete forecast is operationally ambiguous.
- Follow-up: Store fallback identity, source ages, issue time and reason; prevent duplicate publication on retry.
- Acceptable alternative: An earlier complete weather cycle if the approved age and completeness rules allow it.
- Red flag: Reuse yesterday's output as though it were fresh.
- Score: 2 includes identity, escalation and idempotency; 1 has only fallback; 0 conceals failure.
- Support: Section 8 and F3/F6/F9.

#### Q12 — Explain the changed constraint
- Question: What if archived weather receipt times cannot be recovered?
- Answer: Do not claim vintage-correct weather backtesting; evaluate a weather-free candidate or begin a prospective timestamped archive.
- Reasoning: Provider issue time does not prove internal availability.
- Follow-up: Compare fallback quality and operational consequences on a common valid cohort.
- Acceptable alternative: Conservative availability bounds supported by reliable records and a sensitivity analysis.
- Red flag: Replace forecast weather with realized observations.
- Score: 2 explains both loss of evidence and viable alternative; 1 notes only the limitation; 0 backfills with hindsight.
- Support: Section 4 A2, Section 5d.

#### Q13 — Evaluate an intervention
- Question: Does a better historical MAE prove lower procurement costs?
- Answer: No; the link depends on actions, prices, constraints, weather and the counterfactual.
- Reasoning: Operators may override a forecast, and the economically important hours may differ from the average hour.
- Follow-up: Use approved decision replay and then a carefully designed prospective evaluation with independent safety oversight.
- Acceptable alternative: A valid, bounded causal design appropriate to the actual operations and data.
- Red flag: Multiply MAE improvement by an assumed energy price and call it realized savings.
- Score: 2 identifies confounding and a credible next test; 1 only denies causality; 0 claims automatic savings.
- Support: Sections 3, 9 and 11.

#### Q14 — Evaluate reliability and sizing
- Question: What is the nominal numeric output size for 24 hourly targets and three eight-byte quantiles?
- Answer: `24 × 3 × 8 = 576` bytes before metadata, serialization, indexes, and replication.
- Reasoning: Count the quantile scalars first, then apply the assumed storage per scalar.
- Follow-up: A 25-hour day gives `25 × 3 × 8 = 600` bytes, still not a full storage or wire estimate.
- Acceptable alternative: Use a different encoded size if explicitly assumed and recalculate consistently.
- Red flag: Treat 576 bytes as total archive or infrastructure cost.
- Score: 2 includes both calculations and omitted overhead; 1 gives the first; 0 confuses bytes and megabytes.
- Support: Section 8 sizing.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Utility system-boundary day-ahead planning |
| Problem type | Hourly probabilistic net-load forecasting for advisory use |
| Method | As-of weather and load features; proposed quantile model versus same-weekday/hour baseline |
| Primary result | Not measured; the four-hour comparison is illustrative only |
| Key lesson | Align availability, forecast uncertainty and operational objective before trusting aggregate accuracy |
| Tags | Primary `evaluation-mismatch`; secondary `data-leakage`, `model-monitoring` |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-01-10-003 / 1.0 / 2026-11-07 |
| Evidence level | not-measured; constructed arithmetic is illustrative |
| Implementation stage | proposed; document is an unreviewed draft |
| Baseline / primary metric | Prior-week same-hour forecast; paired hourly MAE in MW over a locked future period |
| Key limitation | Original source and actual weather receipts, meter histories, and operational outcomes are unavailable |
| Review trigger / owner | Boundary, weather availability, label revision or planning deadline changes / utility load forecasting lead |
