---
case_id: "CASE-01-10-001"
case_title: "Retail CPG demand forecasting for bounded replenishment"
case_version: "1.0"
created: "2026-11-07"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Retail consumer packaged goods"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["data-leakage", "model-monitoring"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-07"
owner_role: "Retail replenishment lead"
source_index: "INDEX.md"
review_trigger: "Change to order timing, inventory semantics, or available historical snapshots"
---
# Case Study: Retail CPG Demand Forecasting at Scale

## 1. The One-Line Framing
Given point-in-time store-SKU sales, inventory, receipts, and approved promotion plans, propose a daily case-pack order for a replenishment planner, constrained by shelf capacity, pack size, and order deadlines, compared with a seasonal-naive reorder rule.
The task is probabilistic forecasting followed by a bounded replenishment decision, not merely minimizing prediction error.
Primary success is lower realized replenishment cost without unacceptable service degradation; it is not measured.
The primary pattern is evaluation-mismatch; secondary patterns are data-leakage and model-monitoring.
All entities, numbers, policies, calculations, incidents, and architecture below are constructed for teaching; there is no implied retailer, deployment, log, or measured outcome.
The directory entry names a topic, not an accessible implementation or evidence of a real system.

## 2. Context & Background
A fictional grocer sells ambient consumer packaged goods (CPG) through physical stores.
A stock-keeping unit (SKU) identifies one sellable product and pack size.
Each evening a planner reviews orders for the next supplier shipment.
An order placed today arrives before the store opens two mornings later, if the supplier performs as assumed.
Daily decisions seek to cover the interval until the next regular delivery while avoiding excess shelf and backroom stock.
Inventory position means on-hand plus reliable incoming units minus committed units, all as known at the cutoff.
Sales reflect fulfilled purchases, not necessarily customer demand when shelves are empty.
The scenario assumes 1,000 stores and 2,000 active SKUs per store, or up to two million decision series.
This is an illustrative upper bound, not an observed store fleet or benchmark.
The replenishment lead owns the action; store operations own physical counts; data engineering owns feeds; ML operations owns model releases.
Merchandising owns the promotion plan, and supplier operations owns shipment commitments.
The assumed current approach uses the previous matching weekday's sales and a fixed safety allowance, with planner edits.
No previous model attempts, measured baseline weaknesses, legal requirements, or actual company conditions are asserted.
The time zone is local store time for business days; timestamps are recorded in UTC with store-zone history.
The scope excludes perishables, returns to vendor, transfer optimization, price selection, and causal promotion effects.
A planner can reject a recommendation; the scenario does not assume automated purchase orders.

## 3. The Business Question
Illustrative stakeholder question: Which pack-multiple order should each eligible store place before tonight's cutoff to cover its next protection period at acceptable shortage and holding risk?
The decision owner is the replenishment lead, not the model developer.
Available actions are current rule, proposed forecast-assisted order, or defer to a planner when inputs fail.
The daily cutoff triggers one decision per eligible store-SKU.
The proposed objective is total purchase-order-independent shortage and holding cost over the evaluation window.
Purchase cost is omitted only where equal across choices; changing unit cost or discounts requires a new objective.
Hard scenario constraints are nonnegative integer packs, shelf capacity, valid item-store eligibility, and a supplier order deadline.
Business costs and service risk can disagree with weighted absolute percentage error (WAPE); the latter is a diagnostic proxy only.
No-change is the current rule, with its planner override workflow preserved.
A failed forecast does not authorize an unconstrained order.

## 4. Assumptions & Constraints
| ID | Assumption or constraint | Type and status | Verification plan | Impact and fallback | Owner / trigger |
|---|---|---|---|---|---|
| A-01 | Two-day lead time and daily ordering | Business, illustrative | Compare historical arrival timestamps with orders | Late arrival invalidates protection period; planner review | Supplier operations / lead-time shift |
| A-02 | Inventory snapshots represent saleable units | Data, unverified | Reconcile counted, damaged, reserved and shelf stock | Bad position changes order; hold stale item | Store operations / reconciliation failure |
| A-03 | Promotion plan is known at cutoff | Data, unverified | Replay publication and ingestion timestamps | Future-plan leakage; use last published version | Merchandising / late amendments |
| A-04 | Stockouts are observable | Data, unverified | Compare shelf availability with receipts and counts | Sales understate latent demand; flag uncertainty | Data owner / missing availability |
| A-05 | Costs can be approximated per protection period | Business, illustrative | Finance reviews margin, substitution and carrying cost | Wrong quantile and order; default rule | Finance / cost revision |
| C-01 | Orders fit physical maximum and case multiples | Physical, scenario rule | Reconcile pack master and capacity master | Reject invalid proposal, do not silently round beyond cap | Replenishment / master-data change |
| C-02 | Cutoff must be met | Operational, scenario rule | Measure end-to-end timeliness at order handoff | Fallback or defer; missed shipments | Operations / schedule change |
| C-03 | Limit access to operational data | Security, proposed | Review service identity, permissions and retention | Disable unauthorized access, investigate | Security / access change |
The three most consequential assumptions are inventory correctness, lead-time stability, and point-in-time availability of promotion and inventory data.
An approved promotion is an input only if published and ingested by the decision cutoff; approval after that time cannot be backdated into a backtest.
The hard constraints are scenario choices, not claims about an external policy.
Privacy and legal applicability require an actual jurisdiction and assessment; store-SKU aggregation is not a substitute for access controls.

## 5. Data, Tools & Method
### 5a. Data Specification
The decision grain is store ID, SKU ID, and local order date; forecasts cover cumulative demand in the specified protection period.
Inputs are transaction sales, inventory position, shipment status, store calendar, product master, price, and promotion plan.
Each source has event time, publication time, ingestion time, and version; the latest revision is not necessarily the historical truth available to a decision.
The illustrative historical window is two years with weekly rolling origins, and a later untouched four-week test; no dataset was obtained.
Events belong to their local business day using the time zone effective for that store on the event date.
Training labels are final observed unit sales after the outcome-maturity delay, but the label represents sales rather than uncensored demand on unavailable days.
A stockout indicator gates evaluation segments; a latent-demand target requires independently justified estimation and is not treated as observed truth.
Exclude permanently closed stores and discontinued item-store pairs before each decision date, not using future closure status.
Zero sales on an open stocked day differ from missing sales, a closed store, or unavailable inventory.
Join promotion and product history using effective-time intervals and only records available by the historical cutoff.
Enforce one eligible record per decision key; quarantine duplicate master records rather than multiplying transaction rows.
Hold out future time, preserve overlapping-horizon dependence in uncertainty calculations, and avoid randomly splitting daily rows.
#### Point-in-time field contract
Each contract is illustrative and must be mapped to actual source owners and retention rules before implementation.
##### D-01. Store identity
- Meaning: Stable store identifier.
- Validation: Reject unknown or remapped keys.
- Proposed source: Store master effective history; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: A merged store can silently mix locations; quarantine uncertainty instead of silently inventing a value.
##### D-02. SKU identity
- Meaning: Sellable SKU identifier.
- Validation: Reject unmatched item.
- Proposed source: Product master effective history; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: A product transition can corrupt historical series; quarantine uncertainty instead of silently inventing a value.
##### D-03. Decision origin
- Meaning: Local cutoff and UTC instant.
- Validation: Reject absent or ambiguous cutoff.
- Proposed source: Replenishment schedule; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: A date alone does not establish feature availability; quarantine uncertainty instead of silently inventing a value.
##### D-04. Business date
- Meaning: Store-local trading day.
- Validation: Derive using effective zone.
- Proposed source: Store calendar and time-zone history; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: UTC midnight can place a sale in the wrong day; quarantine uncertainty instead of silently inventing a value.
##### D-05. Transaction identifier
- Meaning: Unique source sale event ID.
- Validation: Deduplicate exact delivery retries.
- Proposed source: Point-of-sale event stream; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Conflicting same-ID events require source reconciliation; quarantine uncertainty instead of silently inventing a value.
##### D-06. Sale event time
- Meaning: When the sale occurred.
- Validation: Keep separate from ingestion time.
- Proposed source: Point-of-sale event; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Late-arriving events cannot become historical features early; quarantine uncertainty instead of silently inventing a value.
##### D-07. Sales quantity
- Meaning: Gross fulfilled saleable units.
- Validation: Validate integer units and transaction type.
- Proposed source: Point-of-sale line; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Sales alone do not reveal unmet demand; quarantine uncertainty instead of silently inventing a value.
##### D-08. Return quantity
- Meaning: Units returned by customers.
- Validation: Keep separate from gross sales.
- Proposed source: Returns stream; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Return posting dates need not match sale dates; quarantine uncertainty instead of silently inventing a value.
##### D-09. Record publication time
- Meaning: When a source released a record.
- Validation: Require no later than origin.
- Proposed source: Source publication metadata; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Later corrections cannot be backdated into replay; quarantine uncertainty instead of silently inventing a value.
##### D-10. Record ingestion time
- Meaning: When the pipeline could access the record.
- Validation: Require no later than origin.
- Proposed source: Landing-zone audit log; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Publication alone does not prove pipeline availability; quarantine uncertainty instead of silently inventing a value.
##### D-11. Inventory on hand
- Meaning: Count of stock on premises.
- Validation: Check unit and age.
- Proposed source: Inventory snapshot; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: On-hand includes unsaleable stock unless defined otherwise; quarantine uncertainty instead of silently inventing a value.
##### D-12. Saleable status
- Meaning: Whether units can actually be sold.
- Validation: Exclude damaged and quarantined units.
- Proposed source: Inventory status ledger; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Status changes can occur between snapshots; quarantine uncertainty instead of silently inventing a value.
##### D-13. Committed units
- Meaning: Stock already promised to orders.
- Validation: Deduct only if not already netted.
- Proposed source: Order reservation ledger; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Double subtraction inflates replenishment; quarantine uncertainty instead of silently inventing a value.
##### D-14. Reliable incoming units
- Meaning: Confirmed receipts expected before demand exposure.
- Validation: Use current supplier commitment.
- Proposed source: Shipment and order feeds; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Late shipments invalidate position; quarantine uncertainty instead of silently inventing a value.
##### D-15. Arrival timestamp
- Meaning: Time units became available for sale.
- Validation: Compare with promised arrival.
- Proposed source: Receipt ledger; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Warehouse arrival is not necessarily shelf availability; quarantine uncertainty instead of silently inventing a value.
##### D-16. Availability indicator
- Meaning: Whether customers could purchase during a period.
- Validation: Distinguish unknown from available.
- Proposed source: Shelf availability or reliable proxy; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: A daily flag may hide an intraday stockout; quarantine uncertainty instead of silently inventing a value.
##### D-17. Store operating hours
- Meaning: Intervals when store is open.
- Validation: Use effective schedule known at origin.
- Proposed source: Store calendar; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Closure zero differs from zero customer demand; quarantine uncertainty instead of silently inventing a value.
##### D-18. Promotion identifier
- Meaning: Approved planned promotion.
- Validation: Use revision available at cutoff.
- Proposed source: Merchandising plan history; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Final campaign details can leak into earlier forecasts; quarantine uncertainty instead of silently inventing a value.
##### D-19. Promotion effective interval
- Meaning: Start and end of approved campaign.
- Validation: Join by effective and availability times.
- Proposed source: Promotion version ledger; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Intervals can overlap during amendments; quarantine uncertainty instead of silently inventing a value.
##### D-20. Price
- Meaning: Customer-facing unit price.
- Validation: Normalize currency and unit.
- Proposed source: Price history; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Post-hoc corrections are not historical inputs; quarantine uncertainty instead of silently inventing a value.
##### D-21. Price effective interval
- Meaning: When a price applies in store.
- Validation: Use valid version as of cutoff.
- Proposed source: Price version ledger; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: A price-change event can arrive late; quarantine uncertainty instead of silently inventing a value.
##### D-22. Case-pack size
- Meaning: Saleable units per supplier case.
- Validation: Require positive integer and effective version.
- Proposed source: Supplier item master; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Pack changes can invalidate previously feasible quantities; quarantine uncertainty instead of silently inventing a value.
##### D-23. Capacity
- Meaning: Maximum additional saleable units.
- Validation: Require nonnegative quantity and current scope.
- Proposed source: Store layout master; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Shared receiving capacity is not a per-item limit; quarantine uncertainty instead of silently inventing a value.
##### D-24. Supplier lead time
- Meaning: Interval from accepted order to saleable receipt.
- Validation: Retain a distribution if uncertain.
- Proposed source: Order and receipt history; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Mean lead time can mask tail disruption; quarantine uncertainty instead of silently inventing a value.
##### D-25. Order cutoff
- Meaning: Latest valid handoff instant.
- Validation: Enforce store and supplier calendar.
- Proposed source: Order schedule; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Time-zone and holiday shifts change the deadline; quarantine uncertainty instead of silently inventing a value.
##### D-26. Eligibility interval
- Meaning: Dates when store can replenish SKU.
- Validation: Use status known at each origin.
- Proposed source: Assortment master; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Future delisting must not exclude earlier examples; quarantine uncertainty instead of silently inventing a value.
##### D-27. Store opening date
- Meaning: First active trading day.
- Validation: Audit known launch status.
- Proposed source: Store master history; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Cold-start evaluation must not use future sales; quarantine uncertainty instead of silently inventing a value.
##### D-28. Item launch date
- Meaning: First item availability date.
- Validation: Audit effective assortment.
- Proposed source: Item and assortment history; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Similar-item mapping can itself be revised later; quarantine uncertainty instead of silently inventing a value.
##### D-29. Holiday calendar
- Meaning: Local nonstandard trading dates.
- Validation: Version by region and year.
- Proposed source: Approved store calendar; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Calendar effects vary with region and store hours; quarantine uncertainty instead of silently inventing a value.
##### D-30. Feature missingness
- Meaning: Whether a field was truly unknown.
- Validation: Use explicit missing-state indicator.
- Proposed source: Feature pipeline metadata; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Silently imputing a zero changes the meaning; quarantine uncertainty instead of silently inventing a value.
##### D-31. Label maturity
- Meaning: When sales and corrections are final enough for evaluation.
- Validation: Delay scoring until agreed maturity.
- Proposed source: Source close process; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Premature labels can create false forecast regressions; quarantine uncertainty instead of silently inventing a value.
##### D-32. Model artifact version
- Meaning: Exact forecast estimator and parameters.
- Validation: Bind to a checksum.
- Proposed source: Model registry; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: An unversioned model prevents reproducible replay; quarantine uncertainty instead of silently inventing a value.
##### D-33. Feature schema version
- Meaning: Field names, units, transforms and semantics.
- Validation: Require compatible model bundle.
- Proposed source: Feature registry; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Matching column names do not prove matching meaning; quarantine uncertainty instead of silently inventing a value.
##### D-34. Policy version
- Meaning: Rounding and constraint behavior.
- Validation: Record version on each decision.
- Proposed source: Replenishment rules repository; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Changing policy can move outcomes without model drift; quarantine uncertainty instead of silently inventing a value.
##### D-35. Recommendation revision
- Meaning: Version of proposed quantity for an order date.
- Validation: Reject conflicting duplicate revisions.
- Proposed source: Recommendation store; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Manual correction must not create an untracked new order; quarantine uncertainty instead of silently inventing a value.
##### D-36. Override reason
- Meaning: Structured planner explanation.
- Validation: Allow missing only with explicit reason.
- Proposed source: Planner workflow; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Override selection is not a randomized label; quarantine uncertainty instead of silently inventing a value.
##### D-37. Downstream acknowledgement
- Meaning: Supplier or order system acceptance status.
- Validation: Reconcile ambiguous timeouts.
- Proposed source: Order handoff ledger; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: A timeout does not establish nonreceipt; quarantine uncertainty instead of silently inventing a value.
##### D-38. Source manifest
- Meaning: Expected partitions and extraction watermarks.
- Validation: Require completeness before scoring.
- Proposed source: Data platform control plane; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: An absent partition is not a zero-sales partition; quarantine uncertainty instead of silently inventing a value.
##### D-39. Capacity unit
- Meaning: Units, cases, weight or volume of permitted storage.
- Validation: Convert before comparing quantities.
- Proposed source: Store capacity master; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: Mixed units silently break constraint checking; quarantine uncertainty instead of silently inventing a value.
##### D-40. Prediction horizon
- Meaning: Dates covered by cumulative forecast.
- Validation: Match order and receipt timetable.
- Proposed source: Forecast configuration; the source has not been accessed for this case.
- Availability: retain the record version and the publication and ingestion instants relevant to the forecast origin.
- Failure boundary: A daily forecast is not automatically the protection-period distribution; quarantine uncertainty instead of silently inventing a value.
### 5b. Tools & Stack
The proposed stack is an orchestrated batch pipeline, immutable versioned object storage, a warehouse, a model registry, and a planner-facing order export.
No specific vendor, library version, cloud instance, hardware benchmark, software deployment, or run is asserted.
Warehouse queries prepare point-in-time features; a batch model job emits distributional demand forecasts and lineage.
A separate deterministic policy applies cost, position, capacity, and pack constraints; this separation permits replay.
The data platform owns snapshots, the modeling team owns forecasts, and the replenishment platform owns order validation.
### 5c. Method Summary
The baseline predicts the prior matching weekday's demand, aggregated over the protection period, then applies the same pack and capacity constraints.
The baseline uses the same available data, stockout treatment, and decision deadline as the proposed method.
A seasonal moving average smooths noise but can lag promotions and distribution shifts.
A global quantile regression model shares information across related series; it is only proposed, not fitted.
A hierarchical approach can reconcile store-level and regional aggregate forecasts, but reconciled quantiles need not preserve calibration.
Intermittent-demand methods can beat a global model on sparse items; selection needs comparable rolling-origin backtests.
A deep sequence model may capture nonlinear seasonality but introduces complexity and must beat simpler candidates on decision cost.
Fit quantiles separately or use a distributional objective; check quantile crossing and repair monotonically with recalibration validation.
Hyperparameters are selected on earlier rolling validation origins; the last chronological test remains untouched.
No alternative was tested, and no model-selection claim is made.
### 5d. Provenance, Quality, Exclusions and Leakage Controls
The source directory inventory identifies only the desired topic; its content is unavailable and no source-system records were accessed.
SRC-01: [Hyndman and Athanasopoulos, Forecasting: Principles and Practice, third edition, time-series cross-validation](https://otexts.com/fpp3/tscv.html), online edition; accessed 2026-11-07; supports rolling-origin evaluation, not any retail result.
SRC-02: [scikit-learn common pitfalls, data leakage](https://scikit-learn.org/stable/common_pitfalls.html), online documentation, accessed 2026-11-07; supports fitting transformations without test information, not model efficacy.
Recheck these method references when the cited material or evaluation design changes; neither source describes this fictional architecture.
Retain immutable extraction manifests, source watermarks, missingness, duplicate counts, and schema versions for any real run.
Permission to access an aggregate does not imply permission to join customer-level transactions; use minimum required fields.
Stockout periods are not thrown away without reporting the excluded share by store, item, and season.
A promotion effective tomorrow is usable only if the correct version was actually available today.
A corrected inventory snapshot published tomorrow is prohibited as a feature for today's historical forecast.
Fit imputers, encoders, scaling, and model selection within each training fold; future information remains outside each fit.
### 5e. Reproducibility and Method-Selection Ledger
RUN-01: no training or backtest was run; no code revision, model artifact, seed, dependencies, or data snapshot exists.
A future run must record immutable source manifests, time-zone database version, code revision, dependencies, parameters, fold definitions, seeds, and model checksum.
A seed is insufficient to guarantee identical results across nondeterministic hardware and software environments.
DEC-01 selects a proposed global quantile forecast for evaluation, not production adoption.
DEC-02 retains the seasonal rule as the required baseline, not a claimed inferior result.
CLM-01 is the bounded arithmetic in Section 6; it is derived from constructed inputs, not measured business performance.
CLM-02 is the proposed evaluation requirement in Section 9; it is not evidence that a release passed.

## 6. Analysis Walkthrough
The decisive constructed example shows that decision constraints can change the order even when a demand target is known; it does not establish an improved forecast.
At the cutoff, assume 20 saleable units on hand, 8 reliable units due before the protection period, and 3 units committed to existing orders.
These values are illustrative, mutually exclusive inventory categories and have no backing transaction records.
Inventory position is 20 + 8 - 3 = 25 units.
Suppose a calibrated cumulative demand forecast for the protection period has an illustrative decision target of 53 units.
The unconstrained shortage to target is 53 - 25 = 28 units.
The supplier case pack is 6 units, and remaining physical capacity admits at most 30 additional units.
The following policy chooses the smallest feasible pack multiple that reaches the target when possible; otherwise it chooses the maximum feasible multiple.
This operational rule is not necessarily the exact cost-minimizing discrete action when pack rounding overshoots the demand distribution.
```text
FORMULA [Q-01]: P = H + R - C
WHERE: P inventory position [units]; H saleable on-hand [units];
       R reliable incoming [units]; C commitments [units].
CONSTRAINT: no double counting; R arrives in time; non-saleable units excluded.
NUMERICAL EXAMPLE: P = 20 + 8 - 3 = 25 units.
INTERPRETATION: only 25 units are available to cover future demand under assumptions.
FORMULA [Q-02]: q = min(k*ceil(max(0,T-P)/k), k*floor(M/k))
WHERE: q order [units]; k pack size [units/pack]; T target [units];
       P position [units]; M incremental physical capacity [units].
CONSTRAINT: k positive integer; M nonnegative; no minimum-order or shared-capacity constraints.
NUMERICAL EXAMPLE: q = min(6*ceil(28/6),6*floor(30/6)) = min(30,30) = 30 units.
INTERPRETATION: five packs yield position 55 units, two above the target.
```
Independent check: four packs yield 24 units and position 49, below target; five yield 30 units and position 55; six packs exceed capacity.
If capacity falls to 24, the order is 24 and position is 49; the target is missed by four units and an exception is sent to the planner.
If position increases to 56, the proposed order is zero; negative orders are not returned to suppliers automatically.
The target of 53 must come from a properly calibrated protection-period distribution, not the sum of daily marginal quantiles.
For a single-period idealization with linear underage cost Cu and overage cost Co, the critical ratio is Cu/(Cu+Co).
This result assumes known demand distribution, independent unconstrained unit ordering, risk neutrality, and no salvage or fixed costs beyond the given costs.
```text
FORMULA [Q-03]: alpha = Cu / (Cu + Co)
WHERE: alpha desired demand quantile [dimensionless]; Cu underage cost [currency/unit];
       Co overage cost [currency/unit].
CONSTRAINT: Cu and Co nonnegative and sum positive; continuous-demand idealization.
NUMERICAL EXAMPLE: Cu = 4 and Co = 1, so alpha = 4/(4+1) = 0.8.
INTERPRETATION: the 80th percentile is the ideal unconstrained target under these assumptions.
```
The marginal cost of adding a unit is Co when demand is below the unit and the marginal avoided shortage cost is Cu when demand is above it; equating expected marginal costs yields Q-03.
Independent check: 0.8*1 = 0.2*4, so the expected marginal overage and underage terms balance at the ideal quantile.
If Cu falls to 1 with Co still 1, the ratio is 0.5; the relevant target becomes the median, whose value is not supplied.
If both costs are zero, the ratio is undefined and no cost-based target is identified.
For integer demand or case packs, evaluate expected costs of feasible pack orders directly; the continuous formula is an informative starting point, not a guarantee of the discrete optimum.
Quantiles must cover demand across lead time plus review interval as defined by the actual order and receipt timetable.
A late supplier and uncertain lead time widen or mix that protection-period distribution; the fixed-horizon target may understate shortage risk.
No real baseline, score, revenue, intervention effect, or improvement has been measured.

## 7. The Finding
Do not deploy the proposed model automatically: evaluate it against the current seasonal rule first, with the same cutoff and order constraints.
In CLM-01's constructed example, the feasible policy produces five packs, or 30 units, from a 53-unit target and a 25-unit position.
That arithmetic is not a measured improvement or proof of minimum expected cost under discrete packs.
Inventory accuracy, latent demand during stockouts, and lead-time uncertainty could change the result; the next justified action is a historical point-in-time replay followed by controlled evaluation.

## 8. System/Solution Design
### 8a. Offline / Training / Batch Path
```text
Sales + inventory + shipment + approved plans -> immutable snapshots -> validate/quarantine
   -> point-in-time joins -> rolling train/validation -> model and calibration gate
   -> versioned registry -> scheduled batch forecast -> deterministic bounded policy
   -> planner review -> idempotent order handoff -> receipt and sales feedback
```
Each arrow is proposed, not an assertion of a deployed component.
Features are computed only from the snapshot visible at the forecast origin.
Persist source manifests, schema, lineage, model version, policy version, and decision timestamp alongside every output.
The scheduler partitions by store and date, bounds retry work, and marks incomplete partitions rather than inventing zero forecasts.
### 8b. Online / Serving / Real-Time Path
The system is batch-first; there is no proposed low-latency online inference API.
A planner-facing review and order handoff is an interactive control path, not a separate forecast-serving claim.
```text
Planner identity -> authorized store scope -> latest eligible recommendation + freshness
 -> approve/override/defer -> constraint recheck -> idempotent order export
 -> acknowledgement + audit record; failure -> hold + alert + owned manual process
```
A valid request identifies the store, SKU, business date, recommendation version, actor, and action.
The handoff validates current capacity, item eligibility, and cutoff again because these may change after batch generation.
The idempotency key is the store-SKU-order-date and decision revision; the downstream contract must reject conflicting duplicates.
Approval cannot silently overwrite an already accepted order: reconcile downstream acknowledgements before issuing a correction.
If model, inventory, or registry inputs are stale, mark the recommendation unavailable and route to the approved manual process.
A seasonal fallback is valid only if its own input freshness and hard constraints pass; otherwise defer and escalate.
Log a reason code for every override without recording unnecessary personal information.
### 8c. Key Design Decisions
| ID | Choice | Alternative | Tradeoff and boundary | Revisit |
|---|---|---|---|---|
| DEC-01 | Batch global quantile model proposed | Local or seasonal model | Shared statistical strength versus heterogeneous segments | Fails rolling-origin decision gate |
| DEC-02 | Same bounded policy for candidate and baseline | Different policy per model | Fair comparison; may hide best jointly optimized policy | After isolated model comparison |
| DEC-03 | Planner approval | Automatic ordering | Safer initial release, requires staff capacity | After controlled evidence and approval |
| DEC-04 | Quarantine missing position | Forward-fill indefinitely | Lower silent error, more manual cases | Verified bounded inventory freshness |
### 8d. Architecture Artifacts and Operational Readiness
A proposed on-time recommendation SLI is eligible decision keys with a valid, constraint-checked output before cutoff divided by all eligible keys that day.
Proposed target: at least 99% over a rolling 28-day window; it is a scenario objective, not an observed service level.
Also measure late, quarantined, fallback, and manually resolved keys separately so excluding failures cannot inflate the SLI.
Alert the on-call data engineer when input freshness breaches the configured cutoff; alert replenishment operations when pending orders approach deadline.
Use bounded retries with backoff before cutoff; after cutoff stop automatic retry and escalate rather than sending late duplicate orders.
Hold a previous model and matching feature/policy schema for rollback; verify backward compatibility before any release.
Use least-privilege service identities, encrypted transport and storage, audited role-based planner access, and retention approved by the data owner.
Restore exercises must verify source manifests, replayed decision IDs, order acknowledgements, and no duplicate purchase orders; none has been executed.
For illustrative sizing, two million eligible series times 28 horizon-day forecast values equals 56 million values per daily run.
At four bytes per value the raw numeric payload is 224 million bytes, or approximately 213.6 MiB, before identifiers, metadata, storage encoding, replicas, logs, and intermediate data.
Independent check: 2,000,000 * 28 * 4 = 224,000,000 bytes; divide by 1,048,576 for MiB.
Doubling stores or horizon doubles this raw payload, while cross-features and model state can dominate actual memory; no throughput or latency benchmark follows.
Use partition-level manifests and checkpoints to resume only failed work; ensure every partition sees the correct cutoff and model version.
No price is supplied: compute, storage, egress, retries, and planner labor remain unpriced and require finance measurement.

## 9. Evaluation & Validation
The intended protocol is not executed; no gate has passed or failed.
Use rolling-origin training and validation, followed by a locked final four-week chronological test with mature labels.
Both baseline and candidate see identical point-in-time data, constraints, eligibility, and review deadlines.
The primary proposed decision metric is total shortage-plus-holding cost in currency per eligible store-SKU-period, aggregated as total cost divided by eligible periods.
Its numerator is realized modeled shortage and holding cost from transaction, inventory, and agreed cost records; its denominator includes all eligible periods, including fallback cases.
Actual lost demand is not directly observed during stockouts, so costs need a validated demand estimate or a separately identified experiment; report both uncertainty and censoring.
The illustrative decision gate is at least 2% lower cost than baseline on the locked test with no deterioration beyond agreed guardrails, selected before evaluation.
This is a proposed threshold, not a measured gain or universally appropriate minimum.
Guardrails include units filled divided by units demanded where demand is observable, stockout rate, waste, planner workload, order validity, and on-time recommendation SLI.
WAPE is sum of absolute unit errors divided by sum of observed actual unit sales, over eligible mature store-SKU-periods; undefined when actual total is zero.
For illustration, actual sales [10,20] and forecasts [12,17] give absolute errors 2 and 3; WAPE = 5/30 = 16.67%, not the mean of per-item percentages.
These constructed numbers are not a backtest, and WAPE on censored sales can reward a model that predicts stockouts.
Quantile coverage is the share of mature uncensored observations at or below the forecast quantile; dependence and discreteness affect interpretation.
Check coverage by promotion, store, product family, intermittency, horizon, and availability, with sample counts and uncertainty.
Use clustered or block resampling by store and time as appropriate for dependence; overlapping protection periods are not independent observations.
A model may reduce forecast error while worsening cost if errors shift into expensive shortages; evaluate decisions directly.
For any pilot, randomize at a unit that limits shared supplier and store spillovers, record eligibility before assignment, and track contamination and sample-ratio imbalance.
A pilot's power, minimum detectable effect, allocation, stopping rule, and owner require actual historical variance and operational input; none is supplied.
Use shadow mode to validate interfaces and timeliness but do not claim causal business benefit from shadow predictions.
After shadow, a pre-specified controlled pilot could estimate incremental outcomes if randomization and measurement assumptions hold.
No artifact, sample size, observed interval, production outcome, or causal estimate exists for this scenario.

## 10. Action Taken
No action taken as of 2026-11-07; this is a proposed design exercise with no organization or approval record.
The recommended next decision is whether the replenishment owner authorizes point-in-time replay and a shadow implementation.
The proposed owner would approve the protocol and thresholds before the final test, with data engineering supplying lineage and operations supplying manual fallback.
No store, supplier, model version, or customer is in an actual release scope.
A future rollout would progress from replay to shadow to a small controlled pilot, subject to operational readiness and formal approval.
A future stop trigger is a breached pre-specified service guardrail, invalid order, material input failure, or missed cutoff; operations owns the stop.
Rollback means disable model-assisted recommendations, restore a compatible previous version if safe, and reconcile any exported orders before returning to manual rules.
No automatic ordering is recommended on the basis of this constructed example.

## 11. Outcome & Results
Outcome not yet measured; no implementation or evaluation has occurred.
Figures in this case are constructed teaching values, not measurements of a named organization or evidence of realized impact.
| Metric | Baseline | Candidate | Change and uncertainty | Status / limitation |
|---|---|---|---|---|
| Cost per eligible period | Not measured | Not measured | Not estimable | Proposed locked test; censored demand concern |
| Demand fill rate | Not measured | Not measured | Not estimable | Requires defensible demand denominator |
| On-time eligible recommendations | Not measured | Not measured | Not estimable | Proposed SLI; no service exists |
| Planner override and labor | Not measured | Not measured | Not estimable | Requires action logs and time study |
| Waste and excess inventory | Not measured | Not measured | Not estimable | Requires consistent valuation and measurement window |
The proposed measurement owner is the replenishment analytics lead, using the pre-specified locked test and later controlled pilot if authorized.
The proposed decision trigger is the Section 9 gate together with operational and business guardrails; an inconclusive result does not authorize a claim of improvement.

## 12. Failure Modes & What Went Wrong
No incident or rejected mitigation occurred; the following are hypothetical diagnostic exercises, not observed failures.
Each exercise starts with a symptom, separates plausible causes, and gives a safe intervention plus verification.
### F-01. Stockout-censored sales
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Flat low sales despite repeated empty shelves.
- Competing explanation: True demand decline.
- Discriminating check: Compare shelf availability and receipts with sales; a shelf-empty interval with demand proxies supports censoring.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Hold automatic quantity and route to planner; preserve the original output and decision audit trail.
- Prevention: Estimate demand only with a validated censoring approach; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-02. Inventory snapshot delay
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Recommendations spike after inventory-feed delay.
- Competing explanation: Genuine demand surge.
- Discriminating check: Compare ingestion watermark and snapshot age with transaction timestamps.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Quarantine stale positions and use approved manual review; preserve the original output and decision audit trail.
- Prevention: Set freshness SLA and replay delayed-feed scenarios; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-03. Late promotion amendment
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Promotion forecast is unexpectedly optimistic in backtest.
- Competing explanation: Leakage from a later plan revision.
- Discriminating check: Compare feature publication and ingestion times with each forecast origin.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Rebuild affected folds using available plan versions; preserve the original output and decision audit trail.
- Prevention: Contract immutable promotion revisions and cutoff checks; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-04. Duplicate transaction events
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Sales and order quantities double for a subset.
- Competing explanation: Genuine high volume.
- Discriminating check: Compare transaction IDs and source reconciliation totals.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Deduplicate by source event identity and hold affected partitions; preserve the original output and decision audit trail.
- Prevention: Test retries and duplicate delivery in ingestion; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-05. Missing transaction partition
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: A whole store appears to have zero demand.
- Competing explanation: Store closure.
- Discriminating check: Compare partition manifest, register health, and store open status.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Mark sales missing rather than zero and defer; preserve the original output and decision audit trail.
- Prevention: Alert on missing partitions before forecast deadline; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-06. Wrong store time zone
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Forecast errors peak around midnight or daylight-saving changes.
- Competing explanation: Real late-night shopping shift.
- Discriminating check: Recompute event-to-business-day attribution using effective store-zone history.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Quarantine affected days and rebuild using correct time zone; preserve the original output and decision audit trail.
- Prevention: Test DST gaps, repeated hours, and zone changes; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-07. Unit conversion error
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: One SKU forecast differs by a factor of case size.
- Competing explanation: Demand step change.
- Discriminating check: Compare unit, case and pack master changes to transaction units.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Block incompatible unit conversions and require reconciliation; preserve the original output and decision audit trail.
- Prevention: Version the unit master and validate dimensions; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-08. Conflicting SKU master
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: A SKU appears eligible under multiple pack sizes.
- Competing explanation: Legitimate package transition.
- Discriminating check: Check effective dates and source-system overlap.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Quarantine ambiguous master rows and request resolution; preserve the original output and decision audit trail.
- Prevention: Enforce one effective master per key and cutoff; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-09. Calendar misalignment
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Weekday forecasts lag the true day by one.
- Competing explanation: Unusual holiday behavior.
- Discriminating check: Compare forecast origin and business-calendar mapping around week boundaries.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Recompute calendar features and isolate affected runs; preserve the original output and decision audit trail.
- Prevention: Include holiday and calendar-transition fixtures; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-10. Closed store zeros
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Model learns systematic low demand for closed stores.
- Competing explanation: Genuine low-volume open days.
- Discriminating check: Join operating calendar known at cutoff and compare POS activity.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Separate closed-day from eligible open-day zero labels; preserve the original output and decision audit trail.
- Prevention: Track exclusion counts by store and calendar period; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-11. Sparse intermittent series
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Most predictions are nearly zero with occasional extreme misses.
- Competing explanation: Broken sales feed.
- Discriminating check: Check nonzero-sale intervals and feed completeness independently.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Compare intermittent baseline and global model on sparse cohort; preserve the original output and decision audit trail.
- Prevention: Segment evaluation by demand occurrence and size; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-12. New SKU cold start
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: No item-specific history is available.
- Competing explanation: SKU identifier mapping failure.
- Discriminating check: Check product launch date and parent-item mapping history.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Use category priors or planner decision with clear uncertainty; preserve the original output and decision audit trail.
- Prevention: Backtest pseudo-cold-start cohorts without future history; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-13. New store cold start
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: A new location inherits an unsuitable mature-store profile.
- Competing explanation: Store mapping bug.
- Discriminating check: Compare opening date, analogous-store selection and initial receipts.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Require local planner approval and conservative valid baseline; preserve the original output and decision audit trail.
- Prevention: Backtest historical openings and monitor adaptation; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-14. Demand shift
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Errors rise across stores after an assortment change.
- Competing explanation: Data pipeline break.
- Discriminating check: Compare upstream counts and residual shifts at unaffected control categories.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Escalate affected categories and shorten retrain interval only after diagnosis; preserve the original output and decision audit trail.
- Prevention: Monitor residuals and source-health signals separately; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-15. Promotion confounding
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Predicted promotion uplift looks too large.
- Competing explanation: Misconfigured promotion indicator.
- Discriminating check: Check plan timestamps and compare contemporaneous pricing and placement changes.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Avoid causal claims and constrain reliance until validated; preserve the original output and decision audit trail.
- Prevention: Use randomized or credible quasi-experimental evidence for causal uplift; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-16. Price endogeneity
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Model associates higher prices with high sales.
- Competing explanation: Incorrect currency conversion.
- Discriminating check: Audit effective prices and merchandising choices before interpreting coefficients.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Use forecasts only within observed support and review promotion decisions; preserve the original output and decision audit trail.
- Prevention: Keep predictive associations separate from pricing policy; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-17. Quantile crossing
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Predicted 90th percentile is below 50th percentile.
- Competing explanation: Wrong percentile labels.
- Discriminating check: Inspect raw model outputs and quantile column mapping.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Apply monotonic correction with calibration re-evaluation; preserve the original output and decision audit trail.
- Prevention: Test ordering and coverage on every release; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-18. Miscalibrated tails
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Observed mature coverage misses the intended quantile.
- Competing explanation: Temporary sampling noise.
- Discriminating check: Inspect cohort counts and time-block uncertainty as well as systematic bias.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Use baseline or planner review for affected high-risk cohorts; preserve the original output and decision audit trail.
- Prevention: Monitor conditional coverage after labels mature; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-19. Aggregated quantile misuse
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Summed daily 80th percentiles drive excessive or insufficient inventory.
- Competing explanation: Incorrect inventory position.
- Discriminating check: Compare simulated or direct cumulative-demand quantile with sum of marginals.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Replace aggregation with a joint or direct horizon forecast; preserve the original output and decision audit trail.
- Prevention: Test positive and negative cross-day dependence; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-20. Late supplier receipt
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Stockouts rise despite accurate sales forecasts.
- Competing explanation: Unexpected demand shock.
- Discriminating check: Compare promised and actual arrival timestamps and demand residuals.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Update expected receipts and escalate missing shipment; preserve the original output and decision audit trail.
- Prevention: Model lead-time uncertainty and track supplier reliability; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-21. Commitment double count
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Orders are consistently too high.
- Competing explanation: Legitimate backlog growth.
- Discriminating check: Reconcile whether reserved orders were already subtracted from on-hand.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Recompute position with mutually exclusive categories; preserve the original output and decision audit trail.
- Prevention: Version inventory-position semantics and accounting tests; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-22. Damaged inventory included
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Forecasted position appears healthy but shelves are empty.
- Competing explanation: Delayed shelf replenishment.
- Discriminating check: Compare saleable and damaged quantities and shelf availability.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Exclude unsaleable stock; request count if uncertain; preserve the original output and decision audit trail.
- Prevention: Reconcile status codes with physical inventory; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-23. Capacity master stale
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Orders exceed physical shelf or backroom capacity.
- Competing explanation: Planner override after validation.
- Discriminating check: Compare current layout and capacity effective date to order timestamp.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Stop export and revalidate using confirmed cap; preserve the original output and decision audit trail.
- Prevention: Add capacity version and final pre-export check; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-24. Case pack changed
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Supplier rejects valid-looking orders.
- Competing explanation: Downstream API outage.
- Discriminating check: Compare supplier master effective pack size and order divisibility.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Hold affected SKUs and obtain updated master; preserve the original output and decision audit trail.
- Prevention: Test effective-date transitions at cutoff; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-25. Rounding creates high overage
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Case-rounded order overshoots target by nearly a full pack.
- Competing explanation: Forecast high bias.
- Discriminating check: Compare continuous target with expected cost across feasible discrete packs.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Review feasible neighboring packs under agreed cost model; preserve the original output and decision audit trail.
- Prevention: Evaluate pack-size segments and explicit discrete optimization; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-26. Shared capacity oversubscription
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Individually valid SKU orders exceed store receiving capacity.
- Competing explanation: Single-SKU master error.
- Discriminating check: Sum candidate order volumes by store delivery window.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Hold batch and allocate via approved shared constraint policy; preserve the original output and decision audit trail.
- Prevention: Include cross-SKU capacity as an explicit coupled decision; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-27. Duplicate order export
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Supplier reports duplicate purchase orders.
- Competing explanation: Two legitimate order revisions.
- Discriminating check: Compare idempotency key, acknowledgements, and revision history.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Stop retries and reconcile accepted orders before correction; preserve the original output and decision audit trail.
- Prevention: Test retry-after-timeout and conflicting revisions end to end; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-28. Lost acknowledgement
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Exporter times out after a supplier accepted an order.
- Competing explanation: Supplier never received the order.
- Discriminating check: Query authoritative acceptance state using the idempotency key.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Do not send a new key until state is reconciled; preserve the original output and decision audit trail.
- Prevention: Exercise ambiguous timeout and recovery runbooks; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-29. Late batch completion
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Recommendations miss the purchasing cutoff.
- Competing explanation: Upstream data delay.
- Discriminating check: Inspect job spans and source watermarks to localize critical path.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Use valid fallback before deadline or defer; preserve the original output and decision audit trail.
- Prevention: Set per-stage budgets and alert on deadline risk; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-30. Schema incompatibility
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Rollback fails to read a newer feature snapshot.
- Competing explanation: Corrupt model artifact.
- Discriminating check: Compare schema compatibility and artifact checksums.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Disable affected output and restore compatible version pair; preserve the original output and decision audit trail.
- Prevention: Canary migration and retain versioned rollback bundles; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-31. Planner override surge
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Many recommendations are replaced by manual quantities.
- Competing explanation: New local event absent from data.
- Discriminating check: Analyze reason codes, cohorts, and whether interface defaults changed.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Route impacted cohort to human review; preserve the original output and decision audit trail.
- Prevention: Review override outcomes without treating all overrides as correct labels; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-32. Metric denominator shift
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: WAPE improves while stockouts worsen.
- Competing explanation: Actually better forecasting.
- Discriminating check: Recompute on stable eligible cohorts and inspect missing/censored shares.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Block rollout pending complete denominator reconciliation; preserve the original output and decision audit trail.
- Prevention: Publish exclusions and business metrics with scorecards; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-33. Backtest leakage
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Offline score is implausibly strong.
- Competing explanation: Genuine strong seasonality.
- Discriminating check: Trace newest input availability and compare a strict as-of replay.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Invalidate contaminated folds and preserve untouched test; preserve the original output and decision audit trail.
- Prevention: Add as-of join assertions and train-only transforms; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-34. Overlapping evaluation windows
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Confidence interval appears very narrow.
- Competing explanation: Large truly independent sample.
- Discriminating check: Check repeated store and shared-date dependence among periods.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Recompute clustered or block-resampled uncertainty; preserve the original output and decision audit trail.
- Prevention: Predefine resampling unit and temporal blocks; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-35. Model training skew
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: A subset of forecasts changes after feature pipeline migration.
- Competing explanation: Genuine model improvement.
- Discriminating check: Compare feature values for identical keys and cutoff across pipelines.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Roll back paired model-feature version and hold affected output; preserve the original output and decision audit trail.
- Prevention: Run point-in-time parity fixtures before promotion; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-36. Excess fallback
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Most stores receive baseline while aggregate SLI appears healthy.
- Competing explanation: Actually successful model coverage.
- Discriminating check: Separate model, fallback, quarantine and missing populations.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Escalate input availability and report model coverage separately; preserve the original output and decision audit trail.
- Prevention: Set explicit model-coverage and freshness guardrails; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-37. Security scope error
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: A planner can see another region's recommendations.
- Competing explanation: Legitimate cross-region access.
- Discriminating check: Inspect authorized scope, policy decision logs, and identity mapping.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Revoke access and investigate exposure; preserve the original output and decision audit trail.
- Prevention: Test authorization boundaries and review entitlement changes; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-38. Store substitution
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: A stockout shifts demand to a neighboring SKU.
- Competing explanation: Independent demand changes.
- Discriminating check: Compare nearby substitute sales and availability over the same period.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Avoid attributing all shift to the forecasting model; preserve the original output and decision audit trail.
- Prevention: Design pilot randomization to reduce interference; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-39. Return misclassification
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Net sales become negative for some daily observations.
- Competing explanation: Real corrections and returns.
- Discriminating check: Inspect transaction types and posting dates separately.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Separate gross sales, returns and corrections before modeling demand; preserve the original output and decision audit trail.
- Prevention: Reconcile returns using explicit source semantics; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.
### F-40. Promotion plan missing
- Evidence status: hypothetical; this is a designed diagnostic, not an incident.
- Symptom: Promotion feature defaults to false during feed failure.
- Competing explanation: No promotion scheduled.
- Discriminating check: Distinguish missing plan partition from explicit no-promotion record.
- Interpretation: a mismatch supports investigation but does not by itself establish causation.
- Containment: Hold or flag affected forecasts instead of silently treating as no promotion; preserve the original output and decision audit trail.
- Prevention: Monitor plan completeness and use a tri-state feature; assign the accountable data or operations owner before release.
- Verification: replay an affected historical key and an unaffected control under the same cutoff.
- Exit condition: reconciled inputs, valid constraints, and an owner-approved recovery; otherwise keep manual review.
- Residual risk: a successful replay does not prove future feed completeness or business benefit.

## 13. Transferable Lesson
The evaluation-mismatch pattern separates forecast generation from the physical and economic action it supports.
The constructed example demonstrates constraint arithmetic, not empirical forecast superiority.
Use it when a forecast drives recurring decisions with measurable constraints, credible data availability, and an actionable comparator.
The mechanism is explicit: distribution and position determine a candidate quantity, then the feasible set and operational policy bound it.
Avoid transferring the result directly to perishables, coupled network optimization, highly unreliable suppliers, or unknown costs.
A portability test asks whether the new domain can reconstruct what was known at each historical decision and observe the relevant outcomes.
For power planning, substitute capacity and unserved-energy costs for shelf and shortage costs; do not assume the same lead time or demand distribution.
For products with severe substitution or uncertain arrivals, model the interacting process and validate it before applying a one-item policy.

## 14. Connections & Cross-References
### 14a. Related Methods and Cases
The [collection index](INDEX.md) lists planned comparisons including energy load and ride-hailing; neither is represented here as an authored comparison.
SRC-01 provides rolling-origin evaluation; its time-split principle does not solve stockout censoring or inventory optimization.
SRC-02 describes leakage prevention; historical publication and ingestion times still need domain-specific enforcement.
The local [forecasting topic index](../INDEX.md) identifies planned forecast methods and operating themes.
The [rolling-origin validation sheet](../../../02_data_science_method_playbooks/10_time_series_playbook/Cheat_Sheets/04_Rolling_Origin_Validation_And_Demand_Accuracy.md) develops the replay and metric calculations; it does not validate this proposed deployment.
### 14b. Open Questions
| Question | Why unresolved | Evidence needed | Owner |
|---|---|---|---|
| How much demand is hidden by stockouts? | Observed sales are censored | Availability instrumentation and validated demand estimation | Analytics lead |
| Is inventory position reliable at cutoff? | No reconciliation records exist | Timestamped counts and receipt audit | Store operations |
| Does the candidate reduce economic cost? | No replay or pilot exists | Locked replay and adequately powered controlled pilot | Replenishment lead |
| Are supplier lead times stable? | No shipment histories exist | Order-to-receipt distribution and disruption analysis | Supplier operations |
### 14c. Interview Pack
Each prompt is hypothetical; scores distinguish a complete answer from a plausible but unsafe shortcut.
#### Q-01 — Framing
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: What is the actual decision, and who owns it?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: Store-SKU order quantity at a daily cutoff, owned by the replenishment lead.
- Reasoning: Forecasts only matter through a constrained order and later service/cost outcomes.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: A planner may defer if inventory or supplier inputs are unsafe.
- Follow-up: What if the supplier imposes a minimum order?
- Follow-up answer: Add the minimum to the feasible set and consider zero or joint ordering.
- Red flag: Optimizing forecast error alone omits inventory costs and operations.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Sections 1–3; all scenario outcomes remain not measured.
#### Q-02 — Data leakage
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: A plan becomes effective tomorrow but was ingested after today’s cutoff. Can it be used in today’s replay?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: No; effective date alone is insufficient; it was not available at decision time.
- Reasoning: Both publication and ingestion must precede the origin, with the correct revision.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: Use a prior available version or an explicit unknown indicator.
- Follow-up: What if the plan was published before but ingested after cutoff?
- Follow-up answer: It remains unavailable to this actual pipeline; assess a redesigned pipeline separately.
- Red flag: Joining the latest corrected plan onto historical rows manufactures foresight.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Sections 5a and 5d; all scenario outcomes remain not measured.
#### Q-03 — Censoring
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: A store sells zero units on a day. Is demand zero?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: Not necessarily: closure, stockout, missing data, or true zero demand are distinct.
- Reasoning: Check store hours, source completeness, shelf availability, and receipts.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: Treat a verified stocked open day as a possible true zero.
- Follow-up: What if availability is not instrumented?
- Follow-up answer: Report the unknown; use cautious assumptions and sensitivity rather than pretending demand is observed.
- Red flag: Filling all missing sales with zero biases demand downward.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Sections 5a and 12; all scenario outcomes remain not measured.
#### Q-04 — Calculation
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: With H=20, R=8 and C=3 units, what is position?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: 25 units: 20+8-3.
- Reasoning: Incoming must arrive in time and commitments must not already be removed from on-hand.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: If the inventory feed is net of commitments, omit C to avoid double subtraction.
- Follow-up: What if incoming stock is late?
- Follow-up answer: Remove unreliable receipts from position or explicitly model arrival uncertainty.
- Red flag: Counting late receipts as immediately available hides shortage risk.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Section 6 Q-01; all scenario outcomes remain not measured.
#### Q-05 — Feasible quantity
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: With target 53, position 25, pack 6 and capacity 30, what is the policy order?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: 30 units, or five packs; four packs only reach 49 units of position.
- Reasoning: Need 28 units; ceiling to a six-unit multiple gives 30 and fits capacity.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: An expected-cost optimizer may choose a different feasible pack if overshoot is expensive.
- Follow-up: What if capacity is 24 units?
- Follow-up answer: Four packs, or 24 units; position 49 falls four units short of target.
- Red flag: Ordering 28 violates the pack constraint.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Section 6 Q-02; all scenario outcomes remain not measured.
#### Q-06 — Critical ratio
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: Underage costs 4 per unit and overage costs 1 per unit. What target quantile?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: The continuous ideal is 4/(4+1)=0.8, the 80th percentile.
- Reasoning: The expected marginal overage and avoided shortage costs balance at this quantile.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: Direct expected-cost evaluation is preferable for discrete packs or coupled constraints.
- Follow-up: What if both costs are one?
- Follow-up answer: The ideal quantile is 0.5; its numeric demand value is not provided.
- Red flag: Calling the ratio an achieved service rate ignores distribution and assumptions.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Section 6 Q-03; all scenario outcomes remain not measured.
#### Q-07 — Horizon uncertainty
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: Can daily 80th percentiles simply be summed to obtain the 80th percentile of total demand?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: Generally no; the total depends on cross-day dependence.
- Reasoning: Forecast the cumulative protection-period distribution directly or simulate coherent joint paths.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: Under special dependence assumptions a sum may coincide, but it is not a general rule.
- Follow-up: What if lead time is uncertain?
- Follow-up answer: Include the arrival-time distribution and its dependence with demand in the protection period.
- Red flag: Summing marginal quantiles without justification discards dependence.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Sections 6 and 12; all scenario outcomes remain not measured.
#### Q-08 — Baseline choice
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: Why compare to a seasonal-naive order rather than only a mean forecast?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: It reflects the current operational decision, including the same feasible order policy.
- Reasoning: A simple baseline reveals whether model complexity creates incremental decision value.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: Compare moving averages and intermittent baselines when they reflect real alternatives.
- Follow-up: What if the current rule uses planner overrides?
- Follow-up answer: Report both the unedited rule and actual current workflow where history permits.
- Red flag: Comparing candidate decisions to an unbounded baseline produces an unfair result.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Sections 5c and 9; all scenario outcomes remain not measured.
#### Q-09 — Evaluation
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: Why is random row-level splitting inappropriate?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: It leaks future patterns and shares correlated store and temporal context across splits.
- Reasoning: Use rolling origins and a later untouched test, fitting transforms only inside training folds.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: Entity holdouts may supplement time holdouts to assess new-store generalization.
- Follow-up: What if outcomes overlap across origins?
- Follow-up answer: Use store/time blocks for uncertainty rather than treating rows as independent.
- Red flag: An unseen row is not necessarily an unseen future.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Sections 5 and 9; all scenario outcomes remain not measured.
#### Q-10 — Metric
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: Actuals are 10 and 20; forecasts are 12 and 17. Compute WAPE.
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: Absolute errors are 2 and 3, so WAPE is 5/30=16.67%.
- Reasoning: Sum absolute errors before dividing by total actuals; preserve unit and population.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: Report MAE alongside WAPE for sparse or zero-volume slices.
- Follow-up: What if all actuals are zero?
- Follow-up answer: WAPE is undefined; report absolute error and count without dividing by zero.
- Red flag: Averaging per-row percentage errors is not WAPE.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Section 9; all scenario outcomes remain not measured.
#### Q-11 — Sizing
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: What is the raw payload for two million series, 28 values each, four bytes per value?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: 224,000,000 bytes, approximately 213.6 MiB.
- Reasoning: Multiply series by horizon and bytes/value; convert using 1,048,576 bytes/MiB.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: Compression or sparse representations may change storage, not the raw arithmetic.
- Follow-up: What happens if the horizon doubles?
- Follow-up answer: The raw value payload doubles to 448,000,000 bytes, with metadata still excluded.
- Red flag: Calling this a measured memory footprint ignores indexes and working state.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Section 8d; all scenario outcomes remain not measured.
#### Q-12 — Reliability
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: The order exporter times out after sending. Is a retry safe?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: Only after checking downstream acceptance or using a shared idempotency contract.
- Reasoning: An ambiguous timeout can mean the order was accepted but its acknowledgement was lost.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: Reconciliation with a stable key and bounded retries is an acceptable design.
- Follow-up: What if the downstream lacks idempotency?
- Follow-up answer: Hold automatic retry and reconcile with the supplier or operator before resending.
- Red flag: Generating a new order key on retry can duplicate purchases.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Sections 8b and 12; all scenario outcomes remain not measured.
#### Q-13 — Ambiguous diagnosis
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: Orders spike after a batch. Is the model necessarily wrong?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: No; stale inventory, duplicated sales, late receipts, promotion changes, and true demand shifts compete.
- Reasoning: Compare input watermarks, transaction uniqueness, inventory reconciliation and residuals by cohort.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: Quarantine affected keys while investigating rather than globally reverting without evidence.
- Follow-up: What if only one item pack changed?
- Follow-up answer: Check master effective dates and unit conversion before retraining.
- Red flag: Immediately retraining can propagate a data defect.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Section 12; all scenario outcomes remain not measured.
#### Q-14 — Changed constraint
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: A store has a shared receiving cap across all SKUs. Is per-SKU clipping enough?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: No; individually feasible orders can exceed the combined capacity.
- Reasoning: Form a joint allocation over SKU pack multiples and total receiving capacity.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: A planner-approved allocation heuristic may be used if it respects all constraints.
- Follow-up: What if solver time expires at cutoff?
- Follow-up answer: Use an approved feasible fallback or defer; never export an infeasible partial solution.
- Red flag: Independent rounding is not proof of joint feasibility.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Sections 3 and 12; all scenario outcomes remain not measured.
#### Q-15 — Causal evidence
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: Can an offline backtest prove the intervention reduced stockouts?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: No; it estimates historical predictive or simulated decision performance under assumptions.
- Reasoning: Observed outcomes under the alternative inventory policy are counterfactual and often censored.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: A controlled pilot can estimate incremental impact if interference and measurement are addressed.
- Follow-up: What if stores share a warehouse?
- Follow-up answer: Cluster or otherwise design assignment to limit spillovers and explicitly analyze contamination.
- Red flag: A better backtest score is not a realized causal business gain.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Sections 9 and 11; all scenario outcomes remain not measured.
#### Q-16 — Governance
- Level: practitioner; assess both decision reasoning and operational consequences.
- Candidate prompt: What should happen when a recommendation is stale just before cutoff?
- Given facts: use only the assumptions and constructed quantities stated in this question or referenced case section.
- Clarify: identify the decision time, unit, availability, and relevant physical constraint before answering.
- Strong answer: Do not silently place the stale recommendation; use a freshness-qualified fallback or defer.
- Reasoning: The planner and on-call owner need the reason, deadline, and safe next action.
- Check: reconcile the conclusion with the stated grain, timing, units, and evidence status.
- Acceptable alternative: A recent baseline may be valid only if its inputs and constraints remain valid.
- Follow-up: What if no safe fallback exists?
- Follow-up answer: Escalate and hold the order for the approved manual process.
- Red flag: Assuming stale forecasts are harmless ignores physical inventory and supplier constraints.
- Scoring: 0 = incorrect or unsafe; 1 = partial without boundary; 2 = correct reasoning and boundary; 3 = also handles follow-up and tradeoff.
- Answer support: Sections 8 and 10; all scenario outcomes remain not measured.

## 15. Quick Reference Card
| Field | Case-specific answer |
|---|---|
| Problem | Choose a feasible daily store-SKU replenishment quantity from point-in-time information. |
| Context | Fictional retail CPG; two million possible series and all values illustrative. |
| Method | Compare a global probabilistic forecast and a seasonal-naive baseline under the same bounded policy. |
| Key finding | With target 53, position 25, six-unit packs and capacity 30, the illustrative policy orders 30 units; this is not an observed gain. |
| Action | No action taken; authorize replay and review before any pilot or automatic ordering. |
| Transferable lesson | Forecast quality is not the decision: inventory, economics, constraints and feedback determine value. |
| Evidence and stage | Illustrative computation; outcomes not measured; proposed, not deployed. |
| Review record | Draft, unreviewed; revisit when cutoff, inventory semantics, source snapshots or supplier lead times change. |
