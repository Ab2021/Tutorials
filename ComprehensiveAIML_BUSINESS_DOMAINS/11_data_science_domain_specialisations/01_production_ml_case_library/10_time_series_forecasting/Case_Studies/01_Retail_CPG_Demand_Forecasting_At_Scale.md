---
case_id: "CASE-01-10-001"
case_title: "Retail CPG demand forecasting for bounded replenishment"
case_version: "1.1"
created: "2026-11-07"
last_reviewed: "2026-11-10"
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Retail consumer packaged goods"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["data-leakage", "model-monitoring"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Retail replenishment lead"
source_index: "INDEX.md"
review_trigger: "Change to order timing, inventory semantics, or available historical snapshots"
---

# Retail CPG Demand Forecasting at Scale

Every store count, order quantity, cost, service level, and pipeline latency printed in this case is a constructed teaching input, labelled ILLUSTRATIVE where it drives arithmetic. No retailer, supplier, dataset, deployment, or measured result is described. The arithmetic is recomputed and self-checked (§6, §9), so the derivations can be reused with real inputs, but the numbers themselves must never be cited as an observation. Statements about how systems behave in general are marked as design reasoning. Where a fact would need a real contract, source system, or jurisdiction, the case says `unknown` instead of inventing one.

## 1. The One-Line Framing

A grocer must decide tonight, per store and per item, how many case packs to order so that the shelf survives until the following delivery, and the only defensible question is which order quantity minimises expected shortage plus holding cost under the demand uncertainty known at that hour.
The forecast is a means; the order is the decision; the cost is the objective. Optimising mean absolute percentage error is a proxy that can move in the opposite direction to the cost it is supposed to represent.
A distributional forecast plus a critical-ratio rule outperforms a point forecast plus a fixed safety allowance when the cost asymmetry is stable and the forecast's spread is honest.
The case is a cautionary design exercise: the same pipeline that produces a good backtest can produce a bad order, and the gap opens in five specific places — censored sales, promotion leakage, pack rounding, lead-time variance, and evaluation windows that are not decision windows.
Primary pattern is evaluation-mismatch, because the whole failure mode is a scoring choice that looks reasonable in a notebook and is wrong at the shelf.
Secondary patterns are data-leakage (vintage of promotion and price records) and model-monitoring (detecting when the decision, not the forecast, is degrading).
The intended reader is a forecasting practitioner who already knows how to fit a model and now has to defend an order quantity to a planner who carries the pager.

## 2. Context & Background

The fictional setting is a grocery chain with 1,000 stores and about 2,000 active ambient-grocery items per store, so roughly two million store-item series exist in the catalogue.
Not all of them are replenished daily: about 4% of series need a decision on any given night, which gives 80,000 orders per day and about 29.2 million decisions per year at that rate.
Series count matters for two reasons: the cost of a per-series artefact (storage, monitoring, explanation) multiplies, and any per-series false-alarm rate multiplies into an operational queue.
Deliveries arrive on a two-day cycle for most of the estate, so tonight's order must cover tomorrow and the day after, with the next order opportunity intervening.
Shelf space is finite. A pack that does not fit on the shelf must be stored in the backroom, which is where shrink, misplacement, and unrecorded stock live.
Sales are recorded at the register after the fact; demand exists before the sale does, and the difference is exactly the information the forecast needs.
When a shelf is empty the register records nothing, so the worst service failures remove their own evidence from the training data.
Merchandising publishes promotions with effective dates; the publication timestamp, the ingestion timestamp, and the effective date are three different clocks and only the first two constrain what a backtest may see.
Supplier lead time is contractual on paper and variable in practice. The case treats it as an assumption to test rather than a constant to trust.
Store operations physically counts stock on a schedule that is coarser than the order cycle, so the inventory position used at the cutoff is partly inferred.
Data engineering owns the feeds; ML operations owns model releases; merchandising owns the promotion plan; the replenishment lead owns the order and the override.
The status quo is a rule of thumb: order the previous matching weekday's sales plus a fixed safety allowance, then let planners edit obvious outliers.
That rule is not stupid. It is transparent, it degrades gracefully, and it encodes years of tacit knowledge about local conditions.
The proposed change is narrower than "replace the rule". It is: replace the rule with a forecast-assisted recommendation that a planner can accept, modify, or reject, and measure the combined system.
Scope exclusions are perishables, fresh produce, pharmacy, returns to vendor, inter-store transfers, price optimisation, assortment decisions, and supplier negotiation.
Also excluded: automated purchase orders. The scenario keeps a human on the release path, which is a policy constraint, not a maturity limit.
Legal, privacy, and employment considerations are jurisdiction-specific and unspecified here; the case flags the omission rather than asserting compliance.
The organisation's current measurement practice is weekly weighted absolute percentage error (WAPE) at the item-store-day level, reported by category, with no cost translation and no separation of stocked and unstocked days.
That practice is the seed of the evaluation-mismatch: it measures the forecast on a population, window, and unit that do not match the decision.

The operating rhythm deserves attention because it determines what "protection period" means. Most stores receive on a two-day cycle, some on a three-day cycle, and a minority on a daily cycle for the fastest lines.
If the cycle length differs by store, then a single model trained on a pooled horizon is averaging over three different decision problems with three different variance scalings. The variance scales with the square root of the period, so a three-day store has 1.22 times the spread of a two-day store at identical daily variability.
That is not a huge difference, and that is exactly why it is dangerous: it is small enough to survive a pooled training run and large enough to mis-set the safety stock in one of the two populations.
The second rhythm is the week: demand on Saturday is not demand on Tuesday, and the protection period starting on Friday covers a weekend while the one starting on Monday does not. Day-of-week composition inside the period changes the mean, not just the spread.
A model that predicts per-day demand and sums it can represent this; a model that predicts the period aggregate directly must be given day-of-week composition as a feature or it will learn a weekly average and apply it to every origin.
The third rhythm is the merchandising calendar: promotions cluster on month boundaries and paydays, so promoted and non-promoted periods are not randomly distributed across the test window. A four-week test that happens to miss a major promotion will flatter any model that does not handle promotions.
The fourth rhythm is financial: inventory is counted and valued at period end, so an ordering policy that over-orders in the last week of a period and under-orders in the first will look better on the period-end balance than it is on average. Any cost evaluation must span whole periods.
These four rhythms are the reason the evaluation window cannot be chosen for convenience. A window that is one week long can be dominated by a single promotion; a window that is one year long dilutes a genuine regression into a seasonal average.

## 3. The Business Question

The question the case answers: for each eligible store-item on each order date, which case-pack quantity should be proposed so that expected shortage cost plus expected holding cost over the protection period is minimised, and does a distributional forecast beat the current rule on that quantity?
The decision owner is the replenishment lead. The model developer supplies a distribution, a recommended quantity, and an explanation; the lead supplies the release decision and owns the consequence.
Available actions per series-night are: accept the recommended quantity, substitute a planner-chosen quantity, defer the order and accept a later delivery slot, or mark the series as not orderable tonight.
A deferral is a real action with a real cost, and it must appear in the measurement, otherwise the system is rewarded for declining hard cases.
The proposed objective is expected total cost per store-item-period, composed of shortage cost on unmet demand and holding cost on units carried beyond the period.
Shortage cost is a modelling choice, not an accounting fact: it bundles lost margin, substitution, basket abandonment, and goodwill into one number per unit.
Holding cost similarly bundles capital, storage, handling, and the risk of obsolescence into one number per unit per period.
Because both costs are approximations, the recommended order is only as good as the ratio between them; the case therefore reports the quantity as a function of that ratio rather than as a single number.
Hard constraints are non-negative integer case packs, a per-item shelf capacity, item-store eligibility, and the supplier's order cutoff time.
Diagnostics are WAPE, bias, prediction-interval coverage, fill rate, and the count of planner overrides. None of them is the objective; each explains part of the objective's movement.
The no-change option is the current rule with its override workflow intact, evaluated on the identical dates, series, and cost function.
A forecast that fails the guardrails does not authorise an unconstrained order: the system falls back to the rule and records the fallback reason.
The business question also has a negative form worth stating: if the forecast cannot beat the rule after costs are applied, the correct output is to keep the rule and stop spending on the model, and the evaluation must be capable of returning that answer.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type / status | Impact and fallback |
|---|---|---|---|
| A-01 | Two-day lead, daily ordering, one protection period per order | Business, illustrative | Late arrivals invalidate the period; planner review |
| A-02 | Inventory snapshots represent saleable units at cutoff | Data, unverified | Wrong position shifts the order; quarantine the series |
| A-03 | Promotion plan is published and ingested before cutoff | Data, unverified | Future-plan leakage; use last published version only |
| A-04 | Stockouts are partially observable via availability feed | Data, unverified | Sales understate demand; segment and flag, never patch silently |
| A-05 | Per-unit shortage and holding costs are stable within a category | Business, illustrative | Wrong quantile; recompute the critical ratio per category |
| A-06 | Demand over the protection period is approximately normal above a volume floor | Modelling, testable | Heavy tails and count data break it; use empirical quantiles |
| C-01 | Orders are whole case packs within shelf capacity | Physical, scenario rule | Rounding can cost more than the model saves |
| C-02 | Recommendation must be handed off before the supplier cutoff | Operational, scenario rule | Fallback to rule; measure timeliness end to end |
| C-03 | A named planner remains accountable for the release | Policy, scenario rule | No automation; logging must attribute the human decision |
| C-04 | Operational data access is scoped to named services and roles | Security, proposed | Unauthorised access disabled and investigated |

A-02 deserves emphasis: inventory position is on-hand plus reliable in-transit minus committed, and the in-transit term is the one that silently corrupts recommendations when a receipt is recorded late.
A-03 is the leakage surface. An approved promotion is usable only if its publication timestamp precedes the decision cutoff; approval at 23:40 for a cutoff of 22:00 cannot be backdated into a Monday backtest.
A-04 is the reason service cannot be measured from sales alone. The availability feed is itself noisy, and its noise is correlated with the very stockouts it reports.
A-05 fixes the objective's shape. The critical ratio depends only on the ratio of shortage to holding cost, so a 10% error in both costs is harmless while a 50% error in their ratio moves the order materially.
A-06 is a testable distributional assumption, not a convenience: its failure is demonstrated in §6 with a slow-moving item where the normal approximation produces a negative order.
C-01 has a consequence that is usually ignored: rounding a continuous optimum up to the next pack multiple is not free, and rounding up repeatedly across a large estate converts a small per-series bias into a large inventory bias.
C-02 and C-03 define the fallback and accountability structure. Both must appear in the evaluation, because a system evaluated without its fallback path is evaluated on a population that does not exist.
The three most consequential assumptions are inventory correctness, lead-time stability, and point-in-time publication of promotion and price records.
Each assumption carries a verification plan and a trigger. An assumption with no trigger is a belief, and beliefs do not belong in a decision system's specification.
A-05 is the one assumption that is routinely treated as a technical detail and is in fact a governance question. The ratio Cu/Co is set by finance and merchandising, and changing it changes every recommendation in the estate within one scoring cycle.
The practical implication is that the cost parameter needs an owner, a review cadence, and a version number, exactly like the model. A cost change that is not versioned makes historical recommendations unreproducible and turns every post-incident review into an argument.
There is also a structural problem with A-05: the true shortage cost is not constant across an item's life cycle. A discontinued item has near-zero goodwill cost and a high obsolescence cost; a flagship item has the reverse. A single per-category ratio is an average over both.
The mitigation is not a more elaborate cost model, which would be unfalsifiable. It is a category-level ratio plus an item-level flag for items in run-out or launch, with an explicit different rule for those flags.
Capacity, C-01, has a similar hidden structure. Shelf capacity in units is a derived quantity from facings, depth, and pack dimensions, and it changes when a planogram is reset. A stale capacity figure silently caps the order at a quantity that no longer fits.
The fallback when a capacity figure is missing is not to use a category default. It is to escalate the series, because a default cap is a fabricated constraint and it will bind on exactly the high-volume items where the error matters.
C-03's human accountability has an evaluation consequence that is easy to miss: the measured system is human-plus-model, so a model improvement that the planners ignore produces no measurable benefit, and a planner behaviour change produces a benefit with no model change.
Attribution therefore needs an override log with reason codes, and the analysis in §12 treats the override path as a separate treatment arm rather than as noise.
Security, C-04, is scoped to service identities and read/write separation. The scoring job reads operational data and writes recommendations; it does not write to the purchase-order system, which limits the blast radius of a model fault to a bad recommendation rather than a bad order.

## 5. Data, Tools & Method

### 5a. Data Specification

The decision grain is store identifier, item identifier, and local order date. Each row is one decision, and each decision yields one recommended quantity and one realised outcome.
The label grain is store-item-day units sold, aggregated to the protection-period horizon for the decision that consumed it.
Cutoff is 22:00 store-local time on the order date. Everything used by the feature set must be present with a publication timestamp at or before that instant, in the store's time zone, not the warehouse's.
Population: open stores, active items, items eligible for replenishment on the cycle, and series with at least 180 days of history. New items enter a separate cold-start path and are not scored by this model.
Keys: store identifier, item identifier, and order date form the natural key; a surrogate decision identifier maps to the recommendation and the human release for audit.
Units: demand and supply in single units; orders in case packs; costs in currency per unit; lead time in calendar days; capacity in units of shelf facings converted to units.
Not available at decision time: tonight's actual sales (the business day is still open at some stores), tomorrow's weather observation, competitor stock, supplier production status, and any promotion approved after cutoff.
Also not available: the true demand on a stocked-out day. That value is latent by construction and must be estimated or bounded, never read off a table.
The illustrative history is two years of daily rows with weekly rolling origins and a four-week untouched test period. No source data was obtained for this document.
Events are assigned to the local business day using the store's time zone in force on that date, because daylight-saving transitions otherwise create 23-hour and 25-hour days that distort daily aggregates.
Training labels use finalised sales after the returns and corrections window closes, which is typically several days after the sale. The model must know which label vintage it was trained on.
Exclusions computed per decision date: permanently closed stores, delisted items, items in seasonal range only, and stores under remodel. Exclusions must use information available at that date, not the eventual closure flag.
Zero sales on an open, stocked day are a real observation of zero. Zero sales on a stocked-out day, a closed store, or a missing feed are not, and conflating them is the single most common label corruption in retail forecasting.
Duplicate master records are quarantined, not joined: a duplicated item master multiplies transaction rows and inflates both the label and the feature.
Effective-dated joins are mandatory for price, promotion, pack size, and category. A current-record join leaks the future into the past and is invisible in offline metrics that use one snapshot for everything.
Holdout is strictly forward in time, with an embargo long enough that the training label at the boundary is fully mature before the test period begins.
Overlapping multi-horizon evaluation is permitted but the uncertainty calculation must account for the dependence it creates; treating overlapping folds as independent observations is a standard way to manufacture significance.
The following field contracts are illustrative, and each must be mapped to a real source owner, retention rule, and quality monitor before implementation.

#### D-01 Quantity sold
Meaning: units scanned at the register for the store-item-day, net of voids, before returns processing.
Availability: end of business day plus a settlement delay; not reliable at the 22:00 cutoff for the current day.
Grain: one row per store, item, and business date.
Failure boundary: a partial-day feed looks like demand collapse; the case treats the current day as unavailable rather than as a low value.
Owner: transaction data engineering, with a freshness monitor at the feed boundary.

#### D-02 Inventory position
Meaning: on-hand saleable units plus confirmed in-transit minus committed customer orders, at snapshot time.
Availability: nightly snapshot from the store system, sometimes stale by one cycle when a store sync fails.
Grain: one row per store and item per snapshot, with a distinct snapshot timestamp.
Failure boundary: a stale snapshot reports stock that has already sold; the recommendation then over-orders and the error compounds over the protection period.
Owner: store systems engineering; quality gate on snapshot age, not on row presence.

#### D-03 Receipt and shipment status
Meaning: supplier-confirmed dispatch and arrival events for the store, including quantity and expected arrival date.
Availability: updated when the supplier confirms, which may be after the order cutoff for the same cycle.
Grain: one row per purchase order line, with event time and recorded time.
Failure boundary: a recorded receipt that has not physically arrived creates phantom inventory; reconcile against counted stock before trusting the position.
Owner: supply chain integration; alert on confirmed versus arrived divergence.

#### D-04 Store calendar
Meaning: opening hours, closures, remodels, and local holidays per store and date.
Availability: published in advance, but amended at short notice for weather and staffing.
Grain: one row per store per date, effective-dated.
Failure boundary: an unlisted closure produces a zero-sales row that the label treats as demand destruction.
Owner: store operations; amendments must carry a timestamp so backtests can honour the version in force.

#### D-05 Promotion plan
Meaning: planned mechanic, discount depth, display type, and start and end dates per item and store cluster.
Availability: published by merchandising, ingested by the analytics platform; both timestamps are recorded.
Grain: one row per promotion version per item-store or cluster, effective-dated.
Failure boundary: a plan ingested after the cutoff but before the effective date leaks future intent; the backtest must use the version available at each origin.
Owner: merchandising systems; version history is a hard requirement, not a nice-to-have.

#### D-06 Price and cost
Meaning: shelf price, unit cost, and margin per item per store per date, with effective intervals.
Availability: price changes are approved in advance and applied at a scheduled time; cost updates arrive with supplier agreements.
Grain: one row per item-store-date interval.
Failure boundary: applying today's price to last year's sales rewrites history and corrupts the cost that the objective uses.
Owner: pricing and finance; interval integrity checked by overlap detection.

#### D-07 Item and pack master
Meaning: pack size, case multiple, unit dimensions, shelf capacity, and substitution group.
Availability: master data, changed by range reviews and supplier pack changes.
Grain: one row per item per effective interval, plus one capacity figure per store-item where measured.
Failure boundary: a pack-size change without an effective date makes historical orders incomparable in units; normalise or exclude the window.
Owner: master data management; the case treats pack size as immutable within an evaluation window.

#### D-08 Availability and stockout feed
Meaning: shelf-availability observations from store scans, order pickers, or planogram audits, with observation timestamps.
Availability: sparse and irregular; coverage varies by store and by day of week.
Grain: one row per observation, aggregated to a daily indicator with a coverage count.
Failure boundary: absence of an observation is not evidence of availability; treat unobserved days as unknown and report coverage alongside every service metric.
Owner: store operations analytics; coverage is reported, never imputed silently.

### 5b. Tools & Stack

Storage and compute are deliberately generic: a columnar warehouse for the decision-grade history, an orchestration layer that materialises point-in-time feature snapshots, and a batch scoring job that writes recommendations before the cutoff.
The forecasting component is a quantile or distributional model. The case does not depend on the family; it depends on the output being a predictive distribution over protection-period demand.
A hierarchical reconciliation layer is useful because store-item forecasts are noisy and category totals are comparatively stable, but reconciliation must preserve the item-level spread, not just the mean.
The optimiser is a small piece of arithmetic: invert the distribution at the critical ratio, round to pack, clamp to capacity, and check constraints. It does not need a solver.
The evaluation harness replays decisions rather than scoring forecasts, which means it needs the cost function, the eligibility mask, and the fallback path, all versioned alongside the model.
Experiment tracking records the feature snapshot identifier with each model artefact, so a score can be reproduced from a specific vintage rather than from "the data".
Monitoring needs three surfaces: forecast quality on matured labels, decision quality on realised cost, and process quality on timeliness, fallback rate, and override rate.
Alerting routes by owner: a feed delay goes to data engineering, a bias shift goes to the category analyst, a cost regression goes to the replenishment lead.
Access control is by service identity with least privilege; the recommendation job reads operational data and writes recommendations, and it does not write to the order system.
The order system integration is deliberately one-way with a human release step, so a model fault cannot create a purchase order without a person approving it.
The batch window deserves a budget rather than an aspiration. On the constructed pipeline, the snapshot stage takes 38 seconds of the 100-second budget, scoring 22, optimisation 6, constraint application 4, and handoff 18, leaving 12 seconds of headroom.
Snapshot is 38% of the budget and also the stage most likely to expand, because it fans out across sources. Any change to snapshot logic must be accompanied by a measurement of the cutoff tail, not of the mean.
The handoff stage is the second-largest and is the one most often left unmeasured, because it crosses an organisational boundary between analytics and the order system. An unmeasured handoff is where cutoffs are missed.
A practical guard is to instrument the cutoff as a deadline with a per-stage remaining-time metric, so the alert fires when the remaining budget falls below a threshold rather than when the process finishes late.
The stack also needs a replay harness that runs in minutes, not hours, because the most common production question is "what would the model have said on Tuesday given what we knew on Tuesday". A harness that takes a day to answer that question will not be used.

### 5c. Method Summary

Step one is to rebuild the decision population for history: which series were eligible, on which dates, with which cutoff, and which version of each input was visible.
Step two is to construct the label with the correct vintage and a censoring flag, so the model can be trained and evaluated with awareness of unobserved demand.
Step three is to fit a distributional model with the protection-period aggregate as the target, not the daily value, because the decision covers the aggregate and the aggregate's variance is not the daily variance divided by the horizon.
Step four is to convert the distribution into a quantity using the critical ratio implied by the cost inputs, then apply pack rounding, capacity, and eligibility constraints.
Step five is to replay the current rule and the proposed system over the same origins with the same cost function, and to compute the cost difference with a dependence-aware uncertainty interval.
Step six is a stress pass: re-run the replay with the forecast's spread understated, with the mean biased, and with lead time randomised, and record how the cost difference moves.
Step seven is a shadow period: produce recommendations without releasing them, compare against planner decisions, and investigate the largest divergences before any live release.
The method-selection ledger in §5e records what was considered and rejected, because "we chose a gradient-boosted quantile model" is not a justification without the alternatives it beat.
Every step writes an artefact with a hash, a timestamp, and an owner. The replay must be re-runnable from those artefacts alone.

The target choice deserves a paragraph of its own, because it is the most common modelling error in this domain after leakage.
Predicting daily demand and summing three daily forecasts gives an unbiased mean for the period but a variance that depends on the error correlation between days. If the three daily errors are independent, the period error variance is three times the daily error variance; if they are perfectly correlated, it is nine times.
Neither extreme is realistic. Serial correlation in daily errors is typically positive at short lags, so the truth sits between the two, and a model trained on daily targets with an independence assumption understates the period spread.
Predicting the period aggregate directly avoids the correlation question entirely, at the cost of fewer effective training rows: two years of daily history gives about 730 daily targets but only about 243 three-day aggregates, and the aggregates overlap if computed at every origin.
The overlap reduces effective sample size again, so the aggregate target needs a longer history or a hierarchical structure that borrows strength across items. That is the honest trade: fewer, better-matched targets against more, mismatched ones.
A workable compromise is to train on daily demand with a model that captures serial correlation (lags, day-of-week, promotion state) and then form the period distribution by simulation from the fitted residual structure, rather than by multiplying variances.
Simulation is also the cleanest route to non-normal features such as skew for promoted items, where the period demand distribution is right-skewed because a promotion either works or does not.

### 5d. Provenance, Quality, Exclusions, Leakage

Provenance for each input is a triple: source system, extraction time, and transformation version. A feature without a provenance triple cannot be replayed and therefore cannot be defended.
Quality checks run before scoring and are blocking: snapshot age, row counts by store, duplicate keys, null rates on required fields, and calendar coverage.
A blocked store falls back to the rule for that night; a blocked item falls back for that item. Fallback is a first-class outcome, not an error to be suppressed.
Exclusions are computed per decision date from effective-dated records, which prevents the classic error of excluding stores because they closed later.
Leakage detection is a standing test suite: a feature set is scored against a deliberately futured-shifted label, and any feature that predicts the future too well is flagged for review before it is trusted.
Leakage in this domain has three common shapes: current-record joins on price and promotion, aggregates that include the target day, and normalisation statistics fitted on the full history including the test period.
A fourth shape is subtle: using the supplier's final confirmed quantity as a feature when the confirmation happened after the cutoff. The value is real, but it was not knowable, and using it produces a backtest that cannot be reproduced live.
Censoring is handled by segmentation, not by imputation: stocked-out days are excluded from the demand label and counted in a separate coverage band, and the service metrics are reported for stocked and unstocked segments separately.
Exclusions must be reported as counts, because a model that is good on 60% of the population and silent on the other 40% is not a good model.
The evaluation population is frozen at the start of the test period, with a written list of eligible series and the reason for each exclusion, so that later re-runs cannot quietly change the denominator.
A fifth leakage shape is worth naming because it survives most review processes: aggregating a feature over a window that ends after the cutoff. A twelve-week average sales feature computed as of the label date, rather than as of the order date, contains three days of the very demand it is predicting.
The detection test for that shape is mechanical: for each feature, recompute it using only rows with timestamps strictly before the cutoff and compare. A feature that changes has a window that crosses the cutoff, and the size of the change is an estimate of the leak.
Quality gates are blocking at the store level and informative at the item level. A store missing its inventory snapshot cannot be safely scored; a single item missing a price record can be scored with the rule and labelled.
Null handling follows the same principle: a missing promotion record means "no planned promotion known", which is a valid state, while a missing price means "cannot compute cost", which is a blocking state. Treating both as zero is a common and silent error.
Censoring correction is bounded, not free. The availability feed observes an item on a subset of days, and its coverage is correlated with store size and staffing, so a correction built from it is more reliable in large stores than small ones. The evaluation reports correction coverage by store size band.

### 5e. Reproducibility & Method-Selection Ledger

| Option considered | Why rejected or chosen |
|---|---|
| Seasonal naive plus fixed allowance | Chosen as baseline: transparent, no distribution, no spread |
| Point forecast plus service-level multiplier | Rejected: needs a spread it does not have, and hides the cost ratio |
| Quantile regression at a single quantile | Considered: adequate if the cost ratio is fixed per category |
| Distributional model with full predictive spread | Chosen: supports cost-based order and spread diagnostics |
| Hierarchical reconciliation to category totals | Chosen with caveats: improves stability, must preserve item spread |
| Deep sequence model at store-item grain | Deferred: cost per series and explanation burden unclear at this scale |
| Manual planner heuristics as the only method | Rejected as sole method: not measurable against a cost function |

Reproducibility needs five artefacts: the population mask, the feature snapshot, the model artefact, the cost parameters, and the code revision. Any missing artefact voids the replay.
The ledger records a decision date and an owner for each choice, so a later reviewer can see whether the rejection reason still holds.
A rejected option is not permanent. The deep-model deferral carries an explicit revisit trigger: reconciliation gains exceed 5% of shortage cost in two consecutive categories.
Cost parameters are versioned separately from the model, because finance can change a cost input without retraining, and a changed cost changes the recommended order.
Every number in §11 is derived from these artefacts by the same code path, so the report cannot drift from the evaluation.

## 6. Analysis Walkthrough

The walkthrough starts from a single series so the arithmetic is checkable, then generalises to the estate, then breaks the assumptions deliberately.

### 6.1 Base case: one fast-moving item

The series is store 0412, item 33915, a fast-moving ambient item with a stable shelf. ILLUSTRATIVE inputs: mean demand 12 units per day, day-to-day standard deviation 6 units, protection period 3 days, case pack 6 units, shelf capacity 120 units.
Demand per day is treated as independent across days for this first pass, which is itself an assumption to test later.

```text
FORMULA
  m = mu_d * P                      protection-period mean demand, units
  s = sigma_d * sqrt(P)             protection-period standard deviation, units
WHERE
  mu_d    = 12 units/day   (mean daily demand, > 0)
  sigma_d = 6 units/day    (daily standard deviation, >= 0)
  P       = 3 days         (protection period, integer >= 1)
  independence of daily demand is assumed
CONSTRAINT
  s uses sqrt(P), not P; scaling the standard deviation linearly
  overstates the spread by a factor of sqrt(P) = 1.732 here.
NUMERICAL EXAMPLE
  m = 12 * 3         = 36.0000 units
  s = 6 * sqrt(3)    = 6 * 1.7320508 = 10.3923 units
  coefficient of variation s/m = 0.2887
INTERPRETATION
  The order must cover 36 units on average and roughly 10 units of
  one-sigma variation. Any rule that adds a fixed number of units
  is implicitly asserting a spread; that assertion should be explicit.
```

```text
FORMULA
  F(S*) = Cu / (Cu + Co)            newsvendor critical ratio
  S*    = m + z * s,  z = Phi^-1[ Cu / (Cu + Co) ]
WHERE
  Cu = 3.50 currency units per unit short   (lost margin plus goodwill)
  Co = 0.80 currency units per unit held    (capital, handling, obsolescence)
  Phi^-1 is the standard normal quantile function
CONSTRAINT
  0 < Cu and 0 <= Co; if Co = 0 the ratio is 1 and the quantile diverges,
  so the cost model requires a strictly positive holding cost.
NUMERICAL EXAMPLE
  ratio  = 3.50 / (3.50 + 0.80) = 3.50 / 4.30 = 0.813953
  z      = Phi^-1[0.813953]     = 0.892565
  safety = 0.892565 * 10.3923   = 9.2758 units
  S*     = 36.0000 + 9.2758     = 45.2758 units
INTERPRETATION
  The target is not a 95% service level; it is the point where the
  marginal cost of one more unit equals the marginal cost of one
  fewer unit. That point moves with the cost ratio, not with habit.
```

```text
FORMULA
  E[short]  = s * phi(z) - (S - m) * (1 - Phi(z))
  E[excess] = S - m + E[short]
  Cost      = Cu * E[short] + Co * E[excess]
WHERE
  z = (S - m) / s; phi and Phi are the standard normal density and CDF
  E[short] is expected unmet demand in units per period
CONSTRAINT
  The identity E[excess] - E[short] = S - m holds exactly for any
  distribution, which is a useful arithmetic self-check.
NUMERICAL EXAMPLE at the rounded order S = 48 (eight packs of six)
  z         = (48 - 36) / 10.3923 = 1.15470
  phi(z)    = 0.20482,  1 - Phi(z) = 0.12411
  E[short]  = 10.3923 * 0.20482 - 12 * 0.12411 = 2.1285 - 1.4893 = 0.6393
  E[excess] = 48 - 36 + 0.6393 = 12.6393
  Cost      = 3.50 * 0.6393 + 0.80 * 12.6393 = 2.2376 + 10.1114 = 12.3490
  Fill rate = 1 - 0.6393 / 36 = 0.98224
INTERPRETATION
  Rounding 45.2758 up to 48 buys a 1.2 percentage-point fill-rate gain
  and costs 0.3789 currency units per period against the continuous
  optimum of 11.9701. At 80,000 decisions per night that rounding is
  a real budget line, not a rounding detail.
```

```text
FORMULA
  Cost_rule = Cu * E[short | S = m + k] + Co * E[excess | S = m + k]
WHERE
  current rule: S = previous matching weekday sales + 3 units
  expressed as S = m + 3 with m = 36, so the implied safety is 3 units
CONSTRAINT
  A fixed allowance is a constant in units, so its implied service
  level falls as the series' variability rises; the rule is only
  defensible for series with similar spread.
NUMERICAL EXAMPLE
  S         = 39.0000
  z         = (39 - 36) / 10.3923 = 0.28868
  E[short]  = 10.3923 * 0.38271 - 3 * 0.38641 = 3.9773 - 1.1592 = 2.8181
  E[excess] = 3 + 2.8181 = 5.8181
  Cost      = 3.50 * 2.8175 + 0.80 * 5.8175 = 9.8612 + 4.6540 = 14.5152
  Fill rate = 1 - 2.8181 / 36 = 0.92172
INTERPRETATION
  The rule's implied fill rate is 92.2% against 98.2% for the rounded
  optimum, and its cost is 17.5% higher. Most of the gap is shortage
  cost, which is exactly the cost the WAPE report never shows.
```

### 6.2 General case: many series and a cost-sensitive ranking

The estate-level arithmetic is a sum of per-series costs, so the decision rule is unchanged; what changes is the ranking of where to invest effort.
A useful decomposition separates the cost gap into a mean-error component and a spread-error component, because they have different fixes: better features versus better distribution estimation.
If the forecast mean is unbiased and the spread is understated by a factor k, the recommended order is m + z·(k s) while the true distribution has spread s, so the realised z is z/k and service falls even though the model "looks calibrated" on its own terms.
The numeric size of that effect is in §6.4. It is large enough that a spread diagnostic belongs in the release gate, not in a dashboard.

### 6.3 Variants: pack rounding, capacity, and correlated errors

Pack rounding is a variant of the base problem: the feasible set is a lattice, so the optimum is the best lattice point, not the rounded continuous optimum in general.
Capacity is a second lattice constraint applied from above; when the unconstrained optimum exceeds capacity, the true decision is the capacity limit plus a separate escalation, not a silently clamped order.

```text
FORMULA
  S_k = ceil( (m + z*s) / k ) * k       smallest feasible order at pack k
  Delta = S_k - (m + z*s)                rounding increment, units
  Extra_cost = Cu*(E[short|S*] - E[short|S_k]) + Co*(E[excess|S_k] - E[excess|S*])
WHERE
  k = 6 units per pack; S* is the continuous optimum 45.2758
  the increment is always in [0, k), so the mean increment is about k/2
CONSTRAINT
  Rounding up is not always the cheaper lattice point: at S* near the
  lower edge of a pack interval, rounding up can be worse than rounding down.
NUMERICAL EXAMPLE
  Rounding up:   ceil(45.2758 / 6) = 8 packs = 48 units, Delta = 2.7242
  Rounding down: floor(45.2758 / 6) = 7 packs = 42 units
  Cost at 48 = 12.3490;  cost at 42 = 12.6189
  Rounding up wins here by 0.2699 per period, but the margin is small
  Mean increment over a uniformly positioned S* = 3.0 units = 8.3% of 36
INTERPRETATION
  The lattice forces a choice between two suboptimal points, and the
  better choice depends on where the continuous optimum falls inside
  the pack interval. Evaluating only the rounded-up point silently
  biases the order upward across the estate by about half a pack.
```

At 80,000 decisions per night, a systematic half-pack upward bias of 3 units on every series is 240,000 units of extra inventory per night in the constructed estate, which is why the rounding rule is a first-class decision and not a formatting step.

```text
FORMULA
  S_unconstrained = m + z*s
  S_feasible      = min( ceil(S_unconstrained / k)*k , C )
  Escalation flag = 1 if S_unconstrained > C
WHERE
  C = shelf capacity in units; a fast line with mu_d = 45, sigma_d = 9,
  P = 3, k = 6, C = 120 for this constructed store
CONSTRAINT
  When the optimum exceeds capacity the problem is infeasible, not
  constrained-optimal: clamping silently converts a capacity problem
  into an inventory under-protection that no ordering rule can fix.
NUMERICAL EXAMPLE
  m = 45*3 = 135.0000 units;  s = 9*sqrt(3) = 15.5885 units
  S_unconstrained = 135 + 0.892565*15.5885 = 135 + 13.9136 = 148.9136
  S_feasible = min(150, 120) = 120;  escalation flag = 1
  At C = 120: z = (120-135)/15.5885 = -0.96234
  E[short] = 16.3948 units;  fill = 1 - 16.3948/135 = 0.87856
  Shortfall against the unconstrained optimum = 148.9136 - 120 = 28.9136 units
INTERPRETATION
  Capacity removes 28.91 units of the intended safety, dropping fill to
  87.9% and raising cost to 57.3820 against 17.9551 unconstrained. The
  binding constraint must be reported as a capacity defect to the range
  team, because no forecasting improvement can recover it.
```

The capacity arithmetic is the clearest example in the case of a "model failure" that is not a model failure. A dashboard that reports only fill rate will show this series as a persistent underperformer, and the natural response — retrain, add features, raise the safety factor — is exactly wrong, because the recommended quantity is already at the physical ceiling.
Raising the ceiling does work, and the arithmetic quantifies how much: 132 units gives fill 0.9420, 138 gives 0.9642, and 144 gives 0.9798, so about three units of shelf space buys roughly two points of fill until the curve flattens near the unconstrained optimum.
That trade is a range and planogram decision with a currency price, and stating it in those terms converts a modelling complaint into a merchandising case.

```text
FORMULA
  P(covered) = Phi( z_stated / k )
WHERE
  z_stated is the z used to build the interval, k is the ratio of the
  true spread to the assumed spread (k > 1 means understated spread)
CONSTRAINT
  The identity holds for a symmetric interval on a normal target with
  correctly estimated mean; mean error reduces coverage asymmetrically.
NUMERICAL EXAMPLE
  z_stated = 1.6449 (nominal 95%)
  k = 0.8 (spread overstated): coverage = Phi(2.0561) = 0.98011
  k = 1.0:                      coverage = Phi(1.6449) = 0.95000
  k = 1.2 (spread understated): coverage = Phi(1.3708) = 0.91477
INTERPRETATION
  A 20% spread error moves coverage by 3 points, which is small. The
  §6.4 case of a 2.24x lead-time error moves it far more, which is why
  lead-time variance, not parameter noise, is the dominant spread risk.
```

Correlated errors across series change aggregation, not the per-series rule. Twenty independent series with standard deviation 10.3923 each sum to a standard deviation of 46.4758, while perfectly correlated series sum to 207.846.
The practical consequence: category-level service is far more stable than item-level service, so a category dashboard can look healthy while a specific item is chronically empty.
Reconciliation exploits this, but it must be applied with care, because forcing item forecasts to add up to a stable total can flatten a genuine item-level signal.
The arithmetic of that hazard is worth stating. If the item variance is 108 and the category is the sum of 20 independent items with variance 2160, a naive reconciliation that trusts the category and shrinks items toward it assigns the item a weight of only 108/2268 = 0.0476.
At that weight the model's item-level signal is almost entirely discarded, so a genuine item-level shift of 20% becomes a 1% adjustment. Reconciliation weights must be set from measured error covariances, not assumed.
The intermediate case is the one that matters in practice: items in the same category are positively correlated through shared promotions and shared weather, so the true category variance sits between 2160 and 43200, and the correct weight sits between 0.0476 and 1.

### 6.4 Boundary and counterexample

```text
FORMULA
  Var(D_L) = E[L] * sigma_d^2 + mu_d^2 * Var(L)
WHERE
  L is lead time in days, random; D_L is demand over the lead time
  mu_d = 12 units/day, sigma_d = 6 units/day
  Case A: E[L] = 2, Var(L) = 0 (deterministic)
  Case B: E[L] = 2, Var(L) = 2 (two-day mean, 1.414-day standard deviation)
CONSTRAINT
  The formula assumes L is independent of daily demand. Positive
  correlation (promotions delay deliveries) makes the true variance larger.
NUMERICAL EXAMPLE
  Case A: Var = 2 * 36 + 144 * 0 = 72.0000,  sd = 8.4853 units
  Case B: Var = 2 * 36 + 144 * 2 = 360.0000, sd = 18.9737 units
  Ratio of variances = 360 / 72 = 5.0000
  Ratio of standard deviations = sqrt(5) = 2.2361
INTERPRETATION
  Treating a two-day lead time as deterministic understates the
  protection-period spread by a factor of 2.24 in standard deviation.
  A model fitted on deterministic-lead history and deployed on
  variable-lead reality is systematically under-protected, and the
  shortfall appears first on exactly the days that matter.
```

The formula breaks in the slow-moving regime. For an item with mean 1.2 units per day, a normal approximation with standard deviation sqrt(1.2) = 1.0954 places 13.7% of its mass below zero, and the critical-ratio rule returns S* = 1.2 + 0.892565 × 1.0954 = 2.1778 units, which is not an order anyone can place.
The correct treatment for that item is a count distribution. With Poisson demand at mean 1.2, an order of 2 units covers demand with probability 0.8795 and yields a fill rate of 0.8843, while the same order quantity means something entirely different under the normal assumption.
The boundary is not a technicality: in a catalogue of two million series, the low-volume tail is where most of the series are, and where normal approximations and percentage-error metrics both fail.

```text
FORMULA
  WAPE = sum |y - yhat| / sum y
  E|X - mu| = sigma * sqrt(2/pi) = 0.797885 * sigma   for X ~ N(mu, sigma^2)
WHERE
  y is realised units, yhat is the forecast, sums run over a window
CONSTRAINT
  The second identity holds only for a normal variable with known mean.
  For a skewed or count distribution the constant is different, and
  for a forecast with bias the mapping shifts with the bias.
NUMERICAL EXAMPLE
  WAPE of 0.30 on a series with mean 36 implies, under the identity,
  sigma = 0.30 * 36 / 0.797885 = 10.8 / 0.797885 = 13.5358 units
  whereas the decision model needs sigma = 10.3923 units.
  WAPE of 0.20 implies sigma = 9.0239 units.
INTERPRETATION
  A WAPE target is not a spread input. Converting a WAPE figure
  into a safety stock requires the distribution and the bias, so a
  category WAPE target cannot be translated into a service promise
  without those two extra objects.
```

### 6.5 Leakage and censoring arithmetic

```text
FORMULA
  E[observed sales] = (1 - p) * mu + p * f * mu
WHERE
  p = fraction of days stocked out at some point, f = fraction of
  demand still captured during a partial stockout
CONSTRAINT
  Both p and f are unobserved in a pure sales table; they are
  estimated from the availability feed and are therefore uncertain.
NUMERICAL EXAMPLE
  p = 0.08, f = 0.60, mu = 12
  E[sales] = 0.92 * 12 + 0.08 * 0.60 * 12 = 11.04 + 0.576 = 11.6160
  understatement = 1 - 11.616 / 12 = 0.0320, i.e. 3.2%
  naive fill rate  = 11.616 / 12 = 0.9680
INTERPRETATION
  An item that is empty 8% of days still reports a 96.8% fill rate
  if the ratio is computed from sales against its own forecast. The
  metric is measuring the shelf, not the promise, and it is upward
  biased by exactly the censoring it fails to model.
```

```text
FORMULA
  inflation = 1 + 2 * sum_{l=1..h-1} (1 - l/h)
  n_eff     = n_cells / inflation
WHERE
  h is the forecast horizon in periods, n_cells counts every
  origin-horizon pair, and the correlation between overlapping
  errors decays linearly to zero at lag h
CONSTRAINT
  The linear-decay assumption is a simplification; a strongly
  persistent error process gives a larger inflation and a smaller
  effective sample, so this bound is generous.
NUMERICAL EXAMPLE
  h = 4, origins = 104 weekly, cells = 416
  sum   = 0.75 + 0.50 + 0.25 = 1.50
  inflation = 1 + 3.00 = 4.0000
  n_eff = 416 / 4 = 104.0000
INTERPRETATION
  The 416 fold scores behave like 104 independent observations.
  Confidence intervals built on 416 overstate precision by a
  factor of two in standard error, which is enough to promote a
  model whose improvement is noise.
```

The censoring correction also changes the comparison between the rule and the model, because the rule was tuned against the censored series and inherits its bias.
A model trained on observed sales learns to under-order on frequently empty items, then reports excellent accuracy on those same items, because it is predicting its own constraint.
The diagnostic that separates these cases is a plot of forecast error against stockout frequency; a downward trend in under-forecast as stockout frequency rises is the signature of label censoring.

### 6.6 Estate-level arithmetic

At 80,000 decisions per night, a per-decision cost difference of 2.17 currency units between the rule and the model is 173,600 currency units per night on this constructed cost model, before any implementation cost, and it is far too large to believe without checking the inputs.
That scale is a warning, not a forecast: the arithmetic assumes every series has the same cost parameters and the same demand variability, which is false, and it assumes the model achieves the theoretical optimum, which it will not.
The defensible use of the estate multiplication is as a sensitivity tool: recompute it under three cost ratios and record how the ranking of categories changes, then invest where the ranking is stable.
For the constructed estate, the ranking is stable under Cu between 2.00 and 5.00 but reverses around Cu = 1.00, where the critical ratio 0.5556 pushes the order below the mean and the rule happens to be closer to optimal.

### 6.7 How much evidence the pilot can produce

A pilot that cannot detect the effect it is looking for is a pilot that will end in an argument about noise, so the sample size is part of the design.

```text
FORMULA
  n = ( z_{1-alpha/2} + z_{power} )^2 * sigma_d^2 / delta^2
  MDE = ( z_{1-alpha/2} + z_{power} ) * sigma_d / sqrt(n)
WHERE
  delta  = true mean cost difference per decision, currency units
  sigma_d = standard deviation of the per-decision paired cost difference
  two-sided alpha 0.05, power 0.80; n counts decisions, not days
CONSTRAINT
  Paired differences must be computed per decision on identical rows,
  and the independence assumption is violated by shared promotions,
  so the true n is larger by the variance inflation of the block design.
NUMERICAL EXAMPLE
  Pilot: 200 stores x 300 eligible items x 4% decision rate x 7 days = 16,800 rows/week
  Four weeks gives n = 67,200 decisions.
  With sigma_d = 8.0 currency units per decision:
  MDE = (1.959964 + 0.841621) * 8.0 / sqrt(67,200) = 2.801585 * 8.0 / 259.23 = 0.0865
  For power 0.80 at delta = 2.17: n = (2.801585 * 8.0 / 2.17)^2 = 106.7 decisions
INTERPRETATION
  The pilot is overpowered for the main effect and underwhelming for
  subgroup claims: a 0.0865 per-decision MDE over four weeks does not
  license a per-category conclusion, because each category holds about
  a fiftieth of the rows and its MDE is larger by sqrt(50) = 7.07.
```

The asymmetry matters: the headline comparison is easy to power and the interesting comparisons (one category, one store band, one promotion type) are hard.
The honest response is to pre-register one primary comparison and a small number of secondary comparisons with a stated multiple-comparison adjustment, and to treat everything else as exploratory.
A second power issue is dilution. If 20% of decisions fall back to the rule, the measured effect is 80% of the model's effect, so a pilot sized for the full effect will under-detect by a fifth unless the fallback share is estimated first.
The pilot design should therefore report the fallback share as a primary descriptive statistic, and treat the model-only subset as a pre-specified secondary population rather than as the headline.
Finally, the paired design assumes the same decisions in both arms, which is true in replay but not in a live store-level holdout, where the two arms face different realised demand. Live holdouts need a cluster-randomised sample size, not a paired one.

### 6.8 Promotion mixtures break the single-normal shortcut

A promoted week is not a shifted normal; it is a mixture, because the promotion either lands or does not. Treating it as one normal with the averaged mean and the averaged variance is the most common way a well-built model loses money on promoted items.

```text
FORMULA
  D = (1-p) * N(mu_1, sigma^2) + p * N(mu_2, sigma^2)
  E[D]  = (1-p)*mu_1 + p*mu_2
  Var[D] = (1-p)*(sigma^2 + mu_1^2) + p*(sigma^2 + mu_2^2) - E[D]^2
  F(S)  = (1-p)*Phi((S-mu_1)/s_P) + p*Phi((S-mu_2)/s_P),  s_P = sigma*sqrt(P)
WHERE
  p = 0.30 probability a planned promotion actually executes
  mu_1 = 12 units/day (unpromoted daily mean), mu_2 = 26.4 = 2.2 * 12
  sigma = 6 units/day within each regime, P = 3 days
CONSTRAINT
  The variance identity is the law of total variance and must use the
  unconditional mean; using the average of the two variances alone
  omits the between-regime term p(1-p)(mu_2-mu_1)^2, which dominates.
NUMERICAL EXAMPLE
  E[D]   = 0.7*12 + 0.3*26.4 = 8.4 + 7.92 = 16.3200 units/day
  Var[D] = 0.7*(36 + 144) + 0.3*(36 + 696.96) - 266.3424
         = 126.0000 + 219.8880 - 266.3424 = 79.5456, sd = 8.9188
  Between-regime term alone p(1-p)(14.4)^2 = 0.21 * 207.36 = 43.5456, i.e. 55% of the variance
  Period: mean 48.9600, var 238.6368, sd 15.4479
  Normal shortcut: S = 48.96 + 0.892565*15.4479 = 62.7508
  True mixture quantile at 0.813953: S = 76.0248
  Shortfall at the shortcut: F(62.7508) = 0.7135, not 0.8140
INTERPRETATION
  The single normal matches the mean and the variance and still misses
  the quantile by 13.27 units, because the mixture is skewed: the
  promotion regime contributes a long right shoulder. Ordering at the
  shortcut costs 32.6262 against 29.2962 per period, a 3.33-unit gap
  that is invisible in any mean or variance comparison.
```

The skewness is 0.3535 for this mixture, which is small in absolute terms but large enough to move a high quantile by a fifth of a standard deviation. Skew matters at the quantile, not at the centre, which is why mean and variance diagnostics cannot catch it.
The practical rule is to compute the decision quantile from the mixture directly, or from the simulated distribution, whenever the promotion execution probability is below about 0.9. Above that threshold a single normal is a good approximation, because the between-regime term shrinks with p(1-p).
The failure mode has an uncomfortable property: it is worst for items where promotions are planned but unreliable, which are exactly the items a merchandising team cares most about, so the error concentrates where it is most visible.

## 7. The Finding

The finding is not "the model beats the rule". It is that the evaluation, as practised, cannot distinguish a good order from a good forecast, and that the two differ by roughly 15% of the controllable cost on the constructed example.
Concretely, three claims survive the arithmetic: a cost-based critical ratio moves the order materially away from a fixed allowance; the pack multiple absorbs a meaningful share of the theoretical gain; and a misestimated spread costs more than a modest mean error.
The third claim is the least intuitive and the most consequential. A mean error of 3 units (8.3% of the mean) costs less than a 50% understatement of the spread at the same order quantity, because the spread enters the tail while the mean shifts the whole distribution.
The finding also has a negative branch: at low cost ratios, or for very slow movers, the current rule is competitive, and the honest recommendation is to keep it and narrow the model's scope.
Reporting that branch is part of the deliverable. A comparison that can only conclude in favour of the new method is not a comparison.
The measurement implication is that the unit of evaluation must be the decision (store, item, date, quantity, cost), not the forecast row (store, item, date, error).
Once the unit is the decision, the population, window, comparator, and cost function all become explicit, and the evaluation stops being gameable by choosing a favourable slice.
There is a fourth claim that the arithmetic supports and that is rarely stated: most of the achievable gain comes from the first correction, not the last. Moving from a fixed allowance to a cost-based ratio captures the bulk of the difference; moving from a cost-based ratio to a fully distributional model captures a smaller increment that is harder to defend.
The reason is structural. The fixed allowance embeds an implicit service level that can be far from the cost-optimal one, so correcting it is a large move. Once the service level is approximately right, the remaining error is the shape of the distribution, which matters mainly in the tail.
This ordering has a practical consequence for sequencing: a team that cannot yet emit a distribution can still capture most of the benefit by estimating the mean and the spread separately from residual history, and should do that first rather than waiting for a full probabilistic model.
The second-order claim is about the spread. Among the residual errors, the lead-time variance component is larger than the parameter-estimation component, so the effort should go to measuring lead times and their dispersion before it goes to model refinement.
A simple test separates the two: recompute the recommended quantity with the spread doubled and with the mean shifted by one standard error, and compare the cost impact. In the constructed example the spread change moves fill by 5 points while the mean shift moves it by 1.5, so the ordering is unambiguous here.
The third-order claim concerns the interface. A recommendation presented without its cost curve invites the planner to override toward service, which systematically undoes the cost optimisation. Showing the plateau converts the conversation from "more stock" to "where on this curve do we want to be".
Finally, the finding includes a claim about the measurement regime itself: the existing weekly WAPE report, kept unchanged, will continue to show improvement while cost degrades, and will therefore keep the project alive past the point where it should be stopped or re-scoped.

## 8. System/Solution Design

The design has six stages: point-in-time snapshot, distributional forecast, decision optimisation, constraint application, human release, and measurement.

```text
Stage 1  Snapshot      materialise features as of the 22:00 cutoff, with
                       a recorded vintage identifier per source
Stage 2  Forecast      predict a distribution over protection-period
                       demand per eligible store-item
Stage 3  Decide        invert at the category critical ratio, producing
                       a continuous quantity and its cost curve
Stage 4  Constrain     round to pack, clamp to capacity, apply eligibility
Stage 5  Release       present to the planner with a reason code and a
                       one-click fallback to the rule
Stage 6  Measure       join to realised demand net of censoring, compute
                       cost, and attribute overrides
```

Stage 1 is where most production incidents originate, so it carries blocking quality gates and an explicit "unknown" state that propagates instead of being defaulted.
Stage 2 must emit a distribution, not a number. If the model cannot emit one, the pipeline can still run with an empirical quantile from the residual history, which is a legitimate fallback and must be labelled as such.
Stage 3 computes the recommended quantity and, in parallel, the cost curve around it, so the planner can see whether the recommendation is a sharp optimum or a flat plateau.
Stage 4 is arithmetic with hard rules. A recommendation that violates capacity is escalated with the binding constraint named; it is never silently clamped, because a clamp hides a data or demand problem.
Stage 5 is a workflow with attribution: accept, modify, or reject, each with a reason code, and the modification magnitude recorded. The recorded override is the most valuable training signal the system produces.
Stage 6 joins recommendations to outcomes after the label matures, computes realised cost for the system and for the counterfactual rule on the same rows, and separates the effect of the model from the effect of planner edits.
The system also needs a fallback path that is exercised in production, not just coded: a synthetic fallback drill every week keeps the rule path warm and measured.
Model artefacts are versioned with the snapshot identifier, so any recommendation can be recomputed exactly, which is the precondition for investigating an incident.
Alerts are tiered by blast radius: a single store's feed failure pages the store's support queue, a category-wide bias shift notifies the category analyst, and a system-wide timeliness breach pages the platform owner.
The interface shows a distribution band, the binding constraint, and the top three contributing features. Explanations that cannot be checked are worse than none, because they transfer false confidence.
Nothing in the design writes directly to the purchase-order system. That is a deliberate safety property, and it also means the measured effect is the effect of the human-plus-model system, which is the only effect that exists.

Three design alternatives were considered and rejected, and the reasons are instructive.
The first was a per-night automated release with a planner reviewing exceptions. Rejected because the exception rate cannot be bounded a priori, and an unbounded exception workload converts a control into a bottleneck that planners route around.
The second was a per-series model per store, which gives maximum flexibility and an unmaintainable artefact count: two million models with two million monitoring configurations is not a system, it is a database of incidents.
The third was a single pooled model with store and item embeddings. Chosen for the base design, because it shares strength across the long tail and produces one artefact to monitor, at the cost of weaker local adaptation.
The hybrid chosen in practice is a pooled model plus a small set of category-level corrections that are themselves learned, rather than a separate model per category. This keeps the artefact count low while allowing category cost ratios to enter through the decision step rather than through the forecast.
The decision step is where the category specificity belongs, because the cost ratio is a category property and the demand shape is largely a store-item property. Separating them keeps the model's job simple and makes the cost parameters auditable on their own.
A further design detail is the recommendation's reason code, which must be one of a small closed set: accepted by planner, modified upward, modified downward, rejected, fallback to rule, escalated for capacity, escalated for data quality. Closed sets are analysable; free text is not.
The reason code set also drives the monitoring: a rising "escalated for data quality" count is a data incident, while a rising "modified upward" count is a service-versus-cost disagreement between the planners and the finance-approved ratio, which is a governance conversation rather than an engineering one.
Both of those are more useful than a generic accuracy alert, which tells you something moved but not which conversation to have.

## 9. Evaluation & Validation

Numerators and denominators are defined per metric, because the same word "accuracy" can mean four different arithmetic objects in this domain.

| Metric | Numerator | Denominator / population | Window |
|---|---|---|---|
| Cost per decision | realised shortage plus holding cost | eligible decisions, stocked and unstocked | test period, 4 weeks |
| Fill rate | units demanded minus units short | units demanded, censoring-corrected | same rows |
| WAPE | absolute error sum | realised sales sum, stocked days only | same rows |
| Bias | sum of signed error | count of decisions | same rows |
| Coverage | realised demand at or below stated quantile | decisions with a distribution | same rows |

The population is the frozen eligibility mask from §5d. Decisions that fell back to the rule are included and labelled, because excluding them measures a system that was never deployed.
The comparator is the current rule evaluated on identical rows with identical labels and identical cost parameters, plus a "planner actual" comparator for retrospective context.
Guardrails are: no more than a 2 percentage-point fall in fill rate on any category, no increase in the override rate above 25%, no decision later than the cutoff, and no category with a bias exceeding 5% of mean demand.
Uncertainty is reported for the cost difference using a block bootstrap over order dates, which respects the dependence created by shared promotions and shared supply. Overlapping-horizon cells are never treated as independent.
A second uncertainty statement covers parameter risk: recompute the cost difference under Cu ± 30% and Co ± 30% and report the interval, because the cost inputs are estimates and the conclusion must not be an artefact of a point estimate.
Validation stages are: backtest on frozen history, shadow run against live decisions, then a limited live release with a holdout of stores. Each stage has an exit criterion and each stage's result is recorded.
The backtest exit criterion is a positive cost difference whose uncertainty interval excludes zero, plus all guardrails satisfied in two consecutive four-week windows.
The shadow exit criterion is agreement with planner decisions on at least 85% of rows, and an explanation for every systematic disagreement above 10% of quantity.
The live exit criterion is that realised cost on treated stores is not worse than control stores, with the comparison pre-registered before the release.
Status of each element in this design is `not executed`: no data has been obtained and no model has been trained. The evaluation specification is complete; its results are not.
Ablations are specified in advance: rule-only, forecast-mean-plus-fixed-allowance, forecast-mean-plus-cost-ratio, forecast-distribution-plus-cost-ratio, and the full system with planner overrides. Each answers a different question.
The ablation set is what makes a failure interpretable. If the third variant is good and the fourth is not, the problem is the distribution; if all forecast variants fail equally, the problem is upstream data.

Two further validation elements are often omitted and matter here.
The first is a stability check across time: the same comparison run on four consecutive four-week windows, reported separately rather than pooled. A method whose advantage is concentrated in one window is a method that has captured a specific event, not a general improvement.
The second is a placebo check: apply the decision rule to a randomised version of the forecast that preserves the mean but not the spread, and confirm it performs worse. If the randomised version performs equally well, then the spread is doing no work and the model's cost is unjustified.
The guardrails deserve their own arithmetic, because a guardrail that is not quantified is a sentiment.
A 2 percentage-point fill-rate fall on a category with mean demand 36 units per period corresponds to expected shortage rising from 1.0580 to 1.7780 units, which under the constructed costs is a shortage cost increase of 2.52 currency units per decision. That is a real budget line and should be stated as such rather than treated as a categorical prohibition.
The 25% override-rate guardrail is a process threshold, not an outcome one. Its purpose is to catch a usability regression early: when planners modify a quarter of recommendations, the system is functioning as a reference rather than as a decision aid, and the measured effect will be dominated by planner judgement.
The bias guardrail of 5% of mean demand is 1.8 units on the worked series, which corresponds to about 1.7% of the protection-period standard deviation. That is a tight band, deliberately: a bias of that size is detectable with four weeks of data at the estimated standard error, so the guardrail is enforceable rather than aspirational.
Detection power for the bias guardrail follows from the standard error: with 104 weekly observations and a period standard deviation of 10.3923, the standard error of the mean error is 1.019 units, so a 1.8-unit bias is about 1.8 standard errors and detectable at moderate power.

### 9a. What is executed and what is not

| Element | Status |
|---|---|
| Population definition and frozen mask | specified, not executed |
| Label construction with censoring flags | specified, not executed |
| Distributional model fitting | not executed |
| Cost-based decision step and pack lattice | arithmetic executed on the worked series only |
| Estate-level replay against the rule | not executed |
| Block-bootstrap cost interval | specified, not executed |
| Guardrail instrumentation | specified, not executed |
| Shadow run and live holdout | not executed |

The only executed element is the arithmetic in §6 on a single constructed series. Everything else is a specification, and the specification is the deliverable.
Stating that plainly is part of the case's evidence discipline: a reader can reuse the specification, but cannot cite this document as evidence that a retail forecasting system improved service or cost.

## 10. Action Taken

The action is a decision memo, not a deployment. It proposes a scoped pilot on two categories and 200 stores with a four-week shadow period before any live release.
The memo names the decision owner, the fallback owner, the data-quality gate owner, and the accountable executive for the cost parameters.
It also states what would cause the pilot to stop: a guardrail breach in two consecutive windows, a data-quality gate that cannot be met, or a cost difference indistinguishable from zero after four weeks.
Three changes to existing practice are proposed alongside the model. First, report service on stocked days separately from unstocked days. Second, version promotion and price records so backtests can use the right vintage. Third, record planner overrides with reason codes.
None of those three changes requires the model, and any one of them would improve the current rule's measurement, which is a useful way to argue for them.
The memo explicitly declines three tempting expansions: automated release, promotion-elasticity modelling, and inter-store transfer optimisation, each with a stated reason and a revisit trigger.
A data-quality remediation list accompanies the pilot: snapshot age monitors on three feeds, a duplicate-master quarantine process, and a calendar amendment log.
Training material for planners covers how to read the distribution band, what the binding constraint means, and when a fallback is the right answer.
The release plan includes a rollback that is a configuration switch, tested before the pilot, because a rollback that has never been executed is a hope.

The pilot's governance is deliberately boring, because novelty in governance is how pilots become permanent without a decision.
A weekly review reads five numbers: decisions produced, fallback rate, override rate with reason mix, realised cost difference against the rule, and the current cost-parameter version.
The review has one decision to make each week: continue, adjust, or stop. "Adjust" covers cost parameters, feature windows, and the pack rule, and each adjustment is versioned so the effect of the adjustment is measurable rather than confounded with the effect of the model.
A stop is recorded with its reason and does not require a second pilot to restart; the artefacts remain and the replay can be re-run when the blocking condition clears.
Stakeholder communication is a one-page summary per week with the same five numbers, plus the largest single decision error and its explanation. The largest-error item is included specifically because it is the one a planner will be asked about by a store manager.
The pilot also keeps a control group of stores on the rule for the full period, and the control group is chosen before the pilot from matched store pairs rather than after it from whatever happened to be available.
Matching uses store size band and historical fill rate, because those two variables explain most of the between-store variation in the outcome. Residual imbalance is reported rather than adjusted away.
The final pilot report is written to a template fixed at the start, which prevents the report from becoming a narrative constructed after the results were known. The template has sections for the pre-registered comparison, the secondary comparisons, the guardrails, the incidents, and the recommendation with its negative case.
No part of the pilot writes automated orders, changes the supplier contract, or alters the planner's accountability, which keeps the decision reversible at every point.

## 11. Outcome & Results

ILLUSTRATIVE constructed results for the worked series, produced by the arithmetic in §6 and not by any measurement:

| Quantity | Current rule | Rounded optimum | Continuous optimum |
|---|---|---|---|
| Order quantity, units | 39 | 48 | 45.2758 |
| Expected shortage, units | 2.8181 | 0.6393 | 1.0580 |
| Expected excess, units | 5.8181 | 12.6393 | 10.3338 |
| Cost per period | 14.5152 | 12.3490 | 11.9701 |
| Fill rate | 0.9217 | 0.9822 | 0.9706 |
| Stockout probability | 0.3864 | 0.1241 | 0.1860 |

The cost difference between the rule and the rounded optimum is 2.1662 currency units per period, which is 14.9% of the rule's cost.
The pack constraint consumes 0.3789 of that, or 17.5% of the total gain, which is the part that a continuous-optimum report would overstate.
Stockout probability falls from 38.6% to 12.4% while fill rate rises by 6 percentage points. The two service measures move together here but need not in general, which is why both are reported.
On the estate arithmetic of §6.6 the same per-decision difference implies a large nightly figure on the constructed cost model; that figure is deliberately not presented as a saving, because the uniformity assumptions do not hold and the model will not achieve the theoretical optimum.
Three sensitivities bound the claim. Understated spread by 50% at the continuous optimum reduces fill from 97.1% to 92.6% and roughly doubles expected shortage to 2.6506 units, so the spread estimate carries most of the risk.
A mean understatement of 3 units lowers fill to 95.2%; an overstatement of 3 units raises fill to 98.3% and raises excess cost, so mean bias trades service against inventory in a predictable direction.
At a cost ratio of 0.5556 the optimal packed quantity is 42 units and the rule's 39 costs 7.4715 against 8.0730, so the rule is actually cheaper by 0.6015 per period; the method's advantage inverts for categories where holding dominates shortage.
No operational result is claimed. There was no pilot, no deployment, and no measured change in service or cost.

The cost difference decomposes into three additive parts, which is useful because each part has a different owner and a different fix.

| Component | Size, currency units per period | Owner of the fix |
|---|---|---|
| Service-level correction, continuous optimum, no pack constraint | +2.5451 | category analyst with finance |
| Pack rounding penalty at pack 6 | −0.3789 | range and pack governance |
| Total, rule to rounded optimum | +2.1662 | shared |

The first component is 14.5152 − 11.9701 = 2.5451, the gain from moving the order from 39 units to the unconstrained cost optimum of 45.2758.
The second is 11.9701 − 12.3490 = −0.3789, the cost of being forced onto the pack lattice, which is negative because rounding can only move the order away from the continuous optimum.
An alternative decomposition attributes the whole 2.1662 to the ratio correction evaluated at the packed quantity and then reports the pack penalty separately as an unavoidable constant; the difference between the two presentations is only where the interaction is booked, and both must state which convention they use.
The pack penalty is listed as belonging to range and pack governance, not to the model, and that attribution matters: a team that owns only the model will be held responsible for a cost it cannot influence.
The ratio correction belongs to finance and the category analyst, and in a well-run organisation it is the cheapest available improvement, because it requires no model at all: applying the critical ratio to the existing rule's mean forecast captures most of it.
That is the single most actionable line in this case, and it is also the least attractive to a modelling team, which is a fair summary of why evaluation-mismatch persists in practice.
The tail of the distribution shows where the remaining risk sits. Under the rounded optimum, the probability of a stockout is 12.4%, so roughly one decision in eight still empties the shelf before the next delivery, and the expected shortage conditional on a stockout is 0.6393 / 0.1241 = 5.15 units.
That conditional figure is what a store manager experiences: eight times in ten nothing happens, and the ninth time the shelf is short by about five units. A recommendation presented only as an average understates the visible experience, which is one reason planners override toward more stock.

## 12. Failure Modes & What Went Wrong

The first mechanism is label censoring. Symptom: excellent accuracy concentrated on frequently empty items, with under-forecasting that grows with stockout frequency. Competing causes are a genuinely declining item and a stale price feed. Discriminating check: compare forecast error against the availability feed's stockout indicator; censoring produces a monotone relationship, decline does not. Correction: exclude stocked-out days from the label and evaluate service on a censoring-corrected denominator.

The second mechanism is promotion leakage through vintage. Symptom: backtest performance far above shadow performance, with the gap concentrated in promoted weeks. Competing causes are a distribution shift in promotion depth and a change in the availability feed. Discriminating check: rebuild the feature snapshot using only records published before each cutoff and re-score; a large drop identifies leakage. Correction: version promotion records and join on publication time, not effective date alone.

The third mechanism is pack-rounding drift. Symptom: inventory rises without a matching service gain, and the effect is proportional to the number of series. Competing causes are over-ordering upstream and a genuine demand shift. Discriminating check: compare the continuous recommended quantity with the released quantity and sum the rounding increments; a systematic positive increment identifies the drift. Correction: choose the best lattice point by cost, not the rounded continuous optimum, and report the rounding increment as a cost line.

The fourth mechanism is spread understatement, typically from fitting on deterministic-lead history. Symptom: service below target despite unbiased point forecasts. Competing causes are a cost parameter that is too low and a real demand shift. Discriminating check: compute the empirical coverage of the stated interval; coverage well below nominal at the same bias identifies spread, not mean. Correction: fit the spread on the lead-time distribution actually in force, and gate release on coverage.

The fifth mechanism is evaluation-window mismatch. Symptom: weekly item-day WAPE improves while cost does not, or vice versa. Competing causes are a genuine cost-parameter change. Discriminating check: recompute both metrics on the same decision rows and inspect the disagreement rows. Correction: evaluate on decisions and keep WAPE as a diagnostic only.

The sixth mechanism is override contamination. Symptom: the system's cost looks good while planners quietly fix the worst recommendations, and the model never learns. Competing causes are planners correcting genuine data errors. Discriminating check: stratify outcomes by override reason code; if the model's wins are concentrated in overridden rows, the credit belongs to the planner. Correction: attribute cost to model-only and human-edited paths separately.

The seventh mechanism is aggregation illusion. Symptom: category service is stable while individual items cycle through empty shelves. Competing causes are a genuine mix shift. Discriminating check: plot the distribution of item-level fill rates within the category; a bimodal shape with a low-mode cluster indicates the illusion. Correction: reconcile to the category total while preserving item-level spread, and report the item-level distribution.

The eighth mechanism is cutoff creep. Symptom: recommendations arrive late and fall back to the rule, and the fallback rate rises without a visible incident. Competing causes are genuine supplier changes. Discriminating check: measure each stage's latency against the cutoff budget and find the stage whose tail is growing. Correction: budget the pipeline explicitly and alarm on tail latency, not on mean.

A ninth mechanism is worth recording because it destroys trust faster than any of the others: explanation drift. Symptom: the feature attributions shown to planners stop matching the model's actual behaviour after a retrain, and planners begin to see explanations that contradict their local knowledge.
Competing causes are a genuine change in the drivers and a stale explanation artefact computed from the previous model version. The discriminating check is to recompute the attribution for a fixed historic recommendation using the current model and compare with what was displayed; a mismatch identifies the artefact, while a match identifies a real driver change.
The correction is to compute explanations at scoring time from the same artefact that produced the recommendation, and to include the model version in the explanation payload shown to the planner.

A tenth mechanism is the fallback path that was never exercised. Symptom: the fallback rate spikes during an incident and the fallback itself fails, because the rule depends on a table that the incident has also invalidated or on a code path that has not run since the last release.
Competing causes are a genuine upstream outage and a configuration error in the fallback. The discriminating check is a scheduled drill: force the scoring path to fail for a single store and verify that a valid rule-based quantity is produced and labelled within the cutoff.
The correction is the drill itself plus a fallback that depends on a small, stable table rather than on the same feature store the primary path uses.

The narrative's cost is worth stating in the same terms as the results, because a post-incident review that only lists mechanisms is not actionable.
The first backtest reported a cost improvement that was never measured, only implied by a WAPE improvement of about 4 percentage points. The shadow run produced a cost difference that was real but small, and the reason it was small was the spread understatement of the second mechanism combined with the rounding drift of the third.
Neither of those two mechanisms was visible in the weekly report, because the report had no spread diagnostic and no rounding line. The fix was therefore not a model change but three reporting changes, which is a common shape for this class of failure.

What went wrong in the constructed narrative is a composite of the first, second, and fifth mechanisms: the first backtest looked excellent, the shadow run looked mediocre, and the weekly WAPE report showed improvement throughout, because none of the three measured the same object.

## 13. Transferable Lesson

The transferable lesson is that a forecasting project fails at the seam between the metric and the decision, and the seam is invisible unless the evaluation is expressed in decision units.
Three artefacts make the seam visible and are cheap to build: a frozen decision population, a cost function with named parameter owners, and a fallback path that is measured rather than assumed.
The second lesson is about spread. A decision that consumes a distribution cannot be validated by a metric that scores a point, and a point model plus a fixed allowance is a distribution with an unexamined shape.
The third lesson is about constraints. Pack multiples, capacity, and cutoffs are not implementation details; they change the optimum, and they consume a measurable share of the improvement.
The fourth lesson is about honesty in the negative branch. At low cost ratios and low volumes the rule wins, and a project that cannot say so will over-deploy and then quietly lose credibility.
The fifth lesson generalises beyond retail: any system whose output is a quantity rather than a label needs its evaluation defined on the quantity, with the constraint set applied before scoring.
A sixth lesson concerns what to build first. The cheapest defensible improvement in this case is a cost-parameter review and a two-line decision step, neither of which requires a model; the model then earns its place by improving the mean and the spread, not by introducing the cost logic.
A seventh lesson is about the vocabulary of service. Fill rate, cycle service level, stockout probability, and ready rate are four different numbers calculated from the same decision, and in the worked example they diverge in a pattern that surprises most stakeholders: stockout probability at the rounded optimum is 12.4% while fill rate is 98.2%, because a stockout costs about five units against a period demand of 36.
When a stakeholder says "we need 98% service", the first response must be a question about which denominator they mean, because choosing the wrong one silently changes the target order quantity by several units.
An eighth lesson is that an override is data. The recorded modification, its direction, its size, and its reason code form a labelled dataset of the cases where the model and the expert disagree, and it is the highest-value training signal available in a retail forecasting system because it is generated continuously and for free.
Systems that discard overrides as noise lose that signal and then complain that planners are resistant to change. Systems that log overrides with codes can show, within a quarter, that most modifications cluster on three identifiable causes, which then become three fixable data problems.
A ninth lesson spans domains: the arithmetic of the decision is short and the engineering around it is long. The critical ratio, the lattice, and the identity linking excess to short are a dozen lines; the snapshot discipline, the vintage handling, and the fallback drill are the actual project.

## 14. Connections & Cross-References

Sibling case `./02_Ride_Hailing_Demand_Forecasting.md` applies the same point-in-time discipline to a supply-demand matching decision, where the cost asymmetry is driven by driver incentives rather than shelf space.
Sibling case `./03_Energy_Load_Forecasting.md` shares the critical-ratio idea but operates under a hard physical constraint that dominates the cost trade-off, which is a useful contrast in how constraints change the optimum.
Sibling case `./04_Financial_Volatility_Forecasting.md` treats spread estimation as the primary object rather than a nuisance parameter, and its loss functions are the right reference for anyone tempted to score a variance forecast with squared error.
The catalogue in `./INDEX.md` records evidence level and stage for every case and should be updated when this case's status changes.
The unit-level plan in `../INDEX.md` places this case first in the catalog order because it carries the pattern vocabulary the others reuse.
The cheat-sheet ladder in `../Cheat_Sheets/INDEX.md` is the intended home for the reusable derivations; until those cards exist, §6 of this case is the reference implementation.
The evaluation specification in §9 is the reusable artefact: copy it, replace the cost parameters, and it applies to any quantity-decision forecasting problem.
External reference points that a real implementation would need, and that this document does not supply, are the retailer's own pack and capacity masters, its cost accounting standard, and its supplier terms.
The pattern vocabulary in this case transfers to any recurring quantity decision: staffing rosters, energy procurement, spare-parts stocking, and cash positioning all reduce to a demand distribution, an asymmetric cost, and a lattice of feasible quantities.
The contrast with the volatility case is sharp and instructive. Here the spread is a nuisance that must be estimated well enough to set a quantile; there the spread is the target and the mean is nearly irrelevant, so the loss functions, the validation, and the skill of the practitioner all differ.
The contrast with the energy case is about hard constraints. There, the physical constraint can bind so tightly that the cost optimum is unreachable and the decision becomes a feasibility problem with a cost tiebreak; here the lattice is the only binding structure most of the time.

### Interview pack

**Q1 (recall).** What is the newsvendor critical ratio and which two quantities does it depend on?
- Answer: It is Cu divided by (Cu + Co), the ratio of underage cost to the sum of underage and overage cost, and it defines the service level at which the marginal cost of ordering one more unit equals the marginal cost of ordering one fewer.
- Reasoning: Setting the derivative of expected cost to zero yields the CDF value at the optimum, so the optimum depends only on that ratio, not on the demand distribution's shape.
- Follow-up: What happens to the optimum as Co approaches zero?
- Red flag: Answering "95% service level" as though it were a universal target.
- Scoring: Full credit for the ratio and the marginal-cost interpretation; half for the formula alone.

**Q2 (calculation).** With mean 36, standard deviation 10.3923, Cu 3.50 and Co 0.80, what continuous order quantity is optimal?
- Answer: 45.2758 units, from ratio 0.813953, z 0.892565, safety 9.2758 units.
- Reasoning: The quantile of the standard normal at 0.813953 is 0.892565, and the safety stock is that multiplier times the protection-period standard deviation.
- Follow-up: What does the same calculation give if Co is doubled?
- Red flag: Using a 95% quantile out of habit.
- Scoring: Full credit only with the intermediate z and safety values shown.

**Q3 (calculation).** The optimum is 45.2758 but packs come in sixes. What is the cost of ordering 48 instead of 54?
- Answer: 12.3490 versus 15.1555 per period, so 48 is better by 2.8065 currency units.
- Reasoning: At 48 the expected shortage is 0.6393 units; at 54 it is 0.1757 units, and the excess cost rises faster than the shortage cost falls beyond the optimum.
- Follow-up: At what order quantity does the cost double relative to the optimum?
- Red flag: Assuming that more service is always better.
- Scoring: Full credit for both costs and the direction of the comparison.

**Q4 (diagnosis).** Backtest WAPE is 0.22 but shadow-run cost is worse than the rule. Name the three most likely causes and one check for each.
- Answer: Promotion vintage leakage, label censoring, and a window or population mismatch; checks are a rebuilt snapshot with publication timestamps, an error-versus-stockout-frequency plot, and a recomputation on decision rows.
- Reasoning: All three make the offline metric optimistic without touching the model, which is why the model's own diagnostics look healthy.
- Follow-up: Which of the three can be detected without any new data collection?
- Red flag: Blaming distribution shift first.
- Scoring: Full credit for three distinct causes with a discriminating check each.

**Q5 (judgment).** The forecast is unbiased but intervals cover only 80% of outcomes at a nominal 95%. What do you do before releasing?
- Answer: Do not release; refit the spread on the lead-time distribution actually in force, then verify coverage by category, and only then evaluate the cost.
- Reasoning: The order quantity is a tail quantity, so an understated spread under-protects precisely on the days that produce the largest shortage cost.
- Follow-up: How would you distinguish a spread problem from a mean problem using only realised data?
- Red flag: Widening every interval by a constant factor without diagnosing the cause.
- Scoring: Full credit for blocking release and diagnosing before patching.

**Q6 (calculation).** Lead time has mean 2 days and variance 2. Daily mean is 12 and daily standard deviation is 6. What is the protection-period standard deviation?
- Answer: 18.9737 units, from variance 2 × 36 + 144 × 2 = 360.
- Reasoning: The law of total variance decomposes lead-time uncertainty into the demand variance accumulated over the mean lead time plus the mean-demand-squared term times the lead-time variance.
- Follow-up: What does the deterministic-lead assumption give, and what is the ratio?
- Red flag: Using 6 × sqrt(2) and ignoring the lead-time variance term.
- Scoring: Full credit for 18.9737 and the ratio 2.2361.

**Q7 (diagnosis).** Item fill rates are bimodal within a stable-looking category. What is happening and what is the fix?
- Answer: The category total is being stabilised by a few high-volume items while many low-volume items cycle through stockouts; the fix is item-level reporting plus reconciliation that preserves item spread.
- Reasoning: Aggregation hides variance when series are imperfectly correlated, and the correlation is highest within a category, which masks exactly the tail.
- Follow-up: What does the correlation assumption do to the aggregate standard deviation?
- Red flag: Concluding that the category is fine because the total is fine.
- Scoring: Full credit for the mechanism and a fix that preserves item-level spread.

**Q8 (selection).** Why prefer a distributional model over quantile regression at one quantile?
- Answer: A single quantile is sufficient only if the cost ratio is fixed per category and stable; a distribution supports cost changes without refitting and exposes spread diagnostics.
- Reasoning: The decision needs one quantile, but the review needs to know whether that quantile moved because the mean moved or because the spread moved.
- Follow-up: When is single-quantile regression the better engineering choice?
- Red flag: Claiming a distribution is always superior regardless of cost.
- Scoring: Full credit for the conditional answer and the diagnostic argument.

**Q9 (calculation).** What does a WAPE of 0.30 imply for the spread of a normal series with mean 36?
- Answer: 13.5358 units, using the mean absolute deviation identity with the 0.797885 constant.
- Reasoning: For a normal variable the mean absolute deviation is sigma times sqrt(2/pi), so inverting gives sigma from the percentage error.
- Follow-up: Why is this conversion invalid for a count distribution?
- Red flag: Treating WAPE as a coefficient of variation.
- Scoring: Full credit for the constant and the caveat.

**Q10 (judgment).** The critical ratio is 0.5556 for a category. Should you deploy the model there?
- Answer: Only if it beats the rule on realised cost; at that ratio the rule's order is close to optimal and the upside is small relative to the change cost.
- Reasoning: Low critical ratios put the optimum near or below the mean, where a simple rule already sits, so the model's advantage shrinks toward the pack-rounding noise.
- Follow-up: What would you measure to decide within one month?
- Red flag: Deploying everywhere because the method worked in the high-margin category.
- Scoring: Full credit for a conditional answer with a measurement plan.

**Q11 (diagnosis).** Override rate rises from 8% to 30% after a release. What are the competing explanations?
- Answer: A regression in recommendation quality, a change in what the interface displays, and a demand or supply regime change that the model has not seen.
- Reasoning: Overrides are a human judgement on model output, so they move with the model, the interface, and the world, in that order of likelihood for a recent release.
- Follow-up: Which single query separates the interface explanation from the others?
- Red flag: Treating overrides as noise or as planner resistance.
- Scoring: Full credit for three causes and a discriminating query.

**Q12 (calculation).** With 104 weekly origins and a four-week horizon, how many independent observations does the evaluation have, and why does it matter?
- Answer: 104, not 416, because the variance inflation for a four-period overlapping mean is 4.0.
- Reasoning: Overlapping errors share shocks, so the effective sample is the number of origins, and using 416 inflates every t-statistic by a factor of two.
- Follow-up: How would a block bootstrap handle this differently from a naive interval?
- Red flag: Dividing 416 by two and calling it conservative.
- Scoring: Full credit for 104 and the inflation factor.

**Q13 (judgment).** Finance revises Co upward by 30%. What changes in the system, and what does not?
- Answer: The critical ratio falls, so the recommended quantity falls; the model, features, and code do not change, but the cost parameters do, and the evaluation must be re-run.
- Reasoning: The order is a function of the ratio, so parameter changes propagate through the optimisation step rather than through retraining.
- Follow-up: What happens to the pack-rounding cost as the optimum moves toward a pack boundary?
- Red flag: Retraining the model to compensate for a parameter change.
- Scoring: Full credit for the separation of parameter from model.

**Q14 (diagnosis).** A slow mover with mean 1.2 units per day gets a normal-approximation recommendation of 2.18 units. What is wrong and what replaces it?
- Answer: The normal approximation assigns 13.7% of its mass below zero; replace it with a count distribution, under which an order of 2 covers demand with probability 0.8795 and yields a 0.8843 fill rate.
- Reasoning: The decision rule needs a valid distribution over non-negative integers, and the continuous approximation violates the support of the quantity being forecast.
- Follow-up: How do you detect this class of item automatically across two million series?
- Red flag: Rounding 2.18 to 3 and moving on.
- Scoring: Full credit for the support violation and the count-model replacement.

**Q15 (recall).** Name four leakage shapes specific to retail forecasting.
- Answer: Current-record joins on price and promotion, aggregates that include the target day, normalisation statistics fitted on the full history, and post-cutoff supplier confirmations used as features.
- Reasoning: Each makes a value available in the backtest that was not available at the decision instant, and each is invisible to a metric-only review.
- Follow-up: Which one survives a point-in-time snapshot rebuild and why?
- Red flag: Listing only "using future data" without a concrete mechanism.
- Scoring: Full credit for four mechanisms, half for fewer than three.

**Q16 (calculation).** A pilot covers 200 stores, 300 items, a 4% decision rate, and four weeks. What is the minimum detectable per-decision cost difference at 80% power if the paired difference has standard deviation 8 currency units?
- Answer: 0.0865 currency units per decision, from 67,200 decisions and the 2.801585 multiplier.
- Reasoning: The sample is the count of decisions, so the standard error of the mean difference is 8.0 / sqrt(67,200) = 0.0309, and the 80%-power multiplier is 2.80.
- Follow-up: What happens to that number for a single category holding a fiftieth of the rows?
- Red flag: Computing power from the number of days rather than the number of decisions.
- Scoring: Full credit for the decision count and the resulting MDE.

**Q17 (judgment).** Planners modify a quarter of recommendations, mostly upward. What is your diagnosis and your first action?
- Answer: A governance disagreement, not a model bug: the planners are pricing service above the finance-approved cost ratio; the first action is to reconcile the shortage cost with the planners and the store operations owner, and to log the reason codes that support it.
- Reasoning: Systematic upward modification concentrated on service means the decision's cost parameters disagree with the people who bear the consequence of an empty shelf.
- Follow-up: How would you distinguish that from a systematic under-forecast of mean demand?
- Red flag: Retraining the model to match planner behaviour without examining why.
- Scoring: Full credit for the governance diagnosis and a discriminating follow-up.

**Q18 (diagnosis).** A category shows a stable fill rate and a rising inventory value. What do you check first?
- Answer: The rounding increment and the split between released and recommended quantity, because a growing gap between the continuous optimum and the released order is the signature of pack drift rather than a demand change.
- Reasoning: Fill rate is a shortage-side metric and is nearly flat around the optimum, while inventory is a level metric that moves one-for-one with an upward rounding bias.
- Follow-up: What single number would confirm it across the estate?
- Red flag: Checking demand forecasts first when the shortage side is stable.
- Scoring: Full credit for the rounding hypothesis and a confirming statistic.

## 15. Quick Reference Card

| Item | Value or rule |
|---|---|
| Decision grain | store, item, local order date, one quantity per decision |
| Cutoff | 22:00 store-local; only pre-cutoff publications are usable |
| Objective | expected shortage cost plus expected holding cost per period |
| Order quantity | invert the demand distribution at Cu/(Cu+Co), then round on the lattice |
| Worked optimum | 45.2758 continuous, 48 at pack 6, cost 12.3490 versus 14.5152 for the rule |
| Spread rule | protection-period sigma uses sqrt(P) and adds lead-time variance |
| Slow-mover boundary | normal approximation fails below roughly one unit per day; use a count model |
| Evaluation unit | the decision, with the frozen population and the cost function versioned |
| Fallback | revert to the rule, label the row, and keep it in the denominator |
| Stop condition | two consecutive guardrail breaches, or a cost difference indistinguishable from zero |
