# Long-Tail Forecasting

> Module 21 — E-commerce Supply Chain Analytics  
> Topic 03 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 07, 10 and 12; Module 21 topics 01–02  
> Version 2026.5

## 1. Why this topic exists

An e-commerce assortment can contain many items that sell less than once per week.
A few popular items may account for most units, while a much larger set accounts for customer choice, replacement parts, niche interests, and search coverage.
The commercial value of that tail is not the same thing as the profitability of stocking every item in every fulfillment node.
A forecasting process must support both assortment access and the cost of providing it.

A replenishment planner asks whether to buy a unit for a particular item before the supplier lead time expires.
A network planner asks whether that unit should be in a nearby node or a shared central node.
A merchant asks whether a rarely purchased variant should remain available, be offered only by a supplier, or be retired.
A finance partner asks whether the expected contribution covers inventory, delivery, and obsolescence costs.
A customer-experience owner asks what promise can be made without creating cancellations and late deliveries.
Those are different decisions and can require different levels of aggregation.

A demand history with 27 zero days and three sales days looks precise in a daily table, but it carries little information about a specific item’s future rate.
A conventional model fitted independently to each series may learn noise.
It can also produce a zero forecast for an item that simply happened not to sell in the most recent few days.
A family, brand, category, or attribute pool contains more observations and can provide useful context.
Pooling reduces variance but may hide important differences between items.
The practical question is how much information to share, not whether every item is identical.

Observed orders are not automatically unconstrained demand.
A customer cannot place a normal order for an item that is unavailable, unlisted, unreachable to their location, or shown with an unacceptable delivery promise.
A low-sales history can therefore reflect weak demand, poor exposure, or both.
Removing an out-of-stock day from a rate calculation is a simple correction only when the remaining available days are representative.
It does not recover customers who disappeared after seeing repeated stockouts or changed their choice because another item was unavailable.

Long-tail decisions are also asymmetric.
Holding an unnecessary unit of an inexpensive, nonperishable cable has a different cost from holding a specialized battery with expiry and dangerous-goods restrictions.
Failing to stock an urgently needed repair part may be much more expensive than a missed discretionary accessory sale.
A mean forecast is useful, but it is not enough to choose stock without a lead-time distribution, an inventory state, and a service or cost objective.

This lesson builds an auditable path from sparse observations to a pooled forecast and then to a replenishment decision.
It treats forecasting as an input to a business decision, not a promise that an individual slow-moving item will sell on a particular day.
All numerical business examples are illustrative unless identified as public evidence.

### Decisions this topic enables

- **Buy or wait:** choose an order quantity at a review date, accounting for current and incoming inventory.
- **Where to stock:** compare a local node with a central pool and quantify the promise consequence.
- **How to forecast:** choose direct item, pooled hierarchy, or cold-start analogy based on evidence.
- **How to assess:** score forecasts at multiple levels and simulate the inventory policy that consumes them.
- **When to intervene:** route exceptions to a named owner with evidence, an expiry date, and an audit trail.

### Forecasts are not commitments

A **forecast** is an estimate of future demand under stated conditions.
A **plan** applies supply, capital, capacity, and policy constraints to that estimate.
A **target** is a desired commercial or operational outcome, such as a service goal.
A **commitment** is an accepted obligation, such as a customer delivery promise or a confirmed supplier order.
Do not replace a forecast with a sales target and then treat the target as an unbiased demand observation.
Do not treat a forecasted sale as a customer order or a supplier commitment.

## 2. Learning outcomes

After this lesson you can:

1. **Diagnose** intermittency using exposure, demand intervals, and nonzero demand sizes; see Exercise 1.
2. **Estimate** an exposed-day rate and explain why a stockout correction is conditional rather than a recovery of true demand; see Exercise 2.
3. **Compute** a Croston-style benchmark and identify its initialization and censoring limits; see Exercise 3.
4. **Allocate** a family forecast to items using smoothed shares and reconcile the resulting totals; see Exercise 4.
5. **Choose** an inventory order-up-to level from a lead-time demand distribution and calculate the net order; see Exercise 5.
6. **Design** a leakage-safe rolling-origin backtest that evaluates hierarchy, availability, and inventory cost; see Exercise 6.
7. **Critique** a placement and service proposal for sparse items with explicit decision ownership; see Exercise 7.
8. **Specify** a scalable production and monitoring design, including cold start and retirement; see Exercise 8.

Evidence of competence is not just naming an algorithm.
You should be able to define the target, identify the observation window, state what was knowable at the forecast origin, and explain the policy that consumes the output.
You should also be able to identify a question that the available observations cannot answer.

## 3. The core mental model

```text
Customer need and acceptable promise
                |
                v
  Search / listing / price / stock / delivery eligibility
                |
                v
  Exposed demand -------> unobserved lost or diverted demand
                |
                v
        Recorded orders and cancellations
                |
                v
   As-of data + eligibility + availability history
                |
                v
  Family signal + item evidence + comparable attributes
                |
                v
  Coherent probabilistic item x destination forecasts
                |
                v
  Stock policy + lead time + inventory position + constraints
                |
                v
  Buy / place / promise / do not stock, with named owners
                |
                v
  Service, cost, bias, availability and override feedback
```

The diagram separates **demand**, **observation**, **forecast**, and **decision**.
Demand exists in a commercial context: a price, listing position, promise, and assortment.
The recorded order is the visible part of that process, not a direct measurement of every customer who wanted an item.
Historical stock and promise therefore influence the data used to train the next forecast.
The next forecast influences stock and promise, closing a feedback loop.
If the process interprets every unavailable day as zero interest, it can repeatedly understock the same items.

A hierarchy gives information a useful structure.
A product hierarchy may connect item to product family to category to total assortment.
A geographic hierarchy may connect customer destination to delivery region to country.
A fulfillment network is not necessarily the same hierarchy as customer destination: a single node may serve several regions and a region may be served by several nodes.
Forecasting customer demand first and applying a routing or placement decision afterward avoids treating past warehouse assignments as fixed customer preference.

At the highest level, the family’s sales can contain enough events to estimate a seasonal pattern.
At the item level, recent sales and attributes determine how that family demand is shared.
A share model can shrink sparse item estimates toward a family prior.
The family forecast multiplied by shares produces item forecasts that add back to the family forecast.
Coherence is an accounting property, not proof that the family forecast or shares are accurate.

An inventory decision needs a distribution for demand during the period until the next effective replenishment.
That distribution may be simulated from occurrence and size, an empirical lead-time sample, or a defensible parametric model.
The required stock increases when demand uncertainty, lead time, desired service, or supply uncertainty increases, all else equal.
A point forecast alone does not identify the upper tail.

### Draw it from memory

1. Draw customer need before observed orders.
2. Put availability and promise between need and orders.
3. Put a family pool and item-level evidence next to the forecasting step.
4. Separate the demand forecast from the inventory policy.
5. Add a feedback loop from service and stockouts to the next data snapshot.
6. Label the owner who can act at each handoff.

### Four useful distinctions

| Distinction | Why it matters |
|---|---|
| Zero demand versus unobserved demand | A zero during known full exposure is informative; a stockout day is censored. |
| Demand location versus shipping node | A warehouse assignment is a result of a routing policy, not necessarily a customer preference. |
| Mean versus distribution | The expected number of units does not specify the probability of needing six or more units. |
| Coherent versus accurate | Forecasts can add up perfectly while every level is systematically biased. |

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Stock-keeping unit (SKU) | A sellable item variant identified within a merchant’s catalog. | Treating a product family or a changing listing ID as an immutable item. |
| Long tail | Many low-volume item series in an assortment; the cutoff is decision-specific. | Assuming every item in the tail has the same economics. |
| Demand event | An occurrence of positive requested quantity in a defined bucket. | Counting each unit as a separate arrival when a basket contains several units. |
| Intermittency | Positive demand is separated by one or more zero-demand buckets. | Treating every missing observation as a zero. |
| Lumpy demand | Infrequent events combined with variable positive event sizes. | Equating a high zero proportion with high event-size variability. |
| Bucket | A time interval such as a day or week, with a timezone and closing rule. | Comparing daily and weekly rates without conversion. |
| Exposure | Time and circumstances in which a customer can see and buy an item. | Assuming physical stock alone guarantees customer access. |
| Availability | Whether eligible, sellable inventory and a purchasable offer are present. | Using on-hand stock as a complete customer-facing availability measure. |
| Censoring | Observation is limited by availability or another constraint. | Calling a censored sales value a complete demand value. |
| Lost sale | Demand that would have converted under an alternative availability or promise, but did not. | Claiming an exact lost-sale count from stockout hours alone. |
| Substitution | A customer chooses another item when the preferred option is unavailable or unattractive. | Adding inferred lost sales without checking whether they already appear elsewhere. |
| Cannibalization | Sales shifted between items or offers rather than created incrementally. | Counting a launch’s gross sales as entirely incremental to the family. |
| Hierarchy | A structured mapping of items or regions to aggregate groups. | Assuming a many-to-many attribute graph is a strict tree. |
| Product family | A group of related SKUs whose behavior can share evidence. | Pooling unrelated products solely because they have low volume. |
| Share | An item’s fraction of a defined parent forecast, dimensionless. | Applying shares from a different time period, channel, or eligibility set without review. |
| Shrinkage | Blending noisy local evidence with a pooled prior. | Replacing every item with the same estimate regardless of evidence. |
| Prior | A pre-observation allocation or distribution informed by earlier data or assumptions. | Calling an arbitrary number an empirically validated prior. |
| Cold start | Forecasting an item with little or no usable own history. | Assigning a permanent zero because the item has no sales. |
| Analog | A comparable item or group used to inform a new or sparse item. | Picking an analog with later information unavailable at launch. |
| Croston method | A benchmark that separately smooths positive demand sizes and inter-demand intervals. | Treating its rate estimate as a calibrated demand distribution. |
| Syntetos–Boylan approximation (SBA) | A bias-adjusted Croston-style rate under its formulation. | Assuming the adjustment corrects censoring or every pattern of obsolescence. |
| Teunter–Syntetos–Babai (TSB) | A method that updates demand occurrence probability each period and size on positive periods. | Interpreting non-sale during unavailability as evidence that the item is obsolete. |
| Forecast origin | The date and time at which a forecast is generated. | Using catalog changes, sales, or stock states recorded only afterward. |
| Horizon | The future duration or sequence of periods being forecast. | Evaluating one-day accuracy when the decision needs 28-day lead-time demand. |
| Lead time | Time from order release to usable receipt; measured in the same time basis as demand. | Equating quoted supplier lead time with observed end-to-end receipt time. |
| Review period | Time between inventory decisions. | Ignoring the extra exposure until the next review. |
| Protection period | The period whose demand must be covered by a periodic-review decision, typically review period plus lead time. | Using lead time alone for a periodic-review order-up-to policy. |
| Inventory position | On-hand plus eligible on-order minus backorders, under a defined accounting convention. | Counting canceled purchase orders or double-counting reserved stock. |
| Order-up-to level | Target inventory position after ordering, in units. | Treating it as the order quantity. |
| Cycle service level | Probability of no stockout over a defined replenishment cycle. | Confusing it with the fraction of units immediately filled. |
| Fill rate | Fraction of requested units immediately filled under a specified measurement rule. | Calling the probability of no stockout a fill rate. |
| Reconciliation | Adjustment or construction that makes forecasts satisfy aggregation constraints. | Assuming adding up guarantees unbiased or well-calibrated forecasts. |
| Backtest | Historical simulation of forecasts generated using data available at each past origin. | Randomly splitting time-series rows and leaking future information. |
| Bias | Systematic signed forecast error under a stated sign convention. | Claiming low average bias means errors are small for individual items. |
| Obsolescence | Demand or sellability declines because an item becomes irrelevant or unusable. | Treating a stockout as proof of obsolescence. |
| Minimum order quantity (MOQ) | Supplier’s smallest permitted order, in units or packs. | Rounding demand up without pricing the resulting excess stock. |
| Contribution | Revenue less explicitly included variable costs, expressed per unit or order. | Calling contribution revenue, or excluding delivery and returns while claiming full profitability. |

## 5. The mechanics, step by step

### Step 1 — Define the business question and decision grain

**Input:** a decision calendar, customer promise, supplier terms, catalog, and cost assumptions.

**Action:** identify the decision that a forecast will support.
For purchasing, the relevant object is often future requested units by SKU over supplier lead time plus the review period.
For capacity, it may be order lines per fulfillment node per day.
For regional placement, it may be SKU demand by customer delivery region over a planning horizon.
For assortment, it may be the item’s incremental contribution and service role over several months.
These targets are related, but they are not interchangeable.

Write down the unit of measure.
An order can contain several lines; a line can contain several units; a unit can be sold in a multipack.
Supplier packs and customer-facing units must have a versioned conversion.
Forecast gross requested units when sizing availability; treat cancellations and returns separately unless the decision specifically needs net retained units.
Returns become potentially reusable inventory only after physical receipt, inspection, and disposition.

**Output:** a decision contract containing SKU definition, geographic grain, horizon, unit, refresh cadence, and owner.

**Branch:** if the purpose is a delivery promise rather than buying, include the customer’s destination and feasible service lanes.
If the purpose is supplier capacity, aggregate in the supplier’s agreed unit and calendar.
If the item has legal or safety constraints, first determine whether it is eligible to be stocked or shipped at all.

### Step 2 — Build an as-of catalog and hierarchy

**Input:** effective-dated catalog, product attributes, item identifiers, seller relationships, and destination mappings.

**Action:** build a stable item identity and a time-valid mapping to its parent family.
Record item launch, delist, variant changes, replacement relationships, and pack conversions.
Keep old mappings for historical evaluation rather than rewriting past families using the current catalog.
A product may move to a new family because of a taxonomy correction; the evaluation must use the mapping that was available at the origin, or explicitly restate all periods to a documented consistent hierarchy.

Distinguish identity from similarity.
A globally unique trade item number may identify a trade item when present, but a merchant SKU can be an internal identifier with different scope.
The same physical product may be offered by multiple sellers, and the same listing may have changed its pack or variant.
A clean crosswalk is a governed mapping, not a model feature to infer casually.

Use only one parent per item in a strict tree if simple add-up reconciliation is required.
If a product belongs to several attributes, such as brand, compatibility, and color, preserve them as features or represent the constraints with an explicit grouped aggregation matrix.
Do not sum overlapping attribute groups as if they were disjoint categories.

**Output:** one effective-dated item-to-family mapping per forecast series, plus documented alternative groupings.

**Branch:** when identity is ambiguous, quarantine or aggregate the series rather than inventing a history.
When a replacement item is launched, link its predecessor as an analog but do not silently join histories if price, compatibility, or offer terms changed.

### Step 3 — Establish the observation window

**Input:** order events, cancellation events, listing and inventory states, route eligibility, promise history, and historical timestamps.

**Action:** choose a bucket, time zone, cutoff, and event definition.
Build the item-by-destination-by-bucket spine only for intervals when the item could plausibly be offered.
Prelaunch time is not a sequence of zero-demand observations.
After a permanent delist, the old item’s active history has ended; do not use future empty rows to force the model to zero without recording the lifecycle event.

Deduplicate events using their business keys and event versions.
Choose whether the target is placed orders, accepted orders, or another demand proxy, and preserve other statuses for audit.
A canceled order may represent genuine intent, fraudulent activity, a payment failure, or an inability to fulfill.
Each cause should be mapped deliberately; there is no universal rule that all cancellations are demand or that none are demand.

For each bucket, record at least: observed units, active listing, sellable stock, purchasable channel, eligible destination, known price, shown promise, and observation completeness.
If event or inventory data are delayed, mark the bucket incomplete until the agreed watermark.
Missing inventory history is unknown, not confirmed available or unavailable.
A data-quality flag should travel with the forecast and decision.

**Output:** a versioned training panel with explicit states for eligible exposed zeros, positive observations, constrained periods, and unknown data.

**Branch:** if exact intra-day stock states are missing, use the finest reliable bucket and report a sensitivity range.
If both stock and listing exposure are unreliable, forecast observed orders as such and avoid claiming a latent-demand estimate.

### Step 4 — Quantify availability and censoring

**Input:** timestamped sellable availability and customer-facing offer eligibility.

**Action:** calculate available time in each bucket and distinguish it from in-stock units at the start or end of the day.
A daily snapshot may show stock at midnight but miss a stockout during peak demand.
Likewise, stock in a warehouse does not help a customer whose postcode is excluded or whose delivery promise suppresses the offer.
For a simple exposure calculation, define available fraction as eligible available hours divided by eligible hours.
Use a weighted exposure measure if demand intensity varies systematically by hour or weekday.

An exposed-day rate can be calculated as observed units divided by available days, then multiplied by the forecast horizon.
That estimate requires demand during observed exposure to represent demand during unobserved time after accounting for time and context.
A promotion day that sold out at noon is not exchangeable with an ordinary day that remained available.
Availability can also depend on demand: popular periods are precisely the periods most likely to stock out.
Report the estimate as an assumption-dependent counterfactual, not as a measured lost-sale count.

Clicks, searches, waitlists, rejected orders, and views can provide additional evidence, but each has its own selection bias.
Searches do not prove purchase intent, and impressions are influenced by rank and personalization.
When another SKU satisfies the need, adding a full lost-sale estimate to the original family total may double-count demand.
A controlled availability experiment, where ethical and commercially acceptable, can help identify incremental effects; otherwise provide scenarios.

**Output:** exposure coverage, a censoring flag, and an explicit method or uncertainty range for constrained periods.

**Branch:** for high-coverage series, use exposed observations directly.
For low coverage, borrow family or analog information and flag estimates for review.
For items with systematic early sellouts, a simple exposed-rate correction is insufficient on its own.

### Step 5 — Describe intermittency and event size

**Input:** eligible observed buckets and positive demand quantities.

**Action:** count eligible exposed buckets, positive-demand buckets, units, and the time between positive events.
A simple average demand interval (ADI) is exposed bucket count divided by positive-demand bucket count when each bucket is treated uniformly.
This is an approximate descriptive rate reciprocal, not a full survival estimate when observation begins or ends mid-interval.
For positive quantities, calculate the squared coefficient of variation, the variance divided by the squared mean, when enough events exist to estimate it.
State whether the variance is sample or population variance.

Compare a daily representation with a weekly representation.
Aggregation reduces the fraction of zero buckets but can erase day-of-week, promotion, and fulfillment-capacity detail.
A weekly mean cannot tell a warehouse when a peak day will arrive.
Choose the bucket according to the decision and the available signal rather than the visual preference of the analyst.

Separate structural zeros from observed zeros.
A listing that cannot serve a region is structurally ineligible there; it is not a low-demand signal for a purchasable offer.
Similarly, exclude or mark periods outside the assortment’s active lifecycle.
Analyze intermittent occurrence and the size of positive events separately because a single bulk order can dominate an average.

**Output:** a diagnostic table by item and family, including exposure, event count, ADI, event-size distribution, and lifecycle flags.

**Branch:** if there are no positive events, ADI based on division by event count is undefined.
Use an analog or parent prior, and preserve uncertainty instead of substituting zero or infinity as a valid demand estimate.
If one event supplies most of the units, investigate whether it was a repeatable customer order, a one-off business purchase, or a data defect.

### Step 6 — Establish baselines before complex models

**Input:** leakage-safe panel and rolling forecast origins.

**Action:** compare at least a zero forecast, an exposed historical rate, a pooled family-share forecast, and an intermittent-demand benchmark appropriate to the series.
The zero benchmark can have low absolute error for a very sparse item while delivering no stock; use it to reveal metric weaknesses, not as an automatically acceptable policy.
Compare forecasts on the decision horizon rather than only daily buckets.

For the original Croston approach, let `z` be the smoothed nonzero demand size and `p` the smoothed number of periods between positive events.
At a positive event of size `x` after interval `q`, update `z <- alpha*x + (1-alpha)*z` and `p <- alpha*q + (1-alpha)*p`.
The estimated rate is `z/p` units per bucket, held constant between positive events.
The choice of first interval and smoothing parameter must be documented and fitted without future data.
A common SBA version applies the factor `1-alpha/2` to that Croston rate under its standard smoothing convention.

TSB instead smooths an occurrence indicator each period and smooths positive size when demand occurs; the rate is estimated occurrence probability times estimated positive size.
That permits a declining forecast after a long run of truly observed zero demand.
It will also decay incorrectly if periods of unavailability are presented as observed zero demand.
Croston-family methods provide rate estimates; they do not by themselves supply a calibrated lead-time demand distribution or a guarantee of better inventory performance.

**Output:** reproducible baseline forecasts and their initialization, smoothing parameters, training window, and availability treatment.

**Branch:** for a new SKU with no sales, Croston has no own-event initialization; use a prior or analog.
For a declining lifecycle, compare occurrence-sensitive models with explicit lifecycle information.
For promotional, seasonal, or price-sensitive demand, test covariates and family-level seasonal structure rather than expecting a constant rate to capture the effect.

### Step 7 — Forecast the parent and form item shares

**Input:** family-level demand history, item evidence, lifecycle eligibility, and prior weights.

**Action:** create a family forecast for the needed horizon, using an aggregate baseline or a model whose added complexity is justified in backtests.
Build family demand from a documented observation definition; summing censored sales still produces a censored family series.
Model known future events only when their information was available at the origin.
A parent forecast may use calendar, price, seasonality, and promotion inputs, but planned promotions are not guaranteed execution.

Estimate an item share with shrinkage.
For item `i`, let `y_i` be its usable evidence units, `a_i` its nonnegative prior weight, and `Y` and `A` their sums across eligible items.
Set `share_i = (y_i + a_i) / (Y + A)` when the denominator is positive.
The units of `a_i` are pseudo-units of evidence, not physical sales or inventory.
The forecast is `family forecast * share_i`.
Because shares sum to one, item means sum to the parent mean, apart from numerical rounding.

Choose prior weights using information available at the origin: equal weights, attribute analogs, launch expectations, or a historical cohort.
A new item can receive positive prior share even with no sales.
A planned delist should normally have zero share after its effective end unless demand for a substitute or backorders is modeled separately.
Re-normalize over eligible items at each horizon when assortment membership changes.
A larger prior total shrinks individual shares more strongly; evaluate the strength in rolling historical launches and sparse cohorts.

Share estimation can itself be distorted by stockouts.
Using raw item sales will favor the better-stocked item even if family demand is estimated correctly.
Use comparable exposure, availability-adjusted evidence with explicit assumptions, or a model that jointly accounts for observation and demand.
Do not claim that a coherent allocation is an unbiased allocation.

**Output:** family mean forecast, item shares, parent-to-child reconciliation checks, and a report of prior and exposure coverage.

**Branch:** where the family has enough item evidence and heterogeneity, a direct item model may beat allocation.
Where one item represents most of the family, use a separate direct model plus a tail residual if this improves out-of-sample decisions.
Where all eligible items have zero prior and zero evidence, the share is undefined; choose and document a prior before allocating.

### Step 8 — Handle geography and network routing

**Input:** customer-destination demand, service lanes, network capacity, inventory availability, and cost-to-serve.

**Action:** forecast where customers request products before deciding where those requests will be fulfilled.
Use an origin-known geographic mapping and distinguish destinations with no exposure from destinations with zero observed demand.
A SKU-by-region allocation can borrow category or family geographic shares where item-level geography is sparse.
Consider rural and urban service constraints, carrier coverage, and delivery promises, rather than treating all destinations as equivalent.

Historical fulfillment-node sales are affected by inventory placement and routing rules.
If a node held no stock, it could not record fulfilled sales even if nearby customer demand existed.
Forecasting node sales from that history can perpetuate the old placement decision.
A routing model can transform demand forecasts into expected node workload under an explicitly chosen scenario.

**Output:** destination-level demand scenarios and node-level fulfillment implications under named routing policies.

**Branch:** if orders contain reliable destination and placed-order timestamps, retain demand at destination grain.
If only ship-node data exist, document that source demand and historical routing cannot be separated cleanly; use sensitivity analysis and seek better data.

### Step 9 — Construct uncertainty for the protection period

**Input:** event history, parent and item forecasts, lead-time history, review cadence, and decision cost.

**Action:** define the protection period before selecting a distribution.
For continuous review with immediate ordering at a known reorder point, the relevant demand window is often the replenishment lead time.
For periodic review, a standard order-up-to model protects through review period plus lead time, subject to its assumptions.
If lead time varies, simulate demand over sampled lead times or model the joint distribution where dependency matters.

Possible approaches include empirical rolling sums, resampling occurrence and positive size, a count model, or a hierarchical probabilistic model.
Use the distribution that is calibrated for the business horizon and situation; a normal approximation can be poor for a low, discrete count.
A Poisson example assumes independent arrivals with a constant rate and equidispersion, meaning variance equals mean.
Overdispersion, promotions, bulk baskets, and correlated item demand can invalidate this assumption.

A family mean split into item means does not automatically give each item’s variance or the dependence among siblings.
If total demand is fixed and allocated among items, item demands can be negatively dependent conditional on the total.
If a common promotion raises all items, demands can rise together.
For a central pool, preserve relevant dependence when simulating totals; adding item quantiles is generally not the same as taking the quantile of their sum.

**Output:** calibrated lead-time or protection-period demand scenarios, with assumptions and source data.

**Branch:** if usable history is too short for individual calibration, pool comparable items and widen uncertainty based on holdout performance.
If supply lead time has a long tail or a correlated disruption risk, consider supply scenarios and service risk rather than only an average lead time.

### Step 10 — Convert forecast into an inventory decision

**Input:** demand distribution, service or cost objective, inventory position, supplier constraints, and placement options.

**Action:** identify the service metric before computing stock.
A cycle service target is the probability of no stockout over the specified cycle, not the percentage of units filled.
For an illustrative order-up-to policy with no other constraints, choose the smallest integer `S` whose protection-period demand cumulative probability is at least the cycle service target.
Then calculate `order = max(0, S - inventory position)` before pack and capacity constraints.
Inventory position is on-hand plus eligible on-order minus backorders, with reservations and allocations handled consistently in the definition.

If an item has a minimum order quantity, pack multiple, supplier minimum spend, or limited capacity, the unconstrained order may be infeasible or uneconomic.
Evaluate the total cost and expected service of feasible quantities rather than simply rounding every item up.
Separate gross margin from contribution after pick, pack, delivery, returns, handling, and carrying costs.
For high-value or perishable items, include capital, expiry, and liquidation exposure.

The buyer approves a purchase against budget and supplier terms.
The inventory planner owns parameter selection and exceptions.
The network or placement owner decides where stock can be used.
The customer-promise owner decides which service is offered, informed by feasible inventory and transit.
Finance owns the agreed cost definitions and approves material trade-offs.
These owners may be different people even if one system produces all recommendations.

**Output:** an executable recommendation, assumptions, confidence or scenario range, and an approver.

**Branch:** for a dropship-only item, forecast supplier demand and monitor supplier availability and transit rather than creating owned stock.
For a one-off custom item, make-to-order may dominate replenishment.
For a critical spare, an explicit service requirement may justify stock despite poor standalone expected contribution.

### Step 11 — Backtest both forecasts and decisions

**Input:** historical origins, as-of snapshots, realized orders, actual availability, and policy assumptions.

**Action:** run a rolling-origin evaluation that rebuilds training data at each historical cutoff.
Use the forecast version and hierarchy known at that cutoff.
Do not train on future stock status, revised promotion outcomes, final returns, or later catalog corrections unless the same information was known at the time.
Use the same forecast horizon and refresh cadence as the operational decision.

Evaluate exposed, observable demand separately from constrained intervals.
A low error against stockout-censored sales may reward underforecasting.
Where true unmet demand is not observed, evaluate sensitivity to plausible counterfactuals or use valid experimental evidence; do not score an assumed latent quantity as ground truth.
Record the fraction of evaluation exposure that is actually observable.

Report signed bias, absolute error, and error by family, item cohort, geography, lifecycle, and horizon.
Weighted absolute percentage error (WAPE), `sum(abs(actual-forecast))/sum(actual)`, is undefined when aggregate actual demand is zero.
It can favor high-volume items and mask tail service.
Per-item mean absolute percentage error is unsuitable when many actuals are zero.
For probabilistic forecasts, use a proper distributional score or quantile loss and check empirical coverage at the relevant horizon.

Simulate a policy with the same lead times, receipt rules, MOQ, stock constraints, and decision timing across candidate forecasts.
Track immediate fill, cycle service, lost or backordered units under the selected assumption, holding cost, expiry, and order count.
A simulated service metric is conditional on the assumed demand and supply process.
Where demand is censored, simulation is not a substitute for observing the missing counterfactual.
Compare paired periods and cohorts, include uncertainty, and avoid declaring victory from a small number of tail events.

**Output:** a decision-oriented scorecard, comparison to simple baselines, and an explicit limitation for censored periods.

**Branch:** if accuracy improves but inventory cost or service worsens, investigate calibration and the mapping from forecast to policy.
If a model wins only in one event or one cohort, report the concentration rather than extrapolating to the whole assortment.

### Step 12 — Operate at portfolio scale

**Input:** versioned input snapshots, hierarchy, model configuration, run calendar, and owner directory.

**Action:** partition work by natural independent keys such as family, region, and forecast date.
Maintain compact sufficient statistics or event histories for sparse series instead of materializing every possible SKU-by-node-by-day combination.
Store nonzero demand events and state-change intervals while preserving the ability to reconstruct eligible zeros.
Sparse storage must not erase the distinction between zero, unavailable, prelaunch, and missing data.

Train or update shared models and priors at family or cohort level where justified, then score item shares in batches.
Avoid one heavyweight bespoke model process per item unless measured value exceeds operational cost.
Keep deterministic keys, input cutoffs, model versions, and output checksums or run identifiers for reproducibility.
Set service-level objectives for pipeline freshness, coverage, and completion before the buyer’s cutoff.

Rank exceptions by expected decision impact rather than percentage forecast error alone.
A one-unit deviation from a one-unit forecast is 100% error but may be less material than a small percentage miss on a high-contribution family.
An exception should include the evidence, proposed action, estimated cost or service impact, accountable owner, and expiry.
Log overrides separately from statistical forecasts so their incremental effect can be assessed.

**Output:** a scalable run with lineage, reconciled totals, coverage checks, and a bounded human review queue.

**Branch:** if a run misses its cutoff, use an explicitly approved fallback and mark its age; do not silently publish stale data as current.
If a hierarchy update changes many allocations, compare old and new mappings and get a catalog owner to sign off.

### Step 13 — Monitor and retire

**Input:** realized orders, exposure, stock and promise outcomes, model predictions, and override records.

**Action:** monitor forecast error only where the outcome is observable, plus exposure rate and the fraction of censored intervals.
Track family totals and item allocations separately.
Monitor stockout and excess inventory by commercially meaningful cohorts, including new items, expensive parts, perishables, and supplier groups.
Watch stale product mappings, missing inventory feeds, unexpected new IDs, zero-rate spikes, and data delay.

A drift alert is not itself a purchase instruction.
Investigate whether the change came from demand, price, search rank, stock, promise, promotion, catalog identity, or data latency.
Set separate thresholds for automated data-quality checks and business exceptions.
A low-demand item may be intentionally kept for compatibility, warranty, or customer trust; lifecycle decisions require the relevant business owner.

**Output:** a review record with cause, owner, action, and date for reassessment.

**Branch:** if a decline remains after exposure and listing checks, compare lifecycle and competitor changes.
If an item is superseded, migrate demand carefully to the replacement using a documented analog relationship, not an automatic one-to-one historical merge.

## 6. Worked example

### Case and decision

An illustrative online repair-parts merchant sells three compatible but distinct filter variants, A, B, and C, in one product family.
They are separate SKUs and cannot substitute perfectly for all customers.
The merchant reviews purchase orders continuously, and supplier lead time is assumed to be exactly 14 calendar days.
The supplier can deliver individual units with no MOQ or pack restriction.
The decision date is the end of the current observation window.

The family planner estimates **10 units per next 14 calendar days** under the planned price and standard delivery promise.
That is a **forecast**, not a sales target or supplier order.
The inventory team’s target for variant B is a **95% cycle service level** for the 14-day lead-time window.
The target is not a promised customer fill rate.
The buyer will later decide whether to issue a supplier purchase order, which becomes a commitment only when accepted under the applicable terms.

The example assumes an established mapping, no imminent delists, no promotion, and no supplier constraint.
The family forecast has already been assessed against an aggregate baseline.
We isolate the allocation and inventory calculation; the example does not prove that the aggregate forecast is accurate.

### Observed evidence

During a comparable historical window, the analyst finds the following available evidence.
The column marked evidence is used for allocation, not claimed to be complete unconstrained demand.
Each item gets one pseudo-unit of prior weight, representing an intentionally simple equal prior.

| SKU | Eligible observed evidence | Prior weight | Evidence plus prior |
|---|---:|---:|---:|
| A | 6 units | 1 pseudo-unit | 7 |
| B | 3 units | 1 pseudo-unit | 4 |
| C | 1 unit | 1 pseudo-unit | 2 |
| **Total** | **10 units** | **3 pseudo-units** | **13** |

The prior is illustrative.
In production it should be justified by comparable items and selected using historical backtests.
These observations can still be biased if exposure differs across SKUs.
The calculation assumes the evidence is comparable enough for this example.

### Compute item shares

The denominator is `10 units of evidence + 3 pseudo-units = 13 weighted units`.
The allocated shares are:

- A: `7 / 13 = 0.538462`, or approximately 53.85%.
- B: `4 / 13 = 0.307692`, or approximately 30.77%.
- C: `2 / 13 = 0.153846`, or approximately 15.38%.
- Total: `(7 + 4 + 2) / 13 = 1.000000`.

Without the prior, C would receive 10% of the family forecast.
With the equal prior it receives about 15.38%, reflecting limited confidence in just one observed unit.
Shrinkage does not establish that C’s true share is 15.38%.
The prior is a transparent modeling choice whose consequences can be tested.

### Reconcile the forecast

Multiply the 14-day family mean by each share:

| SKU | Calculation | Mean forecast, units per 14 days |
|---|---:|---:|
| A | `10 * 7 / 13` | 5.384615 |
| B | `10 * 4 / 13` | 3.076923 |
| C | `10 * 2 / 13` | 1.538462 |
| **Family** | `10 * 13 / 13` | **10.000000** |

Keep fractional means through forecasting and reconciliation.
Fractional means are not fractional orders: they represent expectations across possible futures.
Rounding every item forecast before summing can create an artificial mismatch with the family.
Order quantities are integer units and are determined after inventory policy and supplier constraints are applied.

### Show availability bias separately

Suppose another item D recorded six units over 30 calendar days, but its offer was buyable for only 21 equivalent days.
The raw calendar-day rate is `6 / 30 = 0.2000 units per day`.
The exposed-day rate is `6 / 21 = 0.2857 units per available day`.
Multiplying the latter by 30 equivalent fully exposed days gives `8.5714 units per 30 days`.
The difference from six observed units is approximately 2.57 units, but it is not a directly observed lost-sales count.

This extrapolation assumes the 21 available days represent the nine unavailable days after accounting for day mix, price, promise, and demand intensity.
If the unavailable days were a major promotion, the extrapolation may be too low.
If the available days were unusually high-intent days, it may be too high.
If customers purchased substitutes, some demand may already appear in other items’ sales.
A planner should show the exposure and scenario range, not present 2.57 units as an established fact.
D is a separate illustration and is not part of the A–C family allocation above.

### Choose B’s stock level

For demonstration only, suppose B’s demand in the 14-day lead time follows a Poisson distribution with mean `lambda = 40/13 = 3.076923 units`.
This assumes independent, stationary unit arrivals with variance equal to the mean.
No evidence in the example establishes that these assumptions hold; real positive order sizes and promotion effects should be checked.
Let `D` be lead-time demand and choose the smallest integer `S` with `P(D <= S) >= 0.95`.

The Poisson recurrence starts at `P(D=0) = exp(-lambda)` and uses `P(D=k) = P(D=k-1) * lambda / k`.
Summing the probabilities gives approximately:

| Candidate S | `P(D <= S)` | Meets 95% cycle service target? |
|---:|---:|---|
| 4 units | 0.802 | No |
| 5 units | 0.908 | No |
| 6 units | 0.962 | Yes |

Thus `S = 6 units` is the first candidate meeting the stated target under this model.
This is a probability of covering all demand in the modeled lead-time interval, not a 96.2% immediate fill-rate estimate.
The actual service outcome can be lower or higher if the distribution, availability, lead time, or order policy differs from the assumptions.

### Translate stock level into a purchase recommendation

Assume B has 1 sellable unit on hand, 2 confirmed usable units on order due within the modeled lead time, and no backorders.
Under the example’s inventory-position convention:

`inventory position = 1 + 2 - 0 = 3 units`.

`unconstrained order = max(0, 6 - 3) = 3 units`.

The planner recommends ordering 3 units of B.
The buyer checks that the incoming 2 units are genuinely usable and timely, the supplier can accept the order, and the planned cost remains acceptable.
If the supplier has an MOQ of 10 units, the unconstrained recommendation is not executable as written; compare the incremental carrying and obsolescence cost with alternative sourcing or service levels.
If the inventory position has already changed, recalculate at the decision timestamp.
A forecast does not reserve inventory and does not become a purchase order automatically.

### Compare a placement alternative

Suppose the merchant can stock B centrally or at two regional nodes.
A central pool may cover demand from both regions with less total safety stock when regional demand is not perfectly positively correlated.
However, longer transit may make the central offer miss a next-day promise in one region.
The network owner should compare feasible service, demand coverage, incremental shipping, and stock holding under both arrangements.
A central pool is not automatically cheaper after expedited shipping or lost conversion is included.

The family forecast remains 10 units per 14 days in this illustration; placement does not change it by assumption.
In reality, a slower promise may change conversion, so a counterfactual demand scenario should be considered rather than applying the same demand uncritically.
The merchant owns the offer and assortment trade-off; the placement owner owns the node proposal; finance reviews the contribution assumptions.

### Audit the example

- Decision: replenish B, measured in physical units.
- Forecast horizon: 14 calendar days.
- Parent mean: 10 units per 14 days.
- B mean: `40/13`, approximately 3.077 units per 14 days.
- Distribution assumption: Poisson, not estimated from the given sparse history.
- Service target: 95% cycle service over the modeled lead time.
- Order-up-to level: 6 units.
- Inventory position: 3 units.
- Recommended order before supplier constraints: 3 units.
- Missing evidence: observed lead-time distribution, event-size behavior, actual exposure quality, cost and value of service.
- Responsible approver: buyer, informed by inventory planner and supplier terms.

## 7. Code, query, or calculation

The following standalone Python 3 program reproduces the allocation, availability illustration, Poisson cumulative probabilities, and recommended order.
It uses only the standard library.
Save it as `long_tail_example.py` and run `python long_tail_example.py`.
The code is a teaching calculation, not a production inventory optimizer.
It assumes the input evidence is comparable, the supplier lead time is fixed at 14 days, and B’s lead-time demand follows the stated Poisson model.
The known on-order units are assumed usable under the simplified inventory-position policy.

```python
from math import exp


def smoothed_shares(evidence, prior):
    """Allocate a parent mean using nonnegative evidence and pseudo-units."""
    if set(evidence) != set(prior):
        raise ValueError("evidence and prior must have identical SKU keys")
    if any(value < 0 for value in evidence.values()):
        raise ValueError("evidence must be nonnegative")
    if any(value < 0 for value in prior.values()):
        raise ValueError("prior weights must be nonnegative")
    total = sum(evidence.values()) + sum(prior.values())
    if total <= 0:
        raise ValueError("allocation requires positive total evidence or prior")
    return {
        sku: (evidence[sku] + prior[sku]) / total
        for sku in evidence
    }


def poisson_cdf(k, rate):
    """P(D <= k) for a modest nonnegative Poisson rate."""
    if k < 0:
        return 0.0
    if rate < 0:
        raise ValueError("rate must be nonnegative")
    probability = exp(-rate)
    cumulative = probability
    for count in range(1, k + 1):
        probability *= rate / count
        cumulative += probability
    return cumulative


def poisson_order_up_to(rate, cycle_service):
    """Smallest integer stock level meeting the cycle-service target."""
    if rate < 0:
        raise ValueError("rate must be nonnegative")
    if not 0 < cycle_service < 1:
        raise ValueError("cycle service must be between zero and one")
    # This recurrence is appropriate for the small rate in this example.
    probability = exp(-rate)
    cumulative = probability
    stock = 0
    while cumulative < cycle_service:
        stock += 1
        probability *= rate / stock
        cumulative += probability
    return stock


evidence = {"A": 6, "B": 3, "C": 1}
prior = {"A": 1, "B": 1, "C": 1}
family_forecast = 10.0  # Expected units over 14 calendar days.
shares = smoothed_shares(evidence, prior)
forecasts = {sku: family_forecast * share for sku, share in shares.items()}

print("SKU  share     mean units / 14 days")
for sku in evidence:
    print(f"{sku:3}  {shares[sku]:.6f}  {forecasts[sku]:.6f}")
print(f"Family total: {sum(forecasts.values()):.6f}")

observed_units = 6
calendar_days = 30
available_days = 21
raw_rate = observed_units / calendar_days
exposed_rate = observed_units / available_days
full_exposure_estimate = exposed_rate * calendar_days
print(f"Raw rate: {raw_rate:.4f} units/calendar day")
print(f"Exposed rate: {exposed_rate:.4f} units/available day")
print(f"Full-exposure scenario: {full_exposure_estimate:.4f} units/30 days")

rate = forecasts["B"]
for candidate in (4, 5, 6):
    print(f"P(B demand <= {candidate}): {poisson_cdf(candidate, rate):.6f}")
stock = poisson_order_up_to(rate, 0.95)
on_hand = 1
on_order = 2
backorders = 0
inventory_position = on_hand + on_order - backorders
order = max(0, stock - inventory_position)
print(f"Order-up-to: {stock} units")
print(f"Inventory position: {inventory_position} units")
print(f"Recommended order: {order} units")
```

Expected output, allowing for rounding:

```text
SKU  share     mean units / 14 days
A    0.538462  5.384615
B    0.307692  3.076923
C    0.153846  1.538462
Family total: 10.000000
Raw rate: 0.2000 units/calendar day
Exposed rate: 0.2857 units/available day
Full-exposure scenario: 8.5714 units/30 days
P(B demand <= 4): 0.802176
P(B demand <= 5): 0.908129
P(B demand <= 6): 0.962463
Order-up-to: 6 units
Inventory position: 3 units
Recommended order: 3 units
```

The program prints probabilities to six decimal places.
The recurrence is intentionally limited to modest rates such as this example’s rate.
For very large rates, `exp(-rate)` can underflow and this simple recurrence can fail to advance; use a numerically stable vetted distribution implementation in that setting.
Real production code also needs effective-dated inputs, lead-time uncertainty, availability treatment, supplier constraints, and an audit record.
None of those is implicitly solved by this short example.

## 8. Data, standards, and regulatory context

### The minimum data contract

| Domain | Required fields | Why they matter |
|---|---|---|
| Item master | Internal SKU, trade-item identifiers when available, variant, pack, effective dates | Avoid mixing variants, packs, and lifecycle periods. |
| Hierarchy | Effective-dated item-to-family and family-to-category mapping | Reproduce what was known at each forecast origin. |
| Orders | Order and line IDs, timestamp, requested units, channel, destination, status history | Define the demand proxy and eliminate duplicate events. |
| Offer | Listing state, seller, displayed price, promotion and promise | Identify when the item was commercially accessible. |
| Availability | Sellable stock, reservation state, stockout intervals, destination eligibility | Distinguish exposed zeros from constrained observations. |
| Supply | Purchase orders, expected and actual usable receipt times, MOQ, pack sizes | Define the feasible replenishment policy and lead-time distribution. |
| Returns | Return authorization, receipt, inspection, and disposition | Avoid treating all returned units as immediately reusable. |
| Cost | Item, carrying, fulfillment, shipping, return and disposal costs | Compare feasible inventory and placement choices. |
| Governance | Source timestamp, ingestion timestamp, watermark, revision and access owner | Prevent leakage and make results reproducible. |

Use a documented time zone for every timestamp and define how a day closes across daylight-saving transitions.
Retain event time and ingestion time when late-arriving facts are possible.
An as-of reconstruction includes only records available by the forecast cutoff, even if a later corrected record describes an earlier business event.
When historical versions are unavailable, disclose that the backtest may contain hindsight.

### Item identification

GS1’s Global Trade Item Number (GTIN) identifies trade items under GS1 allocation and management rules.
It can help connect product information across trading partners when a valid identifier is present and correctly applied.
A GTIN does not by itself identify a merchant’s offer, a seller relationship, a node’s inventory state, or a customer’s desired variant.
Pack changes and item-identification decisions require attention to the GS1 rules and the actual master data.
The internal SKU-to-GTIN crosswalk should have an owner and effective dates.

A marketplace can have multiple offers for the same trade item.
An offer may differ by seller, price, shipping region, condition, and promise.
Model the demand target at the level of the decision; do not merge offers solely because their identifiers match.
Conversely, duplicative marketplace IDs can fragment a real item’s history if no reliable crosswalk exists.

### Event exchange and data ownership

Electronic Data Interchange (EDI) documents can communicate orders, acknowledgments, shipping notices, and invoices among partners.
The document type does not guarantee that the operational timestamp means the unit was physically received, inspected, or sellable.
Map each partner field to a canonical event and document local timing semantics.
Supplier acknowledgments and anticipated ship dates should be distinguished from actual usable receipts.

Data contracts should identify who can correct a catalog mapping, who owns stock-state definitions, and who resolves missing supplier events.
A forecast engineer cannot independently repair an incorrect product identity without business authority.
A buyer should be able to trace a recommended order to the catalog, demand, inventory, and supply snapshots used to create it.

### Privacy and lawful processing

Customer destination data can be personal data when linked to an identifiable person or household.
The European Union General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, establishes requirements including purpose limitation and data minimisation for processing within its scope; applicability and lawful basis depend on the facts.
Use the least granular location that supports the decision, control access, and define retention and deletion rules with the responsible privacy team.
A postal area can still be personal data in context, so aggregation alone is not an automatic exemption.
The regulation does not prescribe a forecasting model or a minimum safe group size for every use.

Regional privacy, consumer protection, accessibility, product safety, and hazardous-goods obligations can affect an assortment or delivery promise.
They vary by product, jurisdiction, and role.
For regulated goods, involve legal and compliance owners before deploying a policy; this lesson is educational and not legal advice.
A statistically attractive forecast never overrides a shipping restriction or mandatory product recall.

### Quality controls at the interface

1. Confirm that each active SKU has a valid unit of measure and conversion.
2. Check that time-valid parent mappings do not overlap in a strict hierarchy.
3. Reconcile order counts and units to the authoritative order ledger for the same cutoff.
4. Distinguish eligible zero sales from missing or unavailable observations.
5. Verify that inventory events and reservations do not double-count sellable stock.
6. Check that future-dated promotions are plans known at the origin, not hindsight outcomes.
7. Flag incomplete periods and late feeds instead of silently treating them as zeros.
8. Compare promised and actual usable lead time by supplier and item class.
9. Version cost definitions and currency, including the period of any exchange rate.
10. Record lineage and the owner of any manual adjustment.

### What the standards do not do

A shared identifier can improve joins but cannot make a biased sample representative.
A standard message can transmit a purchase order but cannot establish a customer’s latent demand.
A privacy rule constrains data use but does not select the correct aggregation or model.
A business-specific data contract must define these relationships and the evidence needed for the decision.

## 9. Industry practice and case evidence

### Public benchmark evidence: M5

The M5 forecasting competition used hierarchical unit-sales data from Walmart stores and included many item-level series.
Its published data and evaluation design are useful for studying aggregation, item-level forecasts, and external drivers.
The competition is not an e-commerce-only operational deployment and does not prove that a winning model improves a particular merchant’s inventory policy.
Sales records in such a dataset are still observations under the assortment and availability conditions in which they were collected.
The lesson is to evaluate at relevant levels and horizons, not to copy a leaderboard model without testing its decision impact.
See Makridakis and colleagues in Sources for the competition description.

### Public intermittent-demand evidence

Croston’s 1972 paper introduced a procedure that separately updates estimates of demand size and inter-demand interval for stock-control settings.
The Syntetos and Boylan work analyzes the accuracy and bias of intermittent-demand estimates.
Teunter, Syntetos, and Babai discuss intermittent demand linked to obsolescence and propose a probability-based approach.
These are methodological research sources, not proof that any method dominates for all e-commerce catalogs or all censored data.
Each should be evaluated against simpler baselines on an origin-safe sample and in the intended inventory policy.

### Public retail evidence: stockouts

Gruen, Corsten, and Bharadwaj’s industry study on retail out-of-stocks documents that shoppers can respond in several ways, including substitution and purchase elsewhere.
It is evidence that a stockout does not map mechanically to one lost unit for a particular merchant.
The scale, channel, geography, and period of that study differ from any specific online assortment.
Use it to frame possible customer responses, not to import a universal lost-sale multiplier.
A merchant’s own data or experiment is needed to estimate its incremental effect under its offer and promise.

### Illustrative business application: specialist components

A merchant offering appliance replacement filters sees a long catalog of compatible variants.
Some filters sell once a month, while the parent family sells daily.
An item-only daily model is unstable, so the merchant forecasts family units and estimates variant shares using compatible-model and lifecycle information.
A retiring appliance may warrant a declining prior; a newly supported appliance may warrant a positive prior despite no SKU sales.
The service owner may choose central stock for rare variants while preserving a clear standard-delivery promise.
This scenario is illustrative, not a claim about a named company.

An analyst should not pool all filters merely because the word “filter” occurs in their description.
Compatibility, dimensions, and installation requirements affect substitution and customer need.
If items cannot substitute, family-level demand can still provide statistical information, but the allocation must preserve the risk of stocking the wrong variant.
The merchant should monitor family accuracy and variant-level service separately.

### Illustrative business application: apparel sizes

An apparel style can have multiple colors and sizes, each represented by a separate SKU.
A style may sell frequently while a particular size-color combination sells rarely.
A hierarchy can borrow style, size curve, color, and region evidence to allocate demand.
The history of an unavailable size is censored; the remaining sizes may absorb some but not necessarily all demand.
Returns and exchanges arrive after the original purchase and may not be sellable when received.
A size curve fitted only to the sizes kept in stock can reinforce a poor initial allocation.

The merchant should compare allocation choices at launch and during replenishment.
The fulfillment owner considers stock location and delivery speed.
The merchandising owner evaluates size availability and brand expectations.
The finance owner assesses markdown and return exposure.
None of those decisions is solved by a single item-level point forecast.

### Illustrative business application: marketplace dropship

A marketplace may display a rare item that a third-party seller fulfills directly.
The platform may have visibility into listings and orders but not the seller’s real-time stock.
A forecast of observed platform orders can inform capacity or marketing, while the seller’s replenishment decision needs inventory and lead-time information the platform may not possess.
A long quoted delivery time can change conversion, so observed orders are conditional on that promise.
The marketplace should label unavailable supplier-stock data as a blind spot rather than assuming perfect availability.

### Illustrative business application: central versus local stock

A slow-moving replacement part sells in two regions, each with very few weekly events.
Central stock pools demand, but adds transit time for some customers.
Local stock can protect a fast promise but duplicates stock and can create stranded units.
Compare policies with realistic destination demand, demand dependence, replenishment lead times, transfer rules, and service penalties.
Do not add separate regional safety stocks and call the result the central requirement; calculate a pooled demand distribution instead.

### Evidence grading and transfer

Peer-reviewed methodological papers support claims about their models and experiments.
A documented competition supports claims about its dataset and evaluation rules.
An industry study supports observations within its methods and population, subject to transfer limits.
Illustrative examples are teaching scenarios, not empirical case outcomes.
For a real deployment, require local backtests, data-quality evidence, and controlled operational measurement before asserting benefit.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Missing rows treated as zeros | Sudden zero spike after an ingestion delay | Data absence becomes apparent weak demand | Underbuying | Separate missing, unavailable, and exposed-zero states; alert on feed freshness. |
| Stockout sales treated as complete demand | Strong sales only when stock is present | Inventory censors observations | Self-reinforcing underforecast | Track exposure and use justified scenarios or censor-aware models. |
| Day-end stock used as all-day exposure | Frequent intraday sellouts | Snapshot misses peak-hour unavailability | Biased rate and service measure | Use time-stamped stock states and demand-weighted exposure. |
| Promise ignored | Conversion falls after transit slows | Customer willingness depends on offered service | Wrong demand and placement assessment | Include promise scenarios; use experiments where feasible. |
| Wrong item identity | Pack or variant changes mid-history | Different products are combined | False trend and wrong purchase unit | Govern effective-dated identifiers and conversions. |
| Hierarchy rewritten with future knowledge | Historical backtest improves after taxonomy changes | Future mappings leak into past origins | Inflated reported accuracy | Store mappings as of origin or document consistent restatement. |
| Unavailable item gets TSB zero updates | Occurrence estimate decays during a stockout | Censored time is treated as no demand | Persistent understock | Update occurrence only under defined observable conditions. |
| Share allocation ignores stock differences | Better-stocked variant gains share | Sales shares reflect availability | Wrong mix despite accurate family total | Adjust evidence or analyze comparable exposure and uncertainty. |
| Aggregate model hides mix shift | Parent error low, item service poor | Correct total split among wrong items | Wrong variant inventory | Measure allocation and service at item and cohort levels. |
| Item quantiles summed | Regional safety stock added mechanically | Quantiles do not add under general dependence | Excess or insufficient pooled stock | Simulate or estimate the distribution of the aggregate. |
| Poisson assumption unchecked | Variance and bulk orders exceed the mean | Independent unit-arrival model underestimates tails | Target service missed | Compare event sizes and calibrate alternative distributions. |
| Lead time treated as fixed | Late receipts cluster by supplier | Supply uncertainty ignored | Stockouts despite correct demand mean | Use usable receipt history and scenarios. |
| Wrong protection period | Periodic review modeled as lead time only | Time to next review omitted | Insufficient stock | Include review period plus lead time when the policy requires it. |
| Inventory position double-counted | On-order includes canceled or already received units | Ledger states overlap | Order too little | Reconcile open supply and on-hand at decision time. |
| MOQ rounded blindly | Large stock jumps for rare items | Constraint handled without economic comparison | Excess and expiry | Evaluate feasible quantities, supplier alternatives, and service trade-offs. |
| Zero benchmark wins on a sparse metric | Low daily absolute error but zero stock | Frequent zeros dominate score | Poor customer service | Evaluate lead-time demand and policy cost alongside accuracy. |
| WAPE denominator near zero | Extremely volatile percentage results | Few actual units in the cohort | Misleading model ranking | Report numerator, denominator, volume, and complementary metrics. |
| Returns counted as instant stock | Forecasted returns exceed usable receipts | Reverse flow and inspection lag ignored | Phantom availability | Model return receipt and disposition separately. |
| Inconsistent destination and node | Forecast shifts after routing policy change | Historical assignments mistaken for demand | Misplaced stock | Forecast destination demand, then simulate routing. |
| Overridden forecast never expires | Manual uplift persists after event | Exception becomes permanent bias | Unnecessary purchases | Record reason, owner, expiry, and forecast value added. |
| Cold-start zero default | New items never receive stock | No own sales is interpreted as no interest | Launch failure | Use analog or prior with explicit uncertainty and review. |
| Stale model silently published | Run misses buyer cutoff without a warning | Fallback lacks age and status | Decisions based on outdated conditions | Publish timestamped approved fallback and alert owner. |
| Lost sales added twice | Family demand increases by all substitute sales | Original and replacement demand both counted | Overbuying | Model substitution and report incremental family effect separately. |
| Historical experiment confounded | Faster-promise buyers appear more valuable | Customer mix drives both promise and conversion | Wrong causal conclusion | Use randomized or otherwise credible causal designs when feasible. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or manage |
|---|---|---|---|
| Exposed time is representative after adjustment | Supports rate extrapolation to unavailable periods | Stockouts happen during promotions or high-intent hours | Compare by hour, day, and promotion; show sensitivity scenarios. |
| Sales are a useful demand proxy | Enables learning from observed transactions | Fraud, cancellations, suppressed listings, or payment failure | Define target and compare order-status cohorts. |
| Item identity remains stable | Makes a time series comparable | Pack, variant, seller, or identifier changes | Audit effective-dated product and pack mappings. |
| Family members share useful information | Pooling reduces sparse-series variance | Unrelated compatibility or lifecycle patterns | Backtest alternate pools and inspect item-level error. |
| Prior weights reflect plausible allocation | Gives cold items nonzero starting shares | Arbitrary or outdated launch assumptions | Backtest launches and tune prior strength using only past origins. |
| Parent forecast is relevant to current offer | Item shares inherit parent total | Price, promise, promotion, or assortment changes | Test contextual scenarios and monitor family bias. |
| Shares add to one over eligible children | Ensures coherent item means | Overlapping groups or unmodeled new items | Validate eligibility and aggregation matrix by horizon. |
| Demand distribution matches tail behavior | Converts a mean to a service level | Bulk orders, overdispersion, seasonality, or correlated events | Check lead-time coverage and compare alternatives. |
| Lead-time sample represents future supply | Defines exposure until replenishment | Supplier disruption, long-tail delays, or changed terms | Track actual usable receipts and stress supply scenarios. |
| Review timing is known | Defines the protection period | Missed reviews or delayed purchase approval | Include approval lag and monitor decision cadence. |
| Inventory position is accurate | Converts target stock to an order | Canceled supply, misplaced stock, or double reservations | Reconcile ledgers and validate physical counts. |
| Service target matches customer economics | Connects distribution to policy | Arbitrary target ignores margin or criticality | Compare cost and service across classes with owners. |
| Past demand remains relevant | Supports use of history | Discontinuation, compatibility loss, or new competitors | Add lifecycle signals and evaluate recency sensitivity. |
| Routing policy is represented | Translates customer demand to node workload | Policy or network changes | Simulate alternate routing under current constraints. |
| Evaluation outcomes are observable | Supports an unbiased score | Stockouts censor the periods being evaluated | Segment by exposure and label counterfactual uncertainty. |
| Historical data were available at origin | Makes backtest realistic | Late corrections or final promotion results leak backward | Reconstruct as-of snapshots and record unavoidable gaps. |
| Cost inputs cover relevant trade-offs | Allows policy comparison | Excluded returns, capital, or expedite costs | Finance reviews a versioned cost ledger. |
| Decision owners can act on recommendations | Converts analysis into an operational result | Buyer cutoff, supplier rejection, or capacity constraints | Define approvals, fallback, and execution feedback. |

No model can recover exact unmet customer demand from sales and stock flags alone without further assumptions or evidence.
A hierarchical mean is not a complete joint probability model.
A historical backtest estimates performance under the historical environment and observable outcomes, not necessarily under a new promise or placement policy.
The examples here therefore demonstrate a decision process, not a guaranteed service outcome.

## 12. Exercises

### Exercise 1 — Recognize exposed intermittency

**Time:** 10 minutes.

An active SKU has 20 fully observed, fully available days.
Positive daily unit counts are 2, 1, 3, and 2; all other eligible days are zero.
A separate five-day stockout follows, with no observed orders.

**Deliverable:** state the number of exposed demand events, the exposed-day rate, and approximate ADI in days per event.
Explain whether the five stockout days should be treated as observed zeros.

**Acceptance criteria:** use 20 exposed days for the descriptive calculations; distinguish units from events; identify the stockout as censored.

### Exercise 2 — Quantify availability assumptions

**Time:** 15 minutes.

An item records 12 units in 24 equivalent available days out of 30 calendar days.
No other demand or substitution evidence is available.
The six unavailable days include a major promotion.

**Deliverable:** calculate the raw calendar-day rate, exposed-day rate, and 30-day full-exposure scenario.
Give two reasons the difference between the scenario and observed sales is not a measured lost-sale count.

**Acceptance criteria:** show units, arithmetic, and the limitation imposed by the promotion and substitution or another defensible source of selection bias.

### Exercise 3 — Update an intermittent benchmark

**Time:** 15 minutes.

A Croston-style model has smoothed positive size `z = 2 units`, smoothed interval `p = 5 days`, and `alpha = 0.2`.
A new positive demand event of 4 units arrives after 3 eligible days since the previous positive event.
Assume the stated interval convention and no censoring.

**Deliverable:** calculate updated `z`, `p`, the Croston rate, and the SBA-adjusted rate using factor `1 - alpha/2`.

**Acceptance criteria:** show both exponential-smoothing updates and report units per day; do not present either rate as a calibrated demand distribution.

### Exercise 4 — Allocate and reconcile

**Time:** 15 minutes.

A family forecast is 24 units over four weeks.
Item evidence is A: 6 units, B: 2 units, C: 0 units.
Each item gets a prior of 2 pseudo-units and all three remain eligible for the horizon.

**Deliverable:** compute the smoothed shares and each item’s mean forecast.
Explain why C has a positive forecast and why this does not guarantee a sale.

**Acceptance criteria:** item shares sum to one and item forecasts sum to 24 units; calculations use evidence plus prior for both numerator and denominator.

### Exercise 5 — Select an inventory action

**Time:** 15 minutes.

A demand distribution over the relevant protection period has `P(D <= 2) = 0.80`, `P(D <= 3) = 0.93`, and `P(D <= 4) = 0.97`.
The target is 95% cycle service.
On hand is 1 unit, eligible on-order is 1 unit, and backorders are 1 unit.
There are no pack or MOQ constraints.

**Deliverable:** select the smallest order-up-to level and calculate the order quantity.
Identify the target’s service metric.

**Acceptance criteria:** use the first cumulative probability meeting 0.95; calculate inventory position consistently; do not call cycle service fill rate.

### Exercise 6 — Design an honest backtest

**Time:** 25 minutes.

A merchant wants to compare an item-level moving average, a family-share model, and a zero benchmark for a 28-day replenishment horizon.
Stockout history is incomplete for the earliest half of the sample.
A new hierarchy was introduced after some historical origins.

**Deliverable:** sketch a rolling-origin design, including a data cutoff, hierarchy treatment, observable-outcome definition, metrics, and a policy simulation.
State what cannot be established from the censored outcomes alone.

**Acceptance criteria:** prevent future leakage; separate missing availability from observed availability; compare forecasts and policies at relevant levels; report coverage and limitations.

### Exercise 7 — Critique placement and ownership

**Time:** 20 minutes.

A proposal says: “Move every item selling under one unit per week to a central warehouse; use the same next-day promise everywhere; expected savings equal the removed regional inventory value.”
Some items are critical repair parts and some are inexpensive discretionary accessories.

**Deliverable:** write a short review with at least four specific objections, the additional evidence needed, and named decision owners.

**Acceptance criteria:** address service feasibility, customer response, total cost, heterogeneity or pooling risk, and accountability.
Do not assert that centralization is always wrong or always right.

### Exercise 8 — Design the operating system

**Time:** 30 minutes.

A marketplace lists 2 million SKU–seller offers, including many newly launched offers and large quantities of missing supplier stock data.
The purchasing or capacity team needs a daily run before its cutoff.

**Deliverable:** propose the data representation, family or cohort model, cold-start method, monitoring, exception queue, and fallback.
Explain the difference between forecasting observed orders and claiming unconstrained demand.

**Acceptance criteria:** avoid materializing unnecessary full combinations; preserve zero versus missing states; define as-of lineage, owner, and freshness; describe one way to validate both accuracy and downstream decisions.

## 13. Self-check answers

### Exercise 1

There are four positive-demand events, not eight events.
The total positive quantity is `2 + 1 + 3 + 2 = 8 units`.
The exposed-day rate is `8 / 20 = 0.4 units per available day`.
The approximate ADI is `20 / 4 = 5 exposed days per event`.
This is a descriptive reciprocal of event frequency, not an exact observed list of complete inter-arrival times.
The five stockout days are censored and should not be added as confirmed zero-demand days.

### Exercise 2

The raw rate is `12 / 30 = 0.4 units per calendar day`.
The exposed rate is `12 / 24 = 0.5 units per available day`.
At that rate, the full-exposure scenario is `0.5 * 30 = 15 units per 30 days`.
The arithmetic difference is 3 units, but it is a counterfactual estimate, not a measured count.
Promotion days may have higher demand than the available-day average.
Some customers may substitute, abandon purchase, buy elsewhere, or not see the offer; the observed data do not identify those responses.

### Exercise 3

`z_new = 0.2 * 4 + 0.8 * 2 = 2.4 units per positive event`.
`p_new = 0.2 * 3 + 0.8 * 5 = 4.6 days per positive event`.
The Croston rate is `2.4 / 4.6 = 0.521739 units per day`.
The SBA factor is `1 - 0.2/2 = 0.9`.
The adjusted rate is approximately `0.469565 units per day`.
These are rate estimates, not service probabilities or lead-time demand quantiles.

### Exercise 4

Total evidence is 8 units and total prior is 6 pseudo-units, so the denominator is 14.
A has share `8/14 = 4/7` and forecast `24 * 4/7 = 13.714286 units`.
B has share `4/14 = 2/7` and forecast `24 * 2/7 = 6.857143 units`.
C has share `2/14 = 1/7` and forecast `24 * 1/7 = 3.428571 units`.
Shares sum to one and the means sum to 24 units, allowing for displayed rounding.
C’s nonzero prior expresses borrowed belief under the model; it does not guarantee any particular observed sale.

### Exercise 5

The smallest stock level meeting 95% cycle service is 4 units because 0.93 at 3 units is below the target and 0.97 at 4 units exceeds it.
Inventory position is `1 + 1 - 1 = 1 unit`.
The order quantity is `max(0, 4 - 1) = 3 units`.
The 97% modeled probability is a cycle no-stockout probability, not a fill rate.

### Exercise 6 rubric

A strong answer chooses multiple historical origins, freezes data at each cutoff, and evaluates the following 28-day period.
It retains the hierarchy known at each origin or explains a consistent alternative without hindsight.
It treats missing availability as unknown, reports observable coverage, and avoids treating constrained sales as complete demand.
It includes parent and item metrics, signed bias, and a policy simulation with common lead time and cost assumptions.
It explicitly says that exact latent demand and lost sales cannot be established from the censored records alone.
A weak answer randomly splits item-day rows or ranks models only on in-sample fit.

### Exercise 7 rubric

A strong answer notes that next-day feasibility depends on inventory position, transit lanes, and capacity.
It accounts for possible conversion change under a slower promise and separates inventory value from annual carrying-cost savings.
It compares central versus local service, shipping, expedite, obsolescence, and pooling dependence.
It distinguishes critical parts from discretionary accessories and identifies merchant, network, promise, and finance owners.
It proposes scenarios or a controlled rollout rather than assuming all items share the same economics.

### Exercise 8 rubric

A strong answer uses sparse events and state intervals while reconstructing eligible zero periods.
It uses a documented hierarchy or attribute prior for new items, records as-of model and data versions, and identifies missing supplier stock as an observation limitation.
It prioritizes exceptions by impact, defines a cutoff and timestamped fallback, and assigns ownership for catalog, supplier, and forecast issues.
It evaluates accuracy by cohort and horizon, along with simulated or observed service and cost under a specified policy.
A weak answer equates missing stock data with full availability or claims that all customer demand is known.

## 14. Interview questions

### 1. Why can low observed sales be a poor measure of demand?

A strong answer separates customer need from observable orders and names stockouts, listing exposure, price, promise, and substitution.
It describes a counterfactual as assumption-dependent.
A weak answer says that every zero sale proves zero demand.

### 2. When is hierarchical pooling useful?

A strong answer explains the variance reduction from shared evidence, the choice of meaningful groups, and the risk of smoothing away item-specific differences.
A weak answer assumes that every slow item should receive the same forecast.

### 3. How do you preserve add-up consistency?

A strong answer defines valid parent-child relationships and normalizes shares over the eligible children for each horizon.
It notes that coherent means can still be biased.
A weak answer rounds every SKU independently and assumes the total remains unchanged.

### 4. What is the difference between Croston and TSB?

A strong answer explains separate demand-size and interval smoothing for Croston, versus period-by-period occurrence-probability updating in TSB.
It identifies the risk of treating censored zero sales as non-occurrence.
A weak answer says either method solves stockouts automatically.

### 5. Why is a point forecast insufficient for an inventory decision?

A strong answer identifies protection-period demand uncertainty, lead-time variability, service or cost objectives, and inventory position.
A weak answer simply orders the rounded forecast regardless of stock already on order.

### 6. How do cycle service and fill rate differ?

A strong answer defines cycle service as a no-stockout probability for a specified cycle and fill rate as a fraction of requested units immediately filled.
A weak answer treats both as the same percentage.

### 7. What makes a historical backtest leakage-safe?

A strong answer freezes all features, hierarchy, and observations at each origin and uses the actual operational horizon.
It flags late data and unknown availability.
A weak answer uses a random row split or revised future promotions.

### 8. Why can forecasting fulfilled units by warehouse perpetuate bias?

A strong answer explains that warehouse sales depend on historic placement, routing, and stock availability, then proposes destination-demand forecasting and scenario routing.
A weak answer assumes no warehouse sales means no local customers.

### 9. How would you launch a new SKU?

A strong answer uses a justified analog or prior, checks lifecycle and exposure, sets an uncertainty-aware policy, and monitors early outcomes.
A weak answer assigns zero forever because the item has no history.

### 10. What changes at a million-item scale?

A strong answer mentions sparse representations, shared estimation, reproducible batch scoring, exception prioritization, and lineage.
It preserves eligibility and missingness rather than erasing them for speed.
A weak answer proposes a separate heavyweight manual analysis for each item.

### 11. How would you assess a central-pool proposal?

A strong answer compares pooled demand and dependence, delivery promise, shipping, inventory cost, and customer response under feasible policies.
A weak answer assumes that centralization always lowers total cost.

### 12. What would trigger a human override?

A strong answer names decision-impact thresholds, evidence unavailable to the model, a responsible owner, a reason, and an expiry date.
It preserves both the statistical forecast and the override for later evaluation.
A weak answer allows an undocumented permanent uplift whenever a manager prefers it.

## 15. Key terms

- **[ADI](#step-5--describe-intermittency-and-event-size):** approximate eligible bucket count per positive event.
- **[Analog](#step-7--forecast-the-parent-and-form-item-shares):** comparable item or cohort used to inform sparse demand.
- **[Availability](#step-4--quantify-availability-and-censoring):** ability to buy the offer under defined stock and eligibility conditions.
- **[Backtest](#step-11--backtest-both-forecasts-and-decisions):** historical simulation with data frozen at each origin.
- **[Bias](#step-11--backtest-both-forecasts-and-decisions):** systematic signed forecast error under a declared convention.
- **[Censoring](#step-4--quantify-availability-and-censoring):** incomplete observation caused by a constraint such as stockout.
- **[Cold start](#step-7--forecast-the-parent-and-form-item-shares):** estimation with little or no own-item history.
- **[Croston method](#step-6--establish-baselines-before-complex-models):** separate smoothing of positive demand size and inter-demand interval.
- **[Cycle service](#step-10--convert-forecast-into-an-inventory-decision):** probability of no stockout in a defined cycle.
- **[Demand event](#step-5--describe-intermittency-and-event-size):** positive demand occurrence in a specified bucket.
- **[Exposure](#step-4--quantify-availability-and-censoring):** interval and conditions in which an offer can be purchased.
- **[Family](#step-2--build-an-as-of-catalog-and-hierarchy):** related set of items used as an aggregation or pooling unit.
- **[Fill rate](#step-10--convert-forecast-into-an-inventory-decision):** fraction of requested units immediately filled.
- **[Forecast origin](#step-11--backtest-both-forecasts-and-decisions):** timestamp at which a forecast is made.
- **[GTIN](#item-identification):** GS1 global identifier for a trade item.
- **[Hierarchy](#step-2--build-an-as-of-catalog-and-hierarchy):** mapping of items or locations to aggregation groups.
- **[Intermittency](#step-5--describe-intermittency-and-event-size):** positive demand separated by zero-demand periods.
- **[Inventory position](#step-10--convert-forecast-into-an-inventory-decision):** on-hand plus eligible on-order less backorders under a defined convention.
- **[Lead time](#step-9--construct-uncertainty-for-the-protection-period):** duration from ordering to usable receipt.
- **[Lost sale](#step-4--quantify-availability-and-censoring):** purchase that would occur under a specified counterfactual offer but did not.
- **[MOQ](#step-10--convert-forecast-into-an-inventory-decision):** supplier minimum order quantity.
- **[Obsolescence](#step-13--monitor-and-retire):** decline in demand or sellability as an item becomes outdated.
- **[Order-up-to level](#step-10--convert-forecast-into-an-inventory-decision):** desired post-order inventory position.
- **[Poisson distribution](#choose-bs-stock-level):** illustrative discrete count model with equal mean and variance.
- **[Prior](#step-7--forecast-the-parent-and-form-item-shares):** pre-observation weight or distribution used in estimation.
- **[Protection period](#step-9--construct-uncertainty-for-the-protection-period):** demand coverage window implied by the review and supply policy.
- **[Reconciliation](#step-7--forecast-the-parent-and-form-item-shares):** ensuring forecasts obey aggregation relationships.
- **[SBA](#step-6--establish-baselines-before-complex-models):** Syntetos–Boylan approximation to adjust a Croston-style rate.
- **[Share](#step-7--forecast-the-parent-and-form-item-shares):** item’s fraction of an eligible parent forecast.
- **[Shrinkage](#step-7--forecast-the-parent-and-form-item-shares):** blending sparse local evidence with pooled information.
- **[SKU](#step-1--define-the-business-question-and-decision-grain):** merchant-defined stock-keeping unit.
- **[Substitution](#step-4--quantify-availability-and-censoring):** choosing another item when a preferred item is unavailable or unattractive.
- **[TSB](#step-6--establish-baselines-before-complex-models):** intermittent-demand method updating occurrence probability each observed period.
- **[WAPE](#step-11--backtest-both-forecasts-and-decisions):** total absolute error divided by total actual demand, when the denominator is positive.

## 16. Sources

Sources below are checkable starting points, not evidence that any specific model will work in a particular merchant’s data.
Publication context and applicability are noted in Section 9.

1. Croston, J. D. (1972). “Forecasting and Stock Control for Intermittent Demands.” *Operational Research Quarterly*, 23(3), 289–303. [DOI: 10.2307/3007885](https://doi.org/10.2307/3007885). Foundational intermittent-demand method.
2. Syntetos, A. A., and Boylan, J. E. (2005). “The accuracy of intermittent demand estimates.” *International Journal of Forecasting*, 21(2), 303–314. [DOI: 10.1016/j.ijforecast.2004.10.001](https://doi.org/10.1016/j.ijforecast.2004.10.001). Analysis of intermittent-demand estimators.
3. Teunter, R. H., Syntetos, A. A., and Babai, M. Z. (2011). “Intermittent demand: Linking forecasting to inventory obsolescence.” *European Journal of Operational Research*, 214(3), 606–615. [DOI: 10.1016/j.ejor.2011.05.018](https://doi.org/10.1016/j.ejor.2011.05.018). Occurrence-probability method and obsolescence context.
4. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). “The M5 accuracy competition: Results, findings, and conclusions.” *International Journal of Forecasting*, 38(4), 1346–1364. [DOI: 10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013). Hierarchical retail forecasting competition.
5. Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed., “Hierarchical and grouped time series.” [Online textbook](https://otexts.com/fpp3/hierarchical.html). Aggregation and reconciliation concepts.
6. Gruen, T. W., Corsten, D. S., and Bharadwaj, S. (2002). *Retail Out-of-Stocks: A Worldwide Examination of Extent, Causes and Consumer Responses*. [ECR Europe report](https://www.ecr-community.org/wp-content/uploads/2016/05/2002-OOS-Worldwide-Examination-of-Extent-Causes-and-Consumer-Responses.pdf). Industry evidence on out-of-stock responses; transfer to a specific e-commerce setting requires care.
7. GS1. [GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/). Official rules for trade-item identification and changes.
8. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Official legal text; consult relevant counsel for applicability.
9. OpenStax. [Introductory Statistics 2e, Poisson Distribution](https://openstax.org/books/introductory-statistics-2e/pages/4-6-poisson-distribution). Reference for the illustrative count distribution.

## 17. Links

- [Module 21 index](_index.md).
- [Module 21 survey and embedded lab context](../../../course/21_ecommerce_supply_chain_analytics.md).
- [Sibling topic 02: Order-grain forecasting](02_order-grain-forecasting.md).
- [Sibling topic 04: Inventory placement](04_inventory-placement.md).
- [Module 07: Intermittent demand methods](../../02_methods/07_time_series_forecasting/06_intermittent-demand.md).
- [Module 07: Hierarchy and aggregation](../../02_methods/07_time_series_forecasting/05_hierarchy-aggregation.md).
- [Module 07: Rolling-origin backtesting](../../02_methods/07_time_series_forecasting/08_rolling-origin-backtesting.md).
- [Module 07: Error metrics and cost](../../02_methods/07_time_series_forecasting/09_error-metrics-cost.md).
- [Module 30 simulation and optimization labs](../../../course/30_simulation_optimization_production_labs.md).
