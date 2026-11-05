# Demand Forecasting at Order Grain

> Module 21 — E-commerce Supply Chain Analytics  
> Topic 02 of 10  
> Estimated study time: 4 hours  
> Prerequisites: forecasting fundamentals and fulfillment models  
> Version 2026.5

## 1. Why this topic exists

An e-commerce operation can meet a monthly sales target and still fail customers on Tuesday.
The failure occurs when orders arrive faster than a particular node can pick, pack, or hand them to a carrier.
Monthly stock-keeping-unit (SKU) demand does not say how many workers are needed at 15:00 local time.
It also does not identify which orders are routed to a particular building, which will be split, or which require special handling.

The workforce planner needs a staffing decision before the shift begins.
The transportation manager must reserve carrier collection capacity before the truck arrives.
The network planner must decide whether an overloaded node can send work elsewhere without breaking the delivery promise.
The commercial owner needs to know whether a campaign will create profitable demand or simply move the bottleneck.
A forecast is useful only when it arrives before one of those decisions and describes its operational units.

Order forecasting is not a substitute for item forecasting.
One customer order can contain several lines and units, generate more than one shipment, and consume unequal amounts of labor.
Conversely, the same unit demand can be packaged into very different numbers of orders.
The relevant outcome is therefore a distribution of arrivals and work by time, node, and processing stage.

A forecast describes what may happen under stated conditions.
A plan specifies a chosen response, such as a staffing roster.
A target expresses an aspiration, such as a desired on-time rate.
A commitment is an externally or internally accepted obligation, such as a customer delivery promise or contracted carrier pickup.
None of these should silently replace another in a performance report.

## 2. Learning outcomes

After this topic, you can:

1. Define an order, event timestamp, node, slot, and measurable workload without double counting.
2. Translate a business decision into a forecast horizon, grain, cutoff, and loss function.
3. Build a day-to-slot-to-node baseline and reconcile its sums.
4. Estimate workload and capacity from basket, routing, split, and productivity assumptions.
5. Interpret intervention scenarios without presenting correlation as causal effect.
6. Backtest a forecast with as-of information and quantify decision consequences.
7. Recommend an accountable staffing or routing action under uncertainty.
8. Identify limitations, privacy controls, and evidence needed before deployment.

Exercises 1–8 respectively assess these outcomes.

## 3. The core mental model

```text
Customer opportunity and commercial interventions
       |          availability / price / promise
       v
Orders placed by day and local slot
       |          cancellation, fraud review, payment authorization
       v
Eligible orders and order lines
       |          allocation policy + inventory + network constraints
       v
Node assignments and split shipments
       |          basket, product handling, travel, productivity
       v
Work by station and interval --> queue and cutoff risk --> customer outcome
       ^                                 |
       |________ decisions change both arrivals and routing ________|
```

Start at the decision rather than the model.
A daily staffing decision and an hourly release decision use different lead times, refresh rates, and accuracy requirements.
A node manager may control shift hours but not the campaign calendar.
A central network planner may change routing but cannot conjure inventory at a node.
Mapping rights and constraints prevents a forecast from becoming an unactionable dashboard.

The diagram is a causal and accounting map, not a claim that each arrow can be estimated without bias.
Traffic, conversion, product availability, and promise all influence observed order arrivals.
An increase in observed orders after a campaign is not necessarily caused by that campaign: seasonality and concurrent price changes may also matter.
Likewise, historical node shares embed past routing rules and capacity restrictions.
If the rule changes, those shares need not remain stable.

Flows have different denominators.
An order count is a count of distinct accepted commercial orders.
An order-node assignment counts one order once for each node asked to fulfill it.
A shipment count counts parcels or consignments, depending on the declared definition.
If an order is assigned to two nodes, the sum of node assignments can exceed the number of unique orders.
If it is shipped in three parcels, shipment count can exceed both.

The work generated at a node is not necessarily completed in the arrival slot.
A batch release policy can defer work; a queue carries unfinished work forward.
A pick forecast predicts required work, whereas a completion forecast also depends on available capacity and scheduling.
Forecast the unconstrained work offered to the operation, then model the constrained completion process.
Using historical completions alone as demand can hide the queue left behind on overloaded days.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Order | One distinct commercial order identifier, count of orders. | Counting lines as separate orders. |
| Order line | A SKU and quantity on an order, count of lines. | Equating lines with units. |
| Unit | An individual item ordered, count of units. | Treating a multi-pack and a physical each identically without a conversion. |
| Placed | Customer submits an order, timestamp. | Treating all placed orders as payment-approved work. |
| Accepted | Order passes the agreed eligibility rule, timestamp and count. | Changing the rule between training and scoring. |
| Released | Order or line enters a fulfillment workflow, timestamp. | Assuming release occurs when the customer orders. |
| Cancellation | Order or line removed before the relevant work event, count or probability. | Subtracting cancellations that occurred after picking from pick workload. |
| Slot | Defined interval such as 15 minutes in node-local time. | Mixing UTC and local-time intervals. |
| Node | A fulfillment site or other routing destination, coded identifier. | Counting a merchant as a physical building without a mapping. |
| Destination | Customer delivery region, coded geography. | Using today's region boundaries for historical observations without versioning. |
| Allocation | Assignment of order work to one or more nodes. | Assuming historical allocation survives a policy change. |
| Split order | An order fulfilled by multiple nodes or shipments under a stated rule. | Using node split and parcel split as synonyms. |
| Assignment | A unique order-node pair. | Summing assignments and calling the result unique orders. |
| Shipment | A defined carrier handoff entity, count. | Assuming a shipment always equals one parcel. |
| Workload | Expected processing effort, such as standard labor minutes. | Treating all orders as equally difficult. |
| Capacity | Available processing effort or completions in a period. | Using theoretical machine speed as staffed effective output. |
| Productivity | Output per productive labor hour under a stated mix. | Applying an average rate to an incompatible workload mix. |
| Backlog | Uncompleted eligible work at a boundary, in work minutes or units. | Ignoring work already waiting at the start of a slot. |
| Cutoff | Time by which an operational event must occur to meet a constraint. | Confusing the order cutoff with the carrier pickup time. |
| Forecast origin | Time at which a forecast is issued. | Using data published after that time. |
| Horizon | Time between forecast origin and target event. | Comparing forecasts at different lead times as if equally informed. |
| As-of data | Values available at the forecast origin. | Using a later corrected campaign plan in an earlier backtest. |
| Intervention | A controllable change, such as a campaign or a routing rule. | Treating a planned intervention as an observed causal effect. |
| Scenario | A conditional coherent set of assumptions. | Calling an arbitrary stress case a calibrated percentile. |
| Quantile | Value exceeded with a specified probability under a predictive distribution. | Adding marginal quantiles and claiming a joint quantile. |
| Reconciliation | Adjustment so related forecasts obey specified aggregation identities. | Forcing a false identity between distinct orders and node assignments. |
| Bias | Signed average forecast error under a declared convention. | Calling a high absolute error bias. |
| WAPE | Sum of absolute errors divided by sum of actuals. | Averaging individual percentage errors and calling it WAPE. |
| Service risk | Probability or consequence of missing a specified service commitment. | Treating forecast accuracy as service performance. |

## 5. The mechanics, step by step

### 5.1 Start with the decision contract

1. Name the owner, action, deadline, and cost of changing the action.
2. Define the service commitment and the exact event that demonstrates compliance.
3. Define the forecast origin and the first and last target slots.
4. Choose the quantity that directly drives the resource constraint.
5. Record which inputs are known, planned, or uncertain at the origin.
6. Agree on a fallback when data or model output is unavailable.

For example, the staffing lead sets the next-day roster at 14:00 local time.
The immediate action is paid labor hours, not a change to the statistical forecast.
The carrier manager separately reserves parcels by 16:00 local time.
The commercial manager owns campaign changes and the customer-promise owner approves any promise relaxation.
Document escalation rights for actions that cross these boundaries.

A forecast should carry an issue timestamp, target interval, geographic scope, version, and units.
The same target may have several valid forecasts issued at different horizons.
Do not overwrite the version actually used to make a decision.
A late forecast can be more accurate yet less useful if the staffing deadline has passed.

### 5.2 Define the event and grain

Choose a precise event, such as payment-authorized order creation, and a canonical order key.
Explain whether replacements, test orders, subscriptions, exchanges, fraud holds, and marketplace orders belong in the population.
Persist the original event time and the ingestion time separately.
For each record preserve correction and deletion history where legally permitted.

Choose a nonoverlapping slot convention, such as `[09:00, 09:15)`.
An order at exactly 09:15 belongs to the next slot.
Store event times in UTC and convert using the node's time zone for operational shifts.
Keep the UTC offset or interval identifier so the repeated local hour during the autumn daylight-saving transition remains distinguishable.
The spring transition has a missing local hour; do not fabricate observations in that hour.
If a region does not change clocks, its local-day mapping differs from one that does.

Separate the time of demand from the time of processing.
Customer demand may be reported in the customer's local day, while a node roster follows the node's local day.
The same instant can belong to different calendar dates in these two views.
Choose one basis per calculation and map explicitly when combining them.

### 5.3 Construct the point-in-time dataset

The input is a historical event table, versioned routing and campaign information, and relevant calendar and availability data.
For each forecast origin, select only records available by that origin.
Historical outcomes may be corrected later; that is acceptable for an evaluation truth table if corrections are documented, but not as features unavailable at the origin.
Record late-arrival rates and the evaluation truth freeze date.

Deduplicate event retries using a stable event identifier and an explicit precedence rule.
Reconcile accepted order counts against an independent order-management control total by day and channel.
Keep missing slots distinct from confirmed zero-order slots.
A pipeline outage is not evidence of zero demand.
Flag irregular days including system outages, site closures, fulfillment pauses, and measurement changes.
Do not automatically delete those days: they may be relevant stress cases or interventions.

### 5.4 Establish a defensible daily baseline

Use a comparable-day baseline before adding complexity.
For instance, forecast a Tuesday using recent eligible Tuesdays, adjusted for holidays and known structural changes.
A trailing mean is simple, but an unusual promotion in the training window can contaminate it.
A median can be robust but may understate a repeatable high-demand event.
Compare both under rolling-origin backtests and use the decision loss, not model sophistication, to choose.

Segment only when there is enough information and an operational reason.
Channel, destination, and product family can affect mix, but highly granular daily cells are often sparse.
A national daily forecast can be stable while one small node is uncertain.
Pooling or hierarchical forecasting can borrow information across similar cells.
Keep a simple aggregate benchmark to detect spurious improvements in detailed cells.

A model may use traffic forecasts, conversion, calendar, price, campaigns, and availability.
A traffic plan is not realized traffic; a model trained with realized traffic must be evaluated with traffic forecasts available at the origin.
If price is itself changed in response to anticipated demand, a naive regression coefficient may not identify price elasticity.
Do not present that coefficient as an intervention effect without a defensible identification strategy.

### 5.5 Disaggregate daily demand into slots

Estimate slot shares from comparable days and ensure the shares are nonnegative and sum to one over the target day.
A daily forecast of 1,000 orders and a slot share of 0.08 imply 80 orders in that slot before rounding.
Use separate profiles for weekdays, holidays, channels, or campaign types only when data supports the split.
An email sent at 18:00 can change the shape even if daily volume is unchanged.
A slot profile built from ordinary Tuesdays should not automatically be used for a launch Tuesday.

Estimate shares using the same local-time convention as the target.
For an irregular 23- or 25-hour local day, define the actual slots and renormalize an appropriate profile; preserve the repeated hour's offsets.
If separate slot models are used, reconcile their outputs to the daily total when that total is an intended accounting constraint.
Avoid rounding each cell independently before aggregation: the rounded cells may not add to the rounded total.
Keep fractional expectations for planning arithmetic and apply a documented integer allocation method only where needed.

### 5.6 Forecast node routing and splits

A simple share model works only while routing policy, stock, promise, and capacity remain sufficiently comparable.
A better conditional simulation assigns each forecast order or order class to an eligible node based on destination, inventory, promised service, and policy.
When the feasible node set changes, rerun allocation under the proposed state rather than merely applying last month's shares.
The allocation owner supplies the rule; the inventory owner supplies the availability assumption.

Distinguish mutually exclusive primary-node assignments from all-node assignments.
If every accepted order has exactly one primary node, primary-node counts sum to total accepted orders.
If a fraction needs another node, all-node assignment counts exceed distinct orders.
For example, 100 orders with 10 two-node splits produce 110 order-node assignments, provided none uses more than two nodes.
That arithmetic does not imply 110 customers or 110 independent orders.

In practice, some orders cannot be allocated, some are held, and some are fulfilled by a supplier.
Represent those as explicit categories instead of forcing every record into an internal node.
If a node is at capacity, assignment may be delayed or rerouted; that response changes the node distribution.
The forecast should identify whether it describes preferred assignment, feasible assignment, or actual completion.

### 5.7 Translate demand to workload

Start with the object a resource processes.
Pickers may handle lines and units, packers handle orders and parcels, and carriers receive parcels.
A useful standard-minute model can be written as:

`pick minutes = order-node assignments × (minutes per assignment + lines per assignment × minutes per line)`.

This is an illustrative additive engineering model, not a universal labor standard.
Product location, batch picking, travel, congestion, special handling, and learning can make the relationship nonlinear.
If a time standard already includes travel or setup, do not add it again.
Estimate workload by relevant class when the mix varies materially, then sum expected minutes.
Validate standards against actual work and distinguish elapsed time from productive time.

Workload can cross slot boundaries.
Apply a release-delay distribution when the question is when work arrives at a station rather than when the order was placed.
For class `c`, release slot `t`, and order slot `s`, a convolution sums forecast orders in `s` multiplied by the probability of release in `t` and expected work for `c`.
The delay distribution must reflect current cutoffs and holds; a batching-policy change invalidates historical delays.
Use consistent workload units when carrying queues forward.

### 5.7.1 Map each resource to its workload driver

The translation from orders to minutes should be explicit for each station.
It is tempting to use one universal minutes-per-order factor.
That shortcut masks changes in basket composition and packaging.
It also prevents the operations owner from explaining why one station is overloaded.

For customer service, relevant drivers can include orders, delivery exceptions, and contacts per exception.
For picking, they include order lines, physical units, location visits, and handling class.
For packing, they include cartons, packaging type, and value-added services.
For sortation, they include parcels and destination lanes.
For carrier collection, they include parcels, weight, cube, and handoff window.
For replenishment, the drivers may be pick-face depletion and replenishment trips.
These are different processes and should not share a denominator by accident.

Consider two days with 1,000 orders each.
On the first day, each order averages 1.2 lines.
On the second, each order averages 2.0 lines.
If pick work has a fixed component of 1 minute per order and a variable component of 0.5 minute per line, the first day needs 1,600 standard minutes.
The second needs 2,000 standard minutes.
The order count is unchanged, but modeled pick effort increases by 25%.
The extra 400 minutes can require another shift or change the point at which orders miss a cutoff.
This calculation assumes that lines have equal effort and that batching does not change.

A good workload table records the unit, estimate, origin, version, and owner of each factor.
For example, an industrial engineer may own a standard of 0.5 minute per line.
The forecasting team may own the expected line count.
The shift manager may own planned productive availability.
Changing one factor should not silently overwrite another.
If a standard changes after a process redesign, preserve the effective date.

### 5.7.2 Bridge event time to work arrival

Orders do not always enter a node's work queue immediately.
Payment validation, fraud review, inventory allocation, wave planning, and manual holds can delay release.
Forecasting orders by customer checkout time is therefore only the first stage.
The next stage converts checkout time into work-release time.

Suppose 100 orders are accepted in the 10:00–11:00 slot.
Historical evidence under the current process suggests 70% release in that slot, 20% in the next slot, and 10% in the following slot.
The expected releases from that cohort are 70, 20, and 10 orders respectively.
If each released order requires 3 work minutes, those expected releases create 210, 60, and 30 work minutes.
The total is 300 work minutes across all three slots.
It would be wrong to allocate all 300 minutes to the first slot merely because the orders arrived then.

Now suppose another 80 orders arrive in the next slot with the same delay distribution.
That second cohort contributes 56 releases in its arrival slot and 16 in the next.
The second slot therefore receives 20 releases from the first cohort and 56 from the second, for 76 releases.
At 3 work minutes each, the combined incoming work is 228 minutes.
This is a convolution of the arrival series with the delay distribution.
Do not add the cohort total again when combining slots.

The delay distribution must be conditional where the process warrants it.
A fraud hold may delay one channel much longer than another.
A preorder can have a release date unrelated to its purchase date.
A same-day promise may cause a separate priority queue.
Historical release delays observed during a capacity outage may not describe normal conditions.
Compare release curves by origin, channel, promise, and operating regime where sample sizes permit.

A partial day requires an explicit boundary.
Orders placed before midnight can create work after midnight.
Do not truncate their delayed releases simply because the daily forecast ends.
Carry the expected release tail into the next operating day.
Likewise, work arriving from earlier days must be included in the initial queue or release schedule.
Otherwise the model loses work at the edge of its horizon.

### 5.7.3 Decide whether and how to pool sparse cells

Some destination-node-slot combinations have few historical orders.
An apparent 50% increase from two to three orders is not reliable evidence of a structural shift.
Start by asking whether the cell affects a decision materially.
A low-volume special-handling lane may still be important if it has a hard carrier cutoff.
A large aggregate cell may hide a scarce skill or equipment requirement.

One simple pooling method estimates a local slot share and blends it with a broader regional share.
For illustration, suppose the local share is 8%, the regional share is 10%, and the chosen local weight is 0.25.
The blended estimate is `0.25 × 8% + 0.75 × 10% = 9.5%`.
The weight is a modeling choice, not a confidence level.
Select or estimate it using historical validation rather than assigning precision that is not supported.

Pooling should preserve the business definition of the series.
Do not pool frozen-food picking with ambient picking merely because both are sparse.
The operational work standards, shift schedules, and available resources may differ.
Do not pool neighboring destinations if their promised delivery options differ materially.
Use a fallback hierarchy that reflects real similarities.
Document when the model falls back from node to region, then region to network.

If you blend slot shares independently, renormalize them across the full day.
Otherwise the blended shares can fail to sum to one.
Check that all values are nonnegative.
Preserve the original and adjusted values for inspection.
For rare slots, a pooled share can be more stable while still missing an exceptional campaign.
It should be accompanied by scenario analysis rather than treated as certain.

### 5.7.4 Separate inventory constraints from customer preference

Observed orders depend on what customers could actually purchase.
If the popular size of a product is unavailable, demand for that size is not observed in completed orders.
Some shoppers substitute another item; others abandon the session.
Some wait until inventory returns.
The observed sales series alone cannot reveal those unobserved choices.

Suppose a category normally generates 200 orders on a comparable day.
During a stockout, observed orders are 140.
The difference of 60 orders is not automatically lost demand.
Other causes could include price, weather, a campaign change, or ordinary noise.
Moreover, some customers may have substituted within the category, so simple lost-sales addition can double count.
Use availability, traffic, substitution patterns, and a credible comparison when estimating the effect.
When evidence is insufficient, provide a bounded scenario instead of asserting a precise latent demand figure.

Inventory availability also changes routing.
Two regions with identical order demand can generate different node workloads if stock is concentrated differently.
A line that cannot be filled at the preferred node may cause a second-node split.
The resulting additional assignment and parcel can increase both warehouse work and transport cost.
If the promise cannot be met from the alternative node, the order may be held or rejected instead.
The allocation simulation must represent the actual rule and available stock.

### 5.7.5 Keep expected workload and service risk distinct

An expected workload of 480 minutes does not imply that 480 minutes of capacity will suffice on every day.
It is an average across possible outcomes.
If the outcomes are 400 minutes and 560 minutes with equal probability, their expected workload is 480 minutes.
Capacity of 480 minutes leaves an 80-minute deficit in the high outcome.
It leaves 80 minutes unused in the low outcome.
The average signed difference is zero, but the risk of a deficit is 50%.

The business value of reserve depends on the cost and consequences of each outcome.
If all work can wait until the next shift without service impact, excess capacity may be more expensive than temporary backlog.
If the work is due before the last carrier collection, a shortfall may create missed commitments.
The same demand distribution can therefore justify different plans at different cutoffs.
Do not use a universal safety factor without reviewing the underlying consequence.

For a multi-node network, reserve can be pooled only when resources and work are truly transferable.
Stock, labor skill, travel time, transport capacity, and delivery promise may prevent pooling.
Even when an alternate node is technically eligible, routing changes can add cost or generate a second parcel.
Evaluate the entire feasible action, not just the apparently spare minutes.
The network planner confirms feasibility; finance reviews the economic tradeoff.

### 5.8 Convert workload into a capacity decision

Effective slot capacity equals scheduled workers times slot duration times productive availability times the applicable productivity factor.
Breaks, training, absenteeism, equipment downtime, and nonproductive tasks belong in the availability assumption unless already included in the labor standard.
A capacity number derived from total paid hours cannot also be reduced for the same breaks twice.
If workers share equipment or downstream stations constrain output, the effective system capacity is not simply the sum of independent nominal capacities.

For one bottleneck with work expressed in minutes, use:

`ending backlog = max(0, starting backlog + arriving work - effective capacity)`.

This identity is a coarse fluid approximation.
It does not establish that every order meets its individual cutoff, and it does not capture travel, batching, or priority rules.
A separate order-level simulation or timestamp-based queue analysis is needed when cutoff compliance matters.
The next slot's starting backlog is the previous slot's ending backlog after any explicitly modeled transfers or cancellations.

Compare feasible actions: add a shift, redeploy workers, reserve overflow, reroute eligible work, or adjust the commercial offer.
List cost, lead time, constraints, owner, and customer impact for each action.
An overflow node is not a valid option merely because its nominal capacity is free; it must have stock and meet the promised delivery time.
If no feasible action covers the risk, escalate before making a customer commitment that cannot be met.

### 5.9 Quantify uncertainty and dependence

Point forecasts do not tell a planner how much reserve is needed.
Construct prediction intervals from backtest residuals or a probabilistic model at the decision horizon.
Check empirical coverage by node, day type, and peak regime, not just in aggregate.
Small samples yield unstable tail estimates, so disclose the number of comparable historical cases.
A nominal 90% interval is not automatically calibrated during an unprecedented promotion.

Generate joint scenarios for daily volume, slot shape, routing, basket mix, release delay, and productivity.
Correlations matter: a large promotion can increase both orders and the proportion of multi-line baskets.
Do not independently combine every worst marginal percentile and label the result a 95th-percentile outcome.
Simulate coherent paths or use a clearly labeled stress scenario.
For each path, compute workload, backlog, late-risk proxy, and cost under each feasible action.

If undercapacity costs more than excess capacity, the preferred capacity can exceed expected workload.
Under a simplified single-period newsvendor model with per-unit underage cost `Cu` and overage cost `Co`, choose the demand quantile `Cu/(Cu+Co)`.
That result assumes known costs, a single period, and no complicated service constraints.
It is not a replacement for multi-slot scheduling or contractual service obligations.

### 5.10 Evaluate interventions without confusing them with demand

Describe the intervention, eligibility population, start time, and expected mechanism before estimating an effect.
Examples include a promotion, delivery-promise change, site outage, stock restriction, routing rule, and carrier cutoff change.
Some affect customer demand; others mostly change where and when the same work is processed.
An intervention can affect both.

For a campaign, compare treatment and a defensible control where feasible.
Random assignment is preferable when ethical and practical; otherwise check pre-trends, contemporaneous changes, and selection.
A before-after comparison alone is vulnerable to weekday, season, competitor, and inventory changes.
Express uncertain uplift as a range or distribution and say whether it is experimentally estimated, observational, or a planning judgment.
Do not add an uplift to a baseline that already incorporates the same promotion.

For a capacity throttle or a long delivery promise, observed orders can be censored by the intervention itself.
When customers cannot buy or see longer delivery times, fulfilled demand understates potential demand.
Estimate lost demand only with additional evidence and state identification limits.
Treating missing demand as zero will systematically understate future demand after the constraint is removed.

### 5.11 Backtest as it would have been used

Choose multiple rolling origins and recreate each origin's feature snapshot and known intervention schedule.
Preserve the actual decision lead time: a next-day roster forecast should not use same-day actual traffic.
Score daily totals, slot profile, node assignment, workload, and downstream decision outcomes separately.
Compare a comparable-day baseline, a more detailed model, and the operational plan actually used.
Include ordinary days, peak days, and disruptions, but report each regime separately.

Define error as `forecast - actual` if that is the agreed convention.
Mean signed error identifies persistent over- or underforecasting; mean absolute error measures typical absolute magnitude.
WAPE is `sum(abs(forecast - actual)) / sum(actual)` when the denominator is positive.
It is undefined for an all-zero evaluation set and can conceal poor performance in small nodes.
Compare aggregate errors with per-node and per-slot errors, and avoid unweighted averages of percentages over zero or tiny actuals.

Decision evaluation requires historical constraints and a counterfactual policy, not merely an accuracy score.
Reconstruct the action available at each origin and apply both the candidate and baseline action to the same realized workload.
Compare labor cost, overtime, excess capacity, and missed-work proxy under explicitly stated assumptions.
This replay is an approximation when actions would have changed demand, routing, or productivity.
Use a controlled pilot to establish real operational impact where appropriate.

### 5.11.1 Build a rolling-origin scorecard

The backtest is a sequence of historical decisions, not a random sample of rows.
At each origin, rebuild the information set and predict the next decision window.
The following design illustrates a next-day staffing cycle.

1. At 14:00 on each historical day, freeze available orders, inventory, campaign plans, and staffing constraints.
2. Train or update the model using observations available before that origin.
3. Issue predictions for tomorrow's local operating slots.
4. Store the forecast version and the feasible action set.
5. Wait until the evaluation truth has passed its correction window.
6. Calculate demand and workload errors for each target slot.
7. Replay the proposed and baseline staffing actions against the same observed workload.
8. Summarize results separately for normal days and unusual events.

For example, an origin on Monday at 14:00 forecasts Tuesday.
An origin on Tuesday at 14:00 forecasts Wednesday.
Tuesday's actual traffic cannot be an input to Monday's prediction.
Tuesday's finalized order corrections cannot be used by Tuesday at 14:00 unless they were already available then.
An initial data ingestion followed by a correction needs both availability timestamps.
Without those timestamps, a historical replay may be optimistic.

The final evaluation truth should also have a stable definition.
A late cancellation may change the current status of an order but should not erase picking already performed.
For pick workload, compare with actual work performed or released under the chosen target definition.
For accepted-order forecasting, compare with the count meeting the accepted-event rule.
For parcel reservation, compare with handoff parcels under the carrier definition.
Different evaluation truths are appropriate because the operational decisions differ.

Keep all slots in the scorecard, including zero-actual slots.
An actual zero with a positive prediction contributes a positive absolute error.
Percentage errors at that slot are undefined, not zero.
If the sum of actual demand across the evaluation set is positive, WAPE remains computable.
Report the number of observations and actual volume alongside the score.
A 10% WAPE over 100 orders has a different operational scale from the same score over 1 million orders.

Signed bias is particularly useful for detecting a persistent staffing shortfall.
Suppose three slot forecasts are 90, 110, and 100 orders.
Their actuals are 100, 100, and 110 orders.
Under the convention forecast minus actual, the errors are -10, +10, and -10 orders.
Mean signed error is `(-10 + 10 - 10) / 3 = -3.33 orders per slot`.
Mean absolute error is `(10 + 10 + 10) / 3 = 10 orders per slot`.
WAPE is `30 / 310 = 9.68%`, rounded.
The negative signed error indicates underforecasting on average, not an absence of overforecasted slots.

Next, split the evaluation by operating context.
Measure peak promotion days separately from ordinary weekdays.
Measure large and small nodes separately.
Measure day-ahead forecasts separately from same-day revisions.
Inspect error in the last slot before a carrier cutoff, where timing can matter more than daily totals.
Avoid selecting segments only after finding a favorable result.
State how segments were defined and how many observations each contains.

### 5.11.2 Check probability forecasts and decision cost

A prediction interval is useful only if its meaning and calibration are clear.
Suppose 80 of 100 held-out observations fall within intervals labeled 90%.
Empirical coverage is 80%, not 90%.
That is evidence of undercoverage in this sample, subject to sampling uncertainty and the choice of observations.
Also inspect interval width: an interval spanning all plausible outcomes can have high coverage while providing little planning value.

For a 90% interval, a two-sided calibration check should inspect misses below and above separately.
If nearly every miss is above the upper bound, the forecast may systematically understate peaks.
If misses cluster on campaign days, a global calibration statistic may conceal an event-specific weakness.
Coverage also changes when the horizon changes.
Do not evaluate a day-ahead interval using a same-day updated forecast without identifying the version.

Decision replay adds a cost view to forecast evaluation.
Suppose a slot has 500 actual work minutes.
Plan A provides 460 minutes and plan B provides 520 minutes.
Plan A has 40 minutes of shortfall; plan B has 20 minutes of excess.
If shortfall costs USD 2 per minute and excess costs USD 0.50 per minute, plan A's modeled loss is USD 80 and plan B's is USD 10.
This comparison assumes linear costs and ignores downstream and service effects.
The cost rates must be agreed with operations and finance rather than chosen to make a model look good.

Some consequences are constraints, not convenient dollar penalties.
A contract may require a minimum on-time handoff rate.
An action that minimizes estimated cost but violates the obligation is infeasible.
Evaluate contractual eligibility before ranking remaining actions by cost.
Where the constraint cannot be verified from the data, do not claim the action is compliant.
Escalate to the owner who can assess the actual commitment.

### 5.11.3 Detect drift and choose a fallback

Monitoring begins with source health.
If event ingestion stops, the forecast should not interpret the missing events as a sudden collapse in customer demand.
Check event counts, last-arrival time, completeness by channel, and reconciliation to independent control totals.
Define the tolerance and owner for each alert.
Suppress publication or mark the output degraded when the contract is violated.

Next, monitor changes in the business process.
A new assortment can change lines per order.
A new warehouse can change allocation shares.
A promise change can alter both conversion and node selection.
A campaign calendar can shift the arrival profile.
A model that performed well before these changes may no longer be calibrated.

Finally, monitor outcomes after the truth matures.
Inspect signed errors, absolute errors, coverage, workload mix, and action cost.
Look for sustained patterns, not one isolated noisy interval.
Record the observation window and number of slots behind each alert.
An alert is a reason to investigate; it is not proof that a particular feature caused the drift.

A fallback should be executable without the failed component.
For example, use an approved recent comparable-day profile with a documented peak reserve.
The fallback needs a named owner, issue time, decision window, and expiration.
It must also state what data is still required.
If both the model and the source order feed fail, a fallback relying on that feed is not independent.
Escalate when no reliable basis remains for a customer-facing commitment.

### 5.11.4 Separate forecast quality from override quality

Overrides can contain valuable information not present in the model.
An operations planner may learn of a carrier strike after the forecast was issued.
A campaign manager may postpone a send after the campaign calendar snapshot.
The adjustment should be timestamped and visible to the roster owner.
Record the original forecast as well as the final operational number.

Do not interpret every successful override as proof of better judgment.
The intervention may have changed the eventual outcome.
The planner may have received new information unavailable to the original model.
Compare like horizons and like information sets where possible.
Report override frequency, direction, size, reason, and error before and after.
An override that consistently improves a forecast can motivate adding the relevant information to the automated process.

Conversely, an override that consistently increases error should trigger review.
The reviewer should examine whether the apparent deterioration reflects different decisions rather than just arithmetic.
For example, a deliberate capacity cushion is a plan, not an improved point forecast.
Store that cushion in the plan layer rather than overwriting the demand estimate.
This preserves the distinction between expected demand and chosen protection.

### 5.11.5 Translate the forecast into a shift handoff

A useful shift handoff is concise enough to act upon and detailed enough to audit.
It includes target slots, expected assignments, work minutes, existing backlog, effective capacity, and reserve.
It also states which scenario was chosen and why.
List the forecast issue time and the next refresh time.
Show confidence or scenario limitations without implying guarantees.

For each red or amber slot, name a trigger and an action.
An example trigger is a sustained 20% excess of released work over the current plan for two consecutive 15-minute intervals.
The action might be to call a preapproved reserve team, subject to staffing rules and availability.
The threshold is illustrative; validate it against actual lead times and false-alarm costs.
If a worker needs one hour to arrive, an alert five minutes before cutoff is not actionable.

The shift lead acknowledges receipt and records whether the recommendation is feasible.
The carrier manager confirms whether the downstream handoff can accept the resulting parcels.
The inventory or allocation owner confirms whether rerouting is allowed.
The customer-promise owner approves a change in the offer when it affects a commitment.
After the shift, record actual work, capacity, decisions, and incidents for the next review.

### 5.11.6 Apply the method to three different business settings

**Same-day grocery delivery.**
The decision may concern shopper hours and delivery-slot availability rather than warehouse pick waves.
The relevant workload can depend on item count, substitutions, cold-chain handling, and delivery geography.
The customer-selected slot is itself a capacity constraint.
Closing a slot changes observed demand, so subsequent arrivals are not independent of the previous decision.
Measure both accepted orders and rejected or unavailable-slot attempts when permitted and reliable.

**Marketplace orders fulfilled by independent sellers.**
The marketplace may forecast orders but lack direct control over seller staffing.
An order can be accepted by the platform and then rejected or delayed by a seller.
Forecast seller-level handoff volume and uncertainty separately from the marketplace's customer-order count.
The commercial agreement determines who owns the commitment and what remedies are available.
Avoid treating merchant-reported inventory as guaranteed physical availability without a validation process.

**Direct-to-consumer product launch.**
A launch may have little comparable history and a highly concentrated arrival profile.
Traffic, conversion, stock availability, and purchase limits can change quickly.
The initial forecast should combine a transparent baseline with a range of conditional scenarios.
Reserve capacity only where the underlying labor, material, and transport resources can be secured.
After launch, use frequent updates, but retain the earlier forecast used for the original roster decision.

These examples share a basic accounting structure but not a universal model.
The target event, decision owner, constraints, and workload drivers differ.
Transfer the method by re-specifying those elements.
Do not transfer a coefficient or service assumption merely because both businesses sell online.

### 5.12 Operate, monitor, and revise

Publish the forecast, scenario assumptions, capacity recommendation, and override log together.
The forecaster owns data quality and model diagnostics; the operations manager owns the roster.
The network planner owns routing feasibility; the commercial owner owns campaign timing and terms.
The customer-experience owner approves promise changes, and finance validates cost assumptions.
A named incident lead coordinates when thresholds are crossed.

Monitor data freshness, event volume, allocation mix, feature drift, residual bias, interval coverage, and actual versus planned productive hours.
Use thresholds agreed before deployment, with a human owner and a fallback action.
An override must include its reason, author, timestamp, magnitude, and expiration.
Compare override performance with the original forecast without assuming that the two were exposed to identical decisions.
Retire stale overrides after their stated end time.

## 6. Worked example

### 6.1 Decision and assumptions

A retailer must set Tuesday's 16:00–17:00 local pick shift for two nodes, North and South.
At Monday 14:00, the workforce planner receives a forecast of 2,400 accepted orders for Tuesday.
This is an expected count, not a customer commitment or a target.
All numbers below are synthetic and illustrate the calculation, not a claim about a named company.

The historical comparable-day slot share is 12% for 16:00–17:00.
Under the current allocation policy, 60% of those orders have North as their primary node and 40% have South.
Of all orders, 10% need a second node; for this example, 60% of the split orders start at North and the remainder at South.
Every split order uses exactly the other node as its second node.
No order is unallocated and there are exactly two nodes.

| Input | North | South | Interpretation |
|---|---:|---:|---|
| Primary-node share | 60% | 40% | Share of distinct orders. |
| Lines per assignment | 2.5 | 2.0 | Forecast mean for work at that node. |
| Base pick minutes per assignment | 1.2 minutes | 1.2 minutes | Includes assignment-level work. |
| Extra pick minutes per line | 0.8 minutes | 0.8 minutes | Assumed additive. |
| Starting backlog | 100 work minutes | 50 work minutes | Already in the queue at 16:00. |
| Scheduled pickers | 14 people | 9 people | Shift plan before intervention. |
| Productive availability | 80% | 80% | Fraction of paid slot spent on modeled work. |

Assume all relevant work arrives in this hour and each picker supplies 60 paid minutes in the hour.
Assume the labor standard is expressed in productive minutes, so multiplying paid time by 80% does not double-count breaks.
No transfers or cancellations occur within the hour.
The forecasted means are fractional expectations, not a claim that part of a physical order exists.

### 6.2 Calculate slot arrivals and routing

`2,400 orders/day × 0.12 = 288 orders/slot`.
North primary assignments are `288 × 0.60 = 172.8 assignments`.
South primary assignments are `288 × 0.40 = 115.2 assignments`.
Primary assignments sum to 288, the distinct-order expectation.

Expected split orders are `288 × 0.10 = 28.8 orders`.
Of these, `28.8 × 0.40 = 11.52` start at South and add a North assignment.
The other `28.8 × 0.60 = 17.28` start at North and add a South assignment.
North therefore receives `172.8 + 11.52 = 184.32 assignments`.
South receives `115.2 + 17.28 = 132.48 assignments`.
The total is `316.80 assignments = 288 + 28.8`, as the split assumption requires.
Do not call 316.80 the number of customers or distinct orders.

### 6.3 Calculate work and backlog

North's expected standard minutes per assignment are `1.2 + 2.5 × 0.8 = 3.2 minutes`.
Its incoming work is `184.32 × 3.2 = 589.824 work minutes`.
North capacity is `14 × 60 × 0.80 = 672 productive minutes`.
Including starting backlog gives `100 + 589.824 = 689.824 work minutes`.
Ending backlog is `max(0, 689.824 - 672) = 17.824 work minutes`.

South's expected standard minutes per assignment are `1.2 + 2.0 × 0.8 = 2.8 minutes`.
Its incoming work is `132.48 × 2.8 = 370.944 work minutes`.
South capacity is `9 × 60 × 0.80 = 432 productive minutes`.
Including backlog gives `50 + 370.944 = 420.944 work minutes`.
Ending backlog is `max(0, 420.944 - 432) = 0 work minutes`.
South has `432 - 420.944 = 11.056 productive minutes` of unused modeled capacity.

The network has an aggregate modeled deficit of 6.768 minutes, and capacity is not necessarily interchangeable across sites.
North's deficit cannot be cured by simply subtracting South's unused capacity unless work, stock, travel time, and promise constraints allow transfer.
Even if transfer is feasible, 11.056 minutes is less than North's 17.824-minute deficit.
A network total can therefore hide a local queue.

### 6.4 Decide under a campaign scenario

Suppose the commercial owner proposes a campaign with a planning assumption of 15% more orders in this slot, without changing primary shares, split rate, basket mix, or starting backlog.
The 15% is a scenario assumption, not an estimated causal uplift or calibrated percentile.
North's incoming work becomes `589.824 × 1.15 = 678.2976 minutes`.
With starting backlog, North requires `778.2976 minutes`, producing `106.2976 minutes` of backlog at the existing capacity.
South's incoming work becomes `370.944 × 1.15 = 426.5856 minutes`.
With backlog, South requires `476.5856 minutes`, producing `44.5856 minutes` of backlog.

Each additional available picker supplies `60 × 0.80 = 48 productive minutes` in the slot.
North needs `ceil(106.2976 / 48) = 3` additional pickers to clear this modeled scenario in the hour.
South needs `ceil(44.5856 / 48) = 1` additional picker.
If labor costs USD 24 per paid hour per picker, the incremental planned cost is `4 × USD 24 = USD 96` for the hour.
This excludes onboarding, minimum shift lengths, overtime premiums, and downstream bottlenecks.

The staffing owner should check whether four qualified people can actually be scheduled and whether packing and carrier capacity are also available.
If not, the campaign owner can consider a different send time or volume, subject to approval and evidence about the response.
The promise owner must approve any change to advertised delivery commitments.
The backlog calculation does not itself prove a late delivery, because it lacks individual order deadlines and later capacity.
A responsible recommendation is to reserve the additional capacity conditionally and monitor actual arrivals and work mix.

### 6.5 Interpret the recommendation as a decision

The Monday forecast is an estimate of Tuesday demand.
The recommendation to reserve four extra pickers is a conditional plan.
The campaign increase is a scenario, not an assured outcome.
The customer's delivery promise remains a commitment until an authorized owner changes it.
These four statements should appear separately in the shift briefing.

The workforce planner must check the notice period for extra labor.
Four people might be available for an hour mathematically but have a four-hour minimum shift.
At the same USD 24 per paid hour and no premium, a four-hour minimum would cost USD 384 rather than USD 96.
That comparison assumes all four workers must be booked for the full four hours.
If the extra time can be used productively elsewhere, its net opportunity cost requires a separate assessment.
Do not present an hour's modeled cost as a binding supplier quote.

The operations lead should also verify skills.
If North's work requires equipment certification, an available but unqualified worker cannot provide the assumed 48 productive minutes.
If the extra workers need supervision or training, their effective availability may be lower than 80%.
If the available workstations are already saturated, headcount may not increase throughput at all.
The capacity owner must validate those conditions before accepting the recommendation.

The network planner can examine whether North work is eligible for South.
The existing South base-case slack is only 11.056 productive minutes.
Even if every minute of that slack could be used on North-origin work, the network would still have 6.768 work minutes of aggregate base-case deficit.
Any transfer calculation must use the receiving node's own work standard.
It must also account for inventory, transport time, and the customer's destination.
The simple subtraction is only a diagnostic of aggregate capacity, not a feasible routing plan.

### 6.6 Examine a change in basket mix

The campaign scenario assumed that the number of lines per assignment stays constant.
Now consider a separate sensitivity case in which the campaign's North assignments average 2.8 lines rather than 2.5.
All other assumptions remain those of the campaign scenario.
This is a conditional stress case, not an assertion that the campaign causes a mix shift.

North's campaign assignment expectation is `184.32 × 1.15 = 211.968 assignments`.
With 2.8 lines per assignment, expected work per assignment is `1.2 + 2.8 × 0.8 = 3.44 minutes`.
Incoming work is `211.968 × 3.44 = 729.16992 minutes`.
Including 100 minutes of starting backlog gives `829.16992 minutes`.
Subtracting 672 minutes of scheduled capacity leaves `157.16992 minutes`.
At 48 productive minutes per additional picker, clearing this workload takes `ceil(157.16992 / 48) = 4` extra North pickers.

The original campaign scenario needed three extra North pickers.
The higher-mix scenario needs four, even though its expected order count is unchanged.
The additional need arises from work per assignment, not from an order forecast error.
This sensitivity identifies a useful early indicator: observed lines per assignment during the campaign.
The shift lead should compare that indicator with the assumed 2.5 lines before the extra worker call deadline.

South's assumptions are unchanged in this separate case.
It still needs one additional picker to clear its modeled deficit.
The total reserve is therefore five pickers, costing USD 120 for one paid hour at the simplified rate.
This is USD 24 more than the four-picker campaign plan.
It excludes the same minimum-shift and downstream costs as the original example.

### 6.7 Examine backlog in the following slot

Assume the original schedule is used in the base case, without extra pickers.
North carries 17.824 work minutes into the next hour.
Suppose the next hour has 620 incoming work minutes and the same 672-minute effective capacity.
The ending backlog becomes `max(0, 17.824 + 620 - 672) = 0 minutes`.
Under these assumptions the next hour absorbs the modeled backlog.

That calculation does not prove that the delayed orders met their deadlines.
If the carrier cutoff occurs at 17:00, work completed during 17:00–18:00 can miss that handoff.
If the carrier cutoff is later and the orders remain eligible, the temporary queue may have no delivery effect.
Actual priority rules matter: a first-in-first-out queue and a deadline-priority queue can produce different late-order counts.
The single aggregate queue does not identify which orders are delayed.

Suppose instead that the next hour receives 700 work minutes.
The ending backlog is `17.824 + 700 - 672 = 45.824 minutes`.
The queue grows by 28 minutes despite the preceding hour having only a small deficit.
Repeated small deficits can compound over a shift.
This is why a capacity review should examine the full sequence of slots, not isolated peak cells.

The process owner should reconcile the starting backlog at each slot boundary.
An order canceled before release can be removed from future work.
An order canceled after picking may still have consumed the pick effort already recorded.
An order rerouted elsewhere leaves the original queue only when the handoff is confirmed.
Each adjustment needs an event time and a reason code to avoid losing or double-counting work.

### 6.8 Close the loop after the shift

At the end of Tuesday, compare actual accepted orders with the 2,400-order daily forecast.
Compare the actual 16:00–17:00 share with the assumed 12%.
Compare North and South assignment counts with 184.32 and 132.48 base expectations, or with the campaign expectations if the campaign was actually run.
Record whether the chosen baseline includes the campaign before computing the error.
Use counts as realized observations and retain fractional forecasts as expectations.

Next compare actual line mix and measured pick work with the standards.
An accurate assignment forecast can still understate effort if the line mix changed.
An accurate workload estimate can still produce backlog if availability was below 80%.
An accurate capacity estimate can still miss a promise if carrier departure moved earlier.
Separating those mechanisms makes the postmortem useful.

The shift lead records whether the recommended reserve was booked and whether it worked.
The commercial owner records the campaign's actual timing and exposure.
The network owner records allocation changes and stock-related exceptions.
Finance records paid hours and actual premiums rather than substituting the simplified hourly estimate.
The forecaster joins these observations to the archived forecast version after the truth freeze.

If the campaign was canceled, do not score its conditional scenario as though it were the realized business-as-usual forecast.
Instead compare the observed day against the no-campaign forecast and retain the unused scenario as a planning artifact.
If the campaign ran but eligibility or timing differed from the plan, record the difference.
The discrepancy may explain error, but it does not by itself establish causality.
Feed the finding into the next intervention-design and forecast-review cycle.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Save it as `order_workload.py` and run `python order_workload.py`.
It reproduces the expected base and campaign calculations above.
All inputs are deterministic expected values; it is not a probabilistic forecast or a live data pipeline.

```python
from math import ceil

orders_per_day = 2400.0
slot_share = 0.12
primary_shares = {"North": 0.60, "South": 0.40}
split_rate = 0.10
lines_per_assignment = {"North": 2.5, "South": 2.0}
starting_backlog = {"North": 100.0, "South": 50.0}
pickers = {"North": 14, "South": 9}
availability = 0.80
slot_minutes = 60.0
base_minutes = 1.2
line_minutes = 0.8
hourly_cost_usd = 24.0


def calculate(uplift):
    """Return workload and staffing results for the two-node scenario."""
    slot_orders = orders_per_day * slot_share * (1.0 + uplift)
    primary = {node: slot_orders * share
               for node, share in primary_shares.items()}
    # Each split has exactly one additional assignment at the other node.
    assignments = {
        "North": primary["North"] + primary["South"] * split_rate,
        "South": primary["South"] + primary["North"] * split_rate,
    }
    result = {}
    for node in ("North", "South"):
        minutes_per_assignment = (
            base_minutes + lines_per_assignment[node] * line_minutes
        )
        incoming = assignments[node] * minutes_per_assignment
        capacity_per_picker = slot_minutes * availability
        capacity = pickers[node] * capacity_per_picker
        deficit = max(0.0, starting_backlog[node] + incoming - capacity)
        extra_pickers = ceil(deficit / capacity_per_picker)
        result[node] = (assignments[node], incoming, capacity,
                        deficit, extra_pickers)
    return slot_orders, result


for name, uplift in (("Base", 0.0), ("Campaign", 0.15)):
    orders, rows = calculate(uplift)
    print(f"{name}: {orders:.1f} distinct orders")
    total_extra = 0
    for node, (assignments, work, capacity, backlog, extra) in rows.items():
        total_extra += extra
        print(f"  {node}: assignments={assignments:.2f}, "
              f"work={work:.3f} min, capacity={capacity:.0f} min, "
              f"backlog={backlog:.3f} min, extra={extra}")
    print(f"  incremental one-hour cost: USD {total_extra * hourly_cost_usd:.2f}")
```

Expected output, with rounding applied only for display:

```text
Base: 288.0 distinct orders
  North: assignments=184.32, work=589.824 min, capacity=672 min, backlog=17.824 min, extra=1
  South: assignments=132.48, work=370.944 min, capacity=432 min, backlog=0.000 min, extra=0
  incremental one-hour cost: USD 24.00
Campaign: 331.2 distinct orders
  North: assignments=211.97, work=678.298 min, capacity=672 min, backlog=106.298 min, extra=3
  South: assignments=152.35, work=426.586 min, capacity=432 min, backlog=44.586 min, extra=1
  incremental one-hour cost: USD 96.00
```

The base case also requires one extra North picker to clear its modeled deficit within the hour.
The campaign's USD 96 is the cost of four extra pickers relative to the original schedule, not the incremental cost relative to the one-extra-picker base response.
Relative to that base response, the additional modeled cost is USD 72 for the hour.

## 8. Data, standards, and regulatory context

A minimum dataset joins order events, order lines, node assignments, release and completion events, inventory eligibility, staffing, and carrier handoff events.
A useful order event includes a stable pseudonymous order key, event type, event timestamp, ingestion timestamp, channel, and source-system version.
A line table includes SKU, unit quantity, and product handling class as they were known at the relevant time.
An allocation table needs effective timestamps and a history of reassignment, not just the current destination node.
A labor table needs paid time, productive availability, role, and the definition of the standard used.

Data contracts should specify keys, uniqueness, late-arrival tolerances, zero versus missing semantics, and correction rules.
For example, an assignment table may be unique by order, node, and allocation version rather than by order alone.
A carrier handoff event must distinguish a label creation from a physical scan.
Reconcile totals across order management, warehouse management, and carrier systems, acknowledging that their event definitions differ.
Data lineage should make it possible to reproduce the forecast's inputs as of its issue time.

GS1 identifiers such as the Global Trade Item Number (GTIN) can identify trade items, while Global Location Numbers (GLNs) can identify locations and parties in participating systems.
They do not define your order-forecasting target, guarantee inventory accuracy, or replace a versioned internal node mapping.
GS1 Electronic Product Code Information Services (EPCIS) provides a standard for sharing visibility event data; using it does not guarantee all events arrive promptly or have the same business meaning.
Adoption and contractual requirements depend on the trading relationship and jurisdiction.
See the official GS1 references in section 16.

Time-zone rules are operational data, not a cosmetic display setting.
Retain UTC timestamps and the relevant IANA time-zone identifier, and version the conversion rules used to construct training data.
Some business calendars contain regional holidays and movable events; document which calendar applies to each node and destination.
An apparent demand spike can be a duplicate repeated hour if offsets were discarded.

Customer addresses and order histories can be personal data.
The EU General Data Protection Regulation (GDPR), where applicable, includes principles of purpose limitation, data minimisation, accuracy, storage limitation, and security in Article 5.
It does not prescribe a forecasting algorithm or a universal retention period.
Use a lawful basis assessed by the responsible organization, restrict access, aggregate geography where practical, and set retention according to applicable obligations and documented purpose.
Pseudonymisation reduces risk but does not automatically make data anonymous.
This educational material is not legal advice; counsel should assess applicable law and cross-border transfers.

For workforce data, minimize identifiable employee detail in forecasting outputs and separate performance management from aggregate capacity analysis where feasible.
Check local employment rules and collective agreements before changing rosters or collecting detailed activity data.
A statistical forecast is not itself authorization to change a worker's contractual hours.

## 9. Industry practice and case evidence

Amazon's 2012 anticipatory package shipping patent describes techniques for moving packages toward destinations before a customer order is finalized.
A patent is evidence of a proposed system, not evidence that a particular facility uses it or achieved a particular saving.
Its relevance here is conceptual: expected geographic demand can influence where inventory or packages are positioned before demand is fully observed.
That decision differs from the short-horizon labor forecast and requires its own costs and service constraints.

Walmart's public descriptions of its fulfillment network discuss stores, fulfillment centers, and delivery as components of an omnichannel system.
The public materials establish that multiple fulfillment assets exist, not the retailer's internal node-level forecast accuracy or allocation algorithm.
[INFERENCE] A network with multiple fulfillment paths needs consistent definitions of order, fulfillment location, and capacity to compare alternative routing plans.
Do not treat this inference as a disclosed Walmart operating rule.

The M5 forecasting competition used Walmart unit-sales data and evaluated hierarchical forecasts across products and stores.
Its published dataset and competition show the practical importance of hierarchical retail demand forecasting.
It does not provide order-arrival timestamps, basket splits, picker workloads, or proof that its winning model is best for a particular fulfillment shift.
Use its hierarchy and evaluation lessons cautiously rather than substituting item sales for order-level workload.

Retailers often face promotional and holiday peaks, but historical peak performance should be compared with contemporaneous capacity, intervention, and promise conditions.
A public company statement that it increased network capacity is not a causal evaluation of a specific forecast.
A credible internal case should record baseline, counterfactual, dates, sample size, measurement definitions, costs, and service outcomes.
Absent these facts, report only the documented operational change and label proposed explanations as hypotheses.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Lines counted as orders | Lines per order unexpectedly equals one | Join duplicates order records | Overstaffing and false accuracy | Reconcile distinct order keys before aggregation. |
| Missing slot treated as zero | Abrupt identical zero across channels | Pipeline outage | Understaffing | Use completeness checks and a fallback. |
| Late correction leakage | Backtest accuracy far better than live | Future-corrected data used as a feature | False confidence | Replay as-of snapshots. |
| Time-zone error | Shift spikes at clock changes | Local hour mapped without offset | Wrong shift roster | Retain UTC and zone-aware slots. |
| Censored demand | Orders fall when stock is unavailable | Customers cannot order | Depressed baseline | Model availability and flag unidentifiable lost demand. |
| Stale routing shares | Node mix changes after policy rollout | Historical shares embed old policy | Local overload | Simulate current feasible rules. |
| Split ignored | Parcel or assignment counts exceed order counts | One order reaches several nodes | Underestimated work | Model order-node assignments separately. |
| Mix ignored | Orders accurate but pick minutes high | More lines or special handling | Missed cutoffs | Forecast class mix and work standards. |
| Completion used as demand | Output plateaus while backlog grows | Capacity caps completed work | Hidden overload | Forecast offered work and observe queues. |
| Double-counted breaks | Planned capacity consistently too low | Availability and labor rate both include breaks | Excess cost | Document numerator and denominator of productivity. |
| Additive marginal quantiles | Every node assigned its own extreme | Correlations ignored | Misstated network risk | Use coherent joint scenarios. |
| Campaign double counted | Forecast exceeds plausible scenario | Event already in baseline plus extra uplift | Overstaffing | Record baseline conditioning and intervention version. |
| False causal uplift | Before-after comparison ignores season | Concurrent changes confound effect | Misallocated campaign and labor | Use randomized or defensible controlled analysis. |
| Forecast after decision | Accuracy improves but roster unchanged | Publication misses planning deadline | No decision value | Measure timeliness and forecast value at action cutoff. |
| Unlogged override | Repeated manual edits lack reasons | Accountability and expiry missing | Untraceable bias | Record author, reason, version, and expiration. |
| Local capacity assumed movable | Network total appears sufficient | Stock and travel constrain transfer | Node backlog | Check feasibility and customer promise per route. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| Accepted-order definition is stable | Comparable target series | Fraud or payment rule change | Audit event definitions and bridge versions. |
| Origin snapshots are complete | Reproducible backtests | Late-arriving data or overwritten plans | Measure ingestion lag and preserve snapshots. |
| Comparable days are informative | Baseline and slot shares | New campaign, holiday, or outage | Compare regimes and hold out event days. |
| Node allocation follows stated rule | Assignment forecast | Inventory or policy changes | Compare forecast and actual allocation by reason. |
| Split probability is representative | Assignment totals | New assortment or stock fragmentation | Track splits by basket and destination. |
| Work standards transfer to forecast mix | Convert orders into minutes | Batch size, travel, or special handling changes | Compare engineered and observed time by class. |
| Productive availability is known | Translate headcount to capacity | Absence, training, or equipment downtime | Track actual productive time and scenarios. |
| Starting backlog is measured | Queue arithmetic | Unrecorded holds or late scans | Reconcile open work at each boundary. |
| Within-slot arrivals are pooled | Fluid queue approximation | Cutoffs and arrival order matter | Use order-level simulation and cutoff timestamps. |
| Scenario uplift is external to baseline | Avoid double counting | Baseline already includes the campaign | Version and inspect feature conditioning. |
| Forecast errors resemble evaluation period | Interval coverage | Regime shift or unprecedented peak | Monitor coverage by regime and add stress cases. |
| Alternatives do not change demand | Simple replay comparison | Promise or availability changes conversion | Use causal evidence or label replay conditional. |

The numerical example assumes homogeneous labor and a single pick bottleneck.
It does not model individual service deadlines, downstream packing, carrier loading, breaks within the hour, or stochastic dependence.
A good point estimate can coexist with poor tail coverage and missed promises.
Use it as a transparent planning baseline, not as a guarantee of service.

## 12. Exercises

### Exercise 1 — Define the grain

A customer places one order with three lines and four units.
Two lines are routed to North and one line to South; North sends two parcels and South sends one.
Deliverable: count orders, lines, units, order-node assignments, and parcels.
Acceptance: all five counts are correct and the two split concepts are distinguished.
Time: 8 minutes.

### Exercise 2 — Design the decision contract

A carrier requires a parcel reservation at 15:00 for tomorrow; the warehouse roster closes at 12:00.
Deliverable: a two-row contract naming owner, target, horizon, action deadline, and fallback for each decision.
Acceptance: the two targets and deadlines are distinct, and neither forecast uses future actuals.
Time: 12 minutes.

### Exercise 3 — Reconcile the hierarchy

Forecast 1,250 accepted orders for a day; the relevant slot share is 8%.
North is primary for 70% of slot orders; 20% of orders split to the other of two nodes.
Deliverable: slot orders, primary assignments, all-node assignments, and reconciliation totals.
Acceptance: show the arithmetic and do not equate all-node assignments to distinct orders.
Time: 12 minutes.

### Exercise 4 — Translate workload into capacity

A node expects 160 assignments, each requiring 1.0 base minute plus 2.25 lines at 0.8 minute per line.
Starting backlog is 40 work minutes; ten pickers each provide 60 paid minutes at 75% productive availability.
Deliverable: incoming work, capacity, ending backlog, and minimum extra pickers to clear the slot.
Acceptance: units and the rounding of headcount are explicit.
Time: 12 minutes.

### Exercise 5 — Critique an intervention estimate

A promotion starts on a holiday Monday and orders rise 30% versus the prior ordinary Monday.
Inventory availability also improves and the delivery promise shortens.
Deliverable: identify three threats to a causal interpretation and design an improved comparison.
Acceptance: distinguish an observational estimate from an experimental effect and include a measurement plan.
Time: 15 minutes.

### Exercise 6 — Backtest correctly

A model uses realized same-day site traffic to predict tomorrow's shift, but the roster is fixed the previous afternoon.
Deliverable: propose a rolling-origin evaluation and name two metrics and one operational outcome.
Acceptance: all features are available at each origin, and one metric addresses bias or calibration.
Time: 15 minutes.

### Exercise 7 — Recommend an action

A forecast indicates a 20% chance of exceeding node capacity by at least 120 work minutes.
Three additional pickers each provide 48 productive minutes; the node can book them at USD 24 each for the hour.
Deliverable: recommend an action or an information-gathering gate, naming decision owners and one alternative.
Acceptance: calculate reservation cost, explain what cannot be inferred from the single probability, and protect the service commitment.
Time: 15 minutes.

### Exercise 8 — Privacy and data quality

A proposed dataset contains full customer names, addresses, order IDs, node events, and exact timestamps indefinitely.
Deliverable: propose a minimal forecasting dataset, retention decision process, and two quality controls.
Acceptance: distinguish pseudonymous from anonymous data and identify the owner of the legal assessment.
Time: 15 minutes.

## 13. Self-check answers

### 1. Grain

There is one distinct order, three lines, four units, two order-node assignments, and three parcels.
The order is split across nodes, and its North portion also creates more than one parcel.
The number of parcels need not equal either the number of nodes or the number of orders.

### 2. Contract rubric

The staffing lead uses a next-day workload forecast by node and shift before 12:00 and falls back to an approved comparable-day roster plus escalation.
The carrier manager uses a next-day parcel forecast by carrier and handoff window before 15:00 and falls back to a documented reserve or renegotiation.
A strong answer defines the version and units and treats a reservation as an action rather than a forecast.

### 3. Hierarchy

The slot has `1,250 × 0.08 = 100` expected distinct orders.
North primary assignments are 70 and South primary assignments are 30.
There are `100 × 0.20 = 20` expected split orders and hence 20 extra assignments.
North receives `70 + 30 × 0.20 = 76` assignments.
South receives `30 + 70 × 0.20 = 44` assignments.
The total is 120 assignments, which reconciles to 100 distinct orders plus 20 second-node assignments.

### 4. Work and capacity

Minutes per assignment are `1.0 + 2.25 × 0.8 = 2.8`.
Incoming work is `160 × 2.8 = 448 work minutes`.
Available capacity is `10 × 60 × 0.75 = 450 productive minutes`.
Ending backlog is `max(0, 40 + 448 - 450) = 38 work minutes`.
One additional picker supplies 45 productive minutes, so one additional picker is sufficient in this fluid model.

### 5. Intervention rubric

The holiday, availability improvement, and shorter delivery promise are three concurrent differences.
The comparison also has only one prior Monday and may be affected by noise and seasonality.
A stronger design randomizes promotion eligibility where feasible, measures eligible orders and inventory by group, and checks interference and promise exposure.
If randomization is unavailable, use a defensible control and pre-period diagnostics; describe remaining confounding and uncertainty.
A before-after change is an observed association, not automatically the campaign's causal effect.

### 6. Backtest rubric

At each historical roster cutoff, freeze the available traffic forecast, planned campaign, calendar, inventory view, and other inputs.
Predict the next day's relevant slots and repeat across multiple origins.
Report mean signed error and WAPE, with the latter undefined if total actuals are zero; add empirical interval coverage if intervals are produced.
Translate predicted and actual work into capacity shortfall or cost under the same feasible staffing policy.
Do not use same-day realized traffic as if known the prior afternoon.

### 7. Action rubric

Three pickers provide `3 × 48 = 144 productive minutes` and cost `3 × USD 24 = USD 72` for the hour.
They cover a 120-minute shortfall in the simplified calculation if available and qualified.
The 20% figure does not describe the size of all larger shortfalls, the cost of missed service, or the full distribution.
The staffing lead should compare a reservation or contingent call with rerouting or a campaign change, and the network and promise owners must validate feasibility.
A recommendation must state the decision deadline, service risk, cost assumptions, and escalation if reserve cannot be secured.

### 8. Privacy and quality rubric

Use pseudonymous order keys, coarse destination geography where adequate, event and node timestamps, line quantities, relevant product classes, and allocation history.
Do not retain names or full addresses in the forecasting mart merely because a source system has them.
Set retention through a documented purpose, applicable law, and business need with the privacy or legal owner; do not invent a universal duration.
Pseudonymous records can remain personal data if reidentification is possible.
Controls can include uniqueness checks, as-of freshness checks, source total reconciliation, and missing-slot alerts.

## 14. Interview questions

1. **Why forecast orders rather than units?** Strong answers distinguish pick, pack, and carrier drivers; weak answers assume a fixed basket.
2. **How do you reconcile orders and node work?** Strong answers separate primary assignments and splits; weak answers sum node counts and call them customers.
3. **What is an as-of backtest?** Strong answers recreate information available at the decision time; weak answers use final corrected inputs indiscriminately.
4. **How would you handle a missing hour?** Strong answers distinguish clock transitions from data outages and preserve offsets; weak answers fill every missing slot with zero.
5. **How does a new routing policy affect the forecast?** Strong answers recompute feasible assignments with inventory and promises; weak answers assume old shares remain causal.
6. **What is the difference between capacity and throughput?** Strong answers recognize productive time, bottlenecks, and backlog; weak answers equate historical completions with unconstrained demand.
7. **How do you model campaign uplift?** Strong answers distinguish causal evidence from judgment and avoid double counting; weak answers extrapolate a single before-after percentage.
8. **Why might a better WAPE not improve operations?** Strong answers discuss timing, tails, node mix, and decision cost; weak answers treat one aggregate score as the objective.
9. **Can you add node 95th percentiles?** Strong answers explain dependence and joint scenarios; weak answers assert the sum is always the network 95th percentile.
10. **What does an override require?** Strong answers name owner, evidence, timestamp, expiry, and comparison; weak answers allow undocumented permanent edits.

## 15. Key terms

- [Accepted order](#4-definitions-and-vocabulary): eligible order under the stated rule.
- [Allocation](#56-forecast-node-routing-and-splits): assignment to fulfillment nodes.
- [As-of data](#53-construct-the-point-in-time-dataset): information available at issue time.
- [Backlog](#58-convert-workload-into-a-capacity-decision): unfinished work carried forward.
- [Baseline](#54-establish-a-defensible-daily-baseline): simple comparison forecast.
- [Basket](#57-translate-demand-to-workload): lines and units in an order.
- [Capacity](#58-convert-workload-into-a-capacity-decision): available productive effort.
- [Commitment](#1-why-this-topic-exists): accepted service obligation.
- [Cutoff](#4-definitions-and-vocabulary): operational deadline for an event.
- [Forecast origin](#511-backtest-as-it-would-have-been-used): issue time of a forecast.
- [Horizon](#51-start-with-the-decision-contract): lead time to the target.
- [Intervention](#510-evaluate-interventions-without-confusing-them-with-demand): controllable operational or commercial change.
- [Node](#56-forecast-node-routing-and-splits): fulfillment location or destination.
- [Order line](#4-definitions-and-vocabulary): product and quantity within an order.
- [Order-node assignment](#62-calculate-slot-arrivals-and-routing): one order's work at one node.
- [Override](#512-operate-monitor-and-revise): recorded human forecast adjustment.
- [Plan](#1-why-this-topic-exists): chosen operational action.
- [Primary node](#56-forecast-node-routing-and-splits): one designated initial node per order in the example.
- [Productive availability](#58-convert-workload-into-a-capacity-decision): fraction of paid time available for modeled work.
- [Quantile](#59-quantify-uncertainty-and-dependence): threshold with a specified cumulative probability.
- [Reconciliation](#55-disaggregate-daily-demand-into-slots): enforcement of valid aggregation identities.
- [Scenario](#64-decide-under-a-campaign-scenario): conditional set of assumptions.
- [Slot](#52-define-the-event-and-grain): nonoverlapping operational time interval.
- [Split order](#56-forecast-node-routing-and-splits): order fulfilled by multiple nodes or shipments under a declared definition.
- [WAPE](#511-backtest-as-it-would-have-been-used): weighted absolute percentage error.
- [Workload](#57-translate-demand-to-workload): processing effort required by arrivals.

## 16. Sources

- [GS1: Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin) — item identification; not an order-count standard.
- [GS1: Global Location Number](https://www.gs1.org/standards/id-keys/gln) — identification of locations and parties.
- [GS1: EPCIS](https://www.gs1.org/standards/epcis) — event-sharing standard and context.
- [IANA Time Zone Database](https://www.iana.org/time-zones) — time-zone history and rule data.
- [EU GDPR, Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj) — Article 5 principles and broader obligations, subject to applicability.
- [Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*, third edition](https://otexts.com/fpp3/) — forecast evaluation, hierarchical forecasting, and prediction intervals.
- [Makridakis et al., “The M5 accuracy competition: Results, findings and conclusions,” *International Journal of Forecasting*](https://doi.org/10.1016/j.ijforecast.2021.11.013) — retail hierarchical forecasting competition.
- [Amazon, US patent 8,615,473, “Method and system for anticipatory package shipping”](https://patents.google.com/patent/US8615473B2/en) — patent disclosure, not proof of deployment or results.
- [Walmart, Supply Chain](https://corporate.walmart.com/about/supply-chain) — company description of its network; company source, not independent impact evaluation.

## 17. Links

- [Module 21 index](_index.md)
- [Module 21 survey](../../../course/21_ecommerce_supply_chain_analytics.md)
- [Fulfillment models](01_ecommerce-fulfillment-models.md)
- [Long-tail forecasting](03_long-tail-forecasting.md)
- [Inventory placement](04_inventory-placement.md)
- [Delivery promise](05_delivery-promise.md)
- [Peak planning](08_peak-planning.md)
- [Forecasting foundations survey](../../../course/07_time_series_forecasting.md)
- [Simulation and optimization labs](../../../course/30_simulation_optimization_production_labs.md)
