# The Delivery Promise

> Module 21 — E-commerce Supply Chain Analytics  
> Topic 05 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 07, 10, and 12; basic probability, time arithmetic, and contribution-margin analysis  
> Version 2026.5

## 1. Why this topic exists

A customer deciding whether to buy sees a delivery date, a price, and perhaps a range of options.
The customer does not see the inventory balances, warehouse queues, carrier handoffs, and contingency assumptions behind them.
A delivery promise turns that hidden operating system into a customer-facing commitment.
It also changes the demand that the system must handle.

A merchant that advertises an attractive date may increase conversion.
The same date may force expensive expedited transportation, extra labor, or duplicated inventory.
A merchant that displays a conservative date may protect service but lose orders it could have delivered profitably.
Neither the fastest option nor the highest on-time percentage is automatically the best business decision.

The decision is especially difficult because the forecast of arrival is not the same thing as the commitment shown to a customer.
A forecast describes possible outcomes and their probabilities.
A plan specifies the intended actions and resources.
A target expresses the desired performance level.
A commitment is the date or window communicated under a defined customer policy.
The four objects should be recorded separately.

Consider an order placed on a Friday afternoon.
Stock may be visible at two warehouses, but only one may have enough unreserved units for the complete basket.
A carrier pickup may already have departed from that warehouse.
The carrier's Saturday service may not cover the customer's postcode.
A national average transit time hides all three facts.
The product page's next-day message can therefore conflict with the actual checkout decision.

The problem spans several functions.
A product owner defines what the customer is shown and when the promise becomes binding under company policy.
The inventory owner defines which stock can be committed.
The warehouse owner supplies processing capacity and cutoffs.
The transportation owner validates lanes, capacity, calendars, and rates.
Analytics estimates uncertainty and measures outcomes.
Finance determines the contribution boundary and economic approval threshold.
Legal and customer-care owners review claims, remedies, and notifications.
An accountable promise owner reconciles these inputs and can suspend an unsafe promise.

This topic develops the decision from a single eligible order to an experiment and an operating control loop.
The worked data are fictional; no merchant performance is implied.
All monetary examples use U.S. dollars (USD), and dates and time zones are stated where relevant.
Legal discussion is educational and is not legal advice.

## 2. Learning outcomes

After this lesson you can:

1. **Distinguish** a delivery forecast, operational plan, service target, and customer commitment.
   Exercise 1 asks you to classify all four.
2. **Construct** a feasible order-to-delivery path using stock, capacity, calendars, and an explicit deadline.
   Exercise 2 tests the cutoff and calendar arithmetic.
3. **Estimate** the probability of meeting a deadline from joint historical or simulated outcomes.
   Exercise 3 tests a probability and a conditional fallback.
4. **Choose** a promise and fulfillment option subject to a service threshold and a comparable cost boundary.
   Exercise 4 asks for the feasible minimum-cost choice.
5. **Calculate** on-time delivery and completion metrics with explicit cohorts, denominators, and censoring treatment.
   Exercise 5 tests the measurement definitions.
6. **Evaluate** a promise-change business case using incremental contribution per eligible session.
   Exercise 6 tests the worked economics and the break-even threshold.
7. **Design** a randomized test and operational guardrails that account for interference and capacity constraints.
   Exercise 7 requires a test and rollback decision.
8. **Specify** customer communication, governance, and exception handling for a delayed order.
   Exercise 8 asks for an owner-based control plan.

Successful work states the decision point, population, unit, horizon, and uncertainty.
A result that cannot be executed against current inventory and capacity is a forecast, not an executable commitment.

## 3. The core mental model

### 3.1 The promise is a decision under uncertainty

```text
Customer context: basket, destination, request time, displayed options
                         |
                         v
Available stock + capacity + service calendars + restrictions
                         |
                         v
             Eligible fulfillment paths
             node -> handoff -> carrier -> delivery
                         |
                         v
     Joint arrival distribution + complete path economics
                         |
                         v
  Service, policy, capacity and commercial decision rules
              /                          \
             v                            v
  Customer-facing date and price      Reserved operating plan
              \                          /
               v                        v
                  Order accepted and logged
                         |
                         v
      Pick -> handoff -> transit -> delivery or exception
                         |
                         v
  Customer notification, remedy, measurement and learning
                         |
                         +------> update forecasts and controls
```

The path is conditional on what is known at the decision time.
The promise engine should not use a later inventory snapshot to claim that the original promise was feasible.
The time of a customer-facing display can differ from the time of order acceptance.
A page-view estimate may have less precise location and basket information than checkout.
If either changes, the system must show the refreshed terms before the customer accepts them.

A path consists of a source or set of sources, processing steps, transport services, and a delivery calendar.
Each path has feasibility constraints, a cost, and an uncertain arrival time.
The path's feasibility must be checked before ranking it by expected cost.
An infeasible next-day option cannot be rescued by averaging it with a feasible two-day option.

The complete delivery time is the composition of order release, warehouse work, handoff, transit, and final delivery.
The composition depends on opening hours, queue state, capacity limits, and local holidays.
These components can be correlated: high order volume can slow both warehouse handling and carrier collection.
Adding separately estimated high-percentile durations does not generally give the percentile of the total.
Likewise, multiplying marginal on-time rates assumes independence that may not hold.

The customer commitment has a contractual and experiential meaning beyond a model score.
A high predicted probability does not guarantee the delivery of any specific order.
An explicit service threshold is a policy choice about tolerated risk, not a mathematical guarantee.
The customer receives the communicated date or range, not the hidden percentile used to set it.

### 3.2 A promise changes its own operating environment

A faster date can attract more orders, shift customers between shipping options, and change basket mix.
Those orders compete for the same inventory, warehouse slots, and carrier space.
A model calibrated before a large change may understate the delay caused by the change itself.
The business case must therefore consider the extra demand and the resources it consumes.

Product, supply chain, finance, and customer care must approve the same operating boundary.
A product manager cannot infer capacity merely from historical average throughput.
A transport manager cannot assume customers are indifferent to a longer date.
Finance cannot claim all additional revenue as profit.
The responsible owner makes the final trade-off using both evidence and stated risk appetite.

### 3.3 Four different clocks

The display clock starts when a visitor sees an estimate.
The acceptance clock starts when the order is placed under stated terms.
The operational clock tracks release, pick, pack, handoff, and transit.
The customer clock ends at the promised delivery deadline in the relevant local time zone.

Calendar days, business days, and service days are not interchangeable.
A day-count promise must say which days count and whether the order day counts.
A date promise should still identify the end-of-day convention where it affects the result.
All stored instants should preserve an unambiguous offset or coordinated universal time (UTC).
The displayed date should be generated using the customer's delivery time zone and the applicable calendar.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or grain |
|---|---|---|---|
| Promise | Delivery date or window and associated terms communicated to a customer. | Treating an internal ETA as an accepted commitment. | Order, line, or delivery group; local date/time. |
| Estimate | Best available statement of a future outcome before or after acceptance. | Presenting it as a guarantee without qualification. | Instant or date range. |
| Commitment | Customer-facing obligation as defined by applicable policy and law. | Changing it silently after order placement. | Versioned order or delivery group. |
| Forecast | Probability distribution or prediction of a future event. | Treating its mean as a service guarantee. | Probability by arrival time. |
| Plan | Selected operational actions intended to fulfill an order. | Assuming the plan will always execute. | Node, service, and resources. |
| Target | Desired level used to evaluate a policy. | Claiming it describes every individual outcome. | Percent of defined population. |
| Delivery group | Lines that share a delivery commitment and operational path. | Assuming all lines in a basket always arrive together. | Group within an order. |
| Order management system (OMS) | System coordinating order state and sourcing. | Assuming it alone controls carrier execution. | Order and line events. |
| Stock keeping unit (SKU) | Identifier for an inventory item. | Treating aggregate category stock as item availability. | SKU and location. |
| Available to promise (ATP) | Inventory eligible for a new commitment after applicable deductions. | Equating physical on-hand with unreserved sellable stock. | Units by SKU and node. |
| Reservation | Claim on inventory or capacity for a defined period and purpose. | Assuming a checkout quote permanently reserves a unit. | Units or slots, expiration time. |
| Candidate path | A possible source, handling, carrier, and service combination. | Comparing cost before applying restrictions. | Path per delivery group. |
| Eligibility | Whether a path satisfies hard operational and policy requirements. | Treating a high predicted probability as proof of physical feasibility. | Boolean with reason codes. |
| Cutoff | Latest local order or release time eligible for a processing wave or pickup. | Assuming a cutoff is the same at every node. | Local time, date, and time zone. |
| Handoff | Transfer of a parcel from fulfillment operation to carrier. | Equating label creation with physical carrier acceptance. | Event timestamp. |
| Transit | Movement from carrier acceptance to delivery or other terminal outcome. | Including warehouse processing without saying so. | Hours or service days. |
| Calendar | Dates and times on which a specific operation or service is available. | Applying a national business-day calendar to all lanes. | Node, carrier, lane, and date. |
| Deadline | Latest instant that satisfies the promise's stated interpretation. | Comparing dates without time-zone or end-of-day rules. | Time-zone-aware timestamp. |
| Quantile | Arrival time below which a stated fraction of outcomes lies. | Calling the 95th percentile a 95% guarantee for a new population. | Hours or date at percentile. |
| Joint distribution | Distribution of combined outcomes including their dependence. | Multiplying marginal probabilities without checking dependence. | Event sequence or total duration. |
| Calibration | Agreement between predicted probabilities and observed frequencies for a population. | Confusing accurate ranking with calibrated probabilities. | Predicted probability bin and cohort. |
| Coverage | Fraction of a defined population for which a specific option is offered. | Dividing by completed orders rather than eligible visits. | Percent of eligible sessions or orders. |
| On-time delivery (OTD) | Fraction delivered by the original applicable deadline under a documented denominator. | Dropping canceled or still-open orders without disclosure. | Percent of a defined cohort. |
| First-attempt delivery | An attempted delivery satisfying the chosen event definition. | Reporting it as physical delivery to the recipient. | Parcel event and timestamp. |
| Split shipment | One order fulfilled in multiple parcels or delivery groups. | Counting each parcel as an independent order. | Parcels per order. |
| Completion time | Time at which all required delivery groups are delivered. | Using the first parcel's arrival for a whole-order promise. | Timestamp per order. |
| Censoring | Outcome is not yet observed at the measurement cutoff. | Treating an open parcel as on time. | Order or parcel at observation date. |
| Incremental contribution | Change in revenue less decision-relevant variable costs. | Multiplying total revenue by uplift and calling it profit. | USD per session or period. |
| Interference | Treatment changes outcomes of units assigned to another treatment. | Assuming customers are independent while they share capacity. | Experimental cluster or time block. |
| Guardrail | Predefined condition that limits or stops a test or policy. | Choosing a favorable threshold after results appear. | Metric, threshold, and evaluation window. |
| Fallback | Preapproved alternative when an input, path, or policy fails. | Silently advertising the previous fast date. | Policy and trigger. |

Terms in the table are operational definitions for this lesson.
An organization may adopt different event definitions, but the definitions must remain explicit and stable when results are compared.

## 5. The mechanics, step by step

### Step 1 — Establish the customer decision and commitment boundary

**Input:** the customer journey, offer terms, and business objective.
**Output:** a versioned promise contract and an accountable decision owner.

1. Identify where the customer first sees a speed claim: search, product page, cart, checkout, or confirmation.
2. State whether the message applies to one item, the whole basket, or a particular delivery group.
3. Identify the point at which the customer accepts the terms.
4. Specify the date, delivery window, price, destination, exclusions, and change procedure.
5. Document whether a range's latest date is the service deadline or whether another rule applies.
6. Define whether a failed attempt, safe-place delivery, pickup-point availability, or receipt by the customer counts.
7. Involve legal and customer-care owners when defining the public claim and remedy.
8. Assign a product owner who can change or suspend the public promise.

For an order with multiple lines, decide whether the promise is for the first parcel, every line, or an explicitly separated group.
If checkout shows a single date for all items, measure the completion of the entire promised group.
A parcel-by-parcel denominator can disguise a late final item.

The product page may know only a coarse region and a single item.
Checkout may know a complete address, quantities, and the whole basket.
Do not treat a product-page estimate as if it used checkout-quality information.
Refresh it transparently when the information changes.
Log the previous and replacement terms rather than overwriting history.

**Branch:** if an offer is a quote rather than a reservation, disclose its validity and refresh before acceptance.
If the company elects to reserve stock or carrier capacity, define an expiry and release process.
Avoid locking scarce stock indefinitely for abandoned sessions.

### Step 2 — Form the decision-time state

**Input:** event-time basket, delivery location, current inventory, capacity, and master data.
**Output:** a reproducible snapshot of facts known when the promise was generated.

1. Record the order or session identifier and decision timestamp.
2. Capture SKU, quantity, dangerous-goods status, and other relevant item restrictions.
3. Resolve the destination to the routing area needed for service eligibility.
4. Retain only location detail necessary for this decision and its audit.
5. Obtain inventory by SKU and node with its timestamp and source.
6. Deduct reservations, holds, damaged stock, and other unavailable quantities under a defined ATP rule.
7. Exclude inbound inventory unless the policy explicitly allows it and the arrival risk is modeled.
8. Capture processing and pickup capacity already consumed or reserved.
9. Record source freshness, model version, calendar version, and policy version.

A practical ATP expression is sellable on-hand minus existing claims minus policy buffers.
The exact deductions depend on the organization's inventory policy.
Do not subtract the same reservation twice if the upstream quantity is already net of it.
A negative balance is a data or control condition requiring a rule, not a quantity to promise.

Availability checks are vulnerable to races.
Two customers can see the same final unit at the same time.
At order acceptance, an atomic reservation or equivalent concurrency control should prevent double allocation.
If it fails, the system must refresh the offer or follow a documented exception policy.
Do not silently substitute an unapproved slower commitment.

**Branch:** if a critical stock or capacity feed is stale, apply a predefined safe fallback or stop offering the affected service.
The acceptable freshness threshold is a policy setting with a named owner, not a universal number.

### Step 3 — Generate candidate fulfillment paths

**Input:** eligible sources, carriers, services, and delivery-group policy.
**Output:** candidate paths with stock and operational restrictions attached.

1. List each node that can provide the complete delivery group.
2. If policy permits splits, generate feasible multi-node alternatives as separate candidates.
3. Match each source to carrier services that cover the origin, destination, and parcel characteristics.
4. Exclude services that prohibit the item's contents or dimensions.
5. Check that each node can process the order and that the carrier can collect it.
6. Check destination delivery days and any local access restrictions.
7. Include third-party and supplier sources only under their documented availability and handoff rules.
8. Attach cost inputs and a path identifier for subsequent prediction and audit.

A path is not eligible merely because the carrier operates somewhere in the region.
Coverage may vary by postcode, day, product, weight, and requested service.
Some products cannot travel by air or may require special packaging.
Some destinations require appointments or have access restrictions.
A single prohibited item can invalidate a whole-basket single-parcel option.

For a split, record the path and promise for every delivery group.
A whole-order deadline is met only when every required group meets its applicable deadline.
Split economics must include all parcels and any additional handling, packaging, and customer-care cost.

**Branch:** if no path is eligible, show a truthful unavailable or later option.
Do not manufacture an ETA from an ineligible carrier's historical average.

### Step 4 — Build the operating calendar and cutoff clock

**Input:** decision timestamp, node calendar, processing schedule, pickup timetable, and carrier delivery calendar.
**Output:** a feasible sequence of event windows and an unambiguous deadline.

1. Convert the order timestamp into the warehouse's local time zone.
2. Determine whether the order entered the eligible queue before the relevant release cutoff.
3. Place late orders into the next eligible processing wave.
4. Apply the node's operating days, shift hours, and holiday exceptions.
5. Check whether the predicted processing completion can reach an available pickup.
6. Apply the carrier's lane-specific collection, transit, and delivery calendar.
7. Convert the resulting arrival instant to the customer's local date and time zone.
8. Compare it with the customer-visible deadline under the chosen event definition.

A warehouse may process on Saturday while a carrier does not collect until Monday.
A carrier may collect on Saturday but not deliver to a particular destination on Sunday.
A holiday at either the origin or destination can affect a different part of the path.
An advertised cutoff should therefore identify its scope and time zone.

Daylight-saving transitions can produce nonexistent or repeated local times.
Store the original offset and use a time-zone database rather than fixed UTC offsets.
Compare aware timestamps or dates after applying the correct local calendar.
Never compare strings with missing offsets as if they were universally ordered instants.

**Branch:** if the arrival calculation crosses a cutoff, move to the next eligible pickup.
Do not add 24 hours mechanically when the next operating day may be later.

### Step 5 — Model the joint arrival outcome

**Input:** comparable decision-time historical events or a validated simulation of the end-to-end path.
**Output:** an arrival-time distribution conditional on path, context, and available features.

1. Choose the outcome: actual delivery, first attempt, pickup availability, or another documented event.
2. Construct examples at the original decision time, not from information learned later.
3. Segment or model by destination, node, carrier service, weekday, cutoff position, season, and relevant basket attributes.
4. Include cancellations, missing scans, and still-open shipments in the data-quality ledger.
5. Estimate the total order-to-delivery distribution rather than relying solely on component means.
6. Preserve dependence between processing and transit where shared disruptions affect both.
7. Check sample size and stability before trusting a fine-grained lane estimate.
8. Backtest on later time periods and compare predicted probabilities with realized outcomes.

Historical shipments reflect earlier routing decisions.
An unchosen path does not have an observed arrival for the same order.
A model trained on selected shipments is not automatically an unbiased counterfactual model for every candidate.
Use pooled estimates, carefully designed exploration, or operational evidence where lane data are sparse.
Flag extrapolation rather than presenting a small sample as a precise probability.

If using simulation, sample complete correlated scenarios or explicitly model the dependence.
Independently sampling processing and transit is a modeling assumption, not a consequence of having marginal data.
Include rare capacity disruptions if they are within the business decision's risk horizon and can be credibly estimated.
Document excluded events and assess sensitivity to them.

A percentile answers a distributional question.
For a deadline `d`, the relevant number is `P(arrival <= d | information at decision)`.
For a desired probability `q`, a forecast quantile is the earliest time whose modeled cumulative probability reaches `q`.
A finite historical quantile is uncertain, and it can move as the population changes.
Its interpretation depends on the outcome, cohort, and model being well specified.

**Branch:** if the estimate is too uncertain or outside its supported population, use an approved conservative policy or withhold the fast option.
An uncertainty threshold and its owner should be explicit.

### Step 6 — Set the service policy and filter paths

**Input:** predicted arrival distributions, eligible paths, customer deadlines, and service policy.
**Output:** paths that satisfy hard constraints and the selected risk threshold.

1. Define the service objective for the relevant customer population and period.
2. Translate it into a path-level selection rule only if that is the intended policy.
3. Calculate the probability of arriving by the candidate deadline for each path.
4. Apply the threshold using the correct arrival event and local deadline.
5. Apply stock, capacity, legal, and item restrictions independently of the prediction.
6. Record each rejection reason and the policy version.
7. If no path passes, evaluate a later date or decline the offer.
8. Recheck before acceptance if state changes or the quote expires.

A 95% model threshold for each selected path is not a guarantee that realized aggregate OTD will be at least 95%.
Calibration error, common shocks, changing demand, and limited observations can all cause a miss.
Conversely, an aggregate target can conceal poor service for a particular geography or seller.
Define segment-level guardrails where they are materially important and supported by adequate data.

The decision may also require uncertainty bounds.
A 97% observed success rate from 100 prior shipments is a point estimate, not proof that the true probability exceeds 95%.
A rule based on a lower confidence bound is more conservative but will reject more thin-data lanes.
Choose the approach in advance and evaluate the business impact.

**Branch:** if a fast path fails, a slower eligible service may be offered.
If the slow service also fails, do not promise either date just to preserve conversion.

### Step 7 — Compare full path economics

**Input:** feasible paths, rates, handling costs, risk costs, and customer-facing prices.
**Output:** a comparable cost and contribution estimate for each feasible offer.

1. Estimate pick, pack, packaging, and incremental node handling costs.
2. Include actual or contract-based carrier charges for the lane, service, weight, and dimensions.
3. Include fuel, residential, peak, and other applicable surcharges without double counting.
4. Add incremental costs for split parcels, transfers, and required expediting.
5. Include expected delay remedy or support cost only when it is incremental and not already embedded elsewhere.
6. Account for shipping fees charged to the customer as revenue, not as a negative carrier invoice.
7. Use the same time horizon and cost definitions for all options.
8. Separate committed fixed costs from genuinely avoidable or incremental costs.

Capacity has an opportunity cost when an additional order displaces a more valuable use of a constrained resource.
A modeled shadow price can represent that trade-off if the constraint and optimization are explicit.
It should not be added on top of an identical actual incremental capacity charge without explanation.
A quoted carrier rate is not necessarily the final cost after accessorial charges.

Holding and placement costs matter when a promise requires an upstream inventory decision.
Do not charge the full cost of existing inventory to a marginal order if it will be incurred regardless of the decision.
For a network redesign, however, include incremental carrying, transfer, facilities, and implementation costs.
The decision boundary determines which costs are relevant.

**Branch:** among paths meeting policy, choose the least costly path only if the customer offer and other objectives are held constant.
If the paths produce different customer-visible dates or prices, compare their customer and business effects as well.

### Step 8 — Commit resources and publish the offer

**Input:** selected path, offer terms, and reservation policy.
**Output:** a displayed and, after acceptance, recorded commitment with an executable operational plan.

1. Present the delivery date or window, price, and material qualifications before acceptance.
2. Bind the quote to the basket, destination, policy version, and expiry where applicable.
3. Refresh when the basket, destination, capacity, or inventory changes materially.
4. On acceptance, atomically reserve inventory or otherwise enforce the chosen allocation policy.
5. Reserve capacity where the operating model requires it and supports it.
6. Persist the originally accepted commitment separately from later estimates.
7. Send the customer confirmation with the actual accepted terms.
8. Release unused holds when the order is canceled or a quote expires.

A reservation on stock without a realistic warehouse slot or carrier collection may still produce an infeasible plan.
Conversely, reserving every theoretical resource for every page view can exhaust capacity unnecessarily.
The reservation design must match the decision point and expected abandonment.

If the order-management system later reroutes a parcel, the new path must preserve the original commitment or trigger an approved exception.
A reroute is an operational plan change, not permission to overwrite the customer's original date.

**Branch:** if reservation fails during checkout, refresh the offer before the transaction is completed where possible.
If failure is discovered after acceptance, initiate the exception process rather than concealing it.

### Step 9 — Monitor execution and recover exceptions

**Input:** order state, warehouse events, carrier scans, and the original commitment.
**Output:** prioritized interventions, customer notifications, and a durable audit trail.

1. Track release, pick, pack, physical handoff, transit, and delivery events.
2. Compare actual and predicted progress against the remaining time to deadline.
3. Detect missing scans separately from confirmed operational delays.
4. Classify causes using evidence and allow an unknown category.
5. Evaluate feasible recovery actions, such as an alternate carrier or approved expedite.
6. Compare recovery cost with customer and policy consequences, not merely with invoice cost.
7. Notify affected customers using approved timing, language, and channels.
8. Record the offered remedy, owner, action, and final outcome.

A label-created event is not proof the carrier possesses the parcel.
An early risk alert is not proof that delivery will be late.
The monitoring system should distinguish predicted risk, confirmed delay, and final outcome.
Where scans are absent, do not fabricate a delivery timestamp.

The warehouse owner acts on missed processing milestones.
The transportation owner investigates pickup or transit exceptions.
Customer care owns approved notification and remedy execution.
The promise owner reviews systemic patterns and can throttle or suspend the affected offer.
Finance and legal approve economic and policy boundaries in advance.

**Branch:** if a promise becomes unattainable, follow the applicable customer notification and remedy policy.
An updated estimated arrival may be useful, but it must not replace the original promise in the audit record.

### Step 10 — Measure, learn, and govern

**Input:** immutable offer and order history, final delivery outcomes, costs, and experiment assignments.
**Output:** a service scorecard, diagnosis, and approved policy or model changes.

1. Choose the unit: session, order, delivery group, parcel, or line.
2. Define the cohort by acceptance date and original promise, not by the date a parcel happened to arrive.
3. Wait for a specified maturity period or report open cases explicitly.
4. Calculate OTD using the documented treatment of cancellations, failures, and customer-caused exceptions.
5. Publish offer coverage and conversion alongside service and economics.
6. Compare predicted and realized reliability by supported segment and calendar period.
7. Investigate causes before changing thresholds or model parameters.
8. Approve releases, rollback triggers, and ownership through an operating review.

A metric based only on delivered parcels is a delivered-parcel conditional metric.
It should not be labeled whole-order OTD when incomplete or canceled orders are omitted.
Report counts as well as percentages so small denominators remain visible.
Show uncertainty intervals when the data and decision warrant them.

Customer-side interruptions, fraud holds, and address errors require explicit classification.
Do not exclude them solely because they reduce reported performance.
If a contractual metric allows exclusions, publish both that metric and a broader customer-experience view where feasible.
Do not confuse the two denominators.

The model owner maintains backtests, feature definitions, and drift monitoring.
The operations owner verifies that new routing decisions remain executable.
The product owner approves presentation and fallback behavior.
The accountable promise owner decides whether evidence supports continuing the offer.

## 6. Worked example

### 6.1 Decision and assumptions

A fictional retailer is evaluating delivery promises for a single-item order.
The customer accepts an offer at 12:00 UTC on Monday.
The customer-facing deadlines in this simplified example are also specified in UTC.
The item is available and reserved at the selected warehouse.
Both candidate paths satisfy item, destination, pickup, and capacity restrictions.
The policy requires a modeled probability of at least 95% of delivery by the displayed deadline.
This is a selection target, not a guarantee to any individual customer.

There are two candidate offers:

| Candidate | Customer-visible deadline | Time from acceptance | Incremental fulfillment cost |
|---|---|---:|---:|
| Fast | Wednesday 12:00 UTC | 48 hours | USD 8.10 per order |
| Standard | Thursday 12:00 UTC | 72 hours | USD 7.40 per order |

For simplicity, both options charge the customer the same shipping fee.
The cost boundary includes the same pick, packaging, transport, and ordinary surcharge categories for both paths.
It excludes shared fixed overhead that does not change with this decision.
The following empirical outcomes are illustrative joint order-to-delivery samples from comparable past orders.
The same samples are used in the code section.

| Path | Number of comparable outcomes | Arrival duration | Meets its deadline? |
|---|---:|---:|---|
| Fast | 92 orders | 36 hours | Yes: 36 ≤ 48 hours. |
| Fast | 8 orders | 60 hours | No: 60 > 48 hours. |
| Standard | 97 orders | 50 hours | Yes: 50 ≤ 72 hours. |
| Standard | 3 orders | 80 hours | No: 80 > 72 hours. |

The discrete durations make the arithmetic transparent; real arrival distributions are not usually this simple.
The samples do not establish future performance or justify causal comparison across routes.
Assume for this illustration that they are usable estimates of the corresponding future path distributions.
In practice you would check time stability, representativeness, dependence, and statistical uncertainty.

### 6.2 Feasibility and probability

For the fast path, 92 of 100 comparable outcomes meet the 48-hour deadline.
The empirical success estimate is `92 / 100 = 0.92`, or 92%.
The 92% estimate is below the 95% policy threshold.
Therefore the fast offer is not eligible under this simple point-estimate policy.

For the standard path, 97 of 100 comparable outcomes meet the 72-hour deadline.
The empirical success estimate is `97 / 100 = 0.97`, or 97%.
The 97% estimate is above the 95% threshold.
Therefore the standard offer passes this particular service screen.
It is also physically eligible and costs USD 7.40 per order.
The decision is to offer the standard deadline, subject to successful reservation and the stated assumptions.

The 97% figure is an observed fraction from a small illustrative sample.
It is not a lower confidence bound and does not prove that future reliability exceeds 95%.
A policy requiring statistical evidence above 95% might reject this path as well.
The business must choose that policy before looking at the favorable or unfavorable results.

The fast path's average duration is `(92 × 36 + 8 × 60) / 100 = 37.92 hours`.
That average is below its 48-hour deadline, yet 8% of its outcomes miss the deadline.
A mean below the deadline is therefore insufficient to establish the 95% service target.
The standard path's average is `(97 × 50 + 3 × 80) / 100 = 50.90 hours`.
The decision uses each path's probability against its own deadline, not a comparison of means alone.

### 6.3 Interpreting a missed cutoff

Suppose the selected warehouse's carrier collects at 14:00 local time on operating days.
An order released after the processing cutoff cannot reach that pickup even if the carrier has not yet departed.
If the next collection is the following operating day, the distribution must reflect that later handoff.
If a weekend or holiday prevents collection, the delay may be more than 24 hours.
The 97% estimate should not be reused for this changed cutoff state without supporting evidence.
The appropriate answer may be a later displayed date rather than a higher-cost route.

This cutoff illustration is a change of state, not an extra outcome secretly added to the 100-order sample.
A real implementation would calculate a new conditional distribution using the applicable calendar.
It would also verify whether capacity remains available for the next collection.

### 6.4 Whole-order completion

Now suppose a different order is explicitly split into two delivery groups.
Group A arrives by its deadline in 98 of 100 paired historical scenarios.
Group B arrives by its deadline in 97 of the same 100 scenarios.
Suppose the two late sets are disjoint: two scenarios miss only A and three miss only B.
Then both groups meet their deadlines in `100 - 2 - 3 = 95` scenarios.
Whole-order completion is therefore 95%, not 98% or 97%.

If the same two scenarios cause both groups to be late and one additional scenario misses B, the whole-order rate is 97%.
The marginal group rates are identical in both cases, but the dependence differs.
Multiplying `0.98 × 0.97 = 0.9506` would assume independence.
That assumption is not justified by the marginal rates alone.
Use paired outcomes or a justified joint model for a whole-order commitment.

### 6.5 Illustrative change in conversion and contribution

The product owner proposes a faster visible offer for a future eligible population.
The following is a scenario for the business case, not a measured causal effect.
Assume the new offer can be executed and passes all operational and service controls.
Assume 100,000 eligible sessions per month under either policy.
Each converting session produces exactly one order in this simplified scenario.
Contribution before fulfillment and delay remedies is USD 25.00 per order under both policies.
That figure already nets product, payment, and other unchanged variable costs.
Assume the shipping revenue treatment is already included consistently in the USD 25.00 figure.

| Input | Current policy | Proposed faster policy |
|---|---:|---:|
| Eligible sessions per month | 100,000 | 100,000 |
| Conversion rate | 4.0% | 4.2% |
| Orders per month | 4,000 | 4,200 |
| Contribution before fulfillment and remedies | USD 25.00/order | USD 25.00/order |
| Fulfillment cost | USD 8.00/order | USD 10.50/order |
| Miss rate | 4% of orders | 3% of orders |
| Expected remedy cost per missed order | USD 5.00 | USD 5.00 |
| Incremental fixed program cost | USD 0/month | USD 2,000/month |

The current policy's expected remedy cost per order is `0.04 × USD 5.00 = USD 0.20`.
Its expected contribution per order is `USD 25.00 - USD 8.00 - USD 0.20 = USD 16.80`.
Monthly contribution is `4,000 × USD 16.80 = USD 67,200`.

The proposed policy's expected remedy cost per order is `0.03 × USD 5.00 = USD 0.15`.
Its expected contribution per order before program cost is `USD 25.00 - USD 10.50 - USD 0.15 = USD 14.35`.
Monthly contribution after program cost is `4,200 × USD 14.35 - USD 2,000 = USD 58,270`.
The incremental effect is `USD 58,270 - USD 67,200 = -USD 8,930 per month`.
The proposed policy is not financially justified under these specific assumptions.
It does produce 200 additional orders and a lower assumed miss rate, but those benefits do not cover the added cost.

The calculation compares total contribution across all eligible sessions, including sessions that do not convert.
It does not multiply the 200 extra orders by margin while ignoring the higher cost on the other 4,000 orders.
It also does not assume that a 0.2 percentage-point increase in conversion is a 0.2% relative increase.
The relative increase is `(4.2% - 4.0%) / 4.0% = 5%`.

### 6.6 Break-even and decision sensitivity

Let `c` be the proposed conversion rate expressed as a fraction.
With all other assumptions fixed, the proposed contribution is `100,000 × c × USD 14.35 - USD 2,000`.
Set it equal to the current policy's USD 67,200 per month.
The required proposed orders are `USD 69,200 / USD 14.35 = 4,822.30` orders per month.
The break-even conversion rate is approximately `4.8223%` of 100,000 eligible sessions.
Because orders are indivisible, at least 4,823 orders are needed in this fixed-session illustration.

Alternatively, hold conversion at 4.2% and solve for the proposed fulfillment cost `f`.
`4,200 × (USD 25.00 - f - USD 0.15) - USD 2,000 = USD 67,200`.
Thus `f = USD 24.85 - USD 69,200 / 4,200 = USD 8.3738` per order approximately.
The assumed USD 10.50 would need to fall by approximately USD 2.1262 per order.
These one-variable thresholds are not predictions: changing speed or cost may change conversion, service, and capacity simultaneously.

The owner should not launch solely because a point estimate crosses break-even.
They should examine statistical uncertainty, ramp effects, resource constraints, customer harms, and implementation cost.
A decision could be different if there are credible omitted benefits, but each benefit must have an explicit basis and avoid double counting.
Lifetime value, retention, and acquisition effects should not be added as certain facts without a defensible incremental estimate.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Save it as a local `.py` file or run it in a notebook cell.
It reproduces the candidate screen and the business-case arithmetic above.
The input durations are joint order-to-delivery observations, not independently sampled stage times.
The code deliberately uses a point estimate, matching the worked policy; it does not calculate a confidence bound.
It assumes both paths are physically eligible and ignores reservation races, calendars, and statistical estimation error.
Those omissions must be addressed in a production system.

```python
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Path:
    name: str
    deadline_hours: float
    cost_usd: float
    durations_hours: Tuple[float, ...]
    physically_eligible: bool = True

    def on_time_probability(self) -> float:
        if not self.durations_hours:
            raise ValueError("A path needs at least one observed duration")
        successes = sum(
            duration <= self.deadline_hours
            for duration in self.durations_hours
        )
        return successes / len(self.durations_hours)


def choose_path(paths: Tuple[Path, ...], threshold: float) -> Path:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between zero and one")
    feasible = [
        path for path in paths
        if path.physically_eligible
        and path.on_time_probability() >= threshold
    ]
    if not feasible:
        raise ValueError("No path meets the stated policy")
    return min(feasible, key=lambda path: path.cost_usd)


def contribution_per_order(
    before_fulfillment_usd: float,
    fulfillment_usd: float,
    miss_rate: float,
    remedy_per_miss_usd: float,
) -> float:
    return (
        before_fulfillment_usd
        - fulfillment_usd
        - miss_rate * remedy_per_miss_usd
    )


def monthly_contribution(
    sessions: int,
    conversion_rate: float,
    per_order_usd: float,
    program_cost_usd: float,
) -> float:
    return sessions * conversion_rate * per_order_usd - program_cost_usd


fast = Path(
    name="fast",
    deadline_hours=48,
    cost_usd=8.10,
    durations_hours=(36,) * 92 + (60,) * 8,
)
standard = Path(
    name="standard",
    deadline_hours=72,
    cost_usd=7.40,
    durations_hours=(50,) * 97 + (80,) * 3,
)
paths = (fast, standard)
threshold = 0.95

for path in paths:
    probability = path.on_time_probability()
    print(f"{path.name}: on-time estimate {probability:.1%}")
selected = choose_path(paths, threshold)
print(f"selected: {selected.name}; cost USD {selected.cost_usd:.2f}/order")

current_per_order = contribution_per_order(25.00, 8.00, 0.04, 5.00)
proposed_per_order = contribution_per_order(25.00, 10.50, 0.03, 5.00)
sessions = 100_000
current = monthly_contribution(sessions, 0.040, current_per_order, 0.0)
proposed = monthly_contribution(sessions, 0.042, proposed_per_order, 2_000.0)
break_even_conversion = (current + 2_000.0) / (sessions * proposed_per_order)

print(f"current: USD {current:,.2f}/month")
print(f"proposed: USD {proposed:,.2f}/month")
print(f"incremental: USD {proposed - current:,.2f}/month")
print(f"break-even proposed conversion: {break_even_conversion:.4%}")
```

Expected output:

```text
fast: on-time estimate 92.0%
standard: on-time estimate 97.0%
selected: standard; cost USD 7.40/order
current: USD 67,200.00/month
proposed: USD 58,270.00/month
incremental: USD -8,930.00/month
break-even proposed conversion: 4.8223%
```

The code compares each candidate against its own deadline.
It is not comparing two interchangeable routes for an identical customer offer.
If customers value the faster option, comparing only their operational cost is insufficient.
The screen is appropriate here because the fast offer fails the stipulated reliability threshold.

The output is reproducible from the fictional inputs and does not validate the model's real-world performance.
For real use, calculate probabilities from time-stamped observations with appropriate calendars and outcome definitions.
Add audit records, feed freshness controls, reservations, and a well-defined response when no candidate passes.
Version all model and policy inputs so the offer can be reconstructed later.

## 8. Data, standards, and regulatory context

### 8.1 Data grain and lineage

A promise decision requires more than a final delivered timestamp.
The minimum useful record links the offer state to the accepted order and its subsequent event history.
Each record should state its primary key, grain, source, event time, ingestion time, and retention policy.

| Data object | Grain | Important fields | Purpose |
|---|---|---|---|
| Offer impression | Session × display version | Timestamp, coarse destination, basket, dates, prices, policy version. | Reconstruct what a visitor saw. |
| Offer choice | Session × choice | Option, selection time, experiment assignment. | Link alternatives to customer behavior. |
| Accepted commitment | Order × delivery group × version | Original deadline, time zone, price, acceptance time. | Preserve customer terms and denominator. |
| Inventory snapshot | SKU × node × snapshot | Sellable stock, holds, freshness, source. | Reconstruct ATP at decision time. |
| Capacity state | Node or carrier slot × time | Remaining capacity, cutoff, reservation. | Check whether the plan was executable. |
| Path decision | Offer × candidate path | Eligibility, rejection reason, estimate, cost, model version. | Explain the chosen or omitted option. |
| Warehouse event | Order line or parcel × event | Release, pick, pack, handoff timestamps. | Diagnose processing and queue delays. |
| Carrier event | Parcel × scan | Event code, event timestamp, ingestion timestamp. | Determine transit and delivery outcomes. |
| Order status | Order × status event | Cancellation, refund, fraud hold, reason. | Define terminal outcomes and exclusions. |
| Cost | Parcel or order × charge | Rate, surcharge, credit, invoice date. | Compare forecast with realized economics. |
| Contact or remedy | Order × interaction | Contact reason, remedy, cost, timestamp. | Measure customer harm and response cost. |
| Calendar | Node/service/lane × interval | Time zone, operating windows, exceptions. | Reproduce cutoff and transit logic. |

The original promise must be immutable in the analytical record.
Later revised estimates should be additional versions with their own timestamps and reasons.
Otherwise a system can make a missed commitment disappear by replacing it with a later one.

A carrier scan can arrive late or be corrected after ingestion.
Store both the carrier's event time and the time the retailer learned of it.
Deduplicate repeated events without removing meaningful status corrections.
Define a precedence rule when conflicting delivery events appear.
Avoid joining an order to all parcel scans and then counting duplicated orders as extra successes.

### 8.2 Event quality and denominators

A delivery event should identify whether a parcel was delivered, attempted, refused, returned, or made available for pickup.
Those outcomes may have different operational and legal meanings.
A missing scan is not proof of delivery and is not necessarily proof of a carrier failure.
Reconcile carrier events against customer contacts, claims, and final order statuses where authorized.

For an acceptance-date cohort, report at least:

- Number of accepted orders or delivery groups.
- Number delivered by the original deadline.
- Number delivered after the original deadline.
- Number still open at the reporting cutoff.
- Number canceled, returned to sender, or otherwise terminal without delivery.
- Number excluded under any separately defined contractual metric, with reasons.

One transparent customer-experience metric is delivered by deadline divided by all accepted orders in a mature cohort.
Under that definition, cancellations and non-deliveries remain in the denominator unless a predefined population rule excludes them.
Another useful measure is delivered on time among delivered orders, clearly labeled as conditional.
The two rates answer different questions and should not be substituted for one another.

If the cohort is not mature, the unknown outcomes can change the result.
Report a lower bound treating all open cases as misses and an upper bound treating them all as successes.
These bounds are arithmetic descriptions, not statistical confidence intervals.
A survival or competing-risk analysis may be useful for richer timing questions, provided its assumptions are explained.

### 8.3 Standards and identifiers

GS1's Electronic Product Code Information Services (EPCIS) standard specifies a way to share supply-chain event data.
It describes event information such as what happened, when, where, and why, with supporting data structures.
It can help align event exchange among organizations.
Using EPCIS does not itself prove that an event is accurate, timely, or sufficient for a customer's promise.
Whether an EPCIS profile is mandatory depends on the relevant contract or applicable law; the standard alone is not a universal legal mandate.

GS1 identifiers can help trading partners identify products, locations, and logistics units consistently.
Identifier assignment and exchange should follow the applicable GS1 specifications and partner agreement.
A product identifier does not establish that a particular unit is sellable or reserved at a particular node.

For time representation, ISO 8601 is a standard for date and time formats.
A timestamp with an offset is useful for representing an instant, but a named time zone and calendar rules are also needed for future local operating hours.
The lesson's time calculations are conventions for illustration, not a claim that ISO 8601 dictates commercial service deadlines.

### 8.4 United States shipping claims

The U.S. Federal Trade Commission (FTC) enforces the Mail, Internet, or Telephone Order Merchandise Rule.
The FTC's business guidance explains that a seller must have a reasonable basis for a stated shipment time and addresses delay notices, consent, and refunds in covered circumstances.
The guidance also discusses a default period when no shipment time is stated.
The rule concerns shipment and should not be collapsed into a universal legal rule that every displayed delivery estimate has the same meaning.
Other advertising, consumer-protection, and state requirements may also matter.
Consult counsel for scope, current requirements, and application to a particular offer.

A delivery date can be a material customer representation even when it is not itself the shipment deadline addressed by the federal rule.
The merchant should preserve evidence supporting the claim and its terms.
The engineering requirement to track the original promise is a good control, not a substitute for a legal interpretation.

### 8.5 European Union consumer and data context

Directive 2011/83/EU on consumer rights contains information and delivery provisions for covered consumer contracts.
Article 18 addresses delivery timing and consequences under specified conditions, subject to the directive and applicable national implementation.
It is not safe to generalize one remedy or deadline to every European order without analyzing the contract and local law.
Legal review should consider the current national implementation and the relevant transaction.

The EU General Data Protection Regulation (GDPR) applies to personal-data processing within its scope.
A delivery address and linked order or clickstream history can be personal data.
The GDPR includes principles such as purpose limitation, data minimization, accuracy, storage limitation, and security.
Determine the legal basis and applicable rights with qualified counsel; pseudonymized data may still be personal data.

Delivery analytics usually does not require every analyst to view full street addresses.
Use an appropriately coarse routing zone where it is sufficient, with access controls for more detailed operational needs.
Separate experiment reporting from personally identifiable customer records where feasible.
Set retention periods and log access according to the organization's policy and applicable law.
Data minimization should not destroy the evidence needed to resolve a live customer dispute; design retention for both needs.

### 8.6 Practical data controls

An owner should sign off the following before a promise policy is released:

1. The location and item eligibility data are current enough for the selected fallback policy.
2. Inventory state and reservation semantics are documented and tested operationally.
3. Event definitions distinguish carrier acceptance from label creation.
4. Time zones and calendars are versioned, including holiday exceptions.
5. Model features are available at the moment of the promise.
6. Original terms and every subsequent revision are retained with a timestamp.
7. Reporting explicitly handles open, canceled, and missing-outcome records.
8. Personal data access, retention, and sharing have an approved basis.

## 9. Industry practice and case evidence

### 9.1 Amazon: speed is a network decision

Amazon's 2023 shareholder letter describes a shift from a national fulfillment network to regional networks in the United States.
The company states that the change positioned inventory closer to customers and contributed to faster delivery and lower cost to serve.
This is a company account, not an independent causal estimate of the effect of any single change.
The letter also describes the organizational and operating work involved in the change.

The transferable lesson is that delivery speed depends on inventory location and network design, not just a checkout label.
A promise owner should ask whether the item, labor, and transport capacity exist in the region before advertising the faster date.
The case does not establish that every retailer should replicate Amazon's network.
Its scale, membership economics, product mix, and capital access may differ substantially from those of another merchant.

[INFERENCE] Regional inventory can reduce distance but can also fragment stock and raise local stockout risk.
That trade-off must be measured using the merchant's own demand, replenishment, and inventory constraints.
A speed claim cannot be evaluated independently of those costs.

### 9.2 Walmart: store fulfillment expands choices and constraints

Walmart's public materials describe using stores as fulfillment locations for delivery and pickup.
This demonstrates a real operating model in which the store is both a customer-facing retail site and an e-commerce node.
Public descriptions do not establish the profitability or reliability of every store-level delivery promise.
A store's displayed availability may differ from physically pickable stock, especially when in-store shoppers share inventory.

[INFERENCE] A promise system using stores needs item-level availability, store labor, substitution rules, and local delivery capacity.
A short geographic distance does not guarantee a short completion time if the store queue is overloaded.
Compare a store route with a distribution-center route using the same customer deadline and full cost boundary.
A controlled test is needed before claiming that a faster store-based option caused a particular conversion improvement.

### 9.3 Carrier published service information

UPS and FedEx publish service and shipping information, including service-specific restrictions and rate material.
Those publications are inputs to a lane and service model, not a promise that every parcel arrives at a fixed time.
The current contract, destination coverage, holiday calendar, and exceptions determine which rules apply to a specific shipment.
Carrier terms and actual performance should be reviewed at the time of implementation.

A merchant should reconcile its modeled service and rate tables against invoices and event data.
A carrier's advertised transit time may start after pickup, whereas the customer clock starts at order acceptance.
Treat warehouse processing and waiting for pickup as separate components of the customer experience.

### 9.4 Causal evidence and transportability

A fast promise often appears with urban locations, in-stock products, and preferred customer segments.
Those factors can independently affect conversion and delivery reliability.
A simple difference between customers who saw fast and slow promises is therefore not necessarily causal.
Randomization can improve identification if allocation, exposure, and interference are handled correctly.

A credible experiment publishes the unit of randomization, eligibility rules, and primary outcome before analysis.
The primary outcome should normally use all eligible assigned sessions, not only completed orders.
The economic analysis should include both additional orders and higher costs on existing orders.
Capacity and carrier service should be monitored throughout the experiment.

Results are local to the tested population, period, offer, and capacity conditions.
A holiday peak, new carrier contract, or much broader rollout may change both conversion and fulfillment performance.
Treat extrapolation as an assumption and rerun sensitivity analysis at the intended scale.
Do not describe a company's general network disclosure as evidence for the effect size of your own test.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation and owner |
|---|---|---|---|---|
| Mean treated as commitment | Average arrival beats deadline but late tail grows. | Skewed or disrupted outcomes are hidden by the mean. | Missed customer dates. | Model deadline probability; analytics owner. |
| Stale inventory | Reservations fail after checkout. | Stock feed lags or holds are omitted. | Backorders and broken commitments. | Freshness gate and atomic reservation; inventory owner. |
| Capacity ignored | Queue age rises while stock remains available. | Many orders compete for limited labor or pickups. | Late handoff despite available goods. | Slot caps and queue guardrail; operations owner. |
| Calendar mismatch | Friday and holiday misses cluster. | Node, carrier, or destination days differ. | Systematic late deliveries. | Versioned lane calendars; transport owner. |
| Time-zone error | Deadlines shift around midnight or daylight saving. | Naive timestamps or fixed offsets are used. | Wrong customer date. | Aware timestamps and local calendar logic; platform owner. |
| Label counted as handoff | Label timestamps precede physical scans by days. | Electronic label event is misclassified. | Inflated carrier and warehouse service. | Require physical acceptance event; data owner. |
| Promise overwritten | Late rate falls after revised dates increase. | Latest ETA replaces accepted deadline. | Misleading scorecard and audit gap. | Immutable original version; product owner. |
| Split counted by parcel | Parcel OTD is high while customers report incomplete orders. | First parcel masks a late final group. | Poor whole-order experience. | Track completion against group terms; analytics owner. |
| Missing outcomes excluded | Reported OTD rises as open cases accumulate. | Only delivered parcels enter denominator. | Optimistic measurement. | Mature cohort and open-case bounds; reporting owner. |
| Independence assumed | Joint misses exceed model prediction in peak. | Shared weather or congestion correlates delays. | Understated tail risk. | Paired scenarios and stress tests; model owner. |
| Selection bias | New route looks reliable only in easy lanes. | Historical routing selected favorable cases. | Overconfident counterfactual. | Controlled exploration and overlap checks; analytics owner. |
| Conversion correlation called causal | Fast-option buyers already had higher purchase intent. | Customer and geography mix confound comparison. | False benefit claim. | Randomized design and intent-to-treat analysis; experimentation owner. |
| Cost boundary inconsistent | Faster option excludes surcharges or fixed rollout costs. | Alternatives use different accounting scopes. | Misallocated budget. | Common decision boundary and finance sign-off. |
| Shared capacity interference | Control service deteriorates during treatment. | Treatment consumes common resources. | Biased experiment and customer harm. | Cluster or switchback design and guardrails; operations and experiment owners. |
| Poor fallback | A stale feed still displays the last fast date. | Failure defaults to an attractive but unsupported offer. | Scaled customer harm. | Explicit safe fallback and alert; promise owner. |
| Remedy double counted | Support credits appear in two cost lines. | Expected miss cost overlaps actual service cost. | Understated or overstated benefit. | Reconcile cost ledger; finance owner. |
| Unsupported subgroup inference | A lane shows 100% on-time from three parcels. | Small sample creates false precision. | Unsafe expansion. | Pool responsibly and report uncertainty; model owner. |
| Incentives hide cancellations | Teams remove hard orders from service reporting. | Metric rewards an incomplete denominator. | Bad customer experience and gaming. | Report all terminal states and audit exclusions; governance owner. |

Investigate a red flag at the smallest useful operational grain, then verify that the pattern persists in the broader cohort.
A single late parcel may be a customer-specific exception or an early sign of a shared problem.
Do not immediately retrain a model when the actual issue is a broken inventory feed or a changed carrier cutoff.
The correction must address the mechanism, not merely suppress the symptom.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or monitor |
|---|---|---|---|
| Candidate stock is sellable and unreserved. | A path cannot fulfill an unavailable item. | Stale counts, damage, concurrent claims. | Reconcile counts, reservation failures, and freshness. |
| Node capacity is available at acceptance. | Processing must fit before a viable pickup. | Demand spike, labor absence, equipment outage. | Track queue age, slot utilization, and throughput. |
| Carrier service covers the lane and product. | A valid transit path must exist. | Restriction changes, remote-area exclusions, hazardous goods. | Refresh contract and lane eligibility data. |
| Calendar and time-zone rules are correct. | A deadline and pickup sequence require local time. | Holiday updates or daylight-saving errors. | Compare source calendars and inspect boundary dates. |
| Historical examples resemble the decision population. | Empirical rates estimate future outcomes only with relevant data. | New lane, policy, season, customer mix, or weather. | Time-based backtests and segment drift checks. |
| Joint delay dependence is represented. | Whole-path tail risk depends on shared shocks. | Independent component sampling during congestion. | Compare simulated and observed total-delay tails. |
| The customer event matches the metric. | The score must measure the communicated outcome. | Using first attempt for a delivered-to-door commitment. | Audit terms, carrier codes, and customer disputes. |
| Open outcomes are handled consistently. | Unobserved delivery cannot be counted as success. | Recent cohorts or missing scans. | Publish maturity, counts, and open-case bounds. |
| Conversion estimate is incremental. | The business case requires a counterfactual. | Confounding, selection, interference, or novelty. | Randomized test with prespecified analysis. |
| Costs share a common boundary. | Alternatives must be comparable. | Missing surcharge or double-counted remedy. | Reconcile rates, invoices, and finance ledger. |
| Session volume is unaffected in the simple scenario. | A fixed denominator supports the worked break-even. | Acquisition, repeat visits, or channel mix changes. | Measure eligible traffic and vary the assumption. |
| One conversion produces one order in the example. | Converts session rates into order counts. | Multi-order sessions or identity duplication. | Define and inspect session-to-order linkage. |
| Incremental program cost is USD 2,000 per month. | The worked scenario includes fixed change cost. | Different rollout scale or implementation charges. | Obtain approved budget and vary the input. |
| No spillover changes untreated capacity. | Simple customer-level experiments assume no interference. | Shared warehouse or carrier bottlenecks. | Monitor control service; use cluster or switchback design. |

The worked economic model does not forecast returns, lifetime value, or demand displacement across channels.
Those effects may matter in a real decision, but should be added only with a supported incremental estimate.
Similarly, the example does not price carbon, customer trust, or legal risk as if a universal monetary value existed.
An organization may use additional decision criteria, provided their measurement and ownership are explicit.

A target is not a prediction and an empirical point estimate is not a guarantee.
A strong performance result in one season does not establish service for a holiday peak.
A policy should be re-evaluated when a material assumption changes.

## 12. Exercises

### Exercise 1 — Classify the decision objects

**Time:** 10 minutes.  
**Deliverable:** Four labeled statements and one sentence explaining each label.

A retailer says:

1. The model estimates an 88% probability of delivery by Tuesday.
2. The warehouse will pick the item on Monday and use carrier service A.
3. The retailer aims for 95% of accepted orders to meet their original deadline.
4. Checkout tells the customer, before acceptance, that delivery is due Tuesday.

**Acceptance criteria:** Correctly distinguish forecast, plan, target, and commitment; explain why none is interchangeable with another.

### Exercise 2 — Apply the cutoff and calendar

**Time:** 15 minutes.  
**Deliverable:** Earliest feasible pickup and customer deadline with a short timeline.

An order is accepted Friday at 16:10 in the warehouse's local time zone.
The processing cutoff for Friday's pickup is 15:30.
The warehouse and carrier do not process or collect on Saturday or Sunday.
Monday is an operating day and the next pickup is Monday at 18:00.
The carrier delivers on the next operating day after pickup, and Tuesday is an operating day.
Assume sufficient stock and capacity, no holidays, and that the destination shares the same time zone.

**Acceptance criteria:** Identify Monday pickup and Tuesday delivery; explain why adding 24 hours to Friday's missed pickup is incorrect.

### Exercise 3 — Estimate a whole-order outcome

**Time:** 15 minutes.  
**Deliverable:** Whole-order on-time percentage and a statement about dependence.

In 100 paired scenarios, parcel A misses in scenarios 1 and 2.
Parcel B misses in scenarios 2, 3, and 4.
The order is on time only if both parcels meet their respective deadlines.

**Acceptance criteria:** Count the union of misses once; compute 96%; explain why multiplying marginal rates is not justified without an independence assumption.

### Exercise 4 — Screen and select paths

**Time:** 15 minutes.  
**Deliverable:** A path decision with rejection reasons and cost.

A policy requires at least 95% predicted on-time probability.
Path A costs USD 6.00 per order and has a 93% prediction.
Path B costs USD 7.20 per order and has a 97% prediction, but the product is prohibited on its carrier service.
Path C costs USD 8.10 per order and has a 96% prediction and satisfies all hard constraints.
All predictions refer to the same displayed deadline and customer event.

**Acceptance criteria:** Reject A for the threshold and B for a hard restriction; select C at USD 8.10 per order.

### Exercise 5 — Define and calculate service metrics

**Time:** 20 minutes.  
**Deliverable:** Two rates and a denominator note.

A mature cohort has 200 accepted orders.
Of these, 180 arrive by the original deadline, 10 arrive late, and 10 are canceled without delivery.
Assume no open cases and no exclusions under the broad customer-experience definition.

**Acceptance criteria:** Calculate 90% delivered on time among accepted orders and approximately 94.74% on time among delivered orders; identify the 10 cancellations and avoid calling the second rate whole-cohort OTD.

### Exercise 6 — Reproduce and challenge the business case

**Time:** 25 minutes.  
**Deliverable:** Monthly contribution under each policy, incremental effect, and break-even conversion.

Use the data in Section 6.5.
Assume 100,000 eligible sessions, the stated conversion rates and cost inputs, and the proposed USD 2,000 monthly program cost.
Hold all other inputs constant when calculating the break-even proposed conversion.

**Acceptance criteria:** Reproduce USD 67,200 and USD 58,270 per month, the negative USD 8,930 effect, and approximately 4.8223% break-even; state at least two assumptions that may change together.

### Exercise 7 — Design an operational experiment

**Time:** 30 minutes.  
**Deliverable:** A one-page test design and decision rule.

A merchant wants to test whether a faster promise increases purchase conversion.
Customers in neighboring postcodes share warehouse pick capacity and carrier pickups.
The merchant can randomize compatible geographic clusters or alternate treatment in preplanned time blocks.
The fast treatment is allowed only while capacity and reliability guardrails hold.

**Acceptance criteria:** State the eligible population, unit and method of randomization, primary intent-to-treat outcome, minimum detectable effect and power-planning inputs, interference risk, economics, service guardrails, and stop/rollback owner.
State how analysis will account for the randomization unit and time or geography effects.

### Exercise 8 — Handle an at-risk accepted order

**Time:** 20 minutes.  
**Deliverable:** A concise incident playbook with named role types.

A warehouse outage makes an already accepted date unlikely.
The carrier has not yet physically accepted the parcel.
A more expensive alternative route might preserve the commitment, but inventory and capacity must be rechecked.

**Acceptance criteria:** Preserve the original deadline, distinguish risk from confirmed failure, check an authorized recovery path, name operational and customer-communication owners, specify escalation and remedy policy, and retain the outcome for analysis.

## 13. Self-check answers

### Exercise 1

Statement 1 is a **forecast**: 88% is a probability estimate conditional on the model and its inputs.
Statement 2 is a **plan**: it names the intended warehouse and carrier actions.
Statement 3 is a **target**: 95% is the desired aggregate performance level for a specified cohort.
Statement 4 is a **commitment** for the lesson's stated acceptance boundary: it is the date communicated to the customer.

A plan can fail even if it was correctly recorded.
A forecast can be calibrated while some individual orders arrive late.
A target does not prove actual results.
The original commitment remains the relevant customer-facing record after an ETA is revised.

### Exercise 2

The Friday processing cutoff is 15:30 local time.
The 16:10 acceptance occurs 40 minutes after that cutoff.
The order cannot use Friday's pickup under the stated rules.
Saturday and Sunday are not processing or collection days.
The next feasible pickup is therefore Monday at 18:00 local time.
The carrier delivers the next operating day, Tuesday, under the exercise's assumptions.

Adding 24 hours to a Friday pickup would land on Saturday.
That is not an eligible operating day in this scenario.
In a real system, verify both the warehouse and destination calendars and specify the customer deadline's time of day.

### Exercise 3

The union of missed scenarios is `{1, 2, 3, 4}`.
Scenario 2 is late for both parcels but counts only once as a whole-order miss.
There are 4 missed whole orders and 96 successful whole orders out of 100 scenarios.
Whole-order on-time completion is therefore `96 / 100 = 96%`.

The marginal on-time rates are 98% for A and 97% for B.
Their product is `0.98 × 0.97 = 0.9506`, or 95.06%.
That product uses an independence assumption and differs from the paired-scenario result.
Paired observations directly show the overlap in this exercise.

### Exercise 4

Path A has a 93% predicted on-time rate, below the 95% threshold.
Path B exceeds the probability threshold but violates an item restriction.
A prohibited carrier service remains ineligible regardless of its predicted probability or price.
Path C passes both the 95% screen and the hard restrictions.
Choose path C for USD 8.10 per order under the stated policy.

This exercise treats the predictions as policy inputs, not as guarantees.
If no path passed, the appropriate decision would be to consider a later promise or decline the option.

### Exercise 5

The broad accepted-order rate is `180 / 200 = 0.90`, or 90%.
The delivered-only rate is `180 / (180 + 10) = 180 / 190`, or approximately 94.74%.
The 10 canceled orders remain in the denominator of the broad metric by the exercise's definition.
They are absent from the delivered-only denominator, which must be disclosed.

The two metrics answer different questions.
The first asks what fraction of accepted orders arrived on time.
The second asks what fraction of orders that arrived were on time.
Neither alone explains why cancellations occurred or what customers experienced.

### Exercise 6

Current contribution per order is `25.00 - 8.00 - 0.04 × 5.00 = USD 16.80`.
Current volume is `100,000 × 0.040 = 4,000` orders per month.
Current contribution is `4,000 × 16.80 = USD 67,200` per month.

Proposed contribution per order is `25.00 - 10.50 - 0.03 × 5.00 = USD 14.35`.
Proposed volume is `100,000 × 0.042 = 4,200` orders per month.
After program cost, contribution is `4,200 × 14.35 - 2,000 = USD 58,270` per month.
The change is `58,270 - 67,200 = -USD 8,930` per month.

The break-even conversion is `(67,200 + 2,000) / (100,000 × 14.35)`.
That is approximately `0.048223`, or 4.8223%.
This assumes per-order contribution, eligible sessions, and fixed cost do not change as conversion changes.
In practice added demand may raise congestion and miss rates, and changing the promise may alter basket or customer mix.

### Exercise 7 rubric

A strong design defines eligibility before assignment and retains all eligible assigned sessions in the primary analysis.
It chooses a cluster or switchback unit that reduces shared-capacity contamination and documents why that unit is suitable.
It randomizes with an auditable schedule and balances material day-of-week or geographic differences.
It analyzes uncertainty at the randomization unit rather than pretending that every session is independent.

A strong power plan names baseline conversion, minimum detectable effect, significance level, desired power, expected cluster or time-block variation, and duration.
It accounts for clustering and operational constraints rather than reporting a sample size from a simple independent-user formula without justification.
A strong economic endpoint applies incremental contribution and program cost to the eligible population.
It reports OTD, cancellations, queue age, and cost by treatment, with thresholds approved in advance.

A strong stop rule identifies the accountable person, observation window, data source, and rollback action.
It addresses missing data, novelty, carryover between switchback periods, and incomplete exposure.
A weak answer compares only purchasers, ignores shared capacity, or claims a precise causal effect without sufficient randomized units.

### Exercise 8 rubric

A strong playbook retains the accepted deadline and records a separate revised estimate.
It confirms the parcel has not been handed to the carrier and distinguishes a risk alert from a final miss.
The warehouse owner assesses the outage and alternate processing availability.
The inventory and transportation owners validate stock, pickup capacity, lane eligibility, and the recovery cost.
An authorized decision owner approves or rejects an expedite under predefined policy.

Customer care sends the approved notice when policy or applicable requirements call for it.
The communication does not imply that a revised ETA replaces the original commitment.
The record includes timestamps, the chosen action, customer response, remedy where applicable, and final delivery outcome.
The promise owner reviews the incident for an offer throttle or broader policy change.
A weak answer silently moves the deadline or treats label creation as physical handoff.

## 14. Interview questions

### 1. How do you distinguish an ETA from a delivery promise?

A strong answer separates a probabilistic forecast from the customer-facing terms and preserves both versions.
A weak answer treats the latest model estimate as the original accepted commitment.

### 2. Why is a mean transit time insufficient for a 95% reliability target?

A strong answer discusses the deadline tail, full order-to-delivery distribution, and calibration.
A weak answer assumes that an average before the deadline means nearly every order is on time.

### 3. How would you model a split order?

A strong answer defines the customer-visible delivery groups and uses paired or otherwise dependent completion outcomes.
A weak answer reports only the earliest parcel or multiplies marginal probabilities without an assumption.

### 4. What makes an option physically infeasible despite a good ETA?

A strong answer identifies stock, reservation, item restrictions, processing slots, pickup calendars, and carrier coverage.
A weak answer relies solely on a predicted probability.

### 5. How would you define on-time delivery?

A strong answer names the cohort, original deadline, event, time zone, grain, and handling of open or canceled orders.
A weak answer provides a percentage without a denominator or silently removes difficult cases.

### 6. How do you handle a missed cutoff on a Friday?

A strong answer walks through local operating calendars, next pickup, and destination delivery days.
A weak answer always adds exactly 24 hours.

### 7. How do you compare two promises economically?

A strong answer uses incremental contribution per eligible session, full comparable variable costs, and fixed program costs.
A weak answer values only extra orders while ignoring higher cost on existing orders.

### 8. What can go wrong in a user-randomized promise experiment?

A strong answer discusses common capacity, interference, assignment and exposure, and clustered uncertainty.
A weak answer assumes every session is independent because the assignment used a random number.

### 9. How would you assess model calibration during peak?

A strong answer compares predicted and observed deadline success by mature cohort and relevant segment, with counts and uncertainty.
A weak answer reports only a single overall accuracy score or excludes open shipments.

### 10. What should happen if a critical inventory feed is stale?

A strong answer describes an approved fallback, alert, ownership, and a safe way to refresh the offer.
A weak answer continues to show an attractive promise using an old inventory snapshot without disclosure.

### 11. How should an organization communicate an at-risk order?

A strong answer separates risk, confirmed failure, and updated estimate; it names notification and remedy owners.
A weak answer overwrites the original date and waits for the customer to complain.

### 12. When is a regional fulfillment network useful?

A strong answer evaluates speed, inventory pooling, local capacity, transport cost, and the merchant's own demand mix.
A weak answer assumes a large retailer's network design is universally optimal.

## 15. Key terms

- [Accepted commitment](#4-definitions-and-vocabulary) — original customer-facing date or window and associated terms.
- [Available to promise](#step-2--form-the-decision-time-state) — stock eligible for a new commitment after policy deductions.
- [Calibration](#step-5--model-the-joint-arrival-outcome) — agreement between predicted probabilities and observed frequencies.
- [Candidate path](#step-3--generate-candidate-fulfillment-paths) — a source and service sequence that could fulfill a delivery group.
- [Capacity](#step-2--form-the-decision-time-state) — resources available in the relevant processing or transport slot.
- [Censoring](#82-event-quality-and-denominators) — an outcome not yet observed at the reporting cutoff.
- [Commitment boundary](#step-1--establish-the-customer-decision-and-commitment-boundary) — the point and terms under which a customer accepts an offer.
- [Contribution](#65-illustrative-change-in-conversion-and-contribution) — revenue less the decision-relevant costs in a stated boundary.
- [Coverage](#4-definitions-and-vocabulary) — fraction of a stated population offered an option.
- [Cutoff](#step-4--build-the-operating-calendar-and-cutoff-clock) — latest eligible local time for a processing or pickup wave.
- [Deadline](#step-4--build-the-operating-calendar-and-cutoff-clock) — last instant satisfying the stated promise.
- [Delivery group](#step-1--establish-the-customer-decision-and-commitment-boundary) — lines sharing a delivery commitment and path.
- [Eligible path](#step-3--generate-candidate-fulfillment-paths) — a candidate satisfying hard operational and policy constraints.
- [Exception](#step-9--monitor-execution-and-recover-exceptions) — deviation or risk requiring an operational response.
- [Fallback](#step-2--form-the-decision-time-state) — preapproved alternative when critical data or a path fails.
- [Forecast](#31-the-promise-is-a-decision-under-uncertainty) — prediction or probability distribution for an outcome.
- [Guardrail](#exercise-7--design-an-operational-experiment) — predefined condition that limits or stops a policy or test.
- [Handoff](#step-9--monitor-execution-and-recover-exceptions) — physical transfer from fulfillment to the carrier.
- [Incremental contribution](#66-break-even-and-decision-sensitivity) — change in contribution caused by a decision.
- [Interference](#exercise-7--design-an-operational-experiment) — one assignment affecting another unit's outcome.
- [Joint distribution](#step-5--model-the-joint-arrival-outcome) — combined outcomes including their dependence.
- [On-time delivery](#82-event-quality-and-denominators) — delivery by a defined deadline for a stated population.
- [Operational calendar](#step-4--build-the-operating-calendar-and-cutoff-clock) — valid processing, collection, and delivery windows.
- [Original promise](#81-data-grain-and-lineage) — immutable terms accepted before later ETA revisions.
- [Plan](#31-the-promise-is-a-decision-under-uncertainty) — intended operational actions and resources.
- [Quantile](#step-5--model-the-joint-arrival-outcome) — arrival time associated with a cumulative probability.
- [Reservation](#step-8--commit-resources-and-publish-the-offer) — time-bounded claim on stock or capacity.
- [Service target](#step-6--set-the-service-policy-and-filter-paths) — desired reliability for a defined population.
- [Split shipment](#64-whole-order-completion) — one order fulfilled through multiple parcels or groups.
- [Whole-order completion](#64-whole-order-completion) — arrival of all required groups under their applicable commitments.

## 16. Sources

Sources below support the indicated context; they do not validate the fictional numbers or prove the causal effect of a promise change.
Web resources and company terms can change; check the current version when implementing a policy.

1. U.S. Federal Trade Commission, [A Business Guide to the FTC's Mail, Internet, or Telephone Order Merchandise Rule](https://www.ftc.gov/business-guidance/resources/business-guide-ftcs-mail-internet-or-telephone-order-merchandise-rule). Official guidance on covered shipment representations and delay procedures; consult the rule and counsel for application.
2. European Parliament and Council, [Directive 2011/83/EU on consumer rights](https://eur-lex.europa.eu/eli/dir/2011/83/oj), particularly Article 18. Primary EU text; national implementation and later amendments matter.
3. European Parliament and Council, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), particularly Articles 5 and 6. Primary EU data-protection text; applicability depends on the circumstances.
4. GS1, [EPCIS and CBV standards](https://www.gs1.org/standards/epcis). Official event-sharing standard and documentation; it does not guarantee data quality.
5. GS1, [GS1 identification keys](https://www.gs1.org/standards/id-keys). Official descriptions of identifiers for products, locations, and logistics units.
6. ISO, [ISO 8601 date and time format](https://www.iso.org/iso-8601-date-and-time-format.html). Official overview of date and time representation.
7. Internet Assigned Numbers Authority, [Time Zone Database](https://www.iana.org/time-zones). Source for named time-zone rules used by many software systems.
8. Amazon, [2023 Letter to Shareholders](https://www.aboutamazon.com/news/company-news/2023-letter-to-shareholders). Company account of its regionalized fulfillment network and delivery performance; company claims are not independent causal estimates.
9. Walmart, [Pickup and Delivery](https://corporate.walmart.com/about/pickup-delivery). Company description of pickup and delivery services; not evidence of a specific store's economics or reliability.
10. UPS, [Shipping Services](https://www.ups.com/us/en/shipping/services). Carrier service information; verify current contract, coverage, and terms.
11. FedEx, [Shipping Services](https://www.fedex.com/en-us/shipping.html). Carrier service information; verify current contract, coverage, and terms.
12. Kohavi, Tang, and Xu, *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*, Cambridge University Press, 2020. Practical reference for experimental design and interpretation; shared logistics capacity requires additional interference analysis.
13. Hudgens and Halloran, “Toward Causal Inference With Interference,” *Journal of the American Statistical Association*, 2008, DOI [10.1198/016214508000000292](https://doi.org/10.1198/016214508000000292). Peer-reviewed framework for causal inference when units can affect one another.

## 17. Links

- [Module 21 topic index](_index.md).
- [Module 21 survey and Lab 21.3: Delivery promise test](../../../course/21_ecommerce_supply_chain_analytics.md#lab-213--delivery-promise-test).
- [Fulfillment models](01_ecommerce-fulfillment-models.md).
- [Inventory placement](04_inventory-placement.md).
- [Carrier selection, peak planning, and unit economics entries in the module map](_index.md#3-topics).
- [Simulation and optimization labs module](../../../course/30_simulation_optimization_production_labs.md).

The module map provides entries for sibling lessons that may still be planned.
