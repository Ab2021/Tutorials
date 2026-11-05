# Inventory Placement

> Module 21 — E-commerce Supply Chain Analytics  
> Topic 04 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 07, 10, and 12; basic probability and contribution-margin arithmetic  
> Version 2026.5

## 1. Why this topic exists

An e-commerce merchant may have enough inventory nationally and still miss its delivery promise locally.
A product in a distant fulfillment center is not equivalent to a product near the customer.
A single available unit cannot simultaneously protect two regions against uncertain demand.
The merchant must decide where to position stock before the next customer order is known.

The placement decision is distinct from the purchase decision.
Purchasing determines how many units enter the network and when.
Placement determines which stock keeping unit (SKU) occupies which node.
A node can be a fulfillment center, store, third-party warehouse, or other eligible location.
An order management system (OMS) then chooses a source for an individual order.
That online source choice is constrained by the stock and capacity established earlier.

More distributed inventory can shorten transit distances.
It can also reduce parcel charges and make a faster promise feasible.
But each additional stocking point consumes space, handling, working capital, and operational attention.
Fragmenting inventory can increase stockouts even when the network has enough units in total.
An expensive transfer can move a shortage rather than solve it.
A slow-moving item may be more economical in one central location despite longer delivery.

A placement recommendation therefore cannot be based on speed alone.
Nor can it be based on average shipping cost without considering uncertainty and availability.
You must compare complete, decision-relevant economics on the same demand and time horizon.
You must distinguish an expected service outcome from a customer commitment.
You must also check that the chosen inventory can actually be received, stored, picked, and replenished.

A category owner decides which products and customer segments matter.
A supply planner owns stock positioning and replenishment policy.
The network operations owner validates node and labor capacity.
The carrier and transportation owner supplies feasible service lanes and cost schedules.
Finance approves the economic boundary and capital charge.
The digital product or promise owner controls what checkout communicates.
The OMS owner implements sourcing rules and safeguards.
A cross-functional decision owner arbitrates when service, cost, and risk conflict.

This lesson builds a repeatable placement decision, not a universal rule to decentralize.
The worked example is fictional and its figures are illustrative, not evidence of industry performance.
Money is U.S. dollars, quantities are sellable units, and time buckets are stated explicitly.
All sources of uncertainty should be reconsidered for a real network.

## 2. Learning outcomes

After completing this topic you can:

1. **Distinguish** stock positioning, order routing, replenishment, and delivery promising.
   Exercise 1 tests the decision boundaries and owners.
2. **Build** a SKU-by-node eligibility and cost table with compatible units and time horizons.
   Exercise 2 tests physical and operational feasibility.
3. **Calculate** inventory holding, transfer, and outbound cost for alternative placements.
   Exercise 3 reproduces the business case.
4. **Measure** fast-delivery coverage using an explicit eligible-order denominator.
   Exercise 4 compares a coverage target with actual fulfillment.
5. **Explain** inventory pooling and the effects of demand correlation and lead time.
   Exercise 5 asks for a safety-stock calculation.
6. **Evaluate** placement sensitivity to demand, capacity, and transfer cost.
   Exercise 6 identifies a decision threshold.
7. **Design** an executable allocation, rollout, and exception-control process.
   Exercise 7 requests a controlled implementation plan.
8. **Critique** causal and financial claims in a business case.
   Exercise 8 tests measurement and double counting.

A good answer states the population, grain, horizon, constraints, and decision owner.
A mathematically optimal answer to the wrong business problem is not a useful placement plan.

## 3. The core mental model

### 3.1 Position before routing

```text
Demand by SKU, customer region, day and scenario
                 |
                 v
  Forecast + uncertainty + inventory position
                 |
                 v
  SKU-node eligibility and candidate stock targets
                 |
       +---------+----------+
       |                    |
       v                    v
 Capacity, inventory     Cost, speed, capital,
 and replenishment       service and risk
       |                    |
       +---------+----------+
                 |
                 v
     Approved placement and transfer orders
                 |
                 v
     Receive -> verify available-to-promise stock
                 |
                 v
 Customer order -> OMS routing -> pick -> carrier -> delivery
                 |
                 v
  Actual service, cost, exceptions and inventory aging
                 |
                 v
     Diagnose -> update data, assumptions and next plan
```

Placement is a decision made under incomplete information.
Routing is a later decision made with more information about the actual order and current stock.
Do not assume that the later routing system can undo a poor placement without time or cost.
Do not assume that placing a unit at a node guarantees that it is available when requested.
Receiving delays, reservations, damage, and allocation conflicts can all intervene.

### 3.2 A stock unit has an opportunity cost

Suppose one sellable unit can be stocked in the East or Central node.
Stocking it in the East can make an East customer faster to serve.
That same unit is no longer immediately available to serve the West from Central.
A spare unit can reduce that conflict, but it ties up working capital and may become obsolete.
The economically relevant question is which feasible alternative creates the best expected outcome.

The alternative is called the counterfactual.
It must specify its own routing, capacity, inventory, and service behavior.
A comparison of East placement against an impossible zero-cost baseline is not a business case.
A comparison against the current network must retain existing constraints and stockout behavior.

### 3.3 Coverage is not actual service

A lane is **fast eligible** if an explicitly defined combination of origin, destination, cutoff, and carrier service can meet a defined delivery window.
Coverage is the proportion of the chosen demand population assigned to fast-eligible lanes under stated inventory and capacity assumptions.
Actual on-time delivery additionally requires stock availability, timely processing, carrier performance, and correct customer address data.
Fast coverage is a planning proxy; it is not an on-time-delivery guarantee.
A feasibility matrix should be versioned by day and cutoff when the schedule changes.

An eligible lane can still fail because the carrier misses a scan or a warehouse runs out of labor.
An ineligible lane might occasionally arrive early; that does not make it a reliable promise.
The promise owner should use calibrated end-to-end delivery distributions, not nominal transit time alone.
Placement and promise design must share the same definition of the customer population and service window.

### 3.4 Pooling and fragmentation

A central pool can serve demand from several regions.
Regional fluctuations can partially offset one another if they are not perfectly positively correlated.
Splitting the same units into several pools sacrifices some of that flexibility.
The result may be more safety stock for the same target service or lower service for the same stock.
This is not a law that every distributed network is worse: lead times, minimum order quantities, and allocation rules also change.
Quantify the entire system rather than applying a pooling slogan.

A forecast is an estimate of demand and uncertainty.
A plan is the proposed placement and movement schedule.
A target is the desired service or financial threshold.
A commitment is an obligation or customer-facing promise actually made.
None of these terms means the same thing as an observed outcome.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| SKU | Stock keeping unit: an operationally distinct sellable item, usually a variant. | Treating a product family as a single interchangeable SKU. |
| Node | A physical stockholding or fulfillment location, identified by a stable location ID. | Treating an address as proof of fulfillment capability. |
| Customer region | A destination grouping such as postal-code cluster, at a stated geographic version. | Using the node that shipped an order as the customer's natural region. |
| Origin-destination lane | A node-to-region shipping path with mode, cutoff, calendar, cost, and transit assumptions. | Assuming all parcels on a lane have the same delivery distribution. |
| Placement | Decision about SKU inventory by node and effective date, in units. | Confusing a stocking decision with a per-order routing decision. |
| Assignment | Relationship declaring an item eligible for a node, often a binary flag. | Assuming a binary eligibility flag itself creates stock. |
| Stock target | Desired available or inventory-position quantity at a node, in units. | Adding stock-on-hand to a target without accounting for inbound and reservations. |
| Inventory position | Available stock plus eligible inbound minus relevant commitments, using a documented definition. | Including unconfirmed purchase orders or counting reserved units twice. |
| On-hand inventory | Physically recorded quantity at a location, in units. | Treating quarantined or damaged units as sellable. |
| Available-to-promise (ATP) | Quantity eligible for new orders after defined holds, reservations, and controls. | Assuming physical on-hand equals immediately promiseable stock. |
| Safety stock | Buffer intended to protect against specified uncertainty during a protection period, in units. | Treating any extra inventory as a guaranteed fill rate. |
| Cycle stock | Inventory that fluctuates with replenishment lot size and demand, in units. | Charging the entire target as average cycle stock without explanation. |
| Reorder point | Inventory-position threshold triggering replenishment under a defined policy, in units. | Using it as an end-of-period stock target for every system. |
| Protection period | Time over which demand must be covered before replenishment can restore stock, in days. | Using supplier lead time while ignoring review and receiving delay. |
| Lead time | Elapsed time from an explicitly named release event to available receipt, in days. | Measuring only travel time and omitting queue or dock delay. |
| Holding cost | Cost of holding a unit for a defined period, in currency per unit-period. | Adding a full annual capital rate to a weekly comparison. |
| Capital charge | Financing or opportunity cost associated with inventory value over a period. | Adding it again when the holding rate already includes it. |
| Transfer cost | Incremental transport and handling cost for relocating inventory, in currency per move or unit. | Treating a one-time movement as a recurring weekly expense indefinitely. |
| Order line | One SKU quantity within an order; one order can contain several lines. | Equating line fill with complete-order fill. |
| Split shipment | One order fulfilled in multiple packages or from multiple nodes under a stated definition. | Assuming each extra line necessarily creates another parcel. |
| Delivery promise | Customer-facing delivery date or window under specified conditions. | Calling an internal service target a customer commitment. |
| Fast eligibility | A modeled lane can meet the defined delivery window under its stated assumptions. | Treating eligibility as observed delivery performance. |
| Coverage | Eligible demand assigned to fast-eligible fulfillment divided by an explicit demand population. | Excluding unfilled demand from the denominator without disclosure. |
| Fill rate | Fraction of demand units or lines filled, with event time and denominator specified. | Equating unit fill with the chance of no stockout. |
| Cycle service level | Probability of avoiding a stockout in a defined replenishment cycle. | Treating it as unit fill rate. |
| Demand censoring | Sales observed below potential demand because availability or access constrained purchases. | Training a forecast on lost sales as if they were zero demand. |
| Correlation | Co-movement between demand deviations, a dimensionless number from minus one to one. | Assuming independent regional demand without evidence. |
| Scenario | A coherent possible combination of demand, supply, or operational conditions with stated weight if probabilistic. | Treating scenarios as independent SKU forecasts that cannot jointly occur. |
| Capacity | Available resource per period, such as cubic feet, labor hours, or picks per day. | Using storage capacity as a substitute for pick or outbound capacity. |
| Constraint | A condition every accepted plan must meet, with a unit and evaluation period. | Treating an aspiration as guaranteed feasibility. |
| Objective | The stated cost, contribution, risk, or multi-criteria measure used to rank feasible plans. | Optimizing one term and claiming total network profit. |
| Shadow price | Marginal objective improvement from relaxing a binding constraint under local model conditions. | Assuming a mixed-integer model always has a meaningful stable marginal price. |
| Counterfactual | The outcome under the alternative decision on the same population and horizon. | Comparing a holiday pilot against an unadjusted ordinary week. |
| Rebalancing | Moving stock between nodes after placement or demand changes. | Assuming transfers are instant, free, and operationally unlimited. |

## 5. The mechanics, step by step

### 5.1 Write the decision brief

1. Name the decision and its effective date.
   Example: assign eligible SKUs to Central and East for the next four-week cycle.
2. Specify whether the decision is binary eligibility, quantity placement, or both.
   A node may be eligible even if no transfer is authorized in the current cycle.
3. Define the planning horizon and the review cadence.
   A weekly review with a four-week horizon is different from a four-week one-shot decision.
4. Name the decision owner and the operational approval path.
   An analyst can propose a solution; the owner accepts its service and financial trade-offs.
5. Identify what remains fixed.
   A placement analysis may hold national purchase volume and customer demand constant.
6. Record what is outside scope.
   Examples include new warehouse construction, repricing, and a marketing-led promise change.
7. Specify what happens if no feasible plan exists.
   Escalate the conflict instead of silently dropping a constraint.

The output is a signed brief, not merely a spreadsheet.
Its purpose is to ensure that the model solves the decision the business can actually make.
An accepted brief includes a baseline and an implementation window.

### 5.2 Choose the grain and clock

Use SKU by customer region by day when daily cutoffs and changing demand matter.
Include order and line identifiers when estimating multi-item shipments.
Represent node stock by SKU, location, and timestamp.
A warehouse snapshot at midnight may not describe stock available at a 4 p.m. order cutoff.
Specify time zones and whether the day is local, origin-based, or coordinated universal time.
Use explicit carrier and warehouse operating calendars.
Do not let a calendar-day promise accidentally become a business-day promise.

The placement horizon must align with each cost and constraint.
A monthly holding rate cannot be compared directly with annual parcel savings.
A one-time transfer must be amortized only when that is the approved business-case convention.
Operational capacity is usually measured by shift or day, not just as a monthly total.
A plan that fits a monthly average can still exceed Tuesday's receiving dock capacity.

### 5.3 Build a clean demand population

Start with accepted customer orders and retain their original destination and order timestamp.
Separate canceled, fraudulent, test, and replacement transactions according to a documented policy.
Keep original demand independent of which node happened to fulfill the order.
A historical ship-from node is an outcome of past inventory and routing, not a clean demand origin.
Otherwise a node that was often out of stock may appear to have no demand.

Record order lines, quantities, item variants, shipping address region, and promise shown.
Track demand unavailable because the product could not be purchased where possible.
Use search, page views, substitutions, waitlists, and in-stock periods cautiously to estimate lost demand.
None of those proxies is itself a verified customer order.
State how privacy rules and regional aggregation affect the data.

Create a forecast by SKU, region, and period.
Include promotion, price, seasonality, life-cycle, and stockout information when material.
Use a hierarchy for sparse SKUs instead of trusting unstable item-region averages.
Retain an uncertainty distribution or joint scenario set, not just the mean.
Backtest using only information that would have been available at each historical decision time.
Separate forecast error from error caused by an impossible placement or a node outage.

If a faster promise can change conversion, the demand distribution may depend on the placement decision.
Do not automatically apply a conversion uplift from another market or an uncontrolled test.
For a first diagnostic, hold demand fixed and say so.
For a commercial case, include only a supported incremental contribution estimate and its uncertainty.

### 5.4 Create the SKU-node eligibility matrix

For every SKU and candidate node, record whether storage and fulfillment are permitted.
Check physical dimensions, weight, handling equipment, packaging, and pick method.
Check temperature, shelf life, hazardous-material classification, and segregation requirements where relevant.
Check contractual rights, ownership, channel restrictions, and country-specific restrictions.
Check inventory traceability, serial or lot capture, and returns handling where required.
Check whether the node can receive the item before the plan becomes effective.

Eligibility is not an economic score.
If a node is unable to store or ship an item, mark the combination infeasible.
Do not compensate for a legal or physical prohibition with an attractive cost estimate.
Record the reason and owner for every exclusion so the matrix can be corrected.
A missing product dimension is an exception requiring resolution, not a dimension of zero.

Some eligible combinations need a minimum stock or minimum pick face.
Others require a case-pack multiple or a fixed cost to activate a SKU at a node.
A promotional item may require a mandatory reserve at one node.
Document the reason and expiry date for such overrides.
An override without an owner tends to become a permanent hidden constraint.

### 5.5 Assemble demand, stock, and supply inputs

For each SKU, reconcile available on-hand stock to the warehouse management system (WMS).
Deduct confirmed reservations and exclusions according to the ATP definition.
Show pending receipts separately from physically available stock.
Represent inbound quantities with expected date, confidence, and whether diversion is allowed.
Include outbound transfers already committed to avoid assigning a unit twice.

Define the stock balance for each SKU and node:

```text
ending available stock = starting available stock
                       + usable receipts and inbound transfers
                       - shipped units and outbound transfers
                       - other modeled deductions
```

Every term must use the same unit of measure and effective time.
A receipt that arrives after the order cutoff cannot satisfy a morning promise.
Transfers are out of the origin when dispatched and in the destination only when received and made sellable.
Inventory in transit is neither available at both nodes nor necessarily lost.
Its ownership and capital charge depend on the relevant financial policy.

If supply is limited, choose whether to ration stock by contribution, service priority, contractual obligation, or another rule.
Document that rule before reviewing which customer or SKU would be disadvantaged.
If inventory is unlimited in the model, explicitly state how additional units are obtained and funded.
A plan must not create units simply because a node requires them.

### 5.6 Model demand uncertainty and protection stock

The protection period includes the review interval when applicable, replenishment lead time, and receiving delay.
Its precise definition depends on whether orders are placed continuously or periodically.
Estimate the distribution of demand over that period, not just daily forecast error.
Preserve common promotional shocks and correlated geographic demand in scenarios.
A supplier disruption can also increase lead time and demand exposure simultaneously.

For a simplified continuous-review model with normally distributed protection-period demand,
independent lead time already reflected in the standard deviation,
and a chosen cycle-service target, a common approximation is:

```text
reorder point = mean protection-period demand + z × protection-period standard deviation
safety stock  = z × protection-period standard deviation
```

Here `z` is the standard normal quantile for the chosen cycle-service probability.
A 95% cycle-service target corresponds approximately to `z = 1.645`.
This is not automatically a 95% unit fill rate or a 95% on-time delivery rate.
Intermittent, bounded, seasonal, or highly skewed demand can invalidate the normal approximation.
For those cases use an empirically calibrated distribution or scenario simulation.

If regional demand has standard deviations `sigma_1` and `sigma_2` and correlation `rho`,
the standard deviation of their total is:

```text
sigma_total = sqrt(sigma_1² + sigma_2² + 2 × rho × sigma_1 × sigma_2)
```

This equation assumes the two quantities cover the same protection period and units.
At correlation one, the pooled standard deviation equals the sum of standard deviations.
At lower correlation, pooling can reduce the buffer relative to separate local buffers at the same `z`.
The benefit can be eroded by longer central replenishment lead time or lower delivery speed.
Replenishment policies and cross-region emergency sourcing also affect realized service.

### 5.7 Build cost and service inputs

Build the outbound cost table by node, destination, package characteristics, carrier service, and effective date.
Include base freight, fuel and relevant accessorials, and expected multi-parcel effects where measurable.
Do not use a list rate if the actual marginal contracted rate is available.
Do not include recoverable tax as a cost unless the business-case boundary calls for it.
Separate customer-paid shipping revenue from transportation expense.

Build node handling cost from incremental receiving, storage, picking, packing, and dispatch.
Distinguish truly incremental labor from an allocated fixed overhead that will not change with the decision.
If a constraint makes capacity scarce, record the trade-off instead of assuming infinite free capacity.
A new shift, leased space, or outsourced service may have a discrete activation cost.

Holding cost should reflect the average inventory exposure over the period.
A practical approximation is average unit inventory times unit value times a period-adjusted carrying rate,
plus incremental space, insurance, damage, or obsolescence costs not already included in that rate.
If holding is charged directly in dollars per unit-period, document how that figure was built.
Apply the same convention to both baseline and alternative.
Avoid counting the inventory purchase cost as a fresh expense merely because owned stock is moved.

Transfers can include outbound handling, linehaul, inbound receiving, and in-transit capital.
Some costs are per transfer, some per pallet, and some per unit.
Include transfer lead time and the risk of consuming the origin's available stock before arrival.
For a one-time move, distinguish initial migration expense from recurring replenishment expense.

Create a lane-time feasibility table from actual carrier service calendars and warehouse cutoffs.
A two-day shipping label does not imply delivery two days after a late-night order is placed.
Include order capture, release, pick, pack, carrier induction, transit, and delivery calendar.
If using a probability, identify whether it is empirical on-time probability or a planning assumption.
Use a common eligible-order denominator across competing alternatives.

### 5.8 Encode physical and operational constraints

For every SKU, total allocated opening units cannot exceed usable network inventory plus approved receipts.
For every node, occupied cubic feet, pallet positions, or shelf slots must not exceed available storage.
For every shift, expected or scenario pick workload must fit labor and equipment capacity.
For each outbound cutoff, parcel volume must fit dock, sortation, and carrier pickup capacity.
For each inbound period, transfers must fit receiving slots and staffing.
For each item-node pair, eligibility must be respected.

Represent package or case multiples as integer requirements when they actually bind.
A model allowing 3.5 physical units is not an executable SKU allocation.
A model allowing one item in half of a pick face is also misleading when a full face is required.
Account for space used by reserve stock and safety stock, not just average shipped units.
Do not double count a pallet position and its converted cubic feet unless both are distinct limits.

Coverage can be a hard constraint, such as at least 90% of eligible demand on fast lanes.
It can instead be an objective with an explicit cost of service failure.
The two formulations answer different questions.
If a hard target is infeasible, report the best achievable service and the binding constraints.
Only an authorized business owner can accept a relaxed target.

### 5.9 Formulate a transparent decision model

A compact single-period model can use these sets and variables:

```text
I = SKUs; N = stocking nodes; R = customer regions; S = demand scenarios
x[i,n]   = units of SKU i placed at node n before demand
z[i,n]   = 1 if SKU i is activated at node n, otherwise 0
q[i,n,r,s] = units fulfilled from node n to region r in scenario s
u[i,r,s] = unfilled units for SKU i and region r in scenario s
```

The allocation `x` is chosen before the demand scenario is known.
The routing `q` and unmet demand `u` may respond to the realized scenario.
That distinction is called a two-stage decision structure.
It prevents the model from placing different opening stock for each future scenario with perfect foresight.

For each SKU and scenario, fulfillment from a node cannot exceed the placed stock at that node.
For each SKU-region-scenario, fulfilled plus unfilled units equals scenario demand.
Total placement for a SKU cannot exceed available network stock.
Node resource use cannot exceed node capacity.
Link `x` to activation and eligibility using a valid upper bound on possible units.
Use integer variables when units or pack sizes require them.

A generic objective minimizes:

```text
placement holding and activation costs
+ transfer costs
+ scenario-weighted outbound and handling costs
+ scenario-weighted cost of unmet demand
+ any separately modeled late-delivery or split-shipment costs
```

All costs must be expressed on the same horizon and currency basis.
The unmet-demand penalty must represent a documented economic or service consequence.
Do not include both a full lost-contribution penalty and the same contribution loss in another term.
A service floor may supplement the objective when a cost penalty cannot represent a policy obligation.
If scenario weights are probabilities, they must sum to one.

A weighted average service constraint can hide a severe failure in a low-probability scenario.
A per-scenario service constraint is stronger and may be much more expensive or infeasible.
A chance constraint instead limits the probability of falling below a stated threshold.
The correct form depends on the policy and the credibility of scenario probabilities.
Show the decision owner the distribution of outcomes, not only the average objective value.

The generic formulation omits replenishment timing, order-level baskets, and dynamic capacity unless added explicitly.
Do not claim that it solves these issues merely because it contains an inventory variable.
For multiple periods, add inventory balances and lead-time-indexed movement decisions.
For baskets, use order or basket-pattern variables and model whether lines are shipped together.
For very large assortments, clustering or decomposition can help, but validate any approximation.

### 5.10 Solve, compare, and diagnose

First evaluate the current or proposed baseline under the same data and constraints.
Then evaluate the alternative using identical accounting and service definitions.
Use a simple heuristic or enumeration for a small candidate set.
Use a mixed-integer optimizer when the number of combinations and constraints warrants it.
A solver status of feasible is not the same as proof of global optimality.
Record solve time, optimality gap where available, input version, and any constraint relaxation.

Review the recommended SKU-node list with category and operations owners.
Inspect extreme recommendations and high-impact assumptions before accepting the aggregate result.
A single incorrect dimension or transfer cost can move thousands of units in a model.
Check whether the result depends on forecasts that are especially uncertain or censored.
Check that the chosen transfers arrive before their first planned demand.

Compare cost, fast coverage, fill, and resource use separately.
A lower expected cost with a large downside loss may be inappropriate for the business.
A higher modeled fast coverage with an unchanged promise may not generate incremental revenue.
A service improvement can still have value as risk reduction, but define and measure it.

### 5.11 Run sensitivities and value the decision

Vary the assumptions that could reverse the recommendation.
Examples include destination demand share, transfer cost, carrier rate, holding rate, and service target.
Vary correlated demand, replenishment delay, and node capacity together when they are causally related.
Changing one variable at a time is useful for interpretation but does not capture joint stress.
Show both the threshold at which the preferred plan changes and a plausible range if one is supported.

Separate reversible and irreversible costs.
A one-time transfer is sunk after execution but is relevant before the move.
Future replenishment costs remain relevant for recurring placement decisions.
Incremental margin may depend on conversion, cancellation, returns, and customer retention.
Only include causal changes supported by evidence; show a no-uplift case as a reference.

Express the result as a decision and its conditions.
For example: place item A in East if the approved transfer cost remains below the calculated threshold,
East has sufficient capacity, and the service target is measured on the same eligible population.
That is more actionable than saying the optimizer likes East.

### 5.12 Execute, monitor, and reverse

Convert approved targets into transfer orders with item, quantity, origin, destination, and arrival deadline.
Check whether stock is free of reservations before releasing a transfer.
Use a staging status so in-transit units are not promised from both nodes.
Require receipt and quality confirmation before increasing destination ATP.
Ensure the OMS sourcing rule uses the same eligibility matrix and cutoff calendar as the plan.

Start with a controlled subset when operationally appropriate.
Set a service guardrail, a financial guardrail, and a rollback owner before launch.
Choose an observation window long enough to include normal day-of-week and replenishment effects.
Monitor availability, cost, delivery outcomes, and exceptions at both node and customer-region grain.
Do not interpret a short period with no failures as proof of a high reliability level.

A rollback is not just a switch in a dashboard.
State what happens to inventory already in transit, open orders, reservations, and customer promises.
Freeze or amend new transfers, reconcile stock, update ATP, and notify affected owners.
If customer commitments have already been made, preserve or remediate them through the approved process.
Record the reason, effective timestamp, and evidence for any manual override.

## 6. Worked example

### 6.1 Decision and boundaries

A fictional retailer is choosing a four-week opening placement for two SKUs.
It has 100 units of SKU A and 40 units of SKU B available at Central.
There are two customer regions, East and West, and two possible nodes, Central and East.
No replenishment arrives within the four-week comparison.
For this deterministic example, every demand unit is a separate one-unit order.
There are no returns, cancellations, split shipments, or taxes in the stated cost boundary.

The company can move SKU A, SKU B, both, or neither to East.
A candidate move places exactly the forecast East-region demand for that SKU at East.
East can hold at most 75 units for this decision, with each unit consuming one capacity unit.
All other opening stock remains at Central.
The company requires at least 90% of forecast eligible orders on fast-eligible lanes.
This is an internal target, not a promise to customers or an observed delivery rate.

A fast lane is assumed to be East-to-East or Central-to-West.
Central-to-East and East-to-West are not fast eligible.
These are fictional binary assumptions, not statements about real geography or carrier service.
The deterministic comparison routes East demand from East if it is stocked there, otherwise from Central.
West demand always routes from Central.
The forecast quantities below exactly match the network's available units.

### 6.2 Inputs

| SKU | East forecast, units | West forecast, units | Total forecast, units | Opening stock, units |
|---|---:|---:|---:|---:|
| A | 70 | 30 | 100 | 100 |
| B | 10 | 30 | 40 | 40 |
| Total | 80 | 60 | 140 | 140 |

| Origin | East destination, dollars/unit | West destination, dollars/unit | Fast-eligible destination |
|---|---:|---:|---|
| Central | 7 | 4 | West |
| East | 4 | 9 | East |

The four-week holding proxy is $0.30 per opening unit at Central and $0.80 per opening unit at East.
For simplicity this proxy is charged on opening stock for the entire horizon, even though units ship over time.
It is a stated conservative modeling convention, not a claim that actual average inventory equals opening inventory.
The same convention is applied to every candidate.
The $40 transfer activation charge applies once per SKU moved to East for this four-week decision.
It includes the incremental transfer and receiving costs assumed for the candidate quantities.
The example treats it as a fixed charge per moved SKU, not $40 per unit.
There are no additional handling costs or revenue differences in this simplified boundary.

### 6.3 Baseline: no East stock

All 140 units remain at Central.
Central-to-East demand is 80 units at $7 per unit, costing $560.
Central-to-West demand is 60 units at $4 per unit, costing $240.
Outbound cost is therefore $800 for the four-week period.
Holding is 140 units times $0.30 per unit, or $42.
There is no East transfer activation cost.
The modeled total is $800 + $42 = $842.

All 60 West orders are on the assumed fast lane.
None of the 80 East orders is on a fast lane.
Fast coverage is 60 / 140 = 42.86%, rounded to two decimal places.
The baseline misses the 90% target.
It is still needed as the financial and operational counterfactual.

### 6.4 Alternative A: move SKU A's East demand

Move 70 units of A to East and leave 30 A units at Central.
All 40 B units remain at Central.
East storage use is 70 of 75 capacity units.
East-to-East outbound for A is 70 × $4 = $280.
Central-to-West outbound for A is 30 × $4 = $120.
Central-to-East outbound for B is 10 × $7 = $70.
Central-to-West outbound for B is 30 × $4 = $120.
Total outbound is $280 + $120 + $70 + $120 = $590.

East holding is 70 × $0.80 = $56.
Central holding is the remaining 70 units × $0.30 = $21.
The holding proxy is $77 in total.
The single SKU activation transfer costs $40.
The total modeled cost is $590 + $77 + $40 = $707.
The modeled saving versus baseline is $842 − $707 = $135 over four weeks.

The fast-eligible units are 70 East A units plus 60 West units.
Coverage is 130 / 140 = 92.86%.
This exceeds the 90% forecast-coverage target by 2.86 percentage points.
It does not demonstrate 92.86% actual on-time delivery.
It also does not demonstrate that the target will hold when actual demand differs from the forecast.

### 6.5 Alternative B: move SKU B's East demand

Move 10 B units to East, leaving 100 A and 30 B units at Central.
East storage use is 10 of 75 capacity units.
Outbound is $770: $800 baseline outbound less 10 × ($7 − $4) = $30.
Holding is 10 × $0.80 plus 130 × $0.30 = $47.
The activation transfer costs $40.
Total modeled cost is $770 + $47 + $40 = $857.
That is $15 more than the baseline.
Fast coverage is (60 + 10) / 140 = 50.00%.
This alternative misses the target and costs more within the stated boundary.

### 6.6 Both moves and the decision

Moving both items would require 70 + 10 = 80 East capacity units.
The East limit is 75 units, so that candidate is infeasible.
If the limit were relaxed and all other assumptions held, its computed cost would be $722.
Its forecast fast coverage would be 140 / 140 = 100%.
Those figures describe a counterfactual with more capacity, not an executable current plan.

| Candidate | East stock, units | Outbound, dollars | Holding, dollars | Transfer, dollars | Total, dollars | Fast coverage | Feasible? |
|---|---:|---:|---:|---:|---:|---:|---|
| Neither | 0 | 800 | 42 | 0 | 842 | 42.86% | Capacity yes; service no |
| A only | 70 | 590 | 77 | 40 | 707 | 92.86% | Yes |
| B only | 10 | 770 | 47 | 40 | 857 | 50.00% | Capacity yes; service no |
| Both | 80 | 560 | 82 | 80 | 722 | 100.00% | No: East capacity |

Recommend the A-only plan for the deterministic four-week case.
The supply planner owns the 70-unit movement, subject to operations and finance approval.
The promise owner must not equate the modeled coverage with a customer-facing delivery guarantee.
If the transfer cannot arrive before the effective date, the recommendation must be recomputed or delayed.
If expected realized demand or carrier capacity differs materially, re-evaluate before committing.

### 6.7 Break-even and timing

For A, the outbound saving is 70 × $3 = $210.
The incremental holding proxy is 70 × ($0.80 − $0.30) = $35.
Before transfer cost, the modeled economic benefit is $175 over four weeks.
Therefore A's transfer activation cost breaks even at $175 in this cost-only comparison.
At $40, the difference is $175 − $40 = $135.
At a transfer charge above $175, A would cost more than the baseline, although it could still be needed for the 90% service constraint.
The threshold assumes the same demand, routing, and holding convention.

For B, the outbound saving is $30 and the incremental holding is $5.
Its cost-only transfer break-even is $25.
The assumed $40 transfer exceeds that threshold by $15.
The A-only and B-only calculations are not additive when the shared East capacity constraint binds.
A service requirement may justify a cost increase; finance should show the incremental cost explicitly.

### 6.8 Demand mismatch stress

Keep A's opening stock at 70 East and 30 Central; B remains entirely Central.
Suppose actual A demand is 50 East units and 50 West units, with B still 10 East and 30 West.
Assume East cannot send stock to West during this horizon and Central supplies West first.
East supplies 50 A units and retains 20 unsold A units.
Central supplies 30 of 50 West A units, leaving 20 West A units unfilled.
B's 40 units still ship from Central.
Thus 120 of 140 demanded units are filled, and 20 units of network stock remain stranded in East.
Fast-eligible fulfilled units are 50 East A, 30 West A, and 30 West B: 110 units.
Fast coverage over all 140 demanded units is 110 / 140 = 78.57%.
Among only the 120 filled units it is 110 / 120 = 91.67%; that narrower denominator hides the unfilled orders.

Now suppose A demand is 90 East and 10 West, with the same opening allocation and B demand.
East fills 70 A units; Central fills 10 West A and the remaining 20 East A units.
All 140 units are filled, but 20 East A units travel on the slow Central-to-East lane.
Fast-eligible units are 70 East A, 10 West A, and 30 West B, again 110 of 140 or 78.57%.
The first stress exposes a shortage caused by a stranded regional unit.
The second exposes slow fallback despite sufficient total stock.
Neither stress has an assigned probability, so neither is an expected outcome.
A real stochastic business case must define scenario weights and consequences before averaging.

### 6.9 Managerial interpretation

The deterministic plan is the only candidate satisfying both the capacity and service target.
Its expected financial superiority is not established by a single point forecast.
The owner should check the likelihood and cost of the two demand-mismatch patterns.
Emergency transshipment may improve outcomes, but only if its time, capacity, and cost are modeled.
More East stock can protect East demand while making West shortages more likely.
More Central stock preserves pooling while reducing forecast fast coverage.
Those are genuine trade-offs, not a data-quality issue that can be wished away.

## 7. Code, query, or calculation

The following Python 3 program enumerates the four deterministic candidate placements.
It uses only the standard library and does not require a solver or external files.
It reproduces the worked example's four-week cost and coverage numbers.
All demand is filled in the forecast case because moved quantities equal East demand.
It deliberately does not simulate the demand stresses or calculate actual on-time delivery.

```python
from itertools import product

# Four-week quantities and U.S. dollar costs; each demand unit is one order.
demand = {
    "A": {"East": 70, "West": 30},
    "B": {"East": 10, "West": 30},
}
rate = {
    ("Central", "East"): 7.0,
    ("Central", "West"): 4.0,
    ("East", "East"): 4.0,
    ("East", "West"): 9.0,
}
holding = {"Central": 0.30, "East": 0.80}
fast_lanes = {("Central", "West"), ("East", "East")}
transfer_per_sku = 40.0
east_capacity = 75
coverage_target = 0.90
skus = tuple(demand)
total_demand = sum(sum(regions.values()) for regions in demand.values())


def evaluate(moved):
    """Move forecast East demand for each SKU named in moved."""
    outbound = 0.0
    holding_cost = 0.0
    fast_units = 0
    east_units = 0

    for sku in skus:
        east_demand = demand[sku]["East"]
        west_demand = demand[sku]["West"]
        placed_east = east_demand if sku in moved else 0
        placed_central = east_demand + west_demand - placed_east
        east_units += placed_east
        holding_cost += placed_east * holding["East"]
        holding_cost += placed_central * holding["Central"]

        east_origin = "East" if sku in moved else "Central"
        routes = ((east_origin, "East", east_demand),
                  ("Central", "West", west_demand))
        for origin, destination, units in routes:
            outbound += units * rate[(origin, destination)]
            if (origin, destination) in fast_lanes:
                fast_units += units

    transfer = len(moved) * transfer_per_sku
    coverage = fast_units / total_demand
    capacity_ok = east_units <= east_capacity
    service_ok = coverage >= coverage_target
    return {
        "moved": "+".join(sorted(moved)) or "none",
        "east_units": east_units,
        "outbound": outbound,
        "holding": holding_cost,
        "transfer": transfer,
        "total": outbound + holding_cost + transfer,
        "coverage": coverage,
        "capacity_ok": capacity_ok,
        "service_ok": service_ok,
    }


results = []
for flags in product((False, True), repeat=len(skus)):
    moved = {sku for sku, flag in zip(skus, flags) if flag}
    results.append(evaluate(moved))

for row in results:
    print(
        f"{row['moved']:5s} "
        f"east={row['east_units']:3d} "
        f"outbound=${row['outbound']:.2f} "
        f"holding=${row['holding']:.2f} "
        f"transfer=${row['transfer']:.2f} "
        f"total=${row['total']:.2f} "
        f"coverage={row['coverage']:.2%} "
        f"capacity={row['capacity_ok']} "
        f"service={row['service_ok']}"
    )

feasible = [r for r in results if r["capacity_ok"] and r["service_ok"]]
if feasible:
    best = min(feasible, key=lambda r: r["total"])
    print(f"best feasible: {best['moved']} (${best['total']:.2f})")
else:
    print("no feasible plan: escalate the service/capacity conflict")
```

Expected output:

```text
none  east=  0 outbound=$800.00 holding=$42.00 transfer=$0.00 total=$842.00 coverage=42.86% capacity=True service=False
B     east= 10 outbound=$770.00 holding=$47.00 transfer=$40.00 total=$857.00 coverage=50.00% capacity=True service=False
A     east= 70 outbound=$590.00 holding=$77.00 transfer=$40.00 total=$707.00 coverage=92.86% capacity=True service=True
A+B   east= 80 outbound=$560.00 holding=$82.00 transfer=$80.00 total=$722.00 coverage=100.00% capacity=False service=True
best feasible: A ($707.00)
```

The displayed `service` flag is a deterministic forecast-coverage test.
It does not test actual delivery reliability, availability during the period, or a contractual obligation.
The candidate space includes only the all-or-nothing East-demand move for each SKU.
It does not consider moving 35 A units, for example, or using East as a West fallback.
Therefore the program identifies the cheapest feasible candidate in this small set, not the global optimum of every possible network policy.
For a real implementation, add stock balances, period-specific capacity, demand scenarios, and actual routing rules.
Keep data validation and solver status in the production workflow rather than inferring feasibility from a printed cost.

## 8. Data, standards, and regulatory context

### 8.1 Data contracts

A placement dataset needs a stable SKU identifier, node identifier, customer-region identifier, and effective timestamps.
The SKU master should include unit dimensions, unit weight, handling class, shelf life where relevant, and pack hierarchy.
The node master should include physical location, operating calendar, capabilities, and capacity definitions.
A location ID must not silently change meaning when a warehouse is renamed or a store is closed.
The customer-region mapping must carry its geographic version and effective date.

The inventory feed needs on-hand, reserved, damaged, quarantined, and available quantities.
It also needs receipt and adjustment events to reconcile inventory snapshots.
The order feed needs order and line IDs, ordered quantity, order timestamp, customer region, promise, and cancellation status.
The shipment feed needs origin, carrier service, packages, dispatch, delivery, and exception events.
The transfer feed needs release, dispatch, receipt, and sellable timestamps.
Financial data needs cost type, currency, time period, contract version, and whether the amount is incremental.

A data contract should name the producer, consumer, refresh frequency, allowed values, and escalation owner.
It should specify whether quantities are eaches, cases, or pallets and how conversions are maintained.
It should identify the time zone and event semantics for every timestamp.
A `delivered_at` value without a defined event source and correction policy can be misleading.
Missing data must be distinguished from a true zero.

### 8.2 Identifiers and interoperability

The GS1 Global Trade Item Number (GTIN) identifies trade items within the GS1 system.
A merchant's SKU is an internal operational identifier and is not necessarily one-to-one with a GTIN.
The mapping can change with pack size, market, and variant; validate it before joining datasets.
The GS1 Global Location Number (GLN) identifies parties and locations under that standard.
Neither a GTIN nor a GLN establishes that an item is available for sale at a node.

GS1 electronic product code information services (EPCIS) describe a standard approach to sharing visibility event data.
Events can improve traceability of movement across organizations when parties implement compatible semantics.
A standard event format does not guarantee accurate scans or complete custody records.
A retailer must still reconcile source events, late arrivals, and corrections.
Electronic data interchange (EDI) can support trading-partner messages such as orders and shipment notices.
Message agreement and partner-specific implementation rules still determine the practical interpretation.

These standards are interoperability tools, not a mandated universal placement algorithm.
Whether a specific standard is contractually or legally required depends on the product, jurisdiction, and trading relationship.
Check the current standard and applicable contract before declaring a field mandatory.

### 8.3 Financial and legal boundaries

Inventory accounting and managerial opportunity-cost analysis answer different questions.
IAS 2 describes accounting for inventories under the International Financial Reporting Standards framework where applicable.
An internal capital charge used to compare placement options is not automatically an accounting expense under IAS 2.
Finance must reconcile the decision model to the organization's actual accounting framework and reporting policy.
Do not infer that a model's holding proxy is an audited financial statement number.

In the United States, the Federal Trade Commission's Mail, Internet, or Telephone Order Merchandise Rule addresses shipment representations and delays in covered transactions.
The agency's business guidance explains the rule and its conditions.
The rule does not prescribe a particular warehouse location or require the 90% target in the fictional case.
A model's fast-lane coverage is not legal compliance evidence by itself.
The promise owner and counsel should review applicable current obligations and communications.
Different jurisdictions, product categories, and terms may impose different requirements.
This lesson is educational and is not legal advice.

Some products require special storage, transport, product safety, or traceability controls.
Identify the relevant law, classification, carrier rules, and facility permits for the actual goods and locations.
Do not infer a product's regulatory status from a generic `hazmat` flag without an authoritative classification process.
Document who validates the restrictions and how their effective dates reach the eligibility matrix.

### 8.4 Privacy and data retention

Customer addresses are sensitive operational data.
Use the minimum geographic detail necessary for the placement decision.
A postal-code cluster may be sufficient for model training while a protected routing service handles full addresses.
Apply the organization's access, retention, deletion, and incident procedures.
Regional aggregation can reduce exposure but can also hide service disparities within the region.
Review the trade-off with privacy and analytics owners rather than assuming aggregation eliminates all risk.

### 8.5 Reconciliation and data quality

Reconcile total accepted order lines between the OMS and the demand extract.
Reconcile unit movement between WMS inventory snapshots and receipt, shipment, transfer, and adjustment events.
Reconcile carrier invoice totals to the rate assumptions at the chosen grain.
Investigate material unmatched IDs, negative ATP, duplicate shipments, and unexplained inventory adjustments.
Track both coverage of source data and freshness at the plan's decision timestamp.
A daily feed can be too stale for an intraday promise even if its row count is complete.

Version the model inputs, assumptions, code, and output plan together.
Record the forecast issue time to prevent future demand information leaking into a backtest.
Record the effective versions of rates, calendars, and eligibility restrictions.
Keep the approved baseline as well as the recommended plan.
A rerun should explain changes caused by new data separately from changes caused by new policy.

## 9. Industry practice and case evidence

### 9.1 Why public cases need boundaries

A company's network announcement can establish that the company describes a practice.
It rarely reveals the complete item-level optimization, counterfactual, costs, or experimental design.
A named case should therefore not be used as proof that the same approach will save another retailer a given amount.
The examples below identify public claims and separate them from analytical interpretation.
They do not validate the fictional four-week case.

### 9.2 Amazon regionalization

Amazon describes changes to its U.S. fulfillment network that position inventory closer to customers and support faster delivery.
Its company account discusses a regionalized network and the relationship between placement and delivery distance.
This is evidence of the company's stated network approach, not independent proof of a general optimal placement rule.
The public description does not provide the full SKU-by-node cost matrix or an independently audited counterfactual for this lesson.

[INFERENCE] Regional positioning creates a planning problem involving local demand forecasts and inventory availability.
[INFERENCE] The appropriate number of stocking nodes for a specific item depends on demand and costs, not on the existence of regional facilities alone.
A learner should ask which population, time period, and service measure support any claimed speed benefit.
A company-level average cannot establish the result for a slow-moving SKU or a remote region.

### 9.3 Walmart and store fulfillment

Walmart describes stores as part of its fulfillment network, including pickup and delivery services.
Its public materials describe using local stores to reach customers, but the mix of services and availability depends on location.
A store is not automatically interchangeable with a fulfillment center.
It shares labor, space, and inventory with in-store customers and may have different substitution behavior.

[INFERENCE] Ship-from-store or store-based fulfillment can shorten distance for some customers while increasing local workload.
[INFERENCE] A credible placement case must include the opportunity cost of shelf availability and the store's pick capacity.
The company's public description is not an independent estimate of those costs in a particular store.
A practitioner should obtain local demand, service, and operational records before applying the analogy.

### 9.4 Marketplace fulfillment as a placement decision

Amazon's Fulfillment by Amazon program describes the company storing, picking, packing, and shipping seller inventory under program terms.
Walmart Fulfillment Services similarly describes logistics services offered to eligible marketplace sellers.
These are company descriptions of offered services, not independent evidence of profitability for every seller.
Fees, eligibility, inventory ownership, and service terms can change.
A seller choosing such a program must compare fees, storage duration, stock placement rules, and alternatives.

[INFERENCE] Outsourcing execution does not remove the need to manage inventory exposure or stockout risk.
[INFERENCE] A seller may have limited visibility into or control over the platform's internal node-level allocations.
A model should distinguish decisions the seller can actually control from those reserved to the platform.
Do not present an internal node allocation as the seller's executable decision if the contract does not allow it.

### 9.5 What academic evidence contributes

The inventory-pooling literature explains why aggregate uncertainty can be lower than the sum of local uncertainties.
Eppen's work analyzes pooling benefits under distributional and cost assumptions.
Those results are a reason to model dependence and allocation, not a guarantee that centralization always wins.
A network with strict delivery windows or very different replenishment times can favor a different arrangement.

A controlled operational evaluation should compare actual outcomes against a credible counterfactual.
Randomized eligible SKU-region cohorts can help when business and operational constraints allow them.
When randomization is not feasible, use a documented matched or phased comparison and inspect pre-period trends.
Control for promotions, stock availability, seasonality, and network changes.
Report uncertainty and unintended effects, including shifts in workload or stockouts to other nodes.
A before-after cost drop alone cannot isolate the placement decision.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Demand attributed to shipping node | Forecast follows past routing changes | Fulfillment origin is mistaken for customer location. | Wrong regional stock targets. | Forecast from destination and order time. |
| Stockout-censored sales | Zero sales coincide with unavailable inventory | Constrained demand is treated as true demand. | Chronic underplacement and lost sales. | Flag censoring; estimate demand using supported methods. |
| National stock treated as local ATP | Healthy network inventory but regional cancellations | Units are held at the wrong node or reserved. | Promises fail despite adequate total stock. | Use node-time ATP and transfer lead time. |
| Double-counted inbound | Destination ATP rises at dispatch while origin remains available | Units appear at two nodes simultaneously. | Overselling. | Use in-transit state and receipt confirmation. |
| Nominal transit used as delivery | Carrier transit equals displayed promise without processing time | Pick, cutoff, calendar, and delay are omitted. | Late customer orders. | Model end-to-end delivery and cutoff calendars. |
| Holding costs on incompatible horizons | Annual rates compared with weekly shipping | Time units differ across alternatives. | False savings or false losses. | Normalize each component to the decision horizon. |
| Transfer treated as recurring | Same one-time move charged every week | Migration and steady-state costs are conflated. | Good placement rejected. | Separate one-time and recurring cash flows. |
| Transfer ignored | Many small rebalances occur after launch | Frictionless movement assumed. | Parcel savings disappear in handling costs. | Include movement, lead time, and receiving constraints. |
| Capacity averaged across periods | Monthly utilization is low but peak shift is overloaded | Aggregate capacity hides local peaks. | Late dispatch and labor overtime. | Model shift or day bottlenecks. |
| Space modeled without labor | Shelf space available but pick backlog rises | Storage and throughput are distinct resources. | Promise failures and congestion. | Include pick, pack, and dock limits. |
| Fast coverage denominator shrinks | Reported speed rises while fill falls | Unfilled orders excluded from service measure. | Misleading performance and poor customer experience. | Report coverage over all eligible demand plus fill. |
| Cycle service confused with fill | 95% safety-stock target called 95% order fill | Different events have different probabilities. | Misstated commitment and stock level. | Define denominator and event explicitly. |
| Independent regional scenarios assumed | All regions miss forecasts together | Common shocks ignored. | Understated risk and buffer. | Preserve joint scenarios and correlation. |
| Perfect-foresight placement | Each scenario gets a different initial stock plan | Future demand leaks into first-stage decision. | Unrealistically low modeled cost. | Enforce common pre-demand placement. |
| Basket costs omitted | Parcel count rises after regionalization | Lines route to different nodes. | Higher cost and fragmented delivery. | Analyze order-level baskets and split parcels. |
| Conversion uplift double counted | Revenue and margin both rise by full sales value | Incremental sales treated as pure profit. | Overstated business case. | Use incremental contribution after relevant costs. |
| Ineligible SKU sent to a node | Manual holds or compliance blocks increase | Restrictions absent or stale in master data. | Delays, unsafe handling, or violation. | Validate eligibility with the responsible specialist. |
| Plan not reflected in OMS | Units received but source rules remain unchanged | Planning and execution systems are disconnected. | Stock sits idle and shipping costs persist. | Test effective-dated routing and ATP integration. |
| Manual override becomes permanent | Exceptions have no expiry or owner | Temporary business rule is never reviewed. | Hidden constraints and degraded model quality. | Log reason, owner, review date, and outcome. |
| Pilot compared with wrong baseline | Improvement appears only in holiday period | Seasonality or promotion drives the change. | Unsupported causal claim. | Use comparable controls and pre-period diagnostics. |

A red flag is evidence for investigation, not proof of a single root cause.
The same symptom can arise from forecasting, capacity, data, or execution failure.
Assign an owner and timestamp to each exception so that diagnosis leads to action.

## 11. Assumptions and limits

| Assumption | Why needed in the worked case | What breaks it | How to test or relax it |
|---|---|---|---|
| Forecast quantities equal actual demand | Allows deterministic enumeration. | Promotions, seasonality, or regional shocks. | Backtest joint errors and evaluate weighted scenarios. |
| Demand is unchanged by promise | Isolates logistics economics. | Faster dates change conversion or channel mix. | Run a controlled promise experiment and model incremental contribution. |
| Each unit is a separate order | Makes unit and order coverage equal. | Multi-unit or multi-SKU baskets. | Use order-level data and package-level routing. |
| No replenishment arrives | Keeps stock balances single-period. | Regular receipts or supplier delays. | Add time-indexed receipts and lead-time uncertainty. |
| Opening stock is all sellable | Makes total available stock explicit. | Reservations, damage, or quality holds. | Reconcile ATP and model availability by timestamp. |
| East activation costs $40 per SKU | Represents a simple fixed transfer cost. | Per-unit, pallet, or distance-dependent charges. | Fit an actual rate card and include discrete handling. |
| Holding applies to all opening units | Supplies a reproducible common cost proxy. | Units sell throughout the horizon. | Estimate average on-hand inventory by node and time. |
| Lane cost is constant per unit | Makes the arithmetic transparent. | Weight bands, surcharges, multi-parcel orders. | Use package characteristics and effective rate schedules. |
| Binary fast-lane eligibility is accurate | Permits a simple service constraint. | Cutoffs, weather, carrier failures, and calendar changes. | Calibrate delivery-time distributions and monitor drift. |
| East capacity is 75 identical units | Illustrates a binding node constraint. | SKUs have different cubic volume or handling needs. | Model space, slots, labor, and peak throughput separately. |
| Transfers arrive before the horizon | Allows East stock to serve the first demand. | Receiving backlog or transport delay. | Add release, transit, receiving, and availability timestamps. |
| Forecast routing uses the stated lanes | Makes candidate costs reproducible. | OMS chooses a different origin or order priority. | Simulate and test actual sourcing rules. |
| No returns or cancellations | Keeps forward economics focused. | Material reverse volume or order attrition. | Add cohort-based return, recovery, and cancellation costs. |
| No unfilled demand in the forecast | Allows direct cost comparison for candidates. | Misallocation under realized demand. | Include shortage penalties and service denominators in scenarios. |
| Scenario probabilities are credible | Needed for an expected-cost recommendation. | Structural breaks or too little history. | Report stress outcomes and sensitivity without invented weights. |
| Additional capacity has no hidden cost | Needed to interpret the hypothetical both-SKU plan. | Added shifts, rent, or service fees. | Price the actual capacity change before approval. |

The example's baseline and alternative share the same simplifying assumptions.
That makes the arithmetic comparable, but it does not make the assumptions realistic.
A decision owner should judge whether omitted effects are small enough for the decision at hand.
When they are not, expand the model before using its result as a commitment.

## 12. Exercises

### Exercise 1 — Separate the decisions

A retailer forecasts 500 units for a region next week.
A planner proposes moving 200 units to a local node.
The checkout displays a two-day promise, and the OMS chooses an origin after each order arrives.

**Deliverable:** Label forecast, plan, commitment, and routing decision; name one owner for each.

**Acceptance criteria:** All four labels are distinct; the answer explains why the forecast does not guarantee fulfillment.

**Time:** 10 minutes.

### Exercise 2 — Eligibility and capacity

A node has 100 cubic feet of available space.
SKU C uses 2 cubic feet per unit, and 30 units are proposed.
SKU D uses 3 cubic feet per unit, and 15 units are proposed.
The node is not approved to handle SKU D's required temperature range.

**Deliverable:** Calculate required space and state whether the joint placement is executable.

**Acceptance criteria:** The answer checks both the numeric capacity and the independent eligibility restriction.

**Time:** 10 minutes.

### Exercise 3 — Rebuild the business case

Use the worked example's A-only candidate.
Change its transfer activation charge from $40 to $100 for the four-week horizon.
Leave all other assumptions unchanged.

**Deliverable:** Calculate revised total cost, saving or loss versus baseline, and fast coverage.

**Acceptance criteria:** The calculation keeps the baseline at $842 and distinguishes cost from the service constraint.

**Time:** 15 minutes.

### Exercise 4 — Choose the denominator

A region receives 200 eligible one-unit orders.
It fills 180 orders, of which 153 are on a fast-eligible lane.
The remaining 20 orders are unfilled.

**Deliverable:** Calculate fast coverage over all eligible orders, fast share among filled orders, and unit fill rate.

**Acceptance criteria:** Each ratio has a named denominator; explain which ratio could hide shortages.

**Time:** 10 minutes.

### Exercise 5 — Pooling and safety stock

Two regions have protection-period demand standard deviations of 12 and 16 units.
Their forecast errors have correlation zero.
Assume normal demand, the same protection period, and `z = 1.645`.
Ignore integer rounding until the final comparison.

**Deliverable:** Calculate pooled standard deviation and compare pooled with separate safety stock.

**Acceptance criteria:** Show the variance formula and distinguish cycle service from unit fill.

**Time:** 15 minutes.

### Exercise 6 — Sensitivity threshold

For SKU A in the worked example, the per-unit outbound saving is $3 and incremental holding is $0.50.
Seventy units move to East.
Assume the transfer activation cost is the only unknown.

**Deliverable:** Find the four-week transfer-cost break-even and explain what happens if the actual charge is $190.

**Acceptance criteria:** State the $175 threshold and discuss the separate 90% service constraint.

**Time:** 10 minutes.

### Exercise 7 — Design an execution and rollback plan

A merchant wants to move a seasonal item into 20 stores before a promotion.
Store inventory accuracy is uneven, and carrier pickup capacity varies by day.
The checkout promise must remain stable during the migration.

**Deliverable:** Write a one-page implementation plan and rollback decision table.

**Acceptance criteria:** Include data owners, ATP states, capacity checks, pilot scope, guardrails, open-order treatment, and a named rollback owner.

**Time:** 30 minutes.

### Exercise 8 — Critique the claimed uplift

A team reports that a new placement reduced average shipping cost by $0.80 per shipped order.
It compares a promotion week with a non-promotion week.
Unfilled orders and transfer expenses are omitted; the team also credits all extra gross sales as profit.

**Deliverable:** List at least four defects and propose a credible evaluation and financial ledger.

**Acceptance criteria:** Address counterfactual, denominator, transfer cost, and contribution rather than revenue.

**Time:** 20 minutes.

### Exercise 9 — Scenario decision

Plan P costs $700 in a normal scenario and $1,100 in a disruption scenario.
Plan Q costs $800 in a normal scenario and $900 in disruption.
Assume normal and disruption probabilities of 0.8 and 0.2 respectively.
All costs are dollars over the same four-week horizon and include the same cost categories.

**Deliverable:** Calculate expected cost for each plan and identify the lower-cost choice.

**Acceptance criteria:** Show probability weighting and identify a reason the owner might still prefer the other plan.

**Time:** 10 minutes.

## 13. Self-check answers

### Answer 1

The 500-unit estimate is the forecast.
The proposed 200-unit move is the placement plan until approved and executed.
The two-day checkout message is the customer-facing commitment, subject to its stated terms.
The OMS origin choice is the subsequent order-routing decision.
A demand planner can own the forecast, a supply planner the move, and the product or promise owner the displayed date.
An OMS or fulfillment owner can own routing implementation.
A forecast is uncertain and does not establish ATP, capacity, or delivery performance.

### Answer 2

SKU C requires 30 × 2 = 60 cubic feet.
SKU D requires 15 × 3 = 45 cubic feet.
Together they require 105 cubic feet, exceeding the 100-cubic-foot limit by 5 cubic feet.
The joint plan is infeasible even before considering temperature approval.
It is also ineligible because SKU D cannot be handled in the required temperature range.
Reducing quantity to fit space would not by itself fix the temperature restriction.
The operations and product-compliance owners must resolve eligibility before any transfer.

### Answer 3

Outbound remains $590 and holding remains $77.
With a $100 transfer charge, total cost is $590 + $77 + $100 = $767.
The saving versus $842 baseline is $75 over four weeks.
Fast coverage remains 130 / 140 = 92.86% because the routing and demand assumptions do not change.
The candidate still meets the deterministic 90% coverage target and 75-unit East capacity limit.
Actual delivery reliability remains unmeasured.

### Answer 4

Fast coverage over all eligible orders is 153 / 200 = 76.5%.
Fast share among filled orders is 153 / 180 = 85.0%.
The unit fill rate is 180 / 200 = 90.0%.
The filled-only fast share can hide the 20 unfilled orders.
Neither fast-lane metric proves actual on-time delivery.
Report all three measures with the period, order population, and lane definition.

### Answer 5

With zero correlation, pooled variance is 12² + 16² = 144 + 256 = 400 units squared.
Pooled standard deviation is the square root of 400, or 20 units.
Pooled safety stock is 1.645 × 20 = 32.9 units before rounding.
Separate safety stocks sum to 1.645 × (12 + 16) = 46.06 units before rounding.
The difference is 13.16 units before any operational rounding or minimum-pack constraints.
A conservative whole-unit implementation might round each requirement up, but the rounding policy must be explicit.
The `z` value describes the simplified cycle-service calculation, not a 95% unit fill guarantee.

### Answer 6

Gross outbound savings are 70 × $3 = $210.
Incremental holding is 70 × $0.50 = $35.
The transfer-cost break-even is $210 − $35 = $175 for the four-week case.
At a $190 transfer charge, A costs $15 more than the baseline under these assumptions.
It may still be the only candidate meeting the 90% target, so the owner must decide whether to pay the premium or revise the policy.
The cost threshold is not itself authorization to miss a service requirement.

### Answer 7 rubric

A strong plan names a supply planner, store operations owner, OMS owner, finance owner, and promise owner.
It includes a timestamped SKU-store eligibility and inventory accuracy check.
It reserves origin stock, represents in-transit stock, and activates destination ATP only after verified receipt.
It checks daily pick and pickup limits rather than only available shelf space.
It defines a small pilot, comparison group or counterfactual, and monitoring cadence.
It sets numeric service and financial guardrails appropriate to the business rather than inventing them after the pilot.
It names who can stop transfers and how open orders and existing customer commitments are handled.
It requires reconciliation and a documented decision before expansion or rollback.

### Answer 8 rubric

A strong critique identifies promotion-week confounding and the absence of a comparable counterfactual.
It measures service over all eligible orders and separately reports unfilled demand.
It adds incremental transfer and receiving costs on the correct time horizon.
It converts incremental sales into contribution after item, fulfillment, return, and other relevant variable costs.
It distinguishes one-time migration costs from steady-state costs.
It checks whether apparent savings reflect changed SKU mix, destinations, order size, or carrier rates.
A credible design uses randomized cohorts where feasible or a justified matched/phased comparison with pre-period diagnostics.
It reports uncertainty, limitations, and changes in outcomes at other nodes.

### Answer 9

Plan P has expected cost 0.8 × $700 + 0.2 × $1,100 = $780.
Plan Q has expected cost 0.8 × $800 + 0.2 × $900 = $820.
P is $40 cheaper in expected cost over the four-week horizon.
Q has a $900 disruption cost rather than P's $1,100 and may be preferred by a risk-sensitive owner.
A hard service or liquidity constraint could also change the decision.
The expected values depend on the stated probabilities and do not prove those probabilities are accurate.

## 14. Interview questions

1. **How do placement and order routing differ?**
   A strong answer separates a pre-demand stock decision from a later source choice and describes their shared ATP constraints.
   A weak answer assumes the OMS can route from inventory that has not been positioned or received.

2. **Why can local stocking increase total inventory?**
   A strong answer discusses fragmented pools, joint demand variation, safety stock, and replenishment lead times.
   A weak answer says more warehouses always require a fixed percentage more stock without stating assumptions.

3. **How would you decide whether a slow-moving SKU belongs in every node?**
   A strong answer compares incremental speed, transfer, handling, holding, availability, and obsolescence under uncertainty.
   A weak answer uses national sales rank alone and ignores destination demand.

4. **What is wrong with using the shipping node as the demand location?**
   A strong answer explains that routing reflects past stock and policies and can censor demand at unavailable nodes.
   A weak answer assumes historical fulfillment proves where customers live.

5. **How do you measure a two-day delivery promise?**
   A strong answer defines order population, clock start, calendars, actual delivery event, exclusions, and denominator.
   A weak answer reports only carrier transit time or only orders that shipped.

6. **How do you compare a one-time transfer with weekly savings?**
   A strong answer aligns horizons, separates migration from steady-state costs, and evaluates the relevant lifetime and uncertainty.
   A weak answer subtracts a one-time amount from an annual saving without specifying the comparison period.

7. **What would you do if the optimization model says the service target is infeasible?**
   A strong answer identifies binding constraints, reports attainable alternatives, and escalates any policy relaxation to its owner.
   A weak answer silently removes constraints until a solution appears.

8. **How would you include uncertain demand without perfect foresight?**
   A strong answer keeps initial placement common across scenarios and permits later routing to respond within stock and capacity limits.
   A weak answer chooses a different initial allocation for each realized future scenario.

9. **What changes when orders contain multiple SKUs?**
   A strong answer discusses order-level completeness, package count, consolidation, split shipments, and carrier cost.
   A weak answer assumes every SKU line always creates exactly one independent order.

10. **How would you validate a placement pilot?**
    A strong answer defines a comparable counterfactual, eligible population, costs, service measures, and operational guardrails.
    A weak answer treats a favorable before-after average as proof while ignoring promotions and unfilled orders.

11. **Which system controls prevent inventory from being sold twice in a transfer?**
    A strong answer covers origin reservations, in-transit state, destination receipt, ATP activation, and reconciliation.
    A weak answer changes both node balances when the transfer order is created without tracking movement.

12. **When might a more expensive plan be preferred?**
    A strong answer identifies explicit service constraints, downside risk, customer commitments, and the cost of the trade-off.
    A weak answer calls service priceless or ignores a constraint because another plan has lower average cost.

## 15. Key terms

- [SKU](#4-definitions-and-vocabulary): operationally distinct sellable item.
- [Node](#4-definitions-and-vocabulary): location that can hold or fulfill inventory.
- [Customer region](#53-build-a-clean-demand-population): destination grouping used for demand and lane analysis.
- [Placement](#31-position-before-routing): pre-demand allocation of stock to nodes.
- [Routing](#31-position-before-routing): assignment of an individual order to an eligible source.
- [Inventory position](#55-assemble-demand-stock-and-supply-inputs): defined stock balance including relevant inbound and commitments.
- [ATP](#55-assemble-demand-stock-and-supply-inputs): quantity available for new customer commitments.
- [Safety stock](#56-model-demand-uncertainty-and-protection-stock): buffer against specified protection-period uncertainty.
- [Cycle stock](#4-definitions-and-vocabulary): inventory that varies with replenishment and demand.
- [Reorder point](#56-model-demand-uncertainty-and-protection-stock): inventory-position trigger for a replenishment policy.
- [Protection period](#56-model-demand-uncertainty-and-protection-stock): period of demand exposure before stock is restored.
- [Lead time](#56-model-demand-uncertainty-and-protection-stock): time between named replenishment and availability events.
- [Holding cost](#57-build-cost-and-service-inputs): period-aligned cost of carrying inventory.
- [Capital charge](#57-build-cost-and-service-inputs): financing or opportunity cost of invested inventory capital.
- [Transfer cost](#57-build-cost-and-service-inputs): incremental expense of moving stock between nodes.
- [Fast eligibility](#33-coverage-is-not-actual-service): modeled ability of a lane to satisfy a stated window.
- [Coverage](#33-coverage-is-not-actual-service): share of an explicit demand population assigned to eligible lanes.
- [Fill rate](#4-definitions-and-vocabulary): fraction of a defined unit or order demand population fulfilled.
- [Cycle service level](#56-model-demand-uncertainty-and-protection-stock): probability of no stockout in a defined cycle.
- [Pooling](#34-pooling-and-fragmentation): serving multiple demand sources from shared stock.
- [Correlation](#56-model-demand-uncertainty-and-protection-stock): co-movement of regional forecast errors.
- [Scenario](#56-model-demand-uncertainty-and-protection-stock): coherent possible demand and operating outcome.
- [Capacity](#58-encode-physical-and-operational-constraints): available node resource per defined period.
- [Constraint](#58-encode-physical-and-operational-constraints): condition every accepted model solution must satisfy.
- [Objective](#59-formulate-a-transparent-decision-model): metric used to rank feasible alternatives.
- [Counterfactual](#32-a-stock-unit-has-an-opportunity-cost): comparable outcome under another decision.
- [Demand censoring](#53-build-a-clean-demand-population): sales limited by availability rather than actual interest.
- [Split shipment](#57-build-cost-and-service-inputs): a defined order shipped in multiple packages or from multiple nodes.
- [Rebalancing](#512-execute-monitor-and-reverse): subsequent movement to change the stock distribution.
- [Two-stage model](#59-formulate-a-transparent-decision-model): common pre-demand placement followed by scenario-dependent recourse.

## 16. Sources

The standards, regulatory guidance, academic work, and company materials below support the specific claims attributed in this lesson.
They do not supply the numerical assumptions of the fictional example.
Company publications describe their own services and networks and are not independent proof of a general savings rate.
Check current rules, service terms, and source versions before operational use.

1. [GS1, Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin). Standards-body explanation of trade-item identification.
2. [GS1, Global Location Number](https://www.gs1.org/standards/id-keys/gln). Standards-body explanation of location and party identification.
3. [GS1, EPCIS](https://www.gs1.org/standards/epcis). Standards-body description of visibility-event data exchange.
4. [GS1, Electronic Data Interchange](https://www.gs1.org/standards/edi). Standards-body overview of electronic business-message exchange.
5. [IFRS Foundation, IAS 2 Inventories](https://www.ifrs.org/issued-standards/list-of-standards/ias-2-inventories/). Official overview of inventory accounting under the standard.
6. [U.S. Federal Trade Commission, Business Guide to the FTC's Mail, Internet, or Telephone Order Merchandise Rule](https://www.ftc.gov/business-guidance/resources/business-guide-ftcs-mail-internet-or-telephone-order-merchandise-rule). Official U.S. business guidance on covered shipment representations and delays.
7. Eppen, Gary D. (1979), [Effects of Centralization on Expected Costs in a Multi-Location Newsboy Problem](https://doi.org/10.1287/mnsc.25.5.498), *Management Science*, 25(5), 498–501. Academic analysis of pooling under stated assumptions.
8. [Amazon, How Amazon is improving delivery speeds](https://www.aboutamazon.com/news/operations/amazon-delivery-speeds-2023). Company account of network changes and delivery speed; its claims are company-reported.
9. [Walmart, Pickup and Delivery](https://corporate.walmart.com/about/pickup-delivery). Company description of pickup and delivery offerings.
10. [Amazon, Fulfillment by Amazon](https://sell.amazon.com/fulfillment-by-amazon). Company description of seller fulfillment services and program features.
11. [Walmart Marketplace, Walmart Fulfillment Services](https://marketplace.walmart.com/fulfillment/). Company description of marketplace fulfillment services.

The academic pooling result is conditional on its model assumptions and should not be applied as a universal percentage saving.
The FTC guidance is U.S.-specific; it does not replace current legal review in the applicable jurisdiction.
The IFRS overview does not determine the accounting treatment of a particular transaction.

## 17. Links

- [Module 21 index and topic map](_index.md).
- [Module 21 survey](../../../course/21_ecommerce_supply_chain_analytics.md).
- [Sibling topic 01: e-commerce fulfillment models](01_ecommerce-fulfillment-models.md).
- [Sibling topic 02: order-grain forecasting, listed in the module map](_index.md#3-topics).
- [Sibling topic 03: long-tail forecasting, listed in the module map](_index.md#3-topics).
- [Sibling topic 05: delivery promise, listed in the module map](_index.md#3-topics).
- [Sibling topic 09: unit economics, listed in the module map](_index.md#3-topics).
- [Warehouse flow and system boundaries](../../02_methods/13_warehousing_fulfillment_labor/01_warehouse-flow-system.md).
- [Warehouse management system data model](../../02_methods/13_warehousing_fulfillment_labor/02_wms-data-model.md).
- [Simulation and optimization production labs index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).

The links to planned sibling topics point to their entries in the module map until those lessons are published.
