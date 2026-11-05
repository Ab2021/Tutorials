# Promise Tiers and Dark-Store Design

> Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics  
> Topic 1 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 10, 12, and 21  
> Version 2026.5

## 1. Why this topic exists

A delivery promise is an operating commitment, not a label attached after an order is placed.
A retailer choosing a 10-minute promise must make inventory, picking labor, courier availability,
travel time, and handoff fit within a short and variable interval.
A 30-minute promise permits a wider catchment and more operating flexibility, but is still exposed
to short peaks and uncertain customer access.
A scheduled window allows capacity to be reserved and routes to be consolidated ahead of time,
but the business must retain customer trust across a longer order-to-delivery interval.

These tiers do not merely change the speed shown on an app.
They change where inventory sits, how many facilities are required, how much slack is needed,
and what proportion of each customer's basket can be fulfilled from the selected site.
An operator that sells a service it cannot reliably deliver may incur refunds, repeated contacts,
failed deliveries, unsafe courier incentives, and customer churn.
An operator that builds a dense network for demand that would accept a scheduled window may
incur fixed costs and inventory duplication that order contribution cannot cover.

The network planner needs to decide which addresses a store can serve and under which conditions.
The operations manager needs to decide when to stop accepting a tier before a queue becomes a miss.
The commercial owner needs to choose fees and assortment without treating delivery revenue as profit.
The finance owner needs to test whether a proposed footprint can pay for its incremental cost.
The safety owner needs authority to constrain the service when conditions make it unsafe.
The customer needs a clear commitment, not an average travel time disguised as a guarantee.

This topic connects those decisions through a single capacity and economics model.
Its numerical case is illustrative, not a claim about any operator's actual performance.
The analysis is a decision aid: live acceptance still requires current operational data,
local legal review, and accountable human ownership.

## 2. Learning outcomes

After working through this topic, you should be able to:

1. Define the start, end, denominator, and exclusions of each customer promise.
   Exercise 1 tests the definitions and the treatment of cancelled orders.
2. Decompose a delivery deadline into preparation, queue, travel, and handoff time.
   Exercise 2 asks you to compute remaining travel budget and critique an average.
3. Translate a road-network travel-time boundary into a conditional store catchment.
   Exercise 3 asks you to distinguish a feasible area from an operationally open area.
4. Calculate picker and courier capacity in a common time bucket, including a slack policy.
   Exercise 4 tests the bottleneck and the tier that should be throttled.
5. Compare a 10-minute, 30-minute, and scheduled service on reliability and contribution.
   Exercises 5 and 6 test the economic and service implications.
6. Propose a footprint and an operating control with explicit ownership and escalation.
   Exercise 7 tests the plan against a disruption scenario.
7. Specify an experiment that estimates incremental demand without confusing substitution
   between tiers with new demand.
   Exercise 8 tests the counterfactual and the guardrails.
8. Identify relevant data, safety, employment, privacy, and consumer-protection questions
   without presenting a general operational design as jurisdiction-specific legal advice.
   Exercise 9 tests how to escalate uncertain obligations.

## 3. The core mental model

A promise is the intersection of a customer-facing contract and a constrained network.
Demand reaches the network at a particular place and time; a site and a tier are feasible only
if the order can be completed within its commitment using available, safe resources.
The cost of that feasibility must be compared with incremental value over a defined horizon.

```text
Customer location + basket + chosen time
                  |
                  v
       Candidate store and tier
                  |
          +-------+--------+
          |                |
   Stock and safety   Time and capacity
   eligibility        eligibility
          |                |
          +-------+--------+
                  |
          Admission decision
                  |
       Pick -> stage -> assign -> travel -> handoff
                  |
       Delivered outcome and cost
                  |
      Forecast, boundary, and staffing updates
```

Read the diagram from top to bottom when making a commitment.
Read it from bottom to top when learning from outcomes.
The feedback loop must not overwrite what the system knew when it accepted an order.
Otherwise the operator cannot distinguish a bad forecast from an unforeseen disruption
or evaluate whether an admission decision was reasonable at the time.

The clock starts at the event actually stated to the customer.
In this lesson the immediate tiers start when the order is accepted and stop at completed
customer handoff, with a door-delivery assumption.
A scheduled order has a quoted start and end time and is on time when handoff falls within
the window, unless the customer explicitly accepts an early delivery.
An acceptance timestamp is not necessarily the same as payment authorization or checkout click.
Record all three when they differ.

The clock consists of intervals that sometimes overlap.
A courier can be assigned while a picker is still working, so simply adding all elapsed
process durations can double-count overlapping work.
For a conservative planning approximation, the example uses serial durations and says so.
Production systems should reconstruct the actual critical path from event timestamps.
Averages can describe cost or aggregate work, but a promise concerns an individual outcome.
The 90th or 95th percentile can be useful for capacity planning, provided it is measured
for the same tier, geography, and conditions as the commitment.
Adding separate percentile durations does not in general produce the percentile of their sum.
Measure or simulate the joint end-to-end distribution when possible.

A geographically eligible address is not necessarily an address at which a promise should
be sold at this moment.
The order may arrive during a picker queue, courier shortage, inventory discrepancy, storm,
or access disruption.
Static polygons answer where a site could serve under stated conditions.
Live gating answers whether it should accept another order now.
Both are required.

For economic decisions, keep three views apart.
Contribution per order excludes costs explicitly treated as fixed in the analysis.
Site operating profit subtracts the site's avoidable fixed costs over a stated period.
Network incremental profit compares a complete proposal with the best feasible alternative,
including demand shifted from other stores or tiers.
A positive order contribution is therefore necessary for some growth strategies but does not,
by itself, justify a new store.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Promise tier | Customer-facing service class with a clock or window, in minutes or local time | Treating a marketing name as a measurable commitment |
| Acceptance | Event at which the operator accepts an order, timestamp with time zone | Assuming app click and acceptance are identical |
| Handoff | Transfer to the customer or agreed secure location, timestamp | Using courier arrival at building as completed handoff |
| On-time delivery | Handoff within the quoted deadline or agreed window | Counting only delivered orders while ignoring accepted failures |
| Reliability | On-time eligible commitments divided by all eligible commitments, percentage | Reporting an unweighted average of store percentages |
| Eligible commitment | Accepted order in a predeclared service-measurement population | Excluding inconvenient misses after observing outcomes |
| Cancellation | Order ended before completion, with actor, reason, and timestamp | Automatically treating every cancellation as an on-time success |
| Scheduled window | Agreed interval, such as 18:00–19:00 local time | Treating the window end as the only customer constraint |
| Dark store | Fulfillment facility arranged for picking rather than ordinary customer browsing | Assuming it must be a separate legal or physical building |
| Micro-fulfillment site | Small local fulfillment operation, potentially within another facility | Assuming any small store has the staffing and data of a dark store |
| Catchment | Set of locations potentially served under stated tier and conditions | Drawing a fixed-radius circle and calling it a travel-time guarantee |
| Isochrone | Network-reachable region within a travel-time threshold | Treating a map estimate as proof of individual service |
| Footprint | Set of facilities, capacities, and geographical assignments | Counting sites without accounting for utilization or overlap |
| Demand density | Orders per unit area per time bucket | Confusing customer population with orders actually won |
| Basket | Items and quantities in an order | Treating revenue per basket as gross profit |
| Order line | A distinct ordered item record, sometimes including quantity | Assuming one line always equals one unit |
| Pick time | Elapsed or labor time spent locating and collecting items, specified explicitly | Mixing worker-minutes with wall-clock minutes |
| Staging | Holding a prepared order for courier collection | Assuming all staged food can wait indefinitely |
| Dispatch | Assignment and release of a delivery task to a courier | Assuming assignment means immediate courier arrival |
| Deadhead | Courier movement without a customer order, minutes or distance | Omitting repositioning from delivery capacity and cost |
| Handoff time | Time from arrival at destination vicinity to completed transfer, minutes | Ignoring elevators, security, and customer response |
| Capacity | Maximum sustainable work completed in a bucket under stated assumptions | Using theoretical peak throughput as dependable capacity |
| Utilization | Busy resource time divided by available resource time | Assuming 100% utilization is compatible with negligible queues |
| Slack | Capacity or time reserved against variability, expressed in minutes or percentage | Describing idle capacity as automatically wasteful |
| Bottleneck | Constraint limiting system throughput for a specified demand mix | Assuming the bottleneck is permanent or always the courier |
| Queue | Work waiting for a constrained resource, orders or minutes | Counting only orders actively being picked |
| Admission control | Rule that accepts, changes, or closes a promise before commitment | Updating an ETA after accepting an impossible deadline |
| Forecast | Distribution or estimate of future demand before an outcome | Calling an agreed commitment a forecast |
| Plan | Chosen staffing, inventory, and capacity arrangement | Treating an unconstrained forecast as a staffing plan |
| Target | Desired internal performance level | Promising that target to every customer |
| Commitment | Actual time or window communicated and accepted for an order | Replacing the original commitment with a later revised ETA |
| ETA | Estimated time of arrival, which can change as information arrives | Treating an ETA distribution mean as a delivery guarantee |
| Contribution | Revenue less specified variable costs, currency per order | Ignoring fees, refunds, shrink, or the definition of included costs |
| Fixed site cost | Cost not varying with an incremental order within the model horizon | Treating all fixed costs as avoidable on immediate closure |
| Cannibalization | Demand shifted from an existing offer to another, orders per period | Counting a switched order as wholly incremental |
| Incremental profit | Profit under a proposal minus profit under the counterfactual, currency per period | Multiplying total orders by the new tier's margin |
| Safety gate | Operational restriction that prevents unsafe service | Treating safety as a cost to optimize away |
| Service recovery | Communication, remedy, or support after an issue | Erasing the original miss when a customer accepts a credit |

## 5. The mechanics, step by step

### Step 1 — State the decision and counterfactual

Input: a service area, a planning horizon, current operations, and a proposed decision.
Output: a one-sentence decision and a comparison that can be evaluated.

For example: decide whether to add a small dark store for a 10-minute tier for one year,
compared with serving the same demand from the existing network with a 30-minute tier.
This is not equivalent to comparing the proposed store with no business at all.
The counterfactual includes existing site capacity, current fees, and the orders already served.
An opening date, lease term, and exit options affect which costs are avoidable.
If the question is today's operating choice rather than a site investment, compare admitting
one more order with quoting a slower available tier, not with closing a facility.

The network lead owns the proposed footprint.
Finance signs off on cost definitions and the investment comparison.
Operations owns the actual ability to staff the plan.
Product owns the customer-facing wording and preserves the original quoted commitment.
Safety has authority to prohibit an unsafe operating state regardless of projected profit.
Resolve conflicting decisions through a named general manager, not an implicit model score.

### Step 2 — Specify each service contract

Input: intended customer experience, service geography, and legally reviewed terms.
Output: tier definitions and a measurement policy that can be applied to order records.

For an immediate tier define the clock start, deadline, endpoint, and location of handoff.
For example, a 10-minute tier means completed handoff no later than 10 minutes after acceptance.
A 30-minute tier uses the same endpoints with a 30-minute deadline.
These are illustrative labels, not a recommendation that every business should promise them.
Define whether delivery to a lobby, secure locker, or door satisfies the contract.
A promised door delivery is not met merely because a courier reached the street address.

For scheduled delivery define the local-time window and whether arrival before its start is allowed.
Define what happens when a customer chooses a window that crosses a daylight-saving transition.
Store the actual time-zone identifier and unambiguous instant, not just a wall-clock string.
If a recipient consents to an early delivery, retain the original window and consent event.
A revised ETA informs the customer; it does not silently reset the original commitment.

Decide how an operator cancellation, customer cancellation, attempted delivery, unreachable
customer, and failed handoff affect service measurement before calculating the metric.
The illustrative metric in this lesson counts every accepted order as a commitment and counts
an order without an on-time completed handoff as not on time.
Report customer-initiated cancellations separately to aid diagnosis.
This intentionally conservative operational metric is not a statement of legal liability.
Also report a delivered-only on-time rate if useful, clearly labelled and never substituted for
the accepted-order rate.

### Step 3 — Segment demand and geography

Input: historical order events, addresses or privacy-preserving spatial cells, baskets, and time.
Output: a demand forecast by cell, time bucket, tier, and basket characteristics.

Use the same local time zone and bucket definition for demand and resource availability.
Fifteen-minute buckets are useful for immediate services because a daily average hides peaks.
A scheduled service may use reservation windows and a separate dispatch-planning horizon.
Separate forecasts for weather, weekday, events, and promotions when evidence supports it.
Maintain an uncertainty range, not just a point forecast.

History contains only demand the existing offer allowed customers to place.
Closed service areas, stockouts, sold-out slots, and long displayed ETAs suppress observed demand.
Retain exposure and availability data so low observed orders are not automatically read as
low underlying interest.
Where the new tier has no history, run a controlled pilot and state uncertainty rather than
claiming that another city's conversion rate will transfer exactly.

Basket attributes matter because pick effort and inventory availability change with line count.
Building access matters because an apartment handoff can take much longer than a street handoff.
Mode matters because a bicycle, walker, and car have different feasible routes and restrictions.
Do not infer that one demographic group is less serviceable from protected characteristics;
test spatial and service disparities and obtain legal and policy review before deployment.

### Step 4 — Convert a deadline into a time budget

Input: promise clock, event history, location, mode, and chosen reliability target.
Output: an end-to-end time distribution and a conservative candidate travel-time budget.

Start with the full order-to-handoff distribution for comparable accepted orders.
Separate checkout-to-acceptance if the actual customer promise starts before acceptance.
Inspect the sequence of accepted, pick start, pick complete, courier arrival at store,
departure, arrival at destination, and completed handoff events.
For sequential work, the deadline is consumed by queue, picking, staging, outbound travel,
and destination handoff.
For overlapping work, calculate the elapsed critical path instead of adding all durations.

A quick screening budget can subtract planned nontravel minutes from the deadline.
For instance, a 10-minute deadline minus 2.5 minutes of preparation and queue,
1.0 minute of dispatch and pickup, and 1.5 minutes of handoff leaves 5.0 minutes for travel.
This is a deterministic planning screen, not a 95th-percentile guarantee.
Its components must be consistently chosen and supported by joint outcome data.
A route within five estimated minutes can still miss because of correlated congestion,
a missing item, a slow elevator, or a courier not being at the store.

If the residual budget is negative, the tier is not feasible under those assumptions.
If it is positive but very small, examine whether the customer experience and cost justify
that operating model rather than promising on the basis of an optimistic road speed.
A 30-minute tier has more time for travel and work, but the extra minutes are not all free:
slower pickup, larger baskets, and wider delivery areas can consume them.
Scheduled windows permit aggregation and advance assignment, yet still require reliable
arrival inside the window and a feasible cold-chain holding period.

### Step 5 — Construct feasible catchments

Input: candidate sites, road or path network, mode-specific travel estimates, and time budgets.
Output: tier-specific candidate address sets with a stated date and conditions.

Build an isochrone from each candidate facility using the relevant departure-time travel model.
Constrain routes by legal access, one-way streets, bridges, water, and the selected courier mode.
Add destination handoff and building-access assumptions.
Validate the estimated travel times using observed trips not used to fit the model.
Use specific address-level checks where building access makes a spatial cell misleading.

An isochrone is a screening device, not an operational guarantee.
Use a conservative boundary for sparse or uncertain cells, and expose uncertainty to the owner.
A straight-line radius can be a quick visualization but must not substitute for routing evidence.
A small dense site might have a short boundary for the 10-minute tier and a larger boundary
for the 30-minute tier.
An address may be eligible for both; assignment then depends on available capacity, inventory,
cost, and the actual tier the customer chooses.

For a scheduled tier, feasible destinations depend on the route sequence and promised windows,
not just on a single-order isochrone.
A destination that is feasible alone might make a multi-stop route miss another commitment.
Reserve enough slack for correlated travel disruption and service at prior stops.
Do not quote an additional window merely because an isolated route calculation is feasible.

### Step 6 — Design facility and inventory options

Input: feasible demand cells, assortment needs, candidate sites, facility costs, and capacity.
Output: a small set of viable footprint scenarios, including the current network.

Compare an existing retail backroom, a purpose-built dark store, and a larger scheduled hub
when those options exist in the local market.
A backroom can share existing space but may compete with store replenishment or customer traffic.
A dark store can place fast-moving goods close to demand but adds rent, utilities, and management.
A hub can consolidate stock and scheduled routes but may not reach immediate-tier deadlines.
No format is universally cheaper; use local lease, labor, and demand evidence.

For every candidate identify usable pick area, cold and frozen capacity, loading space,
supplier access, opening hours, and zoning or permitting requirements.
A site that fits a map but has no safe courier staging or refrigeration is not feasible.
Estimate demand-weighted basket availability rather than only number of stocked products.
Inventory duplicated across sites ties up working capital and can raise expiration risk.
Model inbound replenishment and losses explicitly where they differ across footprints.
A site that wins orders by promising a narrow assortment may lose baskets that include one
unavailable item; avoid treating unavailable items as zero-cost omissions.

Define capacity in the units of the bottleneck: pick labor-minutes, packing stations,
refrigerated staging slots, dispatch slots, or courier-minutes.
It is not enough for a site to be near an address if its queue is already full.
Capacity also depends on the mix of small and large baskets.
Use a common work measure or scenario-specific workload rather than an order count alone.

### Step 7 — Size picker capacity

Input: available pickers, paid minutes in the bucket, productive fraction, and work per order.
Output: sustainable pick capacity in orders per bucket under the assumed order mix.

Calculate effective productive minutes as staffed minutes times the productive fraction.
The fraction must cover breaks, replenishment interference, setup, and variation not otherwise
included in measured pick work; do not deduct the same allowance twice.
Divide effective minutes by mean picker labor-minutes per order for a first capacity estimate.
This estimate is a workload bound, not a guarantee that every individual order meets its clock.
The shorter the promise and the more variable arrivals are, the more important queue simulation
or measured service curves become.

For example, four workers over 15 minutes provide 60 staffed minutes.
At an 80% productive fraction they provide 48 effective minutes.
If each order requires 2.4 picker labor-minutes on average, the workload bound is 20 orders
per 15 minutes for that mix.
A 20-order arrival burst at the beginning of the bucket may miss a 10-minute deadline even
though the 15-minute aggregate workload fits.
Measure queue age and the distribution of pick completion times, not just aggregate utilization.

When basket size changes, recalculate work using lines, units, temperature zones, and packaging.
For a promotion that doubles heavy baskets, the order-count capacity from ordinary days is wrong.
Training, layout changes, and missing products can all shift pick time.
Use observations from the relevant workflow and preserve a fallback for insufficient samples.

### Step 8 — Size courier capacity

Input: available courier-minutes, expected work per delivery, tier mix, and utilization policy.
Output: a courier workload bound and the slack needed to protect service.

Count pickup wait, loaded travel, handoff, and deadhead or repositioning where applicable.
A courier who completes delivery outside the store catchment cannot be reused instantaneously.
Use paid or economically relevant available time consistently with the cost model.
If contractors can decline assignments or log off, nominal logged-in time overstates supply.
Account for modes, local restrictions, and any practical limits on carrying multiple orders.

Suppose 12 couriers each provide 15 available minutes in a bucket.
At an 80% planning utilization, they provide 144 usable courier-minutes.
If one single-order task uses 12 courier-minutes on average, the workload bound is 12 orders
per 15 minutes.
The calculation is a planning approximation: an individual courier cannot deliver a fraction
of an order, work can cross bucket boundaries, and location-specific availability matters.
An 80% planning utilization is an explicit illustrative assumption, not a universal rule.

Batching can reduce courier-minutes per order but may increase time to pickup or delivery.
Only count the reduction if an actual feasible route respects all accepted commitments,
food safety, carrying constraints, and fair working conditions.
Batching assumptions that depend on a courier rushing must be rejected.
For scheduled windows, reserve route capacity before offering additional slots.
Cancellation and no-show rates should be modelled conservatively; overselling creates exposure.

### Step 9 — Match workload to capacity and gate acceptance

Input: active queues, future reservations, capacity forecast, candidate tier, and order attributes.
Output: an accept, slower-tier offer, scheduled-window offer, or no-service decision.

Calculate remaining time to deadline from the original acceptance clock.
Before acceptance, estimate the new order's impact on already accepted commitments as well as
its own likely completion.
A new order should not be admitted merely because it can be completed if doing so causes
existing orders to miss.
Apply stock, safety, picker, courier, and staging checks in addition to travel-time checks.

A practical gate uses a hard safety block, a hard inventory eligibility check, and an operational
capacity rule supported by measured or simulated end-to-end reliability.
The rule might close 10-minute service when queue age or predicted on-time probability crosses
an approved threshold, while retaining a 30-minute or scheduled option if independently feasible.
The threshold is a decision policy to calibrate, not a universal percentage.
A model score must never override a legal prohibition or an unsafe local condition.

Use hysteresis or a minimum reopening period when a noisy signal causes rapid open-close cycles.
Keep any such control transparent: an internal store closure and a customer-facing offer change
should have timestamps and reasons.
The duty manager may override within an approved authority matrix; record who, why, and duration.
Offer only a tier with genuinely available capacity; a slower label is not a cure for an
unbounded queue.

### Step 10 — Calculate service and economics

Input: accepted orders, handoffs, cancellations, fee and cost data, and fixed-site costs.
Output: tier-level service rates, variable contribution, site profit, and incremental comparison.

Compute on-time service as on-time completed handoffs divided by accepted commitments under
the declared cancellation convention.
Break down late, cancelled, failed, and still-open orders so the total reconciles.
Show the count with the rate and compare like-for-like observation periods.
A 95% rate from 20 orders has very different uncertainty from 95% from 20,000 orders.
Track the distribution of lateness among misses as well as the overall rate.

Revenue includes product sales and delivery fees, net of discounts and applicable taxes
according to the accounting policy.
Deduct product cost, picking, courier pay, packaging, payment cost, waste, and service recovery
if those are variable in the decision horizon.
Avoid double counting a picker allowance already included in a site fixed labor line.
Show whether advertising, technology, corporate overhead, and depreciation are included.
A contribution margin that excludes them can support a tactical order decision but not alone
establish fully loaded profitability.

Calculate monthly or annual site operating profit after site costs.
For a new site, subtract cannibalized contribution from existing sites and include the effect
on fixed costs, working capital, and investments over the chosen horizon.
The decision must compare the entire proposal with the actual alternative, not with zero.
Check sensitivity to demand, order mix, courier cost, waste, and achievable service.

### Step 11 — Pilot and learn causally

Input: a feasible design, control areas or randomized eligible customers, and guardrails.
Output: a staged deployment plan and a decision rule for expansion or withdrawal.

Decide the primary outcome before launch: for example incremental contribution per eligible
household-week, not just orders attributed to the fast tier.
Randomize where feasible, and specify the unit of assignment and observation window.
Prevent spillovers where a courier or store serves both test and control in ways that change
control capacity; if spillovers cannot be prevented, design and analyze for them.
Include all randomized eligible units in the primary analysis, including those that never order.

Measure total orders across all tiers to distinguish conversion from switching.
Record basket value, refunds, accepted-order service, safety events, and courier earnings.
Do not stop a test solely because a noisy early result is favorable unless the analysis plan
explicitly supports that decision.
Set minimum sample size or precision goals using plausible effect sizes and local baseline data.
An unsafe event triggers immediate review irrespective of statistical significance.

### Step 12 — Operate, review, and retire

Input: live control signals, weekly performance, and longer-term investment outcomes.
Output: named decisions, incident actions, and versioned boundaries.

The shift lead acts on live queues and safety conditions.
The network planner reviews recurring edge failures and geography at a defined cadence.
The demand planner reviews forecast bias and changing mix.
Finance reconciles realized contribution against the decision model monthly.
Product reviews offer wording and customer complaints with support and legal partners.

An incident record identifies the time of signal, decision, owner, scope, customer impact,
and restoration conditions.
A boundary change should have an effective time and reason, with the previous version retained.
If the store fails service or investment criteria after a fair observation period, decide whether
to redesign, reduce the footprint, or close, taking contractual obligations into account.
The team should not retain an uneconomic promise merely to avoid acknowledging an earlier error.

## 6. Worked example

### Decision and data

A fictional operator, Harbor Basket, serves a compact urban district.
The network lead must decide whether to operate three tiers from one proposed dark store.
All figures are illustrative and in US dollars (USD); the planning period is one month
of 30 operating days, with a separate peak 15-minute capacity check.
Tax is excluded from sales and fees, and figures are rounded as shown.
The example is not evidence that real 10-minute service is feasible in this district.

The three offers are defined as follows.

| Tier | Customer commitment | Initial geographic screen |
|---|---|---|
| 10-minute | Door handoff within 10 minutes of acceptance | Mode-specific travel estimate at most 5 minutes |
| 30-minute | Door handoff within 30 minutes of acceptance | Mode-specific travel estimate at most 18 minutes |
| Scheduled | Door handoff inside selected 60-minute window | Feasible route and reserved slot |

The scheduled customer may accept early arrival, but otherwise an early handoff is not counted
as within the stated window.
The isochrones are screening assumptions; the operator will not open either immediate offer
without independently meeting capacity and safety gates.
Forecasts, plans, targets, and commitments are different objects:

- Forecast: 240, 600, and 360 accepted orders per day in the three tiers, respectively.
- Plan: resource and slot reservations intended to handle the forecast mix.
- Target: at least 95% of accepted commitments handed off on time in each tier.
- Commitment: each accepted customer's original deadline or selected delivery window.

The forecast is uncertain and is not a guarantee of demand.
The 95% target is a management choice for the example, not permission to intentionally miss
one order in twenty or a customer-facing promise of only 95% reliability.

### Time-budget screen

The proposed 10-minute process uses a conservative serial planning screen:
2.5 minutes for queue and picking, 1.0 minute for courier pickup and departure,
1.5 minutes for building access and handoff, leaving 5.0 minutes for travel.
The arithmetic is 10.0 − 2.5 − 1.0 − 1.5 = 5.0 minutes.
For 30-minute orders the screen reserves 6 minutes for queue and picking,
2 minutes for pickup, and 4 minutes for access and handoff.
The remaining travel budget is 30 − 6 − 2 − 4 = 18 minutes.

These component values are planning inputs, not independently measured percentiles.
The candidate area still needs an end-to-end service test using joint observations.
If a building typically takes 7 minutes rather than 1.5 minutes to access and hand off,
the 10-minute tier's five-minute travel boundary is not appropriate there.
For scheduled orders, the test is route completion within the promised window for every stop,
including loading and customer access; a single-order isochrone does not establish this.

### Peak capacity

Consider a 15-minute peak with the following forecast workload.
The 10-minute tier expects 6 orders, the 30-minute tier 8 orders, and the scheduled tier
4 orders whose picking and courier work must be handled in the same bucket.
Each order is assumed to require 2.4 picker labor-minutes under this mix.
Picker workload is (6 + 8 + 4) × 2.4 = 43.2 labor-minutes per 15-minute bucket.
Four pickers supply 4 × 15 × 0.80 = 48 effective labor-minutes in the bucket.
Picker workload therefore fits the 48-minute workload bound, with 4.8 minutes of margin.
That margin does not prove that individual 10-minute orders meet their deadlines.

Courier tasks use an assumed 12 minutes per 10-minute order, including repositioning;
15 minutes per 30-minute order; and 9 minutes per scheduled order after feasible batching.
These are resource-minutes, not elapsed customer-facing delivery durations.
The 10-minute order can have fewer than 10 elapsed minutes yet consume 12 courier-minutes
because repositioning occurs outside its customer-facing clock.
The scheduled figure is an average allocation of shared route work and must be validated
against actual feasible routes and safe food holding times.

Required courier work is 6 × 12 + 8 × 15 + 4 × 9 = 228 courier-minutes.
Eighteen couriers at 15 minutes each and 80% planning utilization supply
18 × 15 × 0.80 = 216 usable courier-minutes.
The forecast workload exceeds this bound by 12 courier-minutes in the peak bucket.
Thus picker capacity is not the aggregate bottleneck under these particular assumptions;
courier capacity is.

The manager cannot solve the problem by reporting 18 couriers as 270 available minutes
while ignoring the agreed 80% planning-utilization allowance.
With this simplified workload model, one additional 10-minute order of capacity would
cover the 12-minute shortfall, but it would not prove location or deadline feasibility.
Alternatively, reserve one fewer 10-minute order in this bucket if demand and contractual
status permit; never cancel an already accepted order merely to make a planning table fit.
A slower tier can be offered only if its independent capacity calculation supports it.
Twelve minutes of courier work cannot be removed by moving a 10-minute order to a
30-minute tier under these assumptions, because that tier uses 15 courier-minutes per order.
Moving one eligible scheduled order to a different window releases 9 minutes, leaving
3 minutes still to be found.

The shift lead closes the affected 10-minute offer before accepting more commitments if
live capacity is inadequate.
The scheduled slot owner can close a future slot or move unaccepted demand to another window.
The dispatch lead can add genuinely available supply or change feasible routes.
The safety lead can suspend deliveries if conditions deteriorate.
The planner must still check picker queues, courier locations, and end-to-end service.

### Footprint choice under uncertain demand

Before signing the lease, the network lead compares two mutually exclusive designs.
This is a separate illustrative scenario, not an additional site in the preceding capacity calculation.
Both options cover the same district, operate for 30 days, and are compared with the same
existing network; all amounts are USD per month and exclude tax.
The figures are hypothetical forecasts rather than measured service outcomes.

| Input | One larger site | Two smaller sites |
|---|---:|---:|
| Forecast accepted orders per day | 900 | 1,000 |
| Contribution per accepted order | USD 5.00 | USD 4.70 |
| Avoidable fixed operating cost per month | USD 95,000 | USD 125,000 |
| Monthly capital and working-capital equivalent | USD 15,000 | USD 25,000 |
| Existing-network contribution displaced per month | USD 35,000 | USD 40,000 |

The two-site option is forecast to win 100 more orders per day because more addresses
pass the fast-service screen, not because every address receives a faster promise.
Its lower contribution per order reflects the assumed extra inventory and delivery expense.
Its site fixed cost covers both sites and excludes the separately stated capital equivalent.
The displaced contribution is the value lost in the existing network, not all revenue shifted.

For the one-site option, monthly contribution is 900 × 30 × USD 5.00 = USD 135,000.
Incremental value is USD 135,000 − USD 95,000 − USD 15,000 − USD 35,000,
or −USD 10,000 per month against the existing network.
For the two-site option, contribution is 1,000 × 30 × USD 4.70 = USD 141,000.
Incremental value is USD 141,000 − USD 125,000 − USD 25,000 − USD 40,000,
or −USD 49,000 per month against the same counterfactual.
Under these inputs, retaining the existing network is preferable to either proposed design.
Comparing only the two proposals would identify the one-site option as less unfavorable,
but it would miss the fact that neither beats the current network.

The additional monthly contribution from two sites rather than one is USD 6,000.
Additional fixed cost is USD 30,000, capital equivalent is USD 10,000, and displaced
contribution is USD 5,000; the resulting difference is −USD 39,000 per month.
This reconciliation agrees with −USD 49,000 minus −USD 10,000.
It makes clear that extra geographical reach must create enough incremental contribution
to cover the second site's fixed cost and the demand it diverts.

Holding all other two-site assumptions fixed, break-even requires monthly contribution of
USD 125,000 + USD 25,000 + USD 40,000 = USD 190,000.
At USD 4.70 contribution per order, that is approximately 40,426 orders per month,
or 1,348 orders per day over 30 days after rounding up to a whole order.
The forecast of 1,000 orders per day is about 348 orders per day below that threshold.
The calculation is only a diagnostic: additional volume may require more labor, vehicles,
inventory, or space, and it may change the amount of displaced demand.
If any of those assumptions changes, recalculate the break-even point rather than applying
the same unit contribution and fixed costs beyond the relevant capacity range.

The network lead should map which incremental address cells create the additional demand.
The analyst should check whether those cells have sufficient population, acceptable access,
and a valid travel-time boundary at the relevant hours.
The demand planner should identify how much of the extra volume is genuinely new rather than
customers moving their orders from the incumbent store or from a scheduled window.
The operations manager should establish whether the required peak workload can be served
without degrading accepted-order reliability or creating unsafe courier incentives.
Finance should check lease commitments, one-time opening costs, and the monthly equivalent
against a dated cash-flow comparison before treating any proposal as investable.

If a controlled pilot shows higher incremental demand, update the forecast and the
counterfactual together; do not update only the proposed site's order count.
If the pilot shows poor service at the outer edge, shrink the offer boundary or revise the
facility design before quoting those addresses, and update the demand forecast accordingly.
If service is safe and reliable but both options remain negative, the commercial owner can
test a different fee or assortment only with a new demand and contribution estimate.
The accountable general manager then records the chosen alternative, evidence, assumptions,
and review date rather than approving a footprint solely because it increases coverage.

### Contribution per order

The following unit economics are assumptions for the same month.
Product sales and fees are revenue; product cost and the listed operating costs are deductions.
All delivery labor, including the example's assumed repositioning, is included in courier cost.
A separate service-recovery allowance is an expected variable cost, not a guarantee that
actual refunds will equal the allowance on each order.

| USD per order | 10-minute | 30-minute | Scheduled |
|---|---:|---:|---:|
| Product sales | 24.00 | 32.00 | 45.00 |
| Delivery fee | 3.00 | 2.00 | 1.00 |
| Product cost | 16.80 | 22.40 | 31.50 |
| Picking and packing labor | 1.20 | 1.40 | 1.80 |
| Courier cost | 5.00 | 4.50 | 3.50 |
| Packaging and payment | 0.80 | 1.00 | 1.20 |
| Waste and recovery | 0.70 | 0.60 | 0.70 |
| Contribution | 2.50 | 4.10 | 7.30 |

For the 10-minute tier, contribution is 24 + 3 − 16.8 − 1.2 − 5 − 0.8 − 0.7 = USD 2.50.
For the 30-minute tier, it is 32 + 2 − 22.4 − 1.4 − 4.5 − 1 − 0.6 = USD 4.10.
For the scheduled tier, it is 45 + 1 − 31.5 − 1.8 − 3.5 − 1.2 − 0.7 = USD 7.30.
The scheduled tier has the highest assumed contribution, but that does not prove customers
would accept a scheduled window or that its orders are all incremental.

At forecast daily volumes, daily contribution before site fixed cost is:

- 10-minute: 240 × USD 2.50 = USD 600.
- 30-minute: 600 × USD 4.10 = USD 2,460.
- Scheduled: 360 × USD 7.30 = USD 2,628.
- Total: USD 5,688 per day, or USD 170,640 per 30-day month.

Assume avoidable site fixed operating costs of USD 110,000 per month.
The modeled site operating surplus is USD 170,640 − USD 110,000 = USD 60,640 per month,
before omitted corporate costs, financing, tax, and any capital recovery.
This is not the incremental value of opening the site.

Suppose an existing network would have earned USD 45,000 per month in contribution from
orders shifted to the proposed site, and its avoidable fixed costs do not change.
Suppose the incremental monthly equivalent of fit-out and working capital is USD 20,000
under the chosen finance policy, not already in the USD 110,000 operating cost.
Incremental monthly value is USD 60,640 − USD 45,000 − USD 20,000 = USD 4,360.
This comparison assumes all other demand and costs are unchanged and the investment
annualization is appropriate; finance should test cash flows and lease commitments directly.
A small change in demand or delivery cost could reverse a USD 4,360 monthly advantage.

If all volumes fall by 10% with unit contribution and fixed costs unchanged, monthly
contribution is USD 153,576, giving site surplus of USD 43,576.
Holding the counterfactual contribution and capital equivalent constant for this simple
sensitivity, incremental value becomes USD 43,576 − USD 45,000 − USD 20,000
= −USD 21,424 per month.
That is a scenario, not a demand forecast: cannibalization may also change with volume.
Decision: pilot only after closing the peak capacity gap, validating service, and establishing
a causal estimate of incremental demand; the base-case result alone is too fragile.

### Service measurement example

In a completed sample, 1,200 orders were accepted and 1,122 were handed off on time.
Of the remaining 78 orders, 40 were late, 18 were cancelled by the operator,
10 were cancelled by customers, and 10 failed or remained open at the measurement cutoff.
The categories are mutually exclusive and sum to 1,200.
The accepted-order on-time rate is 1,122 / 1,200 = 93.5%, below the 95% internal target.
A delivered-only rate would be 1,122 / (1,122 + 40) ≈ 96.6%; it hides the 38 orders
without completed delivery and must be labelled as a secondary diagnostic.
Separately report the 10 customer cancellations so policy owners can interpret the metric.
Allow a sufficiently mature observation window before classifying open orders as failures.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Save it as a local `.py` file and run it with `python filename.py`.
It reproduces the example's peak workload and monthly economics.
It uses deterministic averages, a single 15-minute bucket, and fixed counterfactual costs;
it is not a queue simulation, routing engine, statistical service forecast, or investment appraisal.
Values are illustrative; no external data or network access is required.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    name: str
    daily_orders: int
    peak_orders: int
    sales: float
    fee: float
    product_cost: float
    pick_cost: float
    courier_cost: float
    packaging_payment: float
    waste_recovery: float
    courier_minutes: float

    def contribution(self) -> float:
        return (
            self.sales
            + self.fee
            - self.product_cost
            - self.pick_cost
            - self.courier_cost
            - self.packaging_payment
            - self.waste_recovery
        )


tiers = [
    Tier("10-minute", 240, 6, 24, 3, 16.8, 1.2, 5, 0.8, 0.7, 12),
    Tier("30-minute", 600, 8, 32, 2, 22.4, 1.4, 4.5, 1, 0.6, 15),
    Tier("scheduled", 360, 4, 45, 1, 31.5, 1.8, 3.5, 1.2, 0.7, 9),
]

bucket_minutes = 15
productivity = 0.80
pickers = 4
couriers = 18
pick_minutes_per_order = 2.4
fixed_site_month = 110_000
counterfactual_contribution_month = 45_000
capital_equivalent_month = 20_000
days = 30

peak_orders = sum(t.peak_orders for t in tiers)
pick_load = peak_orders * pick_minutes_per_order
pick_capacity = pickers * bucket_minutes * productivity
courier_load = sum(t.peak_orders * t.courier_minutes for t in tiers)
courier_capacity = couriers * bucket_minutes * productivity

daily_contribution = sum(t.daily_orders * t.contribution() for t in tiers)
monthly_contribution = days * daily_contribution
site_surplus = monthly_contribution - fixed_site_month
incremental = site_surplus - counterfactual_contribution_month - capital_equivalent_month

for tier in tiers:
    print(f"{tier.name} contribution: USD {tier.contribution():.2f}/order")
print(f"Pick: {pick_load:.1f}/{pick_capacity:.1f} labor-minutes per bucket")
print(f"Courier: {courier_load:.1f}/{courier_capacity:.1f} courier-minutes per bucket")
print(f"Daily contribution: USD {daily_contribution:,.2f}")
print(f"Monthly site surplus: USD {site_surplus:,.2f}")
print(f"Incremental monthly value: USD {incremental:,.2f}")
print(f"On-time accepted orders: {1122 / 1200:.1%}")
```

Expected output:

```text
10-minute contribution: USD 2.50/order
30-minute contribution: USD 4.10/order
scheduled contribution: USD 7.30/order
Pick: 43.2/48.0 labor-minutes per bucket
Courier: 228.0/216.0 courier-minutes per bucket
Daily contribution: USD 5,688.00
Monthly site surplus: USD 60,640.00
Incremental monthly value: USD 4,360.00
On-time accepted orders: 93.5%
```

For production financial accounting use a decimal currency representation and a documented
rounding policy; binary floating point is sufficient for this small teaching calculation.
To extend the calculation responsibly, replace averages with order-level time-stamped work,
model work spanning bucket boundaries, and incorporate courier position and route feasibility.
The model must preserve already accepted orders when assessing a new one.
Evaluate service by tier and location, including cancellations and orders that never complete.

## 8. Data, standards, and regulatory context

### Minimum data contract

An order record needs a stable order identifier, store identifier, tier and offer version,
quoted deadline or window, time zone, acceptance event, and final disposition.
Retain original and revised ETA separately from the actual contractual promise.
Keep the event source and ingestion timestamp so late or corrected events can be reconciled.
Do not infer an accepted order from a payment event if acceptance is a separate action.
For scheduled orders preserve the selected window, slot capacity reservation, and changes.

A fulfillment event stream should distinguish pick queued, pick started, pick completed,
order staged, courier assigned, courier arrived, pickup, delivery arrival, and handoff.
Not every event occurs in every order; document optional events and reason codes.
A courier may arrive before picking completes, so event order and concurrency matter.
Track substitutions, missing items, cancellation actor, failed delivery reason, and credits.
Store event instants in an unambiguous format and retain local zone for reporting.
Monitor duplicate events, clocks out of sync, missing timestamps, and impossible sequences.

A capacity table needs employee or courier availability by time, location, and mode,
without exposing more personal data than the operational purpose requires.
Distinguish scheduled, logged-in, available, assigned, and actually productive time.
Track safety blocks, manual overrides, and store opening status as time intervals.
A forecast table needs the version, issue time, bucket, location, tier, and uncertainty.
A cost table needs currency, period, accounting basis, and whether the cost is avoidable.

### Standards and obligations

[GS1 identification keys](https://www.gs1.org/standards/id-keys) provide a common way to
identify products, locations, and other entities when used within an adopted data architecture.
A Global Trade Item Number (GTIN) can help reconcile item identity across systems;
a Global Location Number (GLN) can identify a location or party when properly assigned.
Neither key certifies stock accuracy, guarantees freshness, or determines a service promise.
Whether a particular operator must use a particular identifier depends on its contracts and law.

For food safety, the [U.S. Food and Drug Administration Food Code](https://www.fda.gov/food/fda-food-code)
is a model code for retail food and food service, not automatically binding in every jurisdiction.
The applicable state, local, or other authority may adopt and amend a version.
Check local rules for temperature control, handling, labeling, allergen information, and transport.
The [FDA Food Safety Modernization Act overview](https://www.fda.gov/food/food-safety-modernization-act-fsma)
describes a separate U.S. regulatory framework whose applicability depends on the entity and activity.
A short advertised delivery time does not replace a validated food-safety procedure.

In the European Union, [Regulation (EC) No 852/2004](https://eur-lex.europa.eu/eli/reg/2004/852/oj)
sets food-hygiene requirements for food business operators subject to its scope.
Its applicability to a specific facility, delivery activity, and jurisdiction needs legal review.
Temperature monitoring, documented procedures, and traceability controls should be designed
with the food-safety owner and applicable local rules; this lesson is not legal advice.

Location and customer-order histories may be personal data.
The [EU General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
contains purpose limitation, data minimization, retention, and other requirements for processing
within its scope; lawful basis and other obligations depend on the actual processing context.
Restrict access, minimize precision where feasible, define retention, and review vendors and
cross-border transfers with privacy counsel.
Aggregating to cells is useful but does not automatically make data anonymous.

Courier classification, working time, pay, insurance, and road safety depend on jurisdiction,
contractual facts, and the actual relationship, not on the name of a platform role.
Engage local labor and safety specialists before applying incentives or scheduling policies.
Consumer-protection rules may also govern delivery representations, fees, and remedies.
Do not claim that a customer waiver, internal target, or model score eliminates those duties.
Record the relevant legal review date and monitor changes; no universal legal clearance is implied.

## 9. Industry practice and case evidence

### Evidence standard

A public company description can establish that a service or operating model was announced.
It does not establish that every order met the promise, that the service was profitable,
or that the same footprint will work in another city.
Where a source is a company statement, treat it as evidence of the company's description,
not as independently audited evidence of the full operating outcome.
The examples below illustrate design choices, not recommended investments.

### Rapid grocery and local fulfillment

[Instacart's company overview](https://www.instacart.com/company/about-us/) describes a
marketplace connecting customers with retailers and delivery and pickup services.
Its model demonstrates that local inventory and a customer-facing promise can be coordinated
without assuming that all inventory is held in a separately owned dark-store network.
[INFERENCE] A retailer-backed model may trade a lower dedicated facility burden for
competition with in-store customers, variable stock accuracy, and store-specific pick conditions.
The relative economics depend on contractual terms and local operating data.
The source does not establish the numerical parameters in the Harbor Basket example.

[DoorDash's DashMart page](https://about.doordash.com/en-us/what-we-do/dashmart) describes
DashMart as a convenience and grocery offering using local fulfillment locations.
That is evidence of a local inventory model, not evidence of a particular 10-minute guarantee
or of profitability at a particular site.
[INFERENCE] Dedicated local facilities offer layout control but create site costs and inventory
exposure, so the same admission, safety, and incremental-profit questions still apply.
Review current company materials and local availability before using the case operationally.

### Scheduled delivery and route density

[Ocado Group's annual reports](https://www.ocadogroup.com/investors/reports-and-presentations/)
describe online grocery operations and fulfillment technology in company disclosures.
Scheduled grocery illustrates a different design space from ultra-fast local delivery:
orders may be assembled and routed against booked windows, but the exact operating model,
site configuration, and economics vary by business and reporting period.
[INFERENCE] Advance windows can support consolidation, while late route disruptions can
propagate to subsequent customers; this is an operational hypothesis to validate locally.
A corporate annual report is not proof that all scheduled services meet a particular target.

### Delivery estimates and customer communication

[Google Maps Platform's Routes API documentation](https://developers.google.com/maps/documentation/routes)
describes route and travel-time computation capabilities.
It is a technical source for a possible routing input, not proof that an operator's pick queue,
courier supply, building access, or full order-to-handoff SLA is feasible.
A route prediction must be combined with current operational state and independently validated
against end-to-end outcomes.
Provider coverage, pricing, traffic assumptions, and legal terms should be checked before use.

### What can and cannot be transferred

Transfer the decision structure: define the commitment, establish feasible resources,
measure the outcome, and compare against the alternative.
Do not transfer a site density, radius, fee, courier utilization, or average speed without
matching local demand, road network, labor arrangements, and customer behavior.
A company's announcement may be obsolete or describe an offer only in selected markets.
Before treating it as current evidence, check the dated source and the actual service geography.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation and owner |
|---|---|---|---|---|
| Deadline starts too late | Checkout-to-acceptance time grows | Internal clock omits customer waiting | Misleading service reporting | Product defines clock; data owner reconciles events |
| Delivery endpoint is vague | Many long building-arrival-to-handoff intervals | Arrival and actual transfer are conflated | Customer sees a miss reported as success | Product specifies endpoint; ops captures handoff |
| Circle replaces network travel | Edge addresses repeatedly miss | Barriers and one-way routes add time | Unreliable catchment | Network lead validates mode-specific isochrones |
| Boundary ignores buildings | Apartment clusters have high late rates | Elevator and access delays are omitted | Unequal customer experience | Ops adds access distributions and exceptions |
| Picker queue hidden | Pick-start delay rises while average pick time stays flat | Utilization and bursts create waiting | Misses despite apparently fast workers | Shift lead gates on queue age and deadline risk |
| Courier supply overstated | Assigned orders wait for acceptance | Logged-in couriers are not all available | Dispatch backlog | Dispatch uses accepted availability and location |
| Repositioning omitted | Couriers finish far from stores | Loaded trip time omits return work | Capacity and cost understated | Finance and dispatch include deadhead |
| Batch model is optimistic | Later stops miss after batching rises | Detours and waiting consume slack | Multiple misses per route | Routing owner checks every stop's commitment |
| Stock accuracy is poor | Substitution and picker search time rise | System stock differs from shelf | Late or incomplete baskets | Inventory owner audits and corrects stock |
| Slot overselling | Scheduled backlog accumulates before window | Reservations exceed feasible route work | Window failure and recovery cost | Slot owner caps validated capacity |
| Cancelled orders disappear | Delivered-only rate rises during disruption | Failed commitments leave denominator | False service improvement | Analytics publishes accepted-order reconciliation |
| Faster tier cannibalizes | Fast orders rise while total households do not | Existing customers switch tiers | Profit falls despite attributed growth | Commercial owner runs incremental experiment |
| Duplicate inventory ages | Waste and markdowns rise at small stores | Demand fragmentation slows stock rotation | Margin erosion and food risk | Inventory lead forecasts and monitors expiry |
| Fixed cost omitted | Positive order margin but site losses | Rent and management excluded | Bad footprint investment | Finance publishes site and incremental views |
| Safety sacrificed for SLA | Rushing reports or incidents rise | Incentives reward speed without limits | Worker and public harm | Safety owner can pause service without penalty |
| Customer window changes silently | Original promise overwritten | Latest ETA replaces commitment | Misleading service measurement | Product stores immutable original and revisions |
| Forecast hides closed periods | Low order counts align with sold-out slots | Observed demand is censored | Understaffing repeats | Forecast owner records exposure and availability |
| Offer oscillates | Rapid open/close cycles | Noisy gate and no hysteresis | Confusing customer experience | Product and ops calibrate reopening rules |
| Privacy scope grows | Raw location retained indefinitely | Convenience overrides retention design | Regulatory and trust risk | Privacy owner limits access and retention |
| One average masks disparity | Aggregate service holds while some cells fail | Weighted average hides local problems | Unfair or inconsistent coverage | Analytics reviews cell and time slices responsibly |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Demand forecast represents the period | Converts per-order economics into a monthly plan | Promotions, competitor entry, or censored demand | Backtest by bucket and run low/high scenarios |
| Unit contributions stay constant with volume | Makes the simple sensitivity calculable | Surge pay, discounts, or volume-based supplier terms | Recalculate costs by volume and tier |
| Peak mix uses 2.4 picker-minutes per order | Converts order count to labor work | Larger baskets or missing products | Measure labor-minutes by basket and temperature zone |
| Planning productivity is 80% | Reserves capacity for nonproductive work and variability | Break patterns or variability change | Calibrate against staffing and observed queues |
| Courier task averages include deadhead | Makes workload and courier cost comparable | Repositioning rises at the edge | Use GPS-derived task cycles with privacy controls |
| Scheduled batch average is feasible | Gives a per-order resource estimate | Windows conflict or cold holding is too long | Validate actual routes and food-safety limits |
| One bucket is representative | Allows a transparent peak calculation | Work crosses bucket boundaries or peaks cluster | Use rolling workload and event-based simulation |
| Delivery screen is serial | Allows simple time-budget subtraction | Courier and picker work overlap | Reconstruct critical path from actual events |
| Time-budget components are stable | Converts a deadline into a candidate travel boundary | Weather, access, or traffic changes | Segment outcomes and use live safety/capacity gates |
| Fixed site costs are avoidable | Makes a site comparison meaningful | Lease or labor obligations persist after closure | Classify avoidability and timing with finance |
| Counterfactual contribution is USD 45,000/month | Quantifies an example of displaced demand | Customer behavior or competitor response differs | Estimate incrementality in a controlled pilot |
| Capital equivalent is USD 20,000/month | Provides a simple investment burden | Cash-flow timing or discount rate differs | Finance performs a dated cash-flow analysis |
| Accepted-order metric counts customer cancellations as misses | Makes the service denominator conservative and reproducible | A different business or regulatory definition applies | Report actor-specific categories and approved alternatives |
| A 95% target is a suitable internal threshold | Illustrates a decision rule | Customer terms or economics require another threshold | Set governance-approved target by tier and population |
| Facilities can actually be permitted and staffed | Makes a proposed footprint actionable | Zoning, recruitment, or supplier constraints | Complete site diligence before commitment |

These assumptions are not facts about a named operator.
The case does not calculate a confidence interval or prove that a 95% service target will be met.
The simple sensitivity holds counterfactual and fixed costs constant for transparency;
a full model should let them change where evidence supports it.
The example also omits tax, financing, corporate allocation, and detailed cash-flow timing.

## 12. Exercises

### Exercise 1 — Contract and denominator

Time: 15 minutes. Deliverable: a one-paragraph 10-minute contract and a four-row metric table.
An operator accepts 200 orders: 182 are handed off on time, 8 late, 6 are cancelled by the
operator, and 4 are cancelled by customers before delivery.
Acceptance criteria: specify clock start, endpoint, denominator, accepted-order on-time rate,
and a separately labelled delivered-only rate; explain the customer-cancellation treatment.

### Exercise 2 — Time budget

Time: 15 minutes. Deliverable: a calculation and two caveats.
A 30-minute offer reserves 5 minutes for queue and picking, 3 for pickup, and 4 for handoff.
Acceptance criteria: compute the remaining travel budget; explain why adding separate 95th
percentiles is not generally the 95th percentile of total elapsed time; address overlapping work.

### Exercise 3 — Catchment and admission

Time: 20 minutes. Deliverable: a four-step decision procedure.
An address is inside a five-minute isochrone but its building has a seven-minute handoff time.
The store has a long picker queue and no immediately available courier.
Acceptance criteria: distinguish geographical screening from live acceptance; state whether
the 10-minute offer should be admitted under these facts and identify a feasible alternative
only if separately validated; name the people responsible for an override and safety.

### Exercise 4 — Peak bottleneck

Time: 20 minutes. Deliverable: a workload table and admission decision.
In 15 minutes, three pickers each provide 80% productive time; each order uses 2 pick minutes.
Ten couriers each provide 80% productive time; each order requires 8 courier-minutes.
Forecast demand is 17 orders in the bucket.
Acceptance criteria: compute both workload capacities, identify the bottleneck, quantify the
courier-minute shortfall, and explain why the answer is not a deadline guarantee.

### Exercise 5 — Unit contribution

Time: 15 minutes. Deliverable: an arithmetic bridge in USD per order.
Sales are USD 30, fee USD 2, product cost USD 21, picking USD 1.50,
courier USD 4, packaging/payment USD 0.80, and waste/recovery USD 0.70.
Acceptance criteria: calculate contribution; say which site and corporate costs remain outside
this defined measure and why positive contribution alone does not prove site profitability.

### Exercise 6 — Incrementality

Time: 20 minutes. Deliverable: a monthly comparison and interpretation.
A proposed store has USD 80,000 monthly contribution and USD 55,000 avoidable monthly fixed
operating cost; USD 20,000 of contribution is displaced from an existing site.
Incremental monthly capital equivalent is USD 7,000 and existing fixed costs do not change.
Acceptance criteria: calculate site surplus and incremental value, and explain why they differ;
state one sensitivity that could change the conclusion.

### Exercise 7 — Disruption design

Time: 30 minutes. Deliverable: an incident playbook with owners, signals, and reopening criteria.
A heavy storm slows travel, couriers report unsafe roads, and a scheduled wave is already booked.
Acceptance criteria: prioritize safety, protect accepted customers, distinguish closing new offers
from modifying existing commitments, define communication and an auditable override,
and specify how to reopen based on measured conditions.

### Exercise 8 — Controlled pilot

Time: 30 minutes. Deliverable: a one-page experimental design.
A commercial leader wants to launch a 10-minute tier and count all its orders as new demand.
Acceptance criteria: specify randomization unit and comparison, primary incremental outcome,
cannibalization measurement, spillover risk, service and safety guardrails, and stopping rule.
State how uncertainty and insufficient sample size will affect the decision.

### Exercise 9 — Data and regulatory review

Time: 20 minutes. Deliverable: a minimum data contract and review checklist.
The team proposes keeping precise courier locations forever and using them for unrelated marketing.
Acceptance criteria: identify purpose, access, retention, event timestamps, and original promise;
name privacy, employment, food-safety, and consumer-protection review where applicable.
Do not assert a jurisdiction-specific legal conclusion without facts and qualified review.

## 13. Self-check answers

### Answer 1

The accepted-order on-time rate is 182 / 200 = 91%.
The delivered-only on-time rate is 182 / (182 + 8) ≈ 95.8%.
The latter excludes 10 cancellations and is not a substitute for the former.
Under the lesson's illustrative convention, customer cancellations remain in the accepted-order
denominator and are reported separately; another approved convention must be explicit.
A complete contract identifies acceptance as the start and completed door handoff as the endpoint.

### Answer 2

The travel budget is 30 − 5 − 3 − 4 = 18 minutes.
The calculation is a screening approximation with sequential components.
If pickup and picking overlap, summing their full durations double-counts elapsed time.
The percentile of a sum depends on the joint distribution and correlation of its components;
separate 95th-percentile values cannot simply be added and called the total 95th percentile.
Use observed end-to-end paths or an appropriately validated joint simulation.

### Answer 3

The address passes only the travel screening condition; it does not pass the full time budget.
Seven minutes of building handoff plus five minutes of travel already exceeds ten minutes,
without counting picking or courier pickup.
A long queue and no available courier make the 10-minute offer still less feasible.
Do not accept it under the stated facts.
A 30-minute or scheduled offer is an alternative only after its own resource, route, stock,
and safety checks pass.
The duty manager owns an authorized operational override; safety retains stop authority.

### Answer 4

Pick capacity is 3 × 15 × 0.8 / 2 = 18 orders per bucket.
Courier capacity is 10 × 15 × 0.8 / 8 = 15 orders per bucket.
Seventeen orders require 136 courier-minutes against 120 effective courier-minutes,
a 16-minute shortfall, so courier work is the modeled bottleneck.
Pick workload is 34 minutes against 36 effective minutes.
Aggregate feasibility does not account for arrival bursts, courier location, work spanning
bucket boundaries, or each customer's deadline.

### Answer 5

Contribution is USD 30 + 2 − 21 − 1.50 − 4 − 0.80 − 0.70 = USD 4.00 per order.
Rent, site management, and corporate costs are not in the stated calculation.
A site may still lose money if total contribution is insufficient to cover relevant fixed cost.
Capital and displacement must also be considered for an investment decision.

### Answer 6

Site surplus is USD 80,000 − USD 55,000 = USD 25,000 per month.
Incremental value is USD 25,000 − USD 20,000 − USD 7,000 = −USD 2,000 per month.
The site can have positive operating surplus while reducing network value after displacement
and capital burden.
A lower demand forecast, changed displacement estimate, or higher courier cost can alter the result.

### Answer 7 rubric

A strong playbook gives the safety lead explicit authority to suspend unsafe work.
It prevents new unserviceable promises without pretending existing bookings disappeared.
It assigns customer communication, capacity reassessment, food holding, and recovery owners.
It retains original commitments, decisions, reasons, and timestamps.
It defines objective reopening checks and reviews missed orders after the incident.
A weak playbook tells couriers to travel faster or silently changes customer deadlines.

### Answer 8 rubric

A strong design randomizes comparable eligible units, measures all tiers and eligible units,
and compares contribution against the control rather than counting attributed fast orders.
It anticipates shared-courier and shared-store spillovers and specifies analysis accordingly.
It predefines precision, duration, service and safety guardrails, and decision ownership.
Insufficient precision leads to an inconclusive result, not proof of no effect or success.
A weak design compares launch-week fast orders with zero while ignoring switched orders.

### Answer 9 rubric

A strong contract records original promise, acceptance, status, handoff, and event provenance.
It separates operational necessity from an unrelated marketing use and sets access and retention.
It seeks legal and specialist review for the relevant jurisdictions and actual processing facts.
It does not equate aggregation with guaranteed anonymity or make unsupported legal claims.
A weak answer assumes that data can be retained forever merely because it might later be useful.

## 14. Interview questions

1. How would you define on-time delivery for a 10-minute offer?
   A strong answer specifies clock start, completed handoff, denominator, cancellations, and revisions.
   A weak answer reports courier arrival time without a customer commitment definition.
2. Why can a five-minute travel isochrone fail to support a 10-minute promise?
   A strong answer includes queue, pick, pickup, handoff, variability, and live supply.
   A weak answer assumes every order starts traveling at acceptance.
3. What changes when customers book a scheduled window?
   A strong answer discusses reservations, route sequencing, early arrival, capacity, and cold chain.
   A weak answer assumes batching always improves both cost and service.
4. How do you identify a peak bottleneck?
   A strong answer converts demand into comparable resource-minutes and tests each constraint.
   A weak answer compares headcount without considering work, availability, or order mix.
5. Why is 100% planned courier utilization risky?
   A strong answer relates variability, queue growth, location, and service slack.
   A weak answer assumes a busy courier is instantly available for every new order.
6. When would you close an immediate tier but leave a slower one open?
   A strong answer uses independent capacity and safety checks while protecting accepted work.
   A weak answer simply changes the label on the same infeasible backlog.
7. How would you compare a new dark store with an existing store network?
   A strong answer includes displaced contribution, avoidable fixed costs, capital, and sensitivity.
   A weak answer multiplies all new-site orders by contribution and calls the result profit.
8. How do you measure whether a fast tier generates incremental demand?
   A strong answer proposes a control, all-tier outcomes, spillover handling, and uncertainty.
   A weak answer counts every fast-tier order as a newly acquired customer.
9. What is wrong with adding the 95th percentile of each activity?
   A strong answer explains overlap, dependence, and the joint end-to-end distribution.
   A weak answer says the sum must always equal the 95th percentile of the total.
10. How should safety affect an otherwise profitable service decision?
    A strong answer treats safety and legal constraints as gates with stop authority.
    A weak answer suggests courier pressure or faster driving to recover the SLA.

## 15. Key terms

- [Acceptance](#4-definitions-and-vocabulary): event starting the example's immediate-tier clock.
- [Admission control](#step-9--match-workload-to-capacity-and-gate-acceptance): decision to accept or withhold an offer.
- [Basket](#4-definitions-and-vocabulary): ordered items and quantities.
- [Bottleneck](#step-8--size-courier-capacity): resource limiting the modeled workload.
- [Capacity](#step-7--size-picker-capacity): sustainable work over a defined period.
- [Catchment](#step-5--construct-feasible-catchments): conditional geographical eligibility.
- [Commitment](#step-2--specify-each-service-contract): original customer-facing deadline or window.
- [Contribution](#contribution-per-order): defined revenue less defined variable costs.
- [Counterfactual](#step-1--state-the-decision-and-counterfactual): best feasible comparison condition.
- [Dark store](#step-6--design-facility-and-inventory-options): local facility organized for fulfillment.
- [Deadhead](#step-8--size-courier-capacity): courier travel without an active customer order.
- [Demand density](#step-3--segment-demand-and-geography): orders per area and time.
- [ETA](#4-definitions-and-vocabulary): evolving estimated time of arrival.
- [Forecast](#decision-and-data): estimate of future demand, not a commitment.
- [Handoff](#step-2--specify-each-service-contract): completed customer transfer under the contract.
- [Incremental profit](#contribution-per-order): proposal profit less counterfactual profit.
- [Isochrone](#step-5--construct-feasible-catchments): network area reachable within a time estimate.
- [On-time reliability](#service-measurement-example): on-time accepted commitments divided by accepted commitments.
- [Pick time](#step-7--size-picker-capacity): labor or elapsed time to collect an order.
- [Plan](#decision-and-data): chosen allocation of resources and reservations.
- [Promise tier](#step-2--specify-each-service-contract): defined customer service class.
- [Queue](#step-9--match-workload-to-capacity-and-gate-acceptance): work awaiting a constrained resource.
- [Scheduled window](#step-2--specify-each-service-contract): agreed delivery interval.
- [Slack](#step-8--size-courier-capacity): reserved time or capacity for variation.
- [Target](#decision-and-data): internal desired performance, distinct from a commitment.
- [Utilization](#step-8--size-courier-capacity): busy time divided by available time.

## 16. Sources

Sources are cited for the specific claims stated above; none validates the illustrative case.
Access current terms, applicable versions, and local law before making an operational decision.

1. GS1, [Identification Keys](https://www.gs1.org/standards/id-keys), standards overview.
2. U.S. Food and Drug Administration, [FDA Food Code](https://www.fda.gov/food/fda-food-code), model code and editions.
3. U.S. Food and Drug Administration, [Food Safety Modernization Act](https://www.fda.gov/food/food-safety-modernization-act-fsma), regulatory overview.
4. European Parliament and Council, [Regulation (EC) No 852/2004 on the hygiene of foodstuffs](https://eur-lex.europa.eu/eli/reg/2004/852/oj), official legal text.
5. European Parliament and Council, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), official legal text.
6. Instacart, [About Instacart](https://www.instacart.com/company/about-us/), company description; not independently audited service evidence.
7. DoorDash, [DashMart](https://about.doordash.com/en-us/what-we-do/dashmart), company description; not independent site-level profitability evidence.
8. Ocado Group, [Reports and presentations](https://www.ocadogroup.com/investors/reports-and-presentations/), company disclosures; select the relevant reporting period.
9. Google Maps Platform, [Routes API documentation](https://developers.google.com/maps/documentation/routes), technical documentation; route estimates are not end-to-end delivery commitments.
10. J. D. C. Little, “A Proof for the Queuing Formula: L = λW,” *Operations Research* 9(3), 1961, pp. 383–387, [DOI: 10.1287/opre.9.3.383](https://doi.org/10.1287/opre.9.3.383). The result relates average work in a stable system, throughput, and time; it does not itself establish a deadline percentile.

## 17. Links

- [Module 22 index](_index.md).
- [Module 22 survey and its lab sections](../../../course/22_quick_commerce_on_demand_delivery.md).
- [Short-horizon forecasting](02_short-horizon-forecasting.md): demand by peak and slot.
- [Dark-store assortment](03_dark-store-assortment.md): inventory breadth and space.
- [Courier supply and demand](05_courier-supply-demand.md): supply availability and incentives.
- [ETA prediction](06_eta-prediction.md): end-to-end uncertainty and measurement.
- [Batching and routing](07_batching-routing.md): multi-stop route feasibility.
- [Unit economics](09_unit-economics.md): contribution and cost definitions.
- [Food safety and welfare](10_food-safety-welfare.md): non-negotiable operating constraints.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md): methods for testing capacity and routing scenarios.
