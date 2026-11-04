# Last Mile: Density, Windows, and Failure

> Module 12 — Transportation, Routing, and Last-Mile Analytics
> Topic 08 of 10
> Estimated study time: 3 hours
> Prerequisites: topics 01–03
> Version 2026.5

## 1. Why this topic exists

A delivery manager must choose a service offer before the number and location of tomorrow's orders are known. A narrow appointment can win an order while destroying the route density that makes the order profitable. A failed attempt can consume a stop, a customer contact, storage, a second trip and possibly a refund. None of those costs is necessarily visible in the original carrier invoice.

The decision is not simply whether a route has low miles. It is whether the promised service, the feasible daily route and the recovery process together produce acceptable contribution and customer outcomes. This topic gives a way to price or redesign the promise using a consistent order cohort and explicit counterfactuals.

All numerical examples are fictional, in 2026 USD, and are assumptions for teaching, not industry benchmarks. Actual cost and service distributions must be measured locally.

### The decisions behind the delivery promise

A retail commercial manager sees checkout conversion and delivery revenue.
A dispatch manager sees tomorrow's vehicles, drivers, stop sequence and access restrictions.
Customer care sees the calls that follow a missed commitment.
Finance sees invoices, refunds and margin, often after the operational choices have been made.
None of these views alone answers whether a delivery product creates value.

Consider a customer who pays for a two-hour appointment.
The paid fee is visible immediately, but the constraint it imposes on other stops is not.
A driver may wait outside the first appointment, travel across the territory for the second,
and return to the first neighborhood for a later appointment.
The customer may receive better service even as miles and paid hours increase.
The decision is whether that service benefit and any additional demand justify the cost.

The opposite error is treating every narrow window as waste.
An appointment that coincides with a dense cluster can fit inside unused route capacity.
It may also prevent an expensive failed handoff for a customer who needs to be present.
Its marginal cost depends on the rest of the route, not simply on its width.
The operator needs a mechanism to quote and reserve feasible capacity before accepting payment.

A failed first visit creates a second commercial decision.
The operator can retry, seek customer instructions, offer a collection location, or return the order.
These alternatives use different resources and place different burdens on the customer.
A cheaper carrier invoice does not automatically mean a cheaper total recovery,
and an efficient depot does not automatically make a pickup point accessible.

This lesson is for the manager choosing the offer and the analyst making the trade-off visible.
It does not prescribe a universal window, density threshold, or driver productivity target.
Those choices depend on local demand, labor rules, travel times and service commitments.
Use the calculations to frame decisions and then measure the actual network response.

## 2. Learning outcomes

After this topic you can:

1. Define attempt, completed order and first-attempt success with compatible denominators.
2. Explain why geographic and temporal density affect stops per paid route hour.
3. Calculate contribution per order including incremental failure recovery.
4. Compare a window premium with the cost and demand effects of offering it.
5. Design a fair comparison of service areas with different demand mix.
6. Specify a pilot, operational controls and a decision rule.
7. Identify when a pooled success rate conceals inequitable service.

The exercises and answers cover each outcome.

### What you should be able to deliver

The first two outcomes lead to an auditable operating scorecard:
the cohort reconciliation in exercise 9 and the unit conversion in exercise 2.
The third and fourth lead to a fee decision with an explicit incremental-cost boundary,
as tested in exercises 3, 4 and 8.
The fifth through seventh lead to a pilot and a responsible service redesign,
as tested in exercises 6, 7 and 10.

A correct answer states the decision horizon and what remains unknown.
It distinguishes a planned route from actual paid time,
and a forecast failure rate from a measured causal effect.
The arithmetic must be traceable to the unit and cohort stated alongside it.

## 3. The core mental model

```text
Demand and customer choices
  -> locations, promised windows, access constraints
  -> feasible routes and paid driver hours
  -> delivery attempts -> completed orders
                         -> failed attempt -> contact / retry / return
  -> service, cost, customer outcome -> next promise and price
```

Each arrow is a decision boundary. A route plan is a forecast of work, not proof that the route is executable. A customer promise is a commitment, not the driver's predicted arrival. Operations can revise the plan but must manage the commitment and notify the customer when a breach becomes likely.

Density has at least three dimensions. Spatial density is the proximity of stops in the actual travel network. Temporal density is the overlap of service windows. Drop density is completed packages or orders per stop. A multi-unit building may have many orders but still require separate access and handoff work. Counting points on a map is not a substitute for measuring paid time.

The same route may contain several orders at one address. Define a stop as an arrival at a service location, an attempt as a genuine effort to hand over a particular order, and a completion as an accepted delivery. Never divide route cost by stops in one period and compare it with cost per order in another without a bridge.

### The three clocks

The customer clock starts when the promise is accepted and ends when the order is received.
The operating clock starts before departure, with loading and route preparation,
and includes every paid minute until the route and its exceptions are closed.
The accounting clock may continue through a retry, a return, a refund or a later invoice.
If the report cuts off at departure or first attempt, it cannot describe the entire customer outcome.

These clocks overlap but do not start and end together.
An order can be on a route that departed on time and still miss its promised interval.
It can miss the interval and later be delivered successfully on the first attempt.
It can also arrive within the interval and fail because the recipient or location is inaccessible.
Report on-time arrival, first-attempt success and eventual completion separately.

### A map is not a schedule

Picture six neighboring addresses with one-hour windows.
If all windows overlap and service takes ten minutes per address,
the stops may fit into a single hour only if travel and parking consume no additional time.
In practice they consume time, so an apparently compact cluster can be infeasible.
If the windows span the whole day, the same six addresses provide more sequencing freedom.
If the appointments alternate between distant neighborhoods, a short window can increase travel as well as waiting.

The alternative is not always a different sequence on the same vehicle.
An extra vehicle, overtime shift, earlier departure, or different delivery day may be required.
Each has a different lead time and avoidable cost.
The route feasibility test must consider these capacity steps, not only average cost per stop.

### The commercial feedback loop

The offer changes which customers buy, when they order and which windows they select.
That choice changes the locations and time constraints that dispatch must serve.
Dispatch outcomes change customer experience and potentially future demand.
Consequently a historical comparison between premium buyers and standard buyers
does not isolate the effect of offering a premium.
The groups may have different buildings, urgency and availability before the offer is made.

An experiment varies the offer or available capacity and keeps the assignment record.
Operational and customer outcomes then flow back into the next offer decision.
A simulation can explore feasible configurations before this experiment,
but its travel times, service durations and demand must be validated against execution.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Order | Commercial demand unit, orders | Treating every parcel as a distinct customer order |
| Parcel | Physical package, parcels | Equating parcels with delivery visits |
| Stop | Vehicle service location visit, stops | Assuming a stop always serves one order |
| Attempt | Documented handoff effort for an order | Counting a label creation as an attempt |
| First-attempt success | Orders delivered on first attempt / eligible orders with a first attempt | Dividing successes by all attempts |
| Route hour | Paid hours assigned to route, hours | Using driving time alone |
| Spatial density | Eligible stops per geographic area or travel-network measure | Assuming area alone captures barriers |
| Window | Contractual interval in local time | Treating estimated arrival as a promise |
| Dwell | Time stopped at a location, minutes | Treating all dwell as customer delay |
| Recovery | Actions after failed attempt | Omitting return and customer-care costs |
| Contribution | Revenue less specified incremental costs, USD | Calling contribution accounting profit |
| Cost to serve | Allocated or incremental cost for defined unit | Mixing fixed and avoidable cost without labels |
| Cohort | Orders grouped by a common start event and period | Comparing mature and immature outcomes |
| Counterfactual | Outcome under the alternative decision | Assuming before/after differences are causal |

Additional terms used in the operating procedures:

| Term | Definition and unit | Common misuse |
|---|---|---|
| Accepted-order cohort | Orders accepted during a defined period, orders | Starting from only successfully scanned deliveries |
| Attempt sequence | Ordered handoff efforts for one order, ordinal number | Treating a repeated scan as a second visit |
| Breach | A failure to meet the defined commitment, orders or events | Calling every revised ETA a breach |
| Customer availability | Time and conditions under which a handoff can occur | Inferring availability from a delivery failure alone |
| Demand response | Change in order volume or mix caused by an offer | Treating paid uptake as net new orders |
| Exception margin | Planned capacity not assigned to scheduled work, minutes or hours | Describing spare capacity as automatically wasted |
| Geocode | Estimated geographic position for an address | Treating a map pin as a verified entrance |
| Intention to treat | Analysis according to assigned offer, including non-users | Comparing only customers who opted in |
| Marginal route cost | Cost of the additional resources caused by a decision, USD | Applying an average cost to every extra order |
| Maturity horizon | Elapsed time allowed for outcomes to resolve, days | Ignoring returns that occur after a report closes |
| On-time delivery | Delivery meeting the specified time commitment, orders / defined cohort | Substituting first-attempt success without checking time |
| Paid uptake | Eligible orders or customers paying the premium / specified eligible population | Treating uptake as causal demand lift |
| Proof of delivery | Evidence supporting a handoff event | Assuming every proof record is valid without audit |
| Route feasibility | Ability to satisfy stated constraints for a proposed route | Guaranteeing execution in uncertain traffic |
| Service duration | Time required for the specified stop activity, minutes | Using driving time as total route time |
| Stem time | Travel between depot and service territory, minutes | Counting it as local stop-to-stop travel |
| Window overlap | Common usable time between commitment and operating constraints, minutes | Assuming nominal windows guarantee adequate service time |

## 5. The mechanics, step by step

### 5.1 Set the decision and boundary

Name the owner, geography, service population, decision date and alternative. For example: should a retailer offer a two-hour window at a USD 4 premium, or retain the all-day offer? The relevant costs are the costs that change because of this decision, not necessarily the fully allocated cost of the depot.

Fix the unit as an eligible order. Determine whether a cancelled order, split shipment or customer-selected collection is eligible before examining success. Use a mature cohort so that a delayed retry does not disappear at the reporting cutoff. Record order, parcel, stop, attempt and route identifiers separately.

### 5.2 Map the operating process

Capture order acceptance, promise, route assignment, departure, arrival, attempt, proof of delivery, failure reason and resolution. Keep event time and ingestion time. Record who supplied the event and whether the event was corrected. Deduplicate events using a durable event identifier, not a timestamp alone.

A driver marking an address inaccessible and a customer declining delivery are different events. A missing scan is not proof of either failure or success. Reconcile the order ledger against the dispatch and proof-of-delivery feeds and quantify the unexplained remainder. Assign owners to resolve mismatches rather than silently dropping them.

### 5.3 Measure route productivity

For each route, paid hours include loading, travel, service, breaks where paid, return and paid exceptions. Completed orders per paid hour equals completed orders divided by paid route hours. Compute a ratio of totals for the aggregate; the unweighted mean of route-level ratios gives every route equal influence regardless of hours.

Separate road travel, stem travel between depot and service area, service at stops and waiting for the window. A dense cluster far from a depot can have low local travel but high stem time. In a sparse area, a small increase in stop count may improve productivity substantially; at a capacity-constrained route the same increase may require another vehicle.

### 5.4 Model failure and recovery

First-attempt success for a mature eligible cohort equals first-attempt completions divided by orders on which a first attempt was made. Separately report orders never attempted, cancellations and unresolved outcomes. If the intended denominator is all accepted orders, label the metric accordingly and include all accepted orders.

Estimate the incremental recovery cost per first-attempt failure from actual recovery pathways: second attempt, collection point, return to sender, refund, customer contact and write-off. Do not sum mutually exclusive paths as if every failed order experiences each. Multiply each pathway cost by its conditional probability, then sum. Include the initial failed attempt's cost in the route cost, not again in the recovery cost.

### 5.5 Test window economics

A shorter promise can reduce waiting for the customer but constrain sequencing. Compare route hours, vehicles, completion and demand under the same order population and traffic assumptions. A route simulation is a model of feasibility, not evidence of achieved savings. Check driver breaks, capacity, access times, loading and realistic service durations.

Incremental contribution equals incremental premium revenue and other incremental revenue, less incremental delivery cost, incremental expected recovery cost and other decision-specific costs. If the premium causes a different set of customers to order, incorporate that demand change separately rather than pretending that all orders would have occurred under both offers. Do not count an avoided refund both as recovery savings and as incremental retained revenue.

### 5.6 Segment without confusing mix and causation

Stratify by service area, building type, access method, day, promised window and order characteristics when these affect the decision. Protect small cells and sensitive location data. Compare like with like, but do not condition on a post-treatment route assignment when estimating the effect of the promise: the promise itself may change assignment.

A high failure rate in one area is an invitation to investigate access, communications and route feasibility, not evidence of customer fault. Compare missing-data rates too. An intervention can raise the reported success rate merely by excluding hard-to-deliver orders.

### 5.7 Run and govern a pilot

Randomize eligible customers or geographic clusters where operational interference makes individual assignment impractical. Predefine the population, primary contribution metric, service guardrails, observation horizon and handling of spillovers. Measure all assigned orders using intention to treat; describe opt-in and actual exposure separately.

Review operational feasibility with dispatch and drivers before launch. Set a stop rule for unsafe workload, unacceptable breach rate or inaccessible service. At the end, report uncertainty, not only the point estimate. A pilot is a local estimate for its own season, demand mix and network configuration.

### 5.8 Translate a promise into operational capacity

Start with the order cutoff and the earliest possible vehicle departure.
An order accepted after route construction may require an insertion, a separate vehicle, or a later promise.
Treat cutoff compliance as part of the offer, not as a driver performance measure.

For each prospective service window, calculate the overlap with the driver's shift and the site's access hours.
The intersection must contain enough time for travel, parking, handoff and required breaks.
If it does not, reject the promise before payment or offer a feasible alternative.
Do not count a customer-selected window as feasible merely because a software calendar displays it.

When a route serves a building with multiple orders, distinguish shared approach time from individual handoff time.
A concierge may accept all parcels at once, while secure individual handoffs may require separate journeys to each floor.
Both have one geographic location but different labor requirements.
Measure the actual access procedure before assigning a fixed productivity credit.

Use capacity reservations to prevent the sales channel from selling the same scarce route minutes repeatedly.
Refresh the reservation when an order is cancelled, while retaining an audit record.
Reserve a documented exception margin for traffic, absent customers and vehicle failures.
Avoid allocating every available minute to planned stops.

### 5.9 Construct the economic ledger

Agree a time horizon with finance before comparing alternatives.
For a daily dispatch decision, a salaried depot lease may not change; a temporary driver shift may.
For a network redesign, the same lease may become avoidable at renewal.
Classify each item according to the decision horizon rather than calling every fixed cost irrelevant.

Use separate lines for order margin, delivery fee, driver time, vehicle use, customer contact and recovery.
Include payment processing or refunds only when the treatment changes them.
Document whether the order margin is gross of product cost and whether delivery fees include tax.
Do not add a fee collected on behalf of another party to retained revenue.

When failure changes the timing of a delivery rather than its eventual completion, follow the order to final disposition.
A second visit may share a route with other deliveries, so its avoidable cost need not equal a dedicated trip.
Estimate the marginal resource consumption or compare capacity-constrained scenarios.
If the second visit causes an additional vehicle shift, include that discontinuity.

### 5.10 Investigate low first-attempt success

Rank failure reasons only after checking their completeness and consistency.
An operational code may be a driver's selection from a short menu, not an independently verified cause.
Sample records against call logs, access notes and proof-of-delivery evidence where authorized.
Separate coding defects from genuine service defects.

For access failures, check address quality, entrance identification, opening hours and the recipient's delivery instructions.
For unavailable recipients, check whether the communicated window matched the committed window.
For damaged freight, investigate handling and packaging rather than scheduling another identical attempt.
For unsafe locations, use the organization's safety escalation instead of instructing a driver to proceed.

Compare interventions by both effectiveness and cost.
A pre-arrival message may prevent some failures but can exclude customers without reliable digital access.
A staffed pickup point may help some customers while imposing travel or accessibility costs on others.
Offer an accessible alternative and measure who cannot use the intervention.

### 5.11 Convert findings into a decision

Give the decision maker the incremental contribution, service outcome and unresolved risks side by side.
Show the accepted-order count, attempted-order count and final disposition count.
Explain whether the estimate is from a randomized offer, an observational comparison or a simulation.
State the time period, geographic scope and uncertainty of each estimate.

If route capacity is indivisible, examine results near the point at which an extra vehicle is required.
An apparently profitable order can trigger a full shift and make the next capacity tier unprofitable.
Conversely, an order that fills unused capacity may cost less than the average historical order.
Use these differences in offer design, but do not represent modeled marginal cost as an observed invoice.

Record an owner, decision date, implementation conditions and review date.
At review, compare the actual cohort with the eligibility rules used in the forecast.
If customer mix or labor availability changed, explain the comparison before updating prices.

### 5.12 Build the cohort before calculating rates

1. Extract all orders accepted in the chosen local calendar week.
   Keep the acceptance instant and the order identifier even if the order never reaches dispatch.
2. Apply the eligibility rules written before outcomes were inspected.
   Mark cancellations and customer collection choices explicitly rather than deleting their rows.
3. Join the first legitimate attempt for each order.
   An operational scan must indicate an actual handoff effort under the carrier's definition.
4. Classify the first attempt as delivered, failed or unresolved.
   A record without sufficient evidence belongs in unresolved, not automatically in failed.
5. Link later attempts and final disposition using the original order key.
   Record the elapsed time and observation cutoff so immature cases are visible.
6. Reconcile accepted orders to cancelled, unattempted and attempted orders.
   Reconcile attempted orders to first-attempt delivered, failed and unresolved.
7. Publish counts as well as rates and show how many records remain unresolved.
   The owner of each unresolved queue should be able to inspect source records.

For example, if 400 orders were accepted, 20 cancelled, 10 were unattempted,
and 370 were attempted, the accepted-order bridge is complete.
If 333 first attempts succeeded, 27 failed and 10 are unresolved,
the reported observed success is 333 / 370 = 90.0% of attempted orders.
It is not 333 / 400, which answers a different accepted-order question.
If all ten unresolved records were actually first-attempt successes, the upper bound would be
343 / 370 = 92.7%; if none were, the lower bound remains 90.0%.
These are missing-outcome bounds, not a statistical confidence interval.

### 5.13 Measure the work behind density

Take a route-day as the initial operating unit and retain its individual stops and orders.
Record route start, loading, departure, each arrival and departure, return and paid end.
Identify events added or corrected after the day closes.
When breaks are paid, include them in the paid-hour denominator.
When a driver performs non-route work, attribute that time consistently rather than
assuming every minute on the payroll belongs to delivery.

Separate the paid hours into mutually exclusive components:

| Component | Example | Measurement question |
|---|---|---|
| Preparation | Loading and route briefing | Is the time allocated to this route only? |
| Stem travel | Depot to first stop and last stop to depot | Does the depot location dominate the day? |
| Local travel | Movement between stops in the service territory | Are road restrictions and congestion represented? |
| Service | Parking, access and handoff | Are multiple orders at the same building distinguishable? |
| Waiting | Early arrival before access or window | Is waiting genuinely unavoidable? |
| Paid break and exception | Break, vehicle issue, assistance | Is the category defined to prevent overlap? |

Suppose a route has 30 minutes of preparation, 60 minutes of stem travel,
120 minutes of local travel, 180 minutes of service, 30 minutes of waiting,
and 60 minutes of paid breaks and exceptions.
The total is 480 minutes, or 8 paid hours.
If the route completes 48 orders at 40 stops, productivity is
48 / 8 = 6.0 completed orders per paid hour and 40 / 8 = 5.0 stops per paid hour.
The 1.2 orders per stop is a drop-density measure, not a measure of local travel speed.

If another route completes 18 orders in 3 hours, its productivity is also 6.0 orders per hour.
Together the routes complete 66 orders in 11 hours, or 6.0 per paid hour.
For unequal route rates, use total orders divided by total hours,
not the unweighted mean of the individual rates.
This is important when one unusually short route has a very high reported ratio.

The components support different decisions.
High stem time suggests testing a different depot or departure point.
High local travel suggests territory design or greater geographic concentration.
High service time suggests examining parking, access and the actual handoff process.
High waiting suggests reviewing windows and sequencing, but only after checking
whether early arrivals, break allocation or missing scans produced the apparent waiting.

### 5.14 Check a proposed appointment against the route

Inputs are the committed interval, the site's access interval, the vehicle capacity,
the planned previous stop, a travel-time estimate and the relevant driver shift.
The output is either a feasible insertion with reserved capacity,
an alternative offer, or a rejection before the customer commits.
Use local time for the customer-facing promise and preserve the corresponding UTC instants.

1. Intersect the customer window with access hours.
   If the intersection is empty, no route sequence can satisfy both constraints.
2. Estimate earliest arrival from the prior stop and latest departure for the next commitment.
   Include parking and access, not just road travel.
3. Add the expected service duration and an appropriate uncertainty allowance.
   A point forecast that fits by one minute is not evidence of a robust promise.
4. Check load, required equipment, driver shift and legally required breaks.
   A geographically convenient insertion can still violate a non-spatial constraint.
5. Recalculate downstream arrival times and the return to depot.
   A feasible new appointment may make an existing commitment infeasible.
6. Reserve the relevant capacity atomically in the booking process.
   Concurrent checkout sessions must not both claim the same remaining capacity.
7. Release a reservation on cancellation or expiry and retain an audit trail.
   If the route changes after sale, the original promise remains an obligation.

Suppose the customer accepts 10:00–12:00 and building access is 10:30–11:15.
The usable interval is 10:30–11:15, or 45 minutes.
If the vehicle can arrive by 10:40 and the handoff needs 20 minutes,
the planned completion at 11:00 fits the access interval.
If a downstream commitment requires departure by 10:55, the same insertion is infeasible.
It is not enough that the first address can be reached inside its own window.

### 5.15 Treat changes after dispatch as controlled decisions

At departure, compare the final route against the sold promises.
Record the route version used for dispatch and its expected arrival range.
An ETA is an estimate and may change; it does not replace the contractual interval.
If a disruption makes breach likely, send the approved communication early,
provide an accessible contact channel, and offer options the operator can actually execute.

Dispatch can resequence, transfer a stop, add an authorized shift or negotiate a revised appointment.
Each action has a cost and can affect customers downstream.
Before moving a stop, check capacity, equipment, access and the effects on other commitments.
The customer-facing team should know whether a proposed alternative has been reserved.
A new commitment should be recorded as a new version, with the original retained for audit.

Do not instruct a driver to skip a break, exceed a safe workload or enter an unsafe location
to protect a productivity metric.
Escalate to the operational safety process and record the resulting service exception.
After the route, compare the predicted breach with actual delivery and identify
whether the issue was a forecast error, a preventable planning gap or a genuine disruption.

### 5.16 Price the incremental capacity rather than the average route

Start with the alternative the customer would otherwise have received.
If the default all-day delivery already occurs, most of its cost is not new to the premium.
The incremental route cost is the difference between feasible scenarios,
including any added shift, overtime, vehicle, waiting or lost capacity.
Do not add an allocated depot cost unless the decision makes it avoidable.

Consider a route with 20 uncommitted paid minutes and an insertion requiring 15 minutes.
If the existing shift and vehicle can absorb the work without displacing another order,
the immediate avoidable labor cost may be small even when average cost per order is high.
Now consider an insertion requiring 25 minutes when no authorized overtime is available.
If the only feasible alternative is a USD 240 additional shift,
the next order can trigger a cost step far above its average share.
The precise economic comparison depends on other orders that can use the new shift.

The price decision should therefore evaluate a portfolio of likely demand,
not quote every order as though it independently purchased a whole vehicle.
Model several uptake levels and check the threshold at which another shift is needed.
Retain a downside case for demand that is less geographically or temporally concentrated.
If the service level is a strategic requirement rather than a profit center,
state the subsidy and the owner accepting it.

### 5.17 Reconcile carrier and customer-care costs

Ask the carrier which charges cover the first visit, additional attempts, storage and returns.
Some contracts include a fixed number of visits; others charge for each additional service.
Map each invoice line to a service event and the order or route to which it relates.
Unmatched invoices and unbilled events belong in a reconciliation queue,
not in a default assumption that the expense is zero.

Customer-care time may be recorded at case level rather than order level.
Link cases where possible, classify shared contacts, and avoid assigning the whole case cost
to every parcel associated with the same order.
If call handling uses spare salaried capacity, distinguish reported allocation from avoidable cost.
If higher volume requires additional staffing, include the relevant incremental shift.

For a refund, distinguish product margin reversal, refunded delivery fee and recovery handling.
An order that is refunded and then returned has multiple real events,
but the same product margin must not be subtracted in two ledger lines.
Agree with finance whether amounts include tax and whether the customer later repurchases.
Any assumed retained future value needs its own horizon and evidence;
do not present it as cash already received.

### 5.18 Investigate mix with a common comparison population

Suppose Area A has 90 first-attempt successes among 100 apartment orders
and 99 among 100 house orders: 189 / 200 = 94.5% overall.
Area B has 72 successes among 80 apartment orders and 19 among 20 house orders:
91 / 100 = 91.0% overall.
Within apartments both areas achieve 90%; within houses A achieves 99% and B 95%.
The overall difference combines the house/apartment mix and the within-house difference.

To standardize both to a 50% apartment and 50% house mix,
Area A is 0.5 × 90% + 0.5 × 99% = 94.5%.
Area B is 0.5 × 90% + 0.5 × 95% = 92.5%.
This describes performance under a common hypothetical mix.
It does not establish that moving B orders to A would produce the same outcome:
other differences, small samples and measurement error remain.

Check the attempt and unresolved denominators within each stratum.
Do not publish a small or sensitive cell if doing so exposes individual locations.
If a building type is poorly coded, report the unknown category and investigate it.
Avoid stratifying on variables caused by the intervention when estimating its total effect.
For example, route assignment may be a consequence of the promise rather than a baseline characteristic.

### 5.19 Design the pilot as an operational agreement

Write a one-page decision charter before assignments are made.
It should name the owner, target population, proposed offer, comparator and planned duration.
Specify whether assignment occurs by customer, building, territory or route-day.
The choice must reflect how treatment can affect shared drivers and remaining capacity.

If neighboring customers share a route, an individual offer can alter the service
that standard-offer customers receive on that same route.
A territory or route-day cluster may better contain that interference,
but yields fewer independent units and can reduce statistical precision.
Use multiple comparable clusters where feasible, balance assignment in advance,
and analyze uncertainty at the assignment level rather than treating every order as independent.

The primary outcome can be incremental contribution per assigned eligible order.
Include non-users of the offer in the assigned population and record paid uptake separately.
Guardrails might include breach rate, unresolved orders, paid overtime and customer access.
Define their thresholds, observation cadence, accountable owner and escalation action in advance.
Safety incidents require the organization's immediate safety process, not a wait for statistical significance.

Before launch, inspect the dispatch workflow for capacity reservation and promise display.
Train customer-care and drivers on how to record exceptions without changing the success definition.
Track exclusions, assignment failures and changes to software or carrier contracts.
At the end, report the estimate, uncertainty, guardrail outcomes and operational feasibility together.
Do not expand solely because a point estimate is positive if its plausible range includes a material loss.

## 6. Worked example

A fictional depot expects 1,000 eligible orders per week. Under the existing all-day offer, route cost is USD 8,200 per week, including the cost of initial attempts. First-attempt failure is forecast at 8% and expected incremental recovery costs USD 12 per failure. There is no window premium.

A proposed two-hour product is forecast to serve the same 1,000 orders, with a USD 3 premium on every order. Its route cost is USD 10,100 because sequencing is more constrained. First-attempt failure is forecast at 5%, with the same USD 12 recovery cost. These are forecasts; they are not observed pilot results.

Baseline expected recovery: 1,000 × 0.08 × USD 12 = USD 960 per week.

Proposed expected recovery: 1,000 × 0.05 × USD 12 = USD 600 per week.

Premium revenue: 1,000 × USD 3 = USD 3,000 per week.

Incremental route cost: USD 10,100 − USD 8,200 = USD 1,900 per week.

Incremental recovery saving: USD 960 − USD 600 = USD 360 per week.

Incremental contribution before other changes: USD 3,000 − USD 1,900 + USD 360 = USD 1,460 per week, or USD 1.46 per eligible order.

The break-even premium is (USD 1,900 − USD 360) / 1,000 = USD 1.54 per order. That result assumes every order pays the premium and the demand mix is unchanged. If only 600 orders pay while the full incremental cost remains, premium revenue is USD 1,800 and contribution is USD 260. That latter scenario is an arithmetic sensitivity, not a forecast of actual uptake.

If the two-hour promise instead adds USD 3,500 in route cost, the same premium and recovery saving produce a USD 140 loss. A manager should validate the route-cost response and uptake, not approve the offer solely on the headline premium.

### Decompose the recovery assumption

Suppose the baseline's USD 12 per failure consists of three mutually exclusive outcomes.
Sixty percent receive a second attempt costing USD 15 each.
Twenty-five percent use a pickup point costing USD 8 each.
The remaining 15% are returned at an incremental cost of USD 6.67 each.
The weighted cost is 0.60 × USD 15 + 0.25 × USD 8 + 0.15 × USD 6.67 = USD 12.0005 per failure.
The USD 12 assumption is therefore a rounded approximation for this hypothetical mix.

The proposed window need not preserve this mix.
For example, if 50% receive a second attempt, 40% use pickup and 10% are returned, the corresponding expected cost is USD 11.367 per failure.
At 50 forecast failures, this is USD 568.35 per week, rather than USD 600.
The resulting incremental contribution would be USD 1,491.65 per week, holding the other assumptions fixed.
This difference is a scenario calculation, not evidence that customers will use pickup more often.

### Check the capacity threshold

Assume a temporary driver shift costs an additional USD 280 per week and is not already included in the USD 10,100 route cost.
If the proposed window triggers that shift, the contribution falls from USD 1,460 to USD 1,180 per week.
The break-even premium rises from USD 1.54 to USD 1.82 per order when all 1,000 orders pay.
If the shift cost was already included in the route forecast, adding it again would double count it.

For the 600-paying-order sensitivity, the same extra shift produces a USD 20 weekly loss.
The conclusion can therefore reverse at partial uptake even though the full-uptake scenario remains positive.
Neither calculation estimates how much route cost would actually change with uptake.
A realistic assessment must simulate or observe the capacity response for each uptake scenario.

### Carry the business case through a pilot

The commercial owner needs more than the USD 1,460 weekly forecast.
The forecast assumes all 1,000 orders pay, all costs are incremental,
and the estimated first-attempt improvement actually occurs.
The dispatch owner needs to know whether existing shifts can support the appointments.
The customer-care owner needs a process for customers whose appointments cannot be met.
Finance needs a ledger that distinguishes the premium from recovered costs.

Assume the depot selects eight comparable route territories for an offer trial.
Four are assigned the premium offer and four retain the standard offer.
The following figures are fictional forecasts for planning the trial, not results.
The expected number of eligible orders is 500 per week per arm.
In the offer arm, 300 orders are forecast to purchase a USD 3 premium.
The remaining 200 receive the standard product.
The control arm receives the standard product for all 500 orders.

The following table is a scenario, not an arithmetic scaling of the all-uptake forecast.
The operations team has supplied a separate hypothetical route-cost estimate for this uptake.
All amounts are USD per week and the arms are assumed comparable in order count and mix.

| Ledger item | Standard arm | Offer arm | Offer minus standard |
|---|---:|---:|---:|
| Eligible orders, orders/week | 500 | 500 | 0 |
| Premium orders, orders/week | 0 | 300 | 300 |
| Premium fees, USD/week | 0 | 900 | +900 |
| Initial route cost, USD/week | 4,100 | 4,850 | +750 |
| First-attempt failures, orders/week | 40 | 34 | −6 |
| Recovery cost, USD/week at USD 12/failure | 480 | 408 | −72 |
| Incremental contribution, USD/week | — | — | +222 |

The offer-arm contribution difference is USD 900 − USD 750 + USD 72 = USD 222 per week.
Dividing by all 500 assigned eligible orders gives USD 0.444 per assigned order.
Dividing by the 300 paying orders gives USD 0.74 per paying order,
but that second figure is not the intention-to-treat outcome.
It omits the non-buyers who share the route and may experience a spillover.

The six fewer forecast failures are a net difference across the whole arm.
They do not establish how many premium buyers personally avoided a failure.
The operator must follow both paying and non-paying customers
and check whether the offer caused any standard-order service deterioration.
Any difference in base merchandise margin or demand is excluded in this fixed-volume scenario.

Suppose the offer arm actually needs a new USD 280 weekly shift
that was not included in its USD 4,850 route estimate.
The scenario contribution becomes USD 222 − USD 280 = negative USD 58 per week.
At 300 premium orders, the fee required to break even is
(USD 750 + USD 280 − USD 72) / 300 = USD 3.19 per paying order, rounded up to cents.
This is a conditional scenario: changing the fee could change uptake and route cost.

Now suppose the failure reduction is zero rather than six orders per week,
while the original USD 750 incremental route cost remains.
The contribution is USD 900 − USD 750 = USD 150 per week before the extra shift.
That case shows that a favorable failure forecast is helpful but not the sole driver.
With the additional USD 280 shift, however, the same case loses USD 130 per week.
The two sensitivities identify operational capacity as a decision-critical input.

### Decide what the pilot must reveal

Before launch, the teams agree how a route cost will be measured.
They identify paid driver time, vehicle charges and any incremental shifts
and agree which resources the depot can avoid over the decision horizon.
They retain a separate fully allocated accounting view for financial reconciliation.
The two views can coexist as long as neither is mislabeled.

They also specify the first-attempt outcome in the accepted-order cohort.
Cancelled and unattempted orders are reported separately.
If unresolved scans are more common in one arm, the reported rate receives a missingness analysis.
Recovery cases are observed long enough to capture their final disposition,
or else unresolved cases are displayed by age at the reporting date.

The decision meeting should ask five questions:

1. Did the offer change contribution per assigned eligible order, and with what uncertainty?
2. Did its premium uptake and route-cost response resemble the planning scenario?
3. Did the offer change on-time service, first-attempt success or eventual completion?
4. Did any area, building type or accessibility group experience worse outcomes?
5. Could dispatch execute the product without unsafe work or recurring promise overrides?

An affirmative commercial answer does not override an unacceptable safety outcome.
A negative point estimate does not prove a product can never work in another territory.
The decision is local to the studied season, offer, network and observed demand.
Record whether to continue, revise or stop, the person accountable and the next review date.

### Trace one order across the ledger

Order A is accepted with a 14:00–16:00 commitment and a USD 3 premium.
The driver arrives at 15:10, cannot access the building and records a legitimate first attempt.
The initial visit is included in the route cost; it is not an extra USD 12 charge.
Customer care confirms access instructions, and the order is delivered the following day.

For this order, first-attempt success is zero and eventual completion is one.
The first attempted arrival was within the window,
but whether the promise was met depends on whether the contract requires actual delivery
rather than attempted arrival; the metric specification must state that distinction.
The next-day retry and contact are incremental recovery actions if they would not otherwise occur.
They must be valued once, at their actual relevant cost, rather than at both the average pathway
cost and the individual invoice cost in the same realized-order ledger.

Order B is accepted for the same window and delivered at 16:20 on its first attempt.
It is a first-attempt success but a late completion if the commitment requires delivery by 16:00.
These two examples show why a single success percentage cannot summarize customer experience.
Review the order-level evidence and the definitions before attributing either outcome to a driver.

## 7. Code, query, or calculation

This Python 3 calculation uses the fixed-cohort assumptions above. It is deliberately deterministic: it does not infer a probability distribution or causal effect.

```python
orders = 1000
baseline_route = 8200.0
proposed_route = 10100.0
baseline_failure = 0.08
proposed_failure = 0.05
recovery_per_failure = 12.0
premium = 3.0

baseline_recovery = orders * baseline_failure * recovery_per_failure
proposed_recovery = orders * proposed_failure * recovery_per_failure
increment = (orders * premium - (proposed_route - baseline_route)
             + baseline_recovery - proposed_recovery)
break_even = ((proposed_route - baseline_route)
              - (baseline_recovery - proposed_recovery)) / orders
print(f"Incremental contribution: ${increment:,.2f}/week")
print(f"Break-even premium: ${break_even:.2f}/order")
```

Expected output:

```text
Incremental contribution: $1,460.00/week
Break-even premium: $1.54/order
```

A production calculation should include cohort reconciliation, cancellation, customer-care cost, uncertainty and the period over which each cost is avoidable. Do not extrapolate the result to another service area without revisiting the inputs.

### A reproducible cohort bridge

The next calculation starts from a hypothetical accepted cohort rather than the forecast.
It treats unresolved first-attempt status as unresolved, not as a successful delivery.
Each order is represented once; counts are already reconciled to the order grain.
The bounds assume unresolved records could all be failures or all be successes.
They do not measure sampling uncertainty or correct an inaccurate attempt classification.

```python
accepted = 1000
cancelled = 40
not_attempted = 30
first_success = 837
first_failure = 73
unresolved = 20

attempted = first_success + first_failure + unresolved
assert accepted == cancelled + not_attempted + attempted
observed = first_success / attempted
upper = (first_success + unresolved) / attempted
print(f"Attempted: {attempted} orders")
print(f"Observed success: {observed:.1%} of attempted")
print(f"Missing-status bounds: {observed:.1%} to {upper:.1%}")
print(f"Unattempted: {not_attempted} orders")
```

Expected output:

```text
Attempted: 930 orders
Observed success: 90.0% of attempted
Missing-status bounds: 90.0% to 92.2%
Unattempted: 30 orders
```

This bridge does not say that 90% of all accepted orders were delivered.
The 40 cancelled orders and 30 unattempted orders require separate business definitions.
The first-attempt result also says nothing about the time of completion.
Add a promise-compliance measure only after defining whether arrival or handoff is required.

### Translate a sensitivity table into a decision

Use the fixed 1,000-order scenario from section 6.
The following changes are one-at-a-time unless the row explicitly combines them.
The baseline is the USD 1,460 contribution forecast, not a measured result.

| Scenario | Calculation, USD/week | Contribution, USD/week |
|---|---|---:|
| Original forecast | 3,000 − 1,900 + 360 | 1,460 |
| Route cost increases by 500 | 3,000 − 2,400 + 360 | 960 |
| No recovery improvement | 3,000 − 1,900 + 0 | 1,100 |
| Only 600 premium payments, costs fixed | 1,800 − 1,900 + 360 | 260 |
| 600 payments plus extra shift | 1,800 − 1,900 + 360 − 280 | −20 |

The rows are not probability-weighted outcomes.
They isolate assumptions that require evidence before scaling the product.
The partial-payment rows deliberately hold costs fixed to show arithmetic exposure;
in reality the capacity and failure effects could move when uptake changes.
Run a separate route scenario for each plausible demand pattern.

### Calculate the effect of route-hour weighting

Suppose Route A completes 8 orders in 1 paid hour and Route B completes
36 orders in 9 paid hours.
Their route-level rates are 8 and 4 orders per paid hour, respectively.
The unweighted mean is (8 + 4) / 2 = 6 orders per paid hour.
The aggregate rate is (8 + 36) / (1 + 9) = 4.4 orders per paid hour.
The latter answers how many orders the ten paid hours produced.
The former answers the average of two route-level ratios and is valid only for that question.

If Route A's one-hour shift is actually a partial route,
its apparent productivity may exclude loading or the return to depot.
Reconcile the paid-hour boundary before using either number for staffing.
Show the distribution as well as the overall rate to identify difficult routes
without allowing the smallest route to dominate the aggregate.

## 8. Data, standards, and regulatory context

The minimum dataset joins order and promise records to route, scan, proof-of-delivery and recovery records. Store a stable order key, parcel keys, route identifier, location identifier, promised start and end, time zone, event timestamps, failure reason, attempt sequence and eventual outcome. Link costs at their native grain and prevent a route-level cost from being duplicated across multiple parcel rows.

GS1's EPCIS standard provides a model for sharing supply-chain event information: what, when, where, why and how. It does not certify that a recorded event is true or decide which party pays for a failed delivery. Preserve source and corrections alongside standardized fields. See [GS1 EPCIS](https://www.gs1.org/standards/epcis).

Customer address, contact details and detailed location traces can be personal data. Apply the applicable privacy rules for the jurisdiction, limit access and retention, and involve counsel where necessary. The [EU GDPR text](https://eur-lex.europa.eu/eli/reg/2016/679/oj) is an example of a binding regional framework, not a universal retention schedule. Delivery promises and consumer remedies also depend on applicable law and contract; this teaching material is not legal advice.

Use local time and an unambiguous UTC instant together. Daylight-saving transitions can make a local time ambiguous or nonexistent. A geocode is an estimate of a location, not evidence that the driver reached the correct entrance.

### Minimum data contract

Each source has a natural grain and an accountable owner.
Join at a declared grain, and aggregate before joining when the relationship is one-to-many.
The following fields are a starting contract for an operator, not a mandatory standard schema.

| Dataset | Grain and key | Required fields | Primary use |
|---|---|---|---|
| Order | One row per order ID | Acceptance time, customer-facing service, cancellation | Cohort and eligibility |
| Promise history | One row per order and version | Window start/end, zone, accepted and superseded times | Original and revised commitments |
| Parcel | One row per parcel ID | Order ID, size, handling class | Physical load and split shipments |
| Route assignment | One row per order and assignment version | Route ID, planned sequence, assigned time | Planned versus actual work |
| Route time | One row per route or paid time segment | Driver shift, departure, return, paid minutes | Productivity and avoidable capacity |
| Attempt event | One row per legitimate order attempt | Attempt ID, sequence, time, outcome, source | First-attempt outcome |
| Proof | One row per proof artifact or reference | Order ID, event time, evidence type | Completion reconciliation |
| Recovery | One row per action | Order ID, action type, event and resolution time | Retry, collection and return pathway |
| Cost | One row per charge or resource segment | Amount, currency, period, source, allocation basis | Contribution ledger |
| Contact | One row per service case | Case ID, linked orders, duration or cost | Customer-care burden |

The relationship between orders and parcels is not necessarily one-to-one.
A split order can have multiple parcel events and multiple partial handoffs.
Define whether the order is complete only when all required parcels are received.
If one parcel fails and another succeeds, a parcel success rate is not an order success rate.
If a stop serves several orders, the visit must not become several paid route hours.

Promise history must preserve the accepted original interval and each later revision.
Otherwise a late delivery can appear on time after an ETA overwrites the original promise.
Store event time and ingestion time separately so late-arriving data can be recognized.
Keep a source identifier and correction lineage when a carrier amends an earlier event.
An analyst should be able to reproduce the report as known at a past cutoff.

### Event checks and exception queues

At ingestion, validate that identifiers exist and timestamps have an interpretable time zone.
Check for duplicate event IDs and attempts with the same order, time and source.
Flag a completion recorded before acceptance or a retry recorded before the first attempt.
These anomalies need investigation; automatic deletion may erase a legitimate correction.

Compare counts at each handoff:

1. Accepted orders in the commercial system.
2. Orders eligible for delivery after explicitly classified cancellations.
3. Orders assigned to dispatch.
4. Orders with a legitimate first attempt.
5. Orders with recorded first-attempt outcome.
6. Orders with final disposition at the maturity cutoff.

Counts need not match at every stage because orders move through different states.
The bridge must explain each difference and show unresolved cases by age.
An attempted-but-unresolved order is different from an order never attempted.
A cancellation after dispatch also differs from one cancelled before work begins.
Record such timing so the economic ledger does not discard incurred cost.

An exception queue needs an owner, priority, source links and resolution codes.
A missing scan might be corrected from verified evidence or remain unknown.
Corrections should retain the prior value and who authorized the change.
If a carrier feed is incomplete for a whole day, quantify the affected cohort
before comparing that day with a complete day.

### Time and location controls

Store the promised interval as both local wall-clock values with a named time zone
and resolved UTC instants when a unique instant is available.
During a daylight-saving transition, some local times repeat and others do not exist.
Resolve the ambiguity at booking using the time-zone rules and preserve the selected offset.
Do not silently interpret every local timestamp with the depot server's time zone.

The location record should distinguish the postal address, geocode,
building entrance, loading restrictions and customer-supplied access instructions.
Two deliveries with the same coordinates can require different entrances or service methods.
Access information can be sensitive, and its operational usefulness does not justify unlimited retention.
Restrict access to staff and systems with a legitimate need and review retention with the relevant owner.

### Standardization and legal boundaries

EPCIS supports interoperable descriptions of supply-chain events;
using it does not by itself supply every contractual or analytical field in this lesson.
The parties still need an agreed mapping for attempt, handoff, cancellation and correction.
They also need contractual responsibility for event quality and exception handling.
A standardized event can be incomplete, late or wrong.

The EU GDPR is an example of a legal framework that can apply to personal data.
Its application depends on the facts, including the data and parties involved.
This lesson does not prescribe a lawful basis, universal deletion deadline or consent workflow.
Ask privacy counsel and the data owner to determine obligations for the actual operation.
Apply purpose limitation, access controls and retention decisions through that governance process.

Similarly, consumer delivery obligations are jurisdiction- and contract-specific.
The operational metric should record the actual promise and evidence,
while the legal interpretation of a breach belongs to the appropriate advisers.
Do not treat a model's estimate of arrival as a legal amendment to a customer agreement.

## 9. Industry practice and case evidence

UPS describes its ORION routing system as using route optimization to help drivers determine delivery sequences; see [UPS on ORION](https://about.ups.com/us/en/our-company/innovation-driven/ground-fleet.html). This is a company description, not independent evidence that the fictional economics above will occur. Published company descriptions do not establish a controlled causal estimate for any particular depot.

An alternative to repeated home attempts is an access-point or collection network. UPS describes its [Access Point network](https://www.ups.com/us/en/ship/ups-access-point.page). The documented existence of the service does not establish customer uptake, incremental cost or accessibility for a specific customer population. [INFERENCE] Such a network can change recovery economics only if customers can reach the collection site and the additional handling and transport costs are counted.

In a local study, obtain the operator's definition of failure, trial population, missing scans and the period over which final outcomes mature. Do not transfer a company-wide operational claim into a guaranteed route-level saving.

### Reading the ORION example as a business case

The cited UPS material describes an operational system for route sequencing.
The source is useful evidence that a large operator uses routing technology in its ground fleet.
It is not a randomized comparison for the fictional depot in section 6.
It does not supply that depot's cost per stop, failure rate or window premium.
Those values remain local assumptions until measured.

To assess a comparable deployment, ask what the optimized sequence actually covers.
Does the route plan include service durations, access hours and vehicle constraints?
Are drivers permitted to make operationally necessary adjustments?
Are actual arrival and departure events available for comparison with the plan?
Can a customer promise be protected when a route is resequenced?

Compare actual paid hours and outcomes against an appropriately constructed alternative.
A before/after report can be affected by seasonality, changes in volume and geography,
new drivers, roadworks or a different service mix.
If assignment is feasible, a staged rollout with comparable units can improve attribution.
Where it is not, disclose the design and alternative explanations.

The implementation cost belongs in the decision as well.
Software charges, mapping data, integration, planner time and training may change the result.
So may the value of freed capacity if it can be used for additional work.
Do not count freed paid minutes as cash savings unless staffing or another cost actually changes.

### Reading the access-point example

The cited UPS page documents the existence of a collection network.
It does not establish that every customer has an equivalent location nearby
or that every product and parcel can be handled at every site.
It also does not establish the effect of the network on a particular retailer's sales or costs.

[INFERENCE] A useful local evaluation compares the whole recovery pathway.
For home retry, measure the cost of contact, added route work and subsequent outcome.
For collection, measure the extra transport leg, site handling and any applicable fees.
Separately describe customer travel time, opening hours, mobility constraints and alternatives.
Do not convert the customer's burden into an operator saving and then omit it from the decision.

An access-point trial should record whether the customer was offered the option,
whether it was chosen, whether the parcel was successfully collected and when.
Orders not collected may incur storage and return costs.
Customers who cannot use the option should have a feasible alternative,
and their outcomes should remain in the assigned cohort.

If opt-in customers are more likely to be available or live near the collection site,
their experience cannot be generalized to everyone from uptake alone.
Compare assigned offers where possible and show outcomes for non-users.
The commercial conclusion depends on adoption, service, accessibility and total cost,
not merely the count of parcels sent to collection.

### Questions to ask a prospective operator

Ask for the definition of a delivery attempt and whether an inaccessible address qualifies.
Ask for the completion evidence and what happens when a driver device is offline.
Ask how original promises and revised ETAs are stored.
Ask which fees apply to retries, collection, storage and returns.
Ask how subcontracted routes and late invoices are incorporated.

Request an anonymized cohort bridge and an exception-rate report, not only an average.
Request results by building type or territory where the population permits safe reporting.
Ask for the maturity horizon and the share of final dispositions still unknown.
Ask whether the reported results come from a controlled trial or an operational trend.
These questions make a vendor's claims comparable without inventing a benchmark.

## 10. Failure modes and red flags

| Failure | Warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Denominator drift | Success improves after exclusions | Hard orders removed | False improvement | Reconcile accepted cohort |
| Duplicate recovery | Cost rises with parcel count | Order cost joined to parcels | Overstated loss | Aggregate at native grain |
| Missed recovery | Cost ends at first attempt | Retry omitted | Underpriced service | Follow mature cohort |
| Wrong window | ETA stored as commitment | Forecast changes | Unmanaged breach | Separate fields |
| Average route | Mean hides extremes | Unequal route hours | Wrong staffing | Ratio of totals and distribution |
| Infeasible route | Breaks or access missing | Plan omits constraints | Late deliveries | Validate with dispatch |
| Selection bias | Premium buyers differ | Choice is not random | Wrong causal estimate | Randomize offer |
| Missing scan | No outcome recorded | Device or integration failure | Biased success | Reconcile and investigate |
| Double-counting refund | Refund in two cost lines | Overlapping categories | Inflated benefit | Define cost ledger |
| Unequal service | One area has more exclusions | Access or data gap | Harm hidden by aggregate | Segment and investigate |
| False density | Stops counted by map area | Travel barriers ignored | Poor territory design | Use network travel and time |
| Immature cohort | Recent orders unresolved | Retry takes time | Understated recovery | Fix maturity window |

### Diagnose the red flags rather than merely list them

When success improves immediately after a reporting-system change,
first compare accepted, attempted, unresolved and excluded counts before and after the change.
If the improvement comes from a smaller denominator, restore the common cohort
and quantify the effect of the missing records.
Only then investigate whether an operational improvement also occurred.

When route cost falls but paid overtime rises, inspect the accounting boundary.
An outsourced route may have moved cost from one invoice line to another.
Some overtime may be attributed to depot work rather than the route that caused it.
Bring payroll, dispatch and carrier records together before concluding that cost fell.

When customer-care contacts rise but the failure code remains unchanged,
sample case records against promised windows and delivery events.
The failure-code menu may not include late completion or misleading communications.
Do not redefine the code merely to make the dashboard agree;
document the mapping and preserve the historical series.

When a premium service fills quickly in one neighborhood but not another,
examine actual availability, price and access to the booking channel.
Low uptake can reflect a missing feasible appointment rather than lack of interest.
Report the number offered the product, the number with a feasible slot and the number buying.
Avoid interpreting the final percentage without the preceding funnel.

When a collection solution reduces route miles but returns increase,
follow the order beyond transfer to the pickup point.
The transfer is an intermediate event, not customer completion.
Measure collection, expiry and return as well as handling and customer contacts.
If the final outcome worsens for a group, review accessibility before expanding.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test |
|---|---|---|---|
| Stable order count | Fixed-cohort comparison | Price changes demand | Randomized offer |
| Recovery cost constant | Simple calculation | Retry mix changes | Cost by pathway |
| Route costs incremental | Contribution decision | Fixed costs allocated as variable | Finance review |
| Failure rates forecast accurately | Expected cost | Season or mix shifts | Holdout and pilot |
| All orders pay premium | Revenue calculation | Partial uptake | Track paid acceptance |
| Promise feasible | Customer service | Traffic and workload variance | Dispatch simulation and pilot |
| Outcomes mature | Complete accounting | Long return cycles | Age unresolved cases |
| No spillover | Simple experiment | Shared route capacity | Cluster assignment |
| Events represent reality | Measurement | Missing or false scans | Audit proof and exceptions |
| Costs in one currency | Comparable arithmetic | Currency or inflation shifts | Normalize period and FX |

### When the forecast should be withheld

The contribution estimate is not decision-ready if route charges cannot be separated
from a one-time invoice correction or if a major carrier feed is missing.
Publish the known components and an explicit unresolved range instead of substituting zero.
If the unknown cost could reverse the sign, name the owner and date for resolving it.

The fixed-volume model is also incomplete if the premium changes conversion.
Additional orders can bring merchandise margin and consume capacity;
lost orders remove both margin and their associated delivery work.
Their contribution must be calculated relative to a credible no-offer counterfactual.
Premium uptake by existing customers alone does not identify this demand effect.

The constant recovery-cost model is incomplete if an offer changes the pathway mix.
For example, a successful appointment may avert a retry,
while the remaining failures may be disproportionately difficult and expensive.
Estimate costs conditional on the observed path and follow each order to disposition.
The same first-attempt improvement can therefore have different economic value in two areas.

Finally, a route simulation can miss practical access constraints and driver knowledge.
Test inputs with dispatch and drivers, compare forecasts with actual execution,
and retain a plan for manual escalation.
If uncertainty in critical inputs is too wide for the decision,
a limited, guarded pilot can be more informative than a precise-looking forecast.

## 12. Exercises

1. **Define the cohort** (10 minutes). Write first-attempt success for 900 successful first attempts out of 1,000 attempted orders, and list two exclusions. Deliver the formula and exclusions; acceptance requires an explicit denominator.
2. **Separate units** (10 minutes). A route serves 60 orders at 45 stops in 9 paid hours. Compute orders per hour and stops per hour. Acceptance requires both units.
3. **Recovery** (15 minutes). Of 100 failed orders, 70 receive a USD 10 retry and 30 a USD 4 collection handling action. Compute average incremental recovery cost per failure. Acceptance requires weighting the mutually exclusive paths.
4. **Window economics** (15 minutes). Use the worked example but increase proposed route cost by USD 500 per week. Compute incremental contribution. Acceptance requires showing the delta.
5. **Critique** (15 minutes). An operator reports a 99% success rate after excluding orders with missing scans. Identify the bias and a reconciliation action. Acceptance requires addressing the excluded population.
6. **Design** (25 minutes). Design a two-area offer trial with shared drivers. Specify assignment, primary metric, service guardrail and spillover response. Acceptance requires a credible causal comparison and protection for customers.
7. **Equity** (15 minutes). An aggregate success rate rises while an apartment-heavy area declines. Give two plausible mechanisms and a safe investigation. Acceptance requires distinguishing access from customer blame.

8. **Capacity discontinuity** (20 minutes).
   A premium offer brings USD 1,800 per week in additional fees and saves USD 360 per week in recovery.
   It adds USD 1,900 per week in route costs before an additional USD 280 driver shift.
   Deliver the contribution with and without that shift.
   Acceptance requires identifying whether the shift is already included and interpreting the sign in both cases.

9. **Cohort reconciliation** (20 minutes).
   A depot accepts 1,000 orders; 40 are cancelled before dispatch, 30 are not attempted, and 930 are attempted.
   Of attempted orders, 837 succeed on the first attempt, 73 fail and 20 have unresolved scan status.
   Deliver a reconciliation and the first-attempt success rate with the denominator stated.
   Acceptance requires retaining unresolved orders in the attempted cohort and separately reporting the other categories.

10. **Service redesign** (30 minutes).
    A retailer is considering consolidating apartment deliveries at a shared pickup point.
    Drivers report repeated access delays, but no accessibility assessment has been completed.
    Deliver a decision memo identifying the customer groups affected, costs and benefits, evidence required, alternatives and a stop rule.
    Acceptance requires an accessible alternative and a plan to measure outcomes for customers who cannot use the pickup point.

### Submission instructions and extensions

For exercise 1, state whether the denominator includes unresolved first-attempt records.
Explain why a cancellation before dispatch is not itself an attempted handoff.
Name an owner or source that could reconcile the two exclusions to accepted orders.
Do not relabel 90% of attempted orders as 90% of all accepted orders.

For exercise 2, explain what additional information is required to turn the two ratios
into an estimate of avoidable cost.
Your answer should mention paid time, the capacity threshold and the decision horizon.
The ratios alone do not establish that reducing one stop saves an hour of pay.

For exercise 3, state whether the retry amount includes the original failed visit.
The exercise assumes that it does not and that the paths are mutually exclusive.
Explain how your ledger would change if a customer first called support
and then received a retry: the support cost is an additional action, not a competing path.

For exercise 4, label the USD 500 as an incremental change to the proposed route cost.
State the resulting weekly amount and whether the answer is a forecast or an observed outcome.
Explain why the baseline recovery estimate does not change in this sensitivity.

For exercise 5, sketch a bridge from accepted orders to attempted and unresolved orders.
Identify at least one source other than the delivery dashboard that can corroborate an outcome.
Describe a plausible direction of bias without asserting that every missing scan is a failure.

For exercise 6, specify the randomization unit and why shared capacity matters.
The scenario supplies only two areas, so explain the limitation of having only two clusters.
Consider repeated randomized time blocks if feasible and if carryover can be controlled;
otherwise state that the two-cluster comparison cannot support precise general inference.
Include a safety escalation that operates independently of the experiment's final analysis.

For exercise 7, distinguish a hypothesis about building access from a verified cause.
Name the disaggregation variables and an approach to protect small or identifying cells.
Propose one operational check with building management or dispatch where authorized.

For exercise 8, include the extra shift in the incremental-cost ledger exactly once.
Explain how the decision changes if the shift would occur regardless of the offer.
That alternative is a different counterfactual, not a reason to ignore an actually triggered shift.

For exercise 9, compute a lower and upper bound for first-attempt success
if every unresolved record is respectively a failure or a success.
Keep the 30 unattempted orders outside that attempted-order denominator,
but show them in the accepted-order reconciliation.

For exercise 10, distinguish an optional collection offer from compulsory redirection.
Specify a customer communication, a home-delivery alternative and an owner for access complaints.
Include site operating hours, transport burden, handling costs and uncollected returns.
Set a measurable threshold and cadence for the stop rule, with a named decision maker.

## 13. Self-check answers

1. 900 / 1,000 = 90% of attempted orders. Examples of separately reported categories are cancelled orders and orders never attempted; do not silently omit them from an accepted-order metric.
2. 60 / 9 = 6.67 orders per paid hour; 45 / 9 = 5 stops per paid hour. These measure different units.
3. (70 × USD 10 + 30 × USD 4) / 100 = USD 8.20 per failure. This assumes every failure follows exactly one of the stated pathways.
4. USD 1,460 − USD 500 = USD 960 per week, assuming all other inputs unchanged.
5. Missing scans can include failures, and excluding them can inflate success. Reconcile all accepted orders against dispatch, proof and final disposition; report unresolved counts.
6. A strong design randomizes comparable territories or time blocks, measures contribution per assigned eligible order and promise breaches, records cross-area interference, and has a stop rule. It does not treat self-selection as random assignment.
7. Possible mechanisms include inaccessible entrances and windows incompatible with building access. Audit access instructions and scan completeness with local operations; do not infer a customer characteristic from the aggregate.
8. Without the extra shift, USD 1,800 + USD 360 − USD 1,900 = USD 260 per week.
   If the shift is additional, the contribution becomes USD 260 − USD 280 = negative USD 20 per week.
   If it is already included in route cost, do not subtract it again.
9. The accepted cohort reconciles as 40 cancelled + 30 not attempted + 930 attempted = 1,000 orders.
   Attempted orders reconcile as 837 successes + 73 failures + 20 unresolved = 930 orders.
   The observed first-attempt successes are 837 / 930 = 90% of attempted orders.
   If all 20 unresolved scans later prove successful, the upper bound is 857 / 930, approximately 92.2%.
   These bounds describe uncertainty from missing outcomes, not statistical confidence intervals.
10. A strong memo quantifies driver time, pickup-site cost and customer travel burden separately.
    It tests the proposed site for physical access, operating hours and alternatives.
    It specifies a comparison population, monitoring period and a stop rule for unacceptable access or service outcomes.
    A weak memo assumes all residents can collect parcels and treats lower route cost as complete evidence of benefit.


### Expanded reasoning and rubrics

**Exercise 1.** The numerator is 900 orders with successful first handoffs.
The denominator is 1,000 orders with a legitimate first attempt.
The remaining 100 attempted orders are not automatically all confirmed failures;
the prompt does not specify whether unresolved outcomes are included.
Report cancelled and unattempted orders separately, with their counts when available.

**Exercise 2.** The order rate measures customer orders completed per paid hour.
The stop rate measures location visits per paid hour.
There are 60 / 45 = 1.33 orders per stop on this route.
That ratio does not tell you whether each order required a separate handoff.
To estimate cost, determine whether changing the work changes a paid shift or vehicle.

**Exercise 3.** The retry path costs 70 × USD 10 = USD 700.
The collection path costs 30 × USD 4 = USD 120.
Together they cost USD 820 for 100 failed orders, or USD 8.20 per failure.
If a support contact costs another USD 2 on every failed order,
and the original path costs exclude it, the total becomes USD 10.20 per failure.
If support is already included in those amounts, adding it would double count.

**Exercise 4.** The original proposed route cost is USD 10,100 per week.
Increasing it by USD 500 gives USD 10,600 per week.
The incremental route cost against USD 8,200 is therefore USD 2,400 per week.
Premium revenue of USD 3,000 and recovery savings of USD 360 remain unchanged.
The result is USD 3,000 − USD 2,400 + USD 360 = USD 960 per week.

**Exercise 5.** A 99% rate on known scans is a conditional statistic.
Without the number and outcome distribution of excluded scans,
the rate for all attempted orders is not known.
Join the accepted ledger, dispatch feed and proof records and report unmatched cases.
Calculate bounds where appropriate; do not claim missing outcomes are known failures.
Investigate whether missingness differs by carrier, area, device or route.

**Exercise 6 rubric.** A good answer protects the comparison from shared-route interference.
It states the eligible population before assignment and keeps non-buyers in the primary outcome.
It explains that two territories alone offer very limited independent replication.
It includes capacity, breach and customer-access guardrails with an escalation owner.
It reports uncertainty at the assignment unit and acknowledges seasonal transfer limits.
A weak answer compares premium buyers with non-buyers or changes the plan after seeing results.

**Exercise 7 rubric.** A good answer presents access and time-window mismatch as hypotheses.
It checks data completeness before interpreting area-level differences.
It works with operations and affected customers without exposing identifiable address data.
It tests an accessible remedy and retains a comparable service option.
A weak answer assigns blame to residents or suppresses the subgroup result.

**Exercise 8.** The baseline calculation is positive USD 260 per week.
When the USD 280 shift is caused by the offer and excluded from the stated route cost,
the result is negative USD 20 per week.
If the shift is already included in USD 1,900, the result remains USD 260.
If the shift would occur under both alternatives, it is not incremental to this decision.

**Exercise 9.** The lower bound is 837 / 930 = 90.0% of attempted orders.
The upper bound is (837 + 20) / 930 = 92.2%, rounded to one decimal place.
The 73 documented failures remain in the denominator under either bound.
The 30 unattempted and 40 cancelled orders remain visible in the accepted-order bridge.
More time or source reconciliation may narrow the missing-status range.

**Exercise 10 rubric.** A complete memo names all affected customers and the operational owner.
It includes carrier and site fees, retry and return costs, customer travel and access constraints.
It compares optional pickup with an accessible home-delivery alternative.
It monitors collection, final disposition, contacts and subgroup outcomes after assignment.
An illustrative stop rule might pause the change if the pre-agreed accessibility complaint
threshold is crossed, with the service owner reviewing each case promptly.
The threshold itself must be selected for the actual operation, not inferred from this lesson.

## 14. Interview questions

1. What is a stop? Strong answers distinguish a location visit from an order; weak answers assume one parcel per visit.
2. How does a narrow window cost money? Strong answers discuss sequencing and idle time; weak answers assume distance alone.
3. How do you price failure? Strong answers include conditional recovery pathways; weak answers count only the first trip.
4. How do you evaluate a premium? Strong answers compare incremental contribution and demand; weak answers compare revenue alone.
5. Why mature a cohort? Strong answers include delayed retry and returns; weak answers call missing outcomes successes.
6. How do you compare areas? Strong answers address mix and data quality; weak answers rank pooled rates without context.
7. What is a promise? Strong answers distinguish it from an ETA; weak answers treat an updated forecast as permission to change the commitment.
8. How do you run a pilot? Strong answers specify assignment, guardrails and spillovers; weak answers rely on an uncontrolled before/after.

### Probes for the interviewer

For question 1, ask how a single apartment building visit with twelve orders
should appear in an order-level report and a route-level report.
Listen for one stop, twelve orders and a separate measure of actual handoff work.

For question 2, ask for a case where a narrow window adds no extra vehicle.
A strong answer uses existing spare capacity and checks downstream constraints.
Then ask when the same window crosses a shift threshold and changes the economics.

For question 3, ask whether a customer contact and a second attempt are mutually exclusive.
They can both occur, so both may be real costs if neither is already embedded in the other.
Ask how the candidate would value a retry that shares capacity with another route.

For question 4, ask whether 60% premium uptake means 60% demand growth.
It does not: many buyers may have placed the order without the premium.
Look for a counterfactual and an intention-to-treat comparison.

For question 5, ask how to report orders whose final outcome is still unknown.
A strong answer retains an unresolved category, reports age and uses a consistent cutoff.
The candidate should not call unresolved orders either successes or failures without evidence.

For question 6, ask how a pooled average can change when within-building rates do not.
Look for a mix-shift explanation and a common-mix calculation.
Ask how small cells and unequal missingness affect the interpretation.

For question 7, present a 14:00–16:00 promise and a revised 16:20 ETA.
The revised estimate can inform communication but does not erase the original commitment.
The candidate should preserve promise versions and explain the escalation process.

For question 8, ask what happens to control customers sharing treated vehicles.
A strong answer recognizes interference and considers cluster or time-block assignment.
They also discuss the loss of precision when there are few independent clusters.

## 15. Key terms

- **Attempt:** documented handoff effort; see section 4.
- **Cohort:** common-start population; see section 5.
- **Completion:** successful order handoff; see section 3.
- **Contribution:** revenue less defined incremental costs; see section 6.
- **Counterfactual:** outcome under the alternative; see section 5.
- **Density:** proximity or concentration of delivery work; see section 3.
- **Dwell:** time at a stop; see section 4.
- **First-attempt success:** first-attempt completions divided by eligible attempts; see section 5.
- **Guardrail:** threshold protecting service or safety; see section 5.
- **Handoff:** transfer of the order to its recipient; see section 3.
- **Incremental cost:** cost changed by the decision; see section 5.
- **Maturity:** enough elapsed time to observe resolution; see section 5.
- **Order:** commercial demand unit; see section 4.
- **Parcel:** physical package; see section 4.
- **Promise:** customer commitment; see section 3.
- **Recovery:** post-failure actions; see section 5.
- **Route hour:** paid time assigned to a route; see section 5.
- **Service window:** promised time interval; see section 4.
- **Spillover:** effect on units outside their assigned intervention; see section 5.
- **Stop:** service location visit; see section 4.

## 16. Sources

- GS1, [EPCIS standard](https://www.gs1.org/standards/epcis), event-sharing standard and documentation.
- European Union, [Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj), official legal text; applicability depends on circumstances.
- UPS, [Ground fleet and ORION](https://about.ups.com/us/en/our-company/innovation-driven/ground-fleet.html), company description, not an independently evaluated local result.
- UPS, [UPS Access Point](https://www.ups.com/us/en/ship/ups-access-point.page), company service description.

## 17. Links

- [Module index](_index.md)
- [Freight modes](01_freight-modes.md)
- [Rate structures](02_rate-structures.md)
- [Lane and tender analytics](03_lane-tender-analytics.md)
- [Module survey](../../../course/12_transportation_routing_last_mile.md)
- [Simulation and optimization labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md)
