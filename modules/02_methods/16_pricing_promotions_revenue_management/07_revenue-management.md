# Revenue Management and Littlewood's Rule

> Module 16 — Pricing, Promotions, and Revenue Management  
> Topic 07 of 10  
> Estimated study time: 4 hours  
> Prerequisites: demand forecasting, probability distributions, and contribution margin  
> Version 2026.5

## 1. Why this topic exists

A refrigerated carrier has five available pallet positions on a particular departure.
Once the truck leaves, an empty position cannot be carried forward to tomorrow.
A customer offers to reserve three positions early at a low contracted rate.
The carrier expects urgent, higher-paying requests closer to departure.
Accepting the early order produces certain contribution but may displace later demand.
Rejecting it preserves capacity but can leave a truck partially empty.
The commercial decision is not simply whether the low price exceeds variable cost.
It is whether selling now is worth more than the expected contribution from keeping the position available.

Revenue management makes this trade-off explicit.
It coordinates demand forecasts, customer access, booking controls, and the physical capacity that can actually be delivered.
It is relevant to freight departures, warehouse appointment windows, hotel rooms, airline seats, production slots, and other capacity that expires.
The method is most useful when some demand arrives before other demand and capacity cannot be expanded cheaply in time.
It does not require a firm to charge every customer a different price.
A firm can instead offer a small, transparent menu of products with different booking conditions.

The people blocked by this decision are not only pricing analysts.
A dispatcher needs to know whether to promise another slot.
Sales needs to explain which product is available and why.
Finance needs contribution rather than gross revenue.
Operations needs the model to respect maintenance, labor, equipment, and safety constraints.
Customer service needs a process for exceptions and disappointed customers.
An executive needs to understand how a policy changes both utilization and service risk.

Without an explicit policy, the business may accept every early request and later refuse valuable urgent freight.
Alternatively, it may protect too much and repeatedly depart with empty capacity.
A forecast of the average number of urgent requests does not reveal either tail of the demand distribution.
A usable control therefore depends on uncertainty, timing, and a credible measure of the contribution at risk.

The central decision in this topic is a booking decision for a specific service and departure.
It is not a recommendation to coordinate prices with competitors, discriminate unlawfully, or renege on an accepted booking.
A booking limit governs requests that have not yet been accepted.
Accepted commitments remain subject to their contract and applicable law.

## 2. Learning outcomes

After completing this topic, you should be able to:

1. Identify expiring capacity, the decision time, and the customer request to which a control applies.
2. Calculate net contribution by product and explain why an available slot has an opportunity cost.
3. Derive and apply a two-class protection level using Littlewood's rule and a discrete demand distribution.
4. Translate a protection level into an actionable booking limit while respecting already accepted bookings.
5. Reconcile a forecast, a plan, a management target, and an actual customer commitment.
6. Critique the demand data, assumptions, and operational controls required to use the rule responsibly.
7. Design an evaluation and exception process that measures both economic and service consequences.
8. Identify when cancellations, network interactions, or customer substitution require a richer model.

Exercises 1–8 test these outcomes in order.
The self-check section supplies arithmetic and reasoning for the numerical exercises and rubrics for open designs.
A correct recommendation includes the input values, units, decision horizon, and a clear rule for boundary cases.

## 3. The core mental model

```text
Physical departure capacity and accepted commitments
                    |
                    v
    Remaining sellable units, by departure and time
                    |
                    +----------------------------+
                    |                            |
                    v                            v
  Later high-class demand distribution     Product contribution
                    |                            |
                    +-------------+--------------+
                                  v
                   Marginal value of holding capacity
                                  |
                    compare with low-class contribution
                                  |
                                  v
                Booking limit and protected capacity
                                  |
                   +--------------+---------------+
                   |                              |
                   v                              v
           Booking-system control         Exceptions and monitoring
                   |                              |
                   +--------------+---------------+
                                  v
          Departures, realized contribution, service, learning
```

Capacity is a per-departure inventory of service opportunities.
Unlike a physical pallet of canned goods, an unused appointment slot generally has no value after its deadline.
A high utilization percentage is not the objective by itself.
An early low-contribution booking can fill a slot and still reduce expected contribution if it displaces a more valuable request.
Likewise, a low utilization percentage is not automatically evidence that the protection policy was wrong: an uncertain high-demand event may simply not have occurred.

Think at the margin.
A protected unit has value only when a later request actually needs it.
If later high-class demand would fit without that unit, the marginal value of protecting it is zero.
If that unit is needed, its benefit is the contribution from serving the additional request.
The probability of needing that particular unit decreases as more units are protected.
Protect until the next unit's expected incremental high-class contribution is below the low-class contribution forgone.

The rule assumes product classes can be controlled meaningfully.
Class is a product and booking condition, not necessarily a customer's identity.
An economy service with an advance-purchase deadline may attract planned freight, while a flexible urgent service may serve later requests.
Some customers will switch products if access changes.
The simplest rule assumes away such buy-down; a business must test that assumption rather than assert it.

The forecast is a probability distribution for demand over a defined remaining window.
The plan is the resulting booking limit and operating response.
The target is a management aspiration, such as contribution per departure or on-time performance.
A commitment is a confirmed customer booking under its terms.
None of these is interchangeable with the others.
In particular, a target is not a demand forecast, and a forecast is not authorization to cancel a commitment.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Capacity | Feasible service units on a specified departure; for example, pallet positions per departure | Treating nominal vehicle volume as sellable despite weight or temperature limits |
| Perishability | Loss of the opportunity to sell a capacity unit after its departure or service deadline | Confusing expiring capacity with expiring physical inventory |
| Departure | A particular scheduled service with a specific cutoff and resource capacity | Pooling different departures that cannot share capacity |
| Remaining capacity | Physical capacity less firm commitments and unavailable units, in positions | Subtracting provisional requests as if confirmed or forgetting accepted bookings |
| Product class | A sellable service with stated price and conditions | Equating a class with a legally protected personal characteristic |
| Segment | Customers or use cases with different requirements, arrival patterns, or willingness to pay | Assuming a segment is perfectly observed or stable |
| Contribution | Revenue less incremental variable costs for a fulfilled unit, in currency per position | Comparing gross fares when service costs differ |
| Opportunity cost | Expected value of the best displaced use of a scarce position, in currency | Treating it as an accounting cash expense |
| High class | The later-arriving product with higher contribution in the two-class model | Assuming all late customers always pay more |
| Low class | The earlier product with lower contribution in the two-class model | Calling any low-margin customer unprofitable |
| High-class demand | Number of eligible high-class requests during the specified future window | Using accepted bookings as unconstrained demand when the service was closed |
| Demand distribution | Probabilities for possible future request counts | Replacing it with one average and claiming uncertainty is modeled |
| Survival probability | Probability demand reaches at least a specified integer number of units | Confusing \(P(D\ge j)\) with \(P(D>j)\) at a discrete boundary |
| Protection level | Capacity reserved for possible higher-class demand, in positions | Treating protected space as already booked |
| Booking limit | Maximum accepted low-class units for the relevant capacity and horizon | Treating the limit as a guaranteed number of low requests |
| Littlewood's rule | Accept a low-class unit when its contribution is at least the expected marginal value of saving it for high-class demand under the model assumptions | Using the two-class formula unchanged for a complex network |
| Displacement | Loss of a feasible alternative booking because the relevant resource is already committed | Counting a rejected request that could never have been served |
| Bid price | Estimated marginal value of a unit of constrained capacity, in currency per resource unit | Treating it as a customer-facing offer price |
| Booking curve | Historical cumulative bookings by time before service | Treating observed bookings after closures as true unconstrained demand |
| Censoring | Observing only accepted demand or a lower bound when requests were blocked | Treating sellout as proof that no extra demand existed |
| Show-up | A booked unit actually requiring capacity at service time | Assuming every booking shows up when cancellations and no-shows matter |
| Overbooking | Accepting more reservations than physical capacity based on uncertain show-up | Calling a protection level an overbooking authorization |
| Spill | Demand not served because a product or resource is unavailable | Calling every lost website view a confirmed customer request |
| Spoilage | Capacity that expires unused | Counting a planned maintenance unit as commercially available |
| Booking horizon | Interval between a decision and service departure | Mixing forecasts for different horizons |
| Cutoff | Time after which a booking or product is no longer accepted | Confusing commercial cutoff with physical loading completion |
| No-show | A committed customer does not use a reserved unit under defined terms | Assuming the booking was cancelled in time to resell |
| Nested control | A lower class can use only capacity left after protection for higher classes | Creating separate rigid buckets that unnecessarily strand capacity |
| Forecast | A stated probability or quantity prediction conditioned on information available at a time | Backfilling later knowledge into the historical decision |
| Plan | The intended allocation or control derived from forecasts and constraints | Presenting an aspiration as an observed fact |
| Target | A desired business outcome, not necessarily the statistical expectation | Tuning forecast values solely to meet budget |
| Commitment | A booking accepted under defined service and commercial terms | Treating a protected but unsold unit as a customer commitment |
| Calibration | Agreement, over suitable comparable cases, between predicted probabilities and observed outcomes | Inferring probability quality from a single departure |
| Counterfactual | Outcome that would have occurred under another policy | Equating an unobserved rejected request with certain realized contribution |

## 5. The mechanics, step by step

### Step 1 — Define the resource and the expiry

Select one decision unit before choosing a model.
A position on a truck departing a specific origin for a specific destination at a specified time can be a decision unit.
The resource expires for this departure when the acceptance or loading deadline passes.
State the time zone and the relevant booking cutoff.
If loading closes 90 minutes before departure, demand arriving afterward may not be operationally eligible even if the truck remains on site.

Input: route, departure date and time, capacity definition, booking cutoff, and service specification.
Output: a resource identifier and a decision window.
Branch: if capacity can be transferred between services or extra vehicles can be purchased at a material cost, model that option rather than assuming a fixed, isolated resource.
For a truck, a pallet-position count is a valid simplification only when weight, volume, axle load, refrigeration, and compatibility do not bind first.
When multiple constraints bind, use a resource vector or a documented conservative equivalent unit.
A control that sells nominal positions beyond a weight limit is not a valid revenue-management policy.

### Step 2 — Reconcile feasible capacity and commitments

Obtain the operating plan from the party accountable for service execution.
Subtract planned maintenance, blocked positions, safety buffers, and other capacity already unavailable to commercial sale.
Subtract accepted commitments when applying a control to remaining capacity.
Decide explicitly how tentative holds are treated, when they expire, and who can release them.
Do not subtract the same block in both the operations feed and the booking system.

Input: physical capacity, operational blocks, confirmed bookings, and holds with timestamps.
Output: remaining sellable capacity and a reconciliation record.
Branch: if systems disagree, stop automated acceptance at a safe threshold and route the discrepancy to operations.
The safe threshold is a business control, not an estimate of true capacity.
A revised plan does not automatically revoke an accepted customer commitment.
If a vehicle change makes commitments infeasible, use the contractually approved disruption process.

### Step 3 — Define products that can be sold and controlled

List each offer's price, eligible customers, booking deadline, flexibility, service promise, and cancellation terms.
For a simple two-class rule, assign a high-contribution later class and a low-contribution earlier class.
The assignments must describe actual offers, not a label retrofitted to realized customers.
A flexible urgent product might include a later acceptance cutoff or stronger service commitment.
A lower-priced product might require advance booking and reduced flexibility.
The differences must be real, communicated, and consistently applied.

Input: current tariffs, contracts, product restrictions, and sales channels.
Output: a product map with price, cost, eligibility, timing, and ownership.
Branch: if low-class requests can arrive after high-class requests or high-class customers can freely buy the low product, the elementary rule is only an approximation.
Measure the size of such overlaps before using the result as a control.
Avoid access restrictions based on protected personal characteristics; legal review depends on jurisdiction and facts.
A commercial segment is not proof that differential treatment is permitted.

### Step 4 — Calculate comparable incremental contributions

For each fulfilled unit, begin with collectible revenue net of discounts and channel fees.
Subtract costs that vary with that unit and the specified decision.
These might include handling, fuel surcharge paid to a subcontractor, insurance, and expected claims costs where defensible.
Retain fixed vehicle costs only when the decision changes them.
If accepting one extra unit forces an additional vehicle, the relevant cost is discontinuous and a constant unit contribution is not enough.

Let \(r_L\) be low-class contribution and \(r_H\) be high-class contribution, both in currency per position.
The basic rule assumes \(0 < r_L < r_H\).
If \(r_L\le 0\), additional demand may destroy contribution unless there is another explicit benefit or constraint.
If \(r_L\ge r_H\), the assumed high-versus-low ordering is wrong and must be revisited.
Use the same currency, tax treatment, time basis, and resource unit for both classes.

Input: unit revenue, variable cost, and the decision's incremental scope.
Output: auditable contributions and a list of excluded costs.
Branch: if high-class service imposes higher operational risk or consumes more than one equivalent unit, adjust the economics and resource consumption before comparing classes.
A gross-revenue-optimal booking can be contribution-destructive.

### Step 5 — Forecast eligible future high-class demand

Define \(D_H\) as the number of eligible high-class requests that would arrive after the decision and before cutoff.
Forecast the full distribution, not just its mean.
Use departures with comparable route, weekday, season, lead time, product conditions, and capacity context.
Include relevant known events only if their information was available at the decision time.
A forecast for an entire booking horizon is not appropriate after part of that horizon has already elapsed.

Input: request-level arrivals, offer and availability history, product terms, operational calendars, and decision timestamp.
Output: a conditional probability distribution for remaining eligible high-class demand.
Branch: when observations are sparse, use pooling across comparable departures or a conservative scenario analysis, with uncertainty about the pooled model made visible.
Do not silently replace a missing distribution with a precise-looking average.
For a count model, check whether the tail is sensible and whether special-event days need separate treatment.

Observed high-class bookings can understate high-class requests when capacity was unavailable.
Record blocked requests where possible, including request time, quantity, channel, product, and reason for refusal.
A quote view is not necessarily an executable request.
Deduplicate retries and distinguish an inquiry from a customer who would have accepted the terms.
Even a recorded refusal is not automatically a confirmed counterfactual sale.
Availability, price, and customer behavior affect whether the customer would actually have booked.
Use appropriately cautious unconstraining methods and sensitivity ranges when historical demand is censored.

### Step 6 — Compare the incremental alternatives

For an isolated capacity unit, compare accepting a low-class request now with retaining it for high-class demand.
Suppose all relevant low requests arrive before the later high requests.
If \(j\) units are protected, the \(j\)-th protected unit is used by high demand exactly when \(D_H\ge j\).
Its expected high-class contribution is therefore \(r_H P(D_H\ge j)\).
The certain low-class contribution sacrificed to reserve the unit is \(r_L\), assuming a low request is available.

Protect the \(j\)-th unit when:

\[
r_H P(D_H\ge j) > r_L.
\]

If equality holds, either decision has the same expected contribution in this static model.
Declare a tie convention before implementation.
The worked example chooses not to protect the tied unit, serving the available low request.
An organization could choose the other tie rule for a separately justified service objective.
The rule does not claim that the alternative has the same service or customer-mix outcome.

The threshold ratio is \(r_L/r_H\).
An equivalent continuous approximation protects the quantile at cumulative probability \(1-r_L/r_H\).
For discrete demand, a quantile shortcut can be ambiguous at mass points and ties.
Enumerating marginal units using \(P(D_H\ge j)\) makes the convention explicit.
If a normal approximation is used, also inspect the implied negative demand, capacity truncation, and integer rounding.

Input: comparable unit contributions and the remaining high-demand probability distribution.
Output: marginal values by protected unit and the selected protection level.
Branch: if there is no low request to reject, holding capacity has no immediate low-class opportunity cost.
If only part of a multi-position request can be accepted, check the customer's all-or-nothing terms before applying a unitwise decision.

### Step 7 — Convert protection into an executable booking limit

For initial capacity \(C\) and a protection level \(y\), the simple low-class booking limit is \(b=C-y\).
Accept low-class requests only until accepted low-class positions reach \(b\), subject to actual remaining physical capacity and all other controls.
The high class can use unoccupied physical capacity, including capacity not taken by low demand.
This is a nested control, not a separate hard allocation that prevents high demand from using idle low capacity.

For a fresh decision after commitments already exist, distinguish the total low-class limit from an additional low-class allowance.
If \(R\) is currently uncommitted sellable capacity and \(y_R\) is protection calculated for remaining high demand, the incremental low allowance is at most \(\max(0,R-y_R)\).
The total cumulative low booking limit would equal accepted low bookings plus that allowance, provided those bookings are already deducted from \(R\).
Do not apply \(C-y\) to the remaining capacity while also subtracting bookings a second time.

Input: capacity basis, existing bookings, protection level, and product mapping.
Output: a timestamped integer limit or incremental allowance in the booking system.
Branch: if capacity falls below accepted commitments, the limit cannot fix the shortfall; escalate to operations and contractual recovery.
If capacity rises, recalculate protection and the low-class allowance rather than simply opening every extra unit.

### Step 8 — Update through the booking horizon

As time passes, high-class demand already realized must be separated from remaining high-class demand.
Reforecast the remaining window at agreed decision times, or after a material event such as a service disruption.
A control can open or close future low requests without cancelling accepted ones.
A late decline in the high-demand forecast may justify releasing previously protected capacity if a low request can still be sold.
If the low product is no longer available after its advance cutoff, the theoretical release does not create demand.

Input: latest bookings, capacity, eligible request history, and forecast information as of the update.
Output: revised control with version, reason, and effective time.
Branch: if late high demand arrives during the low booking window, consider a sequential dynamic program or simulation instead of applying the static early-low, late-high rule without qualification.
An update schedule must balance responsiveness against operational instability and stale data.
Record what the system knew when each decision was made.

### Step 9 — Implement customer-facing controls

Translate the mathematical limit into a product-availability decision.
Specify which booking channels must enforce it, how quickly they receive updates, and what happens during a synchronization failure.
A salesperson should see whether the offer is available before promising it.
Define who can override a control, the reason codes, the approval threshold, and the expiration of the override.
An override may be justified by contractual commitments, a recovery action, or a documented strategic account decision.
Its expected cost should be visible rather than hidden in a later forecast adjustment.

Input: approved limit, channel mapping, terms, and operational controls.
Output: an enforceable, explainable availability rule and an audit event.
Branch: if a channel cannot enforce the rule reliably, restrict its inventory allocation or use an approved manual process until it can.
Do not display a price or service condition that the business is unwilling or unable to honor.

### Step 10 — Measure the result and update the process

Compare realized net contribution and service outcomes with the policy and feasible alternatives.
Track low and high accepted demand, rejected eligible requests, unused capacity, late service failures, overrides, and data delays.
Report the distribution of outcomes across departures rather than only a network average.
Differentiate lost contribution from lost gross revenue.
Use experimental variation or a defensible historical replay when evaluating causal impact.

Input: decision logs, requests, bookings, departures, costs, and customer outcomes.
Output: a policy scorecard and a documented decision to retain, revise, or pause the control.
Branch: if service harm or systematic data errors exceed the approved threshold, suspend automation and use the designated fallback.
One empty departure does not disprove a protection rule, just as one full departure does not validate it.
An evaluation needs enough comparable cases to distinguish ordinary uncertainty from a recurring failure.

### A note on more than two classes

With several classes, businesses often use nested controls and approximations such as expected marginal seat revenue.
A class may consume the same unit as multiple other classes, so its protection cannot be selected independently without checking the combined allocation.
Approximation quality depends on arrival order, demand dependence, and the objective.
For a network, one itinerary or shipment may consume several legs.
The correct trade-off then depends on the combined opportunity cost of every constrained resource, not just the price of the request.
If one request occupies both a scarce origin hub slot and a scarce linehaul slot, add the relevant marginal resource values before comparing with its contribution.
A network bid-price system is a richer control, not a direct application of the two-class formula to each leg separately.

### A note on cancellations and overbooking

Protection protects against future demand; overbooking addresses uncertain consumption by already accepted bookings.
They solve different problems and can interact.
When cancellation and show-up rates differ by class, a booking is not the same as a consumed position.
A valid overbooking decision needs show-up distributions, dependence during disruptions, denied-service costs, customer remedies, and legal and contractual constraints.
Do not infer an overbooking limit from Littlewood's rule.
The example below assumes no cancellations, no no-shows, and no overbooking.

### Step 11 — Review the shape of uncertainty

For each future demand count, retain its probability rather than storing only the recommended limit.
The probability distribution allows a reviewer to reconstruct the marginal protection calculation.
It also permits a risk discussion that is impossible from an expected value alone.
For example, two forecasts can have the same mean and different probabilities of very high demand.
One may place nearly all its probability around the mean.
The other may put substantial probability on both zero and a very large count.
The latter generally exposes the operation to more extreme spill or spoilage.
Which protection level changes depends on the survival probability at the relevant marginal unit, not on variance alone.

Inspect each tail used by the rule.
For a proposed protection of three units, compare the probability of at least three requests against the contribution ratio.
Then inspect the probability of at least four requests to understand the cost of increasing protection by one.
If the tail probabilities are close to the threshold, a small forecast error may change the recommendation.
That is a reason to show a range of decisions, not to report a spurious single-unit precision.
The business can choose a tie or near-tie policy that incorporates a separately approved service objective.
It should not describe the latter choice as maximizing the original expected-contribution objective without qualification.

Consider a second distribution with two possible demand outcomes.
Demand is zero positions with probability 0.50 and four positions with probability 0.50.
The mean is two positions.
Each of the first four units has a 0.50 probability of being needed.
With USD 60 low contribution and USD 120 high contribution per position, all four marginal values tie the low contribution.
Under the no-protection-on-ties convention, the selected protection is zero.
Under the opposite convention, protecting all four is also optimal in expected contribution if four positions are available and low demand fills the rest.
The realized outcomes and service mix, however, differ considerably between those policies.
This example illustrates why a tie rule and a risk discussion belong in the decision record.

Now consider demand that is always exactly two positions.
The mean remains two positions.
The first and second units are needed with probability 1.00.
The third and fourth units are needed with probability 0.00.
With the same contributions, protecting exactly two units is optimal.
The identical mean did not yield the same set of optimal protection levels.
The full probability distribution is decision-relevant.

### Step 12 — Account for uncertainty in the estimate

The demand distribution is an estimate obtained from limited history.
Even if a chosen statistical family fits the historical sample, the parameter values are uncertain.
A route with only ten comparable departures cannot support the same confidence as a stable route with hundreds.
Comparability matters as much as raw sample size.
Many observations from the wrong season or service definition do not necessarily improve the forecast.

Construct a base, low-demand, and high-demand scenario from defensible evidence.
For each scenario, compute the marginal values and protection recommendation on the same contribution basis.
State why each scenario is plausible and which observed events would make it more or less likely.
If every scenario produces the same limit, the decision may be relatively insensitive to the assumed distribution.
If the limits vary, show management the expected contribution and service consequences of each candidate.
Do not average arbitrary scenarios and call the result a calibrated probability distribution.
Stress scenarios are useful for exposure analysis even when their probabilities are unknown.

Backtesting should reproduce the historical decision process.
At each historical decision time, train or retrieve a forecast using only earlier data.
Apply the candidate control to the capacity and product conditions available then.
Compare predicted demand distributions with the best available request observations.
Where historical policy censored demand, record that limitation rather than pretending the true counterfactual is known.
A replay can reveal implementation defects, but its economic comparison depends on the behavioral and demand assumptions used.
It is not a substitute for an experiment when treatment changes customer behavior.

### Step 13 — Define the unit of acceptance

The unit used by the model must match the unit accepted by the customer.
One position per request is convenient but not universal.
A shipper may require three adjacent pallet positions or refuse a partial shipment.
For such a request, a unit-by-unit rule can produce an offer that the customer cannot use.
Compare the expected value of accepting the entire request with the expected value of retaining all the resources it would consume.
This requires evaluating the distribution over the relevant remaining capacities, not merely multiplying one marginal value by the request size when those marginal values differ.

For a three-position request, the three displaced marginal positions need not be equally valuable.
The first held position may have a high chance of use while the third has a much lower chance.
Sum the opportunity costs of the specific positions lost under the feasible alternative.
If the request consumes different resources, each resource constraint must be respected jointly.
A simple sum of independent leg-level values may itself be an approximation when resource use and demand are dependent.
The decision record should state that approximation and its known limitations.

Partial acceptance can be a valid commercial option only when the customer has agreed to it.
Record whether partial fulfillment changes the unit price, handling cost, or service terms.
If it forces an additional pickup or delivery, include that incremental cost.
If the customer rejects the partial offer, the model must not count those units as accepted contribution.
The relevant alternatives are the offers the business can actually make and the customer can actually accept.

### Step 14 — Specify the control's lifecycle

Before a departure opens for sale, verify product mappings and the available capacity feed.
At launch, publish the approved limits to every channel that can accept requests.
During the selling window, monitor booking events and the current capacity version.
When a relevant input changes, calculate a revised decision using information available at that moment.
Record both the old and new limits, the reason for the change, and the person or system authorizing it.

When a control closes, the channel should stop accepting new requests for that product.
It should not silently change the conditions of existing bookings.
When a control reopens, make sure the product remains eligible under its stated cutoff.
An advance product whose deadline has passed cannot be sold merely because protected space is now free.
At the operating cutoff, reconcile confirmations against the departure manifest.
After departure, preserve the final capacity, actual utilization, financial settlement, and any service recovery.

The evaluation record should link each outcome to the decision version that governed it.
If there were several versions, retain the effective interval of each version.
An override needs its own timestamp and authorized reason.
A later correction to a cost or booking record should not erase the historical value used by the original decision.
This event history supports both operational audit and valid backtesting.

## 6. Worked example

### Business situation and assumptions

A regional refrigerated carrier operates one fixed departure on a route.
Operations confirms five equivalent pallet positions can be sold on that departure.
The positions are genuinely interchangeable for the example: weight, cube, temperature, and handling constraints do not bind separately.
There are no prior bookings at the decision time.
The advance product yields USD 80 of incremental contribution per position.
The urgent product yields USD 160 of incremental contribution per position.
The contributions are net of the incremental costs attributable to each service.

Low-class demand is available now and can fill any allowed advance positions.
Later high-class requests arrive before the operational cutoff.
Requests consume exactly one position each, are independent of the chosen advance limit, and are accepted if capacity remains.
There are no cancellations or no-shows.
Customers cannot switch between products in response to availability.
Unused capacity has zero salvage value for this departure.
The objective is expected total contribution for this one departure, not a service-level constraint or network-wide objective.

These are deliberately strong assumptions.
The calculation is a benchmark and decision aid, not a claim that every freight market meets them.

### Forecast, plan, target, and commitment

The forecast is the following distribution for eligible urgent requests remaining before the cutoff.
The carrier's commercial target is USD 540 contribution on this departure; it is an aspiration, not the forecast mean.
The planned booking limit will be computed from the distribution and contributions.
A commitment exists only when a customer actually receives an accepted booking.
The hypothetical plan by itself does not promise a position to any particular customer.

| Urgent demand \(D_H\), positions | Probability |
|---:|---:|
| 0 | 0.05 |
| 1 | 0.15 |
| 2 | 0.30 |
| 3 | 0.25 |
| 4 | 0.15 |
| 5 | 0.10 |
| **Total** | **1.00** |

The mean forecast is \(0(0.05)+1(0.15)+2(0.30)+3(0.25)+4(0.15)+5(0.10)=2.60\) positions.
The mean alone does not decide how many positions to protect.
A unit is useful only in scenarios in which it is actually needed.
The probabilities of reaching each unit are calculated from the distribution's upper tail.

### Marginal protection calculation

The advance-to-urgent contribution ratio is USD 80 / USD 160 = 0.50.
The table compares the expected contribution from the marginal protected position with the certain USD 80 available from an advance sale.

| Protected position \(j\) | \(P(D_H\ge j)\) | Expected urgent contribution of position \(j\) | Decision |
|---:|---:|---:|---|
| 1 | 0.95 | USD 152 | Protect: USD 152 > USD 80 |
| 2 | 0.80 | USD 128 | Protect: USD 128 > USD 80 |
| 3 | 0.50 | USD 80 | Tie: do not protect under chosen convention |
| 4 | 0.25 | USD 40 | Do not protect |
| 5 | 0.10 | USD 16 | Do not protect |

The selected protection level is two positions.
The advance booking limit is \(5-2=3\) positions.
This limit is a maximum, not a commitment that three advance requests will arrive.
Under the stated assumption, three requests do arrive and are accepted, yielding USD 240.
Two positions remain available for urgent requests.

### Scenario-by-scenario contribution

| Urgent requests | Probability | Urgent positions served | Advance contribution | Urgent contribution | Total contribution | Unused positions |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.05 | 0 | USD 240 | USD 0 | USD 240 | 2 |
| 1 | 0.15 | 1 | USD 240 | USD 160 | USD 400 | 1 |
| 2 | 0.30 | 2 | USD 240 | USD 320 | USD 560 | 0 |
| 3 | 0.25 | 2 | USD 240 | USD 320 | USD 560 | 0 |
| 4 | 0.15 | 2 | USD 240 | USD 320 | USD 560 | 0 |
| 5 | 0.10 | 2 | USD 240 | USD 320 | USD 560 | 0 |

Expected urgent positions served are \(0(0.05)+1(0.15)+2(0.80)=1.75\).
Expected contribution is USD 240 + USD 160 × 1.75 = USD 520 per departure.
The probability of leaving at least one protected position unused is 0.20.
The expected number of unused positions is \(2(0.05)+1(0.15)=0.25\).
Expected urgent requests not served are \(2.60-1.75=0.85\) positions.
These are modeled requests, not confirmed lost sales or guaranteed lost customers.

The USD 540 target exceeds the USD 520 expected contribution under these assumptions.
This gap does not authorize changing the forecast.
Management can investigate commercial options, costs, additional feasible capacity, or the target itself.
The number USD 520 is an expected value, not a promise that the next departure will realize that amount.

### Check competing protection levels

For \(y\) protected units, the example's expected contribution is:

\[
80(5-y)+160E[\min(D_H,y)].
\]

The first term assumes enough advance demand to use every unprotected position.
The second term is capped by protected capacity because the advance product fills the remaining positions.

| Protection \(y\), positions | Advance limit, positions | Expected urgent positions served | Expected total contribution |
|---:|---:|---:|---:|
| 0 | 5 | 0.00 | USD 400 |
| 1 | 4 | 0.95 | USD 472 |
| 2 | 3 | 1.75 | USD 520 |
| 3 | 2 | 2.25 | USD 520 |
| 4 | 1 | 2.50 | USD 480 |
| 5 | 0 | 2.60 | USD 416 |

Protecting two or three positions ties on expected contribution.
The chosen tie rule protects two, accepting the third available advance position.
Protecting three gives a different risk profile: more urgent requests can be served in high-demand scenarios, while more capacity can expire unused in low-demand scenarios.
For example, when urgent demand is zero, contribution with three protected positions is USD 160 rather than USD 240.
When urgent demand is at least three, it is USD 640 rather than USD 560.
The expected monetary tie does not settle a separately specified service, risk, or relationship objective.
Management can choose three if an authorized secondary objective warrants it, but should document that the choice changes customer and utilization outcomes.

### What changes when a booking has already been accepted?

Suppose one advance position has been confirmed before a new decision.
Four positions remain, and the same forecast genuinely applies to the remaining future urgent window.
Protection of two positions implies an additional advance allowance of \(4-2=2\) positions.
The cumulative advance limit is then one accepted position plus two additional positions, or three positions.
The already accepted position is not cancelled to achieve the limit.
If instead two urgent positions had already been accepted, the remaining-demand distribution would normally need to be conditioned on that information and the elapsed time.
Using the original distribution without justification could protect for demand already realized.

### Sensitivity and decision boundaries

If the advance contribution rises to USD 100 per position and all else stays fixed, the threshold ratio becomes 0.625.
The first two urgent tail probabilities exceed 0.625; the third, 0.50, does not.
The protection level remains two positions.
At USD 130 advance contribution, the threshold is 0.8125, so only the first unit is protected.
The second unit's expected urgent contribution is USD 128, less than USD 130.
At USD 40 advance contribution, the threshold is 0.25.
The fourth unit ties and the chosen convention protects only the first three units.

If a later forecast puts more probability on high urgent demand, the survival probabilities may rise and more protection may be warranted.
A higher mean by itself is not sufficient to determine every tail probability.
Compare the full distribution at the decision-relevant thresholds.
If urgent customers can book the advance product instead, an increase in the advance limit may reduce observed urgent demand.
The fixed-demand comparison then overstates or misstates the policy's benefit.
Test that behavior with customer-level evidence or an appropriate experiment.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
Save it as a local script and run it with `python script.py`.
It enumerates all feasible protection levels for the same static scenario.
It uses exact decimal arithmetic so that the USD 520 tie is not an artifact of floating-point comparison.
The dictionary maps urgent-request counts to probabilities.
The calculation assumes unlimited advance demand up to the booking limit, no show-up uncertainty, fixed contributions, and no substitution.
It does not estimate demand or authorize an operational booking decision.

```python
from decimal import Decimal

capacity = 5
low = Decimal("80")
high = Decimal("160")
probability = {
    0: Decimal("0.05"),
    1: Decimal("0.15"),
    2: Decimal("0.30"),
    3: Decimal("0.25"),
    4: Decimal("0.15"),
    5: Decimal("0.10"),
}

assert sum(probability.values()) == Decimal("1")
assert all(p >= 0 for p in probability.values())
assert 0 < low < high


def expected_high_served(protection):
    return sum(
        Decimal(min(demand, protection)) * p
        for demand, p in probability.items()
    )


def contribution(protection):
    advance_positions = capacity - protection
    return low * advance_positions + high * expected_high_served(protection)


print("protected  low_limit  high_served  contribution_usd")
results = []
for protection in range(capacity + 1):
    expected = contribution(protection)
    results.append(expected)
    print(
        f"{protection:>9}  {capacity - protection:>9}  "
        f"{expected_high_served(protection):>11.2f}  {expected:>16.2f}"
    )

best_value = max(results)
# On a contribution tie, protect the fewest units.
best = next(y for y, value in enumerate(results) if value == best_value)
print(f"selected protection: {best} positions")
print(f"advance booking limit: {capacity - best} positions")
```

Expected output:

```text
protected  low_limit  high_served  contribution_usd
        0          5         0.00            400.00
        1          4         0.95            472.00
        2          3         1.75            520.00
        3          2         2.25            520.00
        4          1         2.50            480.00
        5          0         2.60            416.00
selected protection: 2 positions
advance booking limit: 3 positions
```

Enumeration is useful as an independent arithmetic check on the marginal rule.
For larger capacity, a demand model can produce a longer probability mass function, but the units and cutoff must remain explicit.
If total probability is incomplete because the upper tail is truncated, add the remaining probability to an appropriate tail representation rather than silently renormalizing without explanation.
If demand can exceed capacity, values above capacity still affect spill even though served urgent positions are capped.
The code's `min` handles such values for contribution, provided their probabilities are supplied.
The calculation assumes each unprotected position is filled by an advance booking.
If advance demand is uncertain or requests arrive in batches, represent those scenarios explicitly; the first term is no longer necessarily \(80(5-y)\).

To adapt the calculation to remaining capacity, use remaining eligible demand and net unit contributions as of the actual decision timestamp.
Reconcile previously accepted bookings outside this function.
For production use, add controlled input validation, source lineage, access permissions, operational exception handling, and independent review appropriate to the system.
Do not interpret the teaching assertions as a complete production validation layer.

## 8. Data, standards, and regulatory context

### Decision-grain data

The minimum useful record is not just a departure-level revenue total.
Store the departure identifier, route, planned service time, cutoff, and physical capacity version.
For each request, record the request timestamp, quantity, product, quote terms, channel, and whether the request was accepted, refused, withdrawn, or expired.
Record the reason for refusal and whether the customer would accept a partial quantity where known.
For each accepted booking, retain confirmation time, modifications, cancellations, actual service, and relevant costs.
Link these events using stable identifiers without overwriting earlier states.

A departure-capacity table should distinguish nominal capacity, operational blocks, sellable capacity, and the reason and time of each revision.
A forecast table should store the version, issue time, remaining horizon, demand distribution, and feature snapshot.
A control table should store the protection, booking limit, effective interval, owner, and override reason.
A settlement table should capture realized net revenue and incremental cost on an agreed accounting basis.
Each table needs a clear grain and unique key.

Time alignment is essential.
A historical evaluation must use only information available when the historical decision was made.
A revised capacity record entered after departure is useful for reconciliation but cannot be treated as a real-time signal available earlier.
Account for time zones, daylight-saving transitions, cutoffs, and delayed channel updates.
State whether a cancellation timestamp is the customer's notice time or the time an agent entered it.
A late-entered cancellation is not necessarily sellable inventory at the earlier time.

### Demand observation and bias

When low availability blocks a customer before a formal request, a request table may still miss demand.
Collect availability displays and relevant quote events where lawful and proportionate.
Do not equate repeated browsing with independent demand.
A customer may ask multiple channels about the same shipment.
A blocked request may have been abandoned even if capacity had been available at the offered price.
Compare unconstraining estimates against periods with broad availability and against genuinely logged refusals.
Document the resulting uncertainty rather than representing recovered demand as observed fact.

Policy affects the data it later learns from.
A restrictive policy may record fewer urgent bookings because it blocks a channel or changes customer behavior.
A generous advance limit may cause urgent customers to buy early rather than reveal their urgent willingness to pay.
When the policy changes, validate whether the relationship between observed bookings and latent requests has also changed.
Separate forecast performance for total eligible requests from performance for accepted bookings.
The latter is partly determined by the control itself.

### Standards and interoperability

A carrier may use electronic data interchange (EDI) messages or application programming interfaces to exchange shipment requests and status events.
The applicable message implementation guide and bilateral trading-partner agreement define the actual fields and event meanings.
A standard message format does not guarantee that every refusal, capacity hold, or customer intention is captured.
Validate code lists, timestamps, unit definitions, duplicate handling, and acknowledgments at each integration boundary.
For physical trade items and logistics units, GS1 identification standards can provide common identifiers where adopted.
Such identifiers support matching across systems; they do not define a revenue-management protection level.
Use the authoritative version of the relevant standard and local implementation agreement rather than assuming one generic format applies to all carriers.

### Privacy, pricing, and consumer protection

Customer-level booking and quote data can be personal data when they identify or relate to an individual.
The European Union's General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, establishes obligations for covered processing, including principles of purpose limitation, data minimization, and accuracy in Article 5.
Its application depends on the facts and territorial scope; a corporate shipment dataset is not automatically free of personal data.
Establish a lawful basis, retention schedule, appropriate access controls, and a process for rights requests with qualified counsel where applicable.
Do not retain personal identifiers merely because they might be useful for an unspecified later model.

The U.S. Federal Trade Commission's rule on unfair or deceptive fees, 16 CFR Part 464, applies to specified live-event ticket and short-term lodging offerings, not every freight quote.
Its total-price disclosure provisions are an example of product-specific consumer-protection requirements, not a universal freight pricing rule.
Other jurisdictions and products can have different transparency, anti-discrimination, contract, and consumer-protection obligations.
Check the current text and effective dates with counsel before implementation.
Fair and transparent product conditions matter even when a particular consumer rule does not govern a business-to-business transaction.

Competition law also constrains independent pricing conduct.
Public sources, internal cost information, and independent demand forecasts are different from exchanging competitively sensitive future pricing plans with rivals.
Do not use this topic as a legal clearance for information sharing or algorithmic pricing.
Document who approves price and availability rules and route proposed collaborations or sensitive exchanges to legal review.
The method is an independent firm's allocation decision, not an invitation to coordinate with competitors.

### Governance controls

Assign accountable owners for the capacity feed, demand forecast, cost inputs, product definitions, and deployment.
Record the model version and tie convention for each departure.
Restrict manual overrides and make them observable to both commercial and operations owners.
Test for disparate or otherwise inappropriate outcomes where customer-level data or eligibility rules make that relevant.
Review any use of personal characteristics or proxy variables with legal and governance teams.
Maintain a practical fallback when the forecast feed is stale, the capacity feed fails, or a channel cannot synchronize.
A documented default control should be safe to execute, not just mathematically convenient.

### Operational data-quality checks

Check that each accepted booking references an existing departure and an active product.
An orphaned booking can be omitted from remaining capacity even though a customer holds a confirmation.
Check that quantity is positive, expressed in the resource unit, and consistent across amendments.
If a booking changes from two to three positions, store the amendment rather than counting both quantities as separate active bookings.
Reconcile the sum of active confirmed quantities against the booking system's own total.
Investigate differences before using the value as a control input.

Check that request events have a plausible order.
A confirmation cannot normally precede the request that generated it.
A cancellation should reference a previously accepted booking.
A refusal reason should identify whether the cause was capacity, product eligibility, customer withdrawal, or another condition.
Do not infer a capacity refusal from a missing confirmation alone.
Customers can abandon an inquiry for reasons unrelated to available space.

Check whether every forecast covers the intended remaining window.
The forecast issue time must precede the decision time.
The forecast end time must not exceed the product's actual operational cutoff without a documented adjustment.
If the feed is stale, show its age explicitly to the decision owner.
The fallback policy should identify the maximum acceptable age and the action when that age is exceeded.
That threshold is an operating decision informed by the speed at which demand and capacity change.

Check the probability distribution itself.
Every probability must be nonnegative, and the probabilities including any represented tail must sum to one within the chosen numerical tolerance.
Demand counts should be nonnegative integers when the modeled unit is indivisible.
If the forecast is continuous but booking units are discrete, state the conversion and boundary convention.
Review any very large upper-tail value against physical demand, even though requests can exceed physical capacity.
Demand is not capped by the number of positions the carrier can serve.

### Reconciliation and access controls

Separate permission to change inputs from permission to approve and publish a commercial control.
An analyst may prepare a forecast while an accountable manager approves the deployment.
Operations should own corrections to physical capacity or explicitly approve them.
Finance should approve changes to the cost mapping that materially alter product contribution.
The booking platform should log which control version it enforced.
These responsibilities can be assigned to the same person in a small operation, but the review and audit trail still need to be explicit.

Reconcile confirmed bookings with fulfilled services after departure.
Record legitimate differences such as cancellations, no-shows, partial fulfillment, or service recovery.
Reconcile actual revenue with discounts, credits, and other settlement adjustments.
An apparent contribution increase caused solely by delayed credits should not be treated as a successful policy.
Likewise, a late accounting correction should update realized performance without rewriting the original decision-time input.
The two records answer different questions: what was known then and what ultimately happened.

Protect access to customer and shipment records according to their sensitivity and applicable obligations.
Give model-development users only the fields needed for their purpose.
Where identifiers are replaced with pseudonyms, maintain controls over the mapping and recognize that pseudonymized data can still be personal data.
Use aggregated reports where individual-level detail is unnecessary.
Set retention and deletion rules with the relevant business and legal owners.
Data minimization must not remove the decision audit trail required for a legitimate operational purpose without an approved alternative.

Finally, establish a route for a customer or internal user to challenge an incorrect availability or service decision.
Correcting an erroneous booking or an inaccurate customer record is an operational responsibility, not merely a model-retraining task.
The review should distinguish an error in data, an authorized policy outcome, and a contractual exception.
Record the resolution without exposing confidential information to unrelated users.

## 9. Industry practice and case evidence

### Airline yield management

Airlines are a canonical setting for revenue management because a seat on a departed flight cannot be sold later.
Littlewood's 1972 paper formulated the two-fare trade-off: a lower fare should be evaluated against the expected value of holding capacity for later higher-fare demand.
Belobaba's published work extends the discussion to multiple fare classes and heuristic protection methods.
These references establish analytical approaches, not a guarantee that a particular airline currently uses the exact simplified rule in this topic.
Actual airline systems face connecting itineraries, multiple fare products, cancellations, no-shows, and frequent updates.

American Airlines describes the development of yield management in its corporate history and public communications.
Smith, Leimkuhler, and Darrow's 1992 Interfaces article documents the practice at American Airlines and describes the business contribution of yield management.
Those are historical public accounts, not independent proof of a current result on any named route.
The relevant business lesson is that controls require both forecasting and a reservation system capable of enforcing availability.
A mathematically attractive protection level that sales channels ignore does not protect anything.

### Hotel room inventory

A hotel room for a particular night is also a perishable service unit.
A customer requesting several consecutive nights consumes several date-specific inventories.
The opportunity cost of accepting a reservation is therefore not necessarily captured by a single-night rate.
Revenue management research on hotels documents decisions involving price, length of stay, and room availability.
A control can limit low-rate availability on a peak night while leaving other dates open.
This is a useful analogy, not evidence that the two-class freight example predicts any particular hotel's outcomes.

### Freight and logistics application

A trucking departure, ocean sailing, or warehouse appointment may exhibit the same time-sensitive capacity trade-off.
The operational constraints can be more heterogeneous than a seat count.
A shipment may consume weight, cube, temperature-controlled space, driver hours, and handling capability at the same time.
It may also have contractual service priorities that an expected-contribution objective alone does not capture.
A carrier can begin with a homogeneous lane and a small set of comparable products before extending the control to a network.
That is a practical design recommendation, not a claimed outcome from a named carrier.

For a warehouse, the relevant scarce resource might be a refrigerated dock-door appointment during a particular shift.
The commercial rule must still respect labor, handling time, and safety constraints.
A high-paying appointment that requires twice the labor time is not necessarily more valuable per constrained hour.
Use a consistent resource unit or a multi-resource optimization.
A position's apparent contribution can be misleading if it occupies a bottleneck needed by several other activities.

### Interpreting public cases

Public case studies often report a revenue improvement relative to a baseline that is not fully observable to the reader.
Ask whether the baseline includes concurrent price, schedule, market, or product changes.
Check the time period, scope, cost treatment, and whether the claimed effect is gross revenue or net contribution.
A vendor's statement that a system improved revenue is a claim by that source, not automatically a causal estimate.
A research article's simplified model may be well established while its empirical assumptions remain unsuitable for a specific lane.
Transfer the decision structure, then validate the input data and constraints locally.

### Decision process in practice

A practical daily review brings together revenue management, operations, sales, finance, and customer service.
Operations confirms capacity and service feasibility.
The analyst presents demand uncertainty, proposed limits, expected contribution, and sensitivity.
Sales checks contract obligations and product availability.
Finance checks contribution definitions and unusual costs.
Customer service prepares explanations and escalation paths.
A designated owner approves exceptions and records the reason.
After departure, the team reviews realized demand and failures without treating every unfavorable outcome as a model defect.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Nominal capacity used as sellable capacity | Operations repeatedly blocks positions after limits are published | Weight, maintenance, or safety constraints were omitted | Overcommitment or emergency recovery cost | Reconcile capacity versions with operations before publication |
| Historical demand is censored | Forecast falls after repeated sellouts | Blocked requests are absent from accepted-booking history | Too little protection and systematic spill | Record availability and refusals; test unconstraining and sensitivity |
| Low customers switch to urgent or vice versa | Product mix changes abruptly after a control change | The fixed-demand assumption fails | Estimated displacement is biased | Measure switching and evaluate policy changes experimentally where feasible |
| Gross revenue replaces contribution | High-price product has unusually high handling cost | Price is compared without incremental cost | The wrong class receives scarce capacity | Reconcile incremental economics with finance |
| Wrong demand window | Forecast includes demand already realized | Remaining-window demand is overstated | Capacity remains unnecessarily protected | Timestamp forecasts and condition on elapsed time and bookings |
| Double subtraction of bookings | Remaining capacity drops twice after a confirmation | A cumulative limit is applied as an incremental limit | Legitimate requests are refused | Reconcile total versus remaining capacity with explicit examples |
| Stale channel control | Sales confirms an offer after central closure | Inventory updates lag or fail | Overbooking or inconsistent customer treatment | Monitor synchronization and define a safe fallback |
| Ties handled inconsistently | Adjacent runs alternate at the same threshold | Rounding or undocumented equality convention changes the result | Unstable controls and disputes | Use exact comparisons and a documented tie rule |
| Batch request treated as divisible | A partial offer is issued to an all-or-nothing customer | Unitwise marginal rule ignores order indivisibility | Lost sale or infeasible service | Capture batch terms and evaluate whole-request alternatives |
| Protection mistaken for commitment | Staff tell an urgent customer that protected space is booked | Unused protected units are confused with customer reservations | Misleading promises or capacity disputes | Separate protected inventory, holds, and confirmed bookings in the interface |
| Policy ignores extra resource | High-value freight repeatedly breaches weight or labor limits | Single-unit model misses another bottleneck | Unsafe or infeasible plan | Use conservative capacity or a multi-resource model |
| Target overwrites forecast | Forecast changes to match the budget without new evidence | Management aspiration is substituted for probability | Biased protection and poor decisions | Separate forecast ownership, target, and approved plan |
| Poor cancellation treatment | High-class bookings frequently cancel after the low cutoff | The model assumes every booking consumes capacity | Protected units expire unused | Model cancellation timing and show-up; separately assess overbooking |
| Model learns from its own restrictions | Apparent demand declines immediately after a restrictive policy | The policy suppresses observed bookings | Self-reinforcing underforecast | Preserve request and availability logs; monitor with independent evidence |
| Manual overrides dominate | Most high-value departures bypass the recommendation | The policy does not fit contracts or lacks trust | No measurable policy and hidden costs | Review override reasons and redesign where evidence supports it |
| Price or eligibility disclosure is misleading | Customer-facing terms differ by channel | Product rules are not synchronized with offers | Complaints, legal exposure, and cancellations | Approve terms, monitor channels, and route issues to counsel |
| Forecast drift goes undetected | Tail misses cluster on event days | Historical comparables no longer represent the market | Persistent spill or spoilage | Monitor conditional calibration and recalibrate or pause |
| Network displacement omitted | A profitable local booking blocks a valuable connecting shipment | The model values only one resource | Lower network contribution | Use network resource values and joint feasibility checks |

A single red flag does not prove that the model is wrong.
It identifies a mechanism to investigate with the decision-time evidence.
Conversely, full vehicles can hide significant displacement and do not prove that the policy is optimal.

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | How to test or respond |
|---|---|---|---|
| Capacity is fixed and known | The booking limit subtracts protection from a definite number of positions | Vehicle substitution, breakdown, or an additional charter | Compare capacity versions and model controllable expansion costs |
| Units are interchangeable | Each request consumes one equivalent unit | Weight, cube, temperature, or handling differs | Check utilization of each resource; use a multi-resource model |
| High demand arrives after low demand | The static rule compares an early certain sale with later uncertain demand | Classes arrive interleaved | Inspect request timestamps; use sequential optimization or simulation |
| Low demand fills the unprotected units | The worked-example revenue table treats every low allowance as sold | Fewer advance requests arrive | Model advance demand jointly and recalculate expected contribution |
| Future high demand is independent of the low limit | The same distribution is compared under each protection level | Customers switch products or book earlier | Examine customer journeys and run an appropriately designed policy test |
| Contributions are constant per position | The marginal comparison uses fixed USD 80 and USD 160 | Volume discounts, step costs, or customer-specific service costs | Use incremental contribution at the relevant quantity |
| No cancellations or no-shows | Booked positions equal consumed positions | Customers cancel or fail to appear | Estimate class- and lead-time-specific show-up; evaluate separately from protection |
| Low requests are divisible | Each marginal position can be accepted independently | An order requires several positions together | Model the entire request and its alternatives |
| Forecast distribution is appropriate | Tail probabilities determine each unit's expected value | Seasonality shift, event, or biased history | Backtest conditional distributions and run scenario sensitivity |
| One departure is independent of other resources | The model optimizes a single inventory | Shared equipment, connections, or network commitments | Model joint constraints and resource opportunity costs |
| Service obligations are represented | Expected contribution is a legitimate objective only within constraints | Contract priorities or protected service commitments | Encode hard obligations before economic optimization |
| Unused capacity has zero salvage value | The comparison values an empty position at zero | A late alternative product or transfer has value | Include feasible salvage contribution and its timing |
| All forecast probabilities sum to one | Expected values require a complete distribution | Missing or truncated demand tail | Account for tail mass and review its effect on spill |
| Historical requests are observable or estimable | Calibration needs a demand reference | Closed channels hide lost requests | Collect availability and refusal events; report bounds |

These assumptions define the scope of the example, not requirements that every practical revenue-management system must satisfy.
When one fails, the response may be better data, a different model, a policy constraint, or a decision not to automate.
Forecast uncertainty and uncertainty about the forecast model are distinct.
A precise probability distribution can itself be poorly supported by sparse data.
Show alternative plausible distributions and the resulting control when that uncertainty can change the decision.

## 12. Exercises

### Exercise 1 — Identify the decision

A warehouse has six refrigerated dock appointments for Tuesday morning.
Four standard customers request appointments on Monday.
Urgent customers may request Tuesday appointments later.
The loading team confirms that all six appointments are operationally feasible.

Deliverable: identify the perishable resource, decision time, low and high products, and the information needed before setting a booking limit.
Acceptance criteria: distinguish a request from a commitment, state the expiry, and name at least two operational constraints.
Estimated time: 10 minutes.

### Exercise 2 — Calculate contribution

A standard appointment brings USD 115 in net revenue and USD 35 in incremental handling cost.
An urgent appointment brings USD 220 in net revenue and USD 60 in incremental handling cost.
Both use one equivalent appointment.

Deliverable: compute each contribution and the standard-to-urgent contribution ratio.
Acceptance criteria: show the arithmetic, units, and why gross-revenue ratio would be the wrong threshold.
Estimated time: 10 minutes.

### Exercise 3 — Set a discrete protection level

There are four positions.
A low request yields USD 60 contribution and a later high request yields USD 120 contribution per position.
The future high-demand distribution is: zero with probability 0.10, one with 0.20, two with 0.35, three with 0.25, and four with 0.10.
Enough low requests exist to fill all unprotected positions.
Use the convention that a tied marginal unit is not protected.

Deliverable: calculate each upper-tail probability, the protection level, and the low booking limit.
Acceptance criteria: use \(P(D\ge j)\), show the threshold, and handle any tie explicitly.
Estimated time: 20 minutes.

### Exercise 4 — Compare realized and expected outcomes

Use Exercise 3.
Assume the selected low-class limit is filled and realized high demand is one position.

Deliverable: calculate realized contribution and unused capacity, and explain why this one outcome does not prove the rule was poor.
Acceptance criteria: identify accepted low and high positions and distinguish the observed outcome from expected performance.
Estimated time: 15 minutes.

### Exercise 5 — Update after a commitment

Use the worked example's five positions and two-position protection level.
One low position has already been confirmed and no high position has been booked.
Assume the same remaining high-demand distribution still applies.

Deliverable: calculate remaining capacity, additional low allowance, and cumulative low limit.
Acceptance criteria: do not subtract the confirmed position twice and do not cancel it.
Estimated time: 10 minutes.

### Exercise 6 — Diagnose censored demand

For eight comparable departures, the urgent product was closed early on six departures.
The analyst uses only accepted urgent bookings and reports a sharp reduction in urgent demand.
No refusal log is available.

Deliverable: write a short critique and a plan for improving the evidence.
Acceptance criteria: explain censoring, distinguish an inquiry from an executable request, and propose at least two data or evaluation improvements without asserting an unobserved demand total.
Estimated time: 20 minutes.

### Exercise 7 — Design an operational control

A carrier sells through its own portal, a sales desk, and a partner interface.
The partner interface updates availability every 30 minutes.
Capacity can change after a vehicle substitution.

Deliverable: design a control, exception workflow, and audit record for the three channels.
Acceptance criteria: state the capacity owner, synchronization fallback, approval and reason for overrides, treatment of existing bookings, and at least three monitoring measures.
Estimated time: 30 minutes.

### Exercise 8 — Evaluate a policy

A manager proposes replacing first-come-first-served with the worked example's protection policy on a set of comparable departures.
Demand is seasonal, and a simultaneous sales campaign may change the request mix.

Deliverable: design an evaluation with an outcome, comparison, assignment or identification strategy, guardrails, and limitations.
Acceptance criteria: measure net contribution and service outcomes, address seasonality and concurrent changes, avoid counting unobserved demand as certain sales, and explain how to handle limited sample size.
Estimated time: 35 minutes.

## 13. Self-check answers

### Exercise 1

The perishable resource is a refrigerated dock appointment for Tuesday morning.
Its commercial value expires when the relevant booking or loading deadline passes.
The decision is made when standard requests are considered on Monday, not after urgent demand is known.
Standard and urgent are product classes only if their terms, contribution, and request timing are defined.
The four standard requests are not commitments until accepted.
Relevant operational checks include staffing, cold-chain compatibility, door availability, handling duration, and the actual cutoff.
The distribution of remaining urgent requests and both products' incremental contributions are also needed.

### Exercise 2

Standard contribution is USD 115 − USD 35 = USD 80 per appointment.
Urgent contribution is USD 220 − USD 60 = USD 160 per appointment.
The threshold ratio is USD 80 / USD 160 = 0.50.
The gross-revenue ratio would be 115 / 220, approximately 0.523, and ignores the different incremental costs.
It would compare different economic quantities from the expected contribution being protected.

### Exercise 3

The threshold is USD 60 / USD 120 = 0.50.
The upper tails are \(P(D\ge1)=0.90\), \(P(D\ge2)=0.70\), \(P(D\ge3)=0.35\), and \(P(D\ge4)=0.10\).
The first two marginal values exceed USD 60: USD 108 and USD 84.
The third and fourth values are USD 42 and USD 12, below USD 60.
Protect two positions and set the low booking limit to \(4-2=2\) positions.
There is no equality at the threshold in this distribution.

### Exercise 4

Two low positions are sold for \(2\times\) USD 60 = USD 120.
One high position is sold for USD 120.
Realized contribution is USD 240, and one of the four positions remains unused.
A policy chosen under uncertainty is judged over comparable repeated decisions and against feasible alternatives, not by one realization.
In particular, the held position was valuable in higher-demand scenarios that did not occur on this departure.

### Exercise 5

One accepted low booking leaves \(5-1=4\) positions.
Protecting two of the remaining positions leaves an additional low allowance of \(4-2=2\).
The cumulative low limit is the one accepted position plus two additional positions, or three.
This answer depends on the stated assumption that the same remaining-demand distribution still applies.

### Exercise 6 rubric

A strong answer says accepted bookings are censored by the closure and cannot alone establish a fall in underlying urgent demand.
It proposes recording product availability and timestamped eligible requests or refusals, with deduplication and a reason code.
It also proposes comparing periods with different availability or testing an unconstraining approach with sensitivity bounds.
A quote view or repeated inquiry is not automatically a counterfactual confirmed sale.
A weak answer invents the missing number of requests or treats zero accepted bookings as zero demand.

### Exercise 7 rubric

A strong answer names operations as the authoritative capacity owner and reconciles each capacity version with confirmed bookings.
It describes how the partner's update delay can produce stale availability and supplies a safe channel allocation, pause, or manual approval fallback.
It specifies authorized overrides, expiration, reason, and a timestamped audit trail.
It preserves existing commitments and routes infeasible ones to the disruption process.
Useful measures include channel lag, oversells, refusal rates, spoilage, contribution, and service failures.
A weak answer simply publishes a mathematical limit without an enforcement or recovery process.

### Exercise 8 rubric

A strong answer defines contribution per comparable departure and service guardrails before launch.
Where feasible, it randomizes comparable departure groups or time blocks to policies and controls for seasonal composition.
It either holds the sales campaign constant across groups or accounts for its exposure and limits causal claims.
It logs eligible requests, availability, actual bookings, capacity, and costs as of each decision.
It reports uncertainty and sample size limitations rather than declaring a winner from a few departures.
A weak answer compares raw revenue before and after the change while ignoring the campaign and service outcomes.

## 14. Interview questions

### 1. Why can accepting a profitable booking reduce expected contribution?

A strong answer explains displacement of a potentially more valuable future booking and compares incremental contribution with the expected marginal value of capacity.
A weak answer assumes any positive contribution must be accepted regardless of scarcity.

### 2. What makes capacity perishable?

A strong answer identifies a specific service opportunity and cutoff after which the unused unit cannot be sold for that departure.
A weak answer confuses the capacity's expiry with physical product shelf life.

### 3. What is the difference between a booking limit and a protection level?

A strong answer describes the low-class maximum and the capacity retained for higher-class demand, including their capacity basis.
A weak answer calls the protected quantity a confirmed reservation.

### 4. How do you handle discrete demand at the Littlewood threshold?

A strong answer computes \(r_H P(D\ge j)\) for each marginal unit and states a tie rule.
A weak answer applies an unspecified continuous quantile and rounds without examining a probability mass point.

### 5. Why might a forecast based on accepted bookings be biased?

A strong answer explains censoring when products or capacity were unavailable and distinguishes blocked requests from realized sales.
A weak answer assumes historical accepted bookings always equal underlying customer demand.

### 6. How do cancellations affect the decision?

A strong answer separates protection for future demand from overbooking based on uncertain show-up and includes cancellation timing and denied-service consequences.
A weak answer adds the expected cancellation count to capacity without a risk or contractual analysis.

### 7. How do you update a limit after bookings are accepted?

A strong answer reconciles remaining capacity, forecasts demand remaining in the new window, and protects existing commitments.
A weak answer reapplies the original total-capacity formula to remaining capacity and subtracts bookings twice.

### 8. When is the single-resource model inadequate?

A strong answer gives an example of weight and volume or multiple network legs, then compares contribution with the combined scarce-resource opportunity cost.
A weak answer assumes every shipment consumes one interchangeable position.

### 9. How would you evaluate the policy?

A strong answer proposes a defensible comparison, net contribution, service outcomes, logging, and uncertainty.
A weak answer uses utilization alone or counts every rejected inquiry as a guaranteed lost sale.

### 10. What happens if a channel receives a stale limit?

A strong answer identifies synchronization monitoring, a safe fallback, an escalation owner, and the treatment of already confirmed bookings.
A weak answer assumes the mathematical recommendation is automatically enforced everywhere.

### 11. How can a product fence go wrong?

A strong answer discusses customer substitution, misleading terms, and potential legal or fairness concerns with eligibility rules.
A weak answer assumes any customer segmentation is both effective and lawful.

### 12. Why is a target not a forecast?

A strong answer distinguishes the desired business outcome from a probability-based prediction and explains why changing the forecast to reach the target corrupts the control.
A weak answer uses the budget as the expected value without supporting evidence.

## 15. Key terms

- [Booking curve](#4-definitions-and-vocabulary) — cumulative booking pattern by time before service.
- [Booking horizon](#4-definitions-and-vocabulary) — time remaining until the service decision expires.
- [Booking limit](#4-definitions-and-vocabulary) — maximum accepted low-class quantity on the stated basis.
- [Capacity](#4-definitions-and-vocabulary) — feasible units of the constrained service resource.
- [Censoring](#4-definitions-and-vocabulary) — incomplete observation when demand is blocked or availability is restricted.
- [Commitment](#4-definitions-and-vocabulary) — accepted booking subject to its terms.
- [Contribution](#4-definitions-and-vocabulary) — net revenue less decision-relevant incremental costs.
- [Counterfactual](#4-definitions-and-vocabulary) — outcome that would occur under an alternative policy.
- [Cutoff](#4-definitions-and-vocabulary) — latest eligible booking or operating time.
- [Demand distribution](#4-definitions-and-vocabulary) — probabilities assigned to possible future demand counts.
- [Displacement](#4-definitions-and-vocabulary) — loss of a feasible alternative use of capacity.
- [Forecast](#4-definitions-and-vocabulary) — prediction based on information available at a stated time.
- [High class](#4-definitions-and-vocabulary) — later, higher-contribution product in the simplified model.
- [Littlewood's rule](#5-the-mechanics-step-by-step) — comparison of early contribution with expected later marginal value.
- [Low class](#4-definitions-and-vocabulary) — earlier, lower-contribution product in the simplified model.
- [Nested control](#4-definitions-and-vocabulary) — availability rule that protects higher classes without rigidly excluding them from unused lower-class capacity.
- [No-show](#4-definitions-and-vocabulary) — booked customer does not consume the reserved unit.
- [Opportunity cost](#4-definitions-and-vocabulary) — value of the best displaced alternative.
- [Overbooking](#4-definitions-and-vocabulary) — acceptance beyond physical capacity based on uncertain consumption.
- [Perishability](#4-definitions-and-vocabulary) — loss of the capacity-sale opportunity after the service deadline.
- [Plan](#4-definitions-and-vocabulary) — intended operating control or allocation.
- [Product class](#4-definitions-and-vocabulary) — offer with a defined price and service conditions.
- [Protection level](#4-definitions-and-vocabulary) — units held for potentially higher-value demand.
- [Remaining capacity](#4-definitions-and-vocabulary) — sellable capacity after existing commitments and blocks.
- [Segment](#4-definitions-and-vocabulary) — group or use case with relevant differences in requirements or behavior.
- [Spill](#4-definitions-and-vocabulary) — demand not served because of unavailability.
- [Spoilage](#4-definitions-and-vocabulary) — sellable capacity expiring unused.
- [Survival probability](#4-definitions-and-vocabulary) — probability demand reaches a specified count.
- [Target](#4-definitions-and-vocabulary) — desired outcome, distinct from a forecast.
- [Tie convention](#6-worked-example) — predetermined treatment of equal marginal expected values.

## 16. Sources

1. Littlewood, Ken. “Forecasting and Control of Passenger Bookings.” *Proceedings of the 12th Annual AGIFORS Symposium*, 1972. Foundational two-class booking-control formulation; see the reprint in *Journal of Revenue and Pricing Management* 4, 111–123 (2005), [https://doi.org/10.1057/palgrave.rpm.5170129](https://doi.org/10.1057/palgrave.rpm.5170129).
2. Belobaba, Peter P. “Application of a Probabilistic Decision Model to Airline Seat Inventory Control.” *Operations Research* 37, no. 2 (1989), 183–197. [https://doi.org/10.1287/opre.37.2.183](https://doi.org/10.1287/opre.37.2.183). Multiple-class airline inventory control and expected marginal seat revenue.
3. Smith, Barry C., John F. Leimkuhler, and Ross M. Darrow. “Yield Management at American Airlines.” *Interfaces* 22, no. 1 (1992), 8–31. [https://doi.org/10.1287/inte.22.1.8](https://doi.org/10.1287/inte.22.1.8). Historical named-company account.
4. Talluri, Kalyan T., and Garrett J. van Ryzin. *The Theory and Practice of Revenue Management*. Springer, 2004. [https://doi.org/10.1007/b139000](https://doi.org/10.1007/b139000). Broader treatment of capacity, demand, and network controls.
5. McGill, Jeffrey I., and Garrett J. van Ryzin. “Revenue Management: Research Overview and Prospects.” *Transportation Science* 33, no. 2 (1999), 233–256. [https://doi.org/10.1287/trsc.33.2.233](https://doi.org/10.1287/trsc.33.2.233). Research overview and modeling limitations.
6. Kimes, Sheryl E. “The Basics of Yield Management.” *Cornell Hotel and Restaurant Administration Quarterly* 30, no. 3 (1989), 14–19. [https://doi.org/10.1177/001088048903000314](https://doi.org/10.1177/001088048903000314). Hotel inventory and yield-management context.
7. European Parliament and Council. Regulation (EU) 2016/679, General Data Protection Regulation, especially Article 5. [Official EUR-Lex text](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Binding scope and obligations depend on the processing and jurisdiction.
8. U.S. Federal Trade Commission. [Rule on Unfair or Deceptive Fees, 16 CFR Part 464](https://www.ecfr.gov/current/title-16/chapter-I/subchapter-D/part-464). Consult current scope and text; it is not a universal freight-pricing rule.
9. GS1. [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications). Reference for applicable identification and data-capture standards, not for booking-control formulas.

The mathematical example and its numeric inputs are instructional assumptions, not reported results from these sources.
Regulatory references are educational and should be checked for current applicability with qualified counsel.

## 17. Links

- [Module 16 topic index](_index.md)
- [Module 16 survey and lab sections](../../../course/16_pricing_promotions_revenue_management.md)
- [Pricing decision stack](01_pricing-decision-stack.md)
- [Markdown optimization with salvage](05_markdown-optimization.md)
- [Price, value, and willingness to pay](02_value-wtp.md)
