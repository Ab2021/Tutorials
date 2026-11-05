# Shipping Cost and Carrier Selection

> Module 21 — E-commerce Supply Chain Analytics  
> Topic 06 of 10  
> Estimated study time: 4 hours  
> Prerequisites: fulfillment models, inventory placement, and delivery promise  
> Version 2026.5

## 1. Why this topic exists

A checkout promise becomes a physical and financial obligation when an order is released.
The merchant must select a service that can accept the parcel, make the promise, and leave enough contribution margin.
The cheapest published base rate is rarely sufficient evidence for that decision.
A rate card may exclude fuel, residential delivery, peak charges, minimums, and dimension adjustments.
Some discounts are conditional on volume earned over a prior period.
A carrier may offer an attractive rate but cap pickup volume on the very day the promotion launches.

The decision has at least three clocks.
Procurement negotiates contracts months before a peak.
Network planning books linehaul and capacity days or weeks before dispatch.
A routing system selects a carrier and service for each eligible parcel at label time.
These decisions interact, but they are not interchangeable.
A rate that requires a quarterly volume commitment cannot be optimized independently for every package.

The transportation manager owns the carrier portfolio and operational escalation.
Procurement owns commercial negotiation, with legal approval of contract language.
The fulfillment manager owns pack accuracy, induction cutoffs, and handover execution.
The promise owner sets customer-facing service requirements.
Finance approves the economic baseline and reconciles savings to invoices.
The data team builds reproducible estimates but does not unilaterally accept service or contractual risk.

This topic provides a method for comparing carrier choices at the same parcel grain.
It also explains when consolidating parcels into a linehaul movement and injecting them nearer the destination can improve economics.
You will calculate billable weight, compare service mixes, assess capacity and uncertainty, and build a negotiation and control plan.
All commercial figures in the examples are illustrative, not actual carrier quotations.

## 2. Learning outcomes

After completing this topic, you can:

1. Translate a rate card and its exclusions into an auditable parcel-level expected cost; see exercises 1 and 2.
2. Calculate dimensional and billable weight using a stated rounding sequence; see exercise 1.
3. Choose an eligible service mix under promise and capacity constraints; see exercise 3.
4. Compute the break-even volume for zone skipping and explain its service risks; see exercise 4.
5. Build a finance-ready business case with forecast, plan, target, and commitment separated; see exercise 5.
6. Design a negotiation package and a contract control that protects net value; see exercise 6.
7. Diagnose invoice and performance exceptions using defensible denominators; see exercises 7 and 8.
8. Explain how evidence, uncertainty, and ownership affect the decision; see exercises 5–8.

## 3. The core mental model

A carrier is not a single price.
It is a capacity-constrained service with a contract, an operating footprint, and a stream of later invoices and events.
The parcel and its customer promise determine which parts of that contract are relevant.

```text
Order and promise
       |
       v
Pack -> measured dimensions, actual weight, destination, handover time
       |
       v
Eligibility: service area, cutoff, package limits, capacity, contract
       |
       v
Rate: zone + billable weight + discounts + minimums + accessorials
       |
       +--> Direct induction ------------------------+
       |                                              |
       +--> Consolidate -> linehaul -> destination injection
                                                      |
       v                                              v
Allocate service mix under promise and commitments -> dispatch
       |
       v
Track handover and delivery -> audit invoice -> learn and renegotiate
```

The comparison should use the same set of parcels, service window, currency, and time period.
A carrier's average invoice cost is not automatically comparable with another's if one receives heavier or more remote parcels.
Re-rate the same shipment population under each feasible offer before attributing a difference to price.
Then add costs that are outside the parcel invoice, including linehaul, sort, packaging changes, and implementation where appropriate.

A parcel can be cheap on paper and expensive in operation.
Missing a cutoff may create an extra day of transit or require an upgrade.
A damaged barcode can create manual rework and delay.
A minimum-volume shortfall can reverse an apparent discount.
For this reason, the routing objective must include hard feasibility constraints and explicitly valued risks rather than only base postage.

Zone skipping changes the injection point, not the physical destination.
The merchant consolidates multiple parcels, pays for a trunk movement to a destination-region facility, and then buys last-mile service from that facility.
The lower local parcel rate must cover linehaul and handling, including the cost of unused space.
Waiting to fill the trailer can erode the delivery promise.

A forecast is an estimate of future parcels.
A plan assigns expected parcels to resources.
A target is the performance or savings the business wants.
A commitment is an obligation accepted by a carrier or merchant in a contract or booking.
Treating a forecast as a carrier commitment makes a capacity model look safer than it is.

## 4. Definitions and vocabulary

The contract, not a generic definition, determines actual rating rules.
Weights below are in pounds (lb), dimensions in inches (in), and money in United States dollars (USD) unless stated otherwise.

| Term | Working definition and unit | Common misuse |
|---|---|---|
| Parcel | One physically labeled handling unit | Calling an order with two cartons one parcel |
| Order | Customer purchase that can produce one or more parcels | Dividing carrier spend by orders and calling it cost per parcel |
| Origin | Contractually rated induction location | Assuming it is always the warehouse street address |
| Destination zone | Carrier-defined origin–destination rate band | Assuming zone numbers match across carriers |
| Service | Carrier product with specified network and terms | Treating a brand name as a delivery guarantee |
| Rate card | Table of prices and conditions for shipment attributes | Treating the table as the final invoice |
| Base charge | Rate before applicable adjustments, USD/parcel | Assuming it includes all accessorials |
| Discount | Contractual reduction applied to a defined charge base | Applying it to every surcharge without authority |
| Minimum charge | Lowest payable amount under a specified rule, USD/parcel | Applying the minimum before a discount when terms say otherwise |
| Accessorial | Charge for a condition or optional service, USD/parcel or percentage | Omitting rare but material exceptions |
| Fuel surcharge | Variable percentage or amount on a defined charge base | Applying a percentage to the entire invoice by default |
| Residential surcharge | Charge for a carrier-defined residential destination | Classifying an address from customer name alone |
| Peak surcharge | Time-bounded additional charge under published or negotiated terms | Applying one peak window to all services |
| Dimensional weight | Volume divided by a contractual divisor, converted to a rating weight | Assuming it is the physical scale weight |
| Dimensional divisor | Cubic inches per lb in this example | Assuming a divisor is universal or does not change |
| Billable weight | Weight used for rating after contractual rounding and comparison | Taking the lower of actual and dimensional weight |
| Cubic volume | Length × width × height, cubic inches | Multiplying unrounded dimensions when the contract rounds each side first |
| Additional handling | Charge for a package characteristic defined in a tariff | Assuming all oversize parcels trigger exactly the same fee |
| Oversize | Package exceeding a carrier-defined size threshold | Assuming oversize and non-conveyable mean the same thing |
| Chargeable dimensions | Dimensions after applicable measuring and rounding rules, in | Using nominal product dimensions as carton measurements |
| Zone skipping | Consolidated linehaul followed by downstream carrier injection | Assuming the carrier automatically grants a local zone |
| Injection | Transfer of parcels into a carrier network at an agreed facility | Treating warehouse manifesting as a carrier acceptance scan |
| Linehaul | Movement between facilities, often priced per trip or mile | Counting only the variable fuel cost of a booked truck |
| Induction cutoff | Latest accepted handover time for a stated processing day | Confusing warehouse label creation with acceptance |
| Capacity | Available parcel or transport units per time bucket | Treating an annual average as a daily commitment |
| Service mix | Distribution of parcels among eligible carrier services | Reporting carrier shares without service-level detail |
| On-time delivery | Deliveries meeting a defined deadline divided by an appropriate cohort | Measuring only delivered parcels and hiding outstanding ones |
| Promise date | Date or timestamp communicated to the customer | Substituting carrier's estimated transit time for the customer promise |
| Billing adjustment | Later change to a charge from carrier measurement or terms | Treating every adjustment as an error |
| Invoice audit | Matching billed charges to expected charges and evidence | Disputing every difference without checking contract version |
| Commitment | Contractual or booked obligation with stated conditions | Equating sales forecasts with accepted capacity |
| Contribution margin | Revenue less defined variable costs, USD/order | Counting a transfer of costs between accounts as savings |
| SLA | Service-level agreement specifying measures and remedies | Assuming a service credit makes the customer whole |
| Tender | Offer of a parcel or batch to a carrier for pickup | Assuming a tender is accepted without confirmation |
| Zone-skip break-even | Volume where direct and consolidated total costs are equal | Ignoring fixed trip and sort costs |

## 5. The mechanics, step by step

### 5.1 Establish the decision and horizon

1. Define the choice: parcel-level routing, seasonal allocation, contract award, packaging change, or zone-skip investment.
2. State the decision owner and the date by which a decision must be made.
3. Identify the affected origin nodes, destination regions, services, package types, and operating days.
4. Select the comparison horizon, such as 13 weeks including a peak period.
5. Specify whether the economic measure is cost per parcel, cost per order, total landed fulfillment cost, or incremental contribution.
6. Set the customer promise and service constraints before minimizing cost.
7. Record what can be changed during the horizon and what is already committed.

Input: decision brief, operating calendar, and promise policy.
Output: a documented scope and a list of decisions that must be made together.
Branch: if packaging is fixed in the short term, compare carrier options on existing cartons; evaluate redesign separately.

### 5.2 Construct the parcel population

1. Obtain shipment records at parcel ID grain, not only order grain.
2. Retain order ID so that split shipments and order-level economics remain measurable.
3. Include origin, destination geography, service, pickup date, actual weight, carton dimensions, residential flag, and delivery outcome where permitted.
4. Attach all invoice lines, adjustments, credits, and payment status to the shipment or a documented unallocated bucket.
5. Preserve the raw carrier values alongside normalized fields and source timestamps.
6. Remove duplicates only with a defined key and a documented rule for corrected records.
7. Reconcile parcel counts to manifests, carrier acceptance records, and finance totals.
8. Report unmatched parcels and unallocated invoice dollars rather than silently dropping them.
9. Separate canceled labels from parcels actually tendered, and check whether voided labels were charged.
10. Track missing dimensions by node and carton type; missingness can be systematic.
11. Use an appropriate historical window and explicitly adjust for known changes in demand, origin, packaging, and carrier coverage.
12. Protect address data and avoid moving full addresses into an unrestricted analysis file.

Input: warehouse management, order, carrier event, and invoice data.
Output: a reconciled parcel cohort and a data-quality exception register.
Branch: if billed parcel IDs cannot be matched, estimate a sensitivity range and prioritize the reconciliation before declaring savings.

### 5.3 Normalize contract terms

1. Capture effective dates and the document hierarchy: agreement, amendment, service guide, rate schedule, and tariff as applicable.
2. Record the carrier's zone mapping by origin, destination, service, and effective date.
3. Capture rate tables at the stated weight increments and service levels.
4. Encode discounts by charge category, eligible service, timing, and volume tier.
5. Encode the order of operations for discounts, minimum charges, fuel, and taxes.
6. Record residential, remote, delivery-area, Saturday, signature, oversize, and additional-handling rules if applicable.
7. Record measurement rules, divisors, rounding, and the possibility of carrier remeasurement.
8. Record excluded lanes, package maximums, declared-value rules, claims terms, and prohibited goods.
9. Include peak windows and explicit rights to amend charges.
10. Capture invoice-level minimum spend, rebates, caps, and true-ups outside the per-parcel rating function.
11. Have procurement and legal confirm interpretations before using the model for an award.
12. Store a versioned rule identifier so every rated parcel can be reproduced later.

Input: executed contract and applicable incorporated schedules.
Output: a versioned rating specification with examples for each rule.
Branch: where the contract is ambiguous, obtain written clarification; do not silently choose the favorable interpretation.

### 5.4 Compute dimensional and billable weight

For one illustrative rule, round each side up to the next whole inch, multiply the rounded dimensions, divide by 139 cubic inches per lb, and round the result up to the next whole lb.
Round actual weight up to the next whole lb, then take the greater of the rounded actual and dimensional weights.
A real agreement may use different units, divisors, dimension rounding, thresholds, or measurement methods.

1. Measure the packed parcel, including protrusions as required by the applicable rules.
2. Apply the carrier's dimension rounding to each side.
3. Multiply the rounded sides to obtain chargeable cubic inches.
4. Divide by the applicable divisor.
5. Apply the specified dimensional-weight rounding.
6. Round actual weight as the contract requires.
7. Take the greater weight where the contract specifies that comparison.
8. Look up the resulting weight band; do not interpolate unless the schedule permits it.
9. Evaluate size limits and accessorial triggers independently from the weight calculation.
10. Retain raw dimensions, transformed dimensions, divisor, and result for audit.

Input: measured parcel dimensions and scale weight plus the contract rule.
Output: billable weight and a calculation trace for each carrier service.
Branch: if the parcel exceeds a service maximum, mark the service ineligible instead of inventing a rate.

### 5.5 Build all-in direct cost

1. Map origin and destination to that carrier's zone.
2. Look up the base charge for service, zone, and billable weight.
3. Apply the negotiated base discount to the eligible charge base.
4. Apply the minimum in the order required by the agreement.
5. Determine applicable residential, remote, handling, signature, and other fixed accessorials.
6. Apply the fuel percentage to the specified eligible subtotal, not necessarily to all charges.
7. Add peak or other time-dependent charges under their own effective dates.
8. Add applicable taxes only where they are a real incremental cost for the comparison.
9. Add merchant-side pickup, sorting, packaging, and operational costs not already in the invoice.
10. Apply rebates or volume true-ups at the proper aggregation level and avoid double counting.
11. Produce a component-level trace and preserve the rate version used.
12. Compare the estimate to actual invoices for a holdout period when such data exists.

Input: parcel characteristics, effective contract, and merchant-side cost assumptions.
Output: expected all-in cost per eligible parcel and service.
Branch: for a conditional volume discount, calculate allocation and discount tier together; a stand-alone marginal rate can be misleading.

### 5.6 Screen service feasibility

1. Identify the customer promise timestamp and the warehouse release timestamp in their correct time zones.
2. Model pick, pack, label, staging, and handover time before transit begins.
3. Apply carrier pickup schedules, origin cutoffs, operating calendars, and destination coverage.
4. Check package limits, restricted goods, insurance, and customer-specific requirements.
5. Estimate the probability of delivery by the promise for the relevant lane and induction day.
6. Base the service measure on complete cohorts, including undelivered or late parcels once the measurement window closes.
7. Separate carrier-controlled transit from merchant lateness without hiding the customer's end-to-end outcome.
8. Require enough observations or use wider uncertainty intervals and a conservative fallback.
9. Check available daily capacity, including already-booked volume and surge rules.
10. Reject infeasible options before comparing their prices.

Input: promise, parcel, operating calendar, carrier events, and committed capacity.
Output: feasible services and their estimated risk for each parcel or segment.
Branch: if no option meets the promise, escalate to the promise owner; do not present an infeasible low rate as a valid selection.

### 5.7 Allocate a service mix

Let x[p,s] equal 1 when parcel p is assigned to service s and 0 otherwise.
Minimize the sum of expected costs multiplied by x[p,s], plus any explicitly modeled fixed trip or shortfall costs.
Each parcel must receive one eligible service; each service's daily assignment must not exceed its available capacity.
Add contractual minimums, customer service constraints, and regional diversification constraints where they actually apply.
Use linear or mixed-integer optimization when the business case warrants it; a transparent ranked allocation can be sufficient for smaller problems.

1. Partition by relevant day, origin, destination, package, and promise.
2. Estimate feasible volume and cost for each service.
3. Set capacity and eligibility constraints from accepted bookings and contracts, not sales claims.
4. Add minimum volume obligations and model the cost of failing them.
5. Reserve contingency for uncertainty if the owner approves it.
6. Solve or allocate by incremental cost, respecting each constraint.
7. Recompute tier discounts and invoice-level minimums on the resulting allocation.
8. Stress-test forecast volume, late release, weather, and carrier capacity loss.
9. Review the allocation with operations, procurement, finance, and the promise owner.
10. Publish a plan with named fallbacks and an effective period.

Input: feasible alternatives, expected cost, capacities, and contract commitments.
Output: a dated allocation plan and an exception path.
Branch: if the allocation violates a carrier minimum or an operational constraint, revise the whole mix rather than manually hiding the deficit.

### 5.8 Evaluate zone skipping

1. Choose a destination region with enough geographically concentrated volume and a carrier willing to accept the intended injection.
2. Confirm the physical injection location, acceptance window, labeling format, service eligibility, and negotiated local rates.
3. Estimate parcels per departure and the distribution of load, not just its average.
4. Price linehaul including empty capacity, minimum trip charge, fuel, waiting, and any return movement that is incremental.
5. Price origin sort, destination handling, equipment, and operational labor.
6. Rate each parcel from the injection location using the agreed zone and service.
7. Add the time from origin release through sort, departure, linehaul, and destination acceptance.
8. Check whether the remaining last-mile time still meets the customer promise.
9. Compare against direct induction on the same parcel cohort and delivery days.
10. Calculate break-even volume as fixed incremental trip cost divided by direct minus skipped variable cost per parcel, when that denominator is positive.
11. Test low-volume days, missed departures, trailer breakdown, and diversion to direct service.
12. Obtain a written operational acceptance and capacity commitment before launch.

Input: direct and local rates, departure-level volume, full linehaul cost, handling, and service timelines.
Output: a lane-specific business case, break-even threshold, and operating plan.
Branch: if skipped variable cost is at least direct variable cost, a positive fixed trip cost cannot be recovered by increasing volume under this simple model.

### 5.9 Negotiate and implement

1. Share a defensible shipment profile: volume distribution, lane mix, weight, dimensions, service, seasonality, and growth uncertainty.
2. Ask carriers to quote the same baseline cohort and disclose exclusions and surcharge treatment.
3. Request capacity and cutoff commitments by origin, day, and peak period, including remedies and exceptions.
4. Model discounts, minimums, annual increases, and fuel on the expected mix, not on the headline discount.
5. Negotiate the terms that drive total value, including dimensional divisor, accessorial definitions, residential eligibility, and billing disputes.
6. Check the marginal economics of each volume tier and model the risk of falling below it.
7. Require change notification, versioned schedules, and an auditable invoice detail format.
8. Include test shipments, label validation, event integration, claims process, and operational escalation in the implementation plan.
9. Run a controlled pilot on agreed lanes and compare delivery cohorts and invoices with the approved baseline.
10. Expand only after the decision owner accepts the evidence and residual risk.

Input: bid package, comparable carrier offers, and operational constraints.
Output: negotiated terms, signed commitments, and launch gates.
Branch: if the best-priced proposal cannot commit adequate capacity, allocate only the confirmed amount and price the remainder explicitly.

### 5.10 Operate, audit, and improve

1. Monitor daily accepted versus planned volume and unused committed capacity.
2. Alert before pickup cutoff if tenders or carrier acceptance are missing.
3. Reconcile delivery performance on cohorts with a defined observation window.
4. Rate invoices against the correct contract version and measurement evidence.
5. Investigate exceptions by cause, amount, age, and owner.
6. Submit disputes inside the contract's deadline and track accepted, rejected, and credited amounts.
7. Compare realized savings with the same baseline mix and adjust for volume or network changes.
8. Review accessorial trends and packaging opportunities with fulfillment.
9. Review service mix, forecast accuracy, capacity commitments, and failure drills at least each planning cycle.
10. Feed confirmed corrections into contract data, packaging practices, or routing logic with an approval trail.

Input: operational events, invoices, contract versions, and dispute outcomes.
Output: controlled payments, service improvement actions, and updated assumptions.

### 5.11 Diagnose the economics of packaging

The transportation team does not control the product's mass, but it can work with packaging and fulfillment to change the dimensions of the packed unit.
The correct object of analysis is the packed shipment that actually leaves the facility.
Catalog dimensions can omit void fill, labels, protective inserts, or irregular protrusions.

1. Rank carton and product combinations by dimensional-weight premium.
2. Define the premium as billed weight minus rounded actual weight, in lb per parcel.
3. Multiply the premium by shipment count to identify where small reductions can affect the most parcels.
4. Re-rate candidate carton dimensions under every eligible carrier's rule.
5. Include discontinuities at whole-inch dimension and whole-pound weight boundaries.
6. Check that a smaller carton still meets protection and handling requirements.
7. Include material, labor, damage, return, and equipment consequences.
8. Pilot the packaging change on representative products and destinations.
9. Measure actual dimensions after packing, not only the empty-carton specification.
10. Compare the resulting invoice adjustments with the original baseline.
11. Obtain packaging engineering approval before making a production change.
12. Keep the carrier decision and packaging decision linked in the business case.

For example, a 16 in × 11 in × 7 in box has 1,232 cubic in of chargeable volume.
At a divisor of 139 cubic in/lb, the dimensional weight is 9 lb after rounding.
If an approved design reduces one rounded side from 7 in to 6 in, the chargeable volume is 1,056 cubic in.
The dimensional weight then rounds from 7.597 lb to 8 lb.
The billed weight falls by 1 lb only if rounded actual weight is no more than 8 lb and no other rule overrides the result.
The financial gain is the difference between the applicable 9 lb and 8 lb all-in rates, less incremental packaging and damage costs.
Do not value the reduction as 1 lb multiplied by a generic price per lb when the rate schedule is stepped.

The packaging owner must decide whether the design protects the product.
The transportation analyst supplies the re-rating and variance evidence.
Finance determines whether the cost reduction is incremental rather than a transfer to a different budget.
Customer care and returns teams should be included if damaged-delivery risk changes.

### 5.12 Translate a promise into dispatch feasibility

The customer sees a delivery deadline, not a carrier's average transit time.
Start with that deadline and work backward through the network.

1. Identify the promise timestamp, including local time and any stated end-of-day convention.
2. Identify the destination calendar, including weekends and applicable holidays.
3. Identify the service's valid delivery days and geographical coverage.
4. Determine the latest feasible carrier acceptance time.
5. Subtract origin staging, pack, and pick times to find the latest feasible release time.
6. For zone skipping, subtract destination acceptance, linehaul, sort, and departure preparation time.
7. Compare the resulting latest release with the actual release distribution.
8. Quantify uncertainty in both origin processing and transit.
9. Evaluate whether an operationally feasible service meets the required confidence level.
10. Reserve a more expensive fallback if a late release can still be recovered.
11. Route exceptions to a named person or system action before the relevant cutoff.
12. Retain the promise version shown to the customer for later measurement.

For a simplified timeline, assume a promised delivery by Friday end of day.
A local carrier requires acceptance by 00:00 Thursday for its planned Friday delivery.
If the trunk drive takes six hours and a one-hour buffer is approved, the vehicle must leave by 17:00 Wednesday.
If sorting and staging take two hours, the origin must release the parcel by 15:00 Wednesday.
A label printed at 16:00 Wednesday is not enough to meet that modeled path.
Whether another service can recover the promise depends on its own cutoff and actual capacity.

The above time budget is a plan, not a statistical guarantee.
Use lane-specific actuals to determine whether the assumed buffer is adequate.
If the observed distribution has a long tail, a mean transit time can hide substantial late-delivery risk.
Do not add independently estimated percentile times and claim the result is the same percentile for the whole journey without a justified joint model.
Weather and facility incidents can affect several stages at once.

### 5.13 Price a volume commitment

Carrier discounts can depend on parcels, eligible revenue, or spend over a defined period.
A tier may apply only to incremental volume, to all eligible volume, or through a later rebate.
The arithmetic must reflect the actual agreement.

Suppose a hypothetical carrier offers USD 8.50 per parcel for all 10,000 monthly parcels if the merchant tenders at least 10,000.
Below that threshold, the rate is USD 8.80 per parcel for all tendered parcels.
At 10,000 parcels, the monthly charge is USD 85,000.
At 9,999 parcels, the monthly charge is USD 87,991.20.
Under these unusual all-volume terms, one fewer parcel increases total spend by USD 2,991.20.
This is a tier cliff, not a normal marginal per-parcel cost.

The example does not mean a merchant should ship an unnecessary parcel.
The correct response is to model realistic routing alternatives, contractual eligibility, and commercial terms.
Check whether the contract has an explicit shortfall fee, a grace band, excluded services, or a rebate instead.
If volume is uncertain, evaluate the probability and financial impact of missing the threshold.
Do not multiply an average price by a forecast and ignore the discontinuity.

A practical negotiation package includes a table for low, base, and high volume.
For each case, report eligible volume, applicable tier, total carrier charge, capacity used, and cost of fallback.
Record whether unused capacity has a take-or-pay cost.
Separate a carrier's operational acceptance commitment from the merchant's commercial volume commitment.
The two obligations can have different measurement periods and remedies.

### 5.14 Assess a fuel formula

Fuel charges may change weekly or monthly and may apply to different charge categories.
The published percentage alone does not tell you the amount due.
The contract must specify both the percentage and the eligible base.

Consider a USD 8.00 discounted base and USD 2.00 of eligible accessorials.
At a 15% fuel rate, applying fuel to base only produces USD 1.20 of fuel.
Applying the same rate to base plus accessorials produces USD 1.50.
The USD 0.30 difference per parcel becomes USD 30,000 over 100,000 parcels if the mix and formula stay constant.
This example excludes rounding and other charges.
It illustrates why two nominally identical fuel percentages need not be economically equivalent.

Maintain effective dates for the fuel index, percentage table, eligible charges, and negotiated adjustment.
Retain the billed fuel line separately from the expected calculation.
When a difference appears, first check the ship date and contract interpretation before treating it as a carrier error.
Procurement should quantify exposure to both the fuel percentage and its charge base during negotiation.
Finance should use scenario ranges rather than presenting a volatile fuel assumption as a fixed commitment.

### 5.15 Establish operational gates for a new carrier

A signed price is not the same as an operationally usable service.
Treat readiness as a set of observable gates, each with an owner.

| Gate | Evidence required | Accountable owner | Failure response |
|---|---|---|---|
| Commercial | Signed rates and effective dates | Procurement | Do not route production parcels |
| Capacity | Accepted origin/day allocation and conditions | Transportation | Limit allocation to confirmed units |
| Data | Stable parcel ID and event mappings | Technology | Continue integration test |
| Label | Approved test labels and routing codes | Fulfillment systems | Block affected service |
| Pickup | Confirmed pickup schedule and contact | Site operations | Use approved fallback |
| Tracking | Acceptance and delivery events received | Customer experience | Limit pilot and investigate |
| Finance | Invoice detail, charge codes, payment setup | Accounts payable | Hold launch or use approved control |
| Claims | Evidence requirements and escalation channel | Risk or claims team | Resolve process before scaling |
| Promise | Lane-level transit validation and cutoff | Promise owner | Restrict eligibility |
| Contingency | Tested alternate carrier or recovery process | Transportation | Do not rely on untested fallback |

A pilot should specify a cohort, comparison baseline, observation window, and stop criteria before the first shipment.
For example, a stop criterion could be a missed acceptance scan for a defined share of the daily cohort.
The actual threshold should be approved by the accountable business owner and supported by the sample size.
Small pilots cannot prove performance at a very high service percentile.
Use operational observations, confidence intervals, and peak stress scenarios together.

### 5.16 Separate recoverable variance from financial benefit

An invoice variance is the difference between an expected charge and a billed charge.
It is not automatically an overcharge.
An approved dispute is not yet a posted credit.
A posted credit is not necessarily incremental saving if the baseline already assumed the lower charge.

1. Match the invoice line to a parcel, service, date, and contract version.
2. Reproduce the expected charge and retain its input values.
3. Classify the difference as data, contract interpretation, carrier adjustment, missing charge, or unresolved.
4. Quantify the gross difference in invoice currency.
5. Check whether the contract allows a dispute and whether the filing deadline remains open.
6. Attach parcel and measurement evidence where relevant.
7. Record submission and carrier decision dates.
8. Match any carrier credit to the original invoice and the general ledger.
9. Escalate recurring root causes to the system or process owner.
10. Report gross variance, disputed value, approved value, and realized credit separately.

If a USD 100 invoice variance is valid under the contract, there is no recoverable USD 100.
If a USD 100 claim is accepted but only USD 60 has been credited, realized recovery is USD 60.
The remaining USD 40 needs follow-up rather than being booked twice.
Use the finance team's accounting policies for recognition and period attribution.

### 5.17 Quantify service-mix risk under disruption

Lowest expected cost is not the same as lowest operational risk.
When several parcels share a carrier facility, pickup, or trunk departure, their outcomes can be correlated.
The same incident can make a whole cohort late.
An independent per-parcel probability model may understate the size of that tail event.

Start with the feasible allocation under normal conditions.
Then select disruptions that the operating team recognizes as plausible.
Examples include a missed pickup, a regional facility closure, an unavailable linehaul vehicle, and a temporary carrier volume cap.
Specify duration, affected origin and destination, and the remaining capacity of each alternative.
Do not assume the entire affected volume can move to a backup unless the backup's cutoff, capacity, and package rules permit it.

For each scenario, calculate:

1. Parcels affected per day.
2. Parcels that can be rerouted before the relevant cutoff.
3. Incremental carrier and handling cost for those parcels.
4. Parcels that cannot be recovered within the promise.
5. Customer communication and operational escalation needs.
6. Expected duration until normal operation resumes.
7. Contractual obligations and any volume-tier effects.
8. Owner and trigger for activating the contingency.

Suppose a service carrying 500 parcels per day becomes unavailable after pickup has been missed.
The alternate carrier can accept 200 parcels before its later cutoff.
The remaining 300 parcels lack a confirmed same-day alternative.
If the alternate costs USD 1.10 more per parcel, the direct incremental transport cost is USD 220 for the recovered 200 parcels.
That USD 220 excludes the cost and customer impact of the 300 unrecovered parcels.
Calling the scenario a USD 220 exposure would therefore be incomplete.

If disruption probabilities are estimated, report their source and period.
A low-confidence probability should not be presented as a precise forecast.
Where probabilities are unavailable, show conditional impact and the minimum capacity needed to recover.
Decision owners can compare contingency costs with the operational risk they are willing to accept.
Service credits may offset part of the invoice but do not guarantee recovery of customer trust or repeat purchases.

### 5.18 Read a carrier bid like a decision record

The bid recommendation should be reproducible without relying on the analyst's memory.
For each carrier-service combination, preserve the following items.

| Bid component | Question for the decision record |
|---|---|
| Coverage | Which origins, destinations, and parcel types are eligible? |
| Service | What pickup and delivery calendar applies? |
| Base rates | Which table and effective date were used? |
| Weight | What dimensions, divisor, and rounding rules apply? |
| Discounts | Which charges qualify and what are the conditions? |
| Minimums | Are they per parcel, per invoice, or per contract period? |
| Accessorials | Which triggers, fees, and exclusions apply? |
| Fuel | What index, frequency, percentage, and charge base apply? |
| Peak | Which dates and services carry additional charges? |
| Capacity | How much is accepted by origin, day, and cutoff? |
| Commitment | What does each party owe, and what happens on shortfall? |
| Service remedies | How are claims, exclusions, and credits defined? |
| Change rights | Who can change terms and with what notice? |
| Integration | Who pays for labels, events, testing, and support? |
| Transition | What is the launch period and rollback option? |

Re-rate the baseline cohort under each proposal.
For missing terms, show the cost as unknown or use a clearly labeled scenario.
Do not fill an unquoted charge with zero merely to complete a comparison.
Show at least the current mix, the proposed mix, and a peak-stress mix if those scenarios are relevant.
Ask each bidder to verify its own interpretation of the quote.
Finance should reconcile the chosen baseline to an actual ledger period before using the result as a savings commitment.

The award note should state what was decided, who approved it, when it takes effect, and which assumptions could change the result.
It should also state what was not decided, such as a future increase in peak capacity or a new destination injection facility.
Keep the executed contract separate from a model's interpretation of it.
The model is a decision aid; legal and procurement owners resolve contractual meaning.

### 5.19 Design a practical operating dashboard

A useful dashboard connects the planned decision with physical execution and eventual payment.
Each measure needs an owner, cadence, grain, and action threshold.
Thresholds should be set from the service and contract context rather than copied blindly from another company.

| Measure | Numerator and denominator or unit | Cadence | Owner and action |
|---|---|---|---|
| Planned allocation | Planned parcels by service and day | Daily | Transportation checks capacity |
| Accepted volume | Accepted parcels / tendered parcels | Daily | Site escalates rejected tenders |
| Cutoff compliance | On-time acceptances / eligible tenders | Daily | Site resolves late handover |
| Capacity utilization | Accepted parcels / confirmed capacity | Daily | Transportation rebooks or diverts |
| Promise attainment | On-time deliveries / matured promise cohort | Weekly | Promise owner investigates failures |
| Open overdue | Undelivered parcels past promise, parcels | Daily | Customer care contacts affected customers |
| All-in unit cost | Matched carrier and incremental cost / accepted parcels | Weekly | Finance investigates change |
| Dimensional premium | Dimensional-billed parcels / measured eligible parcels | Weekly | Packaging reviews carton mix |
| Invoice variance | Expected minus billed, USD by charge type | Invoice cycle | Audit investigates evidence |
| Realized credit | Posted credit, USD, matched to dispute | Invoice cycle | Accounts payable reconciles |
| Departure fill | Parcels and used cube / booked capacity | Each departure | Network team checks economics |
| Missed injections | Late or rejected loads / planned loads | Daily | Transportation activates fallback |

Compare planned and actual values using the same population definition.
Show missing data and late-arriving events rather than silently excluding them.
When an alert fires, provide an owner and a concrete next action.
A dashboard that only reports after the customer promise has failed cannot prevent that failure.
The leading indicators are cutoff readiness, accepted capacity, departure, arrival, and carrier injection.
The lagging indicators include delivered-on-time performance, billed charges, and posted credits.

## 6. Worked example

### Decision and data

A merchant ships from one fulfillment center to a destination region during a four-week planning period.
The forecast is 800 eligible parcels per operating day, with 20 operating days, or 16,000 parcels over the period.
The forecast is not a carrier capacity commitment.
Operations proposes a plan of 800 parcels per day, while finance sets an illustrative savings target of USD 10,000 over the period.
Carrier A has accepted a capacity commitment of 500 parcels per day for this segment.
Carrier B has accepted 400 parcels per day.
Each carrier's acceptance is conditional on the agreed pickup window and eligible package profile.

The existing direct service has a cost of USD 9.20 per parcel for this homogeneous segment, including the modeled accessorials.
Carrier A's direct alternative costs USD 8.70 per parcel.
Carrier B's direct alternative costs USD 9.00 per parcel.
The two alternatives meet the customer promise on the agreed segment and service calendar.
For this illustration, the cost and eligibility assumptions are constant over the period.
All prices are hypothetical and in USD.

### Dimension calculation

Consider one representative parcel with measured dimensions 15.2 in × 10.1 in × 6.4 in and actual weight 6.2 lb.
Assume the contract rounds each dimension up to the next whole inch, uses a divisor of 139 cubic inches per lb, and rounds each calculated weight up to the next whole lb.

- Rounded dimensions: 16 in × 11 in × 7 in.
- Chargeable volume: 16 × 11 × 7 = 1,232 cubic in.
- Unrounded dimensional weight: 1,232 / 139 = 8.8633 lb approximately.
- Rounded dimensional weight: 9 lb.
- Rounded actual weight: 7 lb.
- Billable weight: max(9 lb, 7 lb) = 9 lb.

If the base schedule prices 9 lb at USD 10.00 and the eligible base discount is 20%, the discounted base is USD 8.00.
Suppose the minimum is USD 7.50, so the payable base remains USD 8.00.
The contract in this example adds a USD 1.00 residential charge and applies 10% fuel to the discounted base only.
Fuel is USD 0.80 and the all-in amount is USD 8.00 + USD 1.00 + USD 0.80 = USD 9.80 per parcel.
This separate rating illustration does not set the segment-average carrier rates above.
It shows why the fuel base and minimum sequence must be explicit.

### Direct service allocation

Given the stated costs, allocate the first 500 parcels per day to A and the remaining 300 parcels per day to B.
The plan uses all of A's commitment and 300 of B's 400 parcels per day of accepted capacity.
Unused capacity at B is 100 parcels per day; it is not itself an additional cost in this simplified case.

- A daily spend: 500 parcels × USD 8.70 = USD 4,350.
- B daily spend: 300 parcels × USD 9.00 = USD 2,700.
- Total planned direct spend: USD 7,050 per day.
- Existing baseline: 800 parcels × USD 9.20 = USD 7,360 per day.
- Planned direct saving: USD 310 per day, or USD 6,200 over 20 days.
- Weighted planned cost: USD 7,050 / 800 = USD 8.8125 per parcel.

The direct plan falls USD 3,800 short of the USD 10,000 target.
This is not evidence that a larger saving is guaranteed; it is a gap to investigate.
The plan assumes no minimum-volume penalty, rebate change, or added implementation cost.

### Zone-skip alternative

A third service accepts injection near the destination for this segment.
The illustrative local parcel rate including modeled accessorials is USD 6.60 per parcel.
Incremental origin sort and destination handling total USD 0.40 per parcel.
A dedicated linehaul departure costs USD 1,200 per day regardless of load up to 1,000 parcels.
All 800 daily parcels fit the departure and meet the injection cutoff in the planned schedule.
The merchant must confirm accepted injection capacity and timetable before booking; the example assumes 800 parcels per day have been accepted.

- Skipped variable cost: USD 6.60 + USD 0.40 = USD 7.00 per parcel.
- Daily variable cost: 800 × USD 7.00 = USD 5,600.
- Daily total: USD 5,600 + USD 1,200 = USD 6,800.
- Daily saving against direct allocation: USD 7,050 − USD 6,800 = USD 250.
- Four-week saving against direct allocation: 20 × USD 250 = USD 5,000.
- Four-week saving against existing baseline: USD 6,200 + USD 5,000 = USD 11,200.
- Average skipped cost: USD 6,800 / 800 = USD 8.50 per parcel.

Under the stated forecast and assumptions, the projected USD 11,200 exceeds the USD 10,000 target by USD 1,200.
This is a forecast, not an accounting realization or contractual guarantee.
If one-time launch cost is USD 3,000, the four-week net benefit versus the existing baseline is USD 8,200 and does not meet the target on a net basis.
Finance must specify whether the target includes launch cost before approval.

### Break-even and volume sensitivity

For a simplified comparison to a constant USD 8.8125 per parcel direct alternative, the break-even equation is:

`1,200 + 7.00 × n = 8.8125 × n`.

Thus `n = 1,200 / 1.8125 = 662.07` parcels per departure, or 663 whole parcels.
This approximation holds only if the direct alternative remains USD 8.8125 per parcel at every volume.
It does not reflect A's fixed daily capacity and the resulting changing service mix.

For the actual capacity-constrained alternative, at n above 500 parcels the direct daily cost is `500 × 8.70 + (n − 500) × 9.00 = 9n − 150`.
Equating that with `1,200 + 7n` gives `n = 675` parcels per departure.
At 675 parcels both options cost USD 5,925.
At 674 parcels, direct costs USD 5,916 and zone skip costs USD 5,918, so direct is cheaper by USD 2.
At 676 parcels, direct costs USD 5,934 and zone skip costs USD 5,932, so zone skip is cheaper by USD 2.
Below 500 parcels, direct costs USD 8.70 per parcel and zone skip remains more expensive throughout that range.
These comparisons assume the fixed trip can carry the volume and no other capacity or penalty changes.

At 600 parcels, direct costs USD 5,250 and zone skip costs USD 5,400, a USD 150 loss per departure.
At 800 parcels, zone skip saves USD 250 per departure as calculated above.
At 900 parcels, direct uses 500 A and 400 B parcels for USD 7,950; zone skip costs USD 7,500, saving USD 450.
The 900-parcel comparison uses all accepted direct capacity, and the 1,000-parcel linehaul limit still holds.
Volume above 900 would require a different direct fallback and cannot be evaluated from these commitments alone.

### Time and risk

Assume parcels are released at 14:00 local time, sorted by 16:00, and depart at 17:00.
The linehaul arrives at 23:00, and the destination carrier accepts the load by midnight.
The agreed local service has enough remaining transit time to meet the promised date for the selected region.
If a late origin release misses the 17:00 departure, the team must use an approved direct fallback or explicitly revise the promise.
A failed or late linehaul can affect an entire departure at once, so independent parcel-level failure assumptions may understate risk.

Operations owns departure readiness and escalation.
Procurement owns injection terms and capacity confirmation.
Finance owns baseline and launch-cost treatment.
The promise owner approves any change to the eligible customer promise.
The decision is to pilot the lane only after confirming its operating schedule and contract terms, not to award all volume based on the forecast alone.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the dimensional calculation, direct allocation, and exact capacity-constrained break-even in the worked example.
The functions implement only the stated illustrative contract rules; they are not a general carrier tariff engine.
The direct allocation is optimal here because A is cheaper than B, both are eligible, and there are no volume minimums or other coupling terms.
Values are in USD, inches, lb, and parcels per departure.

```python
from decimal import Decimal, ROUND_CEILING

D = Decimal


def ceil_int(value):
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def billable_weight(dimensions, actual_lb, divisor):
    """Round sides and weights up, then take the greater weight."""
    rounded = [ceil_int(side) for side in dimensions]
    volume = rounded[0] * rounded[1] * rounded[2]
    dimensional_lb = ceil_int(D(volume) / divisor)
    actual_rounded_lb = ceil_int(actual_lb)
    return rounded, volume, dimensional_lb, max(dimensional_lb, actual_rounded_lb)


def direct_cost(n, a_capacity=500, b_capacity=400):
    """A is cheaper; reject volume outside the committed direct capacity."""
    if not 0 <= n <= a_capacity + b_capacity:
        raise ValueError("Volume exceeds modeled direct capacity")
    on_a = min(n, a_capacity)
    on_b = n - on_a
    return on_a * D("8.70") + on_b * D("9.00")


def skip_cost(n):
    """One booked departure with a maximum load of 1,000 parcels."""
    if not 0 < n <= 1000:
        raise ValueError("One departure requires 1–1,000 parcels")
    return D("1200") + n * D("7.00")


rounded, volume, dim_lb, bill_lb = billable_weight(
    [D("15.2"), D("10.1"), D("6.4")], D("6.2"), D("139")
)
print("Rounded dimensions:", rounded)
print("Volume (cubic in):", volume)
print("Dimensional and billable weight (lb):", dim_lb, bill_lb)

base = max(D("10.00") * D("0.80"), D("7.50"))
fuel = base * D("0.10")
print("Illustrative parcel charge (USD):", base + D("1.00") + fuel)

for n in (600, 674, 675, 676, 800, 900):
    direct = direct_cost(n)
    skipped = skip_cost(n)
    print(f"{n}: direct={direct:.2f} skip={skipped:.2f} saving={direct-skipped:.2f}")

break_even = next(
    n for n in range(1, 901) if skip_cost(n) <= direct_cost(n)
)
print("First break-even whole-parcel volume:", break_even)
print("20-day saving versus current baseline (USD):",
      (D("9.20") * 800 - skip_cost(800)) * 20)
```

Expected output:

```text
Rounded dimensions: [16, 11, 7]
Volume (cubic in): 1232
Dimensional and billable weight (lb): 9 9
Illustrative parcel charge (USD): 9.8000
600: direct=5250.00 skip=5400.00 saving=-150.00
674: direct=5916.00 skip=5918.00 saving=-2.00
675: direct=5925.00 skip=5925.00 saving=0.00
676: direct=5934.00 skip=5932.00 saving=2.00
800: direct=7050.00 skip=6800.00 saving=250.00
900: direct=7950.00 skip=7500.00 saving=450.00
First break-even whole-parcel volume: 675
20-day saving versus current baseline (USD): 11200.00
```

The program uses decimal arithmetic to avoid binary floating-point representation surprises in the example.
A production system needs effective-dated contracts, currency-specific rounding, exception handling, actual parcel eligibility, and invoice-level terms.
A zero-volume departure is not evaluated by `skip_cost`: whether a canceled or empty departure incurs a charge depends on the actual booking terms.
The break-even search deliberately stops at 900 parcels because direct capacity beyond that point is unspecified.

## 8. Data, standards, and regulatory context

### Required data and grain

Use parcel ID as the primary operational grain and retain a stable link to order ID.
A shipment can have several invoice lines and several tracking events; joining both raw tables directly can multiply rows and spend.
Aggregate each child table to the parcel or preserve separate fact tables before joining.
A rate version must be effective on the relevant ship date and for the applicable service.
Delivery events need occurrence time, receipt time, time zone, event source, and status interpretation.

| Data set | Minimum fields | Owner and control |
|---|---|---|
| Orders | Order ID, promise timestamp, destination region | Commerce owns promise history; preserve changes |
| Parcels | Parcel ID, order ID, node, actual weight, dimensions, label time | Fulfillment owns measurement calibration |
| Tenders | Parcel or manifest ID, carrier, acceptance or rejection, timestamp | Transportation reconciles daily |
| Carrier events | Parcel ID, event, occurrence timestamp, source | Analytics checks duplicates and late events |
| Invoice | Invoice ID, parcel ID, charge code, amount, currency, tax, adjustments | Accounts payable reconciles to ledger |
| Contract | Service, effective dates, rates, modifiers, capacity terms | Procurement approves versions |
| Capacity | Origin, service, day, committed units, conditions | Transportation confirms with carrier |
| Linehaul | Departure ID, capacity, cost, arrival and acceptance times | Network planning reconciles departures |
| Disputes | Invoice line, reason, deadline, outcome, credit | Audit owner tracks recovery |

For cost per order, sum all eligible parcel charges to order ID before dividing by orders.
For cost per parcel, divide parcel-related costs by accepted parcels with a stated treatment of canceled labels and unallocated fees.
For on-time delivery, select a cohort by promise date or tender date and declare the choice.
Include unresolved parcels once their allowed observation window closes; otherwise a delayed parcel can disappear from the denominator.
A service credit is a financial recovery, not an on-time delivery.

### Standards and conventions

GS1 publishes standards for identifying logistic units and encoding identifiers in barcodes.
A Serial Shipping Container Code (SSCC) identifies a logistic unit; its use does not by itself specify the carrier's price or promise.
Carrier label specifications and the parties' integration agreement govern which identifiers and labels are accepted in a particular flow.
GS1 standards are voluntary unless incorporated into a contract or applicable legal requirement.

ANSI X12 and UN/EDIFACT are families of electronic data interchange standards; a carrier and shipper may agree to use particular transaction sets and implementation guides.
A standard message syntax does not guarantee a complete or correct invoice, nor does it replace contract-specific charge mapping.
Use the agreed implementation guide, preserve raw messages, and reconcile totals to the invoice and ledger.
Application programming interfaces (APIs) can carry equivalent operational data but still require versioning, authentication, and agreed semantics.

In the United States, the Federal Trade Commission's Mail, Internet, or Telephone Order Merchandise Rule addresses shipment representations and delay/consent obligations in covered transactions.
The rule's requirements depend on the facts and the merchant's representations; a carrier's estimate alone does not determine compliance.
Consult the current rule and counsel for operational application, including notices and refunds.
A carrier agreement or industry convention is not itself a statute.

Personal data in destination and tracking records should be limited to what the use requires, access-controlled, and retained according to applicable law and policy.
Privacy obligations depend on jurisdiction, relationship, and data use; a zone-level analytic extract can often avoid full addresses.
Cross-border shipments may add customs, restricted-goods, tax, and data-transfer requirements outside this domestic example.

### Measurement and audit controls

1. Calibrate warehouse scales and dimensioners on a documented schedule.
2. Keep a sample of measured images or other evidence according to retention policy.
3. Compare warehouse and carrier measurements without presuming either is always correct.
4. Flag systematic differences by equipment, node, carton, and service.
5. Validate service-zone mapping against the carrier's effective schedule.
6. Reconcile expected base, minimum, fuel, accessorials, and tax separately.
7. Record whether the carrier has credited an approved dispute; an approved claim is not yet cash recovery.
8. Review missing scans and invoice-only parcels independently.
9. Use immutable contract version references for historical rerating.
10. Restrict edits to master rates and require a second approver for production changes.

## 9. Industry practice and case evidence

### Published carrier rating rules

UPS describes dimensional weight as a pricing technique based on package volume and publishes instructions for determining billable weight.
FedEx similarly publishes guidance on dimensional weight and parcel measurement.
Their current service guides and agreements, not this lesson, determine the exact divisor, rounding, accessorials, and eligibility for a specific shipment.
These are carrier-published descriptions of their own pricing practices, not independent proof that a particular shipper will save money.

**Inference:** A merchant shipping low-density merchandise should test the effect of actual packed dimensions and negotiated divisors on its own mix.
That inference is not a claim that either carrier offers a particular contract to the merchant.

### Postal entry and consolidation

The United States Postal Service (USPS) publishes domestic mail standards and destination-entry conditions for eligible products.
These standards show that entry location, preparation, and eligibility can matter to a mailing's treatment.
They do not establish that a particular private carrier's destination-injection arrangement is available, nor that all parcels receive an automatic discount.
A merchant must verify the applicable product, entry facility, preparation requirements, and prices for its own arrangement.

**Inference:** Consolidation can trade fixed trunk cost and handling time for lower downstream unit cost.
The worked example quantifies that trade-off without attributing its hypothetical rates to USPS or a named company.

### UPS and peak network planning

UPS publishes peak or demand surcharge information and service conditions on its official sites.
This is evidence that time- and service-specific charges can matter to the all-in rate.
It is not evidence that any surcharge applied in the worked example or that the merchant can avoid a charge by changing carriers.
Compare the effective schedules and negotiated agreement for the dates and services in scope.

### FedEx and service availability

FedEx publishes service descriptions, rate information, and service guides.
These materials demonstrate that a service name is accompanied by conditions, operating coverage, and package requirements.
They are not an independently audited measurement of the on-time rate on a particular merchant's lanes.
Use the merchant's own tender and delivery cohorts to estimate that performance, with sufficient sample size and a consistent denominator.

### Evidence hierarchy for a carrier award

- Executed carrier terms are evidence of obligations, subject to their stated conditions and legal interpretation.
- A confirmed daily capacity booking is stronger than an unqualified sales estimate, but still has exceptions that should be read.
- Actual matched invoices provide realized charges for the observed cohort, not a causal estimate of a new mix.
- A prospective pilot provides local evidence but may not represent peak conditions or all geographies.
- Carrier marketing may explain a service, but should not be treated as independent service-performance validation.
- A model is an estimate whose value depends on data quality, assumptions, and validation against later outcomes.

Document which category supports each claim in the award recommendation.
Avoid attributing the example's results to any named carrier.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Base-only comparison | Large difference between model and invoices | Fuel and accessorials omitted | False savings | Rate all charge components |
| Wrong dimensional divisor | Lightweight bulky cartons re-bill | Contract rule not effective-dated | Understated cost | Version rules and retain trace |
| Wrong rounding order | Frequent 1 lb differences | Round volume instead of each side | Wrong rate band | Encode agreed sequence |
| Wrong zone | Distant lanes look unusually cheap | Shared carrier zone assumed | Misallocated services | Carrier-specific zone tables |
| Unpriced minimum | Low-weight invoices exceed estimate | Discount applied below minimum | Overstated value | Model contractual sequence |
| Volume tier cliff | Discount changes after allocation | Spend or volume minimum missed | Retroactive true-up | Optimize tier and mix jointly |
| Missing peak charge | Margin deteriorates in season | Effective window omitted | Budget overrun | Maintain dated schedules |
| Overbooked capacity | Tenders rejected near cutoff | Forecast used as commitment | Late orders | Confirm capacity and fallback |
| Missed cutoff | Labels exist but acceptance is late | Creation time treated as handover | Missed promise | Monitor carrier acceptance |
| Selective delivery denominator | Reported on-time rate rises while backlog grows | Open parcels excluded | Hidden service failure | Cohort with observation window |
| Zone skip underfill | Trailer leaves mostly empty | Average volume masks low days | Fixed cost per parcel rises | Departure-level break-even gate |
| Injection rejection | Local facility refuses a load | No agreed product or capacity | Network delay and reroute | Written acceptance and test |
| Double-counted handling | Finance and operations estimates differ | Same labor included twice | Invalid business case | Map cost ownership |
| Ignored fixed launch cost | Pilot looks profitable, launch does not | Only variable rates compared | Negative net benefit | Separate one-time and recurring costs |
| Duplicate joins | Invoice totals exceed ledger | Event and invoice lines multiply | Inflated spend | Aggregate child facts first |
| Rate version drift | Old invoices reprice differently | Current rules overwrite history | Invalid disputes | Immutable effective-dated rules |
| Carrier concentration | No practical fallback after disruption | Lowest price takes all capacity | Correlated outage | Approve diversification and drills |
| Weak dispute evidence | Many claims rejected | No measurement or contract trace | Lost recoveries | Preserve source evidence |
| Misclassified residential flag | Accessorials cluster on one lane | Address type inferred incorrectly | Persistent invoice variance | Use authoritative agreed signal |
| Packing change ignored | Cost rises after carton switch | Dimensions not refreshed | Repeated dimensional charges | Monitor carton and pack measurements |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| Forecast 800 parcels/day | Sizes the worked plan | Promotion or demand changes | Compare daily distribution with forecast |
| A accepts 500 parcels/day | Defines direct capacity | Cutoff missed or carrier changes booking | Retain daily confirmation and terms |
| B accepts 400 parcels/day | Bounds fallback | Peak restriction | Test booked versus tendered volume |
| Local injection accepts 800/day | Makes zone-skip plan feasible | Rejected manifests or facility cap | Obtain written facility commitment |
| Costs are constant within segment | Enables simple arithmetic | Lane, dimensions, or surcharges vary | Re-rate parcel-level data |
| All services meet the promise | Makes price comparison valid | Late release or transit disruption | Analyze end-to-end delivery cohorts |
| One trip costs USD 1,200 | Gives fixed-cost model | Fuel, waiting, or cancellation changes | Obtain full departure quote |
| One trip fits 1,000 parcels | Bounds capacity | Cube or weight fills vehicle first | Test both cube and weight utilization |
| Handling costs USD 0.40/parcel | Captures incremental processing | Overtime or sort equipment constraints | Time-study incremental labor and supplies |
| A remains cheaper than B | Justifies greedy direct allocation | Tier discounts or changing zone mix | Recompute costs and allocation jointly |
| No direct minimum penalty | Avoids omitted nonlinearity | Volume migration breaches agreement | Review minimums and true-ups |
| No launch costs in gross savings | Isolates recurring operating effect | Integration and training are material | Show net and gross business cases |
| Carrier measurements follow example | Enables calculation | Different contract or remeasurement | Confirm service guide and contract |
| No additional customer failure cost | Keeps cost model narrow | Late delivery causes refunds or churn | Estimate separately with evidence |
| No cross-border charges | Keeps domestic example coherent | Customs and duties apply | Use a separate trade-compliance model |

A result is conditional on these assumptions.
A sensitivity table does not replace a binding carrier commitment or a service validation.
Do not extrapolate a daily break-even to a month by using only average volume when each departure incurs a fixed cost.

## 12. Exercises

### Exercise 1 — Rate one parcel

Time: 15 minutes. Type: calculation.

A parcel measures 12.1 in × 9.0 in × 5.2 in and weighs 4.1 lb.
Round each dimension up to whole inches, divide cubic inches by 139 cubic in/lb, and round both weights up to whole lb.
The 6 lb base rate is USD 9.50; discount the base 10%, then apply an USD 8.00 minimum.
Add USD 1.20 residential and fuel of 12% of payable base only.

Deliverable: calculation trace and all-in charge in USD.
Acceptance: show rounded dimensions, dimensional weight, actual weight, billable weight, base after discount and minimum, fuel, and total.

### Exercise 2 — Critique an invoice comparison

Time: 15 minutes. Type: critique.

A manager reports carrier A at USD 8.00 per parcel and B at USD 9.00 per parcel.
A handles only light local parcels; B handles remote heavy parcels and its invoice includes residential fees.
The manager proposes moving all B volume to A immediately.

Deliverable: a short critique and a replacement comparison design.
Acceptance: identify at least three confounders or feasibility gaps and specify the parcel-level data and cost components needed.

### Exercise 3 — Allocate capacity

Time: 15 minutes. Type: application.

There are 700 identical eligible parcels on one day.
Service A costs USD 7.80 per parcel and accepts at most 300; service B costs USD 8.10 and accepts at most 500.
Both satisfy the promise; there are no minimums, fixed costs, or other constraints.

Deliverable: allocation and total USD cost.
Acceptance: assign all 700 parcels within capacity and demonstrate why the allocation is least cost under the stated assumptions.

### Exercise 4 — Zone-skip threshold

Time: 20 minutes. Type: calculation and interpretation.

Direct service costs USD 10.00 per parcel with sufficient capacity.
A skipped movement costs USD 900 per departure plus USD 7.50 per parcel, including local delivery and handling.
The departure can carry 600 parcels and meets the promise.

Deliverable: break-even whole-parcel volume and costs at 300 and 400 parcels.
Acceptance: show the equation and distinguish equality from strictly positive savings.

### Exercise 5 — Build a business case

Time: 25 minutes. Type: application and design.

A forecast is 500 parcels per day for 20 days.
The current cost is USD 10.00 per parcel; the planned cost is USD 9.40 per parcel.
Implementation costs USD 4,000 once and a USD 5,000 net saving is the target for this 20-day period.
The new carrier has committed only 400 parcels per day; the remaining 100 can use the current service.

Deliverable: gross and net forecast benefit using the actual committed allocation, plus a decision note.
Acceptance: separate forecast, plan, target, and commitment; do not apply the new price to uncommitted volume.

### Exercise 6 — Negotiate a comparable offer

Time: 25 minutes. Type: design.

Two carriers propose different base discounts, minimum charges, fuel bases, peak terms, and daily capacities.
One asks for exclusivity and a quarterly volume minimum.

Deliverable: a one-page bid comparison and negotiation checklist.
Acceptance: request a common cohort re-rate, identify at least four non-base terms, quantify the minimum-volume downside, and assign owners for legal, finance, operations, and procurement.

### Exercise 7 — Audit a disputed charge

Time: 20 minutes. Type: application.

A carrier bills USD 2.00 more than expected for a parcel, reporting 11 lb billable weight versus the merchant's 9 lb.
The package has a dimensional-weight rule and a later carrier measurement record.

Deliverable: an investigation and dispute workflow.
Acceptance: identify source evidence, rounding and contract checks, possible valid causes, deadline, and credit reconciliation.
Do not assume the carrier is wrong.

### Exercise 8 — Design service controls

Time: 25 minutes. Type: design.

A new skipped lane moves 800 parcels on a single daily linehaul.
A delayed departure can affect the entire group.

Deliverable: a monitoring and contingency design.
Acceptance: define an end-to-end on-time cohort, capacity and cutoff alerts, linehaul failure fallback, decision owners, and a way to detect correlated failure.

## 13. Self-check answers

### Exercise 1

Rounded dimensions are 13 in × 9 in × 6 in, or 702 cubic in.
Dimensional weight is 702 / 139 = 5.0504 lb approximately, rounded to 6 lb.
Actual weight rounds from 4.1 lb to 5 lb, so billable weight is 6 lb.
The discounted base is USD 9.50 × 0.90 = USD 8.55, above the USD 8.00 minimum.
Fuel is USD 8.55 × 0.12 = USD 1.026.
Before currency rounding, the total is USD 8.55 + USD 1.20 + USD 1.026 = USD 10.776.
If the contract rounds the final charge to the nearest cent, the total is USD 10.78.
An actual contract may round individual charge lines differently.

### Exercise 2

The manager has compared different parcel mixes, not carrier prices on a common workload.
Distance or zone, weight, and included residential fees all differ.
Capacity, service coverage, and ability to meet the promise are also unknown.
Re-rate the same accepted parcel cohort for both carriers using each service's effective zone, billable-weight, base, discounts, minimums, and accessorial terms.
Then screen each alternative for service and capacity and reconcile the estimated costs to invoices.
The proposed immediate migration is unsupported until these checks are complete.

### Exercise 3

Assign 300 parcels to A and 400 to B.
Cost is 300 × USD 7.80 + 400 × USD 8.10 = USD 2,340 + USD 3,240 = USD 5,580.
A is USD 0.30 cheaper per parcel and is filled to its 300-parcel limit; all remaining parcels fit B's capacity.
No other stated constraint can make an alternate allocation cheaper.

### Exercise 4

Set USD 900 + USD 7.50 × n equal to USD 10.00 × n.
The break-even volume is 900 / 2.50 = 360 parcels per departure.
At 300 parcels, direct costs USD 3,000 and skip costs USD 3,150, so skip loses USD 150.
At 400 parcels, direct costs USD 4,000 and skip costs USD 3,900, so skip saves USD 100.
The two options are equal at 360; the first whole-parcel volume with strictly positive savings is 361.
The calculation is valid only within the 600-parcel capacity and stated service assumptions.

### Exercise 5

The demand forecast is 500 parcels per day; the plan assigns 400 to the new carrier and 100 to current service.
The carrier commitment is 400 parcels per day, not 500.
The gross daily saving is 400 × (USD 10.00 − USD 9.40) = USD 240.
Over 20 days it is USD 4,800; after USD 4,000 implementation cost, net benefit is USD 800.
The USD 5,000 net target is missed by USD 4,200.
Recommend revising the case or period rather than claiming the target has been met.
Actual benefit depends on volume, service, and realized charges.

### Exercise 6 rubric

A strong answer normalizes the same parcels and dates, identifies total cost by component, and compares capacity at the relevant daily grain.
It models the minimum-volume penalty or lost discount over a plausible low-volume scenario.
It requests effective dates, change rights, measurement rules, billing detail, and dispute deadlines.
It assigns procurement to commercial terms, legal to contract risk, operations to feasibility, and finance to baseline and savings.
A weak answer ranks headline discounts without checking conditions or owners.

### Exercise 7 rubric

A strong answer retrieves the original parcel dimensions, scale weight, measurement timestamp, contract version, and carrier measurement evidence.
It recalculates both weights with the correct rounding sequence and checks whether the carrier adjustment was permitted.
It submits a documented dispute before the agreed deadline only if the evidence supports it, tracks the response, and verifies the actual credit.
A valid carrier remeasurement, a warehouse measurement error, or a rule-version error could explain the difference.
A weak answer automatically reverses the charge or assumes all reweighs are fraudulent.

### Exercise 8 rubric

A strong answer defines a cohort and deadline, includes unresolved late parcels, and reports both end-to-end and carrier-controlled intervals.
It alerts on capacity, missed acceptance, departure, arrival, and injection milestones with named escalation owners.
It defines a feasible direct or alternate-departure fallback and its effect on promise and cost.
It groups outcomes by departure to expose common-mode failure and tests the contingency periodically.
A weak answer reports only average carrier transit among already-delivered parcels.

## 14. Interview questions

1. **Why is the largest published discount not necessarily the cheapest offer?**  
   Strong: models minimums, accessorials, fuel bases, mix, tiers, and service eligibility.  
   Weak: compares discount percentages without a common charge base.

2. **How do you calculate dimensional weight?**  
   Strong: asks for the contract's units, dimension rounding, divisor, and weight rounding.  
   Weak: assumes a universal divisor and ignores actual weight.

3. **How would you compare carriers with different parcel mixes?**  
   Strong: re-rates the same cohort and checks feasibility and data quality.  
   Weak: compares raw average invoices as causal evidence.

4. **What limits a parcel-level cheapest-service algorithm?**  
   Strong: mentions capacity, volume tiers, commitments, service windows, and fixed costs.  
   Weak: assumes each parcel is independent and all services are always available.

5. **When is zone skipping attractive?**  
   Strong: compares full per-departure costs, volume distribution, cutoff, and promise.  
   Weak: assumes a shorter last-mile zone guarantees savings.

6. **What changes when a linehaul misses its cutoff?**  
   Strong: explains common-mode delay, fallback capacity, customer promise, and escalation.  
   Weak: treats the event as an isolated parcel exception.

7. **How do you define on-time delivery?**  
   Strong: states the cohort, promise timestamp, observation window, and handling of unresolved parcels.  
   Weak: uses delivered parcels only without a denominator definition.

8. **How do you evaluate a volume commitment?**  
   Strong: calculates tier benefits and shortfall penalties across scenarios and contract periods.  
   Weak: equates a forecast with an enforceable carrier capacity booking.

9. **How would you investigate a billable-weight adjustment?**  
   Strong: checks source measurements, rule version, evidence, and dispute deadline.  
   Weak: assumes either warehouse or carrier measurement is always correct.

10. **How do you distinguish a target from savings?**  
    Strong: reconciles forecast, planned allocation, actual invoice cost, and one-time expense.  
    Weak: reports the target as realized financial benefit.

## 15. Key terms

The following quick index points back to the [definitions](#4-definitions-and-vocabulary) and their application.

- **Accessorial:** conditional charge beyond the base; see [direct cost](#55-build-all-in-direct-cost).
- **Actual weight:** measured scale weight before contractual rounding; see [weight calculation](#54-compute-dimensional-and-billable-weight).
- **Base charge:** schedule price before specified adjustments; see [rate normalization](#53-normalize-contract-terms).
- **Billable weight:** rating weight after the agreed comparison; see [worked example](#dimension-calculation).
- **Capacity:** available volume by relevant time bucket; see [service mix](#57-allocate-a-service-mix).
- **Carrier acceptance:** evidence of transfer into the carrier network; see [service feasibility](#56-screen-service-feasibility).
- **Commitment:** an obligation accepted under stated conditions; see [mental model](#3-the-core-mental-model).
- **Contribution margin:** defined revenue less variable costs; see [decision scope](#51-establish-the-decision-and-horizon).
- **Cutoff:** latest handover time for an operating cycle; see [zone skipping](#58-evaluate-zone-skipping).
- **Dimensional divisor:** volume-to-weight conversion factor; see [dimensional calculation](#54-compute-dimensional-and-billable-weight).
- **Dimensional weight:** volume-based rating weight; see [worked example](#dimension-calculation).
- **Discount:** a contractual reduction on a specified base; see [negotiation](#59-negotiate-and-implement).
- **Forecast:** estimate of future demand, distinct from commitment; see [worked example](#decision-and-data).
- **Fuel surcharge:** variable adjustment on an agreed base; see [direct cost](#55-build-all-in-direct-cost).
- **Injection:** transfer into a downstream carrier network; see [zone skipping](#58-evaluate-zone-skipping).
- **Invoice audit:** reconciliation of charges to terms and evidence; see [controls](#measurement-and-audit-controls).
- **Linehaul:** movement between facilities; see [break-even](#break-even-and-volume-sensitivity).
- **Minimum charge:** contractual lower bound on a charge; see [rate normalization](#53-normalize-contract-terms).
- **On-time delivery:** outcome measured against a defined deadline and cohort; see [data context](#8-data-standards-and-regulatory-context).
- **Parcel:** individually labeled handling unit; see [population](#52-construct-the-parcel-population).
- **Peak surcharge:** time-dependent additional charge; see [failure modes](#10-failure-modes-and-red-flags).
- **Plan:** intended allocation of resources; see [service mix](#57-allocate-a-service-mix).
- **Promise:** delivery expectation communicated to a customer; see [service feasibility](#56-screen-service-feasibility).
- **Rate card:** price schedule with conditions; see [normalization](#53-normalize-contract-terms).
- **Residential surcharge:** address-condition charge under carrier terms; see [direct cost](#55-build-all-in-direct-cost).
- **Service mix:** allocation among available services; see [direct allocation](#direct-service-allocation).
- **Target:** desired outcome, not an observed result; see [business case](#zone-skip-alternative).
- **Tender:** offer of parcels to a carrier; see [operating controls](#510-operate-audit-and-improve).
- **Volume tier:** price condition determined by aggregate volume or spend; see [negotiation](#59-negotiate-and-implement).
- **Zone:** carrier-defined origin–destination rating band; see [direct cost](#55-build-all-in-direct-cost).
- **Zone skipping:** consolidation and destination-region injection; see [mechanics](#58-evaluate-zone-skipping).

## 16. Sources

Sources below are primary references for their respective rules or practices, not support for the hypothetical numerical case.
Commercial terms and online guides change; check the applicable version and executed agreement at the time of use.

1. UPS, [Determine the Weight and Size](https://www.ups.com/us/en/support/shipping-support/shipping-dimensions-weight). Carrier guidance on measurement and billable weight.
2. FedEx, [Dimensional Weight](https://www.fedex.com/en-us/shipping/packaging/what-is-dimensional-weight.html). Carrier guidance on volume-based shipment rating.
3. UPS, [Demand Surcharges](https://www.ups.com/us/en/supplychain/resources/news-and-market-updates/demand-surcharges). Carrier information about time-specific surcharges; verify current applicability.
4. FedEx, [Service Guide](https://www.fedex.com/en-us/service-guide.html). Service and rate documentation, subject to effective dates and individual contracts.
5. United States Postal Service, [Domestic Mail Manual](https://pe.usps.com/text/dmm300/dmm300_landing.htm). Official product, preparation, and entry rules for applicable USPS services.
6. GS1, [Serial Shipping Container Code](https://www.gs1.org/standards/id-keys/sscc). Identifier standard for logistic units; not a rate or service guarantee.
7. X12, [Standards](https://x12.org/products/standards). Electronic data interchange standards; implementation depends on trading-partner agreements.
8. United Nations Economic Commission for Europe, [UN/EDIFACT](https://unece.org/trade/uncefact/introducing-unedifact). International electronic data interchange standard background.
9. U.S. Federal Trade Commission, [A Business Guide to the FTC's Mail, Internet, or Telephone Order Merchandise Rule](https://www.ftc.gov/business-guidance/resources/business-guide-ftcs-mail-internet-or-telephone-order-merchandise-rule). Agency guidance on covered shipment representations and delays.
10. U.S. e-CFR, [16 CFR Part 435](https://www.ecfr.gov/current/title-16/chapter-I/subchapter-D/part-435). Current regulatory text; applicability depends on facts and jurisdiction.

This material is educational and is not legal, tax, or procurement advice.

## 17. Links

- [Module 21 index](./_index.md)
- [Module 21 survey](../../../course/21_ecommerce_supply_chain_analytics.md)
- [Delivery promise](./05_delivery-promise.md)
- [Peak planning](./08_peak-planning.md)
- [Unit economics per order](./09_unit-economics.md)
- [Simulation and optimization production labs survey](../../../course/30_simulation_optimization_production_labs.md)
