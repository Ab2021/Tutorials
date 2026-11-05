# Ship-from-Store Economics

> Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise  
> Topic 04 of 10  
> Estimated study time: 4 hours  
> Prerequisites: inventory visibility, available-to-promise, and order routing  
> Version 2026.5

## 1. Why this topic exists

A retailer with stores near customers may be able to ship an online order from a store rather than from a distribution center (DC).
The store already holds merchandise and may be closer to the customer.
Neither fact establishes that using the store is profitable or operationally safe.

The decision is made at several horizons.
A network leader decides which stores may participate and whether the fixed enablement cost is justified.
A store manager decides how much capacity can be offered in each hour without undermining customers already in the store.
An order-routing service chooses among feasible sources for an individual order.
Finance later reconciles the result to actual cash expenses and the channel profit and loss statement (P&L).
These decisions need related but not identical cost measures.

The relevant immediate question is not whether the store has a low average cost per parcel.
It is whether the next feasible order creates more enterprise value from the store than the best feasible alternative.
If store employees are already paid, their time may still displace selling, receiving, replenishment, breaks, or another online order.
If no work is displaced and no extra labor is purchased, the cash cost of that particular minute may be zero.
If a shift extension or an extra employee is required, the cost arrives in a discrete block.
A single average labor rate cannot represent all three cases.

The same distinction applies to merchandise.
A unit that appears in a perpetual inventory record may have been sold, misplaced, damaged, or reserved.
A unit that can be found may be the last unit a visiting customer intended to buy.
A unit that is unlikely to sell locally before markdown may be particularly valuable to ship.
Inventory reliability, lost local contribution, and markdown avoidance are different effects and should not be collapsed into one unexplained penalty.

Fast delivery is also not free.
Store picks compete with shoppers and store replenishment.
Parcel service has cutoffs, variable charges, dimensional weight, and exceptions.
A late tender can invalidate a delivery promise even when the parcel is picked on time.
An excessive parcel split adds packaging, carrier charges, customer contacts, and potentially separate returns.

This unit gives you a decision procedure for these trade-offs.
The numbers in the principal example are illustrative, not a claim about a real retailer.
The procedure can be implemented as a spreadsheet, a routing rule, or a more formal optimization model.
The important controls are explicit baselines, units, feasibility checks, ownership, and feedback from realized outcomes.

### Decisions and owners

- The **network design owner** approves store participation and network-level investment.
- The **store operations owner** approves workload standards, staffing, cutoffs, and service protections.
- The **merchandising owner** approves local stock protection and markdown assumptions.
- The **transportation owner** validates carrier services, pickup capacity, and rates.
- The **finance owner** approves incremental cost and benefit definitions and reconciles results.
- The **routing product owner** implements eligibility, reservation, exception, and audit logic.
- The **customer experience owner** owns the customer-facing promise and exception communication.

No one owner can independently establish the entire business case.
The store manager's acceptance of a workable pick plan does not establish enterprise profitability.
Finance's estimate of contribution does not authorize an unsafe staffing plan or a carrier service that cannot meet the promise.

## 2. Learning outcomes

After this unit, you should be able to:

1. Map a store order from reservation through carrier tender and identify its capacity and inventory gates.
   Exercise 1 tests the sequence and ownership.
2. Compute paid labor cost and explain when paid minutes do not equal incremental cash cost.
   Exercise 2 tests both the arithmetic and the counterfactual.
3. Calculate the incremental expected value of a store route against a feasible alternative.
   Exercise 3 tests contribution, avoided cost, and risk.
4. Determine whether a time-bucketed store capacity constraint admits an order.
   Exercise 4 tests the distinction between average daily capacity and a binding cutoff.
5. Diagnose uncertainty and selection bias in observed store fulfillment results.
   Exercise 5 asks you to critique a comparison.
6. Design an operating experiment with safeguards, owners, and a reconciliation plan.
   Exercise 6 tests design rather than a single numerical answer.
7. Explain how store fixed costs, step costs, and incremental order costs support different decisions.
   Exercise 7 tests the break-even calculation and its limits.
8. Propose a response to a missing unit without double selling or silently breaking the customer promise.
   Exercise 8 tests exception handling.

Observable performance means that you can show units, time horizon, baseline, assumptions, and decision owner.
It does not mean that a single score applies to every store, product, or service promise.

## 3. The core mental model

The same customer order can be served by different feasible routes, declined, or deferred.
The correct comparison is between complete consequences under the selected route and a stated counterfactual.
Start with hard feasibility, then compare economic value, then monitor execution.

```text
Customer order and promise
          |
          v
Inventory and eligibility ---- no ----> alternative or no promise
          |
         yes
          v
Store labor + pack + pickup capacity ---- no ----> alternate node
          |
         yes
          v
Compare expected incremental enterprise value
          |
          v
Reserve inventory AND a capacity slot
          |
          v
Pick -> verify -> pack -> label -> stage -> carrier acceptance
  |         |                                     |
  +---- exceptions, re-route, or customer care ---+
          |
          v
Actual costs, service, stock outcomes -> learning and control
```

### The alternative matters

A store route is not evaluated against doing nothing unless doing nothing is genuinely the alternative.
If a DC can serve the same order with the same promise, compare store and DC.
If only the store can meet the promise, the relevant alternative may be a later delivery offer or a lost order.
A lost order requires an estimate of customer acceptance, contribution, and future effects; it does not justify treating all revenue as incremental profit.
If the DC can ship only by splitting the basket, include all additional parcels and their consequences.

A customer may order regardless of source.
In that common case the sale price and cost of goods sold are the same under both routes and cancel from the routing comparison.
They still belong in the order P&L.
If source choice changes conversion, cancellation, discount, or product availability, the common-revenue assumption no longer holds.
State and estimate the difference rather than deleting revenue unconditionally.

### Three levels of economics

**Order routing:** compare incremental future consequences of feasible routes.
Sunk store rent and a salaried manager's existing pay usually do not change because one order routes to the store.
Avoided shipping, a shift extension, replacement packaging, expected failed picks, and displaced store sales can change.

**Capacity planning:** compare capacity blocks, workload volatility, and the contribution of the orders they enable.
Hiring an associate for a shift creates a step cost even when the last order uses only a few minutes.
The relevant alternative may be redistributing hours, declining an order, or routing to a DC.

**Program investment:** compare incremental benefits over the investment horizon with implementation, training, equipment, and continuing operating costs.
The program's cost includes the resources necessary to enable the capability even if they are sunk by the time a subsequent order is routed.
A positive per-order differential does not prove that a low-volume store pays for a printer and training.

### From cost to expected value

For a single candidate route, define expected value in a common currency and time horizon:

```text
Expected route value = incremental customer contribution
                     - incremental fulfillment and recovery cost
                     - expected opportunity loss
                     + expected avoided markdown or other incremental benefit
```

For the same order, compare this value with the best feasible alternative.
Equivalently, use the store's incremental advantage over that alternative.
Positive advantage favors the store on the modeled economics, provided hard service, safety, and policy constraints are satisfied.
A negative advantage can still be approved as an explicit service or strategic exception; record who authorized it and why.

Expected value is an average across outcomes, not a guarantee for a particular parcel.
A store's actual missing-unit event may require a costly reroute.
The probability and cost must describe the same conditional event and time period.
A high average benefit can coexist with an unacceptable risk of a severely late order.
Use a promise guardrail and an escalation policy in addition to an average-value rule.

### Four kinds of time

- **Forecast:** predicted orders, arrivals, pick times, and local sales in a future bucket.
- **Plan:** approved labor allocation and capacity offered for that bucket.
- **Target:** desired metrics, such as on-time handoff, with a defined denominator.
- **Commitment:** the delivery promise actually shown and accepted by the customer.

A forecast of 20 orders is not permission to promise 20 orders when only 12 slots are safely available.
A target of 98% on-time handoff is not a guarantee that a given parcel will arrive by its date.
A customer commitment must use feasible pickup, transit, and calendar information at the time it is offered.

### Enterprise value versus attribution

Store credit, channel credit, internal transfer price, and store-manager incentives influence behavior.
They do not create new enterprise profit by themselves.
A transfer from the digital channel to the store is income to one internal unit and expense to another.
Consolidated analysis eliminates it.
Design incentives so stores are not penalized for doing enterprise-beneficial work, but keep that design separate from the economic calculation.

## 4. Definitions and vocabulary

The units below are illustrative defaults; preserve the currency, calendar, and grain used in your own model.

| Term | Definition and unit | Common misuse |
|---|---|---|
| Ship-from-store (SFS) | Online order fulfilled from store stock and tendered to a carrier; order or parcel | Treating pickup at the store as a shipment. |
| DC | Distribution center supplying customers, stores, or both; node | Assuming every DC holds every item. |
| Stock keeping unit (SKU) | Merchant-defined product variant; identifier | Treating different sizes as interchangeable inventory. |
| Order | Customer purchase, potentially containing several lines; order ID | Counting an order once per parcel. |
| Order line | A SKU and quantity within an order; line ID and units | Assuming every line is a separate parcel. |
| Parcel | Physical package handed to a carrier; tracking ID | Treating it as identical to the order. |
| Node | A store or DC that can fulfill an order; node ID | Treating a node as feasible solely because it is near the customer. |
| Available-to-promise (ATP) | Uncommitted eligible inventory available for a specified horizon; units by SKU and node | Using the raw on-hand balance as a customer promise. |
| Reservation | Atomic claim on inventory for an order; units and expiry time | Assuming a displayed balance prevents concurrent sale. |
| Safety stock | Protected quantity or policy buffer; units | Treating it as a substitute for measuring accuracy. |
| Phantom inventory | System stock that cannot be physically located or sold; units | Treating all missing picks as theft. |
| Pick failure | An assigned unit cannot be picked under the defined procedure; failed units or orders | Counting a temporary delay as a final failure without an event definition. |
| Pick success probability | Probability that an assigned unit or order is found and usable; proportion | Applying line-level probabilities to whole orders without a dependence assumption. |
| Tender | Transfer of parcel custody to a carrier; timestamp and event | Treating label creation as carrier acceptance. |
| Carrier cutoff | Latest acceptance time for a specified service and calendar; local timestamp | Assuming the same cutoff for all services and weekdays. |
| Promise | Customer-facing committed delivery date or time window | Substituting the warehouse's internal dispatch target. |
| Service level | Defined share meeting a specified service condition; percentage | Omitting the denominator, timezone, or exception treatment. |
| Pick time | Associate time spent locating and confirming merchandise; labor minutes | Using elapsed clock time as labor time for parallel work. |
| Pack time | Associate time to protect, label, and stage goods; labor minutes | Ignoring rework, replenishing supplies, or exception handling. |
| Standard time | Expected labor minutes for a defined task and conditions | Treating an unvalidated standard as actual observed time. |
| Paid labor rate | Loaded compensation per paid labor hour; currency per hour | Dividing annual salary by an arbitrary productive-hour assumption. |
| Incremental labor cost | Change in future labor cash expense caused by the choice; currency | Charging a second wage when an already-paid idle minute is used. |
| Opportunity cost | Value of the best alternative use of constrained resources; currency | Automatically charging an arbitrary hourly rate as lost sales. |
| Shadow price | Incremental objective value of one additional unit of binding capacity, under a specified model; currency per minute | Assuming a stable price outside the model's valid range. |
| Step cost | Expense incurred when crossing a staffing or capacity threshold; currency per shift or block | Smearing a whole shift over the next order without checking the threshold. |
| Fixed enablement cost | Cost to activate capability over a defined horizon; currency per store | Treating a historical investment as avoidable on every route. |
| Variable cost | Cost that varies with the relevant activity within the stated range; currency per parcel or minute | Assuming all operational expenses vary proportionally at every volume. |
| Marginal order | The next order evaluated against available alternatives; one order | Using average cost at full program volume as its causal cost. |
| Contribution | Revenue less specified variable costs; currency per order | Using a contribution definition without listing included costs. |
| Sunk cost | Past cost that cannot be changed by the decision; currency | Ignoring future renewal or maintenance that remains avoidable. |
| Counterfactual | Outcome that would occur under the alternative decision | Comparing the store with a DC route that is not feasible. |
| Split shipment | Order fulfilled in multiple parcels or from multiple nodes; count and indicator | Assigning one shipping charge to a multi-parcel order. |
| Dimensional weight | Carrier billing weight derived from parcel dimensions under a rate agreement; weight units | Assuming actual product weight alone sets price. |
| Packaging cost | Incremental materials and applicable disposal or handling cost; currency per parcel | Treating packaging material price as the full cost of damage prevention. |
| Markdown avoidance | Incremental contribution from selling a unit before a likely reduction in price; currency per unit | Counting the entire full-price sale as an incremental benefit. |
| Local sales displacement | Contribution lost when a store customer cannot purchase the unit and no equivalent sale is retained; currency per unit | Multiplying the full retail price by a stockout probability. |
| Reroute | Reassignment to a different node following an exception; order or line event | Assuming rerouting restores the original promise automatically. |
| Expected exception cost | Probability-weighted incremental recovery cost under a stated event tree; currency per assigned order | Adding a full recovery cost as well as a risk reserve based on the same event. |
| Capacity bucket | Time interval for which available work and commitments are compared; labor minutes per interval | Using daily spare time to cover a missed pickup cutoff. |
| Utilization | Committed work divided by available capacity for the same interval and resource; proportion | Assuming 100% average utilization leaves room for random arrivals. |
| Store P&L | Store-level attributed revenue and expense for a period; currency per period | Confusing internal credit with consolidated enterprise value. |
| Enterprise P&L | Consolidated revenue and expenses across channels and nodes; currency per period | Counting an internal transfer as external revenue. |

## 5. The mechanics, step by step

### Step 1: Define the decision and its baseline

**Input:** customer basket, commitment requested, location, time, and available route candidates.

**Action:** write the decision in one sentence.
For example: “For this two-unit order, choose Store S or DC D for delivery by Friday.”
Specify whether each candidate can supply the complete order.
Record a no-sale or later-promise alternative only if it is actually available to the customer.

**Output:** a set of mutually exclusive feasible alternatives and a common evaluation horizon.

**Branch:** if the alternatives imply different customer prices, conversion, or delivery commitments, retain those revenue and service differences explicitly.
If the customer has already placed an order at a fixed price and both routes preserve the same promise, those common amounts normally cancel in a route comparison.

Use a decision timestamp.
Past spending is not made avoidable by adding it to a spreadsheet.
Future carrier charges, material consumption, incremental shifts, and recovery work may be avoidable.
The ledger should specify what changes because of the decision rather than merely what is allocated to the store.

### Step 2: Establish the order and parcel grain

**Input:** order ID, line IDs, quantities, SKU characteristics, delivery address zone, and requested service.

**Action:** identify which units can be packed together, which require special handling, and whether a candidate route creates multiple parcels.
Record expected package dimensions and actual and dimensional weight assumptions.
Check carrier eligibility for each combination of commodity, destination, and service.

**Output:** candidate parcel plan with a count, pack type, and expected charge per parcel.

**Branch:** if the basket must split, price each parcel and include extra pick, pack, label, and handoff work.
If one node supplies all units, do not invent a second shipment merely because there are two order lines.
If a damaged or missing unit may create a later split, represent that conditional outcome separately.

A parcel-level carrier invoice is not the same grain as an order-level margin table.
Aggregate charges to the order after linking all tracking IDs and adjustments.
Preserve the underlying parcel rows for audit and carrier reconciliation.

### Step 3: Apply merchandise eligibility

**Input:** SKU-node inventory, condition, product restrictions, reservations, and current feed age.

**Action:** exclude merchandise that cannot legally or operationally be shipped by the available carrier service.
Check expiration, damage, hazardous-material classification, theft controls, temperature requirements, and brand restrictions where applicable.
For regulated items, require the applicable specialist's approval rather than encoding a generic assumption.

**Output:** eligible SKU-node pairs and the reason codes for exclusions.

**Branch:** if an item is eligible only with a special carton or service, use its actual resource and cost requirements.
If no permitted route exists, do not offer that commitment.

Eligibility is different from profitability.
A very inexpensive forbidden shipment remains forbidden.
A negative expected-value shipment can be feasible but should require an explicit business decision.

### Step 4: Compute trustworthy available inventory

**Input:** on-hand record, existing reservations, protected stock, known damage, feed timestamps, and any inbound expected before the pick.

**Action:** derive a time-specific ATP quantity without subtracting the same commitment twice.
Use only inbound stock that will be received and available before the pick window, not merely ordered from a supplier.
Protect a unit only once even if it belongs to multiple policy categories.

**Output:** eligible, unreserved units per SKU, node, and time horizon.

**Branch:** if the feed is stale, apply the documented freshness rule: refresh, reduce availability, or exclude the node.
If the balance is below the protected threshold, route elsewhere or seek an authorized exception.

A buffer is not a measurement of the physical state.
Compare recorded and counted inventory and track pick failures by SKU class and inventory age.
A single store-wide confidence factor can hide a high-risk product category.
The reservation must be atomic with the availability decision or protected by an equivalent concurrency mechanism.
A stale read followed by two reservations can oversell one unit.

### Step 5: Estimate pick reliability and recovery

**Input:** historical assignment, pick, cancellation, exception, and recovery events, segmented by store and merchandise characteristics.

**Action:** define the event precisely, such as “at least one assigned unit cannot be picked before the cutoff.”
Estimate its probability from comparable assignments and quantify the extra outcomes conditional on failure.
Include repeat search, inventory correction, reroute cost, additional packaging if incurred, and customer-care or compensation costs when caused by the failure.

**Output:** a probability and a conditional incremental cost with a data window and sample size.

**Branch:** if the sample is small, pool similar stores or use a conservative scenario and flag uncertainty.
If failures are correlated across lines or a common inventory feed is stale, do not multiply independent line-success probabilities without evidence.
If the promise cannot survive a reroute, apply a service constraint in addition to the expected cost.

Do not compare a probability per line with a cost per order without transforming the event.
Do not add the full replacement shipment charge when the baseline route would have paid the same charge anyway.
Record the difference between the failure recovery and the original plan.

### Step 6: Build a labor task model

**Input:** observed task times by task, order profile, store layout, and associate skill; loaded pay rates; store schedule.

**Action:** separate travel, search, unit verification, packing, labeling, staging, and exception work.
Use labor minutes rather than elapsed minutes when two people work simultaneously.
State whether observations include walking between tasks, supplies restocking, and paid breaks.
Include variable time for additional units and parcels where evidence supports it.

**Output:** expected labor minutes per order and uncertainty, plus a rule for incremental expense and opportunity cost.

**Branch:** if a shift extension is needed, include the actual wage block, premiums, and any applicable constraints.
If existing paid capacity is genuinely idle, incremental wage cash may be zero.
If those minutes displace another valuable activity, include the opportunity cost of the best displaced activity, not an invented second wage.

Time-motion studies can miss unusually difficult searches, label failures, and high-traffic periods.
Sample across days, shifts, store layouts, basket sizes, and associates.
Do not use an individual associate's short sample as a punitive standard.
Review the measurement design with store operations and workforce representatives where appropriate.

### Step 7: Establish a time-bucketed capacity plan

**Input:** staffed minutes, planned non-fulfillment work, protected service coverage, breaks, pick productivity, existing commitments, and cutoff times.

**Action:** convert the staffing plan into usable labor minutes for each bucket.
Subtract protected work and already-accepted orders once.
Reserve capacity for variability and operational exceptions according to the service objective.

**Output:** remaining capacity by store, bucket, task, and cutoff.

**Branch:** if the order's required minutes exceed the available bucket or it cannot finish before tender, exclude that store even if its daily capacity is positive.
If pick and pack use separate constrained resources, test both; one aggregate labor number is insufficient.
If labor can be moved from another bucket, model the actual schedule change and the resulting displaced work.

Capacity is not the number of people scheduled times 60 minutes by default.
Customers, receiving, replenishment, breaks, and task switching consume time.
A single aggregate total may conceal that no trained packer or carrier pickup slot is available.
For a promised date, work backward from carrier acceptance to staging, labeling, packing, and pick start.

### Step 8: Price labor causally

**Input:** task minutes, staffing alternatives, wage rules, store demand, and the relevant time bucket.

**Action:** choose a labor treatment consistent with the counterfactual.
For a shift extension, calculate the full incremental block and any premium.
For an already-funded but scarce minute, estimate the best forgone contribution or modeled shadow price.
For idle paid minutes with no meaningful displacement, assign zero incremental wage while still reporting utilized paid time.

**Output:** incremental labor cash cost and a separate opportunity-cost estimate.

**Branch:** if adding one order triggers a four-hour shift, evaluate the block against all orders enabled by that shift rather than pretending the next order pays exactly its own pro-rata minutes.
If a shadow price already represents displaced value, do not also subtract that same lost sale separately.

A fully loaded wage is useful for budgeting and for comparing work content.
It is not automatically the marginal cash cost of a single assignment.
It is also not automatically the value of a displaced store customer.
Report the basis alongside every labor number.

### Step 9: Price packaging and transportation

**Input:** carton and dunnage requirements, parcel dimensions, destination zone, carrier agreement, and pickup calendar.

**Action:** calculate material cost by parcel and carrier cost for the applicable service, including expected accessorials and fuel adjustments if not already in the quoted rate.
Include damaged-shipment expectation when a packaging choice changes damage probability and recovery expense.
Distinguish an invoice charge from a modeled accrual that will later be reconciled.

**Output:** incremental packaging, transportation, and handling amounts per candidate.

**Branch:** if the carrier bills dimensional weight, use the contract's dimensions, rounding, and divisor.
If the parcel cannot be accepted before the pickup cutoff, choose a later valid promise or another candidate.
If a store pickup incurs a fixed minimum, evaluate whether this order changes the minimum payment before allocating it as incremental.

The cheapest carton is not necessarily the cheapest shipment.
A damaged item can create a replacement, refund, reverse shipment, and customer contact.
A larger carton can increase billed dimensional weight.
Avoid counting the same expected damage in both a packaging allowance and a separate exception reserve.

### Step 10: Value local inventory effects

**Input:** forecast local demand until replenishment, stock position, substitution and retention behavior, unit contribution, and markdown outlook.

**Action:** estimate the expected contribution lost if sending the unit prevents an otherwise retained local sale.
Estimate the expected difference in markdown recovery only when the store route changes whether and when the unit sells.
Use contribution rather than retail price and use the same demand horizon for both estimates.

**Output:** expected local opportunity loss and any non-overlapping markdown benefit.

**Branch:** if local demand is high and replenishment is uncertain, protect stock or charge the estimated scarce-unit value.
If the unit is likely to remain unsold until markdown, compare the store shipment with the actual alternative inventory outcome.
If local demand and markdown outcomes are linked, model the states jointly instead of adding contradictory independent benefits.

A local customer who substitutes another profitable product may not represent a full lost sale.
A customer who buys later after replenishment may impose a delay rather than a permanent loss.
A transfer from one store to another is not necessarily a lost enterprise sale.
Marketing or merchandising can provide scenario assumptions, but finance should make the units and counterfactual explicit.

### Step 11: Compare expected outcomes

**Input:** candidate-specific shipping, materials, labor, inventory, service, and failure outcomes, plus common customer economics.

**Action:** compute store advantage relative to the best feasible alternative.
Use either complete outcome trees or a base-case calculation plus incremental conditional adjustments.
Ensure that every probability is multiplied by a cost conditional on the same event.

**Output:** expected advantage, range under key scenarios, and a clear recommendation.

**Branch:** if the advantage is positive and safeguards pass, route to the store.
If negative, choose the best feasible alternative unless an authorized exception applies.
If uncertainty spans the decision threshold, acquire better data, use a conservative rule, or run a controlled pilot rather than implying precision.

For a same-sale comparison, a concise differential is:

```text
Store advantage = DC avoided fulfillment cost
                - store fulfillment cost
                - store incremental failure cost
                - local inventory opportunity cost
                + non-overlapping incremental benefits
```

If the DC has its own failure risk, subtract the difference between store and DC expected failure cost, not the store total alone.
If the same carrier cost is incurred after either outcome, count it only where it differs.
If route choice changes the probability of the sale, add the probability-weighted change in order contribution.

### Step 12: Reserve, execute, and handle exceptions

**Input:** chosen route, inventory reservation, capacity slot, cutoff, and exception policy.

**Action:** confirm both inventory and capacity with an expiry and an auditable order association.
Release both if the order is canceled, reassigned, or expires.
Direct associates to pick, verify SKU and condition, pack, label, and stage for the documented service.

**Output:** a shipment event trail or an explicit exception with a new owner.

**Branch:** on a missing item, search according to the permitted time budget, correct stock through the controlled process, and test alternative nodes and promises.
Do not silently ship a substitute or promise a date the alternative cannot meet.
Contact the customer when a changed commitment or consent is required under policy and applicable law.

The handoff boundary matters.
“Label created” means a label exists; it does not prove the carrier received the parcel.
A carrier acceptance scan can be late or missing, so reconcile tracking with pickup manifests and investigate gaps.
Use event timestamps and state transitions that can distinguish no scan from no pickup.

### Step 13: Reconcile and update

**Input:** routing decision snapshot, labor observations, carrier invoices, pick outcomes, actual deliveries, returns, and store sales data.

**Action:** compare prediction with actual by store, time bucket, SKU class, basket size, and service.
Reconcile invoice adjustments and late-arriving returns without overwriting the original decision-time snapshot.
Review exceptions and attribution disputes with named owners.

**Output:** calibrated estimates, a financial bridge, and documented changes to policy or capacity.

**Branch:** pause a node when safety, inventory accuracy, tender reliability, or customer harm breaches an agreed guardrail.
Restore it only after the owner verifies corrective action.
If routing changes the mix of orders sent to a store, avoid comparing raw averages as if the mix stayed constant.

The decision audit should preserve candidates considered, eligibility, data freshness, costs, constraints, chosen route, override, and outcome.
A model with excellent average cost but frequent preventable broken promises is not a successful operating policy.

## 6. Worked example

### Situation and decision

A retailer receives a two-unit order for one SKU at 10:30 local time on a Tuesday.
The customer has already accepted a Friday delivery commitment.
Store S and DC D can each ship both units in one parcel and meet that promise if they tender on time.
The retailer must choose one route before reserving stock.
Currency is U.S. dollars (USD).
All numbers are illustrative for this order and its relevant fulfillment horizon.

The customer pays the same price under either route.
The product cost of goods sold and payment fee are also identical.
Therefore those common amounts appear in the order P&L but cancel from the route comparison.
The retailer has no evidence that this already-placed order converts differently by route.
No extra customer revenue is credited to the store route.

### Inventory and commitment

| Input | Store S | DC D |
|---|---:|---:|
| Recorded on hand | 8 units | 46 units |
| Already reserved | 2 units | 10 units |
| Protected stock | 3 units | 5 units |
| Eligible ATP | 3 units | 31 units |
| Units required | 2 units | 2 units |
| Complete order feasible on inventory | Yes | Yes |

For Store S, ATP is 8 − 2 − 3 = 3 units.
The 2-unit order leaves 1 eligible unit above the stated protection threshold.
For DC D, ATP is 46 − 10 − 5 = 31 units.
The buffers and reservations are assumed not to overlap.
Neither calculation proves physical availability; failure risk is handled below.

The merchant has approved the two units for parcel shipment.
No hazardous, age-restricted, temperature-controlled, or other special handling is assumed.
A simultaneous order cannot claim the same units because the system will atomically reserve them before assignment becomes final.

### Labor and capacity

Store S's 11:00–12:00 bucket has 60 staffed labor minutes.
The approved plan protects 28 minutes for customer service, replenishment, breaks, and other necessary work.
Existing fulfillment commitments require 18 minutes.
The remaining capacity is 60 − 28 − 18 = 14 labor minutes.
The order is expected to use 12 labor minutes and therefore fits, leaving 2 minutes in this simplified bucket.
A separate approved variability buffer has already been included in protected work; without that policy, a 2-minute residual could be unacceptably fragile.

The 12 minutes consist of:

| Task | Labor minutes |
|---|---:|
| Travel and locate | 4.0 |
| Verify and pick two units | 2.5 |
| Pack and label | 4.5 |
| Stage and update status | 1.0 |
| Total | 12.0 |

The loaded paid rate is USD 18.00 per labor hour.
The work-content cost at that rate is 12/60 × USD 18.00 = USD 3.60.
For this example, the manager can extend paid labor in one-minute increments within the approved schedule, so USD 3.60 is also the assumed incremental cash wage.
Real staffing may have minimum call-in periods, overtime premiums, or spare already-paid time; those alternatives require different calculations.
No additional labor opportunity cost is charged because the approved staffing extension prevents displacement in this example.

The carrier accepts parcels at 15:00 local time.
The store's planned pick, pack, and staging will finish before 12:00, leaving time for the tender process.
The DC's accepted service is also assumed to meet Friday.
Neither route is considered feasible merely because its average historic transit time is short enough.

### Base cost comparison

| Cost component | Store S, USD/order | DC D, USD/order |
|---|---:|---:|
| Incremental labor | 3.60 | 1.70 |
| Packaging materials | 0.85 | 0.65 |
| Carrier and pickup, one parcel | 4.40 | 6.90 |
| Other incremental handling | 0.20 | 0.15 |
| Base fulfillment cost | 9.05 | 9.40 |

Store base cost is USD 3.60 + 0.85 + 4.40 + 0.20 = USD 9.05 per order.
DC base cost is USD 1.70 + 0.65 + 6.90 + 0.15 = USD 9.40 per order.
The store saves USD 0.35 per order before inventory and failure effects.
The rate assumptions are expected incremental charges, not universal published carrier tariffs.
Both candidates use one parcel, so no split cost is added.

### Failure and inventory effects

On comparable assignments, the planning estimate is a 4% store pick-failure probability.
A store failure is assumed to cost an additional USD 8.00 relative to the normal store route, including the incremental search, recovery, and expected customer-care cost.
This conditional amount already nets out any cost that would have been incurred under normal fulfillment.
Expected store failure cost is 0.04 × USD 8.00 = USD 0.32 per assigned order.

The corresponding DC failure probability is 1%, and its incremental conditional recovery cost is USD 6.00.
Expected DC failure cost is 0.01 × USD 6.00 = USD 0.06 per assigned order.
Both amounts are incremental to the base costs above.
These probabilities concern at least one failed unit in the order, not separate independent unit probabilities.
The scenario assumes all permitted recovery paths still satisfy the Friday commitment; if that is not credible, the service risk must be modeled or the route excluded.

Merchandising estimates an expected USD 0.22 of lost local contribution from allocating the two units to this order rather than preserving them for shoppers until replenishment.
This estimate already reflects local demand, substitution, and the contribution on a retained sale.
It is not USD 0.22 per unit.
The DC alternative is assumed not to impose an additional local stock effect.
No markdown benefit is added because the merchant has not established a separate incremental effect.

### Decision arithmetic

| Component | Store S, USD/order | DC D, USD/order |
|---|---:|---:|
| Base fulfillment cost | 9.05 | 9.40 |
| Expected failure cost | 0.32 | 0.06 |
| Expected local opportunity loss | 0.22 | 0.00 |
| Total modeled incremental cost | 9.59 | 9.46 |

Store total is USD 9.05 + 0.32 + 0.22 = USD 9.59 per order.
DC total is USD 9.40 + 0.06 = USD 9.46 per order.
The store advantage is USD 9.46 − USD 9.59 = **−USD 0.13 per order**.
Choose DC D on the stated expected economics because it is feasible and less costly.
Store S's lower carrier cost alone does not decide the route.

The USD 0.13 difference is small relative to uncertainty in the inputs.
The decision is not evidence that every similar order should go to the DC.
A change in actual packaging, labor, failure risk, or local demand can reverse the result.
The routing owner should retain the comparison and monitor prediction error rather than presenting the result as precise to the cent in every case.

### Sensitivity and management interpretation

If Store S uses genuinely idle paid time without displacing any other work, incremental store labor falls from USD 3.60 to USD 0.00.
Store total then becomes USD 5.99 per order, and the store advantage is USD 3.47 per order.
This is a different staffing counterfactual, not a free-labor claim.
The 12 labor minutes must still fit and the manager must confirm that the alternative use is truly valueless.

If the store's carrier charge rises by USD 1.00, store total becomes USD 10.59 and the DC advantage becomes USD 1.13.
If the store's pick-failure rate falls from 4% to 2%, expected store failure cost becomes USD 0.16.
Store total then becomes USD 9.43 and its advantage becomes USD 0.03, assuming all other values remain unchanged.
That tiny modeled advantage should not override a material service-risk guardrail.

If local opportunity loss rises from USD 0.22 to USD 1.22, store total becomes USD 10.59.
This could occur near a local promotion, but the USD 1.22 amount would need evidence rather than being inferred from the promotion alone.
If the DC can no longer meet Friday, the original comparison is invalid: the business must compare the feasible store route with an explicitly modeled later-promise or no-sale alternative.

### The financial view

Assume this illustrative order has USD 60.00 revenue and USD 36.00 cost of goods sold.
Assume a USD 2.00 payment and other common variable charge.
Its common pre-fulfillment contribution is USD 22.00.
After the modeled route costs, expected contribution is USD 12.41 for Store S and USD 12.54 for DC D.
Those values exclude any other common costs not specified in the example.
The store's internal sales credit is deliberately omitted because it does not change enterprise contribution.

For a monthly business case, sum actual or expected route differences over incremental eligible orders.
Then subtract incremental recurring enablement and fixed capability costs over the same month.
Do not multiply USD 0.13 by all online orders: this comparison applies only to orders with these characteristics and both routes available.
Do not book USD 0.13 as a realized saving before observing which route was actually used and estimating its counterfactual.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It reproduces the preceding expected-cost comparison and evaluates a small set of alternatives.
It uses decimal arithmetic for money and reports values rounded to cents only for display.
The model treats failure cost as an incremental conditional cost and local opportunity loss as an expected amount per order.
It does not optimize a network, calculate ATP from raw records, or infer service feasibility.
Those eligibility decisions are provided as inputs.

```python
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

D = Decimal
CENT = D("0.01")


@dataclass(frozen=True)
class Route:
    name: str
    labor: Decimal
    packaging: Decimal
    carrier: Decimal
    other: Decimal
    failure_probability: Decimal
    failure_increment: Decimal
    inventory_opportunity: Decimal
    feasible: bool = True

    def base_cost(self) -> Decimal:
        return self.labor + self.packaging + self.carrier + self.other

    def expected_cost(self) -> Decimal:
        return (
            self.base_cost()
            + self.failure_probability * self.failure_increment
            + self.inventory_opportunity
        )


def money(amount: Decimal) -> str:
    return f"${amount.quantize(CENT, rounding=ROUND_HALF_UP)}"


def choose(routes: list[Route]) -> Route:
    feasible = [route for route in routes if route.feasible]
    if not feasible:
        raise ValueError("No feasible route; revise the offer or escalate")
    return min(feasible, key=lambda route: route.expected_cost())


store = Route(
    name="Store S",
    labor=D("3.60"),
    packaging=D("0.85"),
    carrier=D("4.40"),
    other=D("0.20"),
    failure_probability=D("0.04"),
    failure_increment=D("8.00"),
    inventory_opportunity=D("0.22"),
)
dc = Route(
    name="DC D",
    labor=D("1.70"),
    packaging=D("0.65"),
    carrier=D("6.90"),
    other=D("0.15"),
    failure_probability=D("0.01"),
    failure_increment=D("6.00"),
    inventory_opportunity=D("0.00"),
)

for route in (store, dc):
    print(f"{route.name}: base {money(route.base_cost())}, "
          f"expected {money(route.expected_cost())}")
print(f"Store advantage: {money(dc.expected_cost() - store.expected_cost())}")
print(f"Chosen route: {choose([store, dc]).name}")
```

Expected output:

```text
Store S: base $9.05, expected $9.59
DC D: base $9.40, expected $9.46
Store advantage: $-0.13
Chosen route: DC D
```

### Using the calculation responsibly

The code minimizes expected cost because the example's customer contribution is identical for both feasible routes.
If the routes change the probability of sale, price, or customer compensation, use expected contribution rather than this cost-only ranking.
A route marked feasible must already pass inventory, time-bucket, carrier, product, and promise checks.
The Boolean is not a replacement for those checks.

The model assumes one mutually exclusive route per order.
It does not reserve stock or capacity and is not safe to connect directly to a live order system.
A production system needs atomic reservation, idempotent event processing, concurrency control, monitoring, and an exception workflow.
A batch optimizer additionally needs shared stock and capacity constraints across orders.
Choosing the individually cheapest route independently for every order can overbook a store.

The data class intentionally keeps labor cash, packaging, carrier, failure, and inventory effects separate.
If a conditional failure cost already includes a refund or extra parcel, do not add the same expense again elsewhere.
If a carrier invoice incorporates fuel and accessorials, do not add those charges a second time.
The `min` call resolves an exact tie by input order; a production policy should state its tie-break rule explicitly.
Negative cost advantages are valid; negative probabilities or missing costs are not valid model inputs.

For a multi-order decision, extend the model with order quantities, SKU-node balances, labor-minute constraints, parcel capacities, and promise feasibility.
Use actual cost curves or staffing blocks when costs are discontinuous.
Keep a snapshot of the inputs available at decision time so later evaluation does not leak future information.

## 8. Data, standards, and regulatory context

### Operational data contract

A usable model needs more than a store's monthly sales total.
Each data source should have an owner, a grain, a timestamp, a freshness expectation, and a reconciliation procedure.
Preserve both the event time and the time the event arrived in the analytical system.
Delayed events can otherwise make a valid decision appear to have used information not available at the time.

| Dataset | Grain | Required fields | Main control |
|---|---|---|---|
| Order | Order and line | Order ID, SKU, quantity, price, accepted promise, destination zone | Reconcile lines and cancellations to commerce ledger. |
| Inventory | SKU, node, time | On-hand, reserved, protected, condition, feed time | Compare to counts and explain adjustments. |
| Reservation | Order line, node, event | Quantity, creation, expiry, release, reason | Enforce atomicity and idempotent release. |
| Store labor plan | Store, role, time bucket | Scheduled and protected minutes, skill, cutoff | Manager approves feasible coverage. |
| Work events | Assignment, task, event | Start, finish, associate role, exception | Distinguish labor time from elapsed time. |
| Packaging | Parcel | Materials, dimensions, actual weight, damage | Reconcile to packaging master and claims. |
| Carrier | Parcel and charge | Service, zone, manifest, acceptance, delivery, invoice adjustments | Match invoice and tracking to parcel. |
| Store sales | SKU, store, time | Sales, stock position, promotions, substitution proxies | Evaluate local-demand counterfactual cautiously. |
| Returns | Order line, return event | Reason, condition, refund, recovery, processing | Avoid attributing unrelated returns to routing. |
| Cost master | Cost type, effective date, scope | Rate, currency, unit, version, owner | Reconcile to contracts and finance assumptions. |

A planning estimate and an actual invoice should not occupy the same unlabeled field.
Keep the cost version used for the route and post actuals separately.
Time zone and daylight-saving rules matter for store cutoffs and customer commitments.
Use a shared identifier mapping when the commerce order, store pick task, carrier label, and invoice use different IDs.

### Inventory and identifiers

GS1 standards describe identifiers and barcode practices used to identify trade items and logistical units.
A Global Trade Item Number (GTIN) can identify a trade item where the retailer and supplier use it.
An internal SKU can still be needed for merchant-specific variants and inventory controls.
A barcode scan improves identification but does not prove that a system balance is correct or that an item is undamaged.

Store and DC systems may use different product masters.
Map each identifier to the correct unit of measure, pack size, and effective dates.
A case of six units is not interchangeable with one sellable unit without an explicit conversion.
Trace changes to the product mapping because an incorrect conversion can create apparent phantom inventory at scale.

### Shipping and product rules

Carrier tariff, dimensional-weight formula, service calendar, and accessorials are contractual and service-specific.
The carrier's public packaging guidance is useful for package design, but the applicable agreement and current service guide determine a merchant's charges and acceptance conditions.
Do not assume an illustrative rate in this unit applies to an actual shipment.

Dangerous goods and other regulated products require classification and compliant packaging, marking, documentation, and transport where applicable.
The United States Pipeline and Hazardous Materials Safety Administration (PHMSA) publishes hazardous-materials resources and regulations.
A store eligibility flag cannot replace product-specific regulatory review or carrier acceptance.
Rules differ by jurisdiction, mode, product, and date.

For orders subject to the U.S. Federal Trade Commission's Mail, Internet, or Telephone Order Merchandise Rule, the applicable shipping representations and delay-consent requirements matter to exception design.
The rule is not a universal promise standard for every jurisdiction or transaction.
Consult the rule and legal counsel for the actual offer and facts.
Record the promise that was actually made and the customer communication, not just an internal target.

### Worker safety and privacy

In the United States, Occupational Safety and Health Administration (OSHA) guidance addresses warehouse safety hazards including materials handling.
Store shipping operations may encounter related hazards even though a retail store is not identical to a warehouse.
Applicable legal obligations depend on the workplace and jurisdiction; use the relevant regulations and employer safety process.
An efficiency target must not require unsafe lifting, obstructed exits, or skipped training.

Order addresses, customer contacts, and associate-level task logs can contain personal information.
Limit access to operational need, retain data under the organization's policy and applicable law, and use aggregated reporting where individual detail is not necessary.
Do not infer associate misconduct from one missing item or one unusually long task.
Ask privacy, legal, labor, and security owners to review the actual design.

### Accounting and attribution

Financial reporting rules and management accounting conventions are not identical to a routing objective.
The finance owner should define recognized revenue, cost of goods sold, returns, and cost allocation under the organization's accounting policy.
For operational decisions, retain a separate incremental ledger and reconcile it to reported expenses.
An internal store credit is a transfer within the enterprise, not external revenue.
A fixed corporate allocation may be relevant for a reported store P&L while being unchanged by the next order.

Data sources and legal references below were selected for their primary-source authority.
Their presence does not certify compliance or validate a particular business case.
Verify current versions and local requirements before implementation.

## 9. Industry practice and case evidence

### Target: store network as a fulfillment asset

Target's annual report describes its stores as part of its fulfillment network and discusses fulfilling digital orders through store-based services and shipping.
The report is a company filing, not a controlled experiment isolating the causal profit of one store shipment.
See the Target annual-report source below for the company's current description and reported figures.

The operational lesson is that store inventory, labor, and physical location can support more than walk-in transactions.
The filing does not establish that every store or order should be used for shipping.
It also does not reveal the incremental cost of a particular SKU, store, hour, or carrier lane.
[INFERENCE] A retailer evaluating a similar network should segment economics and capacity rather than applying a single national average.

Target also offers multiple fulfillment modes, including pickup services.
Pickup and parcel shipment share store labor and inventory but have different handoff, packaging, and customer-travel requirements.
Do not transfer the economics of pickup directly to a parcel route.
A store can be effective at one mode and constrained in another.

### Walmart: stores and digital fulfillment

Walmart's annual report describes an omnichannel model that uses stores, clubs, and fulfillment assets to serve customers through several channels.
It provides company-level evidence of the strategic importance of the network, not a unit-level proof that a particular route produces incremental profit.
Its scale, store density, assortment, technology, and carrier arrangements may differ substantially from a smaller retailer's.

[INFERENCE] The relevant transferable practice is to assess the store and network jointly, not to copy a reported national metric as a local operating target.
The same order may consume store inventory, associate minutes, carrier capacity, and future returns processing.
A well-designed measurement system links these events before attributing benefit to a channel.

### Carrier packaging guidance

UPS publishes guidance on selecting and preparing packaging.
This is operational guidance from a carrier, not evidence that a particular carton is optimal for every merchandise category.
A retailer should validate protection against its product mix, carrier requirements, damage experience, and total parcel charges.
The carrier's quoted price and applicable terms remain specific to the service agreement.

[INFERENCE] A packaging trial should compare total expected cost, including materials, labor, dimensional weight, damage, and customer recovery.
Comparing the purchase price of two boxes alone may select a box that increases total cost.
Keep the trial's products, lanes, and damage definitions consistent across alternatives.

### What the public evidence cannot establish

Public annual reports often report aggregate sales or fulfillment activity.
They generally do not reveal the counterfactual DC cost, local lost-sales effect, per-order shadow price, or failed-pick recovery for each route.
A high share of digital orders fulfilled from stores is therefore evidence of operating scale, not by itself evidence of a positive causal return.
Selection matters: the network may send only suitable orders to stores.

Use public evidence to identify plausible operating designs and questions.
Use internal order-level and store-level data, controlled variation where possible, and financial reconciliation to estimate your own effect.
For claims about performance, record source date, denominator, metric definition, and whether the claim is company-reported or independently evaluated.
Do not infer competitor cost structure from your own wage or parcel rates.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Phantom stock | Rising failed picks at low on-hand levels | Recorded stock differs from usable physical stock | Reroute, delay, cancellation | Cycle count, freshness controls, calibrated buffers. |
| Double reservation | Two tasks claim the last unit | Concurrent reads are not protected | Oversell and broken promise | Atomic reservation and idempotent release. |
| Stale inventory feed | Feed age exceeds the store's policy threshold | Sales or adjustments have not propagated | False eligibility | Refresh, restrict, or exclude stale nodes. |
| Daily capacity hides cutoff | Late tasks cluster before pickup | Labor is available after rather than before the deadline | Missed tender | Bucket capacity and work backward from cutoff. |
| Overstated spare labor | Customer wait or shelf gaps rise | Fulfillment displaces unmeasured store work | Lost sales and poor service | Protect coverage and measure opportunity cost. |
| Shift step ignored | Overtime or call-in hours spike | Fractional order cost ignores a minimum staffing block | Negative program contribution | Model staffing blocks and approval thresholds. |
| Task standard too optimistic | Observed pick times exceed standard during peak | Study omitted travel, search, and exceptions | Overbooking | Sample representative tasks and recalibrate. |
| Carrier label mistaken for tender | Many labels lack acceptance scans | Label creation is treated as custody transfer | False on-time reporting | Reconcile manifest, pickup, and tracking. |
| Pickup missed | Parcels staged after carrier arrival | Incorrect calendar or no pickup capacity | Late delivery | Maintain service calendars and stage buffers. |
| Dimensional weight omitted | Invoices exceed quoted parcel cost | Carton size changes billed weight | Margin erosion | Use contract-specific rate and dimension rules. |
| Fragile product underpacked | Damage claims rise in one carton class | Materials are optimized on purchase price only | Refunds, replacements, dissatisfaction | Test packaging and compare total expected cost. |
| Double-counted failure | Risk charge and recovery allowance rise together | Same event appears in two model components | Good routes rejected | Define conditional event tree and cost ownership. |
| Lost local sale overstated | Opportunity charge approaches full retail price | No allowance for margin, substitution, or demand probability | Stock unnecessarily trapped in stores | Use expected incremental contribution and retention. |
| Markdown benefit overstated | All store shipments credited at full price | Sale would have occurred without route | Inflated business case | Estimate incremental counterfactual outcome. |
| Split parcel hidden | Order cost appears low despite multiple labels | Costs joined at line or first-parcel grain | Understated transport and packaging | Aggregate all parcels by order. |
| Store credit mistaken for value | Store P&L improves while consolidated profit does not | Internal transfer treated as external revenue | Misleading executive decision | Eliminate internal transfers in enterprise bridge. |
| Selected-order bias | Stores appear better only for easy baskets | Routing selects nearby or low-cost orders | Wrong causal conclusion | Compare matched or randomized eligible cases. |
| Returns omitted from attribution | Channel profit changes after delayed returns | Returns arrive after the evaluation cutoff | Inflated early contribution | Mature cohorts and estimate outstanding returns. |
| Untracked override | High manual routing share with no reason | Store or central team bypasses policy | Unexplained cost and inconsistent service | Log authority, reason, duration, and outcome. |
| Safety compromised | Blocked staging area or unsafe lifting | Volume target exceeds physical controls | Injury or operational shutdown | Set safe capacity and stop-work escalation. |

A red flag is a prompt for investigation, not proof of its mechanism.
For example, a missing acceptance scan could indicate a missed pickup, a late scan, or a broken data feed.
Preserve operational evidence before changing routing policy or assigning responsibility.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| Same order value under either route | Allows common revenue and product cost to cancel | Promise or availability changes conversion or discount | Model conversion and contribution by offer. |
| Both routes meet the same commitment | Makes service comparable | Later pickup or transit variability | Test cutoff and delivery distribution; apply guardrail. |
| Conditional recovery cost is incremental | Avoids counting ordinary shipment cost twice | Recovery estimate includes normal costs already in base | Audit an event-level cost bridge. |
| Failure probability is per order | Matches the conditional cost denominator | Estimate is per unit or per pick attempt | Rebuild event denominator and account for dependence. |
| Local opportunity loss is incremental | Makes inventory use comparable | Assumed lost customer would substitute or buy later | Estimate retention and replenishment outcomes. |
| Labor extension is divisible | Supports the example's minute-level cash cost | Minimum shifts or overtime rules | Model actual staffing blocks and premiums. |
| Protected minutes cover variability | Makes 14 minutes available meaningful | Arrival spikes or absence consume protection | Backtest bucket service under peaks. |
| Carrier quote is fully incremental | Supports the parcel comparison | Minimum fees, negotiated tiers, or shared pickup | Reconcile to marginal contract terms. |
| One parcel suffices | Avoids split shipment costs | Items require incompatible packaging or origin | Validate parcel plan and physical rules. |
| Capacity and inventory can be reserved | Prevents simultaneous oversubscription | Non-atomic services or delayed release | Test concurrency and reconciliation controls. |
| Historic observations transport to the decision | Supports estimated time and failure rates | Layout, mix, staffing, or policy changes | Monitor calibration by segment and time. |
| Risk can be summarized in expected cost | Allows a simple ranking | Rare severe customer or safety outcomes | Add hard constraints and tail scenarios. |
| Store and DC costs share one horizon | Prevents mixing current and lifetime amounts | Benefits or return costs mature later | Define horizon and discount where material. |
| No additional markdown benefit is credited | Avoids unmeasured double counting | Route changes end-of-season sell-through | Estimate a joint demand and markdown counterfactual. |

These assumptions are not universal truths.
They define the domain in which the illustrative answer is interpretable.
When one fails, update the model and possibly the decision rather than hiding the exception in a general contingency percentage.

## 12. Exercises

### Exercise 1 — Map the operational decision

A three-line online order arrives at 13:00 local time.
One store has all three items in its recorded inventory and a carrier pickup at 16:00.
The store has a scheduled promotion and two associates on the floor.

**Deliverable:** draw the sequence from customer commitment through carrier handoff, with at least six gates or events and an owner for each exception.

**Acceptance criteria:** distinguish inventory reservation from a physical pick; distinguish label creation from carrier acceptance; include labor capacity before the cutoff; identify an alternative or customer communication path.

**Time:** 15 minutes.

### Exercise 2 — Separate work content from incremental wages

An order uses 7 minutes of picking and 5 minutes of packing.
The loaded pay rate is USD 20.00 per hour.
Scenario A uses an approved 12-minute paid extension.
Scenario B uses already-paid idle time with no displaced work.

**Deliverable:** compute work-content cost and incremental wage cash for each scenario; explain whether opportunity cost is zero or unknown.

**Acceptance criteria:** state units, show the division by 60, and explain why the same 12 minutes can have different marginal cash costs.

**Time:** 10 minutes.

### Exercise 3 — Compare complete expected costs

The store has base cost USD 8.40 per order, 5% probability of pick failure, and USD 6.00 incremental cost conditional on failure.
Its local inventory opportunity loss is USD 0.25 per order.
The DC has base cost USD 8.80 per order, 2% probability of failure, and USD 5.00 incremental cost conditional on failure.
The customer contribution and service commitment are identical and both routes are feasible.

**Deliverable:** calculate each route's expected cost and the store advantage; recommend a route.

**Acceptance criteria:** multiply each probability by its own conditional cost, include local opportunity once, and use the DC as the baseline.

**Time:** 15 minutes.

### Exercise 4 — Test a cutoff bucket

A store has 90 staffed labor minutes from 14:00 to 15:00.
Protected store work requires 45 minutes and accepted fulfillment tasks require 32 minutes.
A new order needs 15 minutes and must be staged by 15:00.
Another 40 labor minutes are available after 15:00, but the carrier collects at 15:00.

**Deliverable:** determine whether the store can accept the order under this plan and describe two feasible policy responses.

**Acceptance criteria:** show the time-bucket arithmetic and explain why later capacity cannot satisfy the existing cutoff without a changed service or schedule.

**Time:** 10 minutes.

### Exercise 5 — Critique a store-versus-DC comparison

A report says store shipments cost USD 7.00 per parcel and DC shipments cost USD 8.00 per parcel.
The store sample contains mainly one-item nearby orders; the DC sample contains more large and distant orders.
The report excludes failed picks and local inventory effects.

**Deliverable:** write a short critique and propose a comparison design.

**Acceptance criteria:** name at least three sources of bias or omitted cost, define the unit of analysis and counterfactual, and describe a feasible matching or randomization approach with service safeguards.

**Time:** 20 minutes.

### Exercise 6 — Design a store pilot

A retailer is considering enabling ten stores for ship-from-store over eight weeks.
Each store has different staffing, inventory accuracy, and carrier schedules.
The business wants to learn whether a capacity-aware rule creates value without degrading the store experience.

**Deliverable:** propose a one-page pilot design, including eligibility, assignment, outcome measures, guardrails, owners, and stopping criteria.

**Acceptance criteria:** include a credible comparison group or randomized eligible assignment where ethical and feasible; measure enterprise contribution and customer promise; track missing picks and labor; protect staff and shoppers; define how outcomes are reconciled and who can stop the pilot.

**Time:** 30 minutes.

### Exercise 7 — Calculate program break-even

A store requires USD 1,200 of incremental enablement cost for a one-year decision horizon.
Expected recurring fixed program expense is USD 300 for that year.
Eligible orders provide an expected USD 2.50 of incremental contribution each relative to the feasible alternative before those fixed costs.
Assume this amount is constant over the relevant range and ignore discounting and taxes.

**Deliverable:** compute the minimum whole number of eligible orders needed to break even and list two reasons the result might not hold.

**Acceptance criteria:** include both fixed amounts, show the equation and rounding, and discuss a volume or cost assumption.

**Time:** 10 minutes.

### Exercise 8 — Handle a missing unit

A store has reserved two units but locates only one before the pick cutoff.
A DC has both units but its next eligible carrier service would miss the accepted promise.
The store's associate can continue searching for five minutes under policy.

**Deliverable:** describe the next actions, system state changes, and customer communication decisions.

**Acceptance criteria:** protect the original reservation while appropriate, prevent duplicate fulfillment, use the approved search and inventory-correction process, reassess the promise, and escalate rather than silently substituting or asserting on-time delivery.

**Time:** 15 minutes.

## 13. Self-check answers

### Exercise 1 — Suggested sequence

First, record the accepted customer commitment and evaluate product, inventory, and delivery eligibility.
The routing service checks the store's time-bucket capacity and carrier cutoff, while store operations owns the staffing plan.
Atomically reserve the units and capacity; a failed reservation returns the order to routing.
An associate picks and verifies all units, reports missing or damaged stock, and follows the approved search procedure.
The store packs, labels, and stages the parcel; the carrier accepts it at pickup.
The customer-experience owner handles any changed promise and the inventory owner corrects a confirmed stock discrepancy.
A valid diagram can use different labels if it preserves these boundaries and owners.

### Exercise 2 — Calculation

Twelve labor minutes at USD 20.00 per hour equal 12/60 × 20 = **USD 4.00** of work content.
In Scenario A, the stated paid extension also creates **USD 4.00** of incremental wage cash.
In Scenario B, the stated already-paid idle time creates **USD 0.00** incremental wage cash.
Under the explicit no-displacement assumption, opportunity cost is also USD 0.00 in Scenario B.
In practice that assumption needs evidence; if valuable work is displaced, its value must be estimated separately.
The calculation does not assert that all store labor is free.

### Exercise 3 — Calculation

Store expected failure cost is 0.05 × USD 6.00 = USD 0.30 per order.
Store expected total is USD 8.40 + USD 0.30 + USD 0.25 = **USD 8.95** per order.
DC expected failure cost is 0.02 × USD 5.00 = USD 0.10 per order.
DC expected total is USD 8.80 + USD 0.10 = **USD 8.90** per order.
Store advantage is USD 8.90 − USD 8.95 = **−USD 0.05** per order.
Choose the DC on these modeled economics, subject to the stated feasibility and service conditions.

### Exercise 4 — Calculation

Remaining pre-cutoff capacity is 90 − 45 − 32 = **13 labor minutes**.
The order needs 15 minutes, so it exceeds the bucket by **2 minutes** and should not be accepted under the stated plan.
The 40 minutes after 15:00 cannot reach the 15:00 pickup.
Possible responses include routing to another feasible node or, if customer and policy permit, offering a later commitment.
An authorized schedule change that safely adds at least two pre-cutoff minutes is another possibility; it must be costed and must preserve protected work.

### Exercise 5 — Rubric

A strong critique identifies basket size, destination, service, selection into route, failure recovery, and local inventory effects as potential sources of noncomparability.
It compares complete order-level outcomes, aggregating every parcel and relevant return or recovery cost.
A randomized choice among orders for which both routes are feasible provides a credible contrast if service and safety safeguards are maintained.
Alternatively, match on pre-decision basket, zone, store, time, promise, and other observed factors, while acknowledging unobserved confounding.
Use assignment-time information rather than outcomes recorded after routing to form the comparison.
Report sample size, uncertainty, exclusions, and the share of orders for which both routes were eligible.

### Exercise 6 — Rubric

A strong design specifies store and SKU eligibility, time-bucket capacity, customer promise, and a baseline route.
It assigns comparable eligible orders or store-time periods to policies in a way that supports causal interpretation.
It measures incremental enterprise contribution, on-time delivery to the accepted promise, failed picks, associate minutes, store customer service, and carrier acceptance.
It specifies exposure and order denominators, the observation and returns-maturity window, and treatment of cancellations.
Store operations owns safe staffing and can pause unsafe activity; customer experience owns promise remediation; finance owns the reconciliation; the experiment owner monitors stopping rules.
The design prespecifies thresholds and reviews uncertainty rather than stopping only when a favorable estimate appears.
It does not force an infeasible route for experimental purity.

### Exercise 7 — Calculation

Total incremental fixed cost is USD 1,200 + USD 300 = **USD 1,500 per year**.
Required volume is USD 1,500 / USD 2.50 = **600 eligible orders** over the year.
Because the quotient is already a whole number, no additional rounding is needed.
At 600 orders, expected incremental contribution before fixed cost is USD 1,500, producing zero modeled profit.
The result fails if contribution per order declines with volume, capacity expansion introduces another step cost, or eligible orders are fewer than forecast.
It also omits uncertainty, timing, taxes, and alternative uses of the investment.

### Exercise 8 — Rubric

Continue the approved search for no more than five minutes if it is safe and does not violate policy.
Maintain a clear state for the located and missing units and avoid releasing or duplicating reservations until the reroute decision is committed.
Record the failed pick and correct the stock through the authorized inventory process after the appropriate checks.
Reassess all feasible alternatives, including whether a split shipment, later date, or cancellation is permitted and acceptable.
Escalate the missed original promise to customer experience and communicate or obtain consent as required by policy and applicable law.
Release or transfer inventory and capacity reservations atomically when the selected recovery path is confirmed.
Do not substitute merchandise or claim the original promise is safe without evidence.

## 14. Interview questions

1. **Why can a nearby store be more expensive than a DC?**
   A strong answer covers labor, packaging, carrier terms, inventory risk, and local opportunity cost.
   A weak answer equates shorter distance with lower total cost.
2. **What is the marginal cost of an order using already-paid associates?**
   A strong answer distinguishes cash wage, displaced work, and capacity thresholds.
   A weak answer says the cost is always zero or always the full hourly wage.
3. **How do you compare store and DC when the customer price is identical?**
   A strong answer cancels genuinely common contribution and compares all differing expected consequences.
   A weak answer compares only the parcel invoice.
4. **How would you value a missing pick?**
   A strong answer defines the order-level event, conditional incremental recovery cost, and service impact.
   A weak answer multiplies incompatible per-unit and per-order measures or counts the replacement twice.
5. **Why is daily spare capacity insufficient?**
   A strong answer uses time buckets, task dependencies, carrier cutoff, skill, and variability.
   A weak answer assumes later labor can serve an earlier pickup.
6. **How do you estimate the opportunity cost of store stock?**
   A strong answer considers local demand, replenishment, contribution, substitution, and the actual counterfactual.
   A weak answer charges the full retail price for every shipped unit.
7. **What is the difference between store P&L and enterprise value?**
   A strong answer eliminates internal transfers and explains incentive design separately.
   A weak answer treats channel credit as new external revenue.
8. **How would you measure whether ship-from-store caused savings?**
   A strong answer compares feasible alternatives with randomized or carefully matched assignment and accounts for selection.
   A weak answer compares unadjusted averages for different order mixes.
9. **What would make you pause a store?**
   A strong answer cites defined safety, inventory, service, and labor guardrails with ownership.
   A weak answer pauses only when average parcel cost rises.
10. **How do you handle a missed carrier pickup?**
    A strong answer distinguishes label from acceptance, re-evaluates the commitment, and assigns customer communication.
    A weak answer marks the parcel shipped because the label exists.

## 15. Key terms

- **ATP:** uncommitted eligible units for a node and horizon; see [definitions](#4-definitions-and-vocabulary).
- **Available capacity:** usable resource time after protected work and commitments; see [capacity](#step-7-establish-a-time-bucketed-capacity-plan).
- **Carrier acceptance:** transfer of custody evidenced by an appropriate event; see [execution](#step-12-reserve-execute-and-handle-exceptions).
- **Contribution:** revenue less specified variable costs; see [economics](#three-levels-of-economics).
- **Counterfactual:** outcome under the alternative decision; see [baseline](#step-1-define-the-decision-and-its-baseline).
- **Cutoff:** latest acceptance time for a service and calendar; see [capacity](#step-7-establish-a-time-bucketed-capacity-plan).
- **Dimensional weight:** contract-specific billing weight based on dimensions; see [packaging](#step-9-price-packaging-and-transportation).
- **Enterprise P&L:** consolidated view eliminating internal transfers; see [attribution](#enterprise-value-versus-attribution).
- **Expected exception cost:** event probability multiplied by its conditional incremental cost; see [reliability](#step-5-estimate-pick-reliability-and-recovery).
- **Feasibility:** satisfaction of hard operational and service constraints; see [mental model](#3-the-core-mental-model).
- **Fixed enablement cost:** capability cost over a specified horizon; see [break-even](#exercise-7--calculate-program-break-even).
- **Incremental cash cost:** future cash expense caused by the choice; see [labor](#step-8-price-labor-causally).
- **Inventory opportunity loss:** expected local contribution forgone due to the allocation; see [inventory](#step-10-value-local-inventory-effects).
- **Labor minute:** one person's minute of work, not necessarily one elapsed minute; see [labor model](#step-6-build-a-labor-task-model).
- **Marginal order:** the next order evaluated against a feasible alternative; see [baseline](#step-1-define-the-decision-and-its-baseline).
- **Markdown avoidance:** incremental benefit of changing a likely markdown outcome; see [inventory](#step-10-value-local-inventory-effects).
- **Opportunity cost:** value of the best alternative use of a constrained resource; see [labor](#step-8-price-labor-causally).
- **Parcel:** one physical package with a carrier identifier; see [grain](#step-2-establish-the-order-and-parcel-grain).
- **Phantom inventory:** recorded stock not physically available; see [inventory](#step-4-compute-trustworthy-available-inventory).
- **Pick failure:** assigned item not picked under a defined event rule; see [reliability](#step-5-estimate-pick-reliability-and-recovery).
- **Promise:** customer-facing delivery commitment; see [four kinds of time](#four-kinds-of-time).
- **Reservation:** claim on a unit or capacity slot; see [execution](#step-12-reserve-execute-and-handle-exceptions).
- **Shadow price:** modeled value of additional binding capacity; see [labor](#step-8-price-labor-causally).
- **Split shipment:** an order sent in multiple parcels or from multiple nodes; see [grain](#step-2-establish-the-order-and-parcel-grain).
- **Step cost:** cost triggered by crossing a staffing or capacity threshold; see [labor](#step-8-price-labor-causally).
- **Tender:** physical handoff to the carrier; see [execution](#step-12-reserve-execute-and-handle-exceptions).
- **Time bucket:** interval used to evaluate resources and deadlines; see [capacity](#step-7-establish-a-time-bucketed-capacity-plan).
- **Work-content cost:** task minutes valued at a stated labor rate; see [worked labor](#labor-and-capacity).

## 16. Sources

Sources are primary where practical; company filings are company-reported evidence, not independent causal evaluations.
Verify current documents, effective dates, and jurisdiction before operational use.

1. Target Corporation, [annual reports and proxy statements](https://corporate.target.com/investors/annual-reports). Company filings describing business and fulfillment operations.
2. Walmart Inc., [annual reports and proxies](https://stock.walmart.com/financials/annual-reports-and-proxies/default.aspx). Company filings describing its business and omnichannel model.
3. GS1, [GTIN identification](https://www.gs1.org/standards/id-keys/gtin). Identifier scope and standards context.
4. U.S. Federal Trade Commission, [A Business Guide to the FTC's Mail, Internet, or Telephone Order Merchandise Rule](https://www.ftc.gov/business-guidance/resources/business-guide-ftcs-mail-internet-or-telephone-order-merchandise-rule). U.S. merchant guidance on shipping representations and delays.
5. U.S. Pipeline and Hazardous Materials Safety Administration, [Hazardous Materials Regulations](https://www.phmsa.dot.gov/hazmat/standards-rulemaking/hazardous-materials-regulations). U.S. hazardous-materials regulatory starting point.
6. U.S. Occupational Safety and Health Administration, [Warehousing](https://www.osha.gov/warehousing). Safety resources relevant to materials handling; applicability depends on the workplace.
7. UPS, [How to Pack a Shipment](https://www.ups.com/us/en/support/shipping-support/shipping-special-care-regulated-items/prohibited-items/how-to-pack-a-shipment.page). Carrier packaging guidance; confirm current contractual service requirements.
8. Marshall L. Fisher, Ananth Raman, and Anna Sheen McClelland, “Rocket Science Retailing Is Almost Here,” *Harvard Business Review*, July–August 2000, [article](https://hbr.org/2000/07/rocket-science-retailing-is-almost-here). Retail demand and inventory decision context; not evidence for the numerical example.

The example values, probabilities, and cost assumptions in this unit are constructed for teaching and are not taken from these sources.
Legal and safety references are educational and do not replace professional advice.

## 17. Links

- [Module 23 index](_index.md).
- [Inventory visibility](01_inventory-visibility.md).
- [Available-to-promise](02_available-to-promise.md).
- [Order routing and distributed order management](03_order-routing-dom.md).
- [Module 23 survey](../../../course/23_omnichannel_fulfillment_returns.md).
- [Lab 23.3 — Ship-from-store economics](../../../course/23_omnichannel_fulfillment_returns.md#lab-233--ship-from-store-economics).
- [Lab 23.2 — Order routing](../../../course/23_omnichannel_fulfillment_returns.md#lab-232--order-routing).
