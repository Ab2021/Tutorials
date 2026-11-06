# Cost-to-Serve: Customer, Channel, and Order Economics

> Module 25 — Product, Economics, and Leadership  
> Topic 06 of 10  
> Estimated study time: 4 hours  
> Prerequisites: [baseline and counterfactual](02_baseline-counterfactual.md), [ROI, NPV, and payback](03_roi-npv-payback.md)  
> Version 2026.5

## 1. Why this topic exists

A sales report can show rising revenue while the operation that supports it loses money.
A profitable-looking customer may order frequently in small quantities, require special packaging,
change delivery windows, return a large share of shipments, and use expensive manual support.
A seemingly smaller customer may submit consolidated orders and accept standard delivery.
Revenue and gross margin alone do not reveal those differences.

Cost-to-serve (CTS) assigns the cost of service activities to the demand that consumes them.
It combines commercial and operational evidence without pretending that every accounting cost
can be causally traced to an individual order.
The result is a decision model, not a substitute for financial statements.

A commercial leader needs to know whether to change a price, service promise, or account plan.
A distribution manager needs to know whether an order minimum or delivery-day change relieves a bottleneck.
Finance needs to know which projected savings will actually change cash spending.
Customer service needs to know whether service requests are defects, contractual obligations,
or customer choices before anyone proposes charging for them.
The customer needs an understandable offer and a fair chance to choose among service levels.

The expensive error is not only underpricing an account.
It is also cutting a useful service because an arbitrary allocation makes it look unprofitable,
or claiming a payroll saving when employees remain on payroll doing other necessary work.
Some customers cover shared capacity and enable attractive network economics even when a
fully allocated report shows a loss.
Some apparently profitable orders have negative incremental economics when expedite costs are included.
You need more than one profit view to make the distinction.

This unit builds an auditable cost bridge from revenue to contribution, then to allocated profit.
It separates customer, channel, and order views, including their overlapping dimensions.
It follows a single monthly case through driver rates, reconciliation, and a service redesign.
It also shows when an apparent saving is merely released capacity rather than a cash benefit.

The case is fictional and educational; its observations are not claims about an actual company.
All amounts are US dollars (USD); the reporting period is one month unless explicitly stated.
A month of observation is not automatically representative of a year.
Finance and the relevant commercial, legal, and operational owners must approve real decisions.

## 2. Learning outcomes

After completing the unit, you should be able to:

1. **Define** the order, customer, channel, and period boundaries of a CTS model.
   Exercise 1 asks you to identify the grain and avoid an overlapping total.
2. **Trace and allocate** service costs using defensible activity drivers and capacity denominators.
   Exercise 2 checks rate and idle-capacity arithmetic.
3. **Reconcile** net sales, cost of goods sold, service pools, and unallocated costs to finance.
   Exercise 3 asks you to bridge the general ledger without double counting.
4. **Calculate** contribution and fully allocated customer profit, and explain their different uses.
   Exercise 4 compares the two margins.
5. **Estimate** incremental avoidable cost and distinguish cash savings from capacity release.
   Exercise 5 applies a staffing step and one-time transition cost.
6. **Critique** biased or incomplete driver data and propose an appropriate sensitivity analysis.
   Exercise 6 tests a missing-scan scenario.
7. **Design** a customer-facing service intervention with a credible counterfactual and safeguards.
   Exercise 7 requires a decision memo and a measurement plan.
8. **Communicate** uncertainty and ownership to finance, sales, and operations.
   Exercise 8 asks for an executive escalation and an acceptance rubric.

A correct calculation without the right decision boundary is not sufficient.
State the currency, time horizon, population, data cut, and whether each figure is observed or assumed.

## 3. The core mental model

```text
Customer demand and contract
        |
        v
Order lines --> picks / packs / deliveries / returns / contacts
        |                         |
        |                         v
        |                 activity quantities x resource rates
        |                         |
        v                         v
net sales - product cost - traceable and driven service costs
        |                         |
        v                         v
contribution by order       reconciliation to cost pools
        |                         |
        +------> customer / channel views <------+
                         |
                add justified shared costs
                         |
              fully allocated profitability
                         |
       compare feasible future choices and avoidable cash flows
```

Start with a demand event: an order line, delivery, return, or support contact.
Each event has a timestamp and identifiers that let you link it to a customer and channel.
A service activity consumes resources; the resource cost may be directly observed or estimated.
An allocation rate converts driver units such as picks or deliveries into a share of a cost pool.
A pool must be used once in a chosen profit bridge, even if it appears in several analytical views.

An order is the best grain only for resources actually attributable to an order.
A monthly account review belongs to a customer-period, not an invented order event.
A marketplace subscription can be channel-period cost, not necessarily a cost caused by each sale.
Corporate finance, strategic account coverage, and spare capacity may be shared across all views.
Display the residual explicitly instead of forcing all resource costs into precise-looking order margins.

Customer and channel are dimensions of the same facts, not independent revenue streams.
An order assigned to customer A and channel web appears in both the A view and the web view.
Summing those two view totals double counts that order.
A customer-by-channel pivot partitions the underlying orders only when every order has exactly one
customer and one channel under the chosen attribution rule.
A multi-touch channel journey needs a documented rule or fractional weights summing to one.

A full-cost view answers how the period's resource cost is distributed under a declared policy.
A contribution view answers what revenue remains after specified traceable or variable costs.
Neither alone answers whether dropping a customer or changing a service promise improves cash.
That decision requires a forward-looking counterfactual and a resource-specific avoidability assessment.
A cost can be allocated to an order without being removable when that order disappears.
Conversely, an incremental cost can be avoidable even if a historical pool used a poor allocation.

Driver rates are not laws of nature.
A per-order rate may approximate a stable workload, but breaks when line count, handling time,
carrier geography, or exception frequency varies materially.
Use direct charges where available, measured activity where credible, and transparent proxy allocation
only for a residual that cannot be traced economically.
When a driver is missing, report the missing share and assess the bias; do not silently set it to zero.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Cost-to-serve | Cost of activities supporting a defined sale or customer, USD per period or event | Treating it as a single universal accounting standard |
| Order | Commercial commitment with an order identifier and date | Assuming one order equals one shipment |
| Order line | Product, quantity, and price within an order | Treating lines as physically identical picks |
| Customer | Contracting or economically relevant account under an identity rule | Combining a parent and subsidiaries inconsistently |
| Channel | Route to market under a mutually exclusive attribution rule | Adding channel totals to customer totals |
| Net sales | Gross invoiced sales less recognized discounts, credits, and returns, USD | Using gross sales with net costs |
| Cost of goods sold (COGS) | Recognized product cost matched to sales, USD | Including freight twice when already in product cost |
| Gross profit | Net sales less COGS, USD | Calling it net operating profit |
| Cost pool | Set of resource costs with a defined period and boundary, USD | Mixing activity costs with unrelated overhead |
| Activity | Work such as a pick, delivery, or customer contact | Confusing the activity with its driver |
| Cost driver | Measurable activity or causal proxy, e.g., picks per month | Using revenue share without testing causality |
| Driver quantity | Count, minutes, distance, weight, or other consumption unit | Assuming missing observations equal zero consumption |
| Rate | Pool cost divided by chosen capacity or activity denominator, USD per driver unit | Dividing a monthly pool by annual volume |
| Direct cost | Cost specifically traceable to the object, USD | Treating all variable costs as direct |
| Allocated cost | Share of a pooled cost assigned by an explicit rule, USD | Calling allocation proof of causation |
| Practical capacity | Achievable available resource activity after normal downtime, driver units per period | Using theoretical 24-hour capacity for a one-shift operation |
| Unused capacity cost | Practical-capacity cost not assigned to consumed activity, USD | Hiding it in an inflated customer rate |
| Contribution | Net sales less costs specified in the contribution definition, USD | Assuming every contribution definition is identical |
| Fully allocated profit | Contribution less assigned shared costs, USD | Treating it as cash released on exit |
| Avoidable cost | Future cost that changes between feasible alternatives in a stated horizon, USD | Equating all allocations with avoidable cost |
| Incremental cost | Difference in future cost between alternatives, USD | Including sunk setup cost in both alternatives as if avoidable |
| Fixed cost | Cost insensitive to activity over a specified range and horizon, USD | Assuming it is fixed forever or at every capacity level |
| Variable cost | Cost that changes with its causal activity within a range, USD per driver unit | Treating a blended historical rate as entirely variable |
| Step cost | Cost that changes at a threshold, such as an additional shift, USD | Applying a fractional employee saving without a staffing plan |
| Sunk cost | Past expenditure unchanged by the decision, USD | Including it as an incremental future cash flow |
| Counterfactual | Outcome expected under the relevant alternative in the same period | Comparing a future intervention to an unmatched historical month |
| Reconciliation | Bridge from source and model totals to the finance control total, USD | Hiding residuals inside a customer allocation |
| Return | Reversal or credit associated with an earlier sale and physical reverse flow | Counting a credit as both a revenue reduction and service cost |
| Allocation key | Rule for distributing a pool across objects, e.g., share of measured minutes | Mistaking an administratively convenient rule for a causal driver |
| Cost object | Entity for which cost is measured, e.g., order, account, or channel | Assuming the same model supports every decision grain |
| Gross-to-net | Bridge from list or invoiced revenue to recognized net sales, USD | Omitting rebates or double counting credit notes |
| Service-level agreement (SLA) | Contractual service commitment such as delivery window or response time | Assuming an internal target overrides the contract |
| Cost hierarchy | Classification into unit, order, customer, channel, and organization activity levels | Assigning every hierarchy level using order count |
| Released capacity | Work time or equipment capability no longer consumed, hours or units | Reporting the entire released resource rate as cash savings |
| Effective date | Period in which a mapping, price, or contract rule applies | Applying today's customer hierarchy to all past transactions |

## 5. The mechanics, step by step

### Step 1 — Name the decision and the alternatives

Write the decision as a comparison between feasible choices.
Example: retain the present delivery promise or move consenting accounts to two weekly delivery days.
Specify the account population, decision owner, geography, reporting currency, and horizon.
Record contract constraints, service obligations, and whether price or product assortment may change.
A backward-looking profitability ranking is descriptive; it is not the proposed intervention.

Input: a decision request and the actual contractual service menu.
Output: a one-page decision charter with a baseline and at least one feasible alternative.
Branch: if the alternative is prohibited or cannot be operationally delivered, redesign it before costing.
If the decision is strategic account retention, include commercial consequences and customer response.
If the decision is a shift roster, focus on workload by location and time, not merely monthly totals.

### Step 2 — Establish time, grain, and identity

Choose an accounting period and a transaction date policy.
Use invoice date for revenue recognition when appropriate and operational timestamps for service events.
Build a period bridge for returns, accruals, and late carrier invoices.
Identify order headers, order lines, shipments, returns, contacts, and invoice lines separately.
Create stable surrogate or source keys and effective-dated customer and channel mappings.

Input: order management, warehouse, transportation, finance, and customer master extracts.
Output: a documented event model and a unique-key test for every source.
Branch: if one order has several shipments, attribute each shipment once to its parent order.
If a return crosses periods, recognize its financial credit according to finance's policy and separately
record its reverse-handling activity when performed; disclose the timing mismatch.
If a parent account owns several legal entities, retain legal-entity identifiers and specify roll-up rules.

A join from order headers to both shipment and contact rows can multiply records.
Aggregate each child table to the intended grain before joining, or use a controlled fact model.
A two-shipment, three-contact order must produce two shipments and three contacts, not six of each.
Keep unmatched events in an exception bucket with owners and an aging threshold.
Never delete unmatched costs merely to make the reconciliation balance.

### Step 3 — Build the gross-to-net and product-cost bridge

Start from the finance-controlled sales ledger.
Separate list price, invoiced discounts, rebates, credit notes, taxes, and recognized returns.
Use finance's treatment of tax and revenue recognition consistently across all accounts.
Match product cost to the recognized sales population and remove internal transfers where appropriate.
Identify freight or packaging already included in COGS before building service pools.

Input: sales ledger, credit and rebate schedules, inventory valuation, and chart of accounts.
Output: net sales and COGS per modeled population, with an explicit unassigned residual.
Branch: if rebates are accrued at customer rather than order grain, keep them at customer-period grain
or allocate using a documented commercial policy; do not invent a precise line-level rebate.
If a return reduces sales and reverses product cost, apply both accounting effects once.
Reverse logistics is a separate service activity, not a second revenue reduction.

### Step 4 — Inventory resources and pools

Map each ledger cost account to exactly one model treatment: direct, pooled, excluded, or residual.
Examples include warehouse handling, outbound freight, contact center, returns handling, and account management.
Split mixed accounts if reliable evidence supports separating labor, supplies, depreciation, and purchased services.
Document whether expense is booked gross or net of supplier credits and how accruals are handled.
Identify cost owner, accounting entity, period, and currency for each pool.

Input: trial balance, payroll or time records, carrier invoices, supplier invoices, and cost-center definitions.
Output: pool register with finance totals, classification, and source lineage.
Branch: keep a corporate pool unallocated if its drivers do not support a decision-relevant assignment.
Use a separate fully allocated management view only when the allocation policy is approved and labeled.
When overhead is allocated to a pool upstream, remove it from any later corporate allocation.

### Step 5 — Map activities to the cost hierarchy

Unit-level examples include handling a sellable unit, product-specific packing, and per-unit marketplace fees.
Order-level examples include order entry, order release, and invoice processing.
Shipment-level examples include a delivery stop, a parcel label, and a proof-of-delivery event.
Customer-level examples include account reviews and customer-specific reporting.
Channel-level examples include storefront fees and dedicated channel service teams.
Facility-level examples include building security and baseline network capacity.

Input: process maps, operational interviews, activity logs, and contract terms.
Output: an activity dictionary linking each pool to one or more plausible driver measures.
Branch: if one warehouse pool includes picking and packing with different drivers, split it using measured
time or a justified resource assignment before applying driver-specific rates.
If evidence cannot support the split, use a simpler rate and quantify the consequences of the approximation.
A complex allocation is not better if its inputs are fabricated or too volatile to reproduce.

### Step 6 — Choose and validate drivers

A good driver varies with the work, is observable at the relevant grain, and can be reconciled.
Picking may use pick lines, units, or measured minutes depending on product and process mix.
Transportation may use carrier invoice, shipment, distance, weight, cube, zone, or stop count.
Contacts may use case minutes where reliable, or case counts stratified by complexity.
Return handling may use return lines and inspection minutes, not simply returned revenue.

Input: driver candidates, sample observations, process evidence, and cost-owner interviews.
Output: a driver selection sheet with unit, coverage, timing, provenance, and alternatives.
Branch: use a direct invoice to an order when a carrier charge is uniquely linked and uncontested.
If a truck serves many stops, allocate shared route cost by a documented route-level rule and test alternatives.
If high-volume records lack customer keys, do not conclude that their cost is zero.
Assign them to an unknown bucket and investigate their size and likely bias.

Test whether each candidate predicts workload within the relevant segment.
A strong association in one facility does not establish the same relationship in another.
Segment by channel or fulfillment method if the same driver implies materially different work.
Compare periods to distinguish a stable relationship from a promotion or outage.
Ask process owners to identify causation; correlation alone cannot establish avoidability.

### Step 7 — Set rates and handle capacity

For a capacity-based activity, divide eligible resource cost by practical capacity in driver units.
Assign consumed units at that rate, and leave unused practical capacity as a visible residual.
For example, USD 12,000 of resource cost and 1,000 practical pick hours imply USD 12 per pick hour.
If demand uses 800 hours, USD 9,600 is assigned and USD 2,400 remains unused capacity cost.
The USD 2,400 is not an error and is not automatically removable.

Input: reconciled pool cost, practical-capacity estimate, and actual eligible driver units.
Output: rate card, assigned amount, and unused or over-capacity variance.
Branch: if actual consumption exceeds stated practical capacity, investigate overtime, measurement error,
a second resource, or an unrealistic capacity estimate; do not hide the overage.
If the model uses actual-volume rates for descriptive allocation, disclose that low utilization inflates
rates and may make a demand reduction appear to raise each remaining customer's unit cost.
A purchased per-delivery invoice may be directly variable without any capacity denominator.

### Step 8 — Assign direct costs, then activity costs

Post each direct cost only once to the event, customer, or channel that incurred it.
Multiply each eligible driver quantity by its corresponding activity rate.
Maintain the pool identifier and rate version on every assignment.
Keep unlinked, unconsumed, and unsupported costs separate from assigned activity.
Roll up only after all facts are expressed at a consistent grain and period.

Input: direct-charge table, validated activity facts, and approved rate card.
Output: cost-object facts with assignment method and confidence flag.
Branch: if a direct freight invoice covers a route rather than one order, it is not directly attributable
to each order; use the route allocation rule and label the result allocated.
If customers share a shipment, define the allocation basis and preserve the shipment total.
If multiple account owners claim the same order, resolve ownership before aggregation.

### Step 9 — Produce multiple profit views

Compute gross profit as net sales minus COGS.
Compute the defined contribution as gross profit minus identified direct and activity-based service costs.
Show which service costs are variable, capacity-based, or otherwise allocated.
Add customer or channel overhead only where a policy and rationale are clear.
Show fully allocated profit separately from contribution and retain unallocated residuals.

Input: revenue and product cost facts, direct costs, assigned service costs, and shared cost policy.
Output: an order, customer, and channel profitability bridge with denominator and period.
Branch: for a promotion, use contribution and incremental support costs for the promotion decision.
For annual budgeting, finance may also need a fully allocated view to fund the entire resource base.
For service redesign, compare future avoidable cash flows; do not use fully allocated profit as the answer.

### Step 10 — Reconcile and control

For each pool, verify direct assignments plus activity assignments plus unassigned or unused residual
and any explicitly signed variance equal the eligible pool cost.
For the entire model, reconcile net sales and COGS to the finance control totals.
Reconcile the count and value of shipments, contacts, and returns to their source control totals.
Explain all material differences with a named owner, an expected resolution date, and a materiality threshold.

Input: source totals, model output, ledger totals, and exception records.
Output: signed reconciliation and a reproducible report version.
Branch: if a pool appears twice in the bridge, remove the duplicate before considering customer rankings.
If the source is incomplete, disclose the amount and population at risk; suspend high-stakes decisions
when the missingness could reverse the outcome.
Keep FX translation policy and cutoff dates aligned with the finance period.

### Step 11 — Translate historical costs into future avoidability

List each affected resource, its owner, future cash outflow, relevant contract, and decision horizon.
Distinguish a per-shipment purchased charge from a salaried employee or leased building.
Estimate the change in driver volume under each alternative, including behavioral response.
Map the volume change to a real cash mechanism: fewer invoices, fewer paid hours, no replacement hire,
a shift closure, a contract renegotiation, or an asset disposal.
Account for termination charges, implementation spending, transition overlap, and capacity constraints.

Input: alternative operating plans, vendor contracts, staffing schedules, and volume forecast.
Output: incremental cash-flow bridge by month, resource, and owner, plus a capacity-release view.
Branch: if staff cannot be reduced or reassigned to displace an approved hire, label time as capacity only.
If a carrier minimum guarantees the same monthly payment, reduced shipment count may not save cash.
If lost customers reduce revenue and product cost, include both effects rather than quoting cost reduction alone.
Sunk historical costs and allocations that do not change must not enter the incremental comparison.

### Step 12 — Choose, communicate, and monitor

Rank feasible alternatives by expected incremental value, service impact, risk, and strategic fit.
Present the status quo counterfactual, assumptions, uncertainty range, and cash versus capacity distinction.
Validate the proposal with sales, operations, finance, customer service, and contract owners.
Test the intervention with an agreed cohort and a comparison design where practical.
Track realized shipments, costs, service outcomes, customer retention, and unintended consequences.

Input: reconciled historical bridge, incremental alternatives, risks, and stakeholder feedback.
Output: decision record, approved implementation owner, and measurement plan.
Branch: if a customer's contract requires the current service, negotiate rather than silently degrade it.
If service deterioration or churn offsets the saving, stop or redesign the intervention.
Compare actuals with the same-period counterfactual, not only with an unusually expensive baseline month.

### Additional mechanics — Selecting the right order-cost grain

An order header can contain multiple products, destinations, and requested delivery dates.
Its commercial identity alone does not describe how the order is fulfilled.
An order line can be picked from multiple locations or shipped in several separate consignments.
A consignment can contain lines from several orders for the same customer.
The operational fact therefore needs an event identifier as well as the commercial identifier.

Begin by identifying whether the management question concerns order acceptance, fulfillment design,
account terms, or channel investment.
For order acceptance, retain order-level price and incremental handling requirements.
For network design, retain shipment-to-route and facility links.
For account negotiations, preserve customer-level activities that have no natural order key.
For channel investment, retain both channel-specific costs and orders flowing through that channel.

Do not apportion a single customer-level contract management expense to each order merely because
an order-level dashboard requires every row to contain a value.
You may display a policy allocation, but retain the original customer-period cost and allocation key.
The analyst must be able to reverse the allocation without losing the underlying fact.
This reversible design supports several management views without pretending that one is the truth.

### Additional mechanics — Driver selection by activity

**Order entry.**
The number of orders can approximate workload when most orders follow the same automated path.
Manual entry minutes are more informative when some orders require rekeying or error correction.
Separate customer-requested manual processing from supplier system failures.
The decision owner can then distinguish a service option from an internal improvement opportunity.

**Picking.**
Order lines are useful if each line requires a comparable number of visits and handling actions.
Unit counts are more relevant for piece picking than full-pallet movements.
Measured labor minutes may be better when travel distance, product weight, or temperature zone differs.
Check whether measured time includes waiting, training, indirect work, and normal breaks.
Include those components in practical capacity and resource pools consistently.

**Packing.**
Packaging material can be traced from bill of materials or consumption records when reliable.
Packing labor may depend on units, carton count, fragility, or special handling.
Separate standard packaging from customer-specific labeling and compliance requirements.
If a supplier chooses extra packaging to correct its own damage issue, investigate that root cause.
Record both the physical event and the reason for the event.

**Outbound transport.**
Parcel cost may depend on zone, billable weight, service level, fuel, and accessorial charges.
Less-than-truckload charges can include minimums and negotiated class or lane rates.
A dedicated route may have a fixed daily charge plus variable distance or stop charges.
Use the contractual tariff to estimate a feasible change, not a flat average where the tariff is nonlinear.
If a route is already operated, removing one stop may save fuel and time but not eliminate the vehicle.

**Delivery failures.**
A failed attempt can generate a second carrier invoice or consume company driver time.
Capture the failure reason before assigning responsibility.
Examples include customer closure, inaccurate address, late supplier dispatch, and carrier failure.
These reasons have different remedies and may have different contractual consequences.
Measure failure rate per attempted delivery and its associated direct charge.

**Returns.**
A return authorization, receipt, inspection, restocking, disposal, and refund are distinct events.
Their costs are not necessarily proportional to the value of the returned goods.
Different product conditions can require different inspection or disposal effort.
If resale is possible, account for recovered inventory under the applicable accounting policy.
Retain a return reason code, but test whether the code is complete and consistently applied.

**Customer support.**
Contact counts are convenient but may mix a one-minute status request with an hour-long dispute.
Minutes, complexity classes, and escalation counts can improve the workload estimate.
Avoid allocating nonproductive time twice if the capacity rate already covers paid breaks and downtime.
Distinguish support caused by a defective delivery from service voluntarily selected by the customer.
If customer-level case links are incomplete, report an unknown bucket rather than a zero.

**Account management.**
A dedicated account manager's time may be directly attributable to a customer or small portfolio.
Shared account management can be assigned using validated time records or a less precise proxy.
Do not use an employee's compensation as an avoidable saving unless the staffing decision is feasible.
Travel and external meeting costs may be directly traceable even when salaries are not.
Review the allocation cadence because account work can be lumpy and seasonal.

**Channel technology.**
Transaction fees may vary with each order, while a storefront license may be fixed for the year.
Payment processing fees can have both percentage and per-transaction components.
Chargebacks, fraud losses, and marketing attribution require separate definitions and controls.
Do not count an agency fee both in channel expense and in a marketing overhead pool.
For a channel-exit decision, include contract termination and customer migration costs.

### Additional mechanics — A route allocation comparison

Suppose a shared route costs USD 600 for one day and serves three customer stops.
The route is a purchased service, but the USD 600 invoice is not directly attributable to one stop.
Stop A receives 10 pallets, stop B receives 6 pallets, and stop C receives 4 pallets.
The total shipment volume is 20 pallets and there are three stops.
An equal-stop allocation assigns USD 200 to each stop.
A pallet allocation assigns USD 300 to A, USD 180 to B, and USD 120 to C.
Both methods reconcile to USD 600, but they answer different allocation questions.

If route cost is mostly a fixed vehicle and each stop adds similar time, stop count may be defensible.
If vehicle capacity and handling depend primarily on pallet volume, pallets may be more informative.
If a long detour is required for one customer, a distance or marginal route model may be preferable.
The analyst should consult dispatch records and test plausible alternatives.
Neither allocation establishes that cancelling one stop saves its assigned share of USD 600.
If the truck must still run, the immediate saving might be only a small fuel or time change.
If the cancellation removes an entire paid route, the avoidable amount may instead be the route charge.

### Additional mechanics — Time-driven rates and resource constraints

Suppose a contact center costs USD 24,000 per month and has 2,000 practical agent hours.
The capacity rate is USD 24,000 / 2,000 hours = USD 12 per hour.
A ten-minute contact consumes one sixth of an hour and is assigned USD 2.
A 45-minute dispute consumes three quarters of an hour and is assigned USD 9.
If 1,600 hours are consumed, assigned capacity cost is USD 19,200.
The remaining 400 hours correspond to USD 4,800 of unused capacity cost.
The USD 4,800 remains part of the accounting pool and may still be paid.

An average contact count rate can obscure case complexity and distort comparisons.
Time measurements also have limitations: agents may multitask or leave cases open between actions.
Validate a sample against actual work rather than interpreting elapsed duration as labor time.
Use a standard time only if the process and mix are stable enough for the intended decision.
If service demand is concentrated at noon, monthly unused hours elsewhere may not relieve the noon peak.
Capacity availability must match the relevant location, skill, and time window.

### Additional mechanics — Avoidability by horizon

The decision horizon determines which costs can change.
A same-day decision may avoid a purchased parcel label but not the warehouse lease.
A three-month decision may avoid temporary labor if a contract permits a reduction.
A one-year decision may avoid an unfilled position or permit a carrier contract to be renegotiated.
A longer network decision may permit a shift closure or a facility consolidation.
Each claim requires a feasible action and an owner, not a generic label of fixed or variable.

Consider a USD 30,000 monthly carrier contract with a minimum of USD 25,000.
If planned volume reduction lowers calculated charges by USD 3,000, the bill can fall to USD 27,000.
The avoidable cash is USD 3,000, assuming there are no other charges or offsets.
If the modeled reduction is instead USD 8,000, the minimum limits the bill to USD 25,000.
The immediate cash saving is then USD 5,000, not USD 8,000.
A change after contract expiry may have different economics and needs a separate scenario.

Consider a team with five paid shifts and enough released work to remove half a shift.
If a shift must remain fully staffed, that change alone does not reduce payroll.
If several approved changes together permit one shift to be removed, quantify the combined saving once.
Do not give every individual project the full value of the same removed shift.
Identify the dependency and designate one benefits owner for the capacity conversion.
Include severance, notice, retraining, and service-risk costs where applicable.

### Additional mechanics — Decision alternatives

**Price adjustment.**
Estimate the revenue effect using the proposed price, eligible volume, and likely customer response.
Compare it with the unchanged-cost baseline and consider contractual notice periods.
A higher price can improve margin but may reduce demand or move orders to another channel.
Present the break-even volume and a downside case to the commercial owner.
Do not assume the full proposed price increase is realized without a customer response model.

**Minimum order or delivery fee.**
Estimate how many orders fall below the threshold and how customers might consolidate or leave.
Model the fee as net revenue only if customers actually pay it and it is recognized under policy.
Model changes in shipment count and picking workload separately from fee revenue.
Check whether the threshold disproportionately affects customers with legitimate constraints.
A pilot can offer a standard free schedule and a priced expedited option where contracts allow.

**Service-tier redesign.**
Specify the service promise for each tier, the price, eligibility, and customer opt-in process.
Measure order cutoff time, transit time, fill rate, exceptions, and access to support.
Ensure the operational design can provide the promised tier before selling it.
An economic saving achieved by missing a promised delivery is not a valid benefit.
Compare customer value and retention across tiers as well as supplier resource cost.

**Process improvement.**
If an internal defect generates contacts or returns, fix the defect before charging the customer.
Estimate the reduction in repeat work and the resource action that turns it into cash.
Include implementation costs, training, and any temporary increase in support demand.
Measure error and rework rates against an appropriate baseline.
Separate productivity improvements from changes in underlying order volume.

**Account exit.**
Model lost net revenue, avoided product cost, avoidable service cash, exit fees, and retained overhead.
Include impacts on shared routes, supplier commitments, and capacity utilization.
Consider strategic effects such as lost cross-selling only where evidence supports an estimate.
An account with negative fully allocated profit may still provide positive incremental contribution.
Conversely, a positive historical contribution can mask an unattractive future contract renewal.

### Additional mechanics — Sensitivity and break-even analysis

Identify the few assumptions most likely to reverse the decision.
For a consolidation proposal these can include shipments avoided, carrier minimums, customer retention,
implementation expense, and the duration of the pilot.
Change one assumption at a time to identify its effect, then combine plausible downside assumptions.
Do not call the combined downside a statistical confidence interval without a probability model.
If inputs are correlated, make the scenario internally consistent rather than choosing impossible combinations.

For the worked case, let avoided shipments per month be the unknown quantity x.
Over three months, the carrier saving is 3 × x × USD 20 per shipment.
With USD 1,500 of one-time cost and no other impacts, break-even is
USD 1,500 / (3 × USD 20) = 25 avoided shipments per month.
At fewer than 25 avoided shipments, the three-month incremental cash result is negative.
At exactly 25, the result is zero before discounting and any other benefits or costs.
At 50, the result is USD 1,500, matching the worked case.

Now suppose a customer concession costs USD 400 per month and is required to secure consent.
Three-month incremental cash becomes 3 × (USD 20 × x - USD 400) - USD 1,500.
The new break-even is x = 45 avoided shipments per month.
At 50 avoided shipments, three-month incremental cash falls to USD 300.
This sensitivity is an illustration, not an assertion that the customer requires a concession.
If the concession changes net sales, record it consistently in the commercial bridge.

### Additional mechanics — Measurement after launch

Before launch, define the eligible cohort and the date on which each customer enters the intervention.
Freeze the baseline definitions and record expected seasonal changes in demand.
Where practical, compare with a similar untreated cohort using the same calendar period.
Document selection differences: customers who volunteer may already be easier to consolidate.
Random assignment can strengthen inference when ethical, contractual, and operational conditions permit.
Where randomization is infeasible, disclose residual selection bias and use sensitivity analysis.

Track shipments per comparable unit of demand, not shipment count alone.
A sales decline can reduce shipment count even when consolidation has no effect.
Track carrier invoice amounts and minimum-charge exposure alongside shipment counts.
Track on-time performance, emergency orders, complaints, revenue, product mix, and retention.
Monitor workload at the constrained facility and time window.
Record implementation and recurring operating expense separately.

At each review, finance compares actual cash to the agreed counterfactual.
Operations confirms whether any staffing or carrier action occurred.
Sales validates whether changed revenue reflects the intervention or unrelated events.
Customer service reviews complaints and exceptions without exposing unnecessary personal information.
The decision owner records whether to continue, revise, or stop based on pre-agreed thresholds.
If assumptions change, preserve the earlier version and explain the revision.
The benefit register should identify the invoice, payroll account, or other cash line that changes.
Its owner should retain evidence of both the intervention and the realized cost reduction.
If another program claims the same invoice reduction, allocate the benefit only once.
An increase in available capacity may still be useful even when it does not reduce expense.
Report its hours, location, and available time separately from the cash benefit.
If that capacity later displaces an approved hire, record the subsequent decision and baseline explicitly.
Do not add the value of the same hours twice when the avoided hire is recognized.

## 6. Worked example

### Case and boundaries

A fictional distributor serves two business customers, Amber and Birch, in one month.
Amber orders through a field-sales channel; Birch uses a web channel.
All dollars are USD and all quantities are monthly, unless stated otherwise.
There are no other customers or channels in this deliberately bounded population.
Actual order, shipment, and contact counts below are assumed complete and reconciled.

| Measure | Amber / field | Birch / web | Total |
|---|---:|---:|---:|
| Net sales, USD | 112,000 | 88,000 | 200,000 |
| COGS, USD | 72,000 | 60,000 | 132,000 |
| Orders, count | 200 | 80 | 280 |
| Warehouse pick hours | 500 | 300 | 800 |
| Outbound shipments, count | 180 | 70 | 250 |
| Support contacts, count | 100 | 20 | 120 |

Net sales already reflects discounts and returns; COGS is matched to the recognized sales.
The warehouse does not include outbound carrier charges or the contact-center pool.
Each shipment is charged directly by a carrier at USD 20 per shipment.
The USD 20 is an illustrative contract rate, not an industry benchmark.
A support contact is assumed to consume a comparable amount of effort in this case.

### Warehouse rate and capacity

The warehouse eligible resource pool is USD 12,000 per month.
Practical capacity is 1,000 pick hours per month.
The activity rate is USD 12,000 / 1,000 hours = USD 12 per pick hour.
Amber consumes 500 hours × USD 12/hour = USD 6,000.
Birch consumes 300 hours × USD 12/hour = USD 3,600.
The assigned warehouse amount is USD 9,600 and unused capacity is 200 hours × USD 12 = USD 2,400.
The control is USD 6,000 + USD 3,600 + USD 2,400 = USD 12,000.

### Direct shipment and support assignments

Carrier cost is 250 shipments × USD 20/shipment = USD 5,000.
Amber's 180 shipments cost 180 × USD 20 = USD 3,600.
Birch's 70 shipments cost 70 × USD 20 = USD 1,400.
The assigned cost equals the carrier invoice total: USD 3,600 + USD 1,400 = USD 5,000.
The support pool is USD 2,400 per month and the driver is 120 contacts.
The descriptive contact rate is USD 2,400 / 120 contacts = USD 20 per contact.
Amber's assignment is 100 × USD 20 = USD 2,000.
Birch's assignment is 20 × USD 20 = USD 400.
The control is USD 2,000 + USD 400 = USD 2,400.
This actual-volume contact rate allocates a period pool; it does not prove USD 20 is avoidable per contact.

### Shared organization cost and profit bridge

A separate USD 10,000 organization pool is assigned in proportion to net sales for the management view.
The rule is a funding convention, not a claim that revenue causes each overhead dollar.
Amber receives 112,000 / 200,000 × 10,000 = USD 5,600.
Birch receives 88,000 / 200,000 × 10,000 = USD 4,400.
The unused warehouse capacity is kept at organization level instead of being allocated to customers.

| Monthly bridge, USD | Amber | Birch | Total |
|---|---:|---:|---:|
| Net sales | 112,000 | 88,000 | 200,000 |
| Less COGS | (72,000) | (60,000) | (132,000) |
| Gross profit | 40,000 | 28,000 | 68,000 |
| Less warehouse activity | (6,000) | (3,600) | (9,600) |
| Less outbound freight | (3,600) | (1,400) | (5,000) |
| Less support assignment | (2,000) | (400) | (2,400) |
| Defined contribution | 28,400 | 22,600 | 51,000 |
| Less shared organization assignment | (5,600) | (4,400) | (10,000) |
| Customer fully allocated profit before idle capacity | 22,800 | 18,200 | 41,000 |

Amber's contribution margin is USD 28,400 / USD 112,000 = 25.36%.
Birch's contribution margin is USD 22,600 / USD 88,000 = 25.68%.
Amber's fully allocated margin before idle capacity is USD 22,800 / USD 112,000 = 20.36%.
Birch's corresponding margin is USD 18,200 / USD 88,000 = 20.68%.
Although Amber has more gross profit, its service pattern consumes substantially more resources.
Neither customer appears loss-making under the stated allocation policy.

The finance reconciliation includes the idle warehouse residual:
USD 41,000 - USD 2,400 = USD 38,600 operating result for this bounded model.
Independently, USD 200,000 - USD 132,000 - USD 12,000 - USD 5,000 - USD 2,400 - USD 10,000
= USD 38,600.
No interest, tax, corporate items outside this population, or other expenses are modeled.
The result is not a statutory profit figure.

### An order-level observation

Suppose one Amber order has net sales USD 420 and product cost USD 270.
It uses 1.5 pick hours, one shipment, and two contacts under the same rates.
Warehouse cost is 1.5 × USD 12 = USD 18; freight is USD 20; support assignment is 2 × USD 20 = USD 40.
Its defined contribution is USD 420 - USD 270 - USD 18 - USD 20 - USD 40 = USD 72.
This order is illustrative within Amber's monthly totals, not additional revenue or cost.
The USD 72 is not necessarily the cash contribution of accepting or rejecting this order.
A single order cannot release fractional salaried labor unless a feasible resource decision makes that true.

### Forward-looking decision

Amber is offered a voluntary consolidated-delivery schedule for a future month.
The baseline forecast is 180 shipments; the alternative forecast is 130 shipments.
These are forecasts, not promises, and the alternative assumes identical net sales and product cost.
The negotiated carrier charges USD 20 for each shipment with no monthly minimum or termination fee.
Avoided carrier cash is (180 - 130) × USD 20 = USD 1,000 per month.
The plan estimates 40 fewer pick hours, but staffing and warehouse rent remain unchanged.
Those 40 hours are released capacity, not a USD 480 cash saving.
Contact volume is assumed unchanged; no support saving is claimed.

A one-time scheduling and customer onboarding expense is USD 1,500 at the start of the pilot.
For a three-month evaluation with stable volumes, undiscounted incremental cash is
3 × USD 1,000 - USD 1,500 = USD 1,500.
The simple undiscounted payback is 1.5 months after launch, assuming savings accrue evenly.
This is not net present value; a longer or material investment case needs a dated cash-flow analysis.
The proposal becomes the operating plan only after approval and customer agreement.
The service target and contractual SLA remain separate from this forecast and are not altered silently.

If the customer requires 20 emergency shipments per month under the new plan, avoided shipments fall
to 30 and monthly carrier savings fall to 30 × USD 20 = USD 600.
Then three-month incremental cash is 3 × USD 600 - USD 1,500 = USD 300 before other impacts.
If retention or revenue changes, include lost contribution and related costs explicitly.
The case does not estimate churn, discount the cash flows, or assert a causal impact from historical ranking.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the two-customer monthly bridge and the consolidation decision.
It models aggregate counts, not transaction-level joins or a production ledger.
The inputs are assumed complete, in USD, and restricted to the two stated customers.
The warehouse rate uses practical capacity; the support rate uses actual contact volume.
Organization costs are assigned by revenue share and idle warehouse cost remains unassigned.
The program prints rounded cents; its internal values are exact decimal amounts.

```python
from decimal import Decimal as D

accounts = {
    "Amber": {
        "channel": "field",
        "sales": D("112000"),
        "cogs": D("72000"),
        "pick_hours": D("500"),
        "shipments": D("180"),
        "contacts": D("100"),
    },
    "Birch": {
        "channel": "web",
        "sales": D("88000"),
        "cogs": D("60000"),
        "pick_hours": D("300"),
        "shipments": D("70"),
        "contacts": D("20"),
    },
}
warehouse_pool = D("12000")
practical_hours = D("1000")
support_pool = D("2400")
carrier_rate = D("20")
organization_pool = D("10000")

sales_total = sum(a["sales"] for a in accounts.values())
hours_total = sum(a["pick_hours"] for a in accounts.values())
contact_total = sum(a["contacts"] for a in accounts.values())
warehouse_rate = warehouse_pool / practical_hours
support_rate = support_pool / contact_total
idle_cost = (practical_hours - hours_total) * warehouse_rate

contribution_total = D("0")
allocated_profit_total = D("0")
for name, a in accounts.items():
    gross = a["sales"] - a["cogs"]
    warehouse = a["pick_hours"] * warehouse_rate
    freight = a["shipments"] * carrier_rate
    support = a["contacts"] * support_rate
    contribution = gross - warehouse - freight - support
    shared = organization_pool * a["sales"] / sales_total
    allocated_profit = contribution - shared
    contribution_total += contribution
    allocated_profit_total += allocated_profit
    print(f"{name}: contribution={contribution:.2f}, "
          f"allocated_profit={allocated_profit:.2f}")

result = allocated_profit_total - idle_cost
ledger_result = (
    sales_total
    - sum(a["cogs"] for a in accounts.values())
    - warehouse_pool
    - sum(a["shipments"] * carrier_rate for a in accounts.values())
    - support_pool
    - organization_pool
)
assert result == ledger_result
print(f"Idle warehouse cost={idle_cost:.2f}")
print(f"Reconciled operating result={result:.2f}")

baseline_shipments = D("180")
planned_shipments = D("130")
one_time_cost = D("1500")
months = D("3")
monthly_cash = (baseline_shipments - planned_shipments) * carrier_rate
incremental_cash = months * monthly_cash - one_time_cost
print(f"Monthly carrier saving={monthly_cash:.2f}")
print(f"Three-month incremental cash={incremental_cash:.2f}")
```

Expected output:

```text
Amber: contribution=28400.00, allocated_profit=22800.00
Birch: contribution=22600.00, allocated_profit=18200.00
Idle warehouse cost=2400.00
Reconciled operating result=38600.00
Monthly carrier saving=1000.00
Three-month incremental cash=1500.00
```

The equality assertion checks this simplified ledger bridge, not the quality of source data or assumptions.
In production, reject zero or negative denominators, investigate over-capacity, and version the rate card.
Reconcile each input table independently before joining; the example deliberately has no joins.
Keep direct carrier charges separate from warehouse and support pool costs.
Do not infer causal savings from the support or warehouse allocation rates.
For a different period, change both pool amounts and denominators consistently.

## 8. Data, standards, and regulatory context

### Data contract

Define a data contract for each source: owner, cadence, grain, key, timezone, cutoff, and permitted use.
The order system supplies order and line identifiers, customer, product, dates, quantity, and selling terms.
The warehouse system supplies actual pick events, work time or calibrated standards, and facility identifiers.
The transportation system supplies shipments, routes, stops, carrier charges, and invoice adjustments.
The customer service system supplies cases, timestamps, case categories, and agent effort when available.
The finance system supplies posted expenses, accruals, credits, exchange rates, and control totals.
Customer master data supplies account hierarchy and effective-dated channel classifications.
Contract records supply price, rebate, minimum-volume, and service commitments.

Do not assume that timestamps across systems describe the same economic event.
An order date, shipment date, invoice date, and return date can belong to different periods.
A design must state whether it reports booked, shipped, invoiced, or recognized activity.
Reconcile both count and amount, because a matching total can hide offsetting duplicate and missing rows.
Retain source record identifiers and the transformation version for audit and reproducibility.

### Identification and electronic messages

GS1 standards define identifiers and data structures used to identify parties, locations, products,
and logistics units; the GS1 General Specifications are a reference for the relevant identifiers.
They can improve cross-system linkage when trading partners actually populate and maintain them.
They do not prescribe a company's cost pool, profit definition, or causal allocation driver.
An identifier can be correct while the hierarchy or effective date used to roll it up is wrong.

Electronic data interchange (EDI) messages can transmit purchase orders, invoices, and shipping notices.
A message standard specifies a representation and business exchange convention, not proof of complete work.
Message acknowledgements and trading-partner mappings need their own control checks.
A shipment notice is not necessarily an invoice, a delivery confirmation, or a carrier charge.
Use source documents and contracts to determine which record is authoritative for each field.

### Accounting and legal boundaries

International Financial Reporting Standard (IFRS) 15 addresses revenue from contracts with customers;
IAS 2 addresses inventories, including the recognition of inventory cost as expense.
These standards apply only where the entity's reporting framework requires or adopts them.
They do not mandate the internal customer-level CTS allocation in this example.
US entities may instead apply US generally accepted accounting principles and their own finance policies.
Ask qualified accounting staff to resolve revenue, rebate, freight, and returns treatment for the entity.
Do not change statutory accounting entries merely because an internal decision model uses a different view.

Customer support records can contain personal data, commercially sensitive pricing, and free-text complaints.
The EU General Data Protection Regulation (GDPR) applies where its territorial and material scope is met;
it imposes requirements including data minimization and appropriate security, subject to its provisions.
It is not a blanket rule that every company worldwide must delete all support data.
Determine the lawful basis, retention policy, access controls, and cross-border rules with counsel.
Aggregate or pseudonymize where suitable, but recognize that pseudonymized data can remain personal data.
This educational material is not legal or accounting advice.

### Quality and controls

Measure missing customer keys, unmatched shipments, duplicated invoices, and driver coverage each period.
Profile by customer, channel, facility, and date rather than reporting only an overall coverage percentage.
A low global missing rate can still conceal a large bias for one customer or one channel.
Maintain a quarantine bucket and assign an owner for corrections and subsequent restatements.
Record when rates, mapping tables, and accounting extracts changed.
Restrict access to named customer profitability and individual contact records on a need-to-know basis.

## 9. Industry practice and case evidence

### Published method: time-driven activity-based costing

Kaplan and Anderson's 2004 Harvard Business Review article describes time-driven activity-based costing.
The published approach uses a cost per unit of supplying capacity and time required by activities.
It distinguishes capacity supplied from capacity used, making unused capacity visible.
The method is a framework for estimation, not a guarantee that any one time equation is accurate.
The fictional warehouse example applies the same practical distinction; it is not a reported company result.

### Public company disclosure: delivery networks

Amazon's 2023 shareholder letter describes regionalizing its US fulfillment network and discusses
reductions in distance and touches associated with deliveries.
That public disclosure illustrates why route, facility, and fulfillment choices can affect service economics.
It does not establish an individual customer's profitability, the allocation rate, or this case's USD values.
[Inference] A CTS model could use regional route and touch data to evaluate similar design alternatives,
but an external reader cannot infer Amazon's account-level cash savings from the letter alone.

### Public company disclosure: parcel pricing

UPS publishes service guides and rate information describing shipment charges and additional fees.
That is evidence that shipment attributes and service choices can affect billed prices under a rate schedule.
Actual shipper invoices can reflect negotiated contracts, minimum charges, discounts, and fuel adjustments.
A published list rate is not proof of a particular customer's contracted marginal cost.
[Inference] An analyst should use the applicable contract and invoice, not a generic average, for a decision.

### Practical operating pattern

Commercial teams often review a waterfall from gross sales through discounts and product margin to service cost.
Operations teams need a complementary view of the activities driving those costs.
Finance needs the bridge back to controlled totals, including unused capacity and corporate residuals.
A governance review should challenge whether a costly activity is demanded by a customer, caused by an
internal defect, required by law or contract, or chosen by the supplier for its own operational convenience.
Do not penalize a customer for supplier-caused rework without evidence and a clear commercial policy.

Consider a promised next-day service that drives extra shipments.
If it is a paid contractual commitment, the business must honor it or negotiate an amendment.
If the supplier splits shipments because of poor inventory availability, the split is not automatically
customer-caused even though the order bears the shipment cost in a descriptive model.
A good redesign identifies both the proximate activity and the organizational cause.
A historical cost allocation is a diagnostic starting point, not a customer blame assignment.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Revenue-share allocation used for every pool | Identical cost percentages across very different orders | Driver ignores workload | Misranked accounts | Trace activity and test alternative drivers |
| Shipment/contact join fanout | Joined counts exceed source counts | Two many-to-one children multiply | Inflated costs | Aggregate each child first and reconcile counts |
| COGS includes freight twice | Same invoice appears in COGS and transport pool | Overlapping pool boundary | Understated margin | Map accounts once and reconcile |
| Idle capacity hidden | Rate rises when throughput falls | Actual volume is used as denominator | Remaining customers inherit idle cost | Show practical capacity and residual |
| Allocated cost called cash saving | Claimed savings have no resource owner | Labor and rent remain committed | Benefits case fails | Tie savings to a specific cash action |
| Customer and channel totals added | Grand total exceeds ledger | Same transaction appears in both dimensions | Double counting | Use one partition or customer-by-channel pivot |
| Missing activity treated as zero | Some accounts have implausibly low effort | Unmatched keys suppress costs | Biased profitability | Quarantine and quantify missing records |
| Returns counted twice | Credit and expense both include same commercial reversal | Revenue and service flows conflated | Understated profit | Separate credit, COGS reversal, and handling |
| Historical mapping applied to all periods | Accounts jump after reorganization | Current hierarchy overwrites old ownership | False trends | Effective-date mapping and restatement policy |
| Blanket order minimum | High-value customers threaten to leave | No segmentation or contractual review | Lost contribution exceeds saving | Compare alternatives with retention sensitivity |
| Published carrier rate substituted | Rate disagrees with invoice | Discounts and minimums omitted | Wrong marginal decision | Use contract and invoice reconciliation |
| Customer blamed for supplier error | Expedites coincide with stockouts | Internal failure causes service activity | Damaged relationship | Assign root cause separately from cost recipient |
| Short period treated as annual pattern | Promotion or outage dominates sample | Seasonality and anomalies ignored | Unstable ranking | Compare representative periods and scenarios |
| Support rate assumed variable | Staff schedule remains unchanged | Pool allocation confused with cash | Phantom savings | Build staffing and redeployment plan |
| Shared fee allocated twice | Channel margin and customer margin both sum full fee | Reaggregation across overlapping dimensions | Inflated expense | Preserve pool identifier and single allocation |
| Contract promise ignored | Proposed service conflicts with SLA | Model omits enforceability | Breach or customer harm | Obtain legal and commercial review |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Two accounts exhaust the modeled population | Makes case totals reconcile | Unmodeled orders or customers exist | Compare source population to ledger |
| Net sales and COGS match the same period | Makes gross profit meaningful | Returns or accruals cross periods | Review cutoff and credit-note bridge |
| Pick hours measure comparable work | Supports one warehouse rate | Product or facility complexity differs | Stratify time observations by process |
| 1,000 hours is practical capacity | Separates activity from idle resources | Overtime or normal downtime differs | Compare schedules and measured availability |
| Carrier price is USD 20 per shipment | Supports direct freight arithmetic | Surcharges, minimums, or zone rates apply | Reconcile invoices and contract |
| Contacts are comparable | Supports one contact allocation rate | Case duration and complexity vary | Sample durations and case categories |
| Sales-share overhead is a funding policy | Completes a management view | Stakeholders interpret it as causation | Label policy and show pre-allocation profit |
| No warehouse cash changes in the pilot | Prevents false benefit claims | A shift can actually close | Review staffing plan and labor terms |
| Same sales and COGS under alternatives | Isolates the shipment change | Customer churn or mix changes | Run retention and mix scenarios |
| Fifty shipments are avoided each month | Supports future benefit estimate | Demand or emergency orders differ | Track matched actual shipment counts |
| One-time implementation cost is USD 1,500 | Completes pilot cash calculation | Integration or training costs omitted | Obtain scoped owner estimates |
| Three months are equally weighted | Allows simple undiscounted payback | Timing or cost of capital is material | Use dated cash flows and discount rate |
| Overhead is not duplicated upstream | Prevents double counting | Allocated costs already embedded in pools | Trace each ledger account once |
| Customer consent and SLA permit consolidation | Makes option feasible | Contract prohibits new schedule | Confirm signed terms and service metrics |

A historical model has measurement uncertainty even when all arithmetic reconciles.
Allocation uncertainty, volume response, price changes, and execution risk should be reported separately.
A reconciliation proves accounting completeness at the selected boundary, not causal validity.

## 12. Exercises

### Exercise 1 — Grain and overlap

A customer has two web orders and one field order; each order has two shipments.
A report shows three customer orders, two web orders, and one field order.
Explain why adding customer and channel totals is invalid and construct a non-overlapping pivot.
Deliverable: a three-row table and a one-sentence grain statement.
Acceptance criteria: the pivot totals three orders and six shipments without double counting.
Time: 10 minutes.

### Exercise 2 — Capacity rate

A warehouse spends USD 18,000 per month for 1,200 practical pick hours.
Two customers consume 450 hours and 550 hours respectively.
Calculate the rate, each assignment, unused capacity cost, and pool control total.
Deliverable: a five-line calculation in USD per month.
Acceptance criteria: the assignments and residual sum to USD 18,000.
Time: 12 minutes.

### Exercise 3 — Reconciliation

Net sales are USD 100,000; COGS is USD 60,000; a warehouse pool is USD 12,000.
Direct freight is USD 5,000 and an organization pool is USD 8,000.
Assigned warehouse cost is USD 9,000 and the rest is unused capacity.
Calculate contribution after assigned warehouse and freight, and reconciled operating result.
Deliverable: a bridge with an explicit idle residual.
Acceptance criteria: the result also matches direct subtraction of all full pools.
Time: 12 minutes.

### Exercise 4 — Customer margin

A customer has net sales USD 50,000, COGS USD 31,000, and service assignments USD 7,000.
An approved shared-cost allocation adds USD 4,000.
Calculate gross profit, contribution, fully allocated profit, and each profit as a share of sales.
Deliverable: a four-row table showing USD and percent.
Acceptance criteria: contribution and allocated profit use different cost boundaries.
Time: 10 minutes.

### Exercise 5 — Cash versus capacity

A proposal removes 120 shipments per month at USD 12 each with no minimum.
It also removes 60 warehouse hours per month, but no staffing or rent changes during the three-month pilot.
One-time implementation is USD 1,800 and all sales and COGS are unchanged.
Calculate three-month incremental cash and describe the noncash operational effect.
Deliverable: a cash bridge and one sentence on released capacity.
Acceptance criteria: no fractional payroll saving is claimed without a staffing action.
Time: 15 minutes.

### Exercise 6 — Missing driver critique

A channel has 1,000 support contacts in the source system, but only 800 join to orders.
The assigned contact rate is USD 10 and the 200 unmatched contacts are not random:
most are associated with a newly onboarded customer.
Quantify the potentially omitted assignment and explain why it is not necessarily cash savings.
Deliverable: a risk statement and two checks for root cause.
Acceptance criteria: identify USD 2,000 at the stated rate and the directional customer bias.
Time: 15 minutes.

### Exercise 7 — Design a service intervention

A sales director proposes a blanket weekly-delivery requirement for every customer.
Design a pilot that compares this proposal with a feasible baseline and protects contracted service.
Include a customer response assumption, a cash mechanism, and an implementation cost.
Deliverable: a one-page decision memo with two scenarios and a monitoring plan.
Acceptance criteria: specify a cohort, period, counterfactual, metrics, owners, and stop rule.
Time: 30 minutes.

### Exercise 8 — Executive communication

A finance executive asks whether a customer showing negative allocated profit should be terminated.
The assigned share includes committed building rent and a corporate allocation.
Deliverable: a concise escalation with a recommendation, missing facts, and next decision date.
Acceptance criteria: distinguish allocation from avoidability, account for lost contribution and contract,
and state an owner for validating the counterfactual.
Time: 20 minutes.

## 13. Self-check answers

### Answer 1

The customer total is already the sum of the web and field orders.
Adding the customer total to the channel totals counts each order twice.
A non-overlapping customer-by-channel pivot has two web orders and one field order.
At two shipments per order those rows have four and two shipments, respectively.
The total is three orders and six shipments at customer-by-channel grain.

### Answer 2

The rate is USD 18,000 / 1,200 hours = USD 15 per hour.
Customer one receives 450 × USD 15 = USD 6,750.
Customer two receives 550 × USD 15 = USD 8,250.
Unused capacity is 1,200 - 450 - 550 = 200 hours, or USD 3,000.
The control is USD 6,750 + USD 8,250 + USD 3,000 = USD 18,000.
The USD 3,000 is unused capacity cost, not necessarily avoidable expense.

### Answer 3

Gross profit is USD 100,000 - USD 60,000 = USD 40,000.
Contribution after assigned warehouse and freight is USD 40,000 - USD 9,000 - USD 5,000 = USD 26,000.
Idle warehouse cost is USD 12,000 - USD 9,000 = USD 3,000.
The operating result is USD 26,000 - USD 8,000 - USD 3,000 = USD 15,000.
Direct subtraction gives USD 100,000 - USD 60,000 - USD 12,000 - USD 5,000 - USD 8,000 = USD 15,000.

### Answer 4

Gross profit is USD 50,000 - USD 31,000 = USD 19,000, or 38% of sales.
Contribution is USD 19,000 - USD 7,000 = USD 12,000, or 24% of sales.
Fully allocated profit is USD 12,000 - USD 4,000 = USD 8,000, or 16% of sales.
Net sales itself is USD 50,000, or 100% of sales, and anchors the four-row table.
The shared allocation is a policy choice, not proof that USD 4,000 disappears on account exit.

### Answer 5

Monthly carrier cash avoided is 120 × USD 12 = USD 1,440.
Three-month gross carrier cash is 3 × USD 1,440 = USD 4,320.
After USD 1,800 implementation, incremental cash is USD 2,520.
The 60 warehouse hours per month are released capacity because no resource spending changes.
A future staffing decision could change the cash conclusion, but it is not part of this scenario.

### Answer 6

The unmatched 200 contacts represent 200 × USD 10 = USD 2,000 of potential assignment.
The customer with missing links likely appears too profitable relative to comparable customers.
Inspect customer identifiers and effective dates; compare source contact totals to joined counts by customer.
Also examine the contact categories and whether the USD 10 rate represents their workload.
Allocation of USD 2,000 does not prove those contacts can be removed or that expense will fall.

### Answer 7 rubric

A strong memo identifies eligible customers and excludes or renegotiates incompatible contracts.
It compares actual outcomes to a same-period baseline or credible control, not just last month's costs.
It quantifies shipment changes, carrier terms, staffing response, transition costs, and customer retention.
It identifies a commercial owner and a finance owner and sets a service or churn stop threshold in advance.
It presents a downside scenario and distinguishes forecast, plan, target, and contractual commitment.
A weak memo uses allocated warehouse cost as cash and assumes customer behavior cannot change.

### Answer 8 rubric

A strong escalation recommends no automatic termination based on allocated profit alone.
It asks for the customer's lost gross contribution, avoidable support and logistics costs, and exit charges.
It identifies rent and corporate allocations that remain in the relevant horizon.
It evaluates price or service renegotiation and identifies contractual and strategic constraints.
Finance owns the cash bridge; sales owns the customer response; operations validates capacity changes.
The decision date should follow completion of those checks and any required contract review.

## 14. Interview questions

1. **How would you define cost-to-serve for an order?**
   A strong answer declares period, grain, pool boundary, and direct versus allocated service costs.
   A weak answer treats all company overhead as a directly caused order expense.
2. **Why can a high-revenue customer have lower contribution than a smaller one?**
   A strong answer discusses net sales, product margin, shipment frequency, handling, and support.
   A weak answer assumes revenue rank determines profitability rank.
3. **How do you select an allocation driver?**
   A strong answer combines process causality, coverage, units, and sensitivity to alternatives.
   A weak answer chooses revenue share solely because it is easy to obtain.
4. **How do you handle a truck serving several customers?**
   A strong answer starts with the route invoice, allocates once with a documented route driver,
   and tests whether an order change actually changes future carrier charges.
   A weak answer assigns the entire invoice to every delivery stop.
5. **What is the role of practical capacity?**
   A strong answer explains the rate denominator, consumed capacity, and explicit idle residual.
   A weak answer spreads unused cost over orders and calls it variable.
6. **What reconciliation would you require?**
   A strong answer bridges sales, COGS, each pool, assigned costs, residuals, and source activity counts.
   A weak answer checks only that the model produces a plausible margin.
7. **When does an allocated cost become avoidable?**
   A strong answer names a feasible action, cash mechanism, horizon, and resource owner.
   A weak answer assumes a customer allocation disappears when the customer leaves.
8. **How would you account for returns?**
   A strong answer separates revenue credits, product-cost reversals, and reverse-handling work by period.
   A weak answer treats every return as a second full sales expense.
9. **How would you design a service consolidation pilot?**
   A strong answer includes customer consent, counterfactual, service metrics, costs, and retention risk.
   A weak answer measures only the historical cost allocated to that customer.
10. **How should channel and customer totals be combined?**
    A strong answer uses a mutually exclusive cross-tab and avoids adding overlapping dimension totals.
    A weak answer sums the customer and channel reports into a larger grand total.
11. **What would you do with incomplete support data?**
    A strong answer quantifies unmatched records, tests concentration, and reports a sensitivity range.
    A weak answer treats missing contacts as zero and ranks accounts anyway.
12. **How would you explain a negative fully allocated margin to sales?**
    A strong answer separates policy allocation, controllable service drivers, and feasible alternatives.
    A weak answer directs sales to terminate the account without a cash or contract analysis.

## 15. Key terms

- [Activity](#4-definitions-and-vocabulary) — work that consumes resources.
- [Allocation key](#4-definitions-and-vocabulary) — a stated rule for assigning a pool.
- [Avoidable cost](#11-assumptions-and-limits) — future cost that differs between feasible options.
- [Capacity rate](#step-7--set-rates-and-handle-capacity) — resource cost per practical capacity unit.
- [Channel](#3-the-core-mental-model) — route-to-market reporting dimension.
- [Contribution](#step-9--produce-multiple-profit-views) — sales less explicitly specified costs.
- [Cost driver](#step-6--choose-and-validate-drivers) — measured proxy for activity consumption.
- [Cost object](#4-definitions-and-vocabulary) — entity receiving a cost measurement.
- [Cost pool](#step-4--inventory-resources-and-pools) — bounded resource costs assigned once.
- [Counterfactual](#step-11--translate-historical-costs-into-future-avoidability) — outcome under an alternative.
- [Customer hierarchy](#step-2--establish-time-grain-and-identity) — effective-dated account roll-up.
- [Direct cost](#step-8--assign-direct-costs-then-activity-costs) — cost traceable to an object.
- [Driver coverage](#data-contract) — share of relevant events with usable activity data.
- [Fully allocated profit](#shared-organization-cost-and-profit-bridge) — contribution after assigned shared costs.
- [Gross profit](#step-3--build-the-gross-to-net-and-product-cost-bridge) — net sales less COGS.
- [Gross-to-net](#step-3--build-the-gross-to-net-and-product-cost-bridge) — bridge from gross to recognized sales.
- [Idle capacity](#warehouse-rate-and-capacity) — practical resource capacity not consumed by activity.
- [Incremental cost](#forward-looking-decision) — future cost difference between alternatives.
- [Order line](#4-definitions-and-vocabulary) — product and quantity entry within an order.
- [Practical capacity](#step-7--set-rates-and-handle-capacity) — achievable capacity after normal downtime.
- [Rate card](#step-7--set-rates-and-handle-capacity) — versioned set of driver conversion rates.
- [Reconciliation](#step-10--reconcile-and-control) — bridge between model and source control totals.
- [Released capacity](#forward-looking-decision) — freed resources, not automatically freed cash.
- [Return](#step-3--build-the-gross-to-net-and-product-cost-bridge) — commercial reversal and possible reverse flow.
- [Service-level agreement](#step-1--name-the-decision-and-the-alternatives) — contractual service commitment.
- [Step cost](#step-11--translate-historical-costs-into-future-avoidability) — cost that changes at a threshold.
- [Sunk cost](#step-11--translate-historical-costs-into-future-avoidability) — past cost unaffected by the decision.
- [Unused-capacity residual](#warehouse-rate-and-capacity) — pool cost not assigned to consumed activity.
- [Variable cost](#4-definitions-and-vocabulary) — cost varying with a specified activity and range.

## 16. Sources

1. Robert S. Kaplan and Steven R. Anderson, “Time-Driven Activity-Based Costing,”
   *Harvard Business Review*, November 2004.
   [Article](https://hbr.org/2004/11/time-driven-activity-based-costing).
   Method reference for capacity cost rates and unused capacity; not validation of the fictional inputs.
2. GS1, [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
   Authoritative standards reference for identifiers and data structures; not a CTS allocation rule.
3. IFRS Foundation, [IFRS 15 Revenue from Contracts with Customers](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-15-revenue-from-contracts-with-customers/).
   Revenue recognition standard; applicability depends on the entity's reporting framework.
4. IFRS Foundation, [IAS 2 Inventories](https://www.ifrs.org/issued-standards/list-of-standards/ias-2-inventories/).
   Inventory accounting standard; consult the applicable current text and professional advisers.
5. European Union, [Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
   Primary legal text for GDPR, including scope, principles, and security provisions.
6. Amazon, [2023 Letter to Shareholders](https://www.aboutamazon.com/news/company-news/2023-letter-to-shareholders).
   Company disclosure about network regionalization; company claims are not independent causal estimates.
7. UPS, [Shipping Services and Rates](https://www.ups.com/us/en/support/shipping-support/shipping-costs-rates.page).
   Public rate information; a shipper's actual contract and invoice govern its realized charges.

## 17. Links

- [Module 25 index](_index.md)
- [Module 25 survey](../../../course/25_product_economics_leadership.md)
- [Stakeholders and the value taxonomy](01_stakeholders-value-taxonomy.md)
- [Baseline and counterfactual](02_baseline-counterfactual.md)
- [ROI, NPV, and payback](03_roi-npv-payback.md)
- [Simulation and optimization production labs index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md)
