# Dual Sourcing and Nearshoring

> Module 14A — Supply Chain Risk, Resilience, and Disruption Analytics  
> Topic 08 of 10  
> Estimated study time: 4 hours  
> Prerequisites: exposure mapping, stress testing, scenario analysis, and buffering versus flexibility  
> Version 2026.5

## 1. Why this topic exists

A procurement manager may have two supplier names in the enterprise system and still have one effective source.
Both suppliers may use the same wafer foundry, port, grid, tooling owner, or software provider.
A second contract does not automatically create a second recovery path.

The decision is whether to fund and maintain another usable production route.
The immediate choices include qualification, geographical location, order shares, capacity reservation, and recovery exercises.
The business consequences include margin, working capital, service, product quality, and the ability to keep a plant running.

A nearshore quotation may look more expensive per unit while reducing transit and demand exposure.
Conversely, a nearby assembly plant may import its critical inputs from the same distant facility as the incumbent.
Distance to the last assembly site is not the total replenishment lead time.

The finance owner needs a like-for-like landed-cost comparison.
The operations owner needs usable capacity by week of a disruption.
The quality owner needs evidence that a substitute is approved for the exact product and manufacturing site.
The commercial owner needs to know what contractual capacity is actually callable.
The planning owner needs to know how fast usable inventory arrives, not how quickly a truck can cross a border.

An expensive mistake is to award a nominal 5% share to a backup that cannot scale beyond that share.
Another is to assume that all lead-time reduction can be converted into inventory savings without changing reorder settings or service targets.
A third is to count the same avoided stockout both as a margin benefit and as a separate revenue benefit.

The unit constructs a decision that distinguishes expected cost from severe-tail exposure.
It treats probability estimates as assumptions, not facts implied by a spreadsheet.
It also distinguishes the cost of creating an option from the cost of using it.
The analysis supports an explicit business choice; it does not prescribe two sources for every item.

## 2. Learning outcomes

After studying this unit you can:

1. Map a product's two proposed sources through critical upstream nodes and identify common-mode exposure.
   Demonstrate this in Exercise 1.
2. Calculate delivered lead time from order release to usable stock and distinguish transit from total lead time.
   Demonstrate this in Exercise 2.
3. Construct a comparable annual landed-cost and qualification-cost baseline for alternative sourcing splits.
   Demonstrate this in Exercise 3.
4. Calculate feasible disruption-period output using qualified capacity, ramp timing, and demand.
   Demonstrate this in Exercise 4.
5. Compare expected annual cost with a tail scenario without presenting assumed probabilities as measured frequencies.
   Demonstrate this in Exercise 5.
6. Diagnose double counting, inadequate volume commitments, and misleading independence assumptions.
   Demonstrate this in Exercise 6.
7. Design a staged qualification and capacity-test plan with decision rights and stopping gates.
   Demonstrate this in Exercise 7.
8. Recommend a sourcing split and identify the evidence that would change the recommendation.
   Demonstrate this in Exercise 8.

The calculations use illustrative data rather than confidential supplier records.
A valid answer states units, the decision horizon, and any material exclusions.

## 3. The core mental model

Dual sourcing buys an additional feasible path, not simply an additional supplier record.
Nearshoring changes the geography of some activities; it does not necessarily change every dependency.
The decision is a portfolio of operating and disruption states.

```text
Demand and product requirements
             |
             v
   Map critical upstream nodes
       /                 \
      v                   v
Existing route       Candidate route
qualification        qualification
capacity, lane       capacity, lane
      \                   /
       v                 v
   Shared dependencies and shocks
             |
             v
  State-specific usable supply by week
             |
             v
  Delivered service, lost contribution,
  landed cost, cash and implementation cost
             |
             v
   Choose split, conditions, and tests
```

A route is available only when all required steps can produce usable material.
In a simple serial chain, failure of a required tier-two input can disable either final assembler.
A nearshore supplier with a shared critical upstream plant can therefore reduce transport time but not eliminate that plant's outage risk.

Baseline orders maintain production knowledge, supplier attention, and a record of product conformance.
They may also make reserved capacity more credible.
However, routine orders do not establish that the source can triple output on demand.
Ramp speed, raw-material availability, labor, tooling, and inspection capacity must all be checked.

The relevant service measure is demand filled with conforming material by its required date.
Capacity that becomes available after the customer's need may reduce recovery time without preventing the initial shortage.
Inventory already in the pipeline may bridge part of the gap.
Inventory assumptions must be consistent with the baseline plan and cannot be spent twice.

The economic comparison has two layers.
First, compare recurring cost in normal operation, including the volume-dependent prices and the maintenance cost of the alternative.
Second, add state-dependent disruption costs, weighted by explicitly assumed probabilities when an expected-value analysis is appropriate.
One-time qualification is treated separately or converted to a comparable annual equivalent.

Do not confuse a forecast, a plan, a target, and a commitment.
A forecast is the estimated demand distribution.
A plan is the proposed order allocation and inventory policy.
A target is the desired service or recovery performance.
A commitment is an enforceable or otherwise agreed supplier obligation, subject to its actual contract terms.
A capacity forecast from a supplier is not automatically a contractual commitment.

Geographical diversification is useful when it changes the hazards that matter and the second route can respond in time.
It is less useful when a common input, common market constraint, or simultaneous demand surge dominates.
The best split depends on normal economics, feasible ramp, exposure, and the risk appetite of the business.

## 4. Definitions and vocabulary

The table uses annual units for the example unless otherwise noted.
A definition does not imply that every company uses the same accounting policy.

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Dual sourcing | Approved use of two routes for the same specified demand, measured by share of units or spend | Two vendor IDs treated as two independent plants |
| Nearshoring | Locating an activity closer to the consuming market, described by actual locations | Assuming a local last step makes the whole chain local |
| Reshoring | Returning an activity to the buyer's home country | Treating reshoring as synonymous with nearshoring |
| Single source | One approved or employed route for the defined item and scope | Assuming there is only one possible supplier in the market |
| Sole source | Only one feasible supplier for the requirement as defined | Using the phrase merely because procurement selected one supplier |
| Primary source | Route receiving the larger planned allocation, in units per year | Assuming its capacity is unlimited |
| Secondary source | Other approved route receiving a planned allocation or held as backup | Assuming it is continuously ready without orders or tests |
| Split | Fraction of baseline units allocated to each route; shares sum to 100% | Applying one share to spend and units interchangeably |
| Qualified capacity | Output per period validated for the specified product, site, and process | Equating nameplate machine capacity with product-specific output |
| Reserved capacity | Capacity set aside under an agreement, in units per period and time window | Assuming a nonbinding forecast reserves capacity |
| Ramp | Change in usable output over time, in units per week | Applying final capacity from the first disruption day |
| Qualification | Evidence and approval that a route meets specified product and process requirements | Treating a supplier questionnaire as production approval |
| First article | Initial item examined against defined requirements | Assuming one sample proves stable production |
| Validation | Demonstration that a process or product meets its intended requirements | Treating a purchase order as technical validation |
| Common-mode risk | A single cause affecting multiple routes, such as one shared facility | Assuming different legal entities have independent failures |
| Correlation | Statistical tendency of outcomes to move together, dimensionless | Treating unknown correlation as zero |
| Exposure | Quantity, contribution, time, or capacity affected by a hazard | Calling exposure alone the probability of disruption |
| Lead time | Elapsed time from a defined start to a defined end, in days or weeks | Reporting only port-to-port transit |
| Usable receipt | Conforming stock released for the intended use | Counting goods still in customs or quality quarantine |
| Cycle stock | Inventory arising from replenishment lot sizing, in units | Treating the full order quantity as average cycle stock |
| Pipeline stock | Inventory between the defined start and end of replenishment, in units | Applying a transit-only time to a total-lead-time formula |
| Safety stock | Inventory held to manage specified uncertainty, in units | Assuming average lead time alone determines it |
| Fill rate | Fraction of demand units supplied under a defined service rule | Mixing line fill, order fill, and unit fill |
| Time to recover | Time until a disrupted node or route returns to a specified capability | Assuming full customer service returns at the same moment |
| Time to survive | Time demand can be covered using available resources under stated conditions | Ignoring allocation priorities and inbound commitments |
| Lost contribution | Unfilled units times contribution lost per unit, in currency, subject to demand disposition | Treating revenue as profit or charging deferred sales as lost sales |
| Expedite premium | Incremental cost of a faster recovery action, in currency per unit or event | Adding the full freight cost when base freight is already included |
| Landed cost | Agreed scope of purchase, freight, duties, handling, and other delivery costs | Comparing one supplier's delivered price with another's factory price |
| Qualification cost | One-time testing, tooling, engineering, audit, and approval cost, in currency | Counting capital expenditure as a recurring annual charge and again as depreciation |
| Annual equivalent | Uniform annual cost with the same present value over a specified horizon and discount rate | Calling an undiscounted average a discounted annuity |
| Working capital | Cash tied up in operating assets less relevant operating liabilities, in currency | Treating the inventory balance itself as an annual expense |
| Carrying charge | Annual cost rate applied to inventory value, with defined inclusions | Adding financing twice when the rate already includes it |
| Expected annual cost | Probability-weighted cost over a stated annual set of states | Presenting it as a guarantee or as a worst-case cost |
| Tail scenario | Adverse outcome examined separately from an average, with a stated severity and duration | Calling a scenario a percentile without a calibrated distribution |
| Trigger | Observable condition that authorizes a specified action | Using an alert without an owner, threshold, or response |
| Contractual allocation | Contract-defined share of scarce output or inventory | Assuming reserved capacity is exclusive without evidence |

## 5. The mechanics, step by step

### Step 1 — Define the decision and the service promise

**Input:** the item specification, demand forecast, customer requirements, planning horizon, and proposed decision date.
**Output:** a decision statement naming the product, consuming locations, volume, service measure, and owner.

Define the unit of analysis narrowly enough that interchangeability is meaningful.
A part number can contain variants with different materials, packaging, or regulatory status.
If one variant cannot use the proposed alternate, split the analysis.
Document whether the service target is unit fill, on-time delivery, plant continuity, or a contractual availability threshold.
Choose a common calendar and currency before comparing bids.

Record the demand forecast as a distribution when the data support one.
The order plan is a proposed allocation, not proof of future demand.
A service target is a management objective, not a guaranteed customer outcome.
A supplier's committed capacity must be evidenced separately from its sales forecast.

If the product has safety or regulatory implications, involve the appropriate quality and compliance owners before modeling replacement supply.
A sourcing decision cannot override required technical or regulatory approvals.
If the product is near retirement, compare the remaining commercial life with the qualification and payback period.

### Step 2 — Map both end-to-end routes

**Input:** supplier disclosures, bills of material, shipping records, process flow, site and lane data.
**Output:** a route map with critical dependencies, confidence, and unknowns.

Identify legal supplier, manufacturing site, tooling location, material origin, sub-tier site, port or border crossing, carrier, and buyer receiving location.
Capture the specific activity at each node rather than only the supplier's headquarters.
Identify shared inputs and shared operational services, including electricity, water, cloud systems, and specialized test equipment where material.
Record the evidence date, source, and owner for each node.
A supplier's refusal to disclose a node is an unknown, not proof of independence.

Separate hazards that affect one site from hazards that affect both routes.
Examples include a plant fire, regional power outage, common resin shortage, and broad demand surge.
A common river basin may matter more than national borders during a drought.
A shared geopolitical constraint can affect geographically distant routes.

If both routes share a bottleneck, estimate how much of each route depends on it.
Explore alternate materials or a third upstream source only if they are within the decision scope and technically permissible.
Avoid assuming that moving final assembly changes the origin of critical inputs.

### Step 3 — Establish the present baseline

**Input:** recent purchase orders, actual receipts, quality releases, cost records, and demand history.
**Output:** a reconciled baseline with distributions and a source-to-use timeline.

Reconcile purchased units to receipts and usage, allowing for inventory movement and scrap.
Separate normal recurring costs from unusual expedites and from one-time project spending.
Identify whether the current price includes delivery, duties, packaging, and inspection.
Translate historical receipts into order-to-usable lead time rather than assuming purchase-order due dates are actual arrivals.

Record the sample period and number of observations.
Averages obscure slow-tail risk, and unusually fast shipments may not represent a repeatable route.
For a changed lane with little history, use tested milestones and scenario ranges instead of pretending that a precise distribution exists.
Exclude incomplete shipments only with a documented censoring treatment; late open orders are not evidence of fast lead time.

### Step 4 — Specify qualification and approval gates

**Input:** engineering specifications, supplier capability, quality plan, audit requirements, and change-control rules.
**Output:** a costed and dated path to usable approval for each site and product.

Identify who owns drawings, tooling, test methods, and intellectual property.
List required material approval, process audit, first articles, production trials, performance tests, and customer sign-off where applicable.
Define acceptance thresholds before seeing test results.
Calculate engineering hours, trial materials, tooling, laboratory fees, audits, travel, and opportunity costs using a consistent accounting scope.

Plan requalification after material, equipment, location, or process changes when the applicable specification requires it.
Separate a supplier's general approval from approval of this particular route.
Where approval is conditional, label the conditions and expiration date.
A backup that is not yet approved is not available in the initial disruption period.

**Branch:** if the alternate cannot satisfy a mandatory requirement, stop the award or redesign the requirement through authorized change control.
Do not assign a low modeled failure probability to an unapproved process and call it approved.
If the route passes trials but has no ongoing orders, budget for periodic production and verification.

### Step 5 — Measure practical lead time

**Input:** timestamped order and shipment milestones for comparable products and lanes.
**Output:** total lead time from release to usable receipt, plus component times and uncertainty.

Use a common start: the buyer releases an order that the supplier can execute.
Use a common end: conforming material is released for the intended operation.
Record order acknowledgment, input procurement, production queue, manufacture, dispatch wait, transit, border clearance, receiving, and quality release.
Add times only when activities are sequential; overlapping work requires a critical-path calculation.

Distinguish average cycle time from an upper-tail planning lead time.
A truck journey may be faster while an infrequent departure schedule, batch minimum, or imported component creates a long wait.
Separate ordinary lead time from emergency lead time, and price any emergency lane independently.

For constant demand and a stable pipeline, the average pipeline quantity is approximately demand per day multiplied by average elapsed replenishment time.
This is a steady-state relationship, not a guarantee for irregular demand or nonstationary lead times.
If the analysis changes average lead time, compute the pipeline reduction using the difference in days and the volume actually placed on the shorter route.
Do not multiply the lead-time reduction by all demand if only part of the demand moves.

Safety stock requires variability and a defined service target as well as mean lead time.
If no defensible variability estimate is available, report a scenario or exclude the benefit explicitly.
Changing reorder points, batching, or planning policies may be necessary to realize an inventory reduction.
The shorter route may also be valuable for postponement and forecast updates even if inventory does not immediately fall.

### Step 6 — Establish ramp and capacity feasibility

**Input:** routine production capacity, available slack, shift plans, raw-material coverage, tooling, labor, and contract terms.
**Output:** a time-phased supply envelope for each scenario.

Measure capacity in conforming units per week, net of normal yield and competing product demand.
Specify when the supplier can start additional production and when those units reach usable stock.
A line's installed output is not automatically available to this customer.
If capacity is shared, document allocation rules and the supplier's other commitments.

Ask for demonstrated production runs at the intended level and evidence of constrained inputs.
Consider whether upstream suppliers and transport providers can support the same ramp.
Check whether a single tool or inspection station caps the route.
Include startup losses, batch constraints, transit, and buyer receiving capacity where relevant.

**Branch:** if ramp starts only after the expected outage ends, its near-term benefit may be small even with a high eventual rate.
If demand is volatile, model weekly demand rather than dividing annual demand evenly.
If the second supplier is itself affected by the scenario, use its reduced capacity rather than its normal ramp schedule.

### Step 7 — Design candidate splits

**Input:** qualification gates, economics, capacity limits, supplier minimums, and demand plan.
**Output:** feasible allocations with an explicit operational rationale.

Compare a single-source baseline with multiple feasible splits such as 90/10, 75/25, and 50/50.
Do not assume that an equal split is automatically optimal.
A small share may preserve practical capability but fail supplier minimums or fail to reserve enough ramp capacity.
A large share may reduce exposure while increasing recurring cost or undermining incumbent pricing.

Use annual unit shares consistently when calculating landed cost.
Check minimum order quantities, lot sizes, seasonal demand, and any take-or-pay minimum.
Apply price tiers to each route's own annual volume, not total portfolio volume.
Check whether suppliers can maintain skill, yield, and quality at the contemplated base load.

A split is a plan, not an outage allocation rule.
Define which source fills which locations, what happens when demand exceeds the plan, and who can authorize an emergency switch.
Document the contract's response-time, notification, allocation, and remedy clauses where agreed.
Legal counsel should interpret enforceability in the relevant jurisdiction.

### Step 8 — Build comparable economics

**Input:** volume-tier quotations, logistics, tariffs or duties where applicable, quality costs, inventory parameters, and qualification budget.
**Output:** a normal-year cost bridge and a separate one-time investment ledger.

Use the same delivery point, currency, tax treatment, product specification, and time horizon for each route.
Model purchased units, yield and scrap consistently; a price per shipped unit differs from a price per accepted unit.
Include incremental freight, packaging, customs brokerage, insurance, and inspection only once.
Separate pass-through duties and recoverable taxes from costs that the buyer ultimately bears.

Add recurring costs of maintaining the route, such as annual audits or reservation fees, where actually incurred.
Treat qualification as a one-time cash outflow at the date paid.
For a one-year operating comparison, either show it separately or compute an annual equivalent with a disclosed horizon and discount rate.
Do not add annualized qualification and the full upfront cost to the same annual total.

Estimate average inventory value using the relevant valuation basis and carrying rate.
The reduction in inventory balance is a cash release, not the same thing as an annual expense saving.
If the carrying rate includes financing, do not add another financing charge for the same inventory.
Show what costs are excluded, including costs that are shared and unchanged across alternatives.

### Step 9 — Build disruption states and calculate service

**Input:** mapped hazards, outage duration, inventory, demand, and route-specific ramp envelopes.
**Output:** a table of usable supply and unmet demand by period and state.

Start with a normal state and at least one primary-only outage.
Add a shared-dependency state and, where relevant, a secondary-only outage or common market constraint.
State the disruption start, duration, scope, demand level, and starting stock for each state.
Avoid assuming every shock begins just after an order unless deliberately testing that adverse timing.

For each period, calculate opening usable stock plus receipts, less fulfilled demand.
Carry the remaining stock forward; never let it be negative.
Unfilled demand becomes lost sales, a backlog, or an allocated shortage according to an explicit business rule.
If demand is backordered, model later recovery and delay penalties rather than treating all unmet units as permanently lost.

When using event-level calculations, compare incremental costs against a baseline that already includes normal purchasing and inventory.
Count emergency purchases at the incremental premium if baseline purchase cost is already included.
Multiply permanently lost units by the applicable lost contribution, not by revenue plus contribution.
Include contractual penalties only when separately applicable and not already embedded in the contribution estimate.

### Step 10 — Compare expected value and tail protection

**Input:** scenario losses, stated probabilities, investment costs, and decision constraints.
**Output:** a cost bridge, sensitivity thresholds, and a separate severe-outcome view.

If scenarios are mutually exclusive and collectively exhaustive within the chosen year, multiply each loss by its annual probability and add the results.
If events can co-occur or recur, a simple mutually exclusive state table is not sufficient without an explicit approximation.
Avoid multiplying individual failure probabilities as if suppliers were independent when they share dependencies.

When probabilities are poorly estimated, use break-even probability and scenario ranges rather than a false point estimate.
A positive expected saving need not imply that the decision meets a liquidity, safety, or service constraint.
A negative expected saving can still be acceptable if leadership explicitly purchases tail protection within its risk appetite.
Do not describe a single selected adverse case as a statistically estimated percentile.

### Step 11 — Decide, contract, and rehearse

**Input:** cost bridge, feasible supply profile, risk appetite, and operational approval.
**Output:** an accountable decision with gates, dates, and a monitoring plan.

Assign an owner for qualification, commercial award, data maintenance, and incident activation.
Specify the planned allocation, minimum call-off, reserved capacity, lead-time commitments, and treatment of shortages if agreed.
Use a change-control process for alternative sites or materials.
Make the activation trigger observable and assign an authorized decision maker and deputy.

Test a realistic switchover using actual ordering and receiving systems.
Verify whether the source can deliver conforming material at the promised rate and by the promised date.
Update the route map after changes in sub-tier suppliers, logistics, or ownership.
Track delivered lead time and quality, not just geographic proximity.
If assumptions fail, revise the split, reserve, or qualification investment rather than hiding the variance in an annual average.

### Operational detail — preserving the option

An alternate route deteriorates when its approval, tooling, staff knowledge, or input allocation lapses.
Design a maintenance plan at the same time as the initial qualification plan.
The purpose of routine activity is to make the future option usable, not merely to report a second supplier.

**Approval maintenance**

1. Keep the approved product revision and manufacturing site in the purchasing master.
2. Link the approval to its test report and approver.
3. Record any customer-specific approval and the products it covers.
4. Identify conditions under which a new trial is required.
5. Review open engineering changes with the alternate supplier before release.
6. Prevent an unapproved site from receiving orders for regulated or controlled product.
7. Track the date of the most recent actual production run.
8. Investigate a long inactivity period before assuming immediate restart.

**Tooling maintenance**

1. Identify the serial number, location, ownership, and authorized custodian of each dedicated tool.
2. Document preventive maintenance and calibration obligations.
3. Check whether one tool is shared between routes.
4. Test whether the tool can support the proposed surge rate.
5. Record the time and practical constraints involved in moving it.
6. Include spare tooling only when technically validated and economically justified.
7. Treat an unavailable or uncalibrated tool as a capacity constraint.
8. Confirm access rights in the applicable agreement rather than assuming title conveys immediate access.

**Input maintenance**

1. List the critical input quantities needed for the normal allocation.
2. List the incremental quantities required by each week of the disruption ramp.
3. Identify which supplier holds the input and where it is stored.
4. Distinguish physical inventory from a supplier promise to procure it later.
5. Check whether another customer can claim the same allocation.
6. Model input replenishment lead time when a shock lasts longer than available stock.
7. Verify shelf life and product-revision compatibility.
8. Include input constraints in the same weekly schedule as the final assembly ramp.

**Commercial maintenance**

1. Reconcile the minimum order quantity with the actual order plan.
2. Review whether the planned share keeps the route commercially viable.
3. Identify which forecast periods are binding, if any.
4. Review reservation fees and the specific rights they purchase.
5. Name the period for which reserved capacity is available.
6. Identify notice requirements and any escalation contacts.
7. Check force-majeure and allocation provisions with legal counsel.
8. Reconcile the annual fee to the finance model.

**Response rehearsal**

1. Trigger a tabletop exercise using a plausible site outage.
2. Identify the person who validates the disruption report.
3. Ask planning to calculate immediate demand and usable inventory.
4. Ask the alternate supplier for time-phased deliverable quantities.
5. Check whether those quantities have the correct product revision.
6. Place a controlled test order through the normal purchasing system.
7. Trace the material to receipt, inspection, and production use.
8. Record actual elapsed time and reasons for delay.
9. Resolve discrepancies between contractual and demonstrated capability.
10. Assign owners and dates to any required corrective actions.

The result of a rehearsal is evidence of performance in that test, not a guarantee for a different shock.
A regional event could also affect labor, roads, energy, or the supplier's other customers.
Repeat the exercise after a material route or process change.

### Operational detail — capacity by week

A useful supply envelope separates capacity at the source from receipts at the buyer.
For each week, start with the supplier's qualified gross production rate.
Subtract planned output for other products and customers where it cannot be reassigned.
Apply expected yield only if the stated rate is gross rather than already in accepted units.
Cap the result at available tooling, labor, material, and inspection capacity.
Apply any contractual allocation limit.
Shift production into future receipt weeks using the appropriate transport and release timing.

For example, an alternate plant may have a nominal 12,000-unit weekly line rate.
If 5,000 units are committed to other products, the initial spare rate is at most 7,000 units per week.
If material availability supports only 4,000 additional units, the input is the binding constraint.
If 4,000 is a gross rate and the expected conforming yield is 95%, usable output is 3,800 units.
If 1,000 of those units are already committed to another customer, the remaining rate is at most 2,800 units.
The precise ordering of these reductions must match how the underlying numbers were measured.
Do not subtract the other-customer commitment twice if the material allocation was already customer-specific.

Production in week 2 is not necessarily a week-2 receipt.
If transport and release require one full week, that production can become usable in week 3 at the earliest.
If dispatch is weekly and the batch misses the cutoff, another week may be needed.
When production happens continuously rather than at the period boundary, use smaller time buckets if weekly timing distorts the decision.
Record whether inventory at the alternate plant is allocated to this buyer or shared.

The mathematical balance for a lost-sales model is:

```text
available[t] = opening[t] + usable_receipts[t]
fulfilled[t] = min(demand[t], available[t])
lost[t]      = demand[t] - fulfilled[t]
closing[t]   = available[t] - fulfilled[t]
opening[t+1] = closing[t]
```

All five quantities are units within the same period and product scope.
Receipts must not include the same in-transit stock in multiple weeks.
If several products compete for the same resource, the receipt envelope must reflect an explicit allocation rule.
If orders are backlogged, carry unmet demand forward rather than calling it lost.
If materials can be substituted, model substitution eligibility and customer priorities explicitly.

### Operational detail — lead-time distributions and policies

The same mean lead time can hide very different outcomes.
Consider two routes with a 30-day average from release to usable receipt.
One arrives close to 30 days on most orders; another alternates between very early and very late arrivals.
The second can require more protection for the same service target even though average pipeline is similar.
The correct comparison requires the distribution of demand during replenishment, not only the average transport duration.

Start by matching completed orders to their actual release and usable-receipt dates.
Separate standard orders from emergency orders if their process and price differ.
Describe the number of observations, time period, and any missing or open orders.
Check whether partial receipts can be used independently.
Compare medians, upper quantiles, and the proportion of orders exceeding the operational threshold.
Use consistent order-size and product filters so that the comparison is not driven by mix.
If the new route has only a handful of trials, report the small sample and use scenario ranges.

Demand and lead time may be dependent.
For example, a demand surge can simultaneously increase supplier queues and customer requirements.
An inventory calculation that assumes independence may understate the upper tail in that situation.
Discuss the direction of the dependence and stress it if observations are too sparse for estimation.
When weekly batches dominate, compare the expected wait to the next batch as well as transportation.
When customs release is variable, measure it separately from physical border crossing.

To realize a steady-state pipeline benefit, review the replenishment parameters and actual order cadence.
A shorter route can leave inventory unchanged if the buyer retains the old reorder point or commits to large batch orders.
The benefit may appear first as a temporary cash release rather than immediately as a permanent cost reduction.
The release can reverse at the end of a finite evaluation horizon if the business restores the former inventory position.
Working-capital improvement is therefore not identical to annual profit improvement.

Document a realization plan:

1. Establish the approved lead-time parameter for each route.
2. Set an appropriate order frequency and minimum order quantity.
3. Change reorder settings through planning governance.
4. Record the baseline pipeline and on-hand inventory by route.
5. Track actual receipt times and late-order frequency.
6. Monitor fill rate while parameters are adjusted.
7. Reconcile realized inventory values with finance.
8. Stop the reduction if service deteriorates or process evidence changes.

### Operational detail — choosing among feasible splits

The correct comparison is not limited to the two options in the numerical example.
Construct a small set of options that are feasible under current supplier terms.
For each option, calculate the routine annual volumes before applying a price tier.
Then check minimum purchases, annual reservation fees, and qualified weekly rates.
Reject an option that fails a mandatory quality or capacity constraint.
Do not use an average cost that assumes volume discounts from mutually incompatible contracts.

A 90/10 plan may be attractive if minimum orders are low and a modest contingency is enough.
It may fail if the alternate requires a larger base load to keep the line staffed.
A 75/25 plan gives the alternate more routine activity, but does not by itself establish its extra capacity.
A 50/50 plan reduces normal reliance on either source, but can sacrifice volume discounts at both.
It can also create more day-to-day exposure to a high-cost or less reliable route.
Each plan must be assessed against the actual demand pattern and the same disruption states.

Include at least these outputs for each feasible split:

1. Annual purchased units by source.
2. Applicable unit price and delivery scope.
3. Recurring audit, reservation, and maintenance charges.
4. Qualification cash outlay and annual equivalent, separately labeled.
5. Average pipeline units under the stated policy.
6. Inventory cash effect and annual carrying effect, separately labeled.
7. Units at risk in a primary-only shock.
8. Units at risk in a secondary-only shock.
9. Units at risk in a common-node shock.
10. Time to first extra usable receipt.
11. Peak qualified and reserved output by week.
12. Approval status and the evidence date.

Plot expected cost against a service or tail-loss measure if management needs a visual trade-off.
An option is dominated only when another feasible option is at least as good on all decision criteria and strictly better on one.
Do not discard an option merely because its expected cost is higher if a binding service constraint changes the decision.
Conversely, do not justify any premium by citing resilience without quantifying the specific protection obtained.

### Operational detail — probability and correlated events

Scenario construction and probability estimation are separate tasks.
First define what occurs, which routes are affected, when capacity returns, and what demand does.
Then decide whether enough evidence exists to assign annual probabilities.
An A-only outage and a shared-chip outage can be mutually exclusive states in a simple annual approximation.
That approximation must say what happens if both events occur or more than one outage occurs in a year.
If overlap is material, enumerate joint states or use an appropriately specified simulation.

Supplier independence has a precise implication in a probability model.
If events A and B were independent, the probability of both would equal the product of their marginal probabilities.
This equality must not be assumed solely because suppliers have different names.
A common upstream node can make both outcomes depend on the same event.
The magnitude and direction of dependence require evidence or a stated scenario assumption.
Shared exposure can also arise from a common buyer demand shock, not only from a supply failure.

Ask three questions before interpreting a probability:

1. What exact event and time horizon does it describe?
2. Does it include short interruptions that inventory can absorb?
3. Does it refer to a site, supplier, lane, or customer-level shortage?

The annual probability of any plant interruption is not necessarily the annual probability of a four-week customer shortage.
Likewise, a historical average across suppliers is not automatically the probability for this specific plant.
Where data are thin, identify plausible ranges with subject-matter owners and disclose their basis.
Calculate the event probability at which the decision changes.
Report tail losses alongside expected costs, including scenarios with no assigned probability when justified.

### Operational detail — monetary comparisons

Use one comparison date and one currency.
Translate supplier quotes using the agreed exchange-rate assumption, and consider a separate currency sensitivity when material.
If a price includes a guaranteed freight component, avoid adding that same freight again.
If a quote is at the factory gate, add the buyer's comparable cost to the agreed destination.
Treat recoverable taxes consistently with company policy.
Label any fixed cost that cannot be avoided over the decision horizon.

The annual-equivalent factor for an upfront cost over `n` years at discount rate `r` is:

```text
factor = r / (1 - (1 + r) ** (-n))
annual equivalent = upfront cost * factor
```

This expression assumes equal annual amounts paid at year-end and a positive discount rate.
It produces an equivalent annual amount with the same present value as the initial payment.
If the rate is zero, divide the upfront amount by the number of years.
If spending occurs in stages, discount each cash flow at its actual date before calculating an equivalent.
If the alternative has salvage value or a required end-of-life payment, include it in the full cash-flow model.

An annual-equivalent comparison is useful for screening, but it can conceal transition cash needs.
The qualification payment may occur before the first operational benefit.
A buyer with a liquidity constraint can reject an otherwise attractive annual equivalent.
Conversely, a cash release from reduced pipeline inventory may be material to financing even if its annual carrying charge is modest.
Show both the annual operating view and the dated cash view when the decision requires them.

## 6. Worked example

### Decision and assumptions

A fictional appliance manufacturer consumes a critical controller in one assembly plant.
The manufacturer forecasts 520,000 controllers per year, or 10,000 units per week over 52 weeks.
The forecast is not a supplier commitment.
The normal order plan is 100% from offshore supplier A in the baseline and 75% A / 25% nearshore supplier B in the proposed alternative.
The management target is to meet weekly controller demand during a four-week A outage.
Supplier B's contracted commitment, if the proposal is approved, is the ramp schedule stated below; the schedule remains subject to a capacity test and executed agreement.
All amounts are illustrative US dollars (USD), and unit prices are per accepted controller delivered to the plant.

Supplier A's comparable landed price is USD 40.00 per unit at the baseline volume.
At the lower volume under the split, A's price is USD 40.30 per unit.
Supplier B's landed price at 130,000 units per year is USD 42.00 per unit.
B requires USD 36,000 per year for ongoing audit and capacity reservation.
Qualifying B costs USD 300,000 upfront at time zero.
The evaluation horizon is five years with a 10% annual discount rate and no terminal value.
The annual equivalent factor is 0.263797, so the annual equivalent qualification cost is approximately USD 79,139.
This factor is the capital-recovery factor, not the simple one-fifth average.

Both routes use the same upstream microchip plant C.
An A-only site outage leaves B's route physically intact.
A C outage stops both assemblers after local component stock is consumed; the simplified scenario below assumes that stock has already been accounted for in the recovery schedule.
That common dependency makes the two routes non-independent in the C scenario.

### Lead-time comparison

The measured or assumed end-to-end components are as follows.
The times are sequential for this example; a real route may have overlap.

| Segment | A, calendar days | B, calendar days |
|---|---:|---:|
| Order acknowledgment and queue | 7 | 5 |
| Input availability | 12 | 10 |
| Production | 8 | 7 |
| Dispatch wait | 5 | 3 |
| Transit and border | 25 | 5 |
| Receipt and quality release | 3 | 2 |
| Total order-to-usable time | 60 | 32 |

Nearshoring shortens the transit-and-border segment by 20 days.
The total route benefit is 28 days because other segments also differ in the illustrative data.
The 28-day difference is not attributable solely to geography.
B still waits 10 days for input availability, partly because chip C is remote.
The figures are average times, not 95th-percentile times or contractual maximums.

The annual demand moved to B is 130,000 units, or 130,000 / 365 = approximately 356.16 units per calendar day.
The steady-state pipeline reduction is 356.16 × 28 = approximately 9,973 units.
At an illustrative inventory valuation of USD 40 per unit, the cash tied up falls by approximately USD 398,904 if planning settings are changed and the transition is completed.
At a 20% annual carrying rate, the illustrative recurring carrying saving is approximately USD 79,781 per year.
The valuation is intentionally held constant for comparison; company accounting may require another basis.
No separate safety-stock saving is claimed because demand and lead-time variability have not been estimated.
The calculation excludes the temporary inventory overlap that may occur during the route transition.

### Normal-year cost bridge

| Item | Baseline, USD/year | 75/25 alternative, USD/year |
|---|---:|---:|
| A purchases | 520,000 × 40.00 = 20,800,000 | 390,000 × 40.30 = 15,717,000 |
| B purchases | 0 | 130,000 × 42.00 = 5,460,000 |
| B annual audit and reservation | 0 | 36,000 |
| Incremental carrying cost relative to baseline | 0 | −79,781 |
| Operating total, relative inventory basis | 20,800,000 | 21,133,219 |
| Annual equivalent qualification | 0 | 79,139 |
| Comparable annual total before disruption | 20,800,000 | 21,212,358 |

The alternative's normal-year premium including annual equivalent qualification is USD 412,358 per year, subject to rounding.
The baseline's absolute inventory carrying charge is omitted from both sides; only the incremental difference is shown.
The inventory balance release of approximately USD 398,904 is not added again as an annual saving.
A full discounted cash-flow model would time the release, its reversal at the horizon, actual qualification outflows, and tax effects explicitly.

### A-only disruption and time-phased supply

Assume A stops supplying the plant for four weeks.
The disruption starts when the plant has 15,000 usable controllers in stock under both alternatives.
B's normal committed receipts are 2,500 units per week under the split.
Extra usable receipts from B are 0, 1,000, 3,000, and 5,000 units in weeks 1 through 4, respectively.
These are receipts at the plant after production, transport, and release, not merely output at B's factory.
Each extra B unit costs USD 3 more than the normal B unit; only the incremental premium is charged below because normal purchases are already in the baseline economics.
Unfilled demand is permanently lost and has USD 80 contribution per unit.
No customer penalty or additional expedite cost is assumed.

| Week | Demand, units | B normal plus extra, units | Split opening stock, units | Split fulfilled, units | Split closing stock, units | Split lost, units |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 10,000 | 2,500 | 15,000 | 10,000 | 7,500 | 0 |
| 2 | 10,000 | 3,500 | 7,500 | 10,000 | 1,000 | 0 |
| 3 | 10,000 | 5,500 | 1,000 | 6,500 | 0 | 3,500 |
| 4 | 10,000 | 7,500 | 0 | 7,500 | 0 | 2,500 |
| Total | 40,000 | 19,000 | — | 34,000 | — | 6,000 |

The baseline has no B receipts and therefore fills 15,000 units from stock and loses 25,000 units.
The split loses 6,000 units and avoids 19,000 lost units.
Its extra B receipts total 9,000 units, so the incremental premium is 9,000 × USD 3 = USD 27,000.
Baseline A-only loss is 25,000 × USD 80 = USD 2,000,000.
Split A-only loss is 6,000 × USD 80 + USD 27,000 = USD 507,000.
The event-level reduction is USD 1,493,000.

The calculation assumes the normal purchase cost of units not purchased from A remains in the annual operating total.
It also assumes B's normal and extra receipts replace those A units without a second full purchase charge.
A more detailed cash model should reverse avoided A purchases and add actual B purchases explicitly.
The premium-only shortcut is appropriate here only because the two normal unit costs are already represented in the annual plan and the incremental emergency differential is defined as USD 3 per extra unit.

### Shared-node disruption and decision

Assume a separate C outage produces a USD 1,600,000 loss under either option.
This is an illustrative scenario cost after accounting for available stock and recovery, not an empirically estimated value.
Assume mutually exclusive annual states: 10% A-only outage, 3% C outage, and 87% normal operation.
These are decision assumptions, not measured frequencies; at most one modeled event occurs per year.

Baseline expected disruption loss is 0.10 × 2,000,000 + 0.03 × 1,600,000 = USD 248,000 per year.
The split's expected disruption loss is 0.10 × 507,000 + 0.03 × 1,600,000 = USD 98,700 per year.
The expected reduction is USD 149,300 per year.
The expected total annual cost is USD 21,048,000 for the baseline and USD 21,311,058 for the split.
The split is therefore USD 263,058 per year more expensive on these assumptions.

Holding all other assumptions constant, the A-only annual event probability needed to offset the USD 412,358 annual premium is 412,358 / 1,493,000 = approximately 27.6%.
This threshold is a conditional sensitivity, not a prediction of outage frequency.
It also changes if ramp, contribution, cost, or lead-time assumptions change.
The shared-node loss does not cancel if the two options have different exposure to C; equality is a specific assumption of this example.

**Recommendation:** retain the single-source baseline on the stated expected-cost criterion, while presenting the A-only service improvement and common-node tail exposure to the risk owner.
The split does not meet the target of zero lost units in the A-only scenario: it still loses 6,000 units.
Management could authorize the premium for a separately stated risk-appetite reason, but should not describe it as a positive expected-cost business case.
Before approval, test B's weekly usable receipts, validate the chip dependency, and verify whether the inventory reduction is actually achievable.
A different outage probability, contribution, reserved capacity, or service constraint can change the decision.

## 7. Code, query, or calculation

The following standard-library Python reproduces the example's annual comparison and A-only stock flow.
It uses constant demand, no backorders, the stated annual-equivalent factor, and mutually exclusive annual states.
The initial inventory is the same in both cases, so the scenario does not automatically inherit the pipeline reduction as a reduction in plant safety stock.
The event calculation uses the premium-only shortcut explained in the example.
Values are rounded for display, not at intermediate steps.

```python
annual_demand = 520_000
share_b = 0.25
lead_a, lead_b = 60, 32  # calendar days
valuation, carrying_rate = 40.0, 0.20
qualification, recovery_factor = 300_000, 0.263797
base_purchase = annual_demand * 40.00
split_purchase = annual_demand * (0.75 * 40.30 + 0.25 * 42.00)
pipeline_units = annual_demand * share_b / 365 * (lead_a - lead_b)
carry_saving = pipeline_units * valuation * carrying_rate
annual_premium = (split_purchase - base_purchase + 36_000
                  - carry_saving + qualification * recovery_factor)

def outage_loss(receipts, opening_stock=15_000, demand=10_000):
    stock, lost = opening_stock, 0
    for received in receipts:
        available = stock + received
        filled = min(demand, available)
        lost += demand - filled
        stock = available - filled
    return lost

base_lost = outage_loss([0, 0, 0, 0])
split_lost = outage_loss([2500, 3500, 5500, 7500])
base_event = base_lost * 80
split_event = split_lost * 80 + 9000 * 3
shared_event = 1_600_000
base_expected = 0.10 * base_event + 0.03 * shared_event
split_expected = 0.10 * split_event + 0.03 * shared_event
base_total = base_purchase + base_expected
split_total = base_purchase + annual_premium + split_expected
print(f"pipeline reduction: {pipeline_units:,.0f} units")
print(f"annual premium: ${annual_premium:,.0f}")
print(f"lost units: baseline {base_lost:,}; split {split_lost:,}")
print(f"event losses: ${base_event:,.0f}; ${split_event:,.0f}")
print(f"expected losses: ${base_expected:,.0f}; ${split_expected:,.0f}")
print(f"annual totals: ${base_total:,.0f}; ${split_total:,.0f}")
print(f"break-even A-only probability: {annual_premium / (base_event - split_event):.1%}")
```

Expected output, rounded to the nearest displayed unit:

```text
pipeline reduction: 9,973 units
annual premium: $412,358
lost units: baseline 25,000; split 6,000
event losses: $2,000,000; $507,000
expected losses: $248,000; $98,700
annual totals: $21,048,000; $21,311,058
break-even A-only probability: 27.6%
```

The code does not optimize the split or infer event probabilities.
It does not model recurring disruptions, substitution, variable demand, price tiers beyond the two stated cases, or an explicit discounted cash flow.
Change one input at a time to understand sensitivity before changing several together.
For example, removing the pipeline carrying saving increases the annual premium by about USD 79,781.
If B's extra receipts arrive a week later, recalculate the full weekly flow rather than subtracting an annual capacity number.
For decision approval, reconcile input values with procurement, operations, quality, finance, and the supplier.

## 8. Data, standards, and regulatory context

### Minimum data contract

A reproducible decision needs identifiers for item, version, consuming site, supplier, manufacturing site, and lane.
A supplier identifier alone does not distinguish two factories or two production processes.
Keep a time-valid mapping because ownership, tooling, and sub-tier sourcing can change.
For each relationship record the information source, effective date, last confirmation, confidence, and accountable owner.
Record unknown upstream nodes as unknown rather than assigning them a fictional independent source.

Purchase-order events should include release, acknowledgment, promised date, dispatch, physical receipt, and quality release.
Capture time zone and calendar definition when timing precision matters.
Mark partial shipments and split receipts separately.
The correct elapsed time ends at the release of usable material, not the first administrative receipt.
If test or customs holds are excluded, state the reason and avoid presenting the resulting duration as end-to-end lead time.

Commercial records should distinguish quoted price, paid price, price tier, currency, freight terms, duty, and nonrecoverable tax.
Record the valuation method and the carrying-rate inclusions.
Maintain an explicit distinction between firm reservation, indicative capacity, and demonstrated output.
Store dates and evidence for product approvals and any customer-specific restrictions.
Access to sub-tier information may be commercially sensitive; limit access under the applicable agreement and policy.

### Standards and contracts

The International Organization for Standardization's ISO 22301 specifies requirements for a business continuity management system.
It is a management-system standard, not proof that two sources are independent or that a particular split is optimal.
Certification, where sought, requires a separate conformity assessment and evidence; using this lesson does not confer it.
ISO 31000 provides risk-management guidelines rather than a numerical probability for a particular supplier outage.
Consult the current official editions and the organization's actual contractual obligations.

ISO 9001 sets quality-management-system requirements but does not by itself certify every individual part or alternate process.
The buyer must follow its own applicable product, customer, and regulatory approval requirements.
Specific industries may impose additional requirements; determine them for the product and jurisdiction rather than assuming they apply to all controllers.
Qualification evidence, test methods, and change approvals remain part-specific.

Incoterms rules published by the International Chamber of Commerce allocate defined responsibilities between buyer and seller when incorporated into a contract.
They are not a substitute for a complete contract and do not alone determine ownership, product compliance, or the buyer's total landed cost.
The selected rule and named place affect which freight and border charges appear on each side of a quote.
Compare quotes at a common destination and avoid adding a charge already included in a delivered price.

Customs origin, duties, sanctions, export controls, and product rules depend on the item, activities, countries, and current law.
A nearshore assembly location does not automatically establish preferential origin or eliminate duties on imported inputs.
Confirm classification, origin, eligibility, documentation, and effective dates with qualified trade-compliance personnel and current official sources.
The educational example does not assert a tariff rate or eligibility for a trade agreement.

### Evidence fields for a sourcing review

The following data elements support an auditable comparison.
The owner column names a typical accountable function; an organization can assign ownership differently.
An unknown field is a prompt for an evidence request, not a license to substitute zero.

| Record | Field and unit | Typical owner | Decision use |
|---|---|---|---|
| Product | Part number and revision | Engineering | Defines interchangeability |
| Product | Approved material specification | Quality | Defines acceptable inputs |
| Product | Customer approval status and date | Quality | Determines eligibility for use |
| Product | Expected end-of-life date | Product management | Sets qualification horizon |
| Demand | Forecast units by week | Planning | Sizes normal allocation and stress |
| Demand | Forecast error by time bucket | Planning | Supports inventory analysis |
| Demand | Lost versus delayed demand rule | Commercial | Determines shortage valuation |
| Supplier | Legal entity and contract identifier | Procurement | Links the offer to enforceable terms |
| Supplier | Manufacturing site address | Supplier management | Maps physical exposure |
| Supplier | Product-specific approval status | Quality | Separates onboarding from qualification |
| Supplier | Ownership and control | Procurement | Identifies potentially related sources |
| Sub-tier | Input identifier and originating site | Supplier management | Reveals common-mode risk |
| Sub-tier | Available input units and allocation | Supplier management | Constrains ramp |
| Tooling | Tool identity and physical location | Engineering | Reveals shared equipment |
| Tooling | Ownership and access terms | Procurement | Assesses recovery feasibility |
| Capacity | Qualified units per week | Operations | Caps conforming output |
| Capacity | Other committed demand, units per week | Supplier management | Tests whether nominal slack is real |
| Capacity | Ramp profile, usable units by week | Operations | Times disruption response |
| Capacity | Reservation period and notice requirement | Procurement | Establishes call-off rights |
| Logistics | Dispatch schedule and cutoff | Logistics | Captures batching wait |
| Logistics | Route and border crossing | Logistics | Maps shared lane exposure |
| Logistics | Transport and border time distribution | Logistics | Supports end-to-end lead time |
| Receipt | Physical receipt timestamp | Warehouse | Measures arrival |
| Receipt | Quality release timestamp | Quality | Determines usable arrival |
| Cost | Price per accepted unit and tier | Procurement | Calculates purchase cost |
| Cost | Freight inclusions and named place | Logistics | Avoids omissions or duplication |
| Cost | Reservation and audit fee per year | Procurement | Captures recurring maintenance |
| Cost | Qualification spending and payment date | Finance | Supports cash and annual-equivalent views |
| Inventory | Units by location and status | Planning | Prevents spending the same stock twice |
| Inventory | Valuation basis and currency | Finance | Converts units into tied-up capital |
| Inventory | Carrying-rate components per year | Finance | Converts balance to annual cost |
| Event | Scenario duration and affected nodes | Risk | Defines disruption scope |
| Event | Assumed probability per year, if used | Risk | Enables a stated expected-value calculation |
| Governance | Activation threshold and decision owner | Operations | Makes the option executable |

Check agreement between systems before using these fields.
For example, the purchasing master may list the contracting entity while the shipment record lists a third-party plant.
Confirm the relationship instead of silently treating either record as the full route.
If capacity is provided in machine hours, convert it to accepted units with the relevant product mix and yield.
Keep the original units and conversion assumptions alongside the result.

### A practical evidence review

An analyst can use the following questions in a joint review with the supplier.
The objective is to identify decision-relevant gaps rather than demand exhaustive disclosure of every supplier relationship.
Record the answer, date, evidence, and any confidentiality restriction.

**Product and process**

1. Which exact product revision did the trial produce?
2. Which facility, production line, and tooling were used?
3. Is the proposed emergency process the same as the approved process?
4. What yield was achieved on a representative run?
5. Which quality checks remain at the buyer's site?
6. Who must approve changes in raw material or manufacturing site?

**Supply and capacity**

1. Which inputs constrain the ramp?
2. Are those inputs physically held or only forecast?
3. What quantities are dedicated to this buyer?
4. What other customers can claim the same equipment?
5. How many usable units can arrive in each of the first four weeks?
6. What happens if the outage lasts longer than four weeks?

**Logistics and receipt**

1. How often do shipments actually depart?
2. What is the normal time from order release to dispatch?
3. Which route and border crossing are used?
4. How long do clearance, receiving, and release usually take?
5. Is an emergency mode available and what does it cost incrementally?
6. Can the receiving plant process the proposed surge without a new bottleneck?

**Commercial and governance**

1. Which volume determines the quoted unit price?
2. Is there a minimum annual purchase or cancellation charge?
3. Is the capacity reservation exclusive, shared, or best efforts?
4. What notice is required to activate the additional volume?
5. Which party pays for the tool, trial materials, and ongoing audits?
6. Who is authorized to approve the additional spend during a disruption?

An answer such as “capacity is available” is insufficient without its time bucket and product scope.
An answer such as “the supplier is approved” is insufficient without the approved site and revision.
A qualitative risk rating can help prioritize review, but it cannot replace these fields in the service calculation.
Update the record when an assumption changes and preserve the prior version for decision auditability.

### Policy and legal boundaries

This learning unit uses international standards as guidance or management-system context.
It does not assert that ISO certification is legally mandatory for this buyer.
Actual obligations can arise from a contract, product regulation, customer specification, or company policy.
Determine which obligation applies, to whom, and in which jurisdiction.
Record the effective version and the function responsible for interpretation.

Prefer a current official customs or regulator source to a generic claim about a region.
For an actual cross-border move, verify the product classification and rules of origin before pricing duties.
Keep a copy of the dated ruling or professional advice where company policy permits.
If origin depends on the value or transformation of components, document the relevant bill of materials and process.
Moving final assembly does not itself establish the answer.

The buyer should also distinguish contractual capacity from legal certainty of performance.
Even a signed allocation can be subject to conditions, remedies, and applicable law.
The operating model should include what happens if a commitment is missed.
Legal remedies can reduce financial loss but do not create controllers on the production line.

### Controls and evidence quality

Use a documented hierarchy: actual plant-level receipt and capacity evidence, supplier declarations, public sources, and assumptions.
A statement from a supplier may be useful but is not equivalent to a witnessed ramp test.
Use audit rights and confidentiality provisions to resolve important unknowns where feasible.
Monitor aging records and revalidate after a material process or route change.
Track actual lead-time distributions and interruption response against the modeled values.
A good decision record preserves the original assumptions even after later outcomes become known.

## 9. Industry practice and case evidence

### Toyota and upstream visibility

Toyota publicly describes its supply-chain resilience activities and supplier information management in its sustainability materials.
Its disclosures discuss business continuity and mapping supplier dependencies beyond direct suppliers.
The documented lesson is that upstream visibility is an active management task, not a property automatically delivered by multiple purchase orders.
Public disclosures do not establish that every Toyota component has independent dual sources, nor do they validate the worked example's probabilities or prices.

**Inference:** a buyer considering dual sourcing should map sub-tier nodes before counting routes as independent.
The appropriate depth depends on criticality, available evidence, and the buyer's ability to act on the information.
A map becomes stale if suppliers change materials, facilities, or transport lanes without a corresponding update.

### Semiconductor concentration and disruption

The Organisation for Economic Co-operation and Development (OECD) has documented the complexity and geographic concentration of semiconductor supply chains.
Fabrication, equipment, materials, assembly, and design are distributed across specialized participants.
This evidence supports examining dependencies at the process and facility level rather than assuming final-assembly location measures exposure.
It does not imply that a single nearshore electronics assembler can immediately replace any chip foundry.

**Inference:** two controller assemblers that buy the same chip can both be constrained by the same fabrication disruption.
A local assembly route may still shorten the later production and transport stages.
The benefit is real only to the extent those stages are on the replenishment critical path and the remaining input is available.

### Japanese earthquake and supplier recovery

The 2011 Great East Japan Earthquake disrupted automotive supply networks, including upstream component suppliers.
The peer-reviewed work by Todo, Nakajima, and Matous examines how supply-chain networks related to the recovery of firms after that disaster.
Its findings are evidence about a specific disaster and network, not a universal percentage reduction in outage duration from a second supplier.
A complex network can transmit disruption while also providing relationships that aid recovery.

**Inference:** supplier relationships and recovery capability should be assessed alongside geographic diversification.
A second supplier without material, tooling, or an approved process cannot necessarily absorb demand during the relevant weeks.
The decision maker should ask which node failed, which firms were connected to it, and when alternate output became usable.

### Nearshoring evidence versus this decision

Public data on merchandise trade can show the importance of trading partners and the movement of goods across borders.
They do not by themselves show a specific buyer's order-to-usable lead time, quality performance, or incremental cost.
Company announcements of new facilities can establish intent or an investment decision; they do not establish qualified output for a given part without more evidence.
Consequently, use public cases to form questions and obtain supplier- and item-level data to make the award.

The worked example is fictional.
Its prices, ramp, event probabilities, and losses are teaching assumptions, not estimates for any named company.
The example's negative expected-value result is not a general finding that dual sourcing is uneconomic.
Different constraints, common-mode dependencies, and business consequences can reverse the answer.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Two names, one site | Different supplier IDs share an address | A site event disables both routes | False diversification | Map actual production sites and ownership |
| Shared chip | Both bills of material name the same chip | Upstream outage stops both assemblers | Common-mode shortage | Map critical tier-two nodes and scenario-test them |
| Unqualified alternate | Only supplier onboarding is complete | Product or site approval is missing | Backup cannot legally or technically ship | Gate use on documented product-specific approval |
| Paper capacity | Capacity is described only in a presentation | Competing customer demand consumes slack | Emergency orders are rejected | Contract and test a time-phased allocation |
| Slow ramp | Full capacity shown without startup weeks | Tooling, labor, and inputs take time | Early demand is lost | Model usable receipts by week |
| Transit-only lead time | Quote shows truck days but no queue | Input waits and batching dominate | Overstated inventory benefit | Measure release-to-usable lead time |
| Wrong split denominator | Spend share is treated as unit share | Different prices distort quantities | Capacity and savings are misstated | Use unit shares and reconcile purchases |
| Lost price tier | A's old unit price is applied at lower volume | Volume discount disappears | Recurring premium understated | Apply each supplier's actual tier |
| Double-counted stock | Cash release and full inventory value treated as annual savings | Balance-sheet asset confused with expense | Business case overstated | Separate cash flow and annual carrying charge |
| Double-counted shortage | Lost revenue and contribution both charged as loss | Variable costs are ignored | Disruption benefit overstated | Use contribution or a fully reconciled profit bridge |
| Ignored common shock | Independent probabilities multiplied by default | Shared geography or input creates dependence | Tail loss understated | Model common-cause states explicitly |
| Stale approval | A process changed since the trial | Prior evidence no longer covers the route | Defective or rejected product | Use controlled change and reapproval rules |
| Minimum-volume breach | Small backup share is below supplier minimum | Supplier cannot sustain the line or contract | Capacity evaporates | Check minimums and maintenance cost |
| Unreleased inventory | Physical receipt counted as available | Quality inspection or customs still pending | False service coverage | Use usable-release timestamps |
| False recovery | Supplier production resumes but transport remains blocked | Downstream lane is still unavailable | Customer shortage continues | Track the entire route to usable receipt |
| Overlapping event costs | Emergency freight included in both price and premium | Duplicate cost lines | Resilience appears too expensive | Reconcile incremental versus full costs |
| Forecast treated as commitment | Supplier sales deck gives a maximum | Capacity was never reserved | Planned response is unavailable | Obtain signed terms and demonstrate output |
| Unsupported probability | One outage in a short history becomes an exact rate | Sparse evidence is overinterpreted | False precision | Show ranges and break-even thresholds |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or change it |
|---|---|---|---|
| Demand is 10,000 units/week | Makes weekly service arithmetic explicit | Seasonality or promotion | Replay actual or simulated weekly demand |
| Both scenarios start with 15,000 usable units | Isolates the sourcing decision in the outage | Different reorder policies or transition stock | Reconcile physical, pipeline, and allocated inventory |
| Unmet demand is permanently lost | Converts shortage into contribution loss | Customers wait or substitute | Model backlog, cancellation, and recovery separately |
| B's ramp arrives as stated | Establishes usable alternative supply | Tooling, inputs, labor, or transport fail | Run a time-stamped capacity and delivery trial |
| Product quality is equivalent | Allows units to substitute one-for-one | Yield differences or unapproved variants | Validate product/site combination and acceptance rates |
| B's price applies at 130,000 units/year | Calculates the normal cost bridge | Minimums or price tiers change | Obtain a dated, volume-specific quotation |
| Lead-time segments are sequential | Allows simple addition of days | Production and input procurement overlap | Build a milestone-level critical path |
| Average demand supports pipeline formula | Estimates steady-state stock | Lumpy orders or nonstationary demand | Simulate actual release and receipt patterns |
| Inventory valuation is USD 40/unit | Converts units to a comparable capital base | Accounting basis or product cost differs | Reconcile with finance and recalculate |
| Carrying rate is 20%/year | Converts a balance to an annual cost | Rate excludes or duplicates financing and storage | Document components and sensitivity |
| Qualification benefits last five years | Annualizes upfront cost | Product changes or supplier exits earlier | Vary horizon and model actual renewal costs |
| Annual states are mutually exclusive | Allows weighted addition of losses | Repeated or simultaneous events | Use a joint-event or simulation model |
| Shared-node loss is equal | Cancels this loss in the comparison | Different chip buffers or allocations | Build route-specific shared-node timelines |
| Extra-unit premium is USD 3 | Prices emergency response incrementally | Base purchase accounting changes | Build a full purchase and avoided-cost bridge |
| A-only probability is 10%/year | Illustrates expected-cost calculation | Sparse or nonstationary evidence | Show ranges and update with credible data |
| No tax or terminal effects | Keeps the annual-equivalent example transparent | Decision depends on tax or end-of-life inventory | Prepare a dated discounted cash-flow model |

The pipeline saving is not automatically a safety-stock saving.
The illustrative event loss is not a statistical estimate of the tail distribution.
The decision can be dominated by a nonfinancial constraint not represented in the USD comparison.
The correct response is to state that constraint and its authority, not to disguise it as a fabricated probability.

## 12. Exercises

### Exercise 1 — Map shared dependencies

A buyer has two approved final assemblers in different countries.
Both use the same specialized chip factory and the same border crossing.
One uses a separate packaging supplier; the other uses the buyer's standard packaging supplier.

**Deliverable:** draw a route map and list two common-mode scenarios and one route-specific scenario.
**Acceptance criteria:** identify the chip factory and border crossing as shared; distinguish the packaging paths; state one unknown that requires evidence.
**Time:** 15 minutes.

### Exercise 2 — Calculate actual lead time

Route X has 6 days of queue, 11 days of input wait, 9 days of production, 4 days of dispatch wait, 22 days of transport and border, and 3 days of release.
Route Y has 5, 12, 8, 3, 6, and 2 days for the same stages, respectively.
All stages are sequential.
Annual demand allocated to Y is 73,000 units, uniformly distributed over 365 calendar days.

**Deliverable:** calculate both totals, the total and transport-only differences, and the steady-state pipeline-unit reduction.
**Acceptance criteria:** show arithmetic, distinguish geography from total lead time, and state why safety stock cannot be calculated from these data alone.
**Time:** 20 minutes.

### Exercise 3 — Compare cost at the split

A buyer purchases 100,000 accepted units per year at USD 20 each.
A proposed 70/30 split changes the primary price to USD 20.40 and the alternate price to USD 22.00 per unit.
The alternate has a USD 12,000 annual maintenance fee and USD 15,000 of annualized qualification cost.
Ignore inventory and disruption effects.

**Deliverable:** calculate the baseline and proposed annual cost and the annual premium.
**Acceptance criteria:** apply each price to its own units and add each annual cost once.
**Time:** 15 minutes.

### Exercise 4 — Apply a capacity ramp

A plant begins an outage with 4,000 usable units.
Demand is 3,000 units each week for four weeks.
The alternate delivers 500, 1,000, 2,000, and 3,000 usable units in weeks 1 through 4.
Unfilled demand is lost and cannot be backordered.

**Deliverable:** provide opening stock, available units, fulfilled demand, closing stock, and lost units for every week.
**Acceptance criteria:** conserve units, never show negative inventory, and report total lost units.
**Time:** 20 minutes.

### Exercise 5 — Find a break-even probability

An alternate route has an annual premium of USD 120,000 after annualized qualification.
A primary-only event costs USD 900,000 in the baseline and USD 300,000 with the alternate.
All other scenario losses are identical between the alternatives.
Assume at most one such event per year and no other incremental benefits.

**Deliverable:** calculate the event probability at which expected annual costs are equal and explain the interpretation.
**Acceptance criteria:** show the incremental event benefit and do not call the threshold a measured probability.
**Time:** 15 minutes.

### Exercise 6 — Critique a business case

A proposal claims 100,000 units of pipeline reduction worth USD 4 million, plus USD 4 million of annual inventory savings.
It also counts USD 2 million of lost revenue and USD 0.8 million of lost contribution for the same unfilled units.
Two suppliers in separate cities buy a sole-source component from one plant.
The proposed emergency capacity is the sum of their factory nameplate capacities.

**Deliverable:** write four corrections and name the information needed for each.
**Acceptance criteria:** address balance versus annual cost, revenue versus contribution, common-mode risk, and usable reserved capacity.
**Time:** 25 minutes.

### Exercise 7 — Design qualification and activation

A second supplier can produce samples within four weeks but needs a dedicated tool and an upstream material allocation before sustained production.
The buyer needs the alternate to be ready for a product whose change approval involves engineering, quality, and a customer.

**Deliverable:** design a staged plan from feasibility to activation, including evidence, owner, gate, and trigger.
**Acceptance criteria:** include product-specific approval, a capacity demonstration, upstream material evidence, ongoing maintenance, and a stop condition.
Do not claim the source is available before the approvals are complete.
**Time:** 30 minutes.

### Exercise 8 — Recommend a split under uncertainty

The worked example's management team values the severe A-only service improvement but has not specified a financial value for the service target.
A 75/25 split still loses 6,000 units in that event, and both sources depend on chip C.

**Deliverable:** write a one-page decision note with a recommendation, the expected annual cost difference, the unmet target, the shared-node limitation, and two evidence requests.
**Acceptance criteria:** distinguish a cost-based choice from an explicitly authorized risk-appetite choice and identify one condition that changes the recommendation.
**Time:** 30 minutes.

## 13. Self-check answers

### 1. Shared dependencies

Both routes depend on the same chip factory and the same border crossing.
A chip-factory outage can affect both assemblers; a border closure can affect both inbound lanes.
A failure of only the independent packaging supplier is route-specific if the other route does not use that supplier or its affected inputs.
A useful unknown is whether each assembler has sufficient chip inventory and contractual allocation to continue during a short outage.
Different stock levels can change timing even when the underlying node is shared.

### 2. Lead-time arithmetic

Route X totals 6 + 11 + 9 + 4 + 22 + 3 = 55 calendar days.
Route Y totals 5 + 12 + 8 + 3 + 6 + 2 = 36 calendar days.
The end-to-end difference is 19 days; transport and border alone differ by 16 days.
Y receives 73,000 / 365 = 200 units per calendar day.
Estimated steady-state pipeline reduction is 200 × 19 = 3,800 units.
Safety stock additionally depends on demand and lead-time variability, their dependence, the service objective, and the inventory policy.

### 3. Cost bridge

The baseline costs 100,000 × USD 20 = USD 2,000,000 per year.
Primary purchases under the split cost 70,000 × USD 20.40 = USD 1,428,000 per year.
Alternate purchases cost 30,000 × USD 22 = USD 660,000 per year.
Including USD 12,000 maintenance and USD 15,000 annualized qualification gives USD 2,115,000 per year.
The annual premium is USD 115,000 before inventory or disruption effects.

### 4. Weekly conservation

| Week | Opening | Receipts | Available | Fulfilled | Closing | Lost |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4,000 | 500 | 4,500 | 3,000 | 1,500 | 0 |
| 2 | 1,500 | 1,000 | 2,500 | 2,500 | 0 | 500 |
| 3 | 0 | 2,000 | 2,000 | 2,000 | 0 | 1,000 |
| 4 | 0 | 3,000 | 3,000 | 3,000 | 0 | 0 |
| Total | — | 6,500 | — | 10,500 | — | 1,500 |

Total available units over the horizon are 4,000 + 6,500 = 10,500.
Total demand is 12,000, so 1,500 units are lost.
Timing matters even if total capacity later increases.

### 5. Break-even probability

The alternate reduces event loss by USD 900,000 − USD 300,000 = USD 600,000 per event.
The break-even annual probability is USD 120,000 / USD 600,000 = 0.20, or 20% per year.
At a lower probability, this simplified expected-cost comparison favors the baseline.
At a higher probability, it favors the alternate, all else equal.
The threshold is not an estimate of actual outage probability and does not account for risk appetite or omitted benefits.

### 6. Critique rubric

A strong answer separates the USD 4 million released inventory balance from the annual cost of carrying that balance.
It requests the valuation basis, carrying rate, timing, and any transition stock.
It uses lost contribution or a reconciled profit model rather than adding lost revenue and contribution for the same units.
It models the sole-source component as a shared node and requests stock and recovery data for both routes.
It replaces nameplate capacity with approved, reserved, time-phased usable capacity supported by input and logistics evidence.
A weak answer merely says that two suppliers are safer without addressing the shared plant or the ramp.

### 7. Qualification rubric

A strong plan identifies technical specifications and establishes who approves them.
It funds and validates the dedicated tool and confirms ownership and transfer rights.
It obtains evidence of upstream material allocation and a product-specific process trial.
It schedules engineering, quality, and customer approvals before emergency use where required.
It tests usable receipts against the ramp, assigns an activation owner, and defines periodic maintenance.
A stop condition includes failure of a mandatory test, absent customer approval, or inability to secure the critical input.
A weak plan treats samples or a signed purchase order as proof of full qualification.

### 8. Recommendation rubric

A cost-based recommendation retains the baseline under the stated numbers because the split is USD 263,058 per year more expensive in expected cost.
It also reports that the split reduces A-only lost units from 25,000 to 6,000 but does not meet the zero-loss target.
The common chip scenario remains unchanged in the example, so geographic diversification is not complete.
A risk owner may choose the split for tail protection if the premium and remaining target gap are explicitly accepted.
Evidence requests can include a witnessed ramp and release test, a verified chip allocation, and an actual lead-time distribution.
A changed event probability above the conditional 27.6% threshold, or a changed cost or service constraint, can alter the decision.

## 14. Interview questions

1. **What makes two suppliers genuinely independent?**
   A strong answer traces sites, inputs, lanes, and shared hazards and distinguishes independence from legal identity.
   A weak answer cites only different vendor codes.
2. **How do you measure a nearshoring lead-time benefit?**
   A strong answer compares release-to-usable milestones and isolates transit from other changes.
   A weak answer substitutes driving time for replenishment time.
3. **How would you select a sourcing split?**
   A strong answer considers price tiers, minimums, qualified capacity, ramp, service, and scenarios.
   A weak answer asserts that 50/50 is always optimal.
4. **Why maintain routine orders at the backup?**
   A strong answer discusses process readiness, quality, supplier economics, and tests while noting that orders do not prove surge capacity.
   A weak answer assumes a dormant supplier can immediately meet all demand.
5. **How do you account for qualification spending?**
   A strong answer distinguishes upfront cash flow from annual equivalent and avoids double counting.
   A weak answer ignores engineering, tooling, and customer approval costs.
6. **What is the difference between pipeline and safety stock?**
   A strong answer ties pipeline to flow and elapsed time, and safety stock to uncertainty and service policy.
   A weak answer calculates safety stock from mean lead time alone.
7. **What happens when both suppliers share a critical sub-tier?**
   A strong answer models a common-cause scenario and the timing of each route's remaining stock.
   A weak answer multiplies independent outage probabilities without evidence.
8. **How do you value an avoided shortage?**
   A strong answer distinguishes permanently lost units, backlog, contribution, penalties, and incremental costs.
   A weak answer adds revenue and contribution for the same loss.
9. **What would invalidate a supplier ramp claim?**
   A strong answer checks product qualification, inputs, other customers, tooling, yield, and delivery timing.
   A weak answer relies on a nameplate machine rate.
10. **How do you make a decision with weak outage probabilities?**
    A strong answer uses ranges, break-even analysis, explicit tail scenarios, and decision-owner risk appetite.
    A weak answer invents a precise frequency or discards risk entirely.

## 15. Key terms

- [Dual sourcing](#4-definitions-and-vocabulary): two approved routes for the same requirement.
- [Nearshoring](#4-definitions-and-vocabulary): moving an activity closer to its consuming market.
- [Sole source](#4-definitions-and-vocabulary): only one feasible supplier under the defined requirement.
- [Split](#4-definitions-and-vocabulary): planned allocation by unit share or another explicit denominator.
- [Qualified capacity](#4-definitions-and-vocabulary): validated conforming output per period.
- [Reserved capacity](#4-definitions-and-vocabulary): capacity set aside under specified terms.
- [Ramp](#4-definitions-and-vocabulary): time profile of usable extra output.
- [Qualification](#4-definitions-and-vocabulary): evidence and approval for the specified route.
- [Common-mode risk](#4-definitions-and-vocabulary): one cause affecting multiple routes.
- [Exposure](#4-definitions-and-vocabulary): quantity or value affected by a hazard.
- [Lead time](#4-definitions-and-vocabulary): elapsed time between specified start and end events.
- [Usable receipt](#4-definitions-and-vocabulary): conforming stock released for use.
- [Pipeline stock](#4-definitions-and-vocabulary): inventory in the replenishment process.
- [Safety stock](#4-definitions-and-vocabulary): inventory held against specified uncertainty.
- [Fill rate](#4-definitions-and-vocabulary): fulfilled fraction under a stated service definition.
- [Time to recover](#4-definitions-and-vocabulary): time to restore a specified capability.
- [Time to survive](#4-definitions-and-vocabulary): coverage time using available resources.
- [Lost contribution](#4-definitions-and-vocabulary): contribution lost from permanently unfilled demand.
- [Landed cost](#4-definitions-and-vocabulary): comparable delivered cost under a stated scope.
- [Qualification cost](#4-definitions-and-vocabulary): one-time cost of approving the route.
- [Annual equivalent](#4-definitions-and-vocabulary): uniform annual amount with the same present value.
- [Working capital](#4-definitions-and-vocabulary): capital tied up in operating assets net of relevant liabilities.
- [Carrying charge](#4-definitions-and-vocabulary): annual cost applied to inventory value.
- [Expected annual cost](#4-definitions-and-vocabulary): probability-weighted annual cost across stated states.
- [Tail scenario](#4-definitions-and-vocabulary): separately examined adverse outcome.
- [Trigger](#4-definitions-and-vocabulary): observable condition that authorizes a response.

## 16. Sources

1. International Organization for Standardization. [ISO 22301:2019, Security and resilience — Business continuity management systems — Requirements](https://www.iso.org/standard/75106.html).
   Official description of business-continuity management-system requirements; not a sourcing optimization method.
2. International Organization for Standardization. [ISO 31000:2018, Risk management — Guidelines](https://www.iso.org/standard/65694.html).
   General risk-management guidance; does not supply event probabilities for this example.
3. International Organization for Standardization. [ISO 9001:2015, Quality management systems — Requirements](https://www.iso.org/standard/62085.html).
   Quality-management-system context; product-specific approval remains necessary.
4. International Chamber of Commerce. [Incoterms 2020](https://iccwbo.org/business-solutions/incoterms-rules/incoterms-2020/).
   Official overview of trade-term rules; consult the contract and applicable law for an actual transaction.
5. Toyota Motor Corporation. [Sustainability data book](https://global.toyota/en/sustainability/report/sdb/).
   Company disclosures on supply-chain management and continuity; company-reported evidence, not independent proof of every supplier relationship.
6. Organisation for Economic Co-operation and Development. [Measuring distortions in international markets: The semiconductor value chain](https://doi.org/10.1787/8fe4491d-en), OECD Trade Policy Papers, 2019.
   Context on semiconductor value-chain structure and specialization.
7. Todo, Yasuyuki, Kentaro Nakajima, and Petr Matous. [How do supply chain networks affect the resilience of firms to natural disasters? Evidence from the Great East Japan Earthquake](https://doi.org/10.1111/jors.12111). *Journal of Regional Science*, 2015.
   Peer-reviewed empirical study of a specific disaster and supply network.
8. Simchi-Levi, David, William Schmidt, and Yehua Wei. [From superstorms to factory fires: Managing unpredictable supply-chain disruptions](https://hbr.org/2014/01/from-superstorms-to-factory-fires-managing-unpredictable-supply-chain-disruptions). *Harvard Business Review*, 2014.
   Practitioner discussion of disruption exposure and recovery; use as conceptual context, not a probability source.

Sources are provided for learning and should be checked against current editions and applicable jurisdiction before operational or legal use.
The numerical case is fictional and is not attributed to any source.

## 17. Links

- [Module 14A index](./_index.md).
- [Module 14A survey](../../../course/14A_resilience_disruption_analytics.md).
- [Mapping and exposure](./03_mapping-exposure.md).
- [Stress testing](./05_stress-testing.md).
- [Buffering versus flexibility](./07_buffering-flexibility.md).
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md).
