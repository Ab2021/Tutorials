# Buffering versus Flexibility

> Module 14A — Supply Chain Risk, Resilience, and Disruption Analytics  
> Topic 07 of 10  
> Estimated study time: 4 hours  
> Prerequisites: risk mapping, time-to-recover and time-to-survive, scenario analysis  
> Version 2026.5

## 1. Why this topic exists

A procurement leader can buy more material before a disruption, reserve capacity to produce during it, or qualify a second supplier.
Each choice spends money at a different time and protects against a different failure.
A treasury team can also hedge a price or exchange-rate exposure, but that contract does not deliver a missing part.
The challenge is to buy the right combination rather than label every expense “resilience.”

A plant making medical devices depends on a single molded component.
A flood at its supplier can stop shipments for weeks.
Extra finished components buy time immediately, but they consume working capital and can expire or become obsolete.
Reserved molding capacity elsewhere may help only after tooling, labor, approvals, and transport are ready.
Dual sourcing creates an alternate path, but two suppliers can still share a power grid, a resin producer, or a port.
A call option on resin price limits some financial exposure; it does not qualify a replacement molder.

The decision maker needs a common comparison: incremental annual cost, lost contribution, customer impact, and physical recovery time under the same scenarios.
Average annual cost alone is not sufficient when a missed delivery creates regulatory or patient consequences.
Likewise, a high service target does not justify a hedge whose capacity cannot be activated in time.
Separate what management forecasts from what it plans, targets, and has contractually secured.
A forecast is uncertain expected demand; a plan allocates resources against it.
A target is a desired outcome; a commitment is an enforceable or operationally verified promise, subject to its terms.

This topic supplies a practical evaluation method, not a universal optimal buffer.
Its illustrations are hypothetical and their probabilities are assumptions, not market statistics.
Tax, financing, legal enforceability, and customer damages require local review.
The point is to make the trade-offs explicit enough for a cross-functional owner to act.

## 2. Learning outcomes

After completing the topic you can:

1. **Distinguish** physical inventory, standby capacity, alternate supply, and financial option payoffs, including the risks each does not cover; see exercises 1 and 7.
2. **Compute** usable stock coverage, shortage, and recovery overlap with units and timing; see exercises 2 and 3.
3. **Compare** annual incremental cost and expected disruption loss across mutually exclusive strategies under an explicit probability model; see exercise 4.
4. **Evaluate** a dual-source plan for common-cause failure, capacity, qualification, and commercially credible commitments; see exercise 5.
5. **Calculate** a simple commodity call option payoff, premium, and basis exposure without mistaking a financial payoff for physical supply; see exercise 6.
6. **Design** an activation and governance plan with triggers, owners, and evidence; see exercise 8.
7. **Critique** a resilience recommendation when demand, loss valuation, or scenario probabilities are uncertain; see exercises 4 and 9.
8. **Explain** how to avoid double counting between option premium, physical purchase cost, and disruption loss; see exercises 6 and 7.

A successful answer shows the decision rule and its boundaries, not merely a formula.
When reliable probabilities are absent, compare scenario outcomes and identify the threshold at which the decision changes.

## 3. The core mental model

```text
Shock -> lost or delayed flow -> compare demand with usable supply each period
                                  |
               +------------------+------------------+
               |                  |                  |
        inventory now      capacity / alternate    uncovered need
        consumes stock     supply after delay      -> loss/backlog
               |                  |                  |
               +------------------+------------------+
                                  |
                     service, recovery, cash

Price shock -> physical purchase cost -> financial option payoff
                                       (cash offset, not material)
```

Inventory is a **stock** measured in units.
Production and consumption are **flows** measured in units per day or week.
Spare capacity is a rate ceiling, not an instant stockpile.
A facility capable of making 500 units per day for five days can deliver at most 2,500 units before lead time and yield constraints, and less if that capacity is shared.
A recovery promise also has a clock: detection, decision, qualification, production, transit, and receipt can each consume time.

Start the clock at a defined event, such as the first missed receipt.
On each day, add usable receipts to opening inventory, satisfy the demand that policy allows, and carry the remainder forward.
If orders can be backlogged, keep the backlog as a separate state variable and account for late-service consequences.
If demand is lost permanently, do not allow subsequent production to serve it.
The time-to-survive is how long a defined service level can be maintained under a specified shock.
Time-to-recover is the time until the disrupted capability returns to the stated level, not necessarily the time until backlog clears.

Inventory generally helps from day one; flexibility helps after activation.
A hybrid can bridge the period before alternate output arrives.
However, its elements may overlap, so the benefit of the combination can be smaller than the sum of stand-alone benefits.
Shared upstream dependencies can also make apparent diversity illusory.
The network, product specifications, and contract matter as much as the supplier count.

A financial hedge shifts cash outcomes across market prices.
It can fund an expensive substitute or reduce earnings volatility, but the hedge counterparty is not necessarily an approved supplier.
The option may settle against an index different from the actual purchase price.
That mismatch is basis risk, and the hedge can be ineffective even when it pays.

A useful decision has four views:

- **Physical:** period-by-period demand, usable supply, inventory, and unmet demand.
- **Commercial:** reservation fee, minimum volume, exercise price, qualification, freight, and supplier capacity rights.
- **Financial:** premium, payoff, financing, cash timing, and incremental loss.
- **Governance:** authority, trigger, tests, customer priorities, and monitoring.

No single view replaces the others.
For example, an expected cost minimum may violate a mandatory customer allocation rule.
A service-compliant alternative may still need treasury approval for a large upfront cash commitment.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Demand | Customer requirement per period, units/day | Treating a forecast as an order commitment |
| Forecast | Estimate of uncertain future demand, units/day | Calling it guaranteed sales |
| Plan | Chosen operating allocation, units/day | Assuming all planned capacity exists |
| Target | Desired performance, percent or time | Treating an aspiration as a contract |
| Commitment | Allocated or contractually promised resource, units or units/day | Counting unreserved vendor capacity as firm |
| On-hand stock | Inventory physically recorded, units | Assuming all recorded units are usable |
| Usable stock | Released, compatible, unexpired stock, units | Counting quarantined or customer-owned stock |
| Safety stock | Stock held against modeled variability, units | Assuming it covers every disruption duration |
| Cycle stock | Inventory arising from ordinary replenishment batches, units | Counting all cycle stock as dedicated disruption cover |
| Inventory position | On hand plus eligible on-order minus backorders, units | Equating it with stock available immediately |
| Coverage | Usable stock divided by a specified net depletion rate, days | Dividing by demand when receipts continue |
| Capacity buffer | Available production rate above planned load, units/day | Confusing nameplate capacity with deliverable output |
| Utilization | Actual load divided by available capacity, percent | Using annual utilization to prove a shift is free |
| Ramp-up | Transition to a required production rate, days | Assuming full capacity on activation day |
| Activation delay | Time from trigger to first usable alternate receipt, days | Omitting approval or transport |
| Lead time | Elapsed time from a defined start to receipt, days | Combining clocks with incompatible start points |
| Time-to-recover | Time until capability returns to a defined performance level, days | Calling backlog clearance recovery of the facility |
| Time-to-survive | Time until a specified service threshold is breached, days | Treating it as a property independent of demand |
| Backlog | Demand awaiting future fulfillment, units | Treating lost sales as recoverable backlog |
| Lost sales | Demand assumed permanently unserved, units | Counting both lost sales and later backlog fulfillment |
| Fill rate | Units served divided by eligible demand, percent | Calling orders served on time and units served the same metric |
| Contribution margin | Incremental revenue less avoidable variable costs, currency/unit | Treating revenue as the economic loss |
| Dual sourcing | Qualifying and using two supply sources | Assuming two vendor names imply independent sources |
| Common cause | One event affecting multiple supposedly distinct paths | Multiplying marginal probabilities without dependence evidence |
| Reservation fee | Payment to secure capacity rights, currency/year | Assuming a fee buys material without exercise |
| Qualification | Demonstrated approval of process, product, and source | Assuming an identical drawing makes a supplier approved |
| Exercise | Use of a contractual right under its terms | Assuming any option holder must exercise |
| Financial call option | Right to a price-linked payoff above a strike, subject to contract terms | Treating a cash-settled call as a delivery guarantee |
| Strike | Contract price threshold, currency/unit | Calling the strike the total cost after premium |
| Premium | Upfront or agreed price of an option, currency | Subtracting the premium twice |
| Notional | Quantity or reference amount covered by a hedge | Assuming notional equals physical supply |
| Basis risk | Mismatch between hedged index and actual exposure | Assuming correlated prices are identical |
| Carrying cost | Cost of holding inventory over time, currency/year | Applying an annual rate to the wrong inventory increment |
| Obsolescence | Loss of usability or value as specifications or shelf life change | Assuming all excess stock can be sold at book value |
| Scenario | Coherent set of conditions and outcomes | Treating an unweighted stress as a probability forecast |
| Expected loss | Probability-weighted loss over mutually exclusive outcomes, currency/year | Multiplying a severity by an unsupported probability |
| Stress test | Analysis of a specified adverse state | Interpreting its severity as its likelihood |
| Incremental cost | Change relative to a common baseline, currency/year | Comparing one alternative's full cost with another's incremental cost |
| Residual risk | Exposure remaining after a control, in a stated metric | Treating residual risk as zero because a hedge exists |
| Risk appetite | Authorized tolerance for specified outcomes | Replacing a quantified service constraint with a slogan |

## 5. The mechanics, step by step

### Step 1: Frame the decision and the counterfactual

**Input:** product, customer requirements, decision horizon, current operating policy, and a named decision owner.
**Output:** a baseline against which all alternative costs and service outcomes can be compared.

Specify a one-year decision horizon if annual cost is the relevant budget.
Use a daily or weekly operating bucket fine enough to capture a meaningful activation delay.
Define whether the baseline includes ordinary cycle stock, existing backup agreements, and normal purchase prices.
Describe the disruption as a change from that baseline, not as total company revenue at risk.

Decide whose shortage matters: a part supplier, the assembly line, a customer distribution center, or the end customer.
A shortfall at one node can sometimes be covered downstream.
Do not add shortage at both nodes as two independent economic losses for the same unit.
Define whether the lost margin is realized immediately, delayed, or recovered on a later sale.
If service penalties or patient impacts matter, show them separately rather than burying them in a guessed margin.

The alternatives must have comparable scope.
A warehouse stock increase protecting one component is not directly comparable to a network-wide alternate manufacturing contract unless the protected products are matched.
Keep unchanged baseline costs out of an incremental comparison.
Include new costs even if they fall in another department's budget.

### Step 2: Map physical exposure and dependencies

**Input:** bill of materials, supplier sites, lanes, available inventory, and validated recovery estimates.
**Output:** a map of the failing node, affected products, substitutes, and dependencies.

Trace the component to the actual manufacturing site and upstream critical inputs.
A supplier's headquarters address is not a production-site location.
Identify common utilities, subcontractors, tooling ownership, logistics corridors, and raw-material origins.
Check whether alternate locations can operate under the same hazard, including a regional electricity interruption or trade restriction.

Classify each item as interchangeable, conditionally substitutable, or non-substitutable.
Substitution may require engineering validation, customer consent, labeling changes, or regulatory review.
Record who can approve the substitution and the elapsed time under real operating conditions.
Capacity that exists technically but cannot make an approved part is not usable protective capacity.

### Step 3: Build a time-phased baseline

**Input:** opening usable inventory, demand forecast, scheduled receipts, and the disruption timeline.
**Output:** a daily or weekly balance and first service breach.

For a lost-sales case with all receipts available at the start of the bucket:

```text
available[t] = usable_opening[t] + usable_receipts[t]
served[t] = min(demand[t], available[t])
shortage[t] = demand[t] - served[t]
usable_opening[t+1] = available[t] - served[t]
```

Adjust the sequence if receipts arrive after demand in the real operation.
State whether partial fulfillment is permitted and how customers are prioritized.
If backlog is allowed, track it separately and specify whether old orders are served before new demand.
Do not use the lost-sales recurrence unchanged for a backlog model.

For constant demand and no receipts, stock coverage equals usable stock divided by daily demand.
With continuing receipts below demand, divide by the net depletion rate instead.
If usable receipts meet or exceed demand, a simple depletion-based coverage measure is not finite; verify capacity and timing rather than reporting infinity as a guarantee.
Averages can conceal a stockout when demand or receipts occur at different times.
Simulate the actual pattern when bucket timing matters.

### Step 4: Define candidate physical hedges

**Input:** approved alternatives and realistic resource constraints.
**Output:** comparable hedge designs with quantities, lead times, and ownership.

An inventory hedge specifies location, part revision, opening quantity, storage conditions, replenishment, ownership, and release rules.
It also specifies whether the stock is dedicated to one customer or shared across products.
Estimate incremental average inventory, not just the peak buffer, when applying an annual carrying rate.
Add expected scrap, inspection, insurance, and storage where they are incremental and not already included in the rate.

A capacity hedge specifies firm reserved hours or units per period, start date, activation delay, yield, shift availability, and shipment time.
Identify the labor, tooling, utilities, maintenance, and raw materials needed to make the capacity usable.
A reservation at the same site as the disrupted asset does not protect against a site-wide outage.
A reserve at a different site may still depend on the same upstream resin.

A dual-source hedge specifies approved volumes and whether the alternate is already producing or only qualified.
Normal ongoing volume may keep operators trained and expose quality problems before an emergency.
It may also cost more, reduce scale discounts, and require more quality audits.
The alternate's ordinary load uses capacity that cannot simultaneously be offered as incremental emergency output.
Model gross capacity less committed load, maintenance, and the ramp-up profile.

A hybrid specifies how inventory bridges the alternate's activation delay.
It must not count the same reserved units as both on-hand inventory and a future receipt.
Test whether the bridge is still sufficient when detection is late or the alternate ramp is slower than planned.

### Step 5: Add a financial option only for its actual exposure

**Input:** price or foreign-exchange exposure, hedge index, strike, notional, premium, tenor, and settlement terms.
**Output:** a price-dependent cash flow, reconciled to the physical purchase.

For a simple cash-settled call with one settlement price, the gross payoff is notional times the positive difference between index and strike.
Net payoff subtracts the premium once.
If the contract has a cap, deductible, averaging window, different currency, or physical-delivery provision, use its actual terms instead.
A call payoff can reduce net spend but does not prevent an allocation shortage.
A cash-settled option is not proof of access to resin, freight, or production capacity.

Compare the hedge tenor and settlement dates with the physical purchase and cash need.
If the physical purchase price follows a different location, grade, currency, or month than the index, show that basis explicitly.
A positive index payoff may be too small or arrive too late to cover the actual purchase.
Counterparty credit, collateral, liquidity, accounting, and approved hedging authority also matter.
This educational illustration is not investment or accounting advice.

### Step 6: Cost every strategy on the same basis

**Input:** commercial quotes, finance assumptions, unit economics, and activation rules.
**Output:** incremental annual cost and scenario-dependent cost for each alternative.

Inventory cost can include annual carrying rate times average incremental stock value.
If that rate already includes warehousing, do not add the same warehousing charge again.
A rate applied to a fully replenished dedicated reserve is reasonable only if its average level is near the stated reserve; model a lower average if drawdown is material.
For a multi-year qualification outlay, either calculate present value across the whole horizon or use a consistent annual equivalent.
A simple annual allocation is a budgeting convention, not a discounted net present value.

Capacity cost includes reservation fee whether used or not, plus incremental exercise cost when used.
Dual-source cost may include higher purchase price on normal allocation, qualification, audits, and incremental freight.
The baseline purchase price must be subtracted before counting a replacement purchase as an incremental cost.
Account for savings from avoided purchases as well as extra spending.
Treasury option premium belongs in the financial cost once; payoff is a scenario-dependent offset, not a physical unit.

Disruption loss should reflect contribution margin lost on genuinely lost demand, plus separately supported penalties and response costs.
If a sale is delayed rather than lost, value delay and financing rather than automatically charging full lost margin.
Avoid charging both full lost margin and the full retail revenue for the same lost unit.
Make customer harm and nonfinancial constraints visible even if they cannot be monetized credibly.

### Step 7: Create coherent scenarios

**Input:** hazard narratives, duration distributions, demand states, dependence assumptions, and evidence quality.
**Output:** a scenario table with physical outcomes for each strategy.

Include no disruption, a short disruption, a severe disruption, and at least one common-cause event where appropriate.
Preserve a coherent demand story: a storm may both interrupt a supplier and change customer demand.
The alternate's activation and transport assumptions should change if the same storm affects its region.
If scenarios are mutually exclusive and exhaustive, their probabilities must sum to 100 percent over the chosen horizon.

A scenario can represent an annual event rather than an individual incident.
If multiple disruptions can occur within a year, a simple one-event model may understate cost and should be extended.
When probabilities are not credible, present scenario losses without assigning arbitrary expected-value precision.
Use stress tests for consequences and breakeven thresholds for decision sensitivity.

### Step 8: Simulate service and loss

**Input:** scenarios, hedge commitments, and time-phased balances.
**Output:** shortage, service, inventory drawdown, recovery, and cost by strategy and scenario.

For each period, recognize only approved receipts available at the relevant time.
Apply yield losses before treating output as usable stock.
Allocate constrained units according to the stated customer policy.
A reservation fee may secure a maximum quantity rather than require taking it; exercise only according to the scenario and contract.

Calculate unit fill rate as served units divided by eligible demand units over the stated horizon.
Also show the maximum consecutive shortage period when continuity matters.
A 99 percent annual fill rate can still conceal a complete shutdown lasting several days.
Report whether unserved demand is lost, delayed, or diverted; these outcomes have different economic effects.

### Step 9: Compare and challenge

**Input:** strategy-by-scenario costs, service results, budget, and risk constraints.
**Output:** a recommended strategy, alternatives rejected, and decision thresholds.

For mutually exclusive annual states, expected annual incremental cost equals fixed annual hedge cost plus the probability-weighted scenario-dependent incremental costs and losses.
Use the same horizon and currency for every term.
Add a service or maximum-loss constraint before choosing the lowest expected cost among feasible options.
A strategy with a lower expected cost can be unacceptable if it fails a contractual or safety requirement.

Calculate the probability at which an incremental hedge pays for itself in the simplified case: extra annual fixed cost divided by loss avoided per disruption.
This threshold is meaningful only for the stated loss and no other changing costs.
If the implied probability exceeds 100 percent, the hedge cannot pay for itself on those assumptions alone, although nonfinancial constraints may still justify it.
Perform sensitivity analysis on demand, recovery duration, capacity start, loss per unit, price, and dependence.

### Step 10: Contract, activate, and learn

**Input:** selected strategy, supplier evidence, operating controls, and approval authority.
**Output:** an implementable agreement and monitored playbook.

Specify the exact event or authorized decision that triggers activation.
Name a primary and deputy owner, notification channels, response time, and escalation path.
A trigger based only on the supplier declaring force majeure can be too late for a short inventory bridge.
Set evidence requirements for capacity, tooling, materials, quality approval, and transportation.

Include quantity, priority, test frequency, ramp schedule, duration, price, cancellation, and remedies in commercial review.
Review exclusivity and shared reservation claims: the same idle machine cannot serve unlimited customers simultaneously.
Agree what happens if forecast demand changes or the backup cannot perform.
Track expiration of qualification, contracts, certificates, raw material, and stock shelf life.

Run a tabletop and, where feasible, a physical trial of the end-to-end path.
Measure trigger-to-first-good-receipt, not just time to issue a purchase order.
Update the model after an actual disruption, product revision, supplier change, or failed test.
Make the risk owner accept residual exposure explicitly rather than treating an approved budget as proof of readiness.

### Applying the procedure to common decision branches

The following short decision cases are independent illustrations.
Each isolates a practical issue that can change a hedge selection.
They do not alter the assumptions or recommendation of the valve example below.

#### 1. Inventory location

A buyer has 6,000 units of reserve and consumes 1,000 units/day at one plant.
If all 6,000 units are at that plant and usable, the reserve covers six days with no receipts.
If the units are at a distant warehouse, transport and receiving time must be added before they are usable.
An inventory location can therefore change service even when the total quantity is unchanged.
The decision is whether to pay for local stock, resilient transport, or both.
Check theft, fire, temperature control, customer ownership, and transit restrictions at the proposed location.
Do not count the same stock simultaneously at both the warehouse and the plant.
The warehouse record is not evidence that the plant can access the goods during a regional closure.

#### 2. Stock pooling

Two facilities each use 500 units/day and hold 2,000 dedicated units.
Each has four days of protection without replenishment.
Pooling the 4,000 units can improve allocation when one facility is unaffected and stock can travel in time.
It does not create more than 4,000 units or guarantee that the shared stock reaches both sites.
Simultaneous demand from both sites consumes the pool at 1,000 units/day, still four days of coverage.
Include dispatch priority and transfer lead time before treating pooling as risk reduction.
The customer allocation rule must identify who receives stock when total requirements exceed the pool.

#### 3. Yield at the alternate

A backup line starts 1,000 parts/day but only 90 percent pass release tests.
Expected good output at that yield is 900 parts/day, not 1,000 parts/day.
If demand is 1,000 good parts/day, usable inventory falls by 100 units/day after the line starts.
The calculation assumes the stated yield is stable and failed parts cannot be recovered.
In practice, yield may initially be lower and improve after process tuning.
Model a dated ramp profile and ask quality engineering for demonstrated good-output evidence.
If the supply agreement promises gross starts rather than good receipts, clarify the difference.

#### 4. Period boundaries

Suppose a plant begins a day with 100 units and must supply 200 units during that day.
A 100-unit receipt arrives at the end of the day, after the service deadline.
The plant can deliver only 100 units on time; the other 100 units are late or lost under policy.
A daily total that adds receipts before demand would incorrectly report no shortfall.
Record actual cut-off times when short windows or perishable material make them important.
Use smaller time buckets or an explicit within-day ordering rule.
State whether receiving inspection can delay the apparently available receipt further.

#### 5. Ongoing partial receipts

Opening stock is 5,000 units, demand is 1,000 units/day, and reliable receipts are 500 units/day.
If receipts arrive before demand and continue each day, net depletion is 500 units/day.
The stock is depleted after ten days, assuming a constant pattern and no other stock restrictions.
If the 500-unit receipt is itself vulnerable to the shock, that coverage estimate is optimistic.
Specify whether it is a firm unaffected path or simply a normal scheduled order.
At the first period after depletion, each day's shortage is 500 units under lost-sales assumptions.
The model should not extrapolate the continuing receipt indefinitely without evidence.

#### 6. Time-to-recover versus backlog clearance

A supplier returns to 1,000 units/day on day 11 after ten days without production.
Normal customer demand is also 1,000 units/day.
If 3,000 orders are backlogged at that point, output equal to current demand cannot clear the backlog.
The supplier's capability has recovered to its pre-event rate, but customer service has not fully recovered.
If incremental output rises to 1,300 units/day and all excess is allocated to backlog, clearance takes ten days.
That ten-day calculation assumes no new disruptions and ignores production or shipment variability.
Track both capability recovery and demand recovery; neither clock substitutes for the other.

#### 7. Capacity reservation with variable exercise cost

A backup agreement costs USD 20,000/year and charges an extra USD 3 per unit over the normal source.
If 8,000 units are exercised in a disruption, incremental exercise cost is USD 24,000.
The disruption-year incremental supplier cost is USD 44,000 before any remaining lost contribution.
If the backup is not exercised, the fixed USD 20,000 reservation fee remains.
The comparison assumes normal purchase cost is already included in the common baseline.
If the alternate avoids expediting cost, include that saving once in the same scenario.
Do not compare the alternate's full purchase price with only the primary's incremental loss.

#### 8. Minimum purchase obligations

A supplier offers a lower reservation fee if the buyer purchases a minimum annual volume.
The minimum can be useful if it maintains process familiarity and verifies product quality.
It can also force purchases when demand drops or when the primary supplier is cheaper.
Estimate the incremental price on the committed volume and any take-or-pay penalty.
Account for goods actually received rather than treating the entire minimum spend as pure loss.
Check whether the minimum is inclusive of emergency units or additional to them.
Include potential volume displacement and lost primary-supplier discounts in the commercial comparison.

#### 9. Tooling rights

A second molder may have free machine hours but lack a compatible tool.
Building a second tool involves upfront cost, qualification, storage, maintenance, and ownership terms.
Moving the existing tool after a disaster may be impossible if it is damaged or inaccessible.
An escrowed drawing is not equivalent to an available, validated production tool.
Model tool lead time and validation as part of activation if no spare exists.
Clarify whether the buyer can access the tool after supplier insolvency.
Inspect the tool periodically; a nominal spare that cannot run is not an operational hedge.

#### 10. Supplier insolvency

A capacity reservation may protect against a competitor's claim but not the supplier's insolvency.
Advance payment can become an unsecured exposure if the supplier fails before providing goods.
Financial monitoring, tool access, alternative inventory, and contract terms address different parts of this risk.
An option payoff from a different counterparty could offset some loss but cannot restore the supplier.
Ask legal and treasury teams to assess enforceability, prepayment, and counterparty concentration.
Model a scenario in which both reserved capacity and prepaid funds are impaired.
Do not assume that a supplier's financial statements are a perfect early-warning system.

#### 11. Demand surge and supply interruption

A severe weather event might interrupt a supplier while increasing demand for emergency equipment.
Using a normal-demand forecast in that scenario understates the rate at which stock depletes.
For example, 5,000 units cover five days at 1,000 units/day with no receipts.
At 1,250 units/day the same stock covers four days under the same receipt assumptions.
The demand increase and supply interruption belong in one coherent scenario, not independent favorable cases.
Check which customers are contractually prioritized when available stock is scarce.
The relevant forecast is scenario-specific; the service target remains a management choice.

#### 12. Product substitution

An engineer identifies another valve that performs a similar function.
The proposal is not an available hedge until compatibility, validation, and required approvals are established.
Substitution may also change bill-of-material cost, manufacturing yield, labeling, and maintenance requirements.
Estimate both one-time qualification cost and scenario-dependent conversion cost.
Test whether the alternate itself has independent sources for critical inputs.
If substitution is not approved today, include the approval lead time in the receipt schedule.
The owner should maintain the approval as designs and supplier sites change.

#### 13. Backup ramp profile

Suppose an alternate delivers 300 good units on its first day, 600 on its second, and 900 on its third.
Its three-day total is 1,800 units, even if its eventual steady-state rate is 900 units/day.
At 900 units/day of demand, those three days create a 900-unit cumulative deficit without opening stock.
A model that assigns 900 units to all three days overstates output by 900 units.
Production, transit, and receiving profiles may each lengthen the ramp to usable receipts.
Request time-phased output evidence rather than one maximum-rate figure.
Also reserve enough raw material and labor to sustain the promised ramp.

#### 14. Shared reservation rights

Several buyers may each pay to reserve the same flexible line.
Each agreement can look plausible in isolation while total claimed volume exceeds the line's capacity.
A broad regional event can trigger all reservations at the same time.
Ask for priority order, total reserved load, and allocation rules in the executed agreement.
Confidentiality may limit disclosure, so request an independent assurance or quantified allocation cap.
If priority is best effort, model a partial-delivery or zero-delivery scenario.
A fee is evidence of payment, not proof that capacity is exclusively available.

#### 15. Inventory financing

An additional 10,000 units cost USD 25 each, tying up USD 250,000.
At an illustrative 20 percent annual carrying rate on the full average value, carrying cost is USD 50,000/year.
That charge is not the same as a USD 250,000 annual operating expense.
The capital outlay and eventual recovery of inventory value belong in a cash-flow view.
If a separate discount rate or financing charge is used, remove overlapping capital charges from the carrying rate.
If stock becomes obsolete, add the expected incremental write-down not already included.
Different finance conventions can be valid if applied consistently and disclosed.

#### 16. Inventory rotation

A dedicated reserve with twelve months of remaining shelf life must be rotated before expiry.
First-expiring-first-out allocation can keep the reserve fresh if replenishment is reliable.
However, ordinary consumption may temporarily lower the protected balance during replenishment.
Record the minimum available stock during the rotation cycle, not just the maximum after receipt.
If the supplier is already failing, planned replacement of consumed reserve may not arrive.
Monitor age, lot release, replenishment lead time, and minimum coverage together.
Do not claim full protection for stock that is due to expire before the modeled event.

#### 17. Foreign exchange hedge

A buyer pays a supplier in euros but budgets in USD.
A currency option can limit some exchange-rate exposure during its tenor, subject to contract terms.
It does not fix the euro purchase price, secure manufacturing capacity, or ensure export clearance.
The economic exposure is the currency amount actually payable on the relevant dates.
If volumes fall, an oversized hedge can create cash flows that are no longer offsets to purchases.
If the supplier raises its euro price, the exchange-rate hedge alone does not cover that price increase.
Coordinate treasury and procurement forecasts without treating either as a guaranteed commitment.

#### 18. Commodity hedge and basis

A contract settles against an index for one resin grade at one location.
The plant buys a different grade at a different location and pays additional freight.
The observed purchase price can move differently from the index even in the same month.
Construct scenarios with adverse basis as well as an adverse index.
Include the premium and gross payoff in one consistent cash-flow bridge.
Verify whether settlement timing matches the emergency purchase and whether collateral may be required.
The hedge owner must monitor counterparty risk as well as commodity price risk.

#### 19. Currency conversion and units

A resin option may be quoted in USD per metric tonne while purchasing records use kilograms.
One metric tonne is 1,000 kilograms, so a 50-tonne hedge references 50,000 kilograms.
Multiply a USD/tonne price by tonnes, not by kilograms, to obtain USD.
If the physical purchase is in another currency, specify the conversion rate and date.
Do not silently convert using a settlement rate that differs from the cash transaction rate.
The conversion and option notional must match the period actually exposed.
Record rounding conventions so that treasury can reconcile modeled and settled cash flows.

#### 20. Cash timing

Emergency material may require advance payment while an option settles after the buying date.
The option can reduce ultimate economic cost but still leave an immediate liquidity gap.
Make a dated cash-flow schedule for deposit, purchase, premium, settlement, and customer payment.
Test the gap against credit facilities and authority to spend during an incident.
If the counterparty posts collateral, distinguish collateral timing from option payoff.
A positive expected net payoff does not guarantee cash arrives when the invoice is due.
Escalate financing risk separately from physical availability.

#### 21. Service metric choice

Suppose 19,000 of 20,000 units ship on time during a disruption.
The unit fill rate is 95 percent, but the missing 1,000 units could all occur on one day.
If a hospital depends on uninterrupted delivery, that one-day gap may be unacceptable.
Add maximum consecutive shortage days or a priority-customer minimum to the decision criteria.
Specify the eligible demand population and whether late deliveries count as failures.
Do not compare a line-item fill rate with a unit fill rate as though they were identical.
The target needs an owner and a measured time window.

#### 22. Customer prioritization

When only 500 units are available against 800 units of demand, 300 units are unserved.
The total shortage does not say which customer bears it.
An allocation rule may prioritize contractual obligations, safety-critical use, or an agreed proportional split.
Legal and commercial owners should approve the rule before the event.
The warehouse needs an executable order and an escalation path for exceptions.
Model consequences by customer if margin, penalties, or safety impacts differ.
Do not imply that a purely economic ranking overrides applicable obligations.

#### 23. Common-cause probability

Two suppliers each have a 10 percent annual probability of a defined disruption.
Multiplying 0.10 by 0.10 gives a 1 percent joint probability only under independence.
If a shared event disrupts both, the joint probability can be materially higher.
Marginal supplier probabilities alone do not determine the joint probability.
Use shared-site evidence and coherent scenarios to assess dependence.
When data are sparse, present a range of joint outcomes instead of a spurious point estimate.
Confirm that all probabilities use the same event definition and annual time horizon.

#### 24. Recovery duration uncertainty

An inventory reserve may cover a four-week median outage but not an eight-week tail.
The relevant loss depends on the duration distribution and the response after stock is exhausted.
A single recovery estimate can obscure both short events and severe prolonged events.
Calculate shortage under several plausible durations, including a tail stress.
Do not call a stress duration a percentile unless evidence supports that percentile.
Check whether the backup reservation lasts long enough for the severe scenario.
Renewal rights and maximum duration can materially affect the physical protection.

#### 25. Horizon and discounting

A qualification program costs USD 120,000 upfront and is expected to provide three years of capability.
Dividing by three gives USD 40,000/year as a simple budget allocation.
That amount is not a discounted annual equivalent and does not include maintenance or residual value.
For an investment decision, compare the dated cash flows over the same three-year horizon.
Include requalification, price changes, and discounting using the organization's approved method.
If the part becomes obsolete after one year, the useful protection horizon is shorter than planned.
Show both the budget allocation and economic evaluation when decision makers need both.

#### 26. Changed design and qualification

An engineering revision can make both buffer stock and an alternate qualification invalid.
The design-change process should identify affected inventory, tools, contracts, and approvals.
Set a transition date and maintain a revision-specific inventory balance.
If old and new revisions cannot substitute, do not aggregate them into one coverage number.
Assess whether the alternate supplier is qualified for the new process, not just the old drawing.
Include qualification lead time before retiring the old source or buffer.
The decision owner should accept any temporary coverage gap explicitly.

#### 27. Exercise authority

A contract may grant the buyer an option to activate reserved capacity within a notice period.
The plant planner may detect the need but lack authority to commit the exercise payment.
A delayed internal approval can consume the very inventory runway the reservation was meant to protect.
Define a delegated spending limit and a named alternate approver.
Record the latest decision time that still permits first receipt before stockout.
Rehearse decisions during weekends and holidays as well as normal office hours.
Contractual rights are useful only if the organization can exercise them in time.

#### 28. Monitoring and expiry

A signed agreement is not permanent evidence of usable capacity.
Supplier ownership, workforce, equipment, sub-tier inputs, and other customer loads can change.
Set review dates for qualification, insurance, capacity reservation, and test results.
Escalate missed evidence or an expired certificate to the risk owner.
Refresh the model after a product revision or meaningful change in demand.
Retain the previous assumptions so decision makers can see why the result changed.
An annual review is insufficient if a known material change occurs earlier.

#### 29. Bounding value without probability

Suppose a hedge costs USD 25,000/year and avoids USD 250,000 in a specified event.
If those are the only changing cash flows, the breakeven annual event probability is 10 percent.
That threshold is a decision boundary, not an estimate of the event's frequency.
If management believes the rate may be between 2 percent and 20 percent, expected-cost preference changes within that range.
Report the severe-state service improvement and the consequences at each boundary.
The decision can still be justified by a binding service constraint or risk appetite.
Do not attach false precision to an uncertain hazard model.

#### 30. Combining physical and financial hedges

A buyer reserves alternate output at a market-indexed price and buys a cash-settled call on that index.
The capacity contract addresses physical access only to the extent its conditions are met.
The call addresses part of the price exposure only to the extent its index and tenor match.
In a disruption, calculate actual physical receipts and purchases first.
Then compute the separate financial settlement and reconcile the combined cash flow.
If the alternate cannot produce, a favorable option payoff does not create a receipt.
If the buyer does not exercise capacity, verify whether the hedge notional still matches actual exposure.

## 6. Worked example

### Decision and facts

A hypothetical manufacturer makes infusion pumps and buys one approved valve per pump.
The values below are invented for teaching; they are not an estimate of a named company's performance.
Management must select a one-year protection strategy for the valve.
All amounts are in US dollars (USD), all physical quantities are valves, and each day is a production day.

The forecast is 1,000 valves/day.
The operating plan allocates enough normal capacity and material to serve that forecast.
The service target is at least 95 percent unit fill during the severe 20-day interruption.
The only baseline physical commitment in the disruption is the opening usable inventory; no alternate receipt is committed.
This distinction matters: the target does not create supply.

The common baseline has 4,000 usable valves on hand at the beginning of the interruption.
Assume the primary supplier delivers zero valves for 20 days and returns at exactly 1,000 valves/day afterward.
Demand is permanently lost if not served on the day it occurs; no backlog is created.
All receipts arrive before demand in their day.
Assume no price change, no salvage loss, no demand substitution, and a USD 80 contribution margin lost per unserved pump.
The interruption begins at day 1; inventory is not replenished during the 20 days unless the selected hedge says otherwise.

The modeled annual states are a severe 20-day disruption with probability 10 percent and no interruption with probability 90 percent.
These probabilities are deliberately assumed to demonstrate arithmetic, not inferred from evidence.
At most one severe event occurs in the year in this simplified model.
The model measures incremental annual cost relative to normal operation, not total company cost.

### Candidate A: baseline

There is no new hedge cost.
Opening stock covers 4,000 / 1,000 = 4 days.
Demand over the disruption is 20 × 1,000 = 20,000 valves.
Served demand is 4,000 valves; shortage is 16,000 valves.
Fill rate is 4,000 / 20,000 = 20 percent.
Lost contribution is 16,000 × USD 80 = USD 1,280,000 in the severe state.
Expected annual incremental cost is 0.10 × USD 1,280,000 = USD 128,000.
Return of primary production on day 21 does not recover those lost sales under the assumed policy.

### Candidate B: inventory buffer

Buy and hold 12,000 additional usable valves in dedicated stock at USD 20 per valve.
The illustrative carrying rate is 25 percent per year on average incremental inventory value.
Assume the average incremental inventory is the full 12,000 valves for this simplified annual calculation.
Incremental carrying cost is 12,000 × USD 20 × 0.25 = USD 60,000/year.
The purchase principal of USD 240,000 is tied-up working capital, not an additional annual expense in this comparison; the carrying rate represents its modeled annual cost.
Assume the rate includes relevant storage and financing and that no additional scrap cost arises.

Opening usable inventory rises to 16,000 valves, enough for 16 days.
Shortage is 20,000 − 16,000 = 4,000 valves.
Fill rate is 16,000 / 20,000 = 80 percent.
Severe-state lost contribution is 4,000 × USD 80 = USD 320,000.
Expected annual incremental cost is USD 60,000 + 0.10 × USD 320,000 = USD 92,000.
The reserve still misses the 95 percent severe-state target.
A separate cash-flow view would show the USD 240,000 investment and the timing of stock replenishment after drawdown.

### Candidate C: reserved alternate capacity

Pay USD 45,000/year for a firm alternate production reservation.
The alternate is already qualified, has independent inputs for this scenario, and can deliver 1,000 good valves/day on days 6 through 20.
Its capacity is incremental to all other committed loads and is available for the entire assumed outage.
For illustration, its unit price and freight equal the avoided primary purchase, so incremental exercise cost is zero.
This is an assumption, not a general property of reserve contracts.

The initial stock serves days 1–4, or 4,000 valves.
Day 5 has no stock and no alternate receipt, so 1,000 valves are lost.
The alternate serves days 6–20, or 15,000 valves.
Total served is 19,000 valves, and severe-state fill is 95 percent.
Severe-state lost contribution is 1,000 × USD 80 = USD 80,000.
Expected annual incremental cost is USD 45,000 + 0.10 × USD 80,000 = USD 53,000.
The target is met exactly, leaving no allowance for a late receipt or higher demand.

### Candidate D: inventory bridge plus reserved capacity

Hold 1,000 additional valves at the same cost and carrying rate, costing USD 5,000/year.
Combine that bridge with the USD 45,000/year reservation under the same activation terms.
Fixed annual hedge cost is USD 50,000.
The 5,000 opening valves serve days 1–5; alternate receipts serve days 6–20.
Shortage is zero and severe-state fill is 100 percent.
Expected annual incremental cost is USD 50,000.
The hybrid costs USD 3,000/year less in expected terms than the reservation alone under this 10 percent disruption probability, while improving severe-state service.

| Strategy | Fixed incremental cost, USD/year | Severe shortage, valves | Severe fill | Severe loss, USD | Expected annual incremental cost, USD/year |
|---|---:|---:|---:|---:|---:|
| A: baseline | 0 | 16,000 | 20% | 1,280,000 | 128,000 |
| B: inventory | 60,000 | 4,000 | 80% | 320,000 | 92,000 |
| C: reserve capacity | 45,000 | 1,000 | 95% | 80,000 | 53,000 |
| D: hybrid | 50,000 | 0 | 100% | 0 | 50,000 |

On the stated expected-cost and target criteria, select D.
This is a conditional recommendation, not evidence that all firms should prefer hybrid protection.
The bridge is worth an extra USD 5,000/year and avoids USD 80,000 of severe-state loss.
Its breakeven annual severe-event probability versus C is USD 5,000 / USD 80,000 = 6.25 percent.
Below that probability, C has lower expected cost, assuming all other inputs remain fixed; both meet the stated target exactly or better.
Above it, D has lower expected cost.
At the 6.25 percent threshold their expected costs are equal.

### Challenge the recommendation

If alternate delivery begins on day 8 rather than day 6, the hybrid's five-day stock bridge is exhausted after day 5.
Days 6 and 7 each lose 1,000 valves, producing 2,000 lost valves and 90 percent fill in the 20-day outage.
The same USD 50,000 fixed cost then accompanies USD 160,000 of severe-state loss.
At 10 percent probability, expected annual cost becomes USD 66,000, before any changed exercise expense.
The severe-state target is missed, so management must either increase the bridge or secure a faster verified ramp.

If demand rises to 1,100 valves/day while the alternate remains at 1,000/day, a simple five-day bridge is not enough.
An end-to-end daily balance is needed because alternate receipts and remaining stock interact.
Likewise, if the alternate uses the primary supplier's resin plant, the independence assumption fails under a resin-site disruption.
A common-cause scenario could erase much of the reservation's benefit.

Qualification evidence, reserved raw material, and first-good-receipt tests are prerequisites for using the modeled day-6 commitment.
Management should approve D only if those prerequisites are evidenced and the cost of any missing elements is included.
If it cannot secure those terms, rerun the decision with the actual ramp and price rather than presenting the hypothetical result as a commitment.

## 7. Code, query, or calculation

The following Python 3 example reproduces the severe-state shortage and annual expected costs.
It uses the same lost-sales policy, start-of-day receipts, 20-day outage, and 10 percent annual severe-event probability as the worked example.
The capacity dictionary contains receipts during the outage, not capacity available after day 20.
The no-disruption state is assumed to have zero incremental loss and no exercise cost.
The fixed inventory carrying and capacity reservation costs are paid in either state.

```python
DAYS = 20
DEMAND = 1_000
MARGIN = 80
PROBABILITY = 0.10

strategies = {
    "baseline": (4_000, 0, 0),
    "inventory": (16_000, 0, 60_000),
    "capacity": (4_000, 1_000, 45_000),
    "hybrid": (5_000, 1_000, 50_000),
}

for name, (stock, alternate_per_day, fixed_cost) in strategies.items():
    shortage = 0
    for day in range(1, DAYS + 1):
        if day >= 6:
            stock += alternate_per_day
        served = min(stock, DEMAND)
        stock -= served
        shortage += DEMAND - served
    fill = 1 - shortage / (DAYS * DEMAND)
    severe_loss = shortage * MARGIN
    expected_cost = fixed_cost + PROBABILITY * severe_loss
    print(f"{name:9} shortage={shortage:5,d} "
          f"fill={fill:.0%} expected=USD {expected_cost:,.0f}/year")
```

Expected output:

```text
baseline  shortage=16,000 fill=20% expected=USD 128,000/year
inventory shortage=4,000 fill=80% expected=USD 92,000/year
capacity  shortage=1,000 fill=95% expected=USD 53,000/year
hybrid    shortage=    0 fill=100% expected=USD 50,000/year
```

This calculation deliberately excludes backlog, stochastic demand, simultaneous disruptions, taxes, discounting, and capacity exercise charges.
It is not a general-purpose production planning model.
Change one assumption at a time and document what the changed result means.
For example, changing `day >= 6` to `day >= 8` delays the alternate; it does not itself model a shared upstream failure.
A common-cause case requires an explicit scenario where both sources fail or recover together.

### Financial option hand calculation

Suppose a treasury team hedges a forecast purchase of 100 metric tonnes of resin with a cash-settled call.
The underlying index and strike are in USD per metric tonne, with strike USD 1,500/tonne.
The premium is USD 40/tonne, or USD 4,000 for the 100-tonne notional.
At settlement, the index is USD 1,800/tonne, so gross payoff is 100 × (1,800 − 1,500) = USD 30,000.
Net payoff after premium is USD 26,000.
If the actual resin purchase is 100 tonnes at USD 1,850/tonne, physical spend is USD 185,000.
Adding premium and subtracting gross payoff produces net spend of USD 159,000, or USD 1,590/tonne.
The USD 90/tonne above strike consists of the USD 40/tonne premium and USD 50/tonne positive basis.
This arithmetic assumes matching quantities, currency, and settlement timing, and that the resin is actually available.
If the index instead settles at USD 1,400/tonne, gross payoff is zero and the premium remains USD 4,000.
The holder can buy at the physical market price if material is available; the option does not force a purchase at the strike.

## 8. Data, standards, and regulatory context

### Evidence inventory

A defensible model combines enterprise data with supplier evidence and contract review.
Each row needs an owner, an as-of date, a unit, a source, and a known uncertainty.
A precise-looking result based on stale or nonbinding inputs is not a defensible commitment.

| Data element | Source or owner | Validation question | Consequence of error |
|---|---|---|---|
| Product-to-part usage | Engineering bill of materials | Is the revision effective for the modeled period? | Wrong shortage exposure |
| Demand by day | Commercial forecast and order book | Which quantities are orders versus forecasts? | Incorrect depletion speed |
| Opening usable stock | Warehouse and quality records | Is stock released, compatible, and unexpired? | False time-to-survive |
| In-transit stock | Logistics milestones | Will it clear the disrupted lane? | Double-counted receipts |
| Primary recovery time | Supplier continuity plan | Does it cover full recovery or initial restart? | Understated outage |
| Backup output | Supplier capacity and load plan | Is the rate net of other commitments and yield? | Overstated recovery |
| Qualification status | Quality and engineering | Is the site, tool, and product revision approved? | Unusable alternate |
| Activation delay | Actual test or validated schedule | Does it include approvals, transit, and receiving? | Gap before first receipt |
| Input dependencies | Supplier mapping | Do both sources share a sub-tier site? | Correlated failures omitted |
| Carrying rate | Finance | Which storage, capital, tax, and insurance items are included? | Double counting |
| Lost contribution | Finance and sales | Is the demand permanently lost or merely delayed? | Inflated loss |
| Customer penalties | Legal and sales | Are amounts incremental and enforceable in this scenario? | Incorrect economic loss |
| Option settlement | Treasury confirmation | Does the index match the physical grade and period? | Unrecognized basis |
| Reservation rights | Executed agreement | Is quantity firm, shared, conditional, or best effort? | False capacity commitment |
| Shelf life | Quality and warehouse | How much usable life remains on the event date? | Unusable buffer |
| Freight capacity | Logistics provider | Can it operate under the same disruption? | Delayed alternate receipt |

Store event and receipt timestamps in a consistent time zone and preserve the operational day boundary.
Use a documented unit conversion when the financial hedge is in tonnes and production consumes kilograms.
Avoid linking a modeled supplier to a legal-entity identifier without confirming its production sites and subcontractors.
Track missing data explicitly; a blank value should not silently become zero delay or zero exposure.

### Standards and their limits

ISO 22301:2019 specifies requirements for a business continuity management system.
It provides a framework for planning, implementing, maintaining, and improving continuity capability.
Certification against that standard does not prescribe a universal safety stock or prove that a particular supplier reservation will perform.
The relevant standard and independent audit scope must be checked before asserting certification.

ISO 31000:2018 provides risk management guidelines, not a product-specific calculation or a certification requirement for this decision.
The framework supports identifying context, evaluating risk, and monitoring treatment, but it does not supply the probabilities in the example.
GS1 identification standards can help distinguish products and locations when organizations implement them consistently.
An identifier by itself does not validate supplier capacity, geospatial independence, or operational readiness.

For regulated medical devices, source or process changes can require assessment under applicable quality-system and market requirements.
In the United States, the Food and Drug Administration's Quality Management System Regulation became effective February 2, 2026, incorporating ISO 13485:2016 by reference with specified modifications.
Whether a particular alternate supplier or process change requires a submission or other action depends on the product, change, jurisdiction, and applicable rules.
The example does not decide those questions; quality and regulatory professionals must assess them before implementation.
A qualified source in one market is not automatically approved for every market.

Derivative contracts may create reporting, accounting, collateral, or authorization obligations depending on jurisdiction and entity.
The financial call example is an economic payoff illustration, not a statement of hedge-accounting eligibility or legal requirements.
Treasury should consult its executed contract, applicable rules, and advisers.
Do not represent a standard, training example, or supplier assurance as an approval from a regulator.

### Data quality controls

Reconcile stock snapshots to the operational cut-off used by the simulation.
Separate physical stock from inventory position and from future supplier promise dates.
Check that duplicate purchase-order lines do not create duplicate receipts.
Flag stock that is restricted to a specific customer or product revision.
For each alternate, retain the dated evidence behind its first-good-receipt estimate.
When evidence is an untested supplier assertion, report it as an assumption, not a proven commitment.

Reconcile the financial notional with the physical quantity actually exposed during the option tenor.
A hedge larger than forecast consumption can create an unintended speculative exposure.
A hedge smaller than consumption leaves a residual unhedged quantity.
Compare the pricing index's location, grade, units, currency, and settlement month with actual purchases.
Where historical data are used to estimate dependence, investigate whether supplier networks or contract terms changed during the sample.

## 9. Industry practice and case evidence

### Toyota: visibility and buffer decisions

Toyota describes its supply-chain initiatives and its response to the 2011 Great East Japan Earthquake in publicly available sustainability materials.
The event exposed dependencies beyond direct suppliers and motivated attention to supply-chain visibility and continuity.
Public accounts describe the development and use of a supplier database known as RESCUE to help identify affected supply chains.
See Toyota's sustainability disclosures and the Reuters report cited below for their respective descriptions.

**Inference:** visibility can make a buffer more selective by identifying parts for which a long interruption has a high consequence.
It does not follow that the right response is the same inventory percentage for every component.
A database also does not manufacture an unavailable component or guarantee the accuracy of its sub-tier mapping.
A useful operational check is whether the named site, part revision, and recovery estimate are current enough to support allocation decisions.

### Ericsson and Nokia: common component disruption

A widely documented 2000 fire at a Philips semiconductor plant in Albuquerque disrupted supply to mobile phone manufacturers.
A peer-reviewed account by Norrman and Jansson discusses Ericsson's response and supply-chain risk management after the event.
Public reporting and case discussions contrast the manufacturers' responses, but company-specific outcomes depend on multiple factors.
The lesson is not that every second source is automatically effective.
It is that early visibility, approved alternatives, and the ability to execute a response can change the consequences of a component shortage.

**Inference:** a nominally qualified alternative that requires lengthy redesign may be too slow to bridge a short inventory runway.
A decision owner should test the time from a credible early signal to usable alternate output, including customer and engineering approval.
A supplier list without this timing evidence is an exposure map, not a recovery plan.

### Semiconductor capacity: long commitment horizons

Semiconductor fabrication capacity requires specialized equipment and long lead times.
The US Semiconductor Industry Association describes the sector's complex global supply chains and capital intensity in its public materials.
Manufacturing output depends on the specific process technology, qualified design, packaging, and test steps.
A generic claim of spare wafer capacity is not equivalent to a qualified supply commitment for a particular device.

**Inference:** a buyer may rationally use a combination of longer inventory coverage, supplier commitments, and redesign options where immediate capacity switching is infeasible.
The correct combination depends on obsolescence, product life, financing, and customer impact.
A financial semiconductor index hedge, if available and suitable, would still not create a qualified fabrication slot.

### Applying case evidence without overclaiming

Cases support mechanisms and questions for due diligence; they do not provide transferable probabilities for the hypothetical valve example.
Company narratives may emphasize successful actions and omit counterfactual costs.
Press reporting may conflate a vendor, plant, product family, or specific event timeline.
Distinguish what the source establishes from the analyst's proposed implication.
If a company does not publicly disclose inventory levels, do not infer an exact buffer from a recovery outcome.

Look for evidence of operational readiness: a repeated production run, quality approval, confirmed tooling, available inputs, and a logistics test.
Request contractual proof of priority under a shared constraint rather than relying on a general statement of partnership.
If the counterparty cannot supply evidence, preserve that uncertainty in the scenario instead of assigning perfect availability.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Counting quarantined stock | Inventory and quality totals disagree | Unreleased units cannot be shipped | Coverage overstated | Reconcile usable stock by lot |
| Mixing stock and flow | Capacity shown as immediate units | Output accrues only over time | False day-one protection | Simulate receipts by day |
| Ignoring activation | Supplier quote gives only daily rate | Approval and transit delay first receipt | Bridge stock exhausted | Test trigger-to-receipt |
| Nameplate capacity | No net-load schedule | Other customers occupy the line | Promised output unavailable | Reserve net capacity and audit |
| Common upstream input | Two suppliers share one resin plant | One event disables both | Dual source fails together | Map sub-tiers and stress common cause |
| Unqualified alternate | Site not on approved list | Product cannot legally or contractually ship | Capacity is unusable | Complete qualification and maintain it |
| Expired buffer | Aging stock has little shelf life | Reserve cannot meet specification | Apparent stock is worthless | Rotate and monitor remaining life |
| Obsolete revision | Engineering change not linked to stock | Old parts no longer fit | Write-off and no protection | Coordinate revision and drawdown |
| Double-counted capacity | Same line reserved to multiple buyers | Simultaneous claims exceed output | Contract cannot deliver all promises | Verify priority and allocations |
| Average-only coverage | Weekly total masks daily trough | Receipts arrive after demand | Short stockout despite adequate total | Use finer buckets and timing |
| Lost versus delayed confusion | Backlog and lost sales both charged | Same units counted twice | Inflated benefit | Define demand disposition |
| Revenue treated as loss | Sales price used as lost margin | Avoided variable costs omitted | Hedge overvalued | Use incremental contribution |
| Premium counted twice | Net payoff and premium both deducted | Accounting duplicates cost | Hedge appears too expensive | Reconcile gross payoff and premium |
| Index mismatch | Settlement index differs from purchase grade | Basis moves adversely | Payoff misses physical cost | Quantify basis scenarios |
| Option mistaken for supply | Treasury contract cited as backup source | Cash settlement delivers no material | Physical shortage persists | Separate financial and physical controls |
| Late settlement | Payoff arrives after emergency purchase | Cash benefit is not timely | Liquidity shortfall | Model settlement and liquidity dates |
| Wrong probability | One assumed event rate presented as fact | Sparse data lack statistical support | Misleading expected value | Use thresholds and scenarios |
| Shared transport | Both suppliers use one closed port | Alternate output cannot reach customer | Delayed receipt | Map alternate lanes and test them |
| Demand surge ignored | Buffer assumes a constant baseline | Demand consumes stock faster | Coverage overstated | Stress demand and supply jointly |
| Exercise cost omitted | Quote lists only reservation fee | Alternate units cost more on use | Total cost understated | Include incremental exercise price |
| Stale trigger | Plan waits for formal force majeure | Detection-to-action delay grows | Missed activation window | Set earlier observable triggers |
| Unowned plan | No deputy or authority to spend | Decision stalls during incident | Lost recovery time | Assign and rehearse authority |
| Service averaged away | Annual fill is high despite a shutdown | Aggregate metric hides outage | Critical customers miss deliveries | Report outage duration and priority |
| Fixed carrying cost misapplied | Peak inventory used as annual average | Inventory varies through year | Annual cost biased | Model average stock or cash flows |
| Unrefreshed model | Part or supplier changes after approval | Assumptions no longer describe network | Invalid recommendation | Reassess on defined change events |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| Demand is 1,000 valves/day | Enables a transparent balance | Demand surge or shutdown | Re-run daily forecast scenarios |
| Opening stock is usable | Determines first-day coverage | Quarantine, expiry, wrong revision | Reconcile released lots |
| Lost sales are permanent | Supports USD 80/unit loss | Customers accept late supply | Model backlog and delay costs |
| Margin is USD 80/unit | Converts lost units to contribution | Product mix or avoided costs differ | Finance validates incremental margin |
| Severe state occurs with 10% annual probability | Allows expected cost | Sparse evidence or changing hazard | Show breakeven and unweighted stress |
| One event at most per year | Keeps annual state model simple | Multiple or overlapping events | Use event-frequency simulation |
| Alternate starts day 6 | Gives reserve a receipt schedule | Qualification, transport, or detection delays | Measure first-good-receipt in a drill |
| Alternate can make 1,000 good units/day | Sets recovery rate | Yield losses or competing orders | Verify net load and a production run |
| Alternate survives the modeled event | Makes the reserve available | Common supplier or regional outage | Add correlated-failure scenarios |
| Alternate exercise has no incremental unit cost | Isolates reservation economics | Premium freight or higher unit price | Add incremental scenario cost |
| Inventory averages its stated level | Allows simple carrying calculation | Drawdown and replenishment change average | Use time-weighted inventory value |
| Carrying rate captures annual holding cost | Avoids adding overlapping charges | Rate excludes scrap or includes unrelated cost | Reconcile components with finance |
| No backlog clears after recovery | Keeps the severe window independent | Customers defer demand | Extend horizon and model priority |
| No financial hedge protects physical output | Separates different exposure types | Contract includes enforceable physical delivery | Model actual delivery terms separately |
| Option settlement is a single index observation | Supports simple payoff | Contract averages prices or caps payoff | Implement executed contract terms |
| Service threshold is 95% | Provides a decision constraint | Customer needs uninterrupted delivery | Add maximum consecutive outage constraint |

Uncertainty in one input can be correlated with uncertainty in another.
For example, the same storm can delay detection, production, and freight simultaneously.
Varying each input independently may understate the worst coherent scenario.
The model also omits capital budgeting effects beyond the stated annual carrying charge.
For multi-year decisions, calculate discounted cash flows and include residual value, tax, and financing consistently.
No model result overrides a binding contractual or safety requirement.

## 12. Exercises

### Exercise 1: Classify the hedge

**Time:** 10 minutes.
**Task:** Classify each of the following as inventory, capacity, alternate sourcing, financial option, or a combination: a dedicated finished-goods reserve; a paid priority slot at a second approved plant; a cash-settled commodity call; a second supplier with recurring orders.
**Deliverable:** Four classifications and one sentence stating the risk each does not necessarily cover.
**Acceptance criteria:** Distinguish physical access from cash payoff and identify at least one common-cause risk.

### Exercise 2: Calculate coverage

**Time:** 10 minutes.
**Task:** A facility holds 7,200 usable units, consumes 900 units/day, and receives no replenishment during an outage.
Calculate coverage and lost units during a 12-day outage when demand cannot be backlogged.
**Deliverable:** Days of coverage, units lost, and fill rate.
**Acceptance criteria:** Show arithmetic and the stock-flow units; assume receipts would arrive before demand if any existed.

### Exercise 3: Bridge an activation delay

**Time:** 15 minutes.
**Task:** Demand is 800 units/day, initial usable stock is 3,200 units, and an approved alternate begins delivering 800 good units/day at the start of day 7.
Find shortage over a ten-day outage with lost sales, then calculate the additional opening stock needed to eliminate the gap.
**Deliverable:** A day-by-day explanation, lost units, and additional units.
**Acceptance criteria:** Count the day-7 receipt once and do not let later supply recover lost sales.

### Exercise 4: Compare expected cost

**Time:** 20 minutes.
**Task:** Strategy X costs USD 30,000/year and loses USD 200,000 if the severe event occurs.
Strategy Y costs USD 48,000/year and loses USD 50,000 in that event.
Assume the sole adverse state has probability 12 percent per year and no other incremental loss.
Compute expected annual cost for each and the event-probability threshold at which their costs are equal.
**Deliverable:** Two expected costs, the threshold, and a short recommendation.
**Acceptance criteria:** Show units and explain the conclusion below and above the threshold.

### Exercise 5: Challenge dual sourcing

**Time:** 20 minutes.
**Task:** Two suppliers use separate factories but the same upstream wafer foundry and the same export port.
Supplier B promises 2,000 units/day but runs 1,700 units/day for existing customers on a 2,200-unit/day qualified line.
**Deliverable:** A short risk review, estimated uncommitted nameplate capacity, and five evidence requests.
**Acceptance criteria:** Identify both common causes, distinguish gross from incremental capacity, and avoid treating the 500 units/day difference as guaranteed deliverable output.

### Exercise 6: Calculate a call option

**Time:** 15 minutes.
**Task:** Buy a cash-settled call on 50 tonnes, strike USD 2,000/tonne, premium USD 60/tonne.
The index settles at USD 2,300/tonne and the actual physical purchase price is USD 2,350/tonne for all 50 tonnes.
**Deliverable:** Premium, gross payoff, net payoff, physical spend, and effective total spend after the hedge.
**Acceptance criteria:** Count premium once, state the USD/tonne basis, and explain why the contract does not guarantee delivery.

### Exercise 7: Audit a proposed business case

**Time:** 20 minutes.
**Task:** A manager claims a USD 100,000 financial payoff means 10,000 physical units will arrive, adds full sales revenue and lost margin for the same missed units, and charges option premium both within net payoff and separately.
**Deliverable:** A correction memo with the three errors and a defensible calculation structure.
**Acceptance criteria:** Separate cash from units, avoid duplicated lost economics, and show exactly where premium enters.

### Exercise 8: Design a playbook

**Time:** 30 minutes.
**Task:** Design an activation playbook for the worked example's hybrid strategy.
Assume the primary supplier reports an expected outage but the duration is initially uncertain.
**Deliverable:** Trigger, owner and deputy, minimum evidence, decision authority, customer allocation rule, supplier notification, and a test schedule.
**Acceptance criteria:** Protect the five-day bridge, specify a first-good-receipt measure, and include an escalation if the alternate cannot meet day 6.

### Exercise 9: Advise under uncertainty

**Time:** 25 minutes.
**Task:** The worked example's annual disruption probability is judged unreliable and the alternate may be delayed by two days in a common-cause event.
Recommend what management should decide now and what evidence would change the choice.
**Deliverable:** A one-page decision brief with a threshold, at least two scenarios, a service constraint, and an evidence plan.
**Acceptance criteria:** Do not assert an unsupported probability; distinguish the 6.25 percent breakeven from a measured event rate and identify the delayed alternate's service failure.

## 13. Self-check answers

### Answer 1

A dedicated finished-goods reserve is inventory; it can run out in a long outage or become obsolete.
A paid priority slot at a second approved plant combines capacity reservation and alternate sourcing; it can fail if the plants share a critical input.
A cash-settled commodity call is a financial option; it may pay cash but does not deliver the commodity.
A second supplier with recurring orders is dual sourcing; it does not automatically guarantee additional surge capacity.
A complete answer separates the normal order allocation from incremental emergency output.

### Answer 2

Coverage is 7,200 units / 900 units/day = 8 days.
Twelve days of demand is 12 × 900 = 10,800 units.
Lost demand is 10,800 − 7,200 = 3,600 units.
Fill is 7,200 / 10,800 = 66.7 percent, rounded to one decimal place.
The answer assumes the reserve is available at the start of day 1 and no later production fills lost sales.

### Answer 3

Stock serves days 1 through 4: 4 × 800 = 3,200 units.
Days 5 and 6 each lose 800 units.
The alternate's day-7 through day-10 receipts each serve that day's demand.
Total shortage is 2 × 800 = 1,600 units.
An additional 1,600 usable units at opening bridges the two uncovered days under the stated timing.
If day-7 receipts arrive after demand instead, the answer changes; timing is part of the contract.

### Answer 4

X costs USD 30,000 + 0.12 × USD 200,000 = USD 54,000/year in expectation.
Y costs USD 48,000 + 0.12 × USD 50,000 = USD 54,000/year in expectation.
The extra fixed cost of Y is USD 18,000/year and it avoids USD 150,000 per severe event.
The breakeven event probability is 18,000 / 150,000 = 12 percent/year.
Below 12 percent X has lower expected cost; above 12 percent Y does.
At exactly 12 percent the costs are equal, so constraints and risk appetite decide.
These conclusions hold only for the stated two states and costs.

### Answer 5

The foundry and the port are separate common-cause exposures.
The arithmetic difference between qualified gross line capacity and existing load is 2,200 − 1,700 = 500 units/day.
That is an upper bound before maintenance, yield, labor, other obligations, and transport; it is not a firm commitment.
Useful evidence includes the foundry map, port and alternate-lane plan, dated net-load schedule, executed priority reservation, and a qualification and ramp test.
Also request raw-material access and supplier allocation rules under simultaneous customer claims.

### Answer 6

Premium is 50 × USD 60 = USD 3,000.
Gross payoff is 50 × (USD 2,300 − USD 2,000) = USD 15,000.
Net option payoff is USD 15,000 − USD 3,000 = USD 12,000.
Physical spend is 50 × USD 2,350 = USD 117,500.
Effective total spend is USD 117,500 + USD 3,000 − USD 15,000 = USD 105,500, or USD 2,110/tonne.
Equivalently subtract the USD 12,000 net payoff from physical spend; do not also add the premium.
The physical price is USD 50/tonne above the index, a positive basis in this scenario.
The cash-settled option supplies cash, not the 50 tonnes.

### Answer 7

First, remove the claim that a cash payoff guarantees units; model approved physical supply separately.
Second, use lost contribution for permanently lost units and add only independently supported incremental damages, not full revenue plus the same margin.
Third, use either physical spend plus premium minus gross payoff or physical spend minus net payoff.
The two forms are equivalent if they describe the same period, notional, and settlement.
A robust memo also names basis, settlement timing, and counterparty risk.

### Answer 8 rubric

A strong playbook identifies a trigger early enough to order and receive alternate output by day 6.
It names authorized primary and deputy decision makers and a way to reach the supplier outside normal hours.
It verifies available stock, current approved revision, raw materials, reserved net capacity, and transport.
It defines allocation among customers and escalation if the alternate misses its commitment.
It specifies a timed drill measuring trigger-to-first-good-receipt, plus how results update the plan.
There is no single correct trigger without the organization's detection and approval timings.

### Answer 9 rubric

A strong brief distinguishes the 6.25 percent C-versus-D breakeven from an observed event probability.
It compares the original day-6 receipt with a coherent delayed-start scenario and states that the day-8 hybrid yields 90 percent fill in the example.
It recommends a provisional choice conditional on the 95 percent service constraint and verified capacity rights.
It requests evidence on ramp, common sub-tier exposure, alternative freight, and actual loss economics.
It states who accepts residual risk if the service constraint cannot be met.
An answer that invents a precise event probability or treats a financial payoff as physical supply is incomplete.

## 14. Interview questions

1. **How do inventory and capacity differ as disruption hedges?** A strong answer contrasts immediate stock with a rate that requires time and inputs. A weak answer treats a daily capacity figure as stock on hand.
2. **When is a hybrid better than either instrument alone?** A strong answer explains bridging an activation delay and evaluates overlap and incremental cost. A weak answer assumes benefits simply add.
3. **How do you compute days of coverage with ongoing receipts?** A strong answer uses net depletion when constant and a time-phased balance when timing varies. A weak answer always divides stock by gross demand.
4. **How would you validate a backup supplier's 2,000 units/day promise?** A strong answer checks qualification, net load, inputs, contractual priority, yield, and receipt timing. A weak answer accepts nameplate output.
5. **Why can two suppliers fail together?** A strong answer identifies shared sub-tiers, geography, utilities, and logistics and models dependence. A weak answer assumes different company names mean independence.
6. **What does a cash-settled call do in a shortage?** A strong answer explains its index-linked payoff, premium, basis, and lack of physical delivery. A weak answer says it guarantees material.
7. **How do you value an unfilled order?** A strong answer distinguishes lost contribution, deferred demand, penalties, and customer consequences. A weak answer counts both full revenue and margin.
8. **How would you act with no trustworthy disruption probability?** A strong answer presents scenario results, constraints, and breakeven thresholds. A weak answer invents a probability to produce a precise expected value.
9. **What can a high annual fill rate hide?** A strong answer recognizes a concentrated shutdown and adds time-window and customer-priority metrics. A weak answer treats the annual average as complete protection.
10. **How should a resilience plan be governed after approval?** A strong answer names trigger, authority, contract, drill, first-good-receipt evidence, and change controls. A weak answer stops at signing a purchase order.

## 15. Key terms

- **[Activation delay](#4-definitions-and-vocabulary):** elapsed time to first usable alternate receipt.
- **[Backlog](#4-definitions-and-vocabulary):** demand awaiting future fulfillment.
- **[Basis risk](#4-definitions-and-vocabulary):** mismatch between the hedge reference and actual exposure.
- **[Capacity buffer](#4-definitions-and-vocabulary):** available output rate beyond planned load.
- **[Carrying cost](#4-definitions-and-vocabulary):** annual cost of holding stock.
- **[Commitment](#4-definitions-and-vocabulary):** a secured resource or promise, subject to its terms.
- **[Common cause](#4-definitions-and-vocabulary):** shared event that disables multiple paths.
- **[Contribution margin](#4-definitions-and-vocabulary):** revenue less avoidable variable costs.
- **[Coverage](#4-definitions-and-vocabulary):** duration that usable stock supports a defined depletion rate.
- **[Demand](#4-definitions-and-vocabulary):** required quantity per period.
- **[Dual sourcing](#4-definitions-and-vocabulary):** qualifying and using two supply sources.
- **[Exercise](#4-definitions-and-vocabulary):** use of a contractual right.
- **[Expected loss](#4-definitions-and-vocabulary):** probability-weighted loss for stated outcomes.
- **[Fill rate](#4-definitions-and-vocabulary):** served units divided by eligible demand units.
- **[Financial call option](#4-definitions-and-vocabulary):** price-linked right with payoff governed by contract.
- **[Forecast](#4-definitions-and-vocabulary):** estimate of uncertain future demand.
- **[Incremental cost](#4-definitions-and-vocabulary):** cost change relative to a baseline.
- **[Inventory position](#4-definitions-and-vocabulary):** stock plus eligible on-order quantities less backorders.
- **[Lost sales](#4-definitions-and-vocabulary):** demand assumed permanently unserved.
- **[Notional](#4-definitions-and-vocabulary):** quantity or reference amount covered by a hedge.
- **[Obsolescence](#4-definitions-and-vocabulary):** loss of usability or economic value.
- **[Premium](#4-definitions-and-vocabulary):** price paid for an option.
- **[Qualification](#4-definitions-and-vocabulary):** approval of a source and process for the intended use.
- **[Ramp-up](#4-definitions-and-vocabulary):** transition to the required production rate.
- **[Reservation fee](#4-definitions-and-vocabulary):** payment for capacity rights.
- **[Residual risk](#4-definitions-and-vocabulary):** exposure remaining after a control.
- **[Scenario](#4-definitions-and-vocabulary):** coherent set of conditions and outcomes.
- **[Strike](#4-definitions-and-vocabulary):** option payoff threshold.
- **[Time-to-recover](#4-definitions-and-vocabulary):** time to restore a defined capability.
- **[Time-to-survive](#4-definitions-and-vocabulary):** time until a defined service threshold is breached.
- **[Usable stock](#4-definitions-and-vocabulary):** released, compatible, unexpired inventory.

## 16. Sources

1. ISO, [ISO 22301:2019, Security and resilience — Business continuity management systems — Requirements](https://www.iso.org/standard/75106.html). Standard scope; certification does not validate this example's numbers.
2. ISO, [ISO 31000:2018, Risk management — Guidelines](https://www.iso.org/standard/65694.html). General risk-management guidance.
3. GS1, [GS1 standards](https://www.gs1.org/standards). Identification and data-sharing standards; implementation does not by itself demonstrate readiness.
4. US Food and Drug Administration, [Quality Management System Regulation](https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr). Official explanation of the medical-device quality system rule and effective date.
5. Norrman, Andreas, and Ulf Jansson, “Ericsson's proactive supply chain risk management approach after a serious sub-supplier accident,” *International Journal of Physical Distribution & Logistics Management*, 34(5), 2004, pp. 434–456. [DOI: 10.1108/09600030410545463](https://doi.org/10.1108/09600030410545463).
6. Tang, Christopher S., “Perspectives in supply chain risk management,” *International Journal of Production Economics*, 103(2), 2006, pp. 451–488. [DOI: 10.1016/j.ijpe.2005.12.006](https://doi.org/10.1016/j.ijpe.2005.12.006).
7. Toyota Motor Corporation, [Sustainability data book and reports](https://global.toyota/en/sustainability/report/). Company disclosure; consult the specific report and year for a claim.
8. Reuters, [Toyota's supply chain database helps it weather chip shortage](https://www.reuters.com/article/us-autos-toyota-chips-idUSKBN2A20NH/), February 2021. Press reporting on Toyota and RESCUE; case evidence is not a probability estimate.
9. Semiconductor Industry Association, [Strengthening the Global Semiconductor Supply Chain in an Uncertain Era](https://www.semiconductors.org/strengthening-the-global-semiconductor-supply-chain-in-an-uncertain-era/), 2021. Industry association analysis; distinguish industry advocacy from audited company data.
10. CME Group, [Introduction to options](https://www.cmegroup.com/education/courses/introduction-to-options.html). General option mechanics; actual settlement follows the executed contract.

The numerical case, option example, and exercises are hypothetical instructional calculations.
The named sources support general concepts or documented cases, not the invented assumptions in those calculations.

## 17. Links

- [Module 14A index](_index.md)
- [Module 14A survey](../../../course/14A_resilience_disruption_analytics.md)
- [Time-to-recover and time-to-survive](04_ttr-tts.md)
- [Stress testing](05_stress-testing.md)
- [Scenario analysis](06_scenario-analysis.md)
- [Dual sourcing and nearshoring](08_dual-sourcing-nearshoring.md)
- [TTR/TTS stress-test lab](../../../course/14A_resilience_disruption_analytics.md#lab-14a2--ttrtts-stress-test)
