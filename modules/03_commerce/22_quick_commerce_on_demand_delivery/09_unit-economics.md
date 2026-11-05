# Unit Economics for Quick Commerce

> Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics
> Topic 9 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 1, 4, 5, and 7
> Version 2026.5

## 1. Why this topic exists

A delivery can be fast, popular, and operationally smooth while losing money on every incremental order.
A dark store can also show a positive average contribution while still failing to cover the cost of remaining open.
The difference matters when a leadership team must decide whether to change an offer, extend a shift, invest in a neighborhood, or exit a site.

Quick commerce couples small baskets to expensive time-sensitive service.
A customer may pay for groceries and delivery, while the operator pays for merchandise, picking, packaging, payment processing, courier availability, refunds, and spoilage.
Each amount responds differently to order size, distance, density, service promise, and promotion.
An average that hides these differences can reward growth in the least economic orders.

A commercial manager might propose a delivery-fee waiver to raise conversion.
An operations manager might propose batching to reduce courier cost.
A store manager might add opening hours to meet demand.
A finance partner must put each proposal against its actual feasible alternative, including demand response, capacity, worker pay, and customer service.
None of these decisions is answered by gross merchandise value alone.

Subsidies make the measurement harder.
A coupon may be funded by the platform, a supplier, or both.
A supplier-funded promotion can still consume platform cash before reimbursement and can come with conditions.
A large discount may attract an incremental customer, shift a purchase that would have happened anyway, or train existing customers to wait for another coupon.
These cases need different counterfactuals even if the accounting entry looks similar.

Fixed cost complicates the story further.
Rent may be fixed for a week yet avoidable when a lease ends.
A salaried manager may be unchanged by one additional order but avoidable when an entire shift closes after notice.
A shared head-office allocation may be useful for accountability without being a cash saving from closing one store.
There is no universal list of fixed and variable costs independent of the decision and horizon.

This lesson builds an operational contribution bridge, then uses it for decisions.
The example is fictional and its currency, period, assumptions, and calculations are explicit.
It is not a financial reporting template or a prediction of any company's margins.

### Decisions and ownership

| Decision | Accountable owner | Required evidence | Output |
|---|---|---|---|
| Define the order ledger | Finance controller with data engineering | Source contracts, reconciliation, timing | Versioned metric specification |
| Set a fee or basket threshold | Commercial owner | Conversion, mix, contribution and customer effects | Approved offer and eligibility |
| Set courier pay and incentives | Marketplace owner with legal and operations | Applicable agreements, availability, pay and working conditions | Pay schedule and monitoring |
| Change batch policy | Dispatch operations | Feasibility, cost, late tail and courier impact | Bounded policy and rollback |
| Approve a coupon | Growth owner with finance | Funding, incremental demand and budget | Experiment and spending cap |
| Extend a shift | Store operations | Incremental orders, staff coverage and avoidable costs | Staffing decision |
| Open or close a site | General manager with finance | Cash flows, contracts, alternatives and transition costs | Decision memo |
| Sign off accounting | Controller and external advisers as needed | Contracts and applicable standards | Financial reporting treatment |

A model can recommend a decision; the responsible owner approves it.
An order forecast is uncertain, a staffing plan allocates resources, and a customer promise is a commitment under the applicable offer.
A contribution target is neither a forecast nor a customer commitment.

## 2. Learning outcomes

After this lesson you can:

1. **Define** the order population, currency, period, basket, and contribution boundary for a comparable analysis.
   Exercise 1 checks the definitions.
2. **Calculate** merchandise gross profit, net promotional funding, variable cost, and contribution per completed order.
   Exercise 2 tests the bridge.
3. **Separate** courier payout, platform delivery cost, tip, and utilization without treating one as a substitute for another.
   Exercise 3 tests the distinction.
4. **Compute** a conditional basket or volume break-even and explain when no valid break-even exists.
   Exercise 4 checks the arithmetic and assumptions.
5. **Compare** a subsidy using incremental orders, funding, cannibalization, and a no-offer counterfactual.
   Exercise 5 requires an incremental profit calculation.
6. **Classify** costs as avoidable or unavoidable for a specific action and time horizon.
   Exercise 6 applies that classification.
7. **Evaluate** an operational intervention using contribution, customer-service guardrails, and courier welfare.
   Exercise 7 requires a decision and stop rule.
8. **Design** a reproducible ledger and experiment with reconciliation and ownership.
   Exercise 8 uses the calculation and measurement sections.

A strong answer states the denominator and does not call a positive order contribution net profit.
It also distinguishes a modeled sensitivity from a causal estimate.

## 3. The core mental model

```text
Customer demand and promise       Merchant/supplier agreement
             |                              |
             v                              v
   Placed orders and basket  <---- prices, costs, funding terms
             |
             v
   fulfillment + courier offer + travel + handoff
             |
             v
   completed, cancelled, refunded and open order events
             |
             v
   order economics ledger, linked to payments and payroll
             |
             +--> earned merchandise margin and customer fees
             +--> less platform-funded discounts and variable costs
             +--> plus earned third-party reimbursement
             |
             v
   contribution per eligible order and total contribution
             |
             +--> less decision-avoidable shift/site costs
             +--> less other site costs and allocated corporate costs
             v
   decision-specific incremental cash result or reported profit
             |
             v
   compare feasible alternatives under service and welfare constraints
```

Follow the cash and the physical work, but do not confuse management economics with audited revenue recognition.
The operating bridge begins with merchandise sales less merchandise cost, not with the full basket treated as profit.
It adds eligible fees and supplier funding, then subtracts platform discounts and the costs needed to serve the order.
The bridge is only meaningful when each amount appears once.

The **unit** in this lesson is a completed customer order.
A completed-order denominator permits comparison of delivered experiences, but canceled orders still generate costs.
Assign canceled-order costs to the relevant completed-order cohort or show them separately in total contribution; never silently exclude them.
If a period has no completed orders, report total costs and an undefined per-completed-order ratio, not zero.

The **boundary** is variable contribution before store-level fixed costs.
Some picking expense is variable only over a planning horizon where staffing can change.
The worked example assumes the stated picking cost is genuinely avoidable for the relevant order-volume change.
If pickers are already scheduled and cannot be redeployed, an immediate dispatch decision uses a different marginal cost.

An extra order is valuable only if its incremental revenue and funding exceed its incremental cost and displaced alternatives.
A full store may reject an apparently positive order if it displaces a more valuable feasible order.
An idle courier does not necessarily make delivery free: guaranteed pay, wait time, insurance, and incentives can still change.

An average describes a population; a decision compares two futures.
If a campaign raises average basket size but reduces order frequency, the total contribution can fall.
If a fee raises contribution on retained orders but drives customers away, the relevant difference includes lost orders.
If a promotion causes orders to move from next week to this week, a longer observation window is needed.

Contribution is not a safety objective.
Unsafe loads, illegal pay practices, inaccessible service, and food-safety breaches are not acceptable trade-offs merely because the spreadsheet improves.
Hard requirements are screened first; a policy can be considered only among feasible alternatives.

### Three useful views

| View | Numerator | Denominator | Question |
|---|---|---|---|
| Order contribution | Variable revenue and funding less variable cost | Completed orders in a defined cohort | What does an average delivered order contribute before site costs? |
| Site operating result | Total contribution less site costs | Site and period, optionally orders | Can the site cover its operating footprint? |
| Incremental decision result | Cash-flow difference between alternatives | Decision and horizon | Does this change improve the feasible counterfactual? |

None of the three views replaces the others.
A positive order contribution can coexist with a loss-making site.
A loss-making site may still be worth operating in the short run if closure saves less than the contribution it forfeits.
A profitable site can make an unprofitable investment if the proposed expansion adds more cost than benefit.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Order | Customer request with a stable identifier | Counting lines, retries, or delivery attempts as distinct orders |
| Completed order | Order meeting the ledger's delivery-completion rule | Including placed orders in a completed-order denominator |
| Order cohort | Orders grouped by a shared placement or completion period | Mixing cohorts between numerator and denominator |
| Gross merchandise value (GMV) | Defined merchandise transaction value, currency per period | Treating GMV as revenue or gross profit without a contract-based definition |
| Basket | Merchandise value per order at a specified price and discount basis | Including delivery fees without saying so |
| Average order value (AOV) | Total defined basket value divided by eligible orders, currency/order | Averaging store averages without their order weights |
| Merchandise cost | Cost attributable to goods sold, currency/order | Subtracting the purchase cost of goods still in inventory |
| Gross profit | Merchandise sales less cost of merchandise sold, currency/order | Confusing gross profit with contribution after delivery |
| Gross margin rate | Merchandise gross profit divided by merchandise sales, percent | Applying a margin rate to fees or to the wrong basket basis |
| Markup | Gross profit divided by merchandise cost, percent | Treating markup as margin |
| Delivery fee | Customer charge for delivery, currency/order | Assuming it covers all delivery cost |
| Service fee | Separately defined customer charge, currency/order | Treating every service fee as merchandise margin |
| Platform-funded discount | Value borne by the platform, currency/order | Subtracting the same coupon twice |
| Supplier reimbursement | Earned payment from a supplier under an agreement | Assuming all nominal coupon value is collectible |
| Subsidy | Benefit paid by one party to reduce another party's effective price or cost | Calling all customer discounts an investment with proven return |
| Courier payout | Amount payable to the courier for a defined period or task | Equating payout with total fulfillment cost or take-home profit |
| Tip | Customer gratuity subject to local terms and law | Assuming it can always offset the platform's wage obligations |
| Guaranteed pay | Contractual or policy payment not necessarily proportional to completed jobs | Assigning no cost to idle time without checking guarantees |
| Incentive | Additional payment tied to work, supply, or other conditions | Treating a one-off incentive as a permanent cost saving |
| Delivery cost | Defined platform cost of fulfilling delivery, currency/order | Omitting waiting, incentives, returns, or batch effects |
| Picking cost | Labor and directly attributable cost of collecting an order | Assuming every scheduled labor minute is avoidable |
| Packaging cost | Bags, insulation, seals, and related material, currency/order | Omitting incremental cold-chain materials |
| Payment cost | Processing charges and related transaction fees | Applying a percentage to an unspecified payment base |
| Refund | Amount returned or credited under the stated policy | Subtracting full merchandise value after already reversing its revenue |
| Waste | Cost of inventory lost or impaired, currency per period | Counting a damaged item's cost both in cost of goods sold and waste |
| Contribution | Defined revenue and funding less specified avoidable costs | Calling it net profit without subtracting other costs |
| Contribution margin | Contribution divided by a named revenue or basket base | Reporting a percentage without its denominator |
| Fixed cost | Cost unchanged over a specified range and horizon | Assuming it is permanently unavoidable |
| Avoidable cost | Cost absent in the counterfactual because of a decision | Treating an allocation as an actual saving |
| Sunk cost | Past cost unchanged by either alternative | Including it as a reason to continue an unprofitable future action |
| Step cost | Cost that changes when a capacity threshold is crossed | Assuming a constant per-order cost at all volumes |
| Allocation | Assignment of pooled cost to units using a chosen driver | Treating an allocated amount as marginal cash cost |
| Capacity | Feasible throughput for a defined resource and period | Using average throughput as an unconstrained maximum |
| Utilization | Busy time divided by a clearly defined available time | Ignoring unpaid or excluded waiting time |
| Batch | Multiple orders served in a combined delivery route | Assuming one route's total cost applies separately to each order |
| Cannibalization | Activity that would have occurred without the intervention | Counting all treated orders as incremental |
| Incrementality | Difference between outcomes under intervention and counterfactual | Equating observed growth with causal impact |
| Break-even | Point where specified revenue and cost totals are equal | Ignoring fixed costs or a changing contribution rate |
| Cohort maturity | Time allowed for refunds, funding, and cancellations to arrive | Treating an immature cohort as final |
| Contribution waterfall | Ordered reconciliation from sales to defined contribution | Mixing gross and net measures in the same bridge |
| Opportunity cost | Value of the best feasible alternative use of a constrained resource | Counting an impossible future order as certain |
| Sensitivity | Model output under a changed assumption | Calling a one-factor sensitivity a causal forecast |
| Scenario | Internally consistent set of assumptions | Combining incompatible best-case inputs |
| Counterfactual | Outcome expected without the proposed change | Using zero activity as the default baseline |

## 5. The mechanics, step by step

### Step 1 — State the decision, unit, and horizon

**Input:** the decision request, available alternatives, service area, and date.
**Output:** one decision statement and an order population.

Write the question before choosing a metric.
“Should we extend opening hours by one hour on Friday for the next eight weeks?” is testable.
“Are orders profitable?” is too broad to identify the relevant costs.

For each alternative record the time horizon, operational constraints, and who can approve it.
Choose a currency and whether all figures exclude indirect taxes.
Use completed orders, placed orders, or customer cohorts deliberately; each answers a different question.
State the treatment of cancellations and the time allowed for late adjustments.

A daily dispatch decision usually treats an existing lease as unchanged.
A decision to renew the lease considers future rent that can actually be avoided.
A one-week extension may use scheduled workers; a year-long expansion may require another shift or site.
The same expense can therefore be fixed in one comparison and avoidable in another.

**Branch:** if the request concerns financial reporting, involve the controller and use applicable accounting policy.
The contribution ledger is a management view and cannot by itself establish gross versus net reported revenue.

### Step 2 — Identify the commercial model

**Input:** customer terms, supplier contracts, merchant agreements, and fee rules.
**Output:** a transaction map identifying who sells what to whom.

An inventory-led operator may purchase goods and sell them to the customer.
A marketplace may arrange a merchant sale and receive a commission plus fees.
Their reported revenue and direct cost can differ even for the same customer basket.
Before doing arithmetic, establish the role of each party and the contractual basis.

For an inventory model, a useful operational gross-profit measure is merchandise sales less merchandise cost of sales.
For a marketplace, the operator may instead earn merchant commission or another contractually defined amount.
Do not combine merchandise gross profit with a commission on the same transaction unless both are legitimately earned and distinct.
Do not infer financial-reporting principal or agent status solely from the word “platform.”

Document whether discounts apply before or after taxes, whether fees are refundable, and who receives tips.
Treat sales tax or value-added tax as an explicitly modeled item rather than silently treating it as margin.
Confirm whether supplier funding requires proof, minimum volume, caps, or a later settlement.

**Branch:** if contract terms are incomplete, report a range or hold the decision until the accountable commercial and finance owners supply the missing terms.
Do not invent a reimbursement.

### Step 3 — Build the order population

**Input:** order, delivery, cancellation, and payment events with timestamps and identifiers.
**Output:** one row per order plus a reconciliation to source totals.

Define the placement period and the status cutoff.
For a placement cohort, allow a maturity window for delivery, returns, refunds, and chargebacks.
Keep open orders visible until their outcome is known.
If a cohort has matured only partially, label it provisional and quantify the unmatched share.

Remove duplicate event messages without deleting distinct financial adjustments.
An order can have multiple delivery attempts, payments, refunds, and coupon lines.
Aggregate each child table to the order level before joining to avoid multiplying dollars.
Maintain a bridge from placed to completed, canceled, failed, and open counts.

A completed-order metric should include the costs of canceled orders generated by that cohort, either allocated explicitly or shown in a companion period total.
If the numerator includes failed delivery costs and the denominator excludes failed deliveries, say so.
For acquisition or conversion decisions, placed or eligible customers may be the better denominator.
Use more than one view when it prevents a selection bias.

**Branch:** if identifiers do not reconcile, quarantine the unmatched records and quantify their value.
Do not set unmatched courier cost or refunds to zero merely to make the join succeed.

### Step 4 — Define basket and merchandise margin

**Input:** item lines, selling prices, quantities, substitutions, and cost of goods.
**Output:** merchandise sales, merchandise cost, gross profit, and basket distribution.

For this lesson, basket is merchandise sales at the pre-coupon price, excluding delivery and service fees and indirect taxes.
The customer coupon is shown separately.
An alternative net-sales definition is valid only if the corresponding discount is not subtracted again.

Calculate gross profit as merchandise sales minus the cost of merchandise sold.
Gross margin rate is gross profit divided by merchandise sales when merchandise sales are positive.
A 28% margin on a $32.00 basket is $8.96, not $32.00 of profit.
A 28% markup on cost would be a different percentage of selling price.

Inspect the basket distribution, not just its average.
A larger basket may add picks, weight, volume, packaging, or a second trip.
Item margins differ by category and supplier funding.
Averages may change because of customer mix rather than because every customer bought more.

**Branch:** if item-level margin is unavailable, use a weighted historical rate with a documented error range.
Do not apply one rate to delivery fees or to refunded items whose sales have been reversed.

### Step 5 — Add fees and promotional funding exactly once

**Input:** customer charges, coupon records, funding agreements, and settlements.
**Output:** a bridge from list-price basket to net economic inflows.

List customer delivery and service fees separately from merchandise.
Subtract the full customer coupon if the basket is pre-coupon.
Then add the portion reimbursed by a third party only when the agreement makes that funding attributable and earned.
In the worked example the $2.20 coupon is shown gross and $0.50 supplier funding is added back.
The net platform-funded discount is therefore $1.70 per completed order.

A coupon directly reducing merchandise sales could instead be netted against sales, with the corresponding funding treatment shown consistently.
Both bridges should yield the same management contribution when they represent the same economics.
Never subtract the $2.20 coupon and then also subtract its $1.70 net platform cost.

Recognize that supplier funding can be capped, disputed, delayed, or contingent.
A receivable is not a cash receipt; a campaign can be contribution-positive but cash-constrained.
Reconcile expected funding against signed terms and actual settlement.

**Branch:** if no agreement supports the claimed funding, exclude it from the base case and show it as a contingent upside only if justified.

### Step 6 — Measure courier pay and delivery cost

**Input:** accepted offers, courier payments, guarantees, incentives, batch routes, time logs, and refunds.
**Output:** total courier payout and platform delivery cost for the same cohort or period.

Start with what the platform owes couriers, including relevant base pay, per-distance or per-time pay, bonuses, and applicable guarantees.
Separate customer tips from platform-funded payout and show the legal and contractual treatment.
Do not assume that tips reduce platform obligations.
Courier earnings are not the same as take-home income: vehicle, fuel, equipment, insurance, and tax costs may be borne elsewhere.

For period economics, divide all courier payouts attributable to the period by the chosen order denominator.
Include idle-time guarantees and incentives if they are real costs of supplying the service.
For an individual order decision, use the actual incremental payout and any capacity opportunity cost instead of a period average.
When a route serves two orders, count the route payout once in the total.
Allocation between orders can help diagnosis, but the allocation does not change total cash.

Include platform-paid insurance, dispatch support, failed attempts, and other delivery costs in their appropriate categories if they are avoidable and not already included.
Record wait time at the store, travel, access, handoff, and return travel where relevant.
An observed high utilization can mask unpaid waiting if the denominator includes only active delivery time.

**Branch:** where the pay rule is hourly or has a minimum guarantee, model the shift-level payout before allocating it.
An extra order can increase the payout by less than the average, by the full per-order rate, or by a step change.

### Step 7 — Include fulfillment, payment, refunds, and waste

**Input:** labor rosters, time observations, invoices, payment fees, inventory loss, and refund records.
**Output:** a non-overlapping avoidable-cost ledger.

Separate variable picking time from fixed scheduled coverage.
A labor standard is not automatically a cash saving if the schedule cannot be changed.
Include onboarding, overtime, and step changes only where the decision changes them.

Include the bag, seal, insulation, and any other packaging that the order actually uses.
Payment fees may include a percentage, a fixed transaction charge, chargeback costs, and reversals.
Apply the fee rule to the correct payment base, including or excluding taxes as specified by the provider.

Refund accounting requires particular care.
If the full sale is reversed, do not also subtract the same full refund as an expense from the original gross sale.
For management economics, either restate sales and cost of goods or calculate the net incremental loss consistently.
In the example, refunds are a $0.45 incremental loss after any related sales and cost reversals.

Waste can be measured as inventory cost lost to spoilage, damage, shrink, or expiry.
A unit already included in cost of goods sold must not be charged a second time as waste for the same loss.
Separate ordinary sales, inventory adjustments, and returned goods that can or cannot be resold.
A per-order allocation of waste is a descriptive average; avoidability depends on how volume changes replenishment and expiry.

### Step 8 — Calculate the contribution waterfall

**Input:** reconciled inflows and outflows under the chosen boundary.
**Output:** total contribution, contribution per order, and a clearly labeled margin percentage if useful.

For the inventory-led example, let:

```text
contribution = merchandise gross profit
             + customer delivery and service fees
             + earned supplier funding
             - full customer discount
             - courier payout and other delivery cost
             - avoidable picking, packaging, payment, refunds and waste
```

All quantities must use the same currency, order cohort, and discount convention.
The equation is a management bridge, not a mandatory financial-statement layout.
For a marketplace, replace the merchandise gross-profit term with the operator's relevant earnings and costs under its contract.

Sum each component over the entire cohort before dividing by the count of completed orders.
If sites have different order volumes, calculate a weighted aggregate rather than taking the unweighted mean of their per-order results.
Show both total and unit contribution.
A small positive unit amount can still produce a site loss after fixed costs.

**Branch:** when contribution is negative, adding identical orders increases the modeled loss.
Volume alone does not produce fixed-cost break-even under unchanged unit economics.

### Step 9 — Move from average contribution to a site result

**Input:** total contribution, site costs, capacity, and the period.
**Output:** a site-level operating bridge with each cost separately labeled.

Subtract site costs not already included in contribution, such as a lease, minimum staffing, equipment, utilities, or local management.
Label shared corporate allocations separately from direct site costs.
Show depreciation, financing, taxes, and exceptional costs where the decision or reporting view needs them.
Avoid mixing noncash accounting expense and incremental cash flow without explaining the distinction.

A simple fixed-cost break-even is fixed cost divided by positive contribution per order.
It assumes the same contribution, prices, service level, and fixed cost across the relevant volume range.
It is invalid when extra volume changes the number of shifts, courier guarantees, or facilities.
If required orders exceed safe capacity, the formula's arithmetic answer is not an executable plan.

**Branch:** if contribution is zero or negative and fixed cost is positive, no positive finite order volume breaks even under these assumptions.
A policy or cost structure must change; merely scaling the same economics worsens or preserves the loss.

### Step 10 — Separate fixed, avoidable, allocated, and sunk cost

**Input:** cost contracts, staffing commitments, notice periods, redeployment alternatives, and decision horizon.
**Output:** incremental cash flows for each alternative and a cost-avoidability schedule.

A cost can be fixed per order and avoidable per site.
For example, a shift supervisor's weekly salary may be unchanged by one additional order but avoidable after a store closure and notice period.
A lease may remain payable for six months after closure, so its short-run avoidable amount is zero even if future renewals can be avoided.
A shared finance allocation does not become a cash saving unless the shared resource or external bill actually changes.

Past leasehold investment is sunk in a forward-looking decision.
A future removal cost or lease termination payment is not sunk if it changes between alternatives.
Cash released from working capital, inventory liquidation, severance, and lost contractual funding may also matter.
Make the sign, timing, and uncertainty of each explicit.

Avoid counting the same lost contribution twice: once as foregone sales and again as a closure cost.
If workers can be redeployed to a valuable alternative, include the realistic value of that alternative rather than assuming redeployment automatically saves their pay.
For longer horizons, compare discounted cash flows using an approved discount rate and consistent timing.

**Branch:** if contractual terms or staff notice are unknown, use a range and request confirmation before claiming a closure saving.

### Step 11 — Analyze basket, fee, and delivery levers

**Input:** baseline economics and evidence on demand, cost, and operations.
**Output:** a scenario comparison with explicit behavioral and capacity assumptions.

Increasing basket value is not the same as increasing contribution by the whole extra basket.
At an unchanged 28% merchandise margin, a $1.00 increase adds $0.28 gross profit before extra fulfillment and payment costs.
A minimum basket can also deter customers or change item mix.
Estimate the response rather than assuming order count stays constant.

A delivery-fee increase adds revenue only on retained orders and may change demand, tips, promotion eligibility, or service expectations.
A fee waiver may increase orders but can subsidize orders that would have happened anyway.
Compare total contribution across the whole eligible population, not only fee-paying survivors.

Batching can share route time but may require waiting, longer detours, extra packaging, or additional compensation.
The feasible savings depend on order density and courier pay rules.
Validate on-time performance, food safety, courier earnings, and actual total payout rather than just modeled travel distance.

**Branch:** if a scenario exceeds capacity or violates a service or safety requirement, reject it regardless of its apparent contribution.

### Step 12 — Estimate subsidy incrementality

**Input:** treatment and comparison groups, offer rules, cost, supplier support, and outcome window.
**Output:** incremental contribution, uncertainty, and budget exposure.

Choose the unit of randomization to match spillovers: customer, geography, or time block.
A delivery system can have shared couriers and inventory, so one customer's offer may affect another customer's waiting time.
Predefine an analysis that includes those spillovers where possible.

Compare outcome totals per eligible randomized unit, including customers who place no order.
Include the discounts on all treated orders, not only those assumed incremental.
Subtract control contribution from treatment contribution and adjust for third-party funding and incremental operating costs exactly once.
Use a window long enough to detect delayed or pulled-forward demand, refunds, and customer retention effects.

An observational before-and-after comparison can be confounded by weather, holidays, competitors, and changes in courier supply.
Use randomization when feasible, or explain the identification assumptions and limitations of a quasi-experiment.
A lifetime value estimate is a forecast with retention and discount-rate assumptions; it is not a realized contribution today.

**Branch:** if a supplier reimburses only up to a cap, model the cap at campaign level rather than multiplying a nominal per-order reimbursement beyond its limit.

### Step 13 — Decide, monitor, and revisit

**Input:** alternative-specific economics, constraints, uncertainty, and operational readiness.
**Output:** an approved decision memo, owner, metrics, and stop conditions.

Show the baseline, each feasible alternative, and a no-change counterfactual.
State expected total contribution, incremental cash flow, service levels, capacity, and distributional effects.
Separate forecasts from commitments and from aspirational targets.

Assign the controller to reconcile the ledger, operations to validate capacity, the marketplace owner to validate courier pay, and the commercial owner to approve the offer.
The accountable executive makes the trade-off after legal, safety, and financial constraints are satisfied.
Predefine a rollback trigger and a date for review.

Monitor both the unit metric and total contribution.
Track refunds after the original cohort matures, plus supplier funding collected against earned.
A dashboard should expose denominator changes, late-arriving events, and the version of the cost model.

**Branch:** if the decision's observed conditions differ materially from the assumptions, stop or re-evaluate rather than silently replacing the baseline.

## 6. Worked example

### Case and assumptions

A fictional inventory-led operator runs one dark store for one week.
All values are US dollars excluding indirect taxes.
The cohort contains 1,000 completed orders placed in that week.
Costs of cancellations and failed attempts attributable to that cohort are included in the stated averages.
The figures are a teaching dataset, not a public company result.

The customer-facing basket is measured at pre-coupon merchandise prices.
The store also charges delivery and service fees.
A customer receives a coupon, and the supplier reimburses a part of it.
The payout to couriers is a platform expense; customer tips are outside this example and are not assumed to offset it.

The store's finance partner has reconciled merchandise cost, supplier funding, and refunds at cohort maturity.
The example treats picking, packaging, payment, and waste as avoidable per order within the small volume range studied.
It treats the $2,400 weekly site cost separately and assumes it does not change within that range.
It reports a further $1,000 corporate allocation separately; that allocation is not presumed avoidable upon site closure.

### Order contribution bridge

| Component | Calculation | USD/completed order | Treatment |
|---|---|---:|---|
| Merchandise sales | Given | 32.00 | Pre-coupon basket, not all profit |
| Merchandise cost | 32.00 × 72% | −23.04 | Cost of sold goods |
| Merchandise gross profit | 32.00 − 23.04 | 8.96 | 28% gross margin |
| Delivery fee | Given | 2.40 | Customer charge |
| Service fee | Given | 0.60 | Customer charge |
| Supplier reimbursement | Earned and collectible | 0.50 | Funding for coupon |
| Customer coupon | Gross coupon | −2.20 | Full discount, shown once |
| Courier payout | Base and allocated guarantees | −6.40 | Includes attributable payout |
| Avoidable picking | Given | −2.10 | Assumed adjustable over horizon |
| Packaging | Given | −0.55 | Materials per order |
| Payment processing | Given | −0.78 | Same payment base across baseline |
| Refund net loss | Given | −0.45 | After related reversals |
| Waste | Given | −0.32 | Excludes goods already charged above |
| **Contribution** | Sum of bridge | **−0.34** | Before weekly site cost |

Check the inflow side: $8.96 + $2.40 + $0.60 + $0.50 = $12.46 per order.
Check the cost side: $2.20 + $6.40 + $2.10 + $0.55 + $0.78 + $0.45 + $0.32 = $12.80 per order.
Thus contribution is $12.46 − $12.80 = −$0.34 per completed order.
The coupon burden net of the supplier's $0.50 is $1.70 per order.
It is not an additional $1.70 deduction on top of the bridge.

At 1,000 completed orders, total contribution is −$340 per week.
After $2,400 of weekly site cost, the site result before corporate allocation is −$2,740 per week.
After the $1,000 corporate allocation, the management result is −$3,740 per week.
These figures do not include unspecified interest, tax, or other corporate items and are not an audited net-income statement.

### Reconcile the customer payment

The customer pays $32.00 merchandise plus $2.40 delivery and $0.60 service less the $2.20 coupon.
That is $32.80 per order before indirect taxes and tips.
The supplier adds $0.50 per order of earned funding, so total operating receipts attributable to these lines are $33.30 per order.
Subtract $23.04 of merchandise cost and $10.60 of other listed costs excluding the coupon.
The result is again −$0.34 per order.

The other cost sum is $6.40 + $2.10 + $0.55 + $0.78 + $0.45 + $0.32 = $10.60.
This second route confirms that the $2.20 coupon and $0.50 reimbursement were not double counted.
It does not prove that the illustrative assumptions describe a real business.

### Basket-size sensitivity

Hold the 28% gross margin rate and all non-merchandise amounts fixed.
The contribution equation in dollars per order is `0.28 × basket + 3.50 − 12.80`.
Here $3.50 is the $2.40 delivery fee, $0.60 service fee, and $0.50 supplier funding.
The $12.80 includes the full coupon and all listed variable costs.

At a $35.00 basket, merchandise gross profit is $9.80.
Contribution becomes $9.80 + $3.50 − $12.80 = $0.50 per order.
The increase from baseline is $0.84, not $3.00.
This is a conditional sensitivity: actual payment fees, pick time, item mix, and demand may change with the basket.

The variable-contribution break-even basket solves `0.28 × basket + 3.50 − 12.80 = 0`.
The exact threshold is $9.30 / 0.28 = $33.214285... per order.
At cent precision, $33.22 is the first basket at or above the threshold under these assumptions.
It produces approximately $0.0016 contribution per order before rounding.
The threshold covers no weekly site cost.

### Courier and batching scenario

Suppose a feasible batch policy reduces average courier payout by $0.80 per completed order.
The team also forecasts that a service-price concession reduces delivery fee revenue by $0.20 per order.
Expected extra refund loss is $0.12 per order because a few deliveries become less reliable.
These are assumptions, not demonstrated causal effects.

The net change is +$0.80 − $0.20 − $0.12 = +$0.48 per order.
New contribution is −$0.34 + $0.48 = $0.14 per order.
At the same 1,000 orders, total contribution is $140 per week before fixed site costs.
The site result before corporate allocation is $140 − $2,400 = −$2,260 per week.
After the unchanged $1,000 corporate allocation, it is −$3,260 per week.

The policy improves modeled contribution but does not make the site profitable.
The implied site-cost break-even volume at constant $0.14 contribution is $2,400 / $0.14 = 17,142.857 orders per week.
Rounding up gives 17,143 completed orders per week, assuming all else stays fixed.
That is an arithmetic threshold, not an operational forecast or a commitment.
If actual safe weekly capacity is 4,000 orders, this simple threshold is infeasible without changing capacity and its associated costs.
Including the $1,000 corporate allocation would produce a different reporting threshold, but that does not make the allocation avoidable.

Before approval, dispatch must establish feasible routes and the service team must evaluate the late-delivery tail.
The marketplace owner checks courier earnings, paid and unpaid waiting, offer acceptance, and applicable obligations.
Food-safety rules and customer commitments remain hard constraints.
If the $0.80 saving comes only from shifting unpaid waiting to couriers, the platform's measured payout may fall while the wider burden rises.

### Closure comparison

Consider a separate four-week closure decision with no transition costs for simplicity.
Assume the baseline weekly contribution is −$340 and the store can avoid $1,500 of its $2,400 weekly site costs immediately on closure.
The remaining $900 is committed during those four weeks.
The corporate allocation does not change in cash.
Closing changes weekly cash by `+$340 avoided negative contribution + $1,500 avoided site cost = +$1,840`.
Over four weeks the illustrative improvement is $7,360, before any transition costs, customer effects, inventory liquidation, or changes elsewhere.

Now suppose a different store contributes +$2,000 per week and has the same $1,500 immediately avoidable site cost.
Closure changes weekly cash by −$2,000 + $1,500 = −$500.
Even if its full allocated report shows a loss, closing it immediately would reduce cash by $500 per week under those assumptions.
The lease and redeployment terms, not the label “fixed,” determine the answer.

### Subsidy counterfactual

Suppose a randomized trial has equal-size eligible groups over one week.
Control customers place 1,000 completed orders; treatment customers place 1,100.
Before the treatment coupon, both groups have $3.00 contribution per order under the same cost boundary.
The coupon costs the platform $1.00 on every treated order after any supplier funding.
Assume no other effect on cost, no spillovers, and all orders mature within the window.

Control contribution is 1,000 × $3.00 = $3,000.
Treatment contribution is 1,100 × ($3.00 − $1.00) = $2,200.
Incremental contribution is $2,200 − $3,000 = −$800.
The 100 additional orders contribute $300 before coupons, but the platform spends $1,100 on coupons.
The 1,000 orders that would have occurred anyway also receive the discount.
An increase in orders is therefore not sufficient evidence of profitable subsidy.

This comparison excludes acquisition benefits after the week and any capacity or service spillovers.
Those effects may matter, but they require evidence and a longer counterfactual rather than an assumed positive lifetime value.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It calculates the worked contribution, the conditional basket threshold, the batching scenario, a site break-even, closure comparison, and coupon experiment.
It is deliberately a small deterministic management model, not a reporting engine or a causal estimator.

Assumptions:

- Every amount is in USD per completed order unless named as a weekly total.
- The basket is merchandise value before the coupon and excludes indirect taxes.
- The 28% merchandise gross margin rate stays constant as basket changes.
- The coupon is deducted once and supplier funding is added once.
- Refunds and waste are incremental net losses, with no duplicated goods cost.
- All non-merchandise order amounts remain constant in the basket sensitivity.
- Site costs remain fixed only within the modeled volume range.
- The batching and experiment inputs are assumptions, not measured treatment effects.
- `Decimal` is used for transparent decimal calculations; no external package or file is needed.

```python
from decimal import Decimal, ROUND_CEILING

D = Decimal


def contribution(basket, gross_margin, fee, service, funding, coupon, costs):
    """USD per completed order, before site fixed costs."""
    merchandise_gp = basket * gross_margin
    return merchandise_gp + fee + service + funding - coupon - sum(costs.values(), D('0'))


def basket_break_even(gross_margin, fee, service, funding, coupon, costs):
    """Basket in USD needed for zero contribution under constant inputs."""
    if gross_margin <= 0:
        raise ValueError('positive gross margin required for this threshold')
    return (coupon + sum(costs.values(), D('0')) - fee - service - funding) / gross_margin


def required_orders(fixed_cost, unit_contribution):
    """Smallest whole order count covering fixed cost, or None if impossible."""
    if fixed_cost < 0:
        raise ValueError('fixed cost must be nonnegative')
    if fixed_cost == 0:
        return 0
    if unit_contribution <= 0:
        return None
    return int((fixed_cost / unit_contribution).to_integral_value(rounding=ROUND_CEILING))


def closure_change(lost_contribution, avoided_cost, transition_cost=D('0')):
    """Incremental cash benefit of closing instead of continuing."""
    return avoided_cost - lost_contribution - transition_cost


def coupon_increment(control_orders, treatment_orders, pre_coupon_contribution, net_coupon):
    """Difference in total contribution for equally sized eligible groups."""
    control = D(control_orders) * pre_coupon_contribution
    treatment = D(treatment_orders) * (pre_coupon_contribution - net_coupon)
    return treatment - control


costs = {
    'courier': D('6.40'),
    'picking': D('2.10'),
    'packaging': D('0.55'),
    'payment': D('0.78'),
    'refund_net_loss': D('0.45'),
    'waste': D('0.32'),
}
base = dict(
    basket=D('32.00'), gross_margin=D('0.28'),
    fee=D('2.40'), service=D('0.60'), funding=D('0.50'),
    coupon=D('2.20'), costs=costs,
)
unit = contribution(**base)
threshold = basket_break_even(
    base['gross_margin'], base['fee'], base['service'],
    base['funding'], base['coupon'], costs,
)
scenario = unit + D('0.80') - D('0.20') - D('0.12')
orders = 1000
site_cost = D('2400')
corporate_allocation = D('1000')

print(f'Baseline contribution/order: ${unit:.2f}')
print(f'Baseline contribution/week: ${unit * orders:.2f}')
print(f'Baseline site result/week: ${unit * orders - site_cost:.2f}')
print(f'Baseline after allocation/week: ${unit * orders - site_cost - corporate_allocation:.2f}')
print(f'Basket threshold: ${threshold:.4f}')
print(f'Batch scenario contribution/order: ${scenario:.2f}')
print(f'Batch scenario site result/week: ${scenario * orders - site_cost:.2f}')
print(f'Site break-even orders/week: {required_orders(site_cost, scenario)}')
print(f'Closure change/week: ${closure_change(D("-340"), D("1500")):.2f}')
print(f'Coupon incremental contribution: ${coupon_increment(1000, 1100, D("3"), D("1")):.2f}')
```

Expected output:

```text
Baseline contribution/order: $-0.34
Baseline contribution/week: $-340.00
Baseline site result/week: $-2740.00
Baseline after allocation/week: $-3740.00
Basket threshold: $33.2143
Batch scenario contribution/order: $0.14
Batch scenario site result/week: $-2260.00
Site break-even orders/week: 17143
Closure change/week: $1840.00
Coupon incremental contribution: $-800.00
```

The program deliberately does not allocate a corporate charge to the closure decision.
If the organization can actually eliminate a corporate cost, add that avoidable cash flow explicitly.
Likewise, if a new order requires a second picker or a courier guarantee, model the resulting step cost rather than extending the simple ratio.
The threshold helper rejects a nonpositive merchandise margin rather than presenting a misleading basket solution.
The volume helper returns `None` when positive fixed cost cannot be recovered by nonpositive unit contribution.

### Extending the calculation responsibly

Replace hard-coded averages with a reconciled cohort-level ledger before making an operational decision.
Aggregate costs before joining one-to-many transaction tables.
Add separate rows for customer refunds, reversals, and received funding so that the result can be audited.
Model tiered delivery fees and payment costs as functions of basket and customer behavior.
For campaigns, compute outcomes per eligible randomized unit and quantify uncertainty.
For site decisions, add timing, lease exit, labor notice, and working-capital cash flows.
Protect individual courier and customer records through access controls and retention limits.

## 8. Data, standards, and regulatory context

### Minimum data contract

| Dataset | Grain and keys | Required fields | Main check |
|---|---|---|---|
| Orders | One order ID | Placement, completion, store, status, currency | Count bridge to source |
| Item lines | Order ID and line ID | Quantity, list price, substitution, tax | Sum to merchandise basket |
| Inventory cost | Item and cost layer or approved cost method | Cost of sold goods and adjustments | Reconcile to inventory ledger |
| Fees | Order and fee line | Fee type, charge, reversal | Reconcile to customer payment |
| Promotions | Order and promotion ID | Gross discount, funder, terms | Reconcile coupon and funding |
| Courier payments | Courier payment and route/order keys | Base, incentive, guarantee, adjustment | Reconcile to payout ledger |
| Route events | Route, stop and timestamp | Assignment, arrival, handoff, cancellation | Detect shared-route duplication |
| Labor | Shift, store and interval | Hours, rate, overtime, role | Match payroll and schedule |
| Packaging | Item and order or consumption period | Quantity and unit cost | Reconcile purchases and stock |
| Payments | Transaction and order | Gross, processing fee, reversal, chargeback | Reconcile processor settlement |
| Refunds | Refund and order | Reason, amount, related sales reversal | Avoid double-counting |
| Waste | Inventory event and cost | Quantity, reason, cost, date | Exclude already expensed goods |
| Fixed commitments | Contract and site | Amount, term, notice, termination cost | Confirm avoidable horizon |

Store event time and processing time separately.
A courier adjustment posted in a later period may belong economically to an earlier order cohort.
Keep both the original version and adjustment, and record the maturity cutoff used in each report.
A corrected report should be reproducible from a versioned snapshot.

Amounts need a consistent currency and tax basis.
When transactions cross currencies, record the exchange-rate source, date, and conversion convention.
Do not average already-rounded unit contributions when the underlying amounts and counts are available.
Use sufficient precision in computation and round for presentation.

### Reconciliation controls

1. Reconcile order counts from placed to completed, canceled, failed, and open.
2. Reconcile merchandise totals to item lines and the payment ledger.
3. Reconcile courier payouts to payment statements, including late adjustments.
4. Reconcile discounts to promotional redemptions and supplier settlement.
5. Reconcile inventory cost and waste to inventory adjustments without overlap.
6. Reconcile payments and refunds to the processor and bank settlement where available.
7. Reconcile site costs to the general ledger and label corporate allocations.
8. Quantify unmatched records and unresolved values before release.

A join that duplicates a two-order route payout produces an apparent cost increase that no courier actually received.
A join that excludes a route without a matching order produces the opposite error.
Use route-level totals for the financial reconciliation even if an allocation is used for operational analysis.

### Accounting standards and management metrics

The IFRS Foundation's IFRS 15 addresses revenue from contracts with customers and includes principal-versus-agent considerations.
Whether an operator reports gross merchandise consideration or a net fee requires analysis of the facts and contracts under the applicable framework.
The operational contribution bridge in this lesson is not a substitute for that analysis.
It can be useful even when its presentation does not match statutory revenue.

The IFRS Foundation's IAS 2 addresses inventory accounting, including cost and net realizable value.
A management waste estimate needs to reconcile with the organization's actual inventory accounting policy.
The example's treatment is a simplified management convention, not an instruction to book a particular journal entry.
IFRS applicability depends on the entity and jurisdiction; other frameworks may apply.

### Labor, tips, and courier classification

Courier classification and pay requirements depend on jurisdiction and facts.
As an example of local regulation, New York City publishes minimum pay information for app-based restaurant delivery workers and specifies how its rules apply.
Do not extrapolate that city's rules to grocery delivery, another city, or a different worker classification without checking coverage and current requirements.
A model must not assume that a customer's tip can satisfy a platform's obligation unless the relevant rules and agreement permit that treatment.

Use paid time and worker experience measures with transparent definitions.
Platform cost per order is not equivalent to a courier's hourly earnings or net income after expenses.
Labor counsel and the marketplace owner should review any proposed pay or dispatch policy before deployment.
Rules and interpretations can change; verify their current text at the decision date.

### Product safety, privacy, and competition

Food handling, temperature control, and allergen obligations depend on the product, location, and applicable law.
Consult the relevant regulator and the site's food-safety owner before changing holding time or delivery batching.
A financial sensitivity cannot waive a mandatory safety requirement.

Order data and courier location can reveal personal patterns.
Restrict access, limit retention, and use aggregation or other appropriate safeguards consistent with applicable privacy law and company policy.
Do not publish individually identifiable earnings or customer-level health-sensitive baskets in a dashboard without an authorized purpose and safeguards.

Promotional funding and pricing agreements can create competition-law and consumer-protection questions.
The commercial team should preserve the actual contract and advertised offer, not infer legal compliance from the margin model.
This material is educational and not legal or accounting advice.

### Standards versus conventions

| Item | Role in this lesson | What it does not establish |
|---|---|---|
| IFRS 15 | Contract-based revenue recognition framework when applicable | A universal per-order contribution formula |
| IAS 2 | Inventory accounting framework when applicable | That every inventory loss is avoidable per order |
| Local delivery-worker pay rules | Binding obligations where their scope applies | A universal courier wage or tip treatment |
| Customer-facing offer terms | Define charges and promotion conditions | A substitute for applicable law |
| Internal metric specification | Consistent management reporting convention | Statutory accounting classification |
| Experiment protocol | Defines causal comparison and safeguards | Guaranteed transferability to another market |

## 9. Industry practice and case evidence

### Evidence and limits

Public reporting often provides company-level financial statements rather than store-level contribution per order.
A company's reported revenue, gross order value, adjusted earnings, and contribution measures are not interchangeable.
Definitions can change across filings and differ by business model.
Use a filing's own definitions and reconciliation before comparing companies.

This section distinguishes documented public disclosures from analytical implications.
No claim below asserts that the fictional example represents an actual company's store economics.
Reported company measures are not necessarily audited individually, even when included in a filing containing audited financial statements.

### DoorDash: marketplace scale and reporting definitions

DoorDash publishes annual reports and securities filings through its investor relations site and the US Securities and Exchange Commission.
Its filings describe a marketplace business, financial statements, risk factors, and company-defined operating measures.
The exact definitions and current numbers should be read from the relevant filing rather than copied from a different period.

**Documented fact:** DoorDash files annual reports that describe its business and financial measures.
**Inference:** a marketplace analyst cannot assume the customer's full food or grocery basket is identical to the platform's recognized revenue.
The actual classification must follow the contracts and applicable accounting policy.
**Limit:** company-level reporting does not reveal the marginal courier cost or contribution of a particular store-hour.

A responsible comparison therefore reconciles the company's reported metric before attempting an operational bridge.
It also avoids applying the fictional inventory-led cost-of-goods formula directly to a merchant marketplace.

### Instacart: gross transaction value versus company economics

Maplebear Inc., which operates as Instacart, publishes filings and investor materials describing its marketplace and financial measures.
Its reports distinguish transaction activity from measures of company revenue and expenses.
The investor-relations site links to current and historical reports.

**Documented fact:** the company publicly files financial information and describes its operating measures.
**Inference:** transaction value can grow while per-order or company profitability moves differently, because fees, promotions, fulfillment costs, and mix can change.
**Limit:** public aggregates cannot establish the economics of a specific local promotional offer without additional transaction-level data.

When interpreting a marketplace filing, read definitions for advertising, transaction revenue, incentives, and fulfillment costs.
Do not assume the same term has the same numerator across two companies.
The course model is a tool to ask questions, not a replacement for the filed reconciliation.

### New York City: a local pay constraint

New York City's Department of Consumer and Worker Protection publishes guidance about minimum pay for app-based restaurant delivery workers.
The published material identifies covered work and explains its pay framework.
The city also publishes updates and guidance, so the decision-maker should check the current version and effective dates.

**Documented fact:** the city has published a minimum-pay framework for covered app-based restaurant delivery work.
**Inference:** a delivery-cost model for covered operations must include the applicable pay obligations and not optimize below them.
**Limit:** this example alone does not determine whether a particular grocery order, worker, or app is covered.

An operations team can still compare feasible delivery routes and manage dispatch quality.
It must do so within the applicable rule and with accurate measurement of the relevant work and pay.

### What these cases teach together

First, external reporting is a useful source of business-model context, not a ready-made store-level ledger.
Second, public transaction value is not automatically margin available to pay couriers.
Third, delivery labor is both an economic cost and a set of obligations to people performing work.
Fourth, a comparative claim needs a defined period, geography, denominator, accounting basis, and source.

A credible decision memo links its own actual ledger and contracts, identifies the applicable regulatory scope, and labels forecasts as forecasts.
It does not infer local profitability from a company-wide earnings release.
It does not cite a pay rule from a different jurisdiction as if it controlled the current site.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Basket treated as profit | GMV equals modeled revenue contribution | Merchandise cost omitted | Attractive but false unit economics | Bridge through merchandise cost and gross profit |
| Discount counted twice | Net sales and coupon both fall | Gross and net definitions mixed | Contribution understated | Choose one discount convention and reconcile |
| Funding invented | Unsettled claims exceed contract cap | Nominal reimbursement treated as earned | Contribution overstated | Validate contracts, caps, and settlement |
| Refund counted twice | Refund and sales reversal both subtracted | Same economic loss recorded twice | False loss attribution | Reconcile original sale, reversal, and net loss |
| Waste counted twice | Spoiled item appears in sold-goods cost and waste | Inventory events overlap | Margin understated | Trace inventory disposition and cost layer |
| Route payout duplicated | Courier total rises after joining order lines | One-to-many join multiplies rows | False delivery-cost signal | Aggregate child tables and reconcile payout total |
| Idle guarantees omitted | Low payout/order despite paid idle shifts | Only active delivery pay counted | Cost understated | Reconcile all shift payouts and incentives |
| Tips offset incorrectly | Platform payout falls when tips rise | Tip treatment assumed rather than verified | Legal risk and misleading economics | Separate tips and review applicable obligations |
| Courier costs shifted outward | Platform payout falls while worker expenses or unpaid waiting rise | Narrow cost boundary hides burden | Unfair or unsustainable policy | Report earnings, time, and expense context |
| Completed-order selection bias | Completion rate falls while contribution/order improves | Failed orders excluded from numerator | Worse total economics hidden | Include failure costs and show full count bridge |
| Average hides mix | Overall contribution rises while every segment worsens | Segment weights change | Wrong policy conclusion | Report weighted segment and total results |
| Fee increase judged on survivors | Per-order fee rises but orders drop | Lost customers excluded | Total contribution falls | Compare eligible-population totals |
| Coupon treated as incremental | All coupon redemptions credited as new | Existing demand subsidized | Campaign loss | Randomize and count coupons on all treated orders |
| Payback window too short | Sales spike followed by a trough | Orders pulled forward | Incrementality overstated | Extend outcome window and track cohorts |
| Fixed allocation used as saving | Closure memo includes unchanged head-office charge | Allocation confused with cash reduction | Bad site decision | Identify actual avoidable resources and timing |
| Sunk investment drives decision | Memo argues to continue to recover past spend | Past cost is unchanged across futures | Further losses | Compare future incremental cash flows |
| Linear break-even beyond capacity | Required volume exceeds feasible throughput | Constant costs and margin extrapolated | Unexecutable target | Model steps, constraints, and incremental capacity |
| Low contribution scaled | Team plans growth with negative unit contribution | More orders repeat a loss | Larger loss | Change economics before assuming scale helps |
| Service harm ignored | Late tail or complaints worsen after batching | Average cost dominates hard constraints | Customer loss and safety exposure | Predefine guardrails and rollback |
| Partial cohort presented as final | Refunds or funding arrive after reporting cutoff | Late events excluded | Biased early contribution | Use maturity lag and revisions |
| Price elasticity ignored | Basket threshold assumes no customer response | Demand and mix assumed constant | Misleading forecast | Experiment or use bounded response scenarios |
| Different time bases mixed | Fees use placement week but payouts use payment week | Timing mismatch | Apparent unexplained margin swings | Cohort costs by event and reconcile settlement timing |
| Corporate profit overstated | Positive contribution labeled net income | Site and corporate costs omitted | Misleading governance | Show full waterfall and name the boundary |
| Unmeasured spillovers | Treatment raises delivery time in control area | Shared capacity contaminates control | Biased experiment estimate | Randomize at suitable cluster and monitor spillovers |

A red flag is a reason to investigate, not proof of fraud or a specific root cause.
Prioritize errors by their possible impact on the decision and reconcile to an independent total.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test or respond |
|---|---|---|---|
| Completed order is the chosen unit | Comparable contribution denominator | Failure costs are omitted | Show placed-to-completed bridge and failure costs |
| Gross margin is 28% | Basket sensitivity needs a cost relationship | Mix, supplier prices, or substitutions change | Estimate item-level margin by basket band |
| Other costs stay fixed per order | Enables a simple basket threshold | Larger baskets need more labor or higher payment fees | Recalculate with volume-sensitive cost functions |
| Supplier funding is earned | Supports adding $0.50 per order | Cap, documentation, or contract conditions fail | Reconcile agreement and settlement |
| Picking is avoidable over horizon | Supports contribution classification | Staff cannot be rescheduled or redeployed | Review roster, notice, and actual staffing response |
| Courier cost includes guarantees | Avoids understating fulfillment expense | Late payout, external cost, or missing incentive | Reconcile route and shift payouts to payment ledger |
| Refund is a net loss | Avoids duplicate revenue reversal | Original sale remains gross while loss is misstated | Trace a sample through accounting entries |
| Waste excludes sold-goods cost | Prevents duplicate cost | Inventory system charges a unit twice | Match inventory events to sale and disposition |
| Site cost remains $2,400/week | Supports simple break-even | New shift, equipment, or site is needed | Build stepwise capacity and cost schedule |
| Corporate allocation is unchanged | Makes closure cash comparison explicit | Closure permits actual reduction in shared resources | Obtain approved resource and contract plan |
| No transition cost in closure example | Isolates arithmetic | Severance, termination, and inventory losses occur | Add dated transition cash flows |
| Treatment and control are comparable | Supports incremental coupon calculation | Spillovers, imbalance, or differential missingness | Randomize, monitor balance, and analyze eligible units |
| Outcomes mature in the trial window | Ensures refunds and later orders are counted | Customers shift purchases across weeks | Extend follow-up and report later adjustments |
| Service remains feasible in scenario | Prevents cost-only recommendation | Late tail, capacity, safety, or worker conditions worsen | Screen constraints and run guarded experiment |
| Currency and tax basis are consistent | Enables addition of amounts | Mixed currencies or gross and net tax values | Validate currency, conversion, and tax fields |
| Unit economics are stable over volume | Enables fixed-cost ratio | Price, labor, route density, or capacity changes | Evaluate scenarios and volume bands |

A break-even is conditional on the assumptions used to produce it.
The more the required volume lies outside observed operating conditions, the less credible a linear extrapolation becomes.
Use ranges and scenarios where uncertainty is material; do not imply false precision from a calculator.

## 12. Exercises

### Exercise 1 — Define the unit and bridge

**Time:** 15 minutes.
A platform reports $40.00 average basket, $3.00 delivery fee, and 25% merchandise gross margin for 500 completed orders.
It also reports 50 canceled orders with $100 total failed-attempt cost.
The basket is pre-coupon and excludes tax.

**Deliverable:** define the population and write the merchandise gross profit per completed order.
State how to present the canceled-order cost without hiding it.
**Acceptance criteria:** your answer uses $10.00 gross profit, identifies the 500-order denominator, and accounts for the $100 once.
Explain why $40.00 is not contribution.

### Exercise 2 — Build an order waterfall

**Time:** 20 minutes.
A $36.00 merchandise basket has a 25% gross margin.
The customer pays a $2.00 delivery fee and $0.50 service fee, receives a $1.50 coupon, and a supplier reimburses $0.40.
Courier payout is $5.50, picking is $1.80, packaging is $0.50, payment cost is $0.60, and refund/waste net loss is $0.40 per order.
All values are USD per completed order and exclude indirect taxes.

**Deliverable:** a component table and contribution per order.
**Acceptance criteria:** subtract the full coupon once, add funding once, and show the resulting $1.60 contribution before site costs.
Provide the net platform-funded coupon amount as a separate explanatory figure.

### Exercise 3 — Interpret courier cost

**Time:** 15 minutes.
A courier is guaranteed $80 for a four-hour shift and receives a $20 platform bonus.
The shift completes 20 orders in 12 routes.
Customers tip $30 in total, paid to the courier separately and not offsetting the guarantee in this exercise.

**Deliverable:** calculate platform courier payout per completed order and explain why dividing by routes answers a different question.
**Acceptance criteria:** include guarantee and bonus once, exclude tips from platform payout, and distinguish payout from worker net income.
Name one missing cost or time measure.

### Exercise 4 — Find break-even and identify infeasibility

**Time:** 20 minutes.
A site has $3,000 fixed cost per week and contribution of $0.75 per completed order.
Its safe capacity is 3,500 completed orders per week.
Assume contribution and fixed costs are constant inside that range.

**Deliverable:** compute the arithmetic break-even and recommend whether the current footprint can achieve it.
**Acceptance criteria:** calculate 4,000 orders per week, compare with capacity, and describe what must change before claiming feasibility.
Explain the answer if contribution becomes negative while fixed cost stays positive.

### Exercise 5 — Test a subsidy

**Time:** 25 minutes.
Equal-sized randomized customer groups are observed for a mature four-week period.
The control group completes 2,000 orders; treatment completes 2,200.
Pre-coupon contribution is $2.50 per order in both groups.
A $0.75 platform-funded coupon applies to every treated order.
Assume no additional costs or spillovers.

**Deliverable:** calculate treatment and control total contribution and the incremental effect.
**Acceptance criteria:** charge the coupon to all 2,200 treated orders, not just the extra 200.
State one reason the result might not transfer to a different market.

### Exercise 6 — Classify closure cash flows

**Time:** 25 minutes.
A site generates $5,000 contribution per week.
A closure would avoid $2,000 weekly staffing and $1,200 weekly rent after notice; an additional $800 corporate allocation would remain unchanged.
Immediate closure also incurs a one-time $4,000 termination cost.
Assume staffing and rent savings begin immediately for a four-week comparison and ignore all other effects.

**Deliverable:** calculate four-week incremental cash change from closing and classify each line.
**Acceptance criteria:** exclude the unchanged allocation, include lost contribution and the one-time cost, and identify the result as negative.
Explain why the past cost of fitting out the store is not part of the comparison.

### Exercise 7 — Evaluate a batch proposal

**Time:** 25 minutes.
A proposed dispatch policy forecasts $0.70 lower courier payout per completed order, $0.10 lower fee revenue, and $0.15 higher refund loss.
The current contribution is $0.05 per order.
The operations team has not yet measured the policy's late tail, food-safety exposure, or courier waiting.

**Deliverable:** calculate the projected contribution and write a conditional decision with one owner and two stop conditions.
**Acceptance criteria:** show $0.50 forecast contribution per order, distinguish modeled from realized savings, and require service and welfare evidence before rollout.
A numeric result alone is not approval.

### Exercise 8 — Design a reconciled experiment

**Time:** 35 minutes.
Your company proposes a fee waiver for customers near one dark store.
The store shares courier supply with an adjacent zone, and supplier funding has a campaign cap.
Refunds can arrive two weeks after delivery.

**Deliverable:** one-page experiment and ledger design.
**Acceptance criteria:** define randomization and eligible population, counterfactual, outcome per eligible unit, funding cap, maturity window, spillover handling, reconciliation, accountable owners, and rollback rules.
Include at least one customer-service guardrail and one courier-welfare measure.
Explain how you will interpret a rise in orders with a fall in total contribution.

## 13. Self-check answers

### Exercise 1

Merchandise gross profit is $40.00 × 25% = $10.00 per completed order.
The defined cohort has 500 completed orders, plus 50 canceled orders that require separate visibility.
The $100 failed-attempt cost can be shown separately or allocated as $0.20 per completed order, provided it is included once in the same cohort's total.
The $40.00 basket is merchandise sales, not profit; merchandise cost and other fulfillment costs remain to be deducted.

A different denominator, such as 550 placed orders, would require all numerator components to use the same population.
Do not silently divide completed-order merchandise margin by placed orders and call it the same metric.

### Exercise 2

Merchandise gross profit is $36.00 × 25% = $9.00.
Inflow components are $9.00 + $2.00 + $0.50 + $0.40 = $11.90.
The coupon and costs sum to $1.50 + $5.50 + $1.80 + $0.50 + $0.60 + $0.40 = $10.30.
Contribution is therefore $11.90 − $10.30 = $1.60 per completed order.
The net platform-funded coupon is $1.50 − $0.40 = $1.10.
If that $1.10 is substituted for the gross coupon and funding lines, the answer remains $1.60.

This result is before site costs, corporate costs, interest, tax, and other excluded items.
It is not evidence that the site earns $1.60 of net income on each order.

### Exercise 3

Platform courier payout is $80 + $20 = $100 for the shift.
Dividing by 20 completed orders gives $5.00 per order.
Dividing by 12 routes gives approximately $8.33 per route; the two denominators answer different questions because routes can serve multiple orders.
The $30 customer tips are shown separately and do not reduce the assumed platform payout.
The courier's receipts before expenses are $130 in this simplified example, but actual net income requires information on expenses and taxes.
Additional relevant information includes waiting time, vehicle costs, insurance, and any other guaranteed compensation.

### Exercise 4

Arithmetic break-even is $3,000 / $0.75 = 4,000 completed orders per week.
Safe capacity is only 3,500, so the calculated break-even is not feasible under the current footprint.
At capacity, contribution is 3,500 × $0.75 = $2,625, leaving a $375 weekly shortfall before any other costs.
The operator must change contribution, fixed cost, or capacity and include the incremental cost of the change.
If contribution is negative and fixed cost is positive, no positive finite volume breaks even under the unchanged linear assumptions.

### Exercise 5

Control contribution is 2,000 × $2.50 = $5,000.
Treatment contribution is 2,200 × ($2.50 − $0.75) = $3,850.
The incremental result is $3,850 − $5,000 = −$1,150 for the period.
The 200 extra orders create $500 before coupons, while 2,200 coupons cost $1,650.
A different market may have different demand response, courier costs, customer mix, or capacity spillovers.
The result is causal only to the extent that randomization, complete measurement, and the stated no-spillover assumption hold.

### Exercise 6

Continuing yields $5,000 contribution per week that closing would forfeit.
Avoidable weekly staffing and rent total $2,000 + $1,200 = $3,200.
The weekly change from closure before the one-time cost is $3,200 − $5,000 = −$1,800.
Over four weeks, add the $4,000 termination cost: 4 × −$1,800 − $4,000 = −$11,200.
The unchanged $800 corporate allocation is not a cash saving in this comparison.
The historical fit-out cost is sunk because it does not differ between the two future alternatives.
The result would change if actual notice delayed savings or if the site could be redeployed.

### Exercise 7 rubric

The modeled change is +$0.70 − $0.10 − $0.15 = +$0.45 per order.
Projected contribution is $0.05 + $0.45 = $0.50 per order.
A strong answer names dispatch operations as an accountable policy owner and requires confirmation of route feasibility.
It specifies a measurable late-tail stop condition relative to the existing policy and a courier-waiting or earnings guardrail.
It also requires food-safety screening and a rollback mechanism.
It does not declare the forecast saving realized merely because the arithmetic is positive.

### Exercise 8 rubric

A strong design defines every eligible customer or geographic cluster before treatment assignment and reports outcomes for the full randomized population.
It chooses a cluster or time design that acknowledges shared couriers, and measures cross-zone waiting and other spillovers.
It includes all treated coupons, applies the supplier cap at campaign level, and reconciles earned funding against settlement.
It allows at least the stated two-week refund lag and specifies how later adjustments revise results.
It records contribution per eligible unit, total contribution, order count, service reliability, courier time and earnings, and safety incidents.
It names finance, growth, operations, and marketplace owners, along with explicit thresholds and a rollback authority.
If orders rise but total contribution falls, the offer may be buying unprofitable or cannibalized volume; longer-term benefits remain hypotheses until measured.

## 14. Interview questions

1. **What is the difference between GMV and contribution?**
   A strong answer defines both, subtracts merchandise and fulfillment costs, and states the denominator.
   A weak answer treats the full basket as cash available to cover fixed costs.

2. **Why can a $1 increase in basket add less than $1 to profit?**
   A strong answer applies the incremental merchandise margin and accounts for additional handling and payment costs.
   A weak answer assumes every additional dollar is profit.

3. **How would you analyze a $2 coupon funded partly by a supplier?**
   A strong answer shows gross discount, earned funding, caps, and a no-coupon counterfactual.
   A weak answer subtracts the net cost after already subtracting the gross coupon.

4. **What belongs in courier cost per order?**
   A strong answer includes relevant payout, guarantees, incentives, the denominator, and route-sharing treatment.
   A weak answer uses only active-trip pay and assumes idle coverage is free.

5. **Why are platform courier cost and courier net earnings different?**
   A strong answer distinguishes platform payments, tips, working time, and worker-borne expenses.
   A weak answer assumes a low platform payout proves acceptable worker earnings.

6. **When can a loss-making store be worth keeping open?**
   A strong answer compares foregone contribution with costs actually avoided over the decision horizon.
   A weak answer treats every allocated corporate charge as an immediate cash saving.

7. **How would you calculate break-even volume?**
   A strong answer divides fixed cost by positive unit contribution, then checks capacity and step costs.
   A weak answer gives a volume despite negative contribution or ignores operating limits.

8. **How can more orders make a coupon campaign less profitable?**
   A strong answer counts discounts on all treated orders and compares total contribution with a counterfactual.
   A weak answer counts every redemption as incremental.

9. **How would you avoid double-counting route pay?**
   A strong answer aggregates route payouts once and reconciles to the payment ledger before allocating.
   A weak answer joins each route payment to every order and sums the repeated amount.

10. **What evidence would justify batching?**
    A strong answer compares feasible alternatives using total cost, late-tail service, safety, and courier conditions.
    A weak answer relies only on shorter simulated distance.

11. **How does the analysis differ for a marketplace?**
    A strong answer starts from the contracts and appropriate operator earnings, and separates management metrics from financial reporting.
    A weak answer assumes all merchant sales are the platform's recognized revenue.

12. **How do you handle a late supplier reimbursement?**
    A strong answer separates earned funding, receivable, settlement, cap, and cash timing.
    A weak answer records an unsupported nominal amount as both revenue and cash received.

## 15. Key terms

Each link returns to the definition in [Section 4](#4-definitions-and-vocabulary) unless another section is named.

- [Order](#4-definitions-and-vocabulary): customer request with a stable identifier.
- [Completed order](#4-definitions-and-vocabulary): unit meeting the stated completion rule.
- [Order cohort](#4-definitions-and-vocabulary): orders grouped by an explicit time event.
- [GMV](#4-definitions-and-vocabulary): defined merchandise transaction value, not automatically revenue.
- [Basket](#4-definitions-and-vocabulary): merchandise amount under a specified price and discount basis.
- [AOV](#4-definitions-and-vocabulary): weighted merchandise value per eligible order.
- [Merchandise cost](#4-definitions-and-vocabulary): cost attributable to goods sold.
- [Gross profit](#4-definitions-and-vocabulary): merchandise sales less goods cost.
- [Gross margin](#4-definitions-and-vocabulary): gross profit as a share of merchandise sales.
- [Markup](#4-definitions-and-vocabulary): gross profit relative to merchandise cost.
- [Delivery fee](#4-definitions-and-vocabulary): customer charge for delivery.
- [Service fee](#4-definitions-and-vocabulary): separately defined service charge.
- [Platform-funded discount](#4-definitions-and-vocabulary): discount borne by the operator.
- [Supplier reimbursement](#4-definitions-and-vocabulary): earned third-party funding for an offer.
- [Subsidy](#4-definitions-and-vocabulary): benefit reducing another party's effective price or cost.
- [Courier payout](#4-definitions-and-vocabulary): amount payable to the courier.
- [Tip](#4-definitions-and-vocabulary): customer gratuity with jurisdiction-specific treatment.
- [Guaranteed pay](#4-definitions-and-vocabulary): minimum or committed courier compensation.
- [Delivery cost](#4-definitions-and-vocabulary): defined cost of the delivery service.
- [Picking cost](#4-definitions-and-vocabulary): attributable cost of collecting items.
- [Refund](#4-definitions-and-vocabulary): customer repayment or credit requiring reversal reconciliation.
- [Waste](#4-definitions-and-vocabulary): inventory cost lost or impaired.
- [Contribution](#4-definitions-and-vocabulary): defined inflows less specified avoidable costs.
- [Contribution waterfall](#5-the-mechanics-step-by-step): bridge from merchandise economics to contribution.
- [Fixed cost](#4-definitions-and-vocabulary): cost unchanged within a specified range and horizon.
- [Avoidable cost](#4-definitions-and-vocabulary): cost removed in the relevant alternative.
- [Sunk cost](#4-definitions-and-vocabulary): past cost unaffected by the future decision.
- [Step cost](#4-definitions-and-vocabulary): cost changing at a capacity threshold.
- [Allocation](#4-definitions-and-vocabulary): assignment of pooled cost by a chosen rule.
- [Utilization](#4-definitions-and-vocabulary): busy time divided by defined available time.
- [Batch](#4-definitions-and-vocabulary): route serving multiple orders.
- [Cannibalization](#4-definitions-and-vocabulary): activity that would have occurred without the intervention.
- [Incrementality](#4-definitions-and-vocabulary): difference from the counterfactual.
- [Break-even](#4-definitions-and-vocabulary): point where specified costs and contribution are equal.
- [Counterfactual](#4-definitions-and-vocabulary): outcome under the alternative without the intervention.
- [Cohort maturity](#4-definitions-and-vocabulary): elapsed time for delayed financial events to arrive.
- [Opportunity cost](#4-definitions-and-vocabulary): value of the best feasible alternative resource use.

## 16. Sources

Sources below are public primary sources. Access and rules should be checked at the time of a real decision.
Company filings describe the reporting company; they do not validate the fictional numbers in this lesson.

1. IFRS Foundation, [IFRS 15 Revenue from Contracts with Customers](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-15-revenue-from-contracts-with-customers/).
   Relevant to contract-based revenue recognition and principal-versus-agent analysis where IFRS applies.
2. IFRS Foundation, [IAS 2 Inventories](https://www.ifrs.org/issued-standards/list-of-standards/ias-2-inventories/).
   Relevant to inventory measurement where IFRS applies; management avoidability still depends on the decision.
3. New York City Department of Consumer and Worker Protection, [Delivery Workers](https://www.nyc.gov/site/dca/workers/workersrights/food-delivery-worker-laws.page).
   Official local information on delivery-worker protections and minimum pay; check scope and current rules.
4. DoorDash, [Investor Relations: Financials and Filings](https://ir.doordash.com/financials/quarterly-results/default.aspx).
   Company-reported business and financial context; review the relevant period and metric definitions.
5. US Securities and Exchange Commission, [DoorDash company filings](https://www.sec.gov/edgar/browse/?CIK=1792789&owner=exclude).
   Primary filings for financial statements and company-specific disclosures.
6. Instacart, [Investor Relations](https://investors.instacart.com/).
   Company-reported marketplace and financial context; use the current filing for definitions.
7. US Securities and Exchange Commission, [Maplebear Inc. company filings](https://www.sec.gov/edgar/browse/?CIK=1579091&owner=exclude).
   Primary filings for Instacart's financial statements and risk disclosures.
8. US Food and Drug Administration, [FDA Food Code](https://www.fda.gov/food/fda-food-code/food-code-2022).
   A model code for food safety, not automatically binding law in every jurisdiction; check local adoption.
9. Federal Trade Commission, [Protecting Consumer Privacy and Security](https://www.ftc.gov/business-guidance/privacy-security).
   US business guidance on privacy and security; applicability depends on facts and jurisdiction.

## 17. Links

- [Module 22 index](./_index.md).
- [Promise tiers and dark-store design](./01_promise-tiers-dark-stores.md).
- [Replenishment, shelf life, and waste](./04_replenishment-waste.md).
- [Courier supply and demand](./05_courier-supply-demand.md).
- [Batching and routing](./07_batching-routing.md).
- [Module 22 survey and Lab 22.5: Store P&L](../../../course/22_quick_commerce_on_demand_delivery.md#lab-225--store-pl).
- [Module 22 survey and Lab 22.3: Courier dispatch simulation](../../../course/22_quick_commerce_on_demand_delivery.md#lab-223--courier-dispatch-simulation).
