# Unit Economics per Order

> Module 21 — E-commerce Supply Chain Analytics  
> Topic 09 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic financial arithmetic and order, fulfillment, and returns concepts  
> Version 2026.5

## 1. Why this topic exists

A delivery promise can win an order and still destroy value.
A fulfillment manager can improve a warehouse cost per parcel while increasing the number of parcels per order.
A finance team can report a positive gross margin while return freight and payment fees exceed it.
An acquisition campaign can attract customers who buy once, return most items, and never repay the acquisition cost.
These are different decisions, but each requires a consistent view of economics at the order grain.

An order is a useful economic unit because checkout creates a customer promise and a set of downstream obligations.
Its revenue can be discounted, partially refunded, or collected in several currencies.
Its items can leave multiple nodes in separate parcels.
Some items can return months later and have different recovery values.
Some payments fail or are disputed.
The order's apparent profitability therefore changes as events arrive.

The commercial owner needs to know whether a promotion produces incremental contribution, not merely revenue.
The network owner needs to know whether faster delivery earns enough incremental demand to cover incremental cost.
The returns owner needs to separate avoidable return costs from the value of legitimate customer protections.
Finance needs a reproducible bridge from order events to management reporting.
None should use a single average without understanding the cohort, maturity, and cost boundary behind it.

This topic builds a contribution bridge after merchandise, fulfillment, delivery, returns, and payment costs.
It then separates allocated reporting cost from avoidable or incremental decision cost.
The distinction prevents two opposite errors: approving an attractive-looking order that consumes a scarce resource without paying for it, and rejecting a profitable incremental order because it was charged an unavoidable fixed allocation.

The decision is not always to maximize immediate contribution per order.
A constrained operation may maximize contribution per unit of bottleneck capacity.
A policy may be required by law or a customer contract despite a negative short-term contribution.
A strategically justified investment may have negative initial contribution and positive later cash flows.
Make such trade-offs explicit rather than hiding them in a cost allocation.

The numbers in the worked example are synthetic, in US dollars, and are not industry benchmarks.
They illustrate a management calculation, not an audited financial statement or legal advice.

## 2. Learning outcomes

After completing this topic, you can:

1. Define a mature order cohort, denominator, currency, observation window, and decision boundary.
   Demonstrate the definition in Exercise 1.
2. Compute net revenue and contribution after merchandise, fulfillment, delivery, return, and payment costs.
   Reproduce the bridge in Exercise 2.
3. Reconcile orders, parcels, payment events, returns, and accounting totals without fanout or double counting.
   Diagnose the join in Exercise 3.
4. Classify a cost as incremental, avoidable, fixed, or allocated for a specified decision and horizon.
   Defend the classifications in Exercise 4.
5. Build an incremental business case and a break-even condition with an explicit counterfactual.
   Calculate and critique the proposal in Exercise 5.
6. Quantify a capacity constraint and compare opportunities on contribution per bottleneck unit.
   Apply the constraint in Exercise 6.
7. Design an experiment and sensitivity analysis that acknowledge returns maturity and uncertainty.
   Prepare the design in Exercise 7.
8. Communicate a recommendation with decision ownership, assumptions, risks, and stop criteria.
   Produce the memo in Exercise 8.

A correct answer states both the number and what it measures.
A sophisticated model with an undefined denominator is not a correct answer.

## 3. The core mental model

```text
             Decision and counterfactual
                        |
                        v
Eligible checkout orders -----> fixed cohort and observation window
       |                                      |
       +--> sales, discounts, shipping --------+--> net revenue
       +--> refunds and concessions -----------+
       +--> merchandise cost less recovery ----------+
       +--> pick, pack, outbound, reverse ------------+--> contribution
       +--> payment, dispute, customer service -------+
       |                                               |
       +--> capacity and shared fixed cost ------------+--> separate views
                                                       |
                                      incremental option minus baseline
                                                       |
                                  sensitivity, ownership, decision, review
```

The bridge begins with a cohort, not with a general ledger total.
Choose the unit before averaging.
This lesson uses an accepted customer order, even if it later cancels or returns, unless the analysis explicitly says otherwise.
Another team may use shipped orders; that is permissible only if it labels the denominator and reconciles it to accepted orders.

The bridge follows event families through a lifecycle.
A single order can have many item lines, packages, refunds, and fees.
Aggregate each family to order before joining it to the order spine.
The spine retains orders with no shipment or no return.
A missing cost is not automatically zero; a truly absent event and an unavailable feed are different states.

Net revenue is what the business retains from the customer under the management convention.
Contribution subtracts costs that the convention includes.
The name alone does not establish its boundary.
Always list included and excluded costs and the attribution window.

Contribution is a descriptive measure.
It answers what the selected cohort earned under the selected cost treatment.
Incremental value is a causal comparison.
It answers what changes if a specific decision is adopted rather than the best available baseline.
The same historical cost may be useful for the first question and irrelevant for the second.

An allocation spreads shared cost using a rule.
For example, rent divided by order count helps explain how a reporting entity absorbs rent.
It does not mean that one extra order causes an additional equal amount of rent.
However, a decision that requires an additional building can make rent incremental at the capacity threshold.
Relevant cost depends on the option and the horizon, not on the account label alone.

The model has three important boundaries.
The order boundary defines which events attach to which order.
The time boundary defines how long to wait or how to estimate unresolved events.
The decision boundary defines what changes between the candidate and the baseline.
Confusing any two produces persuasive but misleading averages.

Keep a parallel operational view of units and capacity.
A positive dollar contribution may be a poor use of a constrained packing minute.
Likewise, a high contribution per minute cannot justify a decision that violates a safety, service, or contractual requirement.
Decision criteria combine economics with explicit constraints.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Accepted order | Customer order admitted to the defined cohort; count in orders | Replacing it with parcel count without changing the denominator |
| Order line | Product and quantity within an order; lines or units | Treating line count as customer-order count |
| Parcel | A physical shipment unit; parcels | Assuming every order has one parcel |
| Cohort | Set selected by a common event and date; orders over a stated period | Mixing sale-date and return-date populations |
| Observation window | Time after cohort entry during which events are observed; days | Assuming a recent cohort is fully returned |
| Maturity | Degree to which expected downstream events have arrived; percent or days | Calling an order final immediately after dispatch |
| Gross merchandise sales | Original merchandise selling amount before discounts and returns; USD | Including shipping charges without labeling them |
| Discount | Reduction in the original selling amount; USD | Subtracting a discount already netted from recorded sales |
| Customer shipping revenue | Delivery amount charged to the customer; USD | Treating shipping revenue as carrier cost |
| Refund | Amount returned or credited to customer under the chosen net-revenue convention; USD | Also recording the same refund as a separate return expense |
| Net revenue | Sales less discounts and refunds plus retained charges; USD | Equating it with cash received in the same month |
| Merchandise cost | Cost of goods sold under the specified management basis; USD | Subtracting purchase orders that have not been sold |
| Recovery | Documented economic value restored when returned goods are resalable or otherwise recovered; USD | Recognizing the same resale value again as immediate revenue |
| Fulfillment cost | Pick, pack, materials and other defined node activity; USD | Hiding it inside outbound carrier charges and subtracting twice |
| Outbound delivery | Freight and related surcharges for delivery to customers; USD | Dividing total freight by parcels and calling it per order |
| Reverse logistics | Return transport, receipt, inspection, and disposition activity; USD | Including merchandise write-down again when already reflected in recovery |
| Payment cost | Processing, transaction, fraud, and dispute costs in scope; USD | Assuming the percentage fee is refunded when the customer is refunded |
| Contribution | Revenue less specified variable or traceable costs; USD | Treating an unlabeled contribution number as operating profit |
| Contribution per order | Total contribution divided by the stated order count; USD/order | Averaging segment averages without order weights |
| Contribution margin | Contribution divided by a stated revenue base; percent | Confusing USD/order with percent of revenue |
| Fixed cost | Cost unchanged within a specified activity range and horizon; USD/period | Assuming every fixed cost is permanent in every decision |
| Variable cost | Cost that changes with an activity driver over a specified range; USD/unit | Assuming all historical average expense is variable |
| Step cost | Cost that changes when capacity or a contract threshold is crossed; USD/step | Applying a smooth per-order charge across an unmodeled threshold |
| Allocation | Assignment of shared cost by a chosen driver; USD | Calling the allocation the causal cost of an order |
| Incremental cost | Difference in total cost between option and baseline; USD | Including sunk or unchanged expense |
| Avoidable cost | Cost eliminated if the specified action is not taken; USD | Classifying a committed lease as avoidable without an exit mechanism |
| Sunk cost | Past cost that does not change with the current decision; USD | Including an already-paid feasibility study in future option value |
| Opportunity cost | Value of the best displaced use of a constrained resource; USD | Assuming unused capacity has the same value as scarce capacity |
| Counterfactual | Outcome under the best specified alternative; USD and units | Comparing a pilot's outcome only with its own prior period |
| Cannibalization | Candidate-induced movement from an existing sale to a new offer; orders or USD | Counting moved orders as newly created demand |
| Contribution per constraint unit | Incremental contribution divided by incremental constrained capacity; USD/minute, for example | Using an order average when scarce capacity differs by order |
| Chargeback | Payment dispute that can reverse funds and impose fees; USD/event | Counting both a chargeback reversal and the same refund as separate losses |
| Reserve | Estimate for future obligations or losses at a stated date; USD | Treating an estimate as an observed completed event |
| Expected value | Probability-weighted outcome under stated scenarios; USD | Calling a single forecast a guaranteed result |
| Forecast | Estimated future outcome with uncertainty | Treating a forecast as a management commitment |
| Plan | Chosen operating actions and resources | Treating an approved plan as observed performance |
| Target | Desired performance level | Using an aspirational target as the baseline forecast |
| Commitment | Approved obligation to deliver a specified action or outcome | Presenting an unapproved forecast as a commitment |

## 5. The mechanics, step by step

### Step 1 — Frame the decision

Write the action as a choice between mutually exclusive alternatives.
For example: offer two-day delivery to eligible customers at no extra charge, or keep the current promise.
State the decision date, geography, channels, customer population, and period of consequences.
Identify the accountable decision-maker before producing a model.
For a delivery promise, the commercial lead may own the offer, operations may own feasibility, and finance may own the economic sign-off.
Name actual individuals or roles in a real decision record.

Separate an economic recommendation from mandatory constraints.
Service contracts, consumer rights, safety, privacy, and system capacity can limit the available options.
Do not infer that a financially negative order should be rejected when the business is already committed to serve it.
Instead ask which future, lawful action can improve economics.

Specify the baseline as what would plausibly happen without the change.
A baseline can include expected growth, existing promotions, labor constraints, and contract changes already committed.
It is not necessarily last year's result or a zero-activity world.
Record which assumptions are shared by both options so they cancel in the difference.

Output: a one-sentence decision, alternatives, owner, horizon, and constraints.
Branch: if the objective is financial reporting rather than a forward decision, build the historical bridge first and use accounting policy for external reporting.

### Step 2 — Define the unit and cohort

Choose a customer-facing order identifier that remains stable across payment, shipment, and return events.
Select orders by accepted date in a specific time zone.
Record whether test orders, employee purchases, fraud rejections, canceled orders, and replacement orders are included.
Retain cancellation outcomes if the question is economics per accepted order.
If you instead exclude cancellations, label the denominator and separately show their cost.

A return exchanged for a replacement is not necessarily a new customer acquisition.
Link the replacement to its originating order or define how standalone replacement orders are treated.
A marketplace can create multiple merchant orders from one cart.
State whether the unit is a customer checkout, merchant order, or fulfillment order.
Do not combine them as if they were identical.

Order count is a denominator, not an additive financial event.
One customer may place multiple orders.
One order may have multiple parcels.
The customer-level decision may require a longer horizon than the order-level calculation.
For instance, a low first-order contribution may be acceptable only if independently supported repeat-order contribution exceeds the acquisition and retention costs.
Do not assume future purchases without evidence.

Output: a cohort specification with inclusion, exclusions, date, time zone, identifier, and expected count.
Branch: if reliable identity linkage is unavailable, report order economics without asserting customer lifetime value.

### Step 3 — Set the lifecycle and observation date

An order may be authorized, captured, shipped, partially refunded, returned, resold, or disputed on different dates.
Choose a cohort date and a data cut-off separately.
Cohort accounting follows an order through its events, while period accounting groups events by posting date.
Both can be correct for different questions but they are not directly interchangeable.

For mature-cohort analysis, choose a window that captures the operational return pattern and late invoices.
Compare cumulative return or refund curves for older cohorts at the same age.
If a current cohort is immature, report observed contribution and an explicitly modeled remaining-event reserve.
The reserve may be based on historical curves stratified by category, geography, and policy, subject to sufficient data.
A change in policy or product mix can invalidate those curves.

Keep the reserve out of observed actuals.
For example, an observed refund of USD 9,000 and a future-refund estimate of USD 3,000 are not the same evidence type.
Show both and explain the uncertainty of the estimate.
Never use a future event in a historical decision's training data as though it was known at that time.

Output: order date, as-of date, maturity rule, actuals, and remaining-event estimate.
Branch: if no defensible reserve is available, label the result incomplete rather than calling it final.

### Step 4 — Build net revenue

Start with original merchandise sales for the cohort.
Subtract promotional and other customer discounts once.
Add customer shipping charges and other included customer fees.
Subtract refunds, concessions, and credits that reduce the retained customer amount.
Decide whether the analysis is exclusive of sales tax and value-added tax (VAT).
For this lesson, amounts exclude tax collected on behalf of authorities.
Legal and accounting treatment depends on jurisdiction and transaction.

A simple management bridge is:

```text
net revenue = gross merchandise sales - discounts
              + customer shipping charges - refunds and concessions
```

This is not a universal revenue-recognition policy.
If a data source reports sales already net of discounts, do not subtract discounts again.
If it reports shipping refunds separately, include them within the refund total once.
If gift-card purchase and redemption both appear as revenue, identify and remove the duplication according to the applicable accounting policy.

Promotional funding from a supplier needs its own treatment.
Determine whether it is earned, collectible, and incremental to the promotion.
Avoid recognizing unapproved vendor funding as certain revenue.
Keep funding visible as a separate line rather than silently reducing merchandise cost.

Output: net revenue and a reconciliation from source transactions to that value.
Branch: if customer charges are collected by a marketplace rather than the merchant, establish principal-versus-agent accounting and contract flows with finance before interpreting reported revenue.

### Step 5 — Calculate merchandise cost and return recovery

Attach original cost to sold units using the organization's approved inventory-cost convention.
For management analysis the cost might be standard cost, while the financial ledger uses a different accepted convention.
Explain differences rather than joining a current purchase price to old sales and calling it historical cost.

A returned item may be resalable, repairable, liquidated, or discarded.
Use an evidence-based recovery value net of necessary incremental disposition costs if those costs are not separately counted.
Distinguish inventory value restored from a future resale receipt.
For this lesson, original cost less inventory recovery is the net merchandise cost in the bridge.
We do not also add the recovery value to revenue.

Consider a USD 30 original unit cost and a USD 18 recoverable inventory value.
Net merchandise cost is USD 12, before separately recorded reverse logistics.
If an inspection cost of USD 2 is deducted in calculating the USD 18 recovery, do not subtract that USD 2 again.
If the USD 18 is a gross recoverable amount, include inspection separately.
The definition must travel with the number.

Refund value and merchandise recovery are separate.
A USD 50 refund is a reduction in customer revenue.
A USD 18 inventory recovery offsets cost of goods sold in this management bridge.
They are not alternative descriptions of the same event.
Partial returns require item-level quantity and cost mapping.
A customer can return one of three units while retaining the others.

Output: original merchandise cost, recovered cost, and net merchandise cost at the order grain.
Branch: if disposition is unresolved, use an explicit estimate and later true it up.

### Step 6 — Attach fulfillment and outbound costs

Identify which node performed each task.
Measure pick, pack, packaging, and other included warehouse activity.
Where available, use actual labor minutes or a causally calibrated activity rate.
Avoid an arbitrary flat charge if one order's 10 lines require substantially more work than another's single line.
An activity-based estimate can be useful even when the actual payroll bill is fixed in the short run.
Its decision relevance still depends on whether the activity changes staffing or displaces productive work.

Attach carrier invoices, surcharges, insurance, and delivery-failure costs to parcels.
Roll parcels up to the order.
If one order ships in two packages, include both packages.
Reconcile estimated charges to later carrier invoices and credits.
A contract minimum, volume rebate, or peak surcharge can make invoice-level cost nonlinear.
Forecast the marginal contract effect of the option, not merely the past average tariff.

Record fulfillment cost by cost family before joining to the order.
A pick-and-pack fee charged by a third-party logistics provider may already include packaging materials.
Do not add an internal packaging estimate to that same included service.
Similarly, know whether outbound delivery includes returns labels; if it does, remove that component from a separate reverse-freight line.

Output: order-level fulfillment and outbound cost, with actual, estimated, and shared components marked.
Branch: if a rate card is used rather than invoices, track rate-card assumptions and reconciliation drift.

### Step 7 — Attach reverse, payment, and service costs

Reverse logistics can include a label, carrier pickup, receiving, inspection, repackaging, disposal, and restocking.
Include each component once.
A no-return order can still incur expected return cost in a forward scenario, but should not receive a fabricated observed return event.
Separate actual event cost from a cohort-level expected reserve.

Payment cost may include an ad valorem processing fee, a per-transaction fee, gateway costs, failed-authorization charges, chargebacks, and fraud losses.
Refund rules differ by processor and contract.
Some transaction fees may not be returned after a refund.
Use contract terms and actual settlements; do not assume every fee is proportional to net revenue.

A dispute and an ordinary refund can both relate to the same sale.
Track their transaction references so that the same principal is not subtracted twice.
Dispute fees and fraud losses are distinct cost components when they are actually incurred.
A separate customer-service activity cost can be included if its scope is defined and its source reconciled.
Do not duplicate service labor already included in a fulfillment or overhead pool.

Output: reverse, payment, fraud, and service costs by order and by event type.
Branch: if fraud or disputes are rare, use a sufficiently long sample or scenario range instead of assuming a short sample rate is stable.

### Step 8 — Construct and reconcile contribution

Under the lesson's convention:

```text
contribution = net revenue
             - net merchandise cost
             - fulfillment cost
             - outbound delivery cost
             - reverse logistics cost
             - payment and dispute cost
```

Divide the sum of contribution by the count of eligible orders.
This is a weighted average across orders.
For a segment comparison, calculate both segment totals and segment counts before computing the overall average.
The overall contribution margin is contribution divided by net revenue, if net revenue is the selected denominator.
Label it accordingly.

Reconcile at least five totals: eligible order count, net revenue, merchandise cost, shipping expense, and refunds.
Explain timing, scope, accrual, foreign-exchange, and allocation differences from the ledger.
A management bridge may legitimately differ from booked operating profit.
The difference must be quantified and owned.

Look at the distribution, not just the mean.
Show negative-contribution order share, percentiles, and total value at risk by category or channel.
Use cohort-weighted means for portfolio totals and do not add percentiles across segments.
A small number of high-value orders can dominate the average.
Investigate extreme values before treating them as business reality; erroneous joins and delayed refunds are common causes.

Output: a versioned cohort bridge, distribution, and ledger reconciliation.
Branch: if reconciliation differences are material and unexplained, do not use a precise order-level result as a decision commitment.

### Step 9 — Separate reporting allocations from decision costs

A management profit-and-loss view may allocate warehouse rent, technology, leadership, and other shared overhead.
Allocation can support full-cost reporting, pricing policy, or accountability when used consistently.
Its driver should reflect the reporting purpose, and the total allocated amount must reconcile to the pool.
An allocation does not itself establish that the business can avoid the cost.

For every material cost ask: would total cash or economic cost differ between the option and baseline within the horizon?
Ask which contract can be amended, which workers can be redeployed, and whether a capacity threshold is crossed.
If the lease remains unchanged under either option, its allocated per-order amount is not incremental to the decision.
If the new option requires an additional facility lease, the new lease is incremental even though rent is usually described as fixed.

An employee's salary may be unchanged while their time has an opportunity cost.
If idle capacity is genuinely available, that time may have no near-term cash cost.
If a scarce worker is moved away from an alternative that would earn contribution, include the displaced value as opportunity cost.
Do not add a separate opportunity cost for the same displacement already represented by lost sales in the forecast.

For a shutdown decision, an overhead allocation may be avoidable only if the cost can actually be terminated within the horizon.
For a multi-year capacity decision, depreciation may be useful for accounting but investment cash flows and terminal value belong in a discounted cash-flow analysis.
Avoid counting depreciation and the same capital purchase as separate economic cash outflows.

Output: two clearly labeled views: allocated reporting contribution and incremental decision cash flow or economic value.
Branch: where the relevant cost behavior is uncertain, model low, base, and high cases and identify the evidence needed.

### Step 10 — Build the incremental business case

Calculate the outcome under the baseline and the candidate with consistent scope.
Subtract baseline from candidate.
Include incremental orders, contribution on those orders, changed contribution on retained orders, cannibalization, implementation expense, and step costs.
The most robust construction models revenue and each cost under both alternatives rather than applying a single average to every changed order.
Use an average contribution only if the marginal orders have comparable mix and cost behavior.

For an upgrade applied to every order, a useful simplified form is:

```text
incremental value = incremental orders * contribution per incremental order
                  - candidate-order count * added cost per candidate order
                  - incremental fixed and implementation costs
                  - value of displaced business
```

This form assumes contribution per incremental order is measured before the separately subtracted added cost.
It also assumes the upgrade has no other effect on revenue, returns, or service costs.
If those change, model them explicitly.
Candidate-order count includes existing and incremental orders where both receive the upgraded service.

Do not count an order moved from another delivery option as wholly incremental to the business.
Instead compare its candidate economics with the baseline economics it displaced.
Likewise, a price discount on all existing orders can outweigh the contribution on new orders.
A before-and-after lift is not causal evidence if seasonality or a concurrent campaign also changed demand.

Output: baseline and candidate cash-flow bridge, explicit causality assumption, and net difference.
Branch: for a multi-period investment, discount future incremental cash flows at an approved rate and disclose the convention.

### Step 11 — Model bottlenecks and uncertainty

If capacity is unconstrained, total incremental contribution may be the appropriate short-run objective.
If a constrained resource is fully utilized, compare contribution per incremental unit of that resource.
For a packing bottleneck, calculate USD per constrained packing minute, not USD per order alone.
Subtract any relevant opportunity cost of displaced work or optimize the feasible portfolio directly, but do not do both for the same displacement.

Include service, fairness, contract, and safety constraints in the feasible set.
An apparently higher-yield order may be ineligible for the same slot or node.
Averages do not model sequencing, indivisible shifts, product compatibility, or uncertainty in realized processing time.
Escalate to an operational scheduling model if those details drive the decision.

Vary the assumptions that can change the recommendation: incremental demand, conversion, parcel count, return rate, refund timing, cost per parcel, and extra staffing.
Document correlated changes.
A promotion that changes category mix may simultaneously change both order value and return rate.
A one-at-a-time sensitivity is useful for diagnosis but does not establish a probability of loss.
Scenario weights require a defendable basis.

Calculate a break-even threshold with the same cost structure as the base case.
Check that it remains within the range where staffing and carrier prices do not step up.
If break-even crosses a threshold, solve again in the correct cost segment.
Record the point at which a finding changes the decision.

Output: constrained capacity calculation, sensitivity table, threshold, and uncertainty owners.
Branch: if a feasible low-risk experiment can resolve the uncertainty before an irreversible investment, prefer an appropriately designed test.

### Step 12 — Decide, implement, and learn

Present the recommendation, alternatives, expected incremental value, downside, and required approvals on one page.
Give operations a capacity plan and finance an event and reconciliation plan.
Give data owners a measurement definition and publication deadline.
A delivery-promise experiment requires customer-experience and legal review where relevant.
A data owner cannot approve a commercial risk merely because the SQL runs.

Before rollout, document the forecast as an estimate, the plan as intended actions, the target as desired performance, and the commitment as an approved obligation.
For example, forecast 1,040 candidate orders, plan upgraded shipping for eligible orders, target a positive incremental contribution, and commit only to a bounded pilot if authorized.
The forecast is not a commitment to obtain 1,040 orders.

Set decision gates: pilot launch, interim safety review, mature-cohort readout, and scale or stop.
An interim conversion gain is not a final economic result while returns remain immature.
Define who can stop the test for fraud, safety, service failure, or loss exceeding an approved exposure.
Store model version, data snapshot, assumptions, and owner in the decision record.
After enough time has elapsed, compare realized versus forecast drivers and update the next decision's evidence.

Output: signed decision record and a dated review plan.
Branch: if no owner accepts the downside or the data needed to monitor it, do not present the option as approved.

## 6. Worked example

### Case and decision

A synthetic retailer is considering an upgraded two-day delivery promise.
Finance first measures the economics of a mature baseline order cohort.
The commercial lead then evaluates a candidate offer over an equivalent future period.
The cohort contains 1,000 accepted orders from one US region and one sales channel.
All amounts below are USD, exclusive of tax, for that cohort.
The data cut-off is sufficiently late for the assumed returns and invoices to be mature.
This is a simplifying assumption, not proof that any real cohort is mature.

The 1,000-order baseline is the best available forecast for the comparable future period.
It is not a sales target or a commitment.
The proposal assumes 40 additional orders under the upgraded promise, for a candidate forecast of 1,040 orders.
That uplift is an assumption to test, not an observed causal result.
The plan is to upgrade all candidate orders.
There is no approved rollout commitment in this example.

### Cohort revenue

| Component | Amount (USD) | Treatment |
|---|---:|---|
| Original merchandise sales | 100,000 | Add |
| Discounts | 8,000 | Subtract once |
| Customer shipping charges | 4,000 | Add |
| Refunds and concessions | 12,000 | Subtract once |
| Net revenue | 84,000 | 100,000 - 8,000 + 4,000 - 12,000 |

The USD 12,000 refund figure includes any shipping refunds in this example.
No additional refund line should be subtracted from the bridge.
The amount is associated with the order cohort rather than the month in which the refund was paid.
The retained revenue is USD 84.00 per accepted order.

### Cohort costs

| Component | Amount (USD) | Treatment |
|---|---:|---|
| Original merchandise cost | 48,000 | Subtract |
| Recoverable returned inventory | 3,600 | Offset original merchandise cost |
| Net merchandise cost | 44,400 | 48,000 - 3,600 |
| Fulfillment | 7,000 | Subtract |
| Outbound delivery | 8,500 | Subtract |
| Reverse logistics | 1,800 | Subtract |
| Payment and dispute | 2,900 | Subtract |
| Total included costs | 64,600 | 44,400 + 7,000 + 8,500 + 1,800 + 2,900 |

The USD 3,600 recovery is inventory value restored, not a separate sale.
The USD 1,800 reverse cost covers transport and processing and is not already deducted from the recovery amount.
The USD 2,900 payment and dispute line reflects contract fees and disputes under the example's convention.
It is not calculated by applying a uniform rate to the net revenue figure.

### Contribution bridge

```text
Net revenue                             USD 84,000
Less net merchandise cost              -USD 44,400
Less fulfillment                        -USD 7,000
Less outbound delivery                  -USD 8,500
Less reverse logistics                  -USD 1,800
Less payment and dispute                -USD 2,900
--------------------------------------------------
Contribution                            USD 19,400
Accepted orders                              1,000
Contribution per accepted order         USD 19.40
Contribution / net revenue                  23.10%
```

The percentage is USD 19,400 divided by USD 84,000, or approximately 23.10%.
It is not USD 19.40 percent, and it is not net profit.
The bridge excludes acquisition, headquarters, technology, taxes on income, financing, and shared fixed rent.
Those omissions are deliberate and must be reconsidered for a decision that changes any of them.

### Allocated reporting view

Suppose a finance report allocates USD 6,000 of an existing monthly warehouse lease to the cohort.
A reporting view after that allocation is USD 13,400, or USD 13.40 per order.
The USD 6,000 is a useful disclosure if the purpose is to show the cohort's share of a cost pool.
It is not automatically a cost caused by this delivery decision.
If the lease is unchanged under both candidate and baseline, the incremental lease cost is USD 0 over this decision horizon.
If the upgrade requires a new facility, the new avoidable lease cost must instead enter the option's incremental case.

### Candidate economics

Assume each of the 40 incremental orders would have USD 19.40 of contribution before the upgrade's extra delivery charge.
This assumes marginal orders have the same pre-upgrade economics as the baseline average.
Assume the upgraded service adds USD 1.20 per candidate order, including the 40 new orders.
Assume a one-time implementation expense of USD 200 within the horizon.
Assume no cannibalization, no change in returns or other costs, and no binding capacity constraint.
These are case assumptions, not established facts.

```text
Contribution on incremental orders: 40 * USD 19.40 =  USD 776
Incremental shipping:            1,040 * USD 1.20 = -USD 1,248
Implementation expense:                             -USD 200
-------------------------------------------------------------
Incremental candidate value:                         -USD 672
```

On these assumptions the candidate is USD 672 worse than baseline.
The baseline's USD 19,400 contribution is unchanged except for modeled differences.
The candidate contribution after the specified changes is USD 18,728.
The correct recommendation is not to roll out on this forecast alone.
A bounded experiment could still be worthwhile if its expected learning value and exposure are separately approved.

### Break-even lift

Let x be incremental orders.
The candidate serves 1,000 + x orders, all receiving the USD 1.20 upgrade.
Before any step changes, break-even satisfies:

```text
19.40x - 1.20(1,000 + x) - 200 = 0
18.20x = 1,400
x = 76.923... orders
```

At least 77 additional whole orders are needed in this simplified period.
That is a 7.7% lift over 1,000 baseline orders, rounded to a whole order.
At 76 additional orders the difference is USD -16.80.
At 77 additional orders the difference is USD 1.40.
This threshold assumes no other changes and no capacity step; it is not a universal rule.

### Sensitivity and ownership

| Incremental orders | Candidate orders | Incremental value (USD) | Interpretation |
|---:|---:|---:|---|
| 0 | 1,000 | -1,400 | Upgrade cost and implementation without demand lift |
| 40 | 1,040 | -672 | Base assumption, below break-even |
| 77 | 1,077 | 1.40 | Approximately break-even |
| 100 | 1,100 | 420 | Positive only under the stated assumptions |

The commercial lead owns the demand-uplift hypothesis and offer decision.
The fulfillment lead validates carrier price, service capacity, and contract thresholds.
Finance validates the bridge, cost treatment, and decision horizon.
The data team owns randomization, cohort linkage, and the mature readout.
The customer-experience owner reviews delivery failures and return experience.
These roles are illustrative; an actual organization must assign named accountable owners.

### Additional scenario: a carrier capacity step

The preceding break-even assumes the existing carrier arrangement can handle every candidate order.
Suppose instead that the carrier charges a USD 500 capacity reservation whenever candidate volume exceeds 1,050 orders.
The reservation is incremental because the baseline forecast is 1,000 orders and does not trigger it.
Assume the same USD 19.40 contribution before the upgrade and the same USD 1.20 upgrade cost per candidate order.
The USD 200 implementation expense also remains unchanged.

For up to 50 incremental orders, the original formula still applies.
At 50 incremental orders, candidate volume is exactly 1,050 and no reservation is charged.
Incremental value is 50 × USD 19.40 - 1,050 × USD 1.20 - USD 200 = USD -490.
At 51 incremental orders, the reservation is triggered.
Incremental value becomes 51 × USD 19.40 - 1,051 × USD 1.20 - USD 200 - USD 500 = USD -971.80.
The value therefore falls at the threshold even though one more order has been gained.

Above the threshold, the new break-even equation is:

```text
19.40x - 1.20(1,000 + x) - 200 - 500 = 0
18.20x = 1,900
x = 104.3956... incremental orders
```

The smallest whole-order break-even is 105 incremental orders, provided the reservation covers that volume.
At 104 extra orders the incremental result is USD -7.20.
At 105 extra orders the incremental result is USD 11.00.
Both lie above the 50-order threshold, so the equation uses the correct cost segment.
The earlier 77-order threshold is invalid under this new contract assumption: at 77 extra orders the reservation makes the result USD -498.60.

The carrier owner should confirm the trigger, price, and capacity ceiling in the actual contract.
The finance owner should test whether the baseline itself could trigger the charge in higher-demand scenarios.
If both alternatives trigger the same reservation, the reservation cancels in their incremental comparison.
The commercial owner should not treat this hypothetical threshold as evidence of actual demand uplift.
This scenario changes the economics, not the evidence supporting the demand forecast.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It creates the synthetic cohort bridge and the candidate comparison above.
Values are aggregate USD for one cohort; counts are accepted orders.
The program uses `Decimal` to avoid binary floating-point surprises in money arithmetic.
It does not query a source system, infer missing records, or determine whether the assumptions are causal.

```python
from decimal import Decimal, ROUND_CEILING

D = Decimal

cohort = {
    "orders": 1000,
    "gross_sales": D("100000"),
    "discounts": D("8000"),
    "shipping_revenue": D("4000"),
    "refunds": D("12000"),
    "merchandise_cost": D("48000"),
    "inventory_recovery": D("3600"),
    "fulfillment": D("7000"),
    "outbound": D("8500"),
    "reverse": D("1800"),
    "payment": D("2900"),
}


def contribution_bridge(data):
    """Return the explicitly defined management bridge for one cohort."""
    orders = data["orders"]
    if orders <= 0:
        raise ValueError("orders must be positive")
    net_revenue = (
        data["gross_sales"]
        - data["discounts"]
        + data["shipping_revenue"]
        - data["refunds"]
    )
    net_merchandise = data["merchandise_cost"] - data["inventory_recovery"]
    costs = (
        net_merchandise
        + data["fulfillment"]
        + data["outbound"]
        + data["reverse"]
        + data["payment"]
    )
    contribution = net_revenue - costs
    return {
        "net_revenue": net_revenue,
        "net_merchandise": net_merchandise,
        "costs": costs,
        "contribution": contribution,
        "per_order": contribution / D(orders),
    }


def incremental_value(base_orders, extra_orders, before_upgrade, extra_cost, fixed):
    """All candidate orders receive the upgrade; contribution excludes it."""
    candidate_orders = base_orders + extra_orders
    return (
        D(extra_orders) * before_upgrade
        - D(candidate_orders) * extra_cost
        - fixed
    )


result = contribution_bridge(cohort)
base = cohort["orders"]
extra_cost = D("1.20")
implementation = D("200")
pre_upgrade = result["per_order"]

# For this constant-cost segment, solve x*(contribution - upgrade)
# = baseline orders * upgrade + implementation.
net_per_extra_order = pre_upgrade - extra_cost
if net_per_extra_order <= 0:
    raise ValueError("no finite positive break-even in this cost segment")
threshold = (D(base) * extra_cost + implementation) / net_per_extra_order
whole_orders = int(threshold.to_integral_value(rounding=ROUND_CEILING))

print(f"Net revenue: USD {result['net_revenue']:.2f}")
print(f"Net merchandise cost: USD {result['net_merchandise']:.2f}")
print(f"Included costs: USD {result['costs']:.2f}")
print(f"Contribution: USD {result['contribution']:.2f}")
print(f"Contribution per order: USD {result['per_order']:.2f}")
for extra in (0, 40, 77, 100):
    value = incremental_value(base, extra, pre_upgrade, extra_cost, implementation)
    print(f"Extra orders {extra}: incremental value USD {value:.2f}")
print(f"Whole-order break-even: {whole_orders}")
```

Expected output:

```text
Net revenue: USD 84000.00
Net merchandise cost: USD 44400.00
Included costs: USD 64600.00
Contribution: USD 19400.00
Contribution per order: USD 19.40
Extra orders 0: incremental value USD -1400.00
Extra orders 40: incremental value USD -672.00
Extra orders 77: incremental value USD 1.40
Extra orders 100: incremental value USD 420.00
Whole-order break-even: 77
```

For production, aggregate event tables separately to order before joining them.
Assert uniqueness of the order spine and of each aggregated order-level table.
Reconcile order counts and event totals before and after the joins.
Use a left join from the cohort to preserve orders without returns.
Missing feeds should fail a completeness check, not silently become zero cost.
A zero on a complete feed means no qualifying event, not that an unknown amount is known to be zero.

The function deliberately does not validate each business input.
Real inputs require contractual rules for refunds, inventory valuation, currency conversion, and event status.
In particular, a recorded return authorization is not necessarily a completed return.
A transaction can be voided, corrected, or posted more than once.
Preserve source event identifiers and event versions so that corrections are not added twice.

The break-even formula is valid only within the stated linear segment.
It does not model a carrier rebate, a new shift, or the causal uncertainty in demand lift.
When those effects matter, calculate each scenario with the correct threshold and use a designed experiment or other credible evidence for uplift.

## 8. Data, standards, and regulatory context

### Data contract

The order spine needs a stable order identifier, accepted timestamp, status, channel, market, and currency.
The item feed needs order-line identifier, product, quantity, price, discounts, tax treatment, and cost basis.
The shipment feed needs parcel identifier, order mapping, node, carrier, service, charge, and shipment dates.
The return feed needs return identifier, original line mapping, quantity, reason, received date, disposition, and recovery value.
The payment feed needs authorization, capture, refund, dispute, fee, settlement, and reversal identifiers.
A general ledger feed supports period reconciliation, not an automatic substitute for event linkage.

A data contract states the grain, owner, delivery schedule, permitted status transitions, and correction behavior for each feed.
It also states the expected arrival delay and the treatment of orphan records.
A unique event key is required for deduplication, but do not discard a legitimate corrected event merely because its business identifier repeats.
Use event version or effective timestamp according to the source contract.

Date dimensions must distinguish order date, ship date, return receipt date, payment settlement date, and accounting posting date.
Use a consistent local time zone for cohort membership and retain original timestamps for audit.
A cohort report and a calendar-month ledger can differ because their event dates differ.
Bridge those timing differences explicitly.

For cross-border comparisons, choose the functional reporting currency and a documented conversion date and rate source.
Do not sum USD and EUR as if they were the same unit.
Separate transactional exchange-rate effects from changes in underlying unit economics.
Taxes collected on behalf of authorities are excluded from this lesson's management revenue convention.
Ask finance and tax specialists how actual transactions should be recognized.

### Join and completeness controls

A common error joins two return rows and three parcel rows directly to one order.
The result contains six joined rows, multiplying both refund and parcel costs.
Aggregate each event family by order first, then join one row per family to the spine.
Check totals in each source before and after aggregation.
A distinct count of orders alone cannot prove that dollar sums survived the join correctly.

Test for duplicate order identifiers, negative or implausible quantities, orphan returns, unmatched carrier invoices, and missing cost rates.
A replacement shipment may be operationally linked to an original order even when its system creates a separate identifier.
Define its attribution rule before analysis.
A negative cost can represent a valid credit; inspect event type rather than automatically dropping it.

Monitor the proportion of orders with known merchandise cost, known delivery cost, and a completed refund feed.
Publish the data freshness and completeness alongside the economic result.
When a cost feed is delayed, provide an estimate separately from actuals and reconcile it after invoices arrive.
If missingness is concentrated in a particular carrier or category, a simple overall fill rate can hide bias.

### Accounting and rights

The IFRS Foundation's IFRS 15 materials describe revenue recognition for contracts with customers; official IFRS 15 implementation guidance addresses sales with a right of return.
For entities applying that standard, the accounting involves a refund liability and an asset for the right to recover products, subject to the applicable requirements.
This lesson's simplified management bridge is not an instruction to replace an entity's accounting policy.
The applicable framework, contract facts, and reporting period determine the booked treatment.
US Generally Accepted Accounting Principles may apply instead; consult the relevant authoritative guidance and finance team.

Consumer cancellation and return rights vary by jurisdiction, product, and sale type.
The EU Consumer Rights Directive contains distance-selling rules including a withdrawal right subject to conditions and exceptions.
Do not generalize its rules to every country or every product.
The retailer must confirm the applicable current law and its obligations with qualified counsel.
An economic model measures the cost of compliant options; it does not authorize removal of a mandatory right.

Payment-card data requires its own security controls.
The PCI Security Standards Council publishes the Payment Card Industry Data Security Standard (PCI DSS).
Do not copy primary account numbers or sensitive authentication data into an analytics extract merely to compute a fee.
Use tokenized identifiers and least-privilege access consistent with the organization's security program.
Applicability and current version must be verified with the security and compliance owners.

Personal data such as addresses and customer identifiers should be minimized and retained only for a defined purpose and period.
Where the EU General Data Protection Regulation applies, the controller must assess the relevant legal obligations, including data-minimization and storage-limitation principles.
A delivery zone or pseudonymous key may suffice for unit economics without storing a full street address.
Pseudonymization does not automatically make a dataset anonymous.
Engage privacy counsel and security owners for actual deployment.

### Management versus statutory reporting

Contribution in this lesson is an internal decision measure.
It is not necessarily gross profit, operating income, cash flow, or an IFRS-defined performance measure.
Do not relabel it as an audited metric without reconciling definitions and obtaining appropriate approval.
Payment fees, returns liabilities, inventory recovery, and overhead can be classified differently in statutory statements.
Maintain a bridge rather than forcing operational data to mimic a financial statement through arbitrary adjustments.

If an estimate is updated, preserve both the original as-of version and the later true-up.
This permits a fair assessment of the forecast with the information available at decision time.
A revised historical result is not evidence that the earlier decision-maker had the revised information.
Document material changes in cost allocation, product mix, and return policy before comparing periods.

## 9. Industry practice and case evidence

### Public company disclosure is not order-level evidence

Amazon's annual reports disclose net sales and operating expense categories including fulfillment and shipping-related information.
They provide useful context about the economic significance of fulfillment and delivery.
They do not disclose this lesson's synthetic contribution per order or establish that the example's costs are representative of Amazon.
Public segments, accounting definitions, third-party services, and cohort boundaries differ from a retailer's operational order records.
A ratio constructed from a public filing must state both numerator and denominator and cannot be assumed to represent an individual shipment.

Shopify's annual reports discuss merchant solutions and payment-related business lines.
Its reporting describes platform economics rather than a representative merchant's fully loaded order contribution.
A merchant's processor contract, shipping agreement, customer mix, and returns determine its own economics.
Do not take an aggregate platform metric as a merchant transaction fee without contract evidence.

Both examples illustrate a limitation: public reporting is useful for identifying material expense categories, but generally insufficient for a precise causal decision on a specific customer cohort.
That conclusion is an inference from the difference between public reporting grain and the order-event grain required above.
It is not a claim that either company operates the synthetic case in this lesson.

### Use external case evidence carefully

A company statement about speed, free shipping, or returns can explain the offered customer proposition.
It does not by itself establish that the proposition increased profit.
A press article reporting order growth also cannot establish incremental value without the counterfactual, cost, and return information.
A vendor case study may have selection bias and may omit failed deployments.
Record who produced the evidence, its time period, its population, and which outcomes were independently verified.

When translating evidence into a business case, distinguish three statements.
First, a source says an offer exists or an aggregate expense was reported.
Second, an analyst infers a possible mechanism, such as higher conversion from faster shipping.
Third, an experiment or credible causal design estimates the effect for the decision population.
Only the third directly informs the causal uplift parameter, subject to its design and uncertainty.

### Experiment in practice

Randomize eligible customers or an appropriate operational unit before the promise is displayed.
Measure all eligible units, not only those who place an order, to avoid conditioning on a post-treatment outcome.
Calculate total contribution per eligible customer or exposure as well as per accepted order.
The former captures conversion and order economics together; the latter diagnoses changes in order mix.
Guard against customers seeing both offers, geographic spillovers, and carrier capacity constraints shared across arms.
Cluster randomization may be appropriate when individual assignment cannot isolate the intervention, but it reduces the number of independent units.

Pre-specify the primary outcome, assignment unit, observation horizon, analysis method, and stopping rules.
Track service and customer-protection guardrails separately from contribution.
Estimate uncertainty at the randomization unit, not by pretending that multiple parcels from one order are independent customers.
Ensure return and dispute windows mature before the final economic readout.
If a decision is required earlier, show observed results plus a separately labeled reserve and uncertainty range.

Promotions often change who orders and what they buy.
A naive comparison of contribution per order can decline even while total contribution per eligible customer rises, or vice versa.
Both denominators answer useful but different questions.
Report order counts, net revenue, cost components, and exposure counts for each arm.
Do not optimize a per-order metric in isolation when the offer changes the number of orders.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Fanout | Revenue grows after joining parcels | Many-to-many rows multiply amounts | Inflated contribution | Aggregate each family to order and reconcile sums |
| Immature returns | Recent cohorts appear unusually profitable | Refunds have not yet arrived | Overstated contribution | Compare same-age curves and show a separate reserve |
| Discount twice | Net sales already match discounted checkout value | Discounts are deducted again | Understated revenue | Document gross versus net source fields |
| Refund twice | Refund and chargeback share a transaction reference | Same principal counted as two losses | Understated contribution | Match reversals to original events |
| Recovery twice | Recovery enters revenue and reduces cost | One inventory asset counted in two places | Overstated contribution | Choose one bridge treatment and reconcile |
| Shipping omission | Parcel total exceeds order-linked carrier charges | Split shipments not fully mapped | Overstated contribution | Reconcile parcel invoices and orphan shipments |
| Fee assumption | Refunds reduce a modeled percentage fee but settlement does not | Contract retains transaction fees | Overstated contribution | Use actual settlement and contract rules |
| Allocation as marginal | Recommendation changes solely with rent driver | Unchanged lease attributed to incremental orders | Rejection of valuable work | Compare total costs under each option |
| Ignored step cost | Forecast exceeds shift or carrier threshold | Linear average hides capacity step | Overstated business case | Model capacity and contract breakpoints |
| Average mix transfer | New offer attracts a different product mix | Marginal orders differ from historical average | Biased incremental contribution | Estimate treatment economics by eligible population |
| Cannibalization | New-service orders rise while standard orders fall | Existing customers switch offers | Inflated uplift | Measure net orders and total contribution across arms |
| Denominator drift | Dashboard alternates accepted and shipped orders | Cancellation costs disappear from one view | False improvement | Publish denominator and reconciliation |
| Period mismatch | Refunds spike in one month but sales cohort differs | Period costs matched to another cohort's revenue | Misleading per-order result | Follow order cohort and bridge to ledger |
| Capacity blindness | High-value orders consume more bottleneck minutes | Per-order ranking ignores scarce resource | Lost portfolio contribution | Optimize contribution per constrained unit with feasibility checks |
| Missing feed treated as zero | One carrier has no invoice costs | Incomplete data mistaken for no charge | Artificially high margin | Completeness checks and explicit unknown status |
| Target as forecast | Business case uses an aspirational uplift | Desired outcome replaces evidence | Understated downside | Separate forecast, target, plan, and commitment |
| Currency mixing | Multi-market totals lack currency metadata | Values added without conversion | Invalid comparisons | Document rates, dates, and functional currency |
| Noncompliant cost cutting | Proposal assumes mandatory returns disappear | Required rights excluded from feasible set | Legal and customer harm | Review policy and jurisdiction before optimization |

Treat a red flag as an investigation trigger, not proof of a particular error.
A valid credit can yield a negative cost and a legitimate high-value order can be an outlier.
The analyst should trace the source event before deleting either.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Cohort identifiers are stable | Events must attach to the correct order | Replacement orders or platform migrations | Audit crosswalks and unmatched events |
| Cohort is mature | Final contribution needs late returns and invoices | Long return windows or late carrier bills | Compare cumulative curves by cohort age |
| Refunds are complete | Net revenue must reflect customer concessions | Missing marketplace or manual credits | Reconcile to settlement and ledger totals |
| Recovery reflects realizable inventory value | Avoids treating worthless returns as full recovery | Damage, obsolescence, or disposition delay | Compare estimated recovery with disposition outcomes |
| Costs are counted once | Bridge requires mutually exclusive categories | Third-party fee bundles packaging and freight | Map invoice components to cost taxonomy |
| Marginal mix matches average | Simplifies candidate calculation | Promotion attracts different products or customers | Compare treatment-arm composition and contribution |
| Upgrade cost is linear | Allows a simple break-even formula | Contract tiers, peak surcharge, or new shift | Inspect contracts and model threshold cases |
| No displacement | Assumes incremental orders use available capacity | Constrained pick or delivery slots | Measure utilization and displaced demand |
| Lift is causal | Candidate-minus-baseline requires a valid counterfactual | Seasonality, concurrent promotion, or selection | Randomize or use a justified quasi-experiment |
| Costs are known at decision time | Needed for a prospective case | Late invoices or uncertain labor rates | Preserve as-of snapshots and scenario ranges |
| USD is the common currency | Allows addition and comparison | Multi-currency transactions | Check currency fields and conversion policy |
| Lease remains unchanged | Excludes allocated rent from incremental case | New facility or early termination | Obtain signed capacity and lease assumptions |
| Return policy is stable | Historical maturity curve remains relevant | A policy or product change shifts return behavior | Stratify and monitor return curves |
| No omitted strategic effect | Keeps horizon limited to this case | Repeat purchases, brand effects, supplier responses | Model separately when evidence supports them |

The ledger is not a claim that these assumptions hold in a real deployment.
An assumption can be acceptable for a bounded pilot and unacceptable for a long-term investment.
Assign an owner and a review date to each material uncertainty.

## 12. Exercises

### Exercise 1 — Define the cohort

A marketplace checkout creates one customer cart, two merchant orders, and three parcels.
One merchant order is canceled before shipment; the other later has a partial return.
You are asked for contribution per accepted merchant order for orders accepted in March.
The data extract arrives in April, before all returns are mature.

Deliverable: a short cohort and observation-window definition.
Acceptance criteria: identify the two-order denominator, preserve the canceled order, distinguish three parcels from two orders, and describe observed versus estimated late returns.
Time: 15 minutes.

### Exercise 2 — Calculate the bridge

A cohort has 200 accepted orders.
Gross merchandise sales are USD 24,000, discounts USD 1,000, customer shipping charges USD 600, and refunds USD 2,000.
Original merchandise cost is USD 11,000 and recovered inventory is USD 500.
Fulfillment is USD 1,400, outbound delivery USD 1,700, reverse logistics USD 300, and payment costs USD 600.
Recovery excludes reverse logistics, and all values are for the same mature cohort.

Deliverable: net revenue, total included costs, contribution, and contribution per accepted order.
Acceptance criteria: show each formula, subtract the refund once, and offset merchandise cost by the recovery once.
Time: 15 minutes.

### Exercise 3 — Diagnose a join

An order has two parcels costing USD 4 and USD 6 and two refund events of USD 8 and USD 12.
A query joins the raw parcel and refund tables directly on order ID.
The analyst sums parcel cost and refunds over the result.

Deliverable: the naive and correct totals, the cause, and a repair.
Acceptance criteria: show why the join creates four rows and reconcile both event-family totals after separate aggregation.
Time: 15 minutes.

### Exercise 4 — Classify relevant costs

A one-month promotion uses spare warehouse capacity.
The existing lease is USD 10,000 per month regardless of the promotion.
Extra packing supplies cost USD 0.50 per promotional order.
A nonrefundable USD 2,000 study was paid before the decision.
If promotional volume exceeds 500 orders, an extra shift costs USD 3,000.

Deliverable: classify each item for the promote-versus-do-not-promote decision.
Acceptance criteria: distinguish allocation from incremental cash flow, identify the sunk cost and conditional step cost, and state the horizon.
Time: 15 minutes.

### Exercise 5 — Evaluate the upgrade

Use the worked example's 1,000 baseline orders, USD 19.40 pre-upgrade contribution per incremental order, USD 1.20 added cost per candidate order, and USD 200 implementation expense.
Assume 60 incremental orders and no other effects.

Deliverable: incremental value and a recommendation relative to the simplified break-even.
Acceptance criteria: charge the upgrade to all 1,060 candidate orders and explain why the 60 new orders are not the only orders incurring it.
Time: 15 minutes.

### Exercise 6 — Allocate a constrained resource

A packing line has 600 available bottleneck minutes.
Product A yields incremental contribution of USD 12 per order and uses 3 minutes per order.
Product B yields incremental contribution of USD 15 per order and uses 5 minutes per order.
There is unlimited demand for each product and no other constraint; orders are indivisible.

Deliverable: choose the feasible mix that maximizes contribution and compute the value.
Acceptance criteria: calculate USD per bottleneck minute for each product and use at most 600 minutes.
Time: 15 minutes.

### Exercise 7 — Design a causal test

The retailer wants to estimate whether its faster delivery offer improves economics.
It can randomize eligible visitors, but returns can arrive 60 days after purchase and parcels can share carrier capacity.

Deliverable: a one-page experiment design.
Acceptance criteria: specify unit and timing of randomization, primary metric and denominator, guardrails, maturity rule, potential interference, uncertainty, and a stop/scale decision owner.
Time: 25 minutes.

### Exercise 8 — Write a decision memo

A proposal claims that an offer adds USD 10 contribution per order after applying an arbitrary USD 8 rent allocation.
The proposal uses a 15% demand target as if it were an experimentally measured forecast.
There is no estimate of return uplift or carrier capacity.

Deliverable: a decision memo with a conditional recommendation, evidence gaps, accountable roles, and review gate.
Acceptance criteria: distinguish allocation from avoidability; forecast from target; and require a returns and capacity assessment without inventing outcomes.
Time: 25 minutes.

## 13. Self-check answers

### Exercise 1

The denominator is two accepted merchant orders in March, including the canceled order.
The cart is a separate customer-checkout unit, and three parcels are a separate physical shipment count.
Attach any cancellation cost to the canceled order and allocate shared cart-level events only by a documented rule.
Use the April cut-off for observed actuals.
Show an estimate of remaining returns separately if supported by older comparable cohorts; otherwise identify the result as immature.

### Exercise 2

Net revenue is USD 24,000 - USD 1,000 + USD 600 - USD 2,000 = USD 21,600.
Net merchandise cost is USD 11,000 - USD 500 = USD 10,500.
Included costs are USD 10,500 + USD 1,400 + USD 1,700 + USD 300 + USD 600 = USD 14,500.
Contribution is USD 21,600 - USD 14,500 = USD 7,100.
Contribution per accepted order is USD 7,100 / 200 = USD 35.50 per order.
A refund is a reduction of revenue; inventory recovery offsets cost under this convention.

### Exercise 3

The raw join produces 2 x 2 = 4 rows.
The parcel costs are repeated for each refund, yielding a naive USD 20 parcel total instead of USD 10.
The refund amounts are repeated for each parcel, yielding a naive USD 40 refund total instead of USD 20.
Aggregate parcels to USD 10 per order and refunds to USD 20 per order before joining them to the unique order spine.
Check both source totals after the join.

### Exercise 4

The existing USD 10,000 lease is unchanged over the one-month decision horizon and is not incremental, although a reporting view may allocate it.
Supplies are incremental at USD 0.50 per promotional order.
The already-paid USD 2,000 study is sunk and does not change with the choice.
The USD 3,000 extra shift is incremental only if the candidate crosses the stated threshold and the baseline does not.
If the baseline already requires that shift, compare the actual change in total cost instead of automatically charging it again.

### Exercise 5

The candidate has 1,060 orders, of which 60 are incremental.
Incremental value is 60 x USD 19.40 - 1,060 x USD 1.20 - USD 200.
That is USD 1,164 - USD 1,272 - USD 200 = USD -308.
The proposed 60 extra orders are below the simplified break-even of 77.
On the stated assumptions the offer should not be rolled out as a positive-value decision.
All 1,060 candidate orders receive the upgraded service, including baseline orders that would have arrived anyway.

### Exercise 6

A yields USD 12 / 3 = USD 4 per bottleneck minute.
B yields USD 15 / 5 = USD 3 per bottleneck minute.
With unlimited demand, no other constraints, and exactly divisible capacity for A's three-minute orders, choose 200 A orders.
They use 600 minutes and contribute USD 2,400.
No mix containing B can exceed USD 4 per available minute under these assumptions.
If demand for A were capped or service requirements applied, recalculate the feasible mix.

### Exercise 7 rubric

A strong answer randomizes eligible visitors before their offer is shown and analyzes them by assigned arm.
It uses total mature contribution per eligible visitor as the primary outcome and per-order contribution as a diagnostic.
It identifies a 60-day or longer appropriate follow-up after the final purchase, with late disputes and invoice delay considered separately.
It explains how shared carrier capacity can create interference and proposes a feasible unit or monitoring plan.
It specifies uncertainty at the assignment unit, customer-service and safety guardrails, an exposure limit, and named commercial, operations, finance, and analysis owners.
A weak answer compares only buyers after treatment, stops at conversion, or treats an immature return result as final.

### Exercise 8 rubric

A strong memo withholds a scale decision until a defensible incremental comparison exists.
It explains that an allocated USD 8 is not necessarily avoidable and that actual capacity-driven costs may still be material.
It labels 15% as a target, not a measured forecast.
It requests a returns-maturity estimate, carrier and labor threshold analysis, and a counterfactual or experiment.
It identifies the commercial decision owner, finance approver, operations capacity owner, and data measurement owner.
It defines a time-bounded review gate and leaves statutory and customer-rights constraints intact.

## 14. Interview questions

1. **What is contribution per order?**
   A strong answer defines the order denominator, cohort, time window, and cost boundary.
   A weak answer gives a universal formula without explaining what is included.
2. **How do you handle returns after the reporting date?**
   A strong answer separates observed events from a supported reserve and checks maturity.
   A weak answer assumes every order with no observed return will never return.
3. **Why is a warehouse rent allocation not always a marginal cost?**
   A strong answer compares total lease obligations under each alternative and considers thresholds.
   A weak answer says rent is never relevant because it is fixed.
4. **How can a many-to-many join distort contribution?**
   A strong answer shows event multiplication and aggregates each family before joining.
   A weak answer relies only on a distinct order count while ignoring duplicated dollar amounts.
5. **How can faster shipping reduce unit contribution but increase total value?**
   A strong answer compares incremental total contribution per eligible population, including conversion and all upgraded orders.
   A weak answer considers only the orders that bought under the new offer.
6. **What makes an order incremental?**
   A strong answer specifies the counterfactual and handles cannibalization and displaced demand.
   A weak answer calls every order observed after launch incremental.
7. **How should scarce capacity change a decision?**
   A strong answer compares feasible contribution per bottleneck unit and accounts for displaced alternatives.
   A weak answer ranks solely by contribution per order.
8. **How do you treat payment fees on refunded orders?**
   A strong answer checks processor terms and settlement events and avoids refund duplication.
   A weak answer assumes all processing fees reverse with the customer refund.
9. **When does a contribution bridge differ from financial statements?**
   A strong answer explains cost boundary, timing, accounting policies, and reconciliation.
   A weak answer presents management contribution as audited net income.
10. **What would you measure in a delivery-promise test?**
    A strong answer identifies assignment, exposure denominator, mature contribution, uncertainty, and service guardrails.
    A weak answer reports only conversion among purchasers.

## 15. Key terms

- [Accepted order](#4-definitions-and-vocabulary) — order included at the specified acceptance event.
- [Allocation](#step-9--separate-reporting-allocations-from-decision-costs) — assigned share of a common cost pool.
- [Avoidable cost](#step-9--separate-reporting-allocations-from-decision-costs) — cost removed by the defined alternative.
- [Baseline](#step-1--frame-the-decision) — outcome expected without the proposed change.
- [Bottleneck](#step-11--model-bottlenecks-and-uncertainty) — resource limiting feasible throughput.
- [Break-even](#break-even-lift) — parameter value where incremental value equals zero.
- [Cannibalization](#step-10--build-the-incremental-business-case) — movement from a baseline sale into the new offer.
- [Chargeback](#step-7--attach-reverse-payment-and-service-costs) — payment dispute that can reverse funds and incur fees.
- [Cohort](#step-2--define-the-unit-and-cohort) — population sharing an entry rule and date window.
- [Commitment](#step-12--decide-implement-and-learn) — approved obligation, not merely a forecast.
- [Contribution](#step-8--construct-and-reconcile-contribution) — revenue minus the explicitly included costs.
- [Counterfactual](#step-10--build-the-incremental-business-case) — outcome under the relevant alternative.
- [Decision horizon](#step-1--frame-the-decision) — period over which consequences are compared.
- [Discount](#step-4--build-net-revenue) — reduction in the original customer price.
- [Forecast](#step-12--decide-implement-and-learn) — estimate of future outcomes with uncertainty.
- [Fulfillment cost](#step-6--attach-fulfillment-and-outbound-costs) — defined cost of node activities.
- [Incremental cost](#step-9--separate-reporting-allocations-from-decision-costs) — difference in cost between options.
- [Maturity](#step-3--set-the-lifecycle-and-observation-date) — completeness of downstream event observation.
- [Merchandise recovery](#step-5--calculate-merchandise-cost-and-return-recovery) — inventory value restored after a return.
- [Net revenue](#step-4--build-net-revenue) — retained customer revenue under a stated convention.
- [Opportunity cost](#step-9--separate-reporting-allocations-from-decision-costs) — value of the best displaced alternative.
- [Order spine](#data-contract) — unique set of eligible order identifiers.
- [Parcel](#step-6--attach-fulfillment-and-outbound-costs) — physical shipment unit, potentially multiple per order.
- [Payment cost](#step-7--attach-reverse-payment-and-service-costs) — included processing and dispute expense.
- [Plan](#step-12--decide-implement-and-learn) — chosen operating actions and resources.
- [Refund](#step-4--build-net-revenue) — returned or credited customer amount.
- [Reserve](#step-3--set-the-lifecycle-and-observation-date) — separately labeled estimate of future obligations.
- [Reverse logistics](#step-7--attach-reverse-payment-and-service-costs) — return transport and processing activities.
- [Step cost](#step-11--model-bottlenecks-and-uncertainty) — expense triggered at a capacity or contract threshold.
- [Sunk cost](#step-9--separate-reporting-allocations-from-decision-costs) — past cost unchanged by the current decision.
- [Target](#step-12--decide-implement-and-learn) — desired performance, not necessarily the forecast.

## 16. Sources

The following sources support the accounting, compliance, and public-disclosure context.
They do not validate the synthetic case numbers or determine the right choice for a particular business.
Check the version and applicability of a standard or law at the decision date.

1. IFRS Foundation, [IFRS 15 Revenue from Contracts with Customers](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-15-revenue-from-contracts-with-customers/).
   Official standard overview and supporting materials; consult the applicable standard and implementation guidance for right-of-return accounting.
2. European Parliament and Council, [Directive 2011/83/EU on consumer rights](https://eur-lex.europa.eu/eli/dir/2011/83/oj).
   Primary EU legal text; national implementation and exceptions require legal review.
3. European Parliament and Council, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
   Primary EU legal text; see Article 5 for processing principles.
4. PCI Security Standards Council, [PCI DSS document library](https://www.pcisecuritystandards.org/document_library/).
   Official source for the standard and current documents; applicability depends on the environment.
5. Amazon, [Annual reports, proxies and shareholder letters](https://ir.aboutamazon.com/annual-reports-proxies-and-shareholder-letters/default.aspx).
   Company filings provide aggregate financial context, not the lesson's order-level data.
6. Shopify, [Financial reports](https://investors.shopify.com/financials/default.aspx).
   Company disclosures provide platform-level context, not a universal merchant unit-cost benchmark.
7. NIST/SEMATECH, [e-Handbook of Statistical Methods: experimental design](https://www.itl.nist.gov/div898/handbook/pri/pri.htm).
   General experimental-design reference; adapt the design to the assignment unit and operational interference.

## 17. Links

- [Module 21 index](_index.md).
- [Fulfillment models](01_ecommerce-fulfillment-models.md).
- [Shipping cost and carrier selection](06_carrier-selection.md).
- [Returns](07_returns.md).
- [Peak planning](08_peak-planning.md).
- [Module 21 survey and its lab section](../../../course/21_ecommerce_supply_chain_analytics.md).

Sibling topic links refer to the module's indexed sequence; consult the index for their current availability.
