# Channel P&L and Shared-Capacity Economics

> Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise  
> Topic 10 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 13 and 21; Topics 03, 04, 05, 07, and 08 are useful  
> Version 2026.5

## 1. Why this topic exists

A retailer can report positive digital sales growth while the enterprise earns less money.
An online order may use a store's inventory, a store associate's time, a parcel carrier, a digital payment processor, and a store returns desk.
Each function sees a different part of the transaction.
If each function records a complete version of the economics, the same revenue, inventory, or cost can appear more than once.
If none records a shared resource, management can mistake borrowed capacity for free capacity.

Channel profit and loss (P&L) reporting helps explain where economic activity occurred.
It does not, by itself, establish which activity caused a cost or what would happen if a channel were expanded, reduced, or removed.
That distinction is essential for decisions about store pickup, ship-from-store, return-to-store, and inventory placement.
A fully allocated P&L can be useful for accountability and reconciliation while still being a poor basis for accepting one more order.

Consider a store that handles online pickups during a quiet afternoon.
A payroll allocation may charge the digital channel an average hourly wage for those picks.
If the associate would otherwise have been paid and no customer service is displaced, the immediate payroll increment can be zero.
The same pick at a peak hour may require overtime, displace a store sale, or delay a promise.
Neither zero nor the average wage is universally correct: the answer depends on the counterfactual and the relevant capacity constraint.

A finance director needs a reconciled enterprise view.
A channel leader needs a comparable trend and an explanation of controllable costs.
A store manager needs recognition for work performed without incentives to reject enterprise-profitable orders.
A network planner needs the cost of a feasible next action, not an arbitrary share of last month's rent.
A customer experience owner needs service outcomes and returns alongside margin.
The analyst's job is to make those views consistent, not force them into one number.

This lesson builds a channel P&L, reconciles it to an enterprise ledger, and then constructs a separate incremental decision.
The examples are illustrative, in U.S. dollars (USD), and do not describe a named retailer.
They exclude income tax and financing unless explicitly stated.
The accounting policies and decision assumptions are part of the result, not footnotes to be hidden.

## 2. Learning outcomes

After this lesson you can:

1. Define a channel, a fulfillment node, and a return location without treating them as interchangeable.
2. Build a net-revenue-to-contribution bridge from transaction and cost records.
3. Reconcile channel results and unallocated costs to an enterprise control total.
4. Select and document an allocation driver, including an explicit treatment of unused capacity.
5. Identify duplicate revenue, inventory cost, recovery, transfer charges, and shared overhead.
6. Compare a reported fully allocated result with the incremental economics of a specific decision.
7. Quantify a capacity bottleneck, an opportunity cost, and a sensitivity threshold.
8. Recommend ownership and incentives that align local behavior with enterprise outcomes.

Exercises 1–8 provide evidence for these outcomes in the same order.
The worked example and code provide a reproducible control calculation before you attempt the exercises.

## 3. The core mental model

```text
Customer order and subsequent events
  | order channel, original sale, discounts, taxes, refunds
  | fulfillment node, picks, packaging, delivery, return and recovery
  v
Canonical event ledger -----> Enterprise financial control total
  |                              ^
  | ownership + cost drivers     | reconcile exactly once
  v                              |
Channel contribution -> allocated shared costs -> reported channel P&L
  |
  +--> separate counterfactual: changed cash flows + scarce-capacity effects
                               -> decision recommendation
```

The canonical ledger describes what happened once.
The attribution layer assigns economic responsibility according to published rules.
The control total checks that all channel values plus residual or unallocated values equal the enterprise result.
The decision layer asks what changes between feasible alternatives.
A channel P&L is an attribution view; a decision model is a comparison of counterfactuals.
Do not add the two views together.

Four dimensions should remain distinct.
**Demand channel** identifies where the sale was placed under the chosen attribution policy.
**Fulfillment node** identifies the store or distribution center (DC) that shipped or prepared it.
**Return location** identifies where the item was accepted.
**Resource owner** identifies the cost center that pays for labor, rent, transport, or systems.
An online order picked and returned in store remains one customer transaction with several operational events.
A store credit can recognize work without creating a second enterprise sale.

Think of costs in layers.
Direct variable costs are linked to identifiable transactions, such as a per-parcel charge.
Shared costs support multiple activities, such as a store's rent or a pooled associate schedule.
An allocation distributes shared costs for reporting; its driver does not prove causality.
Unused capacity can be separately visible rather than forced into the cost of the units that happened to use a facility.
The incremental view includes a cost only if it differs between alternatives over the decision horizon.

A constraint changes the economics.
If a store has spare paid minutes, a pick can consume capacity without increasing cash payroll today.
If the store is full, a pick may force overtime or displace a profitable alternative.
The foregone contribution of the displaced alternative is an opportunity cost, not necessarily a general-ledger expense.
Avoid counting both a displaced sale's contribution and the same lost margin in another penalty.
A shadow price is the value of relaxing a binding capacity constraint by one unit under a specified model and local range; it is not automatically an accounting rate.

The diagram should be redrawable from memory: events, reconciled attribution, decision comparison.
The arrows do not imply that management allocations are statutory accounting entries.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Channel | Classification of demand origin, such as digital or store; orders or USD per period | Treating fulfillment site as channel |
| Fulfillment node | Physical site completing a pick, pack, or handoff; site ID | Calling each node a separate customer sale |
| Return location | Site accepting a returned item; site ID | Assuming the site caused the original demand |
| Gross merchandise sales | Merchandise price before specified reductions; USD per period | Treating sales tax collected for a government as retailer revenue |
| Net revenue | Recognized revenue after relevant discounts and returns under the defined policy; USD per period | Mixing an order-date forecast with recognized revenue |
| Refund | Amount returned to the customer; USD per return | Deducting it once as a refund and again as a separate sales reversal |
| Cost of goods sold (COGS) | Carrying amount of inventory recognized as expense under the accounting policy; USD | Treating a transfer between stores as a new enterprise purchase |
| Gross profit | Net revenue less recognized COGS; USD | Calling it cash contribution after fulfillment expense |
| Contribution | Revenue less a specified set of costs; USD | Using the word without naming included costs |
| Direct cost | Cost traceable to a defined object with acceptable effort; USD | Assuming every direct cost is avoidable in the next hour |
| Shared cost | Cost supporting more than one cost object; USD | Assuming shared means immaterial or unowned |
| Cost object | Item to which a cost is assigned, such as a channel or order | Changing object mid-report without a bridge |
| Allocation pool | Defined shared-cost amount to distribute; USD per period | Allocating a pool that already includes a direct charge |
| Allocation driver | Measurable basis for distributing a pool, such as occupied square feet or minutes | Treating a convenient driver as proof of causation |
| Driver quantity | Usage or capacity amount used by the allocation; minutes, orders, or square feet | Using quantities from incompatible periods |
| Practical capacity | Attainable resource capacity after normal operating allowances; minutes per period | Equating it with theoretical maximum or forecast demand |
| Utilized capacity | Capacity actually consumed under defined measurement rules; minutes per period | Counting paid hours as productive task minutes |
| Idle capacity | Practical capacity not used by the measured activities; minutes per period | Allocating its cost invisibly to active orders |
| Absorption | Assignment of pool cost to products or channels using a rate | Assuming absorbed cost equals incremental cost |
| Residual | Unassigned cost or rounding difference explicitly retained for reconciliation; USD | Silently dropping unmatched cost |
| Incremental cost | Difference in relevant cost between two feasible alternatives; USD | Subtracting allocated fixed rent when rent does not change |
| Avoidable cost | Cost that disappears under a specified alternative and time horizon; USD | Assuming all fixed costs are permanently unavoidable |
| Sunk cost | Past cost unaffected by the alternatives; USD | Ignoring a future cancellation payment because the original commitment is sunk |
| Opportunity cost | Value of the best feasible alternative forgone; USD | Adding revenue rather than lost contribution |
| Shadow price | Marginal improvement in objective from one more unit of a binding resource under a model; USD per minute | Applying one price outside its valid capacity range |
| Transfer price | Internal charge between organizational units; USD | Treating internal revenue and expense as enterprise profit |
| Elimination | Removal of intercompany or internal charges from a consolidated view; USD | Eliminating actual external carrier expense |
| Cost-to-serve | Defined operating costs to serve a customer, order, or channel; USD | Assuming it includes product cost and returns without checking |
| Activity-based costing | Assignment using activity pools and drivers; USD | Assuming more allocation detail guarantees causal accuracy |
| Cohort | Transactions grouped by a common attribute and starting period | Comparing incomplete recent cohorts with mature cohorts |
| Recognition | Recording an amount in the appropriate accounting period under policy | Equating order placement with earned revenue |
| Forecast | Estimate of a future outcome | Calling a forecast a commitment |
| Plan | Chosen intended operating path and resources | Calling the plan a guaranteed outcome |
| Target | Desired metric level | Treating a target as measured performance |
| Commitment | Agreed obligation, such as a delivery promise | Assuming it is achievable solely because it was displayed |
| Cannibalization | Demand or contribution shifted from one option to another | Counting the replacement sale as entirely new demand |
| Recovery | Value obtained from returned inventory through resale or another disposition; USD | Adding gross resale proceeds while omitting carrying value and related costs |
| Normal balance | Expected debit or credit behavior of a ledger account | Reversing a cost sign in one channel but not another |

## 5. The mechanics, step by step

### Step 1 — Define the decision and the reporting view

Write a sentence containing the decision, alternatives, owner, horizon, and objective.
For example: decide whether to accept an additional digital pickup batch tomorrow afternoon, subject to staffing and promised handoff times.
The decision owner is the store operations manager, with finance approving the economic method and customer experience defining service constraints.
The objective is enterprise incremental contribution, not the store's allocated P&L.

Define the reporting object separately: one monthly channel P&L for digital and store demand.
Use a consistent currency, time zone, business calendar, and entity perimeter.
Document whether the report is a management view, a financial close view, or a forecast.
A management allocation may not satisfy financial reporting requirements; the controller determines the statutory treatment.

Input: a decision request and reporting calendar.
Output: a signed decision brief and a P&L perimeter.
Branch: if no feasible alternative exists, stop and identify what operational change is needed before calculating an incremental margin.
Branch: if the decision spans multiple periods, include costs that can change in those periods, not only tomorrow's cash expense.

### Step 2 — Establish dimensions and unique identities

Assign every order a stable enterprise order ID.
Assign each line, shipment, payment, refund, and return its own ID and link it to the order line.
A shipment split across two nodes has one order-line identity and two fulfillment events, not two demand events.
An exchange should link the return and replacement while retaining separate movement events.
Use effective-dated channel and store mappings, because channels, entities, and ownership can change.

Choose one primary channel for net revenue.
A simple rule attributes revenue to the order-entry channel, including refunds to that original channel.
This is a management convention, not a claim that the entry channel alone caused the sale.
Track assistance, pickup, and return location as separate dimensions.
If the business wants shared sales credit, keep it in an incentive ledger and reconcile it to the one enterprise revenue amount.

Input: order, payment, fulfillment, returns, and master data.
Output: a canonical event table with source IDs and effective dates.
Control: duplicate source IDs must be quarantined, not arbitrarily summed.
Control: orphan returns and unmatched credits must remain visible in an exception account.

### Step 3 — Choose period and cohort logic

A calendar-month P&L contains revenue and costs recognized in that month.
A sale made in January and refunded in February can therefore appear in different monthly periods.
An order cohort assigns the return to the sale's original cohort to estimate lifetime economics.
Both views can be useful, but the cohort view is not automatically the general ledger.
Reconcile the difference through timing and estimate bridges rather than calling one incorrect.

Use a maturity window or explicit return estimate when comparing cohorts.
For example, a 30-day-old cohort cannot yet show returns that occur on day 60.
Document expected late returns, exchanges, fraud reversals, and subsequent inventory recoveries.
Do not book a statistical estimate twice if the finance ledger already contains an allowance.
Input: recognized events, event dates, and return-window policy.
Output: separate period and cohort views with a timing bridge.

### Step 4 — Assemble the revenue bridge

Start with merchandise sales at the stated transaction amount.
Subtract discounts and customer refunds according to the reporting definition.
Handle cancellations before fulfillment consistently with the enterprise revenue policy.
Separate sales tax collected on behalf of a taxing authority when it is not retailer revenue under the applicable accounting policy.
Report shipping fees, gift card breakage, marketplace commissions, and loyalty consideration under defined policy rather than assuming identical treatment.

The example below starts at already-recognized net revenue.
Therefore do not subtract its returns or discounts a second time.
If starting from gross sales instead, show each bridge component once and reconcile the result to the net-revenue control account.
For multi-currency operations, use documented exchange rates and show translation effects separately.
Input: sales, discounts, refunds, and relevant accounting adjustments.
Output: channel net revenue and a bridge to the general ledger.

### Step 5 — Establish inventory and return accounting

Link COGS to the recognized sale using the enterprise inventory-cost convention.
For returns, reverse or adjust COGS and record recovered inventory only as required by the applicable policy and the state of the returned goods.
A return may require inspection, markdown, refurbishment, transport, or disposal.
Keep actual service costs separate from the accounting carrying-value adjustment.

Do not add the full resale price of a returned item to its original order's margin and then also recognize the resale as a new order without an offset.
If original and resale orders are both in the same enterprise ledger, the recovery flows through the inventory and sale entries under the chosen convention.
A decision model may instead calculate expected net recovery for a disposition choice, but that estimate is not an extra posting to the P&L.
See [Disposition Economics](08_disposition-economics.md) for action-level alternatives.
Input: inventory movements, recognized COGS, return status, and recovery transactions.
Output: a cost bridge that is consistent with revenue and inventory records.

### Step 6 — Assign identifiable direct costs

Tag payment fees, packaging, carrier invoices, and order-specific contract charges to the relevant channel and order where traceable.
Use actual amounts when reliable and a documented estimate when invoices arrive later.
Identify minimum commitments and volume tiers before assuming the last observed average applies to the next order.
Match service credits or carrier refunds to the original cost without treating them as new revenue unless the controller's policy requires it.

Labor may be direct for management reporting when measured by activity, but it is not necessarily incremental in the next scheduling interval.
The example treats selected variable fulfillment expense as directly assigned and puts paid shared labor in a separate pool.
That separation is deliberate: the same wage cannot be both a direct expense and part of the shared pool.
Input: cost-center postings, transaction detail, and cost mapping.
Output: directly assigned channel costs and a list of excluded shared pools.

### Step 7 — Define shared pools and their boundaries

For each pool, list the general-ledger accounts, cost centers, period, currency, owner, and exclusions.
Split pools when activities have materially different cost drivers or controllability.
For example, pool store picking labor separately from store occupancy if minutes drive one and usable area drives the other.
Do not split pools merely to create apparent precision where measurement is unreliable.

Reconcile each source account to exactly one of direct assignment, shared pool, excluded item, or unallocated exception.
External carrier costs should not be buried in a labor pool if they have already been assigned directly.
Internal transfers included in a pool must be removed or separately identified so they do not add enterprise expense.
Input: account mapping and cost-center detail.
Output: mutually exclusive and collectively exhaustive cost pools with control totals.

### Step 8 — Choose a driver and a denominator

A useful driver is measurable, stable enough for the use, and plausibly related to resource consumption.
Picking minutes can be more informative than sales dollars for allocating shared picking labor.
Occupied area may be more relevant for occupancy, but it is imperfect if location quality or time differs.
Order count can be acceptable for a uniform per-order activity but is weak when a one-line pickup and a complex shipment consume very different work.
Test practicality and materiality before selecting a more granular driver.

Define the denominator explicitly.
An actual-usage denominator allocates the whole pool to observed users, including the cost of unused capacity.
A practical-capacity denominator charges users for measured capacity and leaves the unused portion visible.
Neither denominator changes enterprise cost; it changes where the reported cost appears.
The worked example uses practical capacity for labor and separately reports unused capacity.

Input: pool total, driver measurements, and practical capacity.
Output: documented rate, assigned amounts, and residual.
For a period pool of $12,000 and practical capacity of 6,000 minutes, the rate is $2 per minute.
If measured usage is 4,500 minutes, allocated cost is $9,000 and the unused-capacity residual is $3,000.
Do not call the $2 per minute a cash saving without testing avoidability.

### Step 9 — Allocate with explicit exceptions

Multiply each channel's driver units by the selected pool rate.
Add assigned amounts and compare them with the pool total.
Retain unused-capacity and unmatched-driver amounts as labeled residuals.
If the driver count exceeds practical capacity, investigate the capacity estimate, measurement, and overtime before applying the rate mechanically.
If records are incomplete, estimate only under an approved policy, with a separate estimated flag and owner.

Allocation is not an optimization objective by default.
It is legitimate to report a channel's share of rent for long-run accountability while excluding unchanged rent from tomorrow's routing decision.
Make that distinction visible in the report title and decision memo.
Input: approved pools and channel driver quantities.
Output: allocation journal or reporting table and exception list.

### Step 10 — Reconcile to the enterprise result

Compute channel contribution before shared pools as net revenue less COGS and directly assigned operating costs.
Subtract allocated shared costs to obtain each channel's reported result.
Then subtract unallocated shared costs and reconcile to the enterprise result.
Channel amounts plus residual should equal the enterprise result, subject only to a stated rounding tolerance.
Use a separate internal-transfer elimination when internal charges appear in the source ledger.

A reconciliation should show source totals, mappings, missing IDs, allocation totals, eliminations, and residuals.
Investigate differences at source-record level; do not force-balance with an unexplained plug.
A controller owns the control total and signs off the accounting treatment.
A data owner resolves missing IDs and late feeds; a business owner approves the management attribution rule.
Input: net revenue, costs, allocations, and ledger controls.
Output: a reconciled period P&L and documented exceptions.

### Step 11 — Model the incremental alternative

Name a baseline and a feasible alternative.
Hold all unchanged revenues and costs constant and remove them from the difference.
Include incremental revenue net of cancellations, returns, and expected discounts when the decision changes them.
Include the incremental product cost, packaging, payment, transport, and genuinely additional labor.
If a promised order replaces an in-store sale, include the contribution forgone under the best feasible alternative.
If the two alternatives share the same customer and same sale, do not count that sale's revenue twice.

Identify the scarce capacity explicitly: minutes, shelf space, carrier slots, or inventory units.
Estimate capacity available at the decision time, not only an average over the month.
Use overtime if it is feasible, authorized, and sufficient; otherwise model displacement, delay, or rejection.
Use the least-cost feasible response, considering any service constraints and labor rules.
Input: baseline, alternative, incremental drivers, available capacity, and service obligations.
Output: difference in enterprise contribution, feasibility, and sensitivity.

### Step 12 — Interpret allocation and incremental results together

A positive reported channel result does not prove that every additional order is profitable.
A negative reported result does not prove that closing the channel saves its allocated share of rent.
A route with a low accounting cost can have a high opportunity cost at a constrained store.
The appropriate question for a long-run closure is which leases, contracts, staffing, and sales actually change after closure.
The appropriate question for a next-hour pickup is which immediate costs and capacity opportunities change now.

Show both the reported P&L and the incremental comparison, clearly labeled.
Document the forecast of demand, plan for resources, target for margin or service, and customer commitment as separate fields.
Escalate when the best economic option violates the promised delivery or operational safety constraint.
A decision owner can choose a lower short-run margin for a stated strategic reason, but must not relabel a target as an observed result.

### Step 13 — Design incentives and review

Avoid paying both channel and store teams for a full sale as though the enterprise earned it twice.
Shared credit is permissible in a separate scorecard if the rules distinguish recognition from accounting revenue.
Credit a store for measured pickup and return work, service quality, and controllable operating performance.
Tie digital targets to enterprise contribution and customer outcomes, not gross order volume alone.

Give the store manager authority over staffing and capacity reservations if the manager is accountable for those resources.
Give the central planner authority over enterprise routing rules and make exceptions observable.
Finance owns the reconciliation and changes to allocation policy.
Customer experience owns service standards, while operations owns feasible execution.
Review incentives for rejected profitable orders, low-quality picks, excessive split shipments, and unfair workload transfers.
Input: scorecards, decision rights, and observed behavior.
Output: an incentive design and review cadence with named owners.

### Step 14 — Build an order-to-ledger trace

Choose one sample order and trace it through all systems before trusting aggregated totals.
Record the order ID, line ID, demand channel, and merchandise amount.
Record each shipment or pickup event and its fulfillment node.
Match each payment fee to the payment event rather than inferring it from the order total.
Match each return to the original line, not merely to the customer account.
Identify the inventory carrying-value movement for the original sale and, where appropriate, the return.
Identify every service cost and the source invoice or labor pool from which it came.
The objective is not to create a perfect per-order income statement from incomplete data.
The objective is to prove that each real financial event is classified once and that remaining gaps are visible.

Use the following example event trace.
The values are USD for a single order and are separate from the monthly example.
The channel is digital, and the fulfillment and return node is Store A.

| Event | Operational evidence | Financial amount | Treatment in this trace |
|---|---|---:|---|
| Original merchandise sale | Order line O1-L1 | $100 revenue | Recognize once in digital demand channel |
| Original product cost | Inventory issue I1 | $60 COGS | Recognize once under inventory policy |
| Payment processing | Processor fee P1 | $3 expense | Assign directly to digital |
| Store pick | Activity record A1 | 5 minutes | Use as labor driver; do not create another sale |
| Customer return | Return R1 linked to O1-L1 | $100 refund | Reverse net revenue under the assumed recognition treatment |
| Item restored to saleable inventory | Receipt I2 linked to R1 | $60 COGS reversal | Restore inventory and reverse original product cost under the assumed policy |
| Return processing | Activity record A2 | 3 minutes | Add to labor driver, not direct wages if the same wages are in the labor pool |
| Subsequent resale | New sale O2-L1 | $85 revenue and $60 COGS | Record a new transaction when resale occurs |

The original order's merchandise revenue and COGS net to zero after the assumed full return and full recovery.
The $3 processing fee remains an expense unless it is actually refunded.
The eight labor minutes are resource usage, not another $8 of expense.
Their reporting charge depends on the relevant labor-pool rate and allocation policy.
The later resale produces $25 of gross profit before further service costs.
Adding an extra $85 recovery credit to the original order as well as recording the resale would duplicate value.
If the return is damaged, the amount restored to inventory and subsequent expense can differ; obtain the controller's policy.

Do not assume the example's return treatment applies to every reporting framework or return circumstance.
For example, reporting may require an estimate of returns at the time of the original sale.
If that estimate is already recorded, later actual-return activity must be reconciled against it rather than independently duplicating it.
The sample trace should identify the allowance or liability account as well as the transaction record where relevant.
The financial statement and management views may classify timing differently but must be bridged.

### Step 15 — Select a driver with an explicit comparison

For each shared pool, compare at least two plausible bases before accepting one.
Consider causal plausibility, availability, cost to measure, auditability, and gaming risk.
Document the result even if the chosen basis is an imperfect compromise.
The following comparison applies to shared store picking labor, not all store expenses.

| Candidate basis | Advantage | Limitation | Reasonable use |
|---|---|---|---|
| Orders | Widely available | Large and small orders consume different effort | Stable, homogeneous order mix |
| Units picked | Reflects line quantity better than order count | Ignores travel and handoff time | Similar products and processes |
| Observed productive minutes | Relates directly to work performed | Samples and timestamps can be biased | Measured work with adequate controls |
| Sales dollars | Easy to reconcile | High-priced items need not consume more labor | A temporary proxy only with justification |
| Gross margin dollars | Aligns a financial scorecard | Does not measure labor consumption | Profit attribution, not a labor driver by default |
| Equal share | Simple and reproducible | Ignores differences in use | Immaterial pool with broadly similar users |

Assume a $9,000 pool, 300 digital orders, and 700 store orders.
An order-count allocation assigns $2,700 to digital and $6,300 to store.
Now suppose digital work consumed 2,000 minutes and store work consumed 1,000 minutes.
An actual-minute allocation assigns two thirds, or $6,000, to digital and one third, or $3,000, to store.
The difference in digital cost is $3,300 for the same $9,000 pool.
This is not evidence that minutes are always correct.
It shows that a choice of driver can materially change reported channel results even when enterprise cost is unchanged.

Investigate why digital orders consume more time before deciding whether to change operations.
Complex item assortment, longer travel paths, substitution rules, and handoff steps are possible mechanisms.
Sampling error, inaccurate event timestamps, and unmeasured store activity are also possible explanations.
Use a representative sample across hours, locations, and order complexity.
If the pool or driver is immature, report a range and an explicit exception rather than overstating precision.

Changing drivers can improve economic interpretation but disrupt trend comparability.
Version the method, identify its effective date, and show a prior-period restatement or a bridge where feasible.
Explain whether any management compensation is based on the old or new convention.
Finance approves the accounting mapping, while operations validates the activity measurement.
A channel leader can challenge the driver but should not unilaterally redefine a shared pool.

### Step 16 — Treat shared capacity as a portfolio

Capacity is shared across activities, locations, and time windows.
A 9 a.m. idle minute at Store A does not automatically satisfy a 5 p.m. pickup commitment at Store B.
The employee must have the necessary training and be available in the relevant interval.
Inventory and space must also be available.
A capacity model that aggregates an entire week may hide the only hour in which the promise is at risk.

Divide capacity into suitable decision buckets.
For pickup, the bucket might be 30 minutes at a specific store.
For parcel dispatch, it might be one carrier cutoff at a specific node.
For shelf storage, it might be cubic feet over a week, subject to product compatibility.
For return inspection, it might be available technician minutes per shift.
Choose the bucket to match the constraint and the decision, not merely the easiest data extract.

For each bucket, estimate scheduled capacity, normal allowances, committed work, and remaining usable capacity.
Subtract activities already promised to customers before offering the same capacity to another channel.
Distinguish a forecast of expected arrivals from a commitment to serve an accepted order.
Hold appropriate buffers for variability if the service policy requires them.
Do not count the same overtime shift as available to two stores or two competing queues.

Consider a store with 200 usable minutes in a one-hour window.
Committed store work requires 110 minutes and previously accepted pickups require 50 minutes.
The remaining uncommitted capacity is 200 − 110 − 50 = 40 minutes.
A new 60-minute batch cannot be accepted on the assumption that the store has 200 total minutes.
It needs 20 additional feasible minutes, a lower-priority task to displace, a revised promise, or a smaller batch.
If none is feasible, the batch is infeasible even when its paper contribution is positive.

When activity varies, an average requirement alone is inadequate for a hard promise.
A six-minute average can include orders that require substantially longer.
Use an appropriate distribution or a conservative operational buffer, and monitor actual service.
The correct buffer depends on acceptable failure probability, arrival correlation, and operational recovery options.
This lesson does not prescribe a universal service level or assume independent order times.

### Step 17 — Compare outsourcing, overtime, and displacement

Suppose a batch requires 120 minutes beyond idle capacity.
The baseline contribution before extra capacity cost is $640.
Three feasible ways to provide those minutes are proposed.
Each scenario below is an alternative, not an additive charge.

| Alternative | Incremental resource cost | Other incremental cost | Batch contribution |
|---|---:|---:|---:|
| Authorized overtime | 120 × $0.60 = $72 | $0 | $568 |
| Outsourced pick support | 120 × $0.80 = $96 | $24 coordination | $520 |
| Displace store work | 120 × $5 = $600 | $0 | $40 |

Under these assumptions, overtime yields the highest contribution.
If overtime cannot be scheduled, outsourced support yields more contribution than displacement.
If outsourced support cannot satisfy training or security requirements, it is not a feasible alternative.
If an alternative changes customer wait time, add the explicitly modeled service effect or enforce a service constraint.
Avoid adding an uncalibrated service penalty simply because a service outcome is inconvenient.

Suppose overtime requires a four-hour minimum shift rather than exactly 120 minutes.
At $0.60 per minute, the minimum shift costs 240 × $0.60 = $144.
Its contribution becomes $640 − $144 = $496 if the other 120 minutes have no incremental value.
Outsourcing at $520 then becomes preferable, assuming equivalent service and feasibility.
If the remaining overtime minutes support other incremental work, include that work in the same decision boundary.
Do not allocate the minimum shift to two decisions while also crediting both decisions for the full unused remainder.

The comparison may also change with a future staffing plan.
If repeated batches create enough sustained demand to hire, the relevant cost includes the feasible change in staffing and its timing.
If a contract can be renegotiated only at renewal, include its actual notice period and termination terms.
Fixed and variable are properties of a cost relative to a decision and horizon, not permanent labels.

### Step 18 — Separate a route choice from a demand choice

Sometimes the customer order is already accepted and the decision is where to fulfill it.
In that case, revenue and product cost may be the same under both feasible routes.
Do not present the entire sale as incremental to each route.
Compare the route-specific differences, including handling, transportation, capacity, and expected service effects.

Suppose the same accepted order can ship from a DC or from a store.
The DC route costs $3.20 handling and $6.80 parcel transportation.
The store route costs $1.50 packaging and $5.10 parcel transportation.
Both routes meet the promised date and use the same inventory carrying value.
Before capacity effects, DC cost is $10.00 and store cost is $6.60.
The store route saves $3.40 per order relative to the DC.

Now suppose the store pick uses six constrained minutes with an opportunity cost of $0.75 per minute.
The store route's economic cost becomes $6.60 + $4.50 = $11.10.
The DC is then $1.10 cheaper per order, assuming no additional DC constraint.
The $4.50 is foregone contribution, not a second payroll posting.
If the opportunity-cost estimate already reflects lost store contribution, do not add that lost contribution again.
If the store route instead uses truly idle suitable capacity, the $4.50 opportunity cost does not apply.

A demand-generation decision is different.
If a promotion creates an additional order, include incremental revenue, product cost, promotion cost, fulfillment, returns, and substitution.
If the promotion only shifts an existing customer from store purchase to digital pickup, the baseline includes the store purchase.
The route example cannot prove demand incrementality.
The batch example cannot prove that one specific node is the best route without node-level comparisons.

### Step 19 — Examine long-run channel exit without allocation shortcuts

For a proposed channel exit, begin with the baseline enterprise forecast.
Estimate which customer demand would be retained through other channels and which would be lost.
Estimate the contribution on retained demand under its new fulfillment pattern.
Estimate the contribution on lost demand only once.
Identify contracts, staff, systems, leases, and equipment that can actually change.
Include notice periods, severance, termination fees, migration costs, and the time at which savings begin.
Treat previously incurred implementation spending as sunk unless a future payment changes with the decision.

Suppose a channel's reported loss is $100,000 per year after $180,000 of allocated occupancy.
Management cannot infer $180,000 of occupancy savings from that allocation.
If the lease remains unchanged and no space can be sublet, immediate occupancy saving is $0.
If $70,000 of other future expenses are avoidable but $90,000 of contribution is lost, the annual change is −$20,000 before transition costs.
The reported $100,000 loss does not establish that closure improves the enterprise outcome.
Conversely, if a lease can actually be terminated with a net $100,000 annual saving, include that saving in the relevant future periods.
All amounts here are illustrative and annual, not a forecast of a named company.

Build a time-phased comparison when costs change at different dates.
A one-time $30,000 exit payment and an annual recurring saving should not be combined without specifying the horizon.
Where a multi-year investment decision requires it, finance should select an appropriate discount rate and timing convention.
Include uncertainty ranges for customer retention and contract outcomes.
The owner must approve the business decision; the analyst should not silently assume a legal right to terminate.

### Step 20 — Make scorecards resistant to gaming

Begin with an enterprise contribution metric using the same reconciliation perimeter as the monthly P&L.
Add a service measure such as on-time pickup readiness with an explicit denominator and exclusions.
Add a workload measure such as validated pickup and return minutes handled by each store.
Add a quality measure such as incorrect or missing items per fulfilled order.
Use a balanced set so no single measure rewards harmful behavior.

Digital gross orders alone can reward discounts or orders that lose money.
Store sales alone can reward refusing profitable pickup work.
Low measured handling time alone can reward incomplete scans or rushed picks.
On-time readiness alone can reward setting unduly long customer promises.
Return-processing speed alone can reward skipping inspection.
Every proposed metric needs a countermeasure or review for predictable unintended behavior.

An operational credit may be based on accepted and completed work rather than demand that never reaches the store.
It should distinguish actual work from estimated work.
The scorecard should state whether a cancellation after picking earns partial credit.
For a return, it should recognize processing workload without assigning the original digital sale to the store as new revenue.
For a shared customer journey, it can show assisted sales separately without summing multiple full credits into enterprise revenue.

The store manager should be able to adjust schedules, request overtime, or escalate a capacity conflict within defined limits.
The routing owner should see stores' stated constraints and actual acceptance performance.
Finance should review allocation-policy changes independently of the teams receiving performance credit.
Customer experience should own the definition of a valid promise and monitor exceptions.
Human resources should review any compensation plan for applicable labor and employment requirements.

### Step 21 — Govern estimates and exceptions

An allocation can be mathematically exact while its inputs are uncertain.
Classify each input as observed, estimated, forecast, planned, or manually overridden.
Record its owner, measurement period, source, and last approval date.
For sampled labor, document the sample frame and whether peak periods are represented.
For estimated returns, document the maturity window and how subsequent actuals are compared with estimates.
For occupancy, document the area or other use basis and when it was measured.

Set a materiality threshold for unresolved exceptions in both USD and percentage terms.
For example, a controller might require investigation of an unmatched pool above a locally approved threshold.
That threshold is an internal control setting, not an external accounting requirement supplied by this lesson.
Keep small exceptions visible in aggregate; many individually small exceptions can become material.
Assign an owner and resolution date to each material exception.
Do not silently reclassify unmatched costs to the largest channel just to make a chart balance.

Maintain two controls after every close.
First, the sum of all channel sales and explicitly unassigned sales must match the relevant revenue control total.
Second, direct costs, allocated pools, unallocated pools, and approved eliminations must reconcile to the corresponding expense control total.
Both controls need a documented sign convention.
If rounding creates a difference, display the rounding amount separately and define the precision used.
If a source posts late, either reopen or adjust the later period under a documented policy.
Do not rewrite historical metrics without recording the method and date of restatement.

### Step 22 — Write the decision memo

Use a short memo that a finance director and store manager can both audit.
State the baseline and alternative in the first paragraph.
State the decision horizon and the enterprise boundary in the second.
Show a compact table of incremental revenue, COGS, variable costs, capacity costs, and total contribution.
List service constraints separately from financial outcomes.
Name the owner of the recommendation and the owner of each key input.

Include the forecast of demand and its uncertainty.
Include the operational plan for staffing and inventory.
Include the management target, if one is relevant, without treating it as a forecast.
Include the customer commitment and the conditions required to meet it.
State which pool allocations are displayed for reporting but excluded from the incremental comparison.
Show at least one break-even calculation and one material assumption that could reverse the recommendation.

For the worked batch, a concise recommendation is conditional.
Accept the batch if the 240 usable minutes are truly available and the displayed pickup promise remains feasible.
If only 120 minutes are idle, prefer authorized overtime at $72 over displacing $600 of contribution.
If overtime is not feasible, the expected $40 margin before additional uncertainty is narrow and deserves review.
If the displacement estimate already includes lost customer demand, do not deduct that demand again.
If evidence suggests material cannibalization not included elsewhere, reduce the incremental benefit accordingly.
The store operations manager makes the next-day decision within delegated limits.
Finance validates the method, and the customer experience owner validates the promise constraint.

### Step 23 — Test the effect of forecast error

A contribution estimate is conditional on the number and mix of orders that actually arrive.
It is not enough to test only one average volume.
For a capacity-sensitive decision, test demand, available minutes, and cost together.
The downside can be nonlinear because overtime minimums and service constraints can activate at thresholds.

Use the batch's $16 contribution per order before additional labor.
At six minutes per order, 120 idle minutes support 20 orders.
Assume additional minutes can be staffed at $0.60 per minute with no minimum.
Assume all orders remain genuinely incremental and all can meet the promise.

| Demand forecast | Required minutes | Overtime minutes | Pre-labor contribution | Overtime cost | Incremental contribution |
|---|---:|---:|---:|---:|---:|
| 15 orders | 90 | 0 | $240 | $0 | $240 |
| 20 orders | 120 | 0 | $320 | $0 | $320 |
| 30 orders | 180 | 60 | $480 | $36 | $444 |
| 40 orders | 240 | 120 | $640 | $72 | $568 |
| 50 orders | 300 | 180 | $800 | $108 | $692 |

The table is a sensitivity, not a probability distribution.
It assumes overtime can scale continuously and that there is no overtime ceiling.
If the store can provide at most 120 overtime minutes, the 50-order scenario is infeasible as stated.
Either cap accepted orders at 40, find another feasible resource, or revise the promise.
The financial result for 50 orders should not be presented as executable until capacity is resolved.

The expected result across scenarios requires probabilities that sum to one.
Those probabilities should come from an appropriate forecast or be labeled judgmental.
Do not average the five rows equally unless equal probabilities are actually intended.
Also assess downside risk: an average profit does not excuse a systematic failure of the promised service level.
Separate controllable staffing choices from demand uncertainty when assigning accountability.

### Step 24 — Evaluate a partial batch under scarce capacity

The all-or-nothing result can conceal a better feasible choice.
Suppose there are 40 potential orders, each worth $16 before scarce capacity and requiring six minutes.
The store has 120 suitable idle minutes and no overtime.
Those idle minutes can support 20 orders.
The remaining 20 orders require 120 minutes that would displace other work.

Accepting only 20 orders uses the idle capacity and produces 20 × $16 = $320.
Accepting all 40 produces $640 before the displacement cost.
At $5 of displaced contribution per minute, the full batch produces $40.
The 20-order option therefore exceeds the full-batch result by $280.
The right comparison is $320 versus $40, not $40 versus zero.
If the retailer cannot selectively accept 20 orders without violating a commitment, the partial option is not feasible.
If orders differ in contribution or effort, choose the feasible mix subject to fairness and business rules rather than assuming identical orders.

For the extra 20 orders, incremental contribution before displacement is $320.
Their displaced contribution is 120 × $5 = $600.
Thus expanding from 20 to 40 orders changes enterprise contribution by −$280.
The break-even displacement rate for the extra orders is $320 / 120 = $2.6667 per minute.
This differs from the $5.3333 per minute threshold comparing the full batch with accepting no orders.
The correct threshold depends on the alternative, and the no-batch threshold cannot answer the partial-batch question.

### Step 25 — Close the loop after the decision

Record the chosen alternative and the assumptions available at decision time.
After the operating window, compare actual orders, minutes, overtime, and service outcomes with the forecast and plan.
Recalculate the result using actual incremental costs where they can be observed.
Keep a clear distinction between the original decision estimate and the later actual result.
An unfavorable outcome alone does not prove the decision was unreasonable given the information available.

Investigate the largest variance before changing the allocation method.
If order minutes were higher because the mix changed, improve the work-content estimate.
If overtime was booked but not used for pickup, correct the attribution and verify time records.
If store sales fell, assess whether demand was actually displaced rather than assuming causality from coincidence.
If service failed, determine whether the constraint was capacity, inventory accuracy, customer arrival clustering, or an execution issue.
The improvement owner should have authority to act on the identified mechanism.
Finance keeps the reconciliation evidence; operations owns capacity corrections; the digital team owns its demand assumptions.

## 6. Worked example

### Case and accounting boundary

A fictional retailer reports one calendar month in USD.
The two demand channels are digital and store.
Digital orders can be fulfilled by a DC or store; store sales can require store labor.
All figures below represent the same legal-entity perimeter and the same month.
Recognized net revenue already reflects the month's recognized discounts and refunds.
COGS already reflects the month's recognized inventory adjustments.
The example excludes sales tax, income tax, financing, and other operating expenses outside the stated perimeter.

| Monthly item | Digital | Store | Total |
|---|---:|---:|---:|
| Recognized net revenue | $96,000 | $144,000 | $240,000 |
| Recognized COGS | $57,600 | $86,400 | $144,000 |
| Direct operating costs | $14,400 | $9,600 | $24,000 |
| Contribution before shared costs | $24,000 | $48,000 | $72,000 |

Digital contribution is $96,000 − $57,600 − $14,400 = $24,000.
Store contribution is $144,000 − $86,400 − $9,600 = $48,000.
The $24,000 of direct operating costs excludes the shared labor and occupancy pools below.
This exclusion prevents a direct charge from being allocated again.
The table is a reported actual, not a demand forecast, staffing plan, margin target, or delivery commitment.

### Shared labor pool

The store has $12,000 of shared paid labor in the month.
The measured activity is 1,500 minutes for digital orders and 3,000 minutes for store sales.
Practical capacity is 6,000 minutes, leaving 1,500 minutes unused.
These are illustrative productive-capacity minutes, not total clocked minutes.
The $12,000 pool is the labor cost associated with that defined capacity after normal allowances.

The practical-capacity rate is $12,000 / 6,000 = $2 per minute.
Digital receives 1,500 × $2 = $3,000.
Store receives 3,000 × $2 = $6,000.
Unused capacity is 1,500 × $2 = $3,000.
The pool reconciles: $3,000 + $6,000 + $3,000 = $12,000.

An alternative actual-use allocation would divide $12,000 by 4,500 minutes, or $2.6667 per minute.
It would assign approximately $4,000 to digital and $8,000 to store, with no visible unused-capacity residual.
Both methods allocate the same $12,000 enterprise cost.
The second method makes the cost per active minute rise when activity falls, even if the resource cost is unchanged.
It can obscure the idle-capacity decision and make channel trends misleading.

### Occupancy pool

Monthly shared occupancy is $18,000.
The approved management rule assigns 25% to digital support and 75% to store activity.
The percentages are an illustrative area-use convention, not proof that a quarter of the lease would disappear if digital demand ceased.
Digital receives $18,000 × 25% = $4,500.
Store receives $18,000 × 75% = $13,500.
All occupancy is assigned in this example, so its residual is $0.

### Channel and enterprise bridge

| Monthly bridge | Digital | Store | Unallocated | Enterprise |
|---|---:|---:|---:|---:|
| Contribution before shared pools | $24,000 | $48,000 | $0 | $72,000 |
| Shared labor | −$3,000 | −$6,000 | −$3,000 | −$12,000 |
| Shared occupancy | −$4,500 | −$13,500 | $0 | −$18,000 |
| Result after shared costs | $16,500 | $28,500 | −$3,000 | $42,000 |

Digital reported result is $24,000 − $3,000 − $4,500 = $16,500.
Store reported result is $48,000 − $6,000 − $13,500 = $28,500.
The two channel results total $45,000, not the enterprise result, because $3,000 of unused labor remains unallocated.
Subtracting that residual yields $42,000.
The independent enterprise bridge is $240,000 − $144,000 − $24,000 − $12,000 − $18,000 = $42,000.
The two routes reconcile exactly.

A dashboard that reports $45,000 as total enterprise profit has dropped the idle-capacity cost.
A dashboard that adds the $3,000 residual to labor already fully assigned under the actual-use method counts the same cost twice.
A dashboard that adds an internal store service charge to external labor without elimination also overstates enterprise expense.
The unallocated amount is a management choice about visibility, not missing money.

### Next-day pickup decision

The digital team forecasts a batch of 40 incremental pickup orders tomorrow.
The operations plan reserves 240 store labor minutes for the batch.
The planning target is nonnegative incremental contribution while meeting the existing pickup commitment.
The commitment is customer collection within the displayed two-hour window; feasibility depends on the actual staffing and arrival pattern.
All 40 orders are assumed genuinely incremental to the enterprise, with no lost store sale and no incremental return costs beyond those embedded in the net figures below.
This is a strong assumption to test, not a conclusion from channel attribution.

Expected net revenue is $50 per order, or $2,000 for the batch.
Expected COGS is $30 per order, or $1,200.
Incremental payment, packaging, and other variable expense is $4 per order, or $160.
Before extra labor or capacity displacement, incremental contribution is $2,000 − $1,200 − $160 = $640.
Each order requires six productive minutes, so 40 × 6 = 240 minutes.

If 240 suitable minutes are genuinely idle within the promised window and paid staffing is unchanged, incremental cash labor is $0.
Incremental contribution is therefore $640, before any service or risk effects omitted by the assumptions.
The monthly allocation of 240 × $2 = $480 is not an extra cash outflow in this scenario.
It may still appear in a management P&L for resource accountability.
The decision result and reported allocation are different views of the same resource, not amounts to add together.

Now assume only 120 suitable minutes are idle.
The other 120 minutes can be covered by authorized overtime at $0.60 per minute, including relevant payroll on-costs.
Overtime cost is 120 × $0.60 = $72.
The batch yields $640 − $72 = $568 of incremental contribution.
The $0.60 rate is a decision-specific future cash rate; it differs from the $2 historical practical-capacity allocation rate.

If overtime is unavailable and the extra 120 minutes instead displace store activity worth $5 contribution per minute, opportunity cost is $600.
The resulting contribution is $640 − $600 = $40, assuming no further service penalty.
If the displaced contribution rises above $640 / 120 = $5.3333 per minute, accepting the full batch reduces enterprise contribution under these assumptions.
At exactly that threshold the two choices are economically tied before risk and service considerations.
A smaller batch could be preferable; compare feasible batch sizes rather than forcing an all-or-nothing choice.
Do not subtract both $72 overtime and $600 displacement if overtime actually prevents all of that displacement.
If overtime covers only part, calculate the remaining displaced minutes separately.

The digital leader proposes the batch.
The store manager confirms capacity and overtime availability.
Finance validates the assumptions and avoids treating allocated occupancy as avoidable.
The customer experience owner confirms whether the two-hour promise is feasible.
The accountable decision owner is the operations manager for the next-day plan, within approved financial and service limits.

### Counterfactual check for cannibalization

Suppose 10 of the 40 pickup orders would otherwise have been store sales with $12 contribution each.
Assuming these are the only displaced sales and the incremental digital figures above still apply, subtract $120 of forgone store contribution.
The idle-capacity scenario then yields $640 − $120 = $520.
The overtime scenario yields $640 − $72 − $120 = $448.
The $5-per-minute displacement scenario needs special care: if its $600 already includes those 10 store sales, do not deduct the $120 again.
A causal estimate of substitution requires evidence; the order-entry channel alone cannot establish it.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the monthly bridge and the three next-day scenarios above.
It uses `Decimal` for currency and explicit checks for reconciliation.
Values are illustrative monthly USD and next-day USD; there is no external input file.
The opportunity-cost scenario assumes overtime is unavailable and does not additionally deduct the cannibalization sensitivity.

```python
from decimal import Decimal

D = Decimal
channels = {
    "digital": {"revenue": D("96000"), "cogs": D("57600"),
                "direct": D("14400"), "minutes": D("1500"),
                "occupancy_share": D("0.25")},
    "store": {"revenue": D("144000"), "cogs": D("86400"),
              "direct": D("9600"), "minutes": D("3000"),
              "occupancy_share": D("0.75")},
}
labor_pool = D("12000")
practical_minutes = D("6000")
occupancy_pool = D("18000")
assert sum(x["occupancy_share"] for x in channels.values()) == D("1")
used_minutes = sum(x["minutes"] for x in channels.values())
assert used_minutes <= practical_minutes
labor_rate = labor_pool / practical_minutes
unallocated_labor = labor_pool - used_minutes * labor_rate

results = {}
for name, x in channels.items():
    contribution = x["revenue"] - x["cogs"] - x["direct"]
    allocated_labor = x["minutes"] * labor_rate
    allocated_occupancy = occupancy_pool * x["occupancy_share"]
    results[name] = contribution - allocated_labor - allocated_occupancy

enterprise_from_channels = sum(results.values()) - unallocated_labor
enterprise_direct = (
    sum(x["revenue"] - x["cogs"] - x["direct"]
        for x in channels.values())
    - labor_pool - occupancy_pool
)
assert enterprise_from_channels == enterprise_direct
print(f"Digital reported result: ${results['digital']:.2f}")
print(f"Store reported result:   ${results['store']:.2f}")
print(f"Unallocated labor:       ${unallocated_labor:.2f}")
print(f"Enterprise result:       ${enterprise_direct:.2f}")

orders = D("40")
base = orders * (D("50") - D("30") - D("4"))
required_minutes = orders * D("6")
short_minutes = required_minutes - D("120")
idle = base
overtime = base - short_minutes * D("0.60")
displacement = base - short_minutes * D("5")
threshold = base / short_minutes
print(f"Idle-capacity increment: ${idle:.2f}")
print(f"Overtime increment:      ${overtime:.2f}")
print(f"Displacement increment:  ${displacement:.2f}")
print(f"Break-even displacement: ${threshold:.4f}/minute")
```

Expected output:

```text
Digital reported result: $16500.00
Store reported result:   $28500.00
Unallocated labor:       $3000.00
Enterprise result:       $42000.00
Idle-capacity increment: $640.00
Overtime increment:      $568.00
Displacement increment:  $40.00
Break-even displacement: $5.3333/minute
```

The assertions are controls for this example, not a complete production validation system.
A production implementation needs effective-dated mappings, late-arriving records, precision and rounding rules, and a controlled reconciliation to the general ledger.
Do not use this static example as an automated routing policy.
If practical minutes, shares, or costs are estimated, preserve their provenance and sensitivity in the decision record.

## 8. Data, standards, and regulatory context

The order-management system supplies order IDs, channels, fulfillment nodes, and timestamps.
The point-of-sale system supplies store sales, returns, discounts, and tender events.
The warehouse and store execution systems supply picks, handoffs, and labor activity where measured.
The returns system supplies return authorization, receipt, condition, and disposition.
The general ledger supplies controlled account totals; accounts payable and payroll can provide supporting detail.
Inventory records supply carrying values and physical movements.
Payment processors and carriers supply transaction or invoice references and fees.
None of these systems alone is guaranteed to contain the whole economic transaction.

Minimum transaction grain is an immutable event ID linked to an order line where feasible.
Record event time, posting time, source system, currency, amount, legal entity, channel attribution, node, and mapping version.
For labor, retain activity minutes, site, time window, measurement method, and whether time is actual, sampled, or estimated.
For pools, retain ledger accounts, cost center, exclusions, driver numerator and denominator, rate, residual, and approver.
For returns, retain the original sale link, refund, condition, disposition, and relevant inventory movements.
For every dataset, document late-arrival cutoffs and the process for restating prior periods.

International Financial Reporting Standard (IFRS) 15 addresses revenue from contracts with customers for entities applying IFRS Accounting Standards.
Its treatment of rights of return includes refund liabilities and assets for rights to recover products, subject to its requirements.
International Accounting Standard (IAS) 2 addresses inventories, including their cost and subsequent expense recognition.
These standards do not prescribe this lesson's particular channel-allocation percentages or management scorecard.
Applicable requirements depend on the entity, jurisdiction, reporting framework, facts, and effective version; the controller must determine the treatment.
Consult the official standard pages in Sources and the authoritative text where applicable.

For entities applying U.S. generally accepted accounting principles, Accounting Standards Codification (ASC) Topic 606 governs revenue recognition; consult the Financial Accounting Standards Board (FASB) and the applicable authoritative text.
Do not assume a management P&L is a statutory segment disclosure or that this illustration determines a reporting obligation.
For sales taxes, privacy, labor rules, and consumer returns, obligations vary by jurisdiction and transaction.
Engage the relevant legal, tax, privacy, and labor specialists instead of deriving legal obligations from an allocation model.
This lesson is educational, not legal or accounting advice.

Payment-card data should not be copied into analytical tables merely to link fees to an order.
Use tokenized references and least-privilege access under the organization's security policy.
Personal data should be minimized, retained only as needed, and handled under applicable privacy requirements.
Financial controls include source-to-ledger reconciliation, separation of duties, change approval, audit trails, and exception ownership.
Allocation drivers are management estimates: label estimates, retain versions, and avoid false precision.

## 9. Industry practice and case evidence

Public company reporting offers context, but it rarely exposes enough detail to reconstruct a channel P&L.
Walmart's annual report describes its omnichannel business and financial results at the consolidated and segment levels.
Its public disclosures should not be interpreted as the fictional store-labor allocations used in this lesson.
Target's annual report describes a business in which stores support shopping and fulfillment and discusses sales and operations.
Those disclosures are evidence that physical and digital activities interact, not evidence for a specific marginal cost or incentive rate.
Read the relevant report for the reporting year and definitions before comparing periods.

The IFRS Foundation's published standards pages establish the authoritative framework's topic and scope.
They do not validate any one management allocation driver.
The FASB's Accounting Standards Codification is the authoritative source for U.S. GAAP, subject to its access conditions and applicability.
Official source documents should be preferred over vendor claims about a retailer's profitability.

[INFERENCE] A retailer with stores serving both walk-in shoppers and pickup orders needs a rule for shared store resources if it publishes internal channel economics.
The public reports do not establish that either named company uses the practical-capacity example shown above.
[INFERENCE] Channel-only sales targets can create incentives to transfer workload across organizational boundaries; the actual impact depends on scorecard design and authority.
[INFERENCE] A mature return cohort can improve comparability with another mature cohort, provided the underlying demand mix and policies are also considered.
These are analytical implications, not claims of measured outcomes at the named companies.

When evaluating a published case, ask which entity and period the numbers cover.
Ask whether the company reports demand channel, fulfillment location, or a blend.
Check whether returns, delivery fees, advertising, and corporate costs are included.
Check whether a reported improvement is causal or merely coincident with a policy change.
A case that lacks a counterfactual can describe practice without proving the incremental benefit.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Duplicate revenue | Store and digital sales sum above ledger revenue | Pickup gets credited as two sales | Inflated enterprise margin | One canonical sale; separate scorecard credit |
| Duplicate refund | Net revenue drops twice for one return | Refund and reversal both deducted | Understated channel result | Trace refund IDs and bridge once |
| Duplicate shared wages | Labor appears in direct costs and pool | Pool boundaries overlap | Inflated cost-to-serve | Reconcile accounts to exclusive classifications |
| Dropped idle capacity | Channel results exceed enterprise total | Practical-capacity residual omitted | False enterprise profit | Report unallocated residual explicitly |
| Actual-use denominator drift | Cost per minute rises as volume falls | Idle cost forced onto fewer activities | Misleading channel trend | Show capacity and residual alongside rate |
| Allocated rent treated as avoidable | Closure memo assumes full rent savings | Allocation confused with lease change | Overstated closure benefit | Review lease and horizon with facilities |
| Free labor assumption | Peak-time orders delay store service | Monthly spare capacity assumed available at peak | Lost sales and missed promises | Check capacity by time window |
| Double-counted opportunity cost | Both displaced margin and identical penalty deducted | Two proxies describe one loss | Rejected profitable work | Map each loss to a distinct counterfactual |
| Internal transfer retained | Enterprise expense rises after internal recharge | Both payer and receiver entries retained | False enterprise loss | Eliminate internal charges at consolidation |
| Return recovery doubled | Original order gets full recovery and resale has full margin | Same recovered value credited twice | Inflated lifetime margin | Reconcile inventory and resale postings |
| Immature return cohorts | New cohorts look unusually profitable | Returns have not yet arrived | Premature channel expansion | Use comparable maturity and estimates |
| Missing cost driver | Pool cost is allocated only to matched orders | Orphan events silently dropped | Biased channel economics | Quarantine and disclose exceptions |
| Wrong period | February refunds attributed to January close without bridge | Cohort and accounting views mixed | Ledger mismatch | Maintain separate period and cohort views |
| Incentive conflict | Stores reject pickup despite positive enterprise contribution | Store bears cost without credit or authority | Local optimization | Align shared scorecard and decision rights |
| Unsupported incrementality | All digital orders labeled new demand | Channel attribution substituted for causal evidence | Inflated decision benefit | Estimate substitution and run sensitivities |
| Overprecision | Driver rate shows cents despite sampled time | Estimate uncertainty hidden | False confidence | Show ranges and measurement provenance |
| Inventory counted twice | Transfer becomes a second COGS event | Internal movement treated as sale | Inflated expense | Reconcile movement and sale events |
| Unfeasible staffing | Plan uses overtime without authorization | Cost model ignores operational constraints | Promise failure | Confirm labor availability and service feasibility |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Channel is order-entry channel | Enables one primary revenue owner | Assisted sales have multiple meaningful touchpoints | Compare alternative attribution and preserve assistance tags |
| Net revenue already includes refunds | Avoids a second deduction | Source contains gross sales instead | Bridge transactions to recognized ledger revenue |
| COGS includes recognized return adjustments | Avoids separate unsupported inventory reversal | Return inventory feeds lag or use different policy | Reconcile inventory movements and closing balances |
| Direct costs exclude shared pools | Ensures each expense enters once | Payroll or occupancy appears in both mappings | Test account and cost-center membership |
| Practical labor capacity is 6,000 minutes | Makes unused capacity measurable | Capacity is estimated poorly or shifts with roster | Compare engineering assumptions with observed available minutes |
| Digital uses 1,500 minutes and store 3,000 | Supports the allocation | Time samples omit activities or bias peaks | Compare time studies with independently observed work |
| Occupancy shares are 25% and 75% | Provides a reporting convention | Area, time, or use changes materially | Refresh measurements and sensitivity-test shares |
| 40 orders are incremental | Provides a baseline batch comparison | Customers switch from store or another digital route | Estimate substitution with suitable causal design |
| Each order takes six minutes | Sizes resource demand | Order complexity and arrival bursts vary | Measure distribution by order type and interval |
| Idle minutes are suitable for pickup | Makes immediate payroll increment zero | Wrong skills, breaks, tasks, or timing | Check schedule and workload within promise window |
| Overtime is available at $0.60 per minute | Makes overtime alternative feasible | Approval, contracts, or payroll rates differ | Obtain manager and payroll confirmation |
| Displacement is $5 per minute | Values scarce capacity in the scenario | Alternative work has nonlinear or uncertain value | Compare feasible alternative work and contribution ranges |
| No incremental return cost beyond net figures | Keeps batch calculation concise | Return frequency or processing differs | Add expected incremental return flows without duplicating net revenue |
| No other variable cost changes | Makes arithmetic bounded | Carrier, support, fraud, or advertising changes | Review the causal cost inventory with owners |
| Static one-month pool is representative | Simplifies teaching | Seasonal labor and lease effects differ | Compare several periods and structural changes |
| All values are one currency and entity | Allows direct summation | Foreign exchange or intercompany transactions exist | Translate and eliminate using approved policy |

These assumptions are scenario inputs, not empirically validated facts.
A decision must be revisited when a binding constraint or counterfactual changes.
The $2 allocation rate does not establish the labor price of an additional minute.
The $5 displacement value is illustrative and may vary sharply across the day.

## 12. Exercises

### Exercise 1 — Distinguish dimensions

Time: 10 minutes.
An order is placed online, picked at Store A, and returned at Store B.
Deliverable: identify demand channel, fulfillment node, and return location under this lesson's convention.
Acceptance criteria: use three distinct fields and explain why the transaction is not two enterprise sales.

### Exercise 2 — Calculate contribution

Time: 10 minutes.
A channel reports $75,000 net revenue, $43,000 COGS, and $12,000 direct operating expense for a month.
Deliverable: calculate contribution before shared costs and show the arithmetic.
Acceptance criteria: result is in USD for the month and does not deduct an unspecified shared pool.

### Exercise 3 — Reconcile the pools

Time: 15 minutes.
A $10,000 labor pool has practical capacity of 5,000 minutes.
Digital uses 1,000 minutes and store uses 3,000 minutes.
Deliverable: calculate the rate, both channel allocations, and unused-capacity residual.
Acceptance criteria: all allocations plus residual equal the original pool and units are stated.

### Exercise 4 — Compare denominators

Time: 15 minutes.
Using Exercise 3, allocate the same pool entirely using actual usage.
Deliverable: calculate each channel's allocation and compare it with practical-capacity allocation.
Acceptance criteria: show both denominators, reconcile $10,000, and explain why enterprise cost does not change.

### Exercise 5 — Find double counting

Time: 15 minutes.
A report subtracts $4,000 of direct order-picking wages and allocates a $10,000 labor pool that already includes those wages.
Deliverable: identify the overlap and give one corrected pool amount if the direct assignment remains.
Acceptance criteria: explicitly preserve the original total cost rather than simply deleting the wages.

### Exercise 6 — Make an incremental decision

Time: 20 minutes.
A batch adds $1,500 net revenue, $900 COGS, and $180 other variable costs.
It requires 100 minutes; 40 minutes are idle and the remainder can be covered by overtime at $0.75 per minute.
A $200 rent allocation is unchanged by the decision.
Deliverable: calculate incremental contribution and explain the rent treatment.
Acceptance criteria: show the baseline difference and count the overtime once.

### Exercise 7 — Test scarce capacity

Time: 20 minutes.
For Exercise 6, assume overtime is unavailable and the 60 non-idle minutes displace work contributing $8 per minute.
Deliverable: calculate the incremental result and the break-even contribution per displaced minute.
Acceptance criteria: do not count overtime as well as full displacement; interpret the sign.

### Exercise 8 — Design incentives and controls

Time: 30 minutes.
A digital team is rewarded for gross orders; stores absorb pickup and return work but have no capacity authority.
Deliverable: propose a one-page scorecard, decision-rights matrix, and monthly reconciliation control.
Acceptance criteria: include enterprise contribution, service, workload recognition, a named owner for each decision, and a control for duplicate revenue.
Explain how the proposal avoids turning shared credit into duplicate financial revenue.

## 13. Self-check answers

### Answer 1

Demand channel is digital, fulfillment node is Store A, and return location is Store B.
The original order is one enterprise sale; fulfillment and return are associated events.
The return may reduce recognized revenue according to the applicable accounting policy, but it does not create an independent second sale.

### Answer 2

$75,000 − $43,000 − $12,000 = $20,000 per month.
This is contribution before shared costs, not net income and not the next order's incremental profit.

### Answer 3

The rate is $10,000 / 5,000 = $2 per minute.
Digital receives 1,000 × $2 = $2,000.
Store receives 3,000 × $2 = $6,000.
Unused capacity is 1,000 minutes, or $2,000.
The control is $2,000 + $6,000 + $2,000 = $10,000.

### Answer 4

Actual use is 1,000 + 3,000 = 4,000 minutes.
The actual-use rate is $10,000 / 4,000 = $2.50 per minute.
Digital receives $2,500 and store receives $7,500; there is no residual under that method.
Compared with Exercise 3, the extra $500 and $1,500 distribute the $2,000 idle-capacity cost to active channels.
Total enterprise cost remains $10,000 under either method.

### Answer 5

The $4,000 wages are present in both the direct charge and the original labor pool.
If all $4,000 genuinely belong to the direct charge and no other overlap exists, reduce the pool to $6,000.
Direct wages plus the corrected pool remain $4,000 + $6,000 = $10,000.
Validate the source accounts and labor records before making the adjustment.

### Answer 6

Contribution before capacity cost is $1,500 − $900 − $180 = $420.
There are 100 − 40 = 60 overtime minutes, costing 60 × $0.75 = $45.
Incremental contribution is $420 − $45 = $375.
The unchanged $200 rent allocation is excluded because it is the same in both alternatives.
It can still appear in the reporting P&L.

### Answer 7

Opportunity cost is 60 × $8 = $480.
Incremental contribution is $420 − $480 = −$60.
The break-even displaced contribution is $420 / 60 = $7 per minute.
At $8 per minute, the batch reduces enterprise contribution under these assumptions.
The $45 overtime cost from Exercise 6 is not also charged because overtime is unavailable in this alternative.

### Answer 8 rubric

A strong answer assigns finance ownership of source-to-ledger reconciliation and allocation governance.
It gives store operations authority over feasible capacity and credits measurable work and service outcomes.
It assigns digital leadership an enterprise-contribution measure rather than only gross order count.
It uses a single financial revenue record and labels any shared incentive credit separately.
It includes a customer promise measure and a process to resolve conflicts between service, capacity, and economics.
It specifies exception thresholds, an owner, and a review cadence rather than only naming metrics.
A weak answer credits the full sale as new enterprise revenue twice or makes stores accountable for work they cannot schedule.

## 14. Interview questions

1. How can a channel be profitable in a fully allocated report but unprofitable for its next order?
   A strong answer distinguishes historical allocation from incremental variable and capacity costs.
   A weak answer assumes the reported average always applies to the next order.
2. When can paid store labor have zero incremental cash cost?
   A strong answer identifies suitable idle paid capacity and a specific time horizon.
   A weak answer declares all store labor free.
3. When is unused capacity an explicit residual?
   A strong answer describes a practical-capacity denominator and reconciles the residual to the pool.
   A weak answer removes idle cost from the enterprise P&L.
4. How would you prevent a pickup sale from being counted twice?
   A strong answer uses a canonical order ID, one revenue owner, and separate incentive credits.
   A weak answer adds store and digital scorecard sales into enterprise revenue.
5. What is the right cost of store rent when deciding whether to accept one order tomorrow?
   A strong answer asks what rent changes, usually excluding unchanged rent from the increment.
   A weak answer mechanically applies allocated rent as a cash saving or cost.
6. How do you model a binding labor constraint?
   A strong answer compares feasible overtime, displacement, rejection, and service effects without double counting.
   A weak answer uses a monthly average without checking the relevant time window.
7. How do you compare a new order cohort with an older one?
   A strong answer aligns maturity or estimates late returns and shows uncertainty.
   A weak answer treats incomplete returns as a permanent improvement.
8. What controls establish that shared costs are counted exactly once?
   A strong answer maps ledger accounts to exclusive pools, reconciles residuals, and eliminates internal transfers.
   A weak answer relies only on a dashboard total.
9. How can incentives recognize store support without duplicating sales?
   A strong answer separates operational credit and compensation from the financial ledger.
   A weak answer records full revenue in two channel P&Ls and sums them.
10. How do you know a digital order is incremental to the enterprise?
    A strong answer discusses substitution, an explicit counterfactual, and evidence or sensitivity analysis.
    A weak answer equates order-entry channel with causation.

## 15. Key terms

- **Allocation:** Distribution of a shared pool to cost objects; see [mechanics](#step-8--choose-a-driver-and-a-denominator).
- **Allocation driver:** Quantity used to assign a shared pool; see [definitions](#4-definitions-and-vocabulary).
- **Avoidable cost:** Cost that disappears under a specific alternative and horizon; see [incremental decision](#step-11--model-the-incremental-alternative).
- **Capacity:** Available resource measured in a stated unit and interval; see [mental model](#3-the-core-mental-model).
- **Cannibalization:** Demand shifted from a feasible alternative; see [counterfactual check](#counterfactual-check-for-cannibalization).
- **Channel:** Demand-origin classification; see [definitions](#4-definitions-and-vocabulary).
- **Cohort:** Transactions grouped by a common starting attribute; see [period logic](#step-3--choose-period-and-cohort-logic).
- **Commitment:** Agreed service obligation, distinct from a forecast; see [next-day decision](#next-day-pickup-decision).
- **Contribution:** Revenue less a stated set of costs; see [worked example](#6-worked-example).
- **Cost object:** Entity to which costs are assigned; see [definitions](#4-definitions-and-vocabulary).
- **Cost-to-serve:** Defined operating costs of serving an object; see [definitions](#4-definitions-and-vocabulary).
- **Direct cost:** Traceable cost; see [direct costs](#step-6--assign-identifiable-direct-costs).
- **Elimination:** Removal of internal charges from the consolidated view; see [reconciliation](#step-10--reconcile-to-the-enterprise-result).
- **Forecast:** Estimate of an uncertain outcome; see [next-day decision](#next-day-pickup-decision).
- **Fulfillment node:** Site performing fulfillment; see [mental model](#3-the-core-mental-model).
- **Idle capacity:** Unused practical capacity; see [shared labor pool](#shared-labor-pool).
- **Incremental cost:** Cost difference between alternatives; see [incremental decision](#step-11--model-the-incremental-alternative).
- **Internal transfer:** Charge between organizational units; see [failure modes](#10-failure-modes-and-red-flags).
- **Net revenue:** Revenue after specified reductions under policy; see [revenue bridge](#step-4--assemble-the-revenue-bridge).
- **Opportunity cost:** Contribution from the best feasible alternative forgone; see [next-day decision](#next-day-pickup-decision).
- **Practical capacity:** Attainable resource capacity after normal allowances; see [shared labor pool](#shared-labor-pool).
- **Residual:** Cost not assigned to a channel; see [channel bridge](#channel-and-enterprise-bridge).
- **Return location:** Site accepting a returned item; see [definitions](#4-definitions-and-vocabulary).
- **Shadow price:** Modeled local value of one more constrained resource unit; see [mental model](#3-the-core-mental-model).
- **Shared pool:** Defined cost amount supporting multiple objects; see [shared pools](#step-7--define-shared-pools-and-their-boundaries).
- **Sunk cost:** Past cost unaffected by the choice; see [definitions](#4-definitions-and-vocabulary).

## 16. Sources

1. IFRS Foundation, [IFRS 15 Revenue from Contracts with Customers](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-15-revenue-from-contracts-with-customers/). Official standard overview, including scope and access to standard materials. Consult the applicable authoritative version for rights-of-return accounting.
2. IFRS Foundation, [IAS 2 Inventories](https://www.ifrs.org/issued-standards/list-of-standards/ias-2-inventories/). Official overview of inventory accounting; the management allocation example is not prescribed by this page.
3. Financial Accounting Standards Board, [Accounting Standards Codification](https://asc.fasb.org/). Authoritative U.S. GAAP source; consult Topic 606 and the applicable inventory guidance for the actual facts.
4. Walmart Inc., [Annual Reports and Proxies](https://stock.walmart.com/financials/annual-reports-and-proxies/default.aspx). Primary company disclosures for its reporting periods; not a source for the fictional cost rates.
5. Target Corporation, [Annual Reports](https://investors.target.com/financial-information/annual-reports). Primary company disclosures; review the specific annual report and definitions used.
6. Robert S. Kaplan and Steven R. Anderson, *Time-Driven Activity-Based Costing*, Harvard Business Review, November 2004, [article](https://hbr.org/2004/11/time-driven-activity-based-costing). Management-costing discussion of time equations and practical capacity; a method, not an accounting standard.

Source pages and standards can change; identify the applicable reporting period and version before using them for an actual accounting conclusion.

## 17. Links

- [Module 23 index](_index.md)
- [Module 23 survey](../../../course/23_omnichannel_fulfillment_returns.md)
- [Order routing and distributed order management](03_order-routing-dom.md)
- [Ship-from-store economics](04_ship-from-store.md)
- [BOPIS and curbside](05_bopis-curbside.md)
- [Returns networks](07_returns-networks.md)
- [Disposition economics](08_disposition-economics.md)
- The module survey contains the applicable lab material and assessment context.
