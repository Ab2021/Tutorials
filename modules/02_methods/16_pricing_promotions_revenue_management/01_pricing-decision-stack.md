# The Pricing Decision Stack

> Module 16 — Pricing, Promotions, and Revenue Management  
> Topic 01 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 04, 05, and 07; basic contribution-margin arithmetic  
> Version 2026.1

## 1. Why this topic exists

A supplier and a retailer can both say an item sells for $20 while describing different transactions.
The supplier's list price may be $20 per unit before customer allowances.
The retailer's shelf price may also be $20 per unit, including its own margin and perhaps tax.
A coupon might bring the consumer's payment down to $18 without changing the supplier's invoice.
A late rebate can reduce the supplier's receipts months after the shipment.
All of those numbers can be correct and still be useless if their owners and timing are not named.

The commercial director needs a price decision before the next customer negotiation.
Finance needs to understand the margin after discounts, rebates, logistics and returns.
The demand planner needs a credible volume scenario to reserve capacity and inventory.
The retailer needs sufficient time to load an authorized offer into its point-of-sale system.
Legal and privacy specialists need to see the mechanics before a personalized offer goes live.
The analyst's job is to connect these requests without treating every price lever as interchangeable.

A list price is a durable reference for an item and market.
A trade term changes what an intermediary pays or earns for specified performance.
A markdown changes a selling price to help clear an exposure over a finite horizon.
A personalized offer is an eligibility-dependent consumer incentive or price.
Each lever has a different decision horizon, customer, data grain and owner.
A discount approved for a one-week event is not permission to lower a year-round list price.

A profit waterfall makes that distinction operational.
It starts with a specified quantity at a specified reference price and subtracts mutually exclusive leakage and relevant costs.
It should reconcile to invoices, credits, accrued obligations and operational costs.
A waterfall is descriptive unless a credible counterfactual is added.
It tells you what happened to the observed transactions; it does not prove that an offer caused the observed sales.

A pricing stack is therefore both an economic model and a control system.
The economic part asks whether the proposed intervention improves the chosen objective relative to an alternative.
The control part makes sure the right person approves the correct intervention, the price is implemented, and the financial result can be reconstructed.
Neither an optimization model nor a price file can substitute for that process.

**Business decision:** approve, revise, test or reject a proposed commercial action for a defined product, customer, geography and interval.
**Costly error:** approving a gross-margin improvement while missing a rebate, fulfillment expense or transfer of sales from another period.
**Audience:** commercial, category, finance, supply, operations, data and control owners.
**Boundary:** this is an educational decision framework, not legal, accounting or tax advice.

## 2. Learning outcomes

After this topic you can:

1. Classify an intervention as list price, trade term, markdown or personalized offer and identify the commercial counterparty.
   Exercise 1 asks you to sort four interventions and identify the decision owner.
2. Draw the price-to-pocket contribution waterfall with consistent units, signs and timing.
   Exercise 2 asks you to compute the components and reconcile the result.
3. Compare a proposal with a counterfactual using incremental contribution, not revenue or gross discount alone.
   Exercise 3 asks you to quantify the decision and explain the causal assumption.
4. Design effective-date, eligibility and funding controls for overlapping offers.
   Exercise 4 asks you to diagnose a stackability problem.
5. Assign accountable owners and specify escalation when decision rights conflict.
   Exercise 5 asks you to draft a decision-rights matrix.
6. Explain why a historical relationship between price and sales may not predict a new intervention.
   Exercise 6 asks you to critique an observational estimate.
7. Design a limited experiment and a rollback trigger with customer and legal safeguards.
   Exercise 7 asks you to build a test plan.
8. Explain the difference among forecast, plan, target and commitment and use them correctly.
   Exercise 8 asks you to label a planning packet.

These outcomes require both calculation and judgment.
A correct total with incorrect ownership or an unapproved customer-data use is not a complete recommendation.

## 3. The core mental model

```text
Customer need + product value + alternatives + costs + inventory
                           |
                           v
              Define decision and counterparty
                           |
       +-------------------+--------------------+-------------------+
       |                   |                    |                   |
  LIST PRICE          TRADE TERM            MARKDOWN        PERSONAL OFFER
  reference level     business contract     finite stock     eligibility/benefit
       |                   |                    |                   |
       +-------------------+--------------------+-------------------+
                           |
                           v
          Stack rules, timing, funding and tax treatment
                           |
                           v
  Invoice -> credits/accruals -> net revenue -> variable cost -> contribution
                           |
                           v
  Compare with no-action counterfactual; constrain inventory and risk
                           |
                           v
  Accountable approval -> publish -> observe -> reconcile -> learn/rollback
```

The diagram begins with a business problem rather than a price number.
The same observed price could result from several decisions, and the repair depends on which one is responsible.
For example, a lower invoice can arise from a new list, a customer allowance or an incorrect off-invoice deduction.
The first calls for price governance; the second for contract verification; the third for correction and recovery.

The stack is not a universal chronological order in a checkout engine.
Specific contracts define whether discounts apply sequentially, to a common base or only after a threshold.
The diagram instead separates decisions and shows that their effects must be brought into one reconciled economic view.
A list price may be set by a manufacturer while a retailer independently sets its consumer price.
A retailer's markdown is not automatically a manufacturer discount; the manufacturer bears it only if a contract or credit transfers that cost.
A coupon redeemed by a shopper may be funded by either party or a third party.
Its economic treatment depends on the actual funding and settlement terms.

There are two distinct questions at the bottom of the diagram.
First, what contribution did the observed transactions generate under an agreed management accounting definition?
Second, what additional contribution did the proposed action cause compared with what otherwise would have occurred?
The first is a reconciliation question; the second is a counterfactual question.
A clean ledger does not identify demand response, and a good test does not excuse an unreconciled ledger.

Time links the layers.
List prices may remain effective for months; a one-off allowance may end in a week.
A markdown could remove inventory now but sacrifice later full-price sales.
A personalized coupon could pull a purchase from next week rather than create one.
A rebate earned in one quarter may be invoiced or settled in the next.
Align economic performance to the earned period and cash performance to settlement, while preserving both dates.

Profit is not a single universal line in the waterfall.
Gross margin, contribution after variable fulfillment, operating profit and cash flow differ.
For decisions here, contribution means net revenue less avoidable product and transaction-variable costs, unless explicitly stated otherwise.
Fixed investment, shared overhead and taxes may matter but must not be smuggled into a per-unit variable rate without justification.
The example later reports contribution, not statutory net income.

A disciplined recommendation specifies the intervention, alternative, objective, constraints, evidence, owner, execution and review.
It also specifies what would change the decision.
When the evidence is too weak, the appropriate output can be a bounded test instead of a price rollout.

## 4. Definitions and vocabulary

| Term | Definition and unit or key | Frequent misuse |
|---|---|---|
| Stock-keeping unit (SKU) | Internal sellable item identifier; map to pack and unit of measure | Equating a case with an each |
| List price | Published or contracted reference amount per unit for a market and interval | Assuming every customer pays it |
| Shelf price | Retailer's displayed consumer amount per item | Treating it as the manufacturer's invoice price |
| Transaction price | Amount actually charged for a specific sale, with tax treatment identified | Using an annual list as transaction evidence |
| Gross sales at list | Units times list price, before deductions; currency | Calling the hypothetical reference amount cash collected |
| Off-invoice allowance | Contractual reduction on the invoice; currency per unit or percentage of specified base | Subtracting it again as a later rebate |
| Bill-back | Claim or credit after invoice for a specified earned allowance | Assuming a claim has been earned merely because it was submitted |
| Rebate accrual | Estimated earned but unpaid obligation at the reporting date | Treating accrual and settlement as two expenses |
| Trade term | Contractual price, allowance, fee or condition between commercial counterparties | Assuming a consumer coupon is necessarily a supplier term |
| Price waterfall | Reconciliation from defined starting reference to a defined net result | Mixing per-unit and total values |
| Pocket price | Realized price after agreed deductions, under an explicitly defined company policy | Assuming it always includes freight and cost of goods |
| Net revenue | Revenue after defined commercial deductions, under the applicable accounting policy | Equating it automatically to cash received |
| Contribution | Net revenue less specified avoidable costs for a decision; currency | Calling it audited operating profit |
| Incremental contribution | Difference in contribution between action and counterfactual over a consistent boundary | Treating all promoted sales as incremental |
| Baseline | Estimate of outcomes without the action, for the same units and horizon | Assuming last year's actual is necessarily a causal baseline |
| Cannibalization | Sales or contribution displaced from other products, channels or locations | Counting shifted volume as wholly new business |
| Pull-forward | Purchases shifted from a later period into the event | Concluding an event created all of its observed uplift |
| Markdown | Reduction in a selling price, commonly for clearance; percentage or currency per unit | Confusing a retailer markdown with a manufacturer-funded credit |
| Salvage value | Net recoverable value per unsold unit at the end of the decision horizon | Ignoring disposal and handling charges |
| Personalized offer | An incentive or price targeted according to person- or account-level eligibility | Assuming that every segmented campaign charges an individually optimized price |
| Eligibility | Conditions determining who may receive or redeem an offer | Using a model score without a documented legal basis and policy |
| Redemption | A qualifying offer is applied to a transaction; count or currency | Counting issued coupons as cost incurred |
| Funding | Party contractually responsible for the incentive and associated costs | Assigning the entire offer to both supplier and retailer |
| Stackability | Rules specifying whether and how multiple incentives combine | Adding percentages that apply to different bases |
| Effective interval | Start-inclusive and end-exclusive period in a stated time zone | Loading ambiguous end-of-day dates |
| Forecast | Conditional prediction with uncertainty; units per time bucket | Treating it as a promised order |
| Plan | Coordinated intended course of action; units and dates | Presenting it as an unbiased forecast |
| Target | Desired performance level used for management | Using the target as empirical demand evidence |
| Commitment | Authorized obligation, such as a purchase order or capacity reservation | Treating a forecast as a legally binding commitment |
| Elasticity | Percentage demand change divided by percentage price change under a specified intervention | Transferring promotional response directly to list price |
| Causal lift | Difference between outcomes under action and no action for the same population and horizon | Equating correlation or raw before/after change with lift |
| Decision right | Authority to recommend, approve, execute or veto a defined action | Listing everyone as accountable |
| Floor | Lowest authorized economic or policy threshold for a defined scope | Treating a contribution floor as legal authorization |
| Guardrail | Constraint that prevents or limits a harmful action | Assuming a guardrail maximizes profit |
| Reference price | Comparison price perceived or displayed to the buyer | Assuming any historic price supports a lawful savings claim |
| Settlement | Payment or credit resolving an earned obligation | Booking settlement as an additional expense after accrual |
| Opportunity cost | Value of the best forgone alternative, such as later full-price demand | Omitting it because no invoice records it |
| Return | Reversal of a sale, subject to stated credit and cost recovery rules | Retaining revenue on returned items without an adjustment |
| Waterfall grain | Identifiers and period to which every line refers | Combining different customer contracts and claiming one rate |

## 5. The mechanics, step by step

### Step 1 — Frame the actual decision

**Input:** request, business objective, product, geography, channel and decision deadline.
**Action:** write a single decision question that states the lever and the party changing it.
**Output:** one accountable decision statement and a no-action alternative.

For example: should the supplier fund a one-week customer-specific rebate on a particular case pack?
That is different from asking whether the retailer should change its shelf price.
The supplier can propose terms but must not assume it controls an independent retailer's price.
Identify whether the requested action changes the published reference, contract net, customer checkout price or inventory clearance path.
Name the firm whose economics the analysis measures.

**Branch:** if more than one lever changes, define a combined intervention and estimate its combined effect.
Do not apply four separately estimated volume lifts and add them without evidence about interactions.
If the business has not defined an objective, obtain a decision rule before optimizing a price.
A sales target can be a constraint, but its existence does not imply that revenue maximization is the agreed objective.

### Step 2 — Establish the reference and decision horizon

**Input:** price books, contracts, stock position, future events and reporting calendar.
**Action:** specify what happens if the proposal is not implemented, including current terms and likely operating conditions.
**Output:** a documented counterfactual and time boundary.

The baseline should include normal customer terms, likely returns, usual fulfillment and pre-existing promotions.
If a current price is already expiring, doing nothing may mean reverting to a different valid price, not holding today's price forever.
Include a post-event window long enough to capture material pull-forward and returns.
For clearance, include residual inventory and its net salvage value at the end of the horizon.
For a durable list-price change, consider customer notice, effective dates and contracts that delay realization.

A baseline is an estimate, not a record of an observed alternate world.
A randomized holdout can support a causal estimate if the intervention and analysis satisfy the design assumptions.
Where randomization is unavailable, show the comparison group, adjustments and remaining selection risks.
A judgmental baseline is acceptable as a scenario, not as proof of causal impact.

### Step 3 — Specify the price architecture

**Input:** price book, customer agreements, promotional mechanics and order-to-cash rules.
**Action:** list every price component, its base, unit, counterparty and effective interval.
**Output:** an executable price stack and a mapping to ledger lines.

Record a percentage discount's base explicitly.
For example, 10% of list followed by 5% of the reduced price is a 14.5% total discount, not 15%.
Two percentage discounts each calculated on the same list base do add to 15%, assuming the contract permits both.
The sequence must match the contract and the checkout or invoicing engine.
For threshold rebates, record the accumulation period, qualifying units and whether the rate is marginal or retroactive.
A retroactive rebate at a threshold can produce a discontinuous increase in liability.

Specify whether an offer is a fixed currency amount, percentage or free good.
Specify whether it is per order, per item, per account or limited by inventory.
Record whether a consumer sees the price before taxes, delivery and mandatory fees.
Do not assume all offers may be combined merely because each is valid in isolation.

### Step 4 — Construct the observed waterfall

**Input:** invoice lines, credits, returns, accruals, units, cost records and funding agreements.
**Action:** reconcile from a reference value to realized net revenue and contribution.
**Output:** a signed bridge in both currency total and currency per unit.

Use a common grain such as SKU-pack × customer × channel × currency × earned week.
Keep a unique source identifier for every invoice, allowance, credit and accrual.
Group invoice-level rebates separately from later bill-backs.
Map actual payments against accrued liabilities rather than deducting both.
Use negative signs for reductions and retain positive reversals when they occur.

A possible management bridge is:

```text
gross sales at list
  - off-invoice discounts
  = invoice sales before returns
  - returns and invoice corrections
  - earned rebates and promotional claims
  = net revenue under the stated policy
  - variable product cost
  - variable freight and handling
  - other incremental transaction-variable cost
  = contribution
```

This is a management model, not a substitute for revenue-recognition policy.
Some firms use a different definition of pocket price or classify freight differently.
Explain the definition rather than comparing labels across firms.
When no units remain after returns, report total currency only; a per-unit denominator would be undefined.

### Step 5 — Reconcile across systems and periods

**Input:** commercial event IDs, invoices, credit memos, general ledger, claims and settlement records.
**Action:** trace the reference, transaction, accrual and settlement to a common contract and earning period.
**Output:** reconciliation with documented differences and owners.

Separate an estimate of an earned liability from the cash settlement that extinguishes it.
An earned $1,000 rebate accrued this month and settled next month is one economic deduction, not $2,000.
When the eventual claim differs from the estimate, book the difference as a true-up under policy.
Keep settlement date for cash forecasting and earning date for performance analysis.
Do not allocate an aggregate claim to specific SKUs without documenting the basis and materiality.

Check units of measure, foreign exchange rate dates, tax inclusivity and returns policy.
An invoice in cases and a price book in eaches need an explicit pack conversion.
Never average unit prices across currencies without conversion at an agreed rate and date.
If actual costs are unavailable, use a labeled provisional estimate and assign a close date for updating it.
If the unexplained reconciliation difference is material, pause automated optimization until it is resolved.

### Step 6 — Build the decision comparison

**Input:** action and no-action demand scenarios, waterfall rates, incremental costs and operational constraints.
**Action:** calculate the contribution of each scenario on the same economic boundary and horizon.
**Output:** incremental contribution, sensitivity and any binding constraints.

Use total contribution, not just contribution per unit.
A discount may reduce contribution per unit while increasing total contribution if enough genuinely incremental units are sold.
The converse is also possible: a revenue increase can destroy contribution.
The full comparison includes the subsidy on baseline units that would have bought anyway.
It also includes any incremental fixed event fee, if avoidable because of the action.

When product cost is avoidable per incremental unit, include it for each incremental sale.
When an already-paid fixed cost will not change with the decision, do not charge it only to the action.
Where supply is scarce, the best alternative may be allocation to another customer, not a zero-sale baseline.
When displaced sales or inventory are material, extend the boundary to the relevant products and periods.
Do not combine a baseline from one market with costs from another without an explicit bridge.

### Step 7 — Quantify uncertainty and threshold conditions

**Input:** plausible demand, redemption, cost and claim ranges and their evidence.
**Action:** calculate break-even thresholds and compare low, central and high scenarios.
**Output:** a range and the observations that would trigger a different decision.

For an action with contribution per unit $m_a and baseline contribution per unit $m_b,
with baseline quantity $Q_b and incremental fixed action cost $F$,
the required action quantity is `(Q_b * m_b + F) / m_a`, provided $m_a is positive.
If `m_a` is zero or negative and the baseline plus fixed cost is positive, volume alone cannot meet this break-even target.
The formula assumes the same boundary, constant unit margins, no tier discontinuity and no displaced contribution.
Round a required unit count upward when units are indivisible.
Include lost contribution from another product or period in the numerator when it is not already included elsewhere.

A confidence interval for historical elasticity is not a guarantee for a new price mechanic.
The commercial team may know that a competitor is changing pack size or that a contract is about to expire.
Record such judgment and test its influence instead of silently replacing evidence with a target.

### Step 8 — Check feasibility and execution

**Input:** inventory, replenishment, shelf and online price deployment systems, funding and contracts.
**Action:** verify that the proposed price and quantity can be delivered within the authorized time and capacity.
**Output:** an implementation plan with owners, deadlines and stop conditions.

Available-to-promise is not the same as on-hand inventory.
Subtract inventory reserved for existing commitments and include receipts only when their arrival is credible.
If expected demand exceeds available stock, cap sales and consider what the stock would have earned elsewhere.
Check whether a customer or retailer must accept a trade term before any announcement.
Test the actual invoice and checkout paths, including returns, multiple coupons, time zones and tax treatment.

Record a unique price version and retain the old version for rollback.
Schedule publication and expiry with the responsible operator, not merely with the analyst.
Where the retailer controls shelf price, verify its acceptance and avoid asserting control over its independent decision.
A model recommendation is not an executed commercial agreement.

### Step 9 — Assign decision rights and controls

**Input:** approval policy, delegation of authority, contract and risk assessment.
**Action:** assign one accountable decision owner, with named recommenders, approvers and operators.
**Output:** signed decision record and exception route.

A possible allocation, subject to company policy, appears below.
Accountability for a price does not remove the need for legal, privacy or finance review when required.

| Decision | Recommends | Accountable decision owner | Required consultation | Executes |
|---|---|---|---|---|
| Supplier list price | Pricing | Commercial director | Finance, sales, supply, legal as needed | Price operations |
| Customer trade agreement | Account manager | Authorized commercial signatory | Finance, legal, customer operations | Contract and billing operations |
| Retail markdown | Retail category manager | Retail merchandise owner | Inventory, finance, supplier if funded | Retail price operations |
| Personalized offer | CRM or marketing analyst | Customer-offer owner | Privacy, legal, finance, security | Campaign operations |
| Rebate accrual | Sales operations | Finance controller | Account owner and contract administration | Accounting |
| Decision measurement | Analytics | Named business decision owner | Finance, data governance, experiment owner | Analytics and data operations |

The actual authorization matrix may differ by firm, value, jurisdiction and contract.
An analyst cannot approve their own exception merely because the model recommends it.
If two owners disagree, escalate to the designated commercial authority before deployment.
If a mandatory legal or privacy gate is not satisfied, commercial approval alone is insufficient.

### Step 10 — Launch, monitor, settle and learn

**Input:** approved decision record, versioned price and offer, and monitoring schedule.
**Action:** compare implemented behavior, sales, claims and customer experience with the plan.
**Output:** reconciled outcome, exception log and retained learning.

At launch, verify a sample of live transactions, not only a successful file upload.
Monitor eligible population, redemptions, price leakage, stockouts, returns and complaint rate.
Use the pre-agreed stop condition if the realized price or legal eligibility is wrong.
Do not repeatedly peek at a test and stop only when the result becomes favorable without a valid sequential design.
After settlement, reconcile actual claims and update any accrual true-up.
Report actuals and causal estimates separately.

A useful close-out has four dates: decision, effective start, earned end and final settlement.
It also identifies the next decision, the forecast error and the reason for any override.
A failed experiment can still improve a future decision if its design and limitations were retained.

### Worked decision-control prompts

The following prompts turn the procedure into specific operational checks.
Use the named evidence at the moment of approval rather than relying on a later forensic exercise.

**Control prompt 01 — List price.**

- Evidence: An annual price book lists $24 per each; the account buys cases of six.
- Risk: A case/each mismatch would multiply the apparent invoice by six.
- Action: Verify pack conversion in both the signed price schedule and invoice master.

**Control prompt 02 — List price.**

- Evidence: A customer contract fixes $22 per each until its renewal date.
- Risk: A new published price cannot be assumed to supersede that contract.
- Action: Segment eligible volume by contract expiry and obtain the authorized interpretation.

**Control prompt 03 — List price.**

- Evidence: A regional price change is scheduled for Monday in local time.
- Risk: A UTC-only timestamp can activate the wrong price during the overnight window.
- Action: Store the named time zone and test the first and last eligible transaction.

**Control prompt 04 — List price.**

- Evidence: The proposed price rise coincides with a new pack size.
- Risk: A nominal increase may mask a different price per gram.
- Action: Report both price per pack and a consistent physical-unit comparator.

**Control prompt 05 — List price.**

- Evidence: Finance provides standard cost from the previous financial year.
- Risk: A changed input cost may make the resulting margin stale.
- Action: Use the decision-relevant cost estimate and retain its effective date.

**Control prompt 06 — List price.**

- Evidence: Sales forecasts 8,000 units; management targets 10,000 units.
- Risk: Replacing forecast with target inflates the economic case.
- Action: Present both numbers, the conditions behind each and the decision sensitivity.

**Control prompt 07 — Trade allowance.**

- Evidence: A customer receives $0.70 off each invoice unit.
- Risk: An additional $0.70 claim for the same allowance doubles the deduction.
- Action: Match invoice discounts to claim IDs before recognizing bill-backs.

**Control prompt 08 — Trade allowance.**

- Evidence: A display allowance is payable only with verified in-store execution.
- Risk: A submitted claim may not prove the contractual condition was met.
- Action: Obtain proof of performance and record approval or dispute status.

**Control prompt 09 — Trade allowance.**

- Evidence: A 3% rebate applies to invoice sales after the off-invoice discount.
- Risk: Applying the percentage to list overstates the liability.
- Action: Calculate the base from the signed agreement and retain the calculation.

**Control prompt 10 — Trade allowance.**

- Evidence: An annual rebate changes retroactively at 50,000 units.
- Risk: The next order can change the allowance on all prior eligible units.
- Action: Model the full liability at each threshold, not only the next order.

**Control prompt 11 — Trade allowance.**

- Evidence: A credit memo arrives after the quarter in which the rebate was earned.
- Risk: Booking expense again at payment duplicates the earned accrual.
- Action: Clear the liability and recognize only the difference from the estimate.

**Control prompt 12 — Trade allowance.**

- Evidence: A lump-sum customer payment covers several product lines.
- Risk: An arbitrary SKU split can misstate product-level contribution.
- Action: Document the allocation basis and preserve the unallocated source total.

**Control prompt 13 — Markdown.**

- Evidence: A retailer holds 1,200 seasonal units and expects 300 full-price sales.
- Risk: A blanket markdown can sacrifice margin on units that would have sold anyway.
- Action: Compare a timed markdown with the expected full-price and salvage alternatives.

**Control prompt 14 — Markdown.**

- Evidence: The retailer plans a 25% reduction but the supplier funds none of it.
- Risk: A manufacturer model can wrongly charge itself for the retail margin sacrifice.
- Action: Analyze each counterparty separately and include only the agreed transfer.

**Control prompt 15 — Markdown.**

- Evidence: A supplier agrees to credit the retailer for 400 verified unsold units.
- Risk: Crediting all historical shipments overpays the contractual amount.
- Action: Verify eligible stock, date, ownership and claim evidence before crediting.

**Control prompt 16 — Markdown.**

- Evidence: A clearance event overlaps with a regular coupon campaign.
- Risk: A coupon may apply on the already reduced price and breach a floor.
- Action: Simulate the exact checkout sequence and approve or disable stacking.

**Control prompt 17 — Markdown.**

- Evidence: A perishable product has a disposal cost of $0.40 per remaining unit.
- Risk: Treating salvage as zero exaggerates the attractiveness of holding stock.
- Action: Use net salvage of minus $0.40 where the cost is avoidable.

**Control prompt 18 — Markdown.**

- Evidence: A chain has 100 units in one store and none in another.
- Risk: A chain-wide average conceals a local stockout or excess.
- Action: Analyze stock and feasible transfer at store and SKU level.

**Control prompt 19 — Personal offer.**

- Evidence: A customer receives a $2 coupon valid once on one eligible unit.
- Risk: Applying it once per item in a five-unit basket creates unintended exposure.
- Action: Test multiple-item baskets and enforce the per-account redemption limit.

**Control prompt 20 — Personal offer.**

- Evidence: A loyalty segment is selected using prior purchase history.
- Risk: The segmentation may use data outside the approved purpose or retention period.
- Action: Obtain privacy review of the data source, purpose, notice and retention.

**Control prompt 21 — Personal offer.**

- Evidence: A consumer qualifies for two 10% percentage offers.
- Risk: Sequential application gives a different price from adding both rates.
- Action: Define order of operations and test both the display and receipt.

**Control prompt 22 — Personal offer.**

- Evidence: A third party funds $1 of a $3 coupon and the retailer funds $2.
- Risk: Charging $3 to each funder doubles the economic cost.
- Action: Match a unique redemption ID to the agreed $1/$2 settlement split.

**Control prompt 23 — Personal offer.**

- Evidence: A campaign forecasts 20,000 redemptions against 12,000 available units.
- Risk: The revenue estimate assumes a volume that supply cannot fulfill.
- Action: Cap the eligible offer or reserve capacity and re-estimate the outcome.

**Control prompt 24 — Personal offer.**

- Evidence: A household has two eligible loyalty accounts.
- Risk: An account limit does not necessarily enforce a household limit.
- Action: Choose the intended identity scope and validate it lawfully before launch.

**Control prompt 25 — Governance.**

- Evidence: A price exception is approved in a chat but not in the price system.
- Risk: The operator cannot reliably distinguish authorization from discussion.
- Action: Link the signed approval to a versioned price record and effective dates.

**Control prompt 26 — Governance.**

- Evidence: A customer asks for the same rate given to a different customer.
- Risk: A copied term may have different obligations, costs and competition-law implications.
- Action: Review the actual agreement, economics and applicable law before signing.

**Control prompt 27 — Governance.**

- Evidence: The model uses a competitor quote supplied without provenance.
- Risk: The quote may be inaccurate, restricted or competitively sensitive.
- Action: Quarantine the input and obtain counsel-approved sourcing guidance.

**Control prompt 28 — Governance.**

- Evidence: An automated job publishes a price below the authorized floor.
- Risk: A correct recommendation can be altered by conversion or loading errors.
- Action: Enforce a deployment gate and stop the job on a failed validation.

**Control prompt 29 — Governance.**

- Evidence: An offer ends on a public holiday when operations are unavailable.
- Risk: The end date may pass without a monitored reversal or settlement owner.
- Action: Schedule and test automatic expiry and name an on-call escalation owner.

**Control prompt 30 — Governance.**

- Evidence: A campaign materially increases sales but also increases returns.
- Risk: An immediate sales-only report can overstate net contribution.
- Action: Hold a return window open and refresh the waterfall after credits settle.

**Control prompt 31 — Measurement.**

- Evidence: A promotion runs only in unusually strong stores.
- Risk: A raw comparison to other stores mixes treatment and store selection.
- Action: Randomize eligible stores or justify a comparable control and diagnostics.

**Control prompt 32 — Measurement.**

- Evidence: A discount runs alongside a prominent display.
- Risk: The total event effect does not identify the price-only effect.
- Action: Define the combined treatment or test price and display separately.

**Control prompt 33 — Measurement.**

- Evidence: Competitor prices changed during the same campaign.
- Risk: Before/after sales differences can reflect the external market change.
- Action: Use a contemporaneous control when credible and record market context.

**Control prompt 34 — Measurement.**

- Evidence: An offer shifts purchases from next week to this week.
- Risk: A one-week uplift overstates incremental volume over the cycle.
- Action: Measure the event and post-event window under a consistent baseline.

**Control prompt 35 — Measurement.**

- Evidence: The item sells out halfway through the event.
- Risk: Observed sales cap demand and can bias response estimates.
- Action: Record availability and analyze the stockout as a constraint, not zero demand.

**Control prompt 36 — Measurement.**

- Evidence: A promoted item gains units while a substitute loses units.
- Risk: SKU-only contribution can be positive while the category loses money.
- Action: Include the measured or scenario-based displaced contribution.

**Control prompt 37 — Supply.**

- Evidence: An offer is approved after the last replenishment order cutoff.
- Risk: The forecasted lift cannot be supplied in the event window.
- Action: Use confirmed lead times and available inventory in the decision.

**Control prompt 38 — Supply.**

- Evidence: The offer consumes capacity reserved for a contracted customer.
- Risk: The apparent new contribution ignores an existing commitment.
- Action: Check allocation and include the value and consequences of displacement.

**Control prompt 39 — Supply.**

- Evidence: A large order moves freight into an expensive expedited mode.
- Risk: A constant per-unit shipping cost misses a step change.
- Action: Use the relevant freight schedule and include the incremental charge.

**Control prompt 40 — Supply.**

- Evidence: The campaign encourages returns through a generous guarantee.
- Risk: Gross shipment volume overstates retained sales and may add handling costs.
- Action: Estimate net retained units and recoverable product value separately.

## 6. Worked example

### Situation and decision

A fictional packaged-food manufacturer, Northstar Foods, supplies a retailer with a single shelf-stable SKU.
The decision is whether to fund a one-week trade event in the retailer's eastern region.
The manufacturer controls its list and the offered trade terms; the retailer independently decides its shelf price.
The retailer has indicated it intends to support the event, but the supplier does not treat that indication as a binding price commitment.
All prices and costs below are in U.S. dollars (USD), excluding sales tax, per individual consumer unit unless stated otherwise.
The horizon includes the event week and a two-week post-event window.
The economic boundary is the manufacturer's contribution for this SKU, including an explicit estimate of lost post-event contribution.

**Forecast:** without the action, 8,000 units ship over the event week at existing terms.
**Plan:** if approved, the team plans the event and inventory allocation described below.
**Target:** the sales team wants 11,000 event-week units; this is not evidence of demand.
**Commitment:** the retailer has a purchase order for 7,000 units; the remaining demand is not committed.

The central scenario estimates 10,500 units in the action week.
This is an assumed demand scenario, not an established causal result.
Existing sellable stock and replenishment support up to 11,200 units after prior commitments.
The proposal is therefore feasible under this stated stock assumption.
An additional 300 units of ordinary post-event sales are expected to be pulled forward.
Those 300 units are included in the event's 10,500 units and would otherwise earn the baseline unit contribution later.
No other category cannibalization is assumed in the central scenario; this is a material uncertainty, not a known fact.

### Price and cost inputs

| Input | Baseline | Proposed action | Interpretation |
|---|---:|---:|---|
| List price | $20.00/unit | $20.00/unit | Unchanged supplier reference |
| Off-invoice allowance | $1.20/unit | $1.20/unit | Existing customer term |
| Earned normal rebate | $0.60/unit | $0.60/unit | Estimated earned obligation |
| Event bill-back | $0.00/unit | $1.00/unit | Supplier-funded, applies to every eligible event unit |
| Variable product cost | $10.40/unit | $10.40/unit | Avoidable per unit shipped |
| Variable freight and handling | $0.80/unit | $0.80/unit | Same for both scenarios |
| Additional event administration | $0 | $1,500/event | Avoidable action-only cost |
| Event-week volume | 8,000 units | 10,500 units | Forecast versus assumed action scenario |
| Post-event displaced volume | 0 units | 300 units | Action effect relative to baseline |

The $1.00 event bill-back is after invoice and is not included in the off-invoice allowance.
It is earned by each eligible unit and is not a percentage.
For simplicity, assume no returns, taxes, financing costs or claims disputes.
The normal rebate is an earned deduction, not a second invoice payment.
The administration charge is an action-only fixed cost for this event.

### Baseline waterfall

| Line | Calculation | USD total | USD/unit |
|---|---:|---:|---:|
| Gross sales at list | 8,000 × $20.00 | 160,000 | 20.00 |
| Off-invoice allowance | 8,000 × $1.20 | −9,600 | −1.20 |
| Invoice sales | 160,000 − 9,600 | 150,400 | 18.80 |
| Normal rebate accrual | 8,000 × $0.60 | −4,800 | −0.60 |
| Event bill-back | Not applicable | 0 | 0.00 |
| Net revenue | 150,400 − 4,800 | 145,600 | 18.20 |
| Variable product cost | 8,000 × $10.40 | −83,200 | −10.40 |
| Variable freight and handling | 8,000 × $0.80 | −6,400 | −0.80 |
| Event-week contribution | 145,600 − 83,200 − 6,400 | 56,000 | 7.00 |

The baseline event-week contribution is $56,000.
The post-event baseline includes ordinary sales, but the only post-event difference modeled here is the 300-unit displacement.
Common post-event contribution cancels from the comparison.

### Proposed event waterfall

| Line | Calculation | USD total | USD/unit |
|---|---:|---:|---:|
| Gross sales at list | 10,500 × $20.00 | 210,000 | 20.00 |
| Off-invoice allowance | 10,500 × $1.20 | −12,600 | −1.20 |
| Invoice sales | 210,000 − 12,600 | 197,400 | 18.80 |
| Normal rebate accrual | 10,500 × $0.60 | −6,300 | −0.60 |
| Event bill-back accrual | 10,500 × $1.00 | −10,500 | −1.00 |
| Net revenue | 197,400 − 6,300 − 10,500 | 180,600 | 17.20 |
| Variable product cost | 10,500 × $10.40 | −109,200 | −10.40 |
| Variable freight and handling | 10,500 × $0.80 | −8,400 | −0.80 |
| Event-week contribution before event fee | 180,600 − 109,200 − 8,400 | 63,000 | 6.00 |
| Event administration | Action-only fee | −1,500 | Not a constant unit rate |
| Event-week contribution after event fee | 63,000 − 1,500 | 61,500 | 5.8571 |

The supplier's invoice is $197,400; its modeled net revenue is $180,600.
The difference is $16,800 of earned normal rebate and event bill-back, not an additional invoice discount.
The rebate and bill-back are accrued when earned and cleared upon settlement.
If the final verified claim is $10,200 instead of $10,500, the $300 favorable true-up corrects the accrual; it is not a second $10,200 deduction.

### Incremental comparison

The event-week action contribution after the fee is $61,500.
Subtract the no-action event-week contribution of $56,000 to get $5,500 before the post-event effect.
The 300 pulled-forward units would otherwise have earned $7.00 each in the post-event window.
Their lost future contribution is 300 × $7.00 = $2,100.
Therefore estimated incremental contribution over the chosen boundary is $61,500 − $56,000 − $2,100 = **$3,400**.
This is a forecast-based decision estimate, not a measured causal effect.

The same answer can be checked by decomposing the action:

1. The 8,000 baseline event-week units receive a $1.00 event allowance, costing $8,000.
2. The 2,500 extra event-week units each contribute $6.00, adding $15,000.
3. The event administration costs $1,500.
4. The 300 units moved from the future lose $2,100 of baseline contribution there.
5. Net impact is −$8,000 + $15,000 − $1,500 − $2,100 = **$3,400**.

This decomposition is consistent because the extra event-week units include the 300 shifted units.
Do not subtract their event-week contribution a second time.

### Break-even and sensitivity

Holding the 300-unit future displacement fixed, action quantity must satisfy:

```text
$6.00 × action units − $1,500 − $2,100 >= $56,000
break-even action units >= $59,600 / $6.00 = 9,933.333...
minimum whole-unit quantity = 9,934 units
```

This break-even condition includes the $1,500 event fee and the $2,100 displaced future contribution.
At 9,933 units the result is −$2; at 9,934 units it is +$4.
The threshold is conditional on the fixed 300-unit displacement and constant rates.
If pull-forward changes with volume, the threshold must be recomputed from that relationship.

| Action-week units | Event-week contribution after fee | Less baseline and displaced future contribution | Incremental contribution |
|---:|---:|---:|---:|
| 9,000 | $52,500 | $58,100 | −$5,600 |
| 9,934 | $58,104 | $58,100 | $4 |
| 10,500 | $61,500 | $58,100 | $3,400 |
| 11,200 | $65,700 | $58,100 | $7,600 |

The last scenario uses the stated inventory limit; no demand above 11,200 units is assumed fulfilled.
A further $3,400 of unmodeled lost contribution would eliminate the central scenario's gain.
For example, a loss of $1.70 per unit on 2,000 displaced substitute units would equal $3,400.
This is a sensitivity illustration, not an additional loss already included in the central estimate.

### Decision and ownership

The commercial director can authorize a limited test, conditional on the retailer's agreement and contract review.
Finance approves the accrual method and reconciles claims; supply confirms the 11,200-unit limit.
Account operations documents the bill-back and proof of performance.
Analytics measures event and post-event sales against a credible control and reports uncertainty.
The retailer owns its independent shelf-price decision.
Stop deployment if the price file applies the event allowance twice, stock falls below committed quantities, or the authorized terms are not accepted.
Do not declare the $3,400 a realized gain until actuals and a defensible counterfactual are available.

## 7. Code, query, or calculation

The following Python 3 program reproduces the worked waterfall, impact and conditional break-even.
It uses only the standard library and `Decimal` for currency arithmetic.
Copy it to a file such as `pricing_stack.py` and run `python pricing_stack.py`.
Input quantities are whole units; rates are USD per unit and the event fee is USD per event.
The program is deliberately a scenario calculator, not an elasticity estimator or accounting system.
It assumes constant rates, one eligible SKU and a fixed number of displaced post-event units.

```python
from decimal import Decimal, ROUND_CEILING

D = Decimal
LIST = D('20.00')
OFF_INVOICE = D('1.20')
NORMAL_REBATE = D('0.60')
PRODUCT_COST = D('10.40')
FREIGHT = D('0.80')
EVENT_REBATE = D('1.00')
EVENT_FEE = D('1500')
BASE_UNITS = 8000
ACTION_UNITS = 10500
DISPLACED_UNITS = 300


def waterfall(units, event_rebate=D('0')):
    q = D(units)
    gross = q * LIST
    invoice = gross - q * OFF_INVOICE
    net_revenue = invoice - q * NORMAL_REBATE - q * event_rebate
    product = q * PRODUCT_COST
    freight = q * FREIGHT
    contribution = net_revenue - product - freight
    return {
        'gross': gross,
        'invoice': invoice,
        'net_revenue': net_revenue,
        'product': product,
        'freight': freight,
        'contribution': contribution,
    }


def dollars(amount):
    return f'${amount:,.2f}'


base = waterfall(BASE_UNITS)
action = waterfall(ACTION_UNITS, EVENT_REBATE)
base_margin = base['contribution'] / D(BASE_UNITS)
action_margin = action['contribution'] / D(ACTION_UNITS)
displaced = D(DISPLACED_UNITS) * base_margin
increment = (action['contribution'] - EVENT_FEE
             - base['contribution'] - displaced)
if action_margin <= 0:
    raise ValueError('Positive action margin required for this threshold')
required = ((base['contribution'] + EVENT_FEE + displaced)
            / action_margin).to_integral_value(rounding=ROUND_CEILING)

for label, values in [('Baseline', base), ('Action', action)]:
    print(label)
    for key in ('gross', 'invoice', 'net_revenue',
                'product', 'freight', 'contribution'):
        print(f'  {key}: {dollars(values[key])}')
print(f'Action-only fee: {dollars(EVENT_FEE)}')
print(f'Displaced contribution: {dollars(displaced)}')
print(f'Incremental contribution: {dollars(increment)}')
print(f'Break-even action units: {required}')
```

Expected output:

```text
Baseline
  gross: $160,000.00
  invoice: $150,400.00
  net_revenue: $145,600.00
  product: $83,200.00
  freight: $6,400.00
  contribution: $56,000.00
Action
  gross: $210,000.00
  invoice: $197,400.00
  net_revenue: $180,600.00
  product: $109,200.00
  freight: $8,400.00
  contribution: $63,000.00
Action-only fee: $1,500.00
Displaced contribution: $2,100.00
Incremental contribution: $3,400.00
Break-even action units: 9934
```

The cost values printed by the program are positive expenses; the tables display them as negative deductions.
The contribution result is before the action-only event fee in each waterfall.
The code subtracts the fee once in the incremental comparison.
It does not model taxes, return credits, multi-tier rebates, changes in displaced units or randomized-test uncertainty.
For an actual production decision, retain contract identifiers, effective intervals, provenance and authorized versions outside this small illustration.

## 8. Data, standards, and regulatory context

### Minimum decision data

| Object | Recommended grain | Required fields | Main control |
|---|---|---|---|
| Product master | SKU × pack × effective interval | Internal ID, GTIN where available, units per case, unit of measure | Validate pack conversion |
| Price book | SKU × market × channel × interval | List amount, currency, validity, owner, version | Prevent overlapping active records |
| Contract | Customer × product scope × interval | Allowance, qualifying conditions, signature, amendment | Match to approved version |
| Invoice line | Invoice × line | Quantity, amount, price, tax, unit of measure | Reconcile to ledger |
| Credit or return | Credit × original line | Reason, quantity, earned and posting dates | Avoid duplicates and apply return policy |
| Rebate accrual | Contract × earning period × basis | Estimated liability, calculation base, revision | Clear rather than re-expense on payment |
| Settlement | Claim × payment or credit | Claim ID, gross amount, funding split, date | Match to accrual and true-up |
| Cost | SKU × location × interval | Variable product, freight, handling, source | Distinguish avoidable from allocated cost |
| Inventory | SKU × location × date | On hand, reserved, receipts, expiry | Separate available stock from commitments |
| Offer | Campaign × version × eligibility rule | Dates, benefit, cap, stack rules, funding | Test actual checkout path |
| Experiment | Unit × treatment assignment × interval | Assignment, exposure, outcomes, consent controls | Preserve assignment and avoid leakage |
| Market context | Market × date × source | Public competitor observations, availability, provenance | Review legal source and permitted use |

The contract and transaction stores need stable identifiers across systems.
A valid contract number alone does not prove that a particular invoice line qualifies.
Match on product, counterparty, date, quantity and applicable conditions.
Protect personal identifiers by minimizing collection and restricting access to the roles that need them.
Aggregate reporting does not automatically make an underlying personal-data use permissible.

### Standards and conventions

GS1's Global Trade Item Number (GTIN) identifies a trade item, including relevant packaging distinctions.
It helps match supplier, retailer and transaction records; it does not specify contract pricing or guarantee a correct unit conversion.
Use a maintained product master and check the unit of measure separately.

ISO 4217 defines currency codes such as USD and EUR.
A code identifies the currency but does not choose an exchange rate, conversion date or financial policy.
Specify whether reporting uses transaction-date, budget or closing rates and keep the source amount.

ISO 8601 supplies a standard way to represent dates and times.
An unambiguous timestamp still needs a business time-zone rule and a clear interval convention.
Here, start-inclusive/end-exclusive intervals avoid double eligibility at a boundary.
That is a design convention, not a claim that ISO 8601 mandates it.

The International Accounting Standards Board's IFRS 15 addresses revenue from contracts with customers for entities applying that standard.
The Financial Accounting Standards Board's Accounting Standards Codification Topic 606 is the corresponding U.S. revenue-recognition framework.
The classification and timing of variable consideration and consideration payable to a customer can be fact-specific.
Finance must apply the relevant framework and entity policy; the management waterfall is not an accounting conclusion.

### Consumer, competition and privacy controls

The EU Price Indication Directive, as amended by Directive (EU) 2019/2161, contains rules for prior-price indications when price reductions are announced.
For goods covered by the rule, Article 6a generally refers to the lowest price applied during a period not shorter than 30 days before the reduction.
Member-state implementation and exceptions, including rules for perishable goods and progressive reductions, need local review.
A private business-to-business allowance is not automatically the same thing as a public consumer price-reduction announcement.
The European Commission's interpretive notice is guidance; the directive and applicable national law are the binding sources.

In the United States, the Federal Trade Commission (FTC) enforces laws against unfair or deceptive practices, including deceptive price claims.
The FTC's Guides Against Deceptive Pricing describe principles for former-price and comparative-price advertising.
Guides are administrative guidance, not a substitute for the underlying law and the applicable facts.
Substantiate a reference-price claim and review the complete disclosure, not only the numerical discount.

Competition law is jurisdiction- and fact-specific.
Horizontal agreements among competitors to fix prices can raise serious antitrust concerns.
Do not exchange non-public forward prices or accept a shared vendor's data without an authorized legal assessment.
A retailer's independent pricing decision must not be silently treated as the supplier's decision right.
Not every algorithm, common vendor or parallel price is itself proof of an unlawful agreement.
Escalate proposed information sharing, resale-price restrictions and sensitive contractual terms to counsel.

Personalized offers may involve personal data and profiling.
The EU General Data Protection Regulation (GDPR), where applicable, contains requirements including lawful processing and transparency.
The appropriate legal basis and any additional rules depend on the facts, jurisdiction and method.
Consent is not universally the only possible legal basis, nor does obtaining it automatically cure other defects.
Review purpose limitation, retention, notice, access controls and applicable automated-decision rules with privacy counsel.
A segment identifier or postal code can still carry privacy or fairness risk when it acts as a proxy.
Assess material differences in outcomes, document legitimate business rationale and provide an escalation or correction path.

These regulatory descriptions are educational and reflect cited source texts, not a current legal opinion as of a particular deployment date.
Before launch, the responsible legal owner must check current law, local implementation, contracts and the actual mechanic.

## 9. Industry practice and case evidence

### Case 1: J.C. Penney's pricing reset

J.C. Penney's 2012 annual report describes the company's introduction of a new pricing strategy and reports a large decline in comparable-store sales for fiscal 2012.
The annual report is a company disclosure; its results describe the company's overall performance rather than an isolated randomized price experiment.
The change accompanied other merchandising and organizational changes.
A before/after sales decline therefore cannot identify the causal effect of removing coupons by itself.

**Decision lesson:** changing the entire price architecture can change how customers interpret value, not merely the arithmetic on a receipt.
**Inference:** a staged experiment with measured customer response could have reduced uncertainty before a full rollout.
**Not established by the case:** that coupons always increase profit, or that every everyday-low-price strategy fails.
The distinction matters when an analyst uses a famous case as evidence for a different product, retailer or customer segment.

### Case 2: Zara's operating model and markdown exposure

Ferdows, Lewis and Machuca describe Zara's rapid-response supply chain in their Harvard Business Review article, “Rapid-Fire Fulfillment.”
The article describes frequent deliveries, short lead times and tightly coordinated design, production and distribution.
It is a published business case, not a randomized estimate of the benefit of a particular markdown policy.

**Decision lesson:** a markdown decision is downstream of the buying and replenishment decisions that create inventory exposure.
**Inference:** smaller, faster commitments can reduce the amount of inventory that must be cleared if execution and demand conditions support them.
**Not established by the case:** a universal markdown percentage or a guarantee that fast replenishment improves all categories.
A long-lead-time producer cannot simply assume the same operating constraints as a vertically coordinated fashion retailer.

### Case 3: Airlines and perishable capacity

The revenue-management literature describes airlines allocating limited seats across booking classes as departure approaches.
A seat not sold before departure generally has little value after the flight leaves; a seat sold cheaply cannot also be sold later at a high fare.
This is a capacity-allocation problem, not merely a list-price change.
Talluri and van Ryzin describe the analytical foundations and practical complexity of revenue management.

**Decision lesson:** the opportunity cost of consuming inventory can be more important than unit production cost.
**Inference:** an analogous opportunity-cost view can help a manufacturer allocate constrained promotional inventory.
**Not established by the analogy:** that airline fare classes or booking curves transfer unchanged to a grocery promotion.
The data, segment fences, cancellation rights and customer expectations are different.

### Translating practice into a decision record

A mature commercial process commonly combines a price book, a trade-promotion workflow, an accrual ledger and a review meeting.
The organizational names vary; the important feature is a trace from authorized terms to observed outcomes.
A tool can store a contract, but a human owner still must resolve conflicting versions and determine whether a claim was earned.
A demand model can estimate a range, but the business owner must select the objective and accept the uncertainty.

A company-wide “pocket price” may be useful for monitoring leakage.
Its denominator and deductions must be identical before comparing customers or periods.
Freight, payment terms, taxes, returns and service levels may differ for legitimate reasons.
A low pocket price is therefore a prompt for review, not proof of an error or an unauthorized concession.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| List treated as realized price | Invoice averages below book price | Contract deductions omitted | Margin overstated | Reconcile invoice to contract |
| Double-counted claim | Same event ID on invoice and bill-back | Two systems recognize one allowance | Contribution understated | Match claims and deductions |
| Missing accrual | Claims arrive after period close | Earned liability not estimated | Current margin overstated | Accrue and true up by earning period |
| Settlement expensed twice | Paid claim and prior accrual both in cost | Liability not cleared | Margin understated | Reconcile settlement against liability |
| Mixed units | Price per each joined to case quantity | Pack conversion absent | Sales and discount miscalculated | Versioned unit conversion |
| Wrong discount order | Receipt differs from approved rate | Sequential and additive bases confused | Unapproved realized price | Test checkout and invoice scenarios |
| Unauthorized stacking | Multiple coupons accepted together | No interaction rule | Unexpected margin loss | Explicit precedence and caps |
| Gross sales mistaken for lift | Big event-week spike | Baseline units receive subsidy | Event can lose contribution | Counterfactual and subsidy calculation |
| Pull-forward omitted | Sales decline after event | Customers advance purchases | Net uplift overstated | Include post-event window |
| Cannibalization omitted | Substitute sales fall | Demand shifts within portfolio | Category profit overstated | Include affected products |
| Inventory cap ignored | Stockout during event | Forecast exceeds available stock | Lost sales and distorted evaluation | Validate available-to-promise |
| Confounded price test | Display and price change together | Treatment bundles multiple actions | Price-only effect unsupported | Test components or label bundle |
| Inappropriate historical elasticity | Clearance uses regular-price model | Customer context differs | Wrong markdown | Match evidence to intervention |
| Poor reference-price claim | “Was” price lacks history | Comparison unsubstantiated | Consumer harm and legal exposure | Retain price history; counsel review |
| Privacy breach | Offer audience lacks purpose record | Unapproved personal-data processing | Customer harm and enforcement | Privacy gate and data minimization |
| Fairness proxy | Offer outcomes differ by geography | Targeting variable proxies protected traits | Unequal impact and loss of trust | Review features and outcomes |
| Competitor-data exposure | Vendor offers future competitor prices | Sensitive information may enter decisions | Competition-law risk | Provenance controls and counsel review |
| No decision owner | Several teams assume another approved | Diffuse accountability | Unauthorized execution | One accountable owner per decision |
| Forecast treated as order | Unconfirmed demand drives purchase | Forecast and commitment conflated | Excess stock or missed commitments | Label and reconcile each planning number |
| Currency mixing | Multiple currency symbols in one total | Missing conversion policy | Nonsensical waterfall | Convert at documented rate and date |
| Late price rollback | Offer remains live after end | Time-zone or scheduling error | Unbudgeted discount | Automated expiry and live checks |
| Conflicting retailer role | Supplier claims to set retail shelf price | Counterparty autonomy ignored | Contract and legal risk | Separate supplier and retailer rights |
| Excluding opportunity cost | Cheap orders consume scarce stock | Alternative contribution ignored | Lower total business contribution | Evaluate feasible alternative allocation |
| Reporting revenue as profit | Revenue rises while contribution falls | Costs and subsidies omitted | Bad approval | Compare incremental contribution |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Event unit cost is constant | Makes per-unit contribution linear | Overtime, freight tier or capacity step | Obtain cost schedule and model tiers |
| Baseline volume is 8,000 units | Defines the no-action comparison | Seasonality, distribution changes or market shocks | Compare matched controls and forecast history |
| Action volume is 10,500 units | Quantifies central scenario | Lower retailer support or stockout | Run scenario range and measure actual availability |
| 300 future units are displaced | Captures one pull-forward estimate | Stockpiling lasts longer or not at all | Measure a longer post-event window |
| Other cannibalization is zero | Keeps central boundary tractable | Substitute or channel sales fall | Include category and cross-channel analysis |
| All 10,500 units qualify | Determines total bill-back | Returns, exclusions or incomplete evidence | Match qualifying unit records and contract |
| Allowances are mutually exclusive | Prevents duplicate deductions | Event credit is already in invoice price | Reconcile invoice, claim and contract IDs |
| No returns or tax in example | Keeps arithmetic focused | Actual return rates or tax treatment change | Add return timing, credits and applicable tax policy |
| Supply supports 11,200 units | Makes central event feasible | Delayed receipts or earlier commitments | Check dated available-to-promise and reservations |
| Event fee is avoidable | Makes fee incremental | Fee is already committed regardless of event | Verify cancellation rights and spending approval |
| Same currency and pack | Allows direct arithmetic | Mixed invoices, foreign exchange or pack changes | Convert using documented rate and pack master |
| No strategic long-run effects | Restricts analysis to stated horizon | Reference prices or relationships change | Add longer-run scenarios and monitor repeat behavior |
| Settlement equals initial accrual | Simplifies close-out | Claims differ or are disputed | Compare settlement to accrual and true up |
| Price response generalizes | Supports forecast-based proposal | Different geography, audience or mechanic | Use comparable experiments or bounded pilot |

The break-even calculation is a conditional financial threshold, not a probability of success.
Even if a point estimate is above break-even, uncertainty, policy constraints or strategic objectives may justify delaying the launch.
The best available decision may be to test a smaller, pre-approved exposure while preserving a holdout.

## 12. Exercises

All amounts are USD, excluding sales tax, unless a question explicitly says otherwise.
Use contribution definitions and the economic boundary in the question.
Show calculations; avoid relying on a headline percentage alone.

### Exercise 1 — Classify the lever

A. A manufacturer changes its published reference from $12.00 to $12.50 per unit for next quarter.
B. The manufacturer pays a retailer $0.40 per verified unit for a display.
C. The retailer cuts a seasonal item's own shelf price to clear stock.
D. A shopper receives a one-use $3.00 loyalty coupon.

**Deliverable:** classify each action and identify the decision owner and counterparty.
**Acceptance criteria:** distinguish supplier and retailer authority; identify the coupon's unknown funder rather than inventing one.
**Time:** 10 minutes.

### Exercise 2 — Build a waterfall

A supplier ships 2,000 units at list $15.00 per unit.
It gives $1.00 per unit off invoice and accrues a separate $0.50 per unit rebate.
Variable product cost is $8.00 per unit and freight is $0.75 per unit.
No returns, fees or other costs apply.

**Deliverable:** gross sales, invoice sales, net revenue and contribution, in total USD and USD per unit.
**Acceptance criteria:** subtract each allowance once; show the denominator and reconcile all totals.
**Time:** 15 minutes.

### Exercise 3 — Compare an event with baseline

Baseline sales are 1,000 units at $5.00 contribution per unit.
The event yields $4.00 contribution per unit, costs an additional $400 in fixed event fees, and is expected to sell 1,500 units.
A further $300 of contribution is lost after the event from pull-forward.
No other effects apply.

**Deliverable:** incremental contribution and the minimum whole-unit event volume that breaks even.
**Acceptance criteria:** include the fee and post-event loss exactly once and state the constant-margin assumption.
**Time:** 15 minutes.

### Exercise 4 — Diagnose offer stacking

An item lists at $100.00.
Two eligible 10% offers can either both apply to the original price or apply sequentially.
The policy owner has not specified the intended method.

**Deliverable:** calculate the final price under each interpretation and propose a deployment control.
**Acceptance criteria:** distinguish the bases, calculate both prices and require a signed rule plus checkout test.
**Time:** 10 minutes.

### Exercise 5 — Assign decision rights

A retailer proposes a clearance price, requests supplier funding and plans a personalized loyalty coupon.
Inventory is constrained and the supplier contract has not yet been amended.

**Deliverable:** a decision-rights table for the retail markdown, supplier funding, coupon, inventory allocation and financial accrual.
**Acceptance criteria:** one accountable owner per decision; distinguish the two firms; include required control consultations and an escalation route.
**Time:** 25 minutes.

### Exercise 6 — Critique causal evidence

A team reports that 12 promoted stores sold 30% more units than 12 unpromoted stores.
The promoted stores were selected because they had the highest previous sales, and they received prominent displays.

**Deliverable:** a short critique and a better measurement design.
**Acceptance criteria:** name selection and bundled treatment as separate problems; identify an assignment method, outcome and horizon.
**Time:** 20 minutes.

### Exercise 7 — Design a bounded offer test

A marketing manager proposes a personalized coupon for a seasonal item with limited stock.
The team wants to learn whether it increases contribution, not merely redemption.

**Deliverable:** one-page test plan with population, randomization unit, treatment, control, outcome, duration, inventory cap and stop rules.
**Acceptance criteria:** include a post-event window, funding, privacy/legal review, customer experience checks and an analysis of displaced sales.
**Time:** 30 minutes.

### Exercise 8 — Label the planning packet

The demand team estimates 4,500 units next month.
The approved supply plan reserves 4,000 units.
The commercial team wants 5,500 units.
A signed purchase order obligates delivery of 3,000 units.

**Deliverable:** label each number as forecast, plan, target or commitment and identify one operational risk.
**Acceptance criteria:** preserve the original figures and explain why they cannot be substituted for each other.
**Time:** 10 minutes.

## 13. Self-check answers

### Exercise 1

A is a **list-price** decision owned by the manufacturer's authorized pricing or commercial owner.
The customer is the buyer under the relevant agreement; any existing fixed-price contract must be checked.
B is a **trade term**, specifically a performance-conditioned allowance, authorized by the supplier's commercial signatory and accepted by the retailer.
C is a **retailer markdown** owned by the retailer's merchandise or pricing authority.
If the retailer seeks supplier funding, that is a separate supplier decision and requires an agreement.
D is a **personalized offer** owned by the retailer or campaign owner that issues it, subject to the actual arrangement.
The funding party cannot be inferred from the coupon's existence; inspect its settlement agreement.

### Exercise 2

Gross sales are 2,000 × $15.00 = **$30,000**, or **$15.00/unit**.
Off-invoice allowances are 2,000 × $1.00 = $2,000.
Invoice sales are $30,000 − $2,000 = **$28,000**, or **$14.00/unit**.
The rebate is 2,000 × $0.50 = $1,000, yielding **$27,000** net revenue, or **$13.50/unit**.
Product cost is 2,000 × $8.00 = $16,000; freight is 2,000 × $0.75 = $1,500.
Contribution is $27,000 − $16,000 − $1,500 = **$9,500**, or **$4.75/unit**.
A subsequent $1,000 settlement clears the accrual; it does not create a second $1,000 expense.

### Exercise 3

Baseline contribution is 1,000 × $5.00 = **$5,000**.
Event contribution before fees is 1,500 × $4.00 = $6,000.
After the $400 fee and $300 displaced contribution, event-relative contribution is $5,300.
Incremental contribution is $5,300 − $5,000 = **$300**.
Break-even requires `4 × Q − 400 − 300 >= 5,000`.
Thus `Q >= 1,425` units, and **1,425 units** is the minimum whole-unit threshold.
The calculation assumes constant $4.00 event contribution and a fixed $300 post-event loss.

### Exercise 4

If both offers apply to the $100.00 original base, each is $10.00 and the final price is **$80.00**.
If they apply sequentially, the first yields $90.00 and the second subtracts 10% of $90.00, giving **$81.00**.
The $1.00 difference is real exposure, not rounding noise.
Obtain an authorized precedence rule and test the live checkout and receipt before publishing.
Tax and rounding rules must also be specified in a real implementation.

### Exercise 5 rubric

A strong response separates the retailer's shelf-price authority from the supplier's funding approval.
It assigns the coupon to the campaign owner with privacy, legal and finance consultation.
It names an inventory owner who protects commitments, and finance as the accrual owner under entity policy.
It requires a signed amendment before supplier funding is executed.
It names a route for unresolved conflicts rather than assigning accountability to every participant.

### Exercise 6 rubric

A strong answer identifies selection bias because the promoted stores were selected on prior sales.
It separately notes that display and price were bundled, so the result is not a price-only effect.
A credible alternative randomly assigns comparable eligible stores, or uses another defensible design if randomization is impossible.
It records assignment, availability, event and post-event contribution, and reports uncertainty and spillovers.
It does not claim that one design automatically eliminates all interference or execution failures.

### Exercise 7 rubric

A strong plan randomly assigns eligible customers or clusters before exposure and defines the control's experience.
It measures incremental contribution after funding, product and fulfillment cost, returns and displaced sales.
It includes a sufficient post-event window, inventory reservation, legal/privacy review and pre-specified guardrails.
It specifies owner, monitoring cadence, exposure cap, price-error rollback and a method for interpreting uncertainty.
It acknowledges interference, eligibility changes and stockouts as threats to inference.

### Exercise 8

The **forecast** is 4,500 units, the **plan** is 4,000 units, the **target** is 5,500 units, and the **commitment** is 3,000 units.
The plan is 500 units below the forecast, so service risk exists if the forecast materializes and no further supply is available.
The target does not create 5,500 units of demand or an obligation to deliver them.
The signed order is a commitment subject to its actual contractual terms, not a guarantee that all forecast demand will materialize.

## 14. Interview questions

1. **How can list sales rise while contribution falls?**
   A strong answer traces discounts, variable costs and mix through a consistent waterfall.
   A weak answer assumes list revenue is collected cash or profit.
2. **Who controls a retailer's shelf price when a supplier offers a trade allowance?**
   A strong answer separates the firms' decision rights and checks the actual agreement and law.
   A weak answer assumes the supplier automatically controls the retailer's price.
3. **Why is an accrued rebate not deducted again when paid?**
   A strong answer distinguishes earning, liability settlement and any true-up.
   A weak answer treats every payment as new commercial expense.
4. **How do you calculate a break-even promotional volume?**
   A strong answer divides baseline contribution plus incremental costs and displaced contribution by positive action unit margin.
   A weak answer ignores the subsidy paid on baseline sales or mixes time horizons.
5. **What does an increased event-week volume prove?**
   A strong answer says it describes observed sales and needs a counterfactual for causal lift.
   A weak answer counts all event units as incremental.
6. **How should two percentage discounts combine?**
   A strong answer asks for their bases, ordering, eligibility and rounding.
   A weak answer always adds the percentages.
7. **When can a markdown improve value despite a lower price?**
   A strong answer compares sale contribution with expected future full-price demand, salvage and stock constraints.
   A weak answer assumes any discount is either always bad or always good.
8. **What is different about a personalized offer?**
   A strong answer identifies eligibility, funding, privacy, fairness and customer-experience controls.
   A weak answer treats a model score as automatic authorization to discriminate prices.
9. **What would stop a profitable forecast from being approved?**
   A strong answer names legal, contract, inventory, execution and uncertainty constraints.
   A weak answer treats the forecast point estimate as a guarantee.
10. **How do forecast, plan, target and commitment differ?**
    A strong answer distinguishes conditional prediction, intended action, desired outcome and obligation.
    A weak answer treats them as interchangeable demand numbers.

## 15. Key terms

- [List price](#4-definitions-and-vocabulary) — reference price for a defined scope and interval.
- [Shelf price](#4-definitions-and-vocabulary) — retailer's displayed consumer price.
- [Transaction price](#4-definitions-and-vocabulary) — amount charged on a particular transaction.
- [Trade term](#4-definitions-and-vocabulary) — commercial condition between counterparties.
- [Off-invoice allowance](#4-definitions-and-vocabulary) — deduction applied directly to an invoice.
- [Bill-back](#4-definitions-and-vocabulary) — qualifying later credit or claim.
- [Rebate accrual](#4-definitions-and-vocabulary) — estimate of an earned unpaid obligation.
- [Settlement](#4-definitions-and-vocabulary) — payment or credit clearing an obligation.
- [Price waterfall](#3-the-core-mental-model) — bridge from reference to a defined financial result.
- [Pocket price](#4-definitions-and-vocabulary) — realized price after defined deductions.
- [Contribution](#4-definitions-and-vocabulary) — net revenue less specified avoidable costs.
- [Incremental contribution](#6-worked-example) — contribution difference versus the no-action scenario.
- [Baseline](#4-definitions-and-vocabulary) — estimate of the no-action outcome.
- [Counterfactual](#5-the-mechanics-step-by-step) — outcome that would occur under an alternative action.
- [Markdown](#4-definitions-and-vocabulary) — selling-price reduction often used for clearance.
- [Salvage value](#4-definitions-and-vocabulary) — net value of residual stock at the horizon.
- [Personalized offer](#4-definitions-and-vocabulary) — eligibility-dependent incentive or price.
- [Eligibility](#4-definitions-and-vocabulary) — rules defining who qualifies.
- [Redemption](#4-definitions-and-vocabulary) — application of a qualifying benefit.
- [Funding](#4-definitions-and-vocabulary) — contractual responsibility for an offer's cost.
- [Stackability](#4-definitions-and-vocabulary) — whether and how offers combine.
- [Cannibalization](#4-definitions-and-vocabulary) — displaced sales from another product or channel.
- [Pull-forward](#4-definitions-and-vocabulary) — purchases shifted into the event from later.
- [Decision right](#4-definitions-and-vocabulary) — authority over a defined decision or control.
- [Forecast](#4-definitions-and-vocabulary) — conditional prediction of an outcome.
- [Plan](#4-definitions-and-vocabulary) — coordinated intended action.
- [Target](#4-definitions-and-vocabulary) — desired performance level.
- [Commitment](#4-definitions-and-vocabulary) — authorized obligation to act or deliver.
- [Opportunity cost](#4-definitions-and-vocabulary) — value of the best forgone feasible alternative.
- [Reference price](#4-definitions-and-vocabulary) — price against which a buyer compares an offer.

## 16. Sources

1. GS1, [GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/).
   Product identification and trade-item changes; a standard, not a pricing contract.
2. ISO, [ISO 4217 — Currency codes](https://www.iso.org/iso-4217-currency-codes.html).
   Currency-code convention; does not prescribe the financial exchange rate.
3. ISO, [ISO 8601 — Date and time format](https://www.iso.org/iso-8601-date-and-time-format.html).
   Date and time representation; business time-zone policy remains necessary.
4. IFRS Foundation, [IFRS 15 Revenue from Contracts with Customers](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-15-revenue-from-contracts-with-customers/).
   Authoritative standard for entities applying IFRS; consult the full standard and finance policy.
5. FASB, [Revenue Recognition](https://www.fasb.org/page/PageContent?pageId=/standards/implementing/revenue-recognition.html).
   Official resources for Topic 606 and implementation; specific accounting treatment depends on facts.
6. European Parliament and Council, [Directive (EU) 2019/2161](https://eur-lex.europa.eu/eli/dir/2019/2161/oj).
   Amends EU consumer-protection rules, including price-reduction announcement rules.
7. European Commission, [Guidance on Article 6a of the Price Indication Directive](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52021XC1229%2806%29).
   Interpretive guidance on prior prices, scope and exceptions.
8. FTC, [Guides Against Deceptive Pricing, 16 CFR Part 233](https://www.ecfr.gov/current/title-16/chapter-I/subchapter-B/part-233).
   U.S. administrative guidance on former and comparative price representations.
9. European Parliament and Council, [General Data Protection Regulation, Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
   Primary EU legal text on personal-data processing, subject to applicability and national context.
10. U.S. Department of Justice, [Price Fixing, Bid Rigging, and Market Allocation Schemes](https://www.justice.gov/atr/price-fixing-bid-rigging-and-market-allocation-schemes).
    Enforcement agency overview of horizontal competition-law risks.
11. J. C. Penney Company, Inc., [2012 Annual Report](https://www.annualreports.com/HostedData/AnnualReports/PDF/NYSE_JCP_2012.pdf).
    Company-reported strategy and financial results; not causal identification of any single change.
12. Kasra Ferdows, Michael A. Lewis and Jose A. D. Machuca, [“Rapid-Fire Fulfillment,” Harvard Business Review, November 2004](https://hbr.org/2004/11/rapid-fire-fulfillment).
    Published business case on Zara's operating model; not a controlled markdown experiment.
13. Kalyan T. Talluri and Garrett J. van Ryzin, *The Theory and Practice of Revenue Management*, Springer, 2004.
    [Publisher record](https://link.springer.com/book/10.1007/b139000); conceptual basis for capacity and opportunity-cost decisions.

## 17. Links

- [Module 16 topic map](./_index.md).
- [Module 16 survey, including Lab 16.1 and Lab 16.2](../../../course/16_pricing_promotions_revenue_management.md#7-labs).
- [Module 16 survey and pricing-stack overview](../../../course/16_pricing_promotions_revenue_management.md).
- [Promotion mechanics in Module 09](../09_demand_sensing_promotions_npi/03_promotion-mechanics.md).
- [Module 09 survey and demand-sensing context](../../../course/09_demand_sensing_promotions_npi.md).
- Sibling topics on willingness to pay, markdown optimization and personalized pricing are listed in the [Module 16 topic map](./_index.md#3-topics); follow the individual links as those planned files become available.
