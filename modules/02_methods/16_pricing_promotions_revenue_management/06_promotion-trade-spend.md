# Promotion Optimization and Trade Spend

> Module 16 — Pricing, Promotions, and Revenue Management  
> Topic 06 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 04, 05, and 07; [own- and cross-price elasticity](03_elasticity-identification.md)  
> Version 2026.5

## 1. Why this topic exists

A promotion is a commercial intervention, not a synonym for a sales increase.
A supplier may pay for a temporary price reduction, a display, an advertisement, or a retailer's performance.
The resulting orders and store sales are visible.
The sales that would have happened without the event are not.

A category manager must decide whether to fund the next event before all its consequences are visible.
A supply planner needs a demand forecast and a feasible replenishment plan.
A finance partner needs the full obligation, including payments that arrive after the event.
An account manager needs an agreement that makes performance and settlement auditable.
A retailer needs to decide whether shelf space and labor are worth allocating to the event.
These are related decisions, but they do not have the same objective or information.

Suppose a snack brand sells 45% more units during a featured week.
That is not automatically 45% incremental demand.
Some units would have sold at the regular price.
Some replace purchases of another pack in the supplier's portfolio.
Some are purchased early rather than in the following weeks.
The supplier may also fund every eligible unit, including the units that would have sold anyway.

A shipment spike can be even less informative.
The retailer may have bought inventory before the event and drawn it down afterward.
Treating the shipment spike as consumer demand can inflate the apparent lift.
Paying again on consumer scans can also double-count the same discount if the invoice allowance is not reconciled.
A credible evaluation must track both the physical flow and the money flow.

The decision is not simply whether total promoted revenue exceeded the cost of a display.
The relevant economic question is whether contribution over the defined portfolio, customer, and time horizon is higher than in a credible no-event alternative.
The practical decision is whether that improvement justifies the budget, working capital, service risk, and alternative use of capacity.

This topic builds that comparison from a baseline, explains which effects can be measured, and shows how a business should settle and learn from an event.
All numeric business cases in this lesson are illustrative, not claims about a named company.
They use US dollars (USD), consumer units, and weekly periods unless stated otherwise.

## 2. Learning outcomes

After completing this topic, you can:

1. **Define** the decision, economic owner, unit, affected portfolio, and evaluation horizon for a promotion.
   - Evidence: Exercises 1 and 7 identify an appropriate scope and decision owner.
2. **Construct** a no-promotion baseline and critique its identification assumptions.
   - Evidence: Exercises 2 and 5 examine baseline validity and comparison groups.
3. **Compute** incremental contribution, including funding on baseline units, cannibalization, and a post-event dip.
   - Evidence: Exercises 3 and 4 require an auditable event bridge.
4. **Distinguish** consumer sell-through, retailer orders, accruals, deductions, and cash settlement.
   - Evidence: Exercises 6 and 7 specify reconciliation checks.
5. **Choose** between candidate events under budget, capacity, and uncertainty constraints.
   - Evidence: Exercise 8 requests a feasible choice and a sensitivity condition.
6. **Design** a post-event review and experiment that can improve future decisions.
   - Evidence: Exercises 5, 7, and 9 require a control design and a governance procedure.
7. **Explain** legal, accounting, data, and competitive boundaries without treating an analytical rule as legal advice.
   - Evidence: Exercise 6 asks for an appropriate escalation and evidence record.

You should be able to reproduce the main example with a calculator or the Python program.
A correct answer exposes the counterfactual and cost scope instead of only reporting a return percentage.

## 3. The core mental model

```text
Commercial decision and eligibility
              |
              v
No-event outcome <----- comparable history / controlled stores
              |
              | compare with actual event and post-event outcomes
              v
Incremental units by product, store, and week
              |
              v
Incremental contribution before event-specific spend
              |
              +--> subtract scan funding, fixed fees and execution costs
              +--> include affected sibling and later-period contribution
              v
Incremental contribution under a stated boundary
              |
              +--> sensitivity, capacity and strategic constraints
              v
Decision -> contracted execution -> claim settlement -> learning
```

The left side of the comparison is not a target.
It is an estimate of the outcome that would have occurred under the same conditions without the promotion.
It includes ordinary demand, seasonality, and any expected distribution or availability changes that would also occur without the event.
It does not include demand caused by the event under review.

The right side is the observed outcome under the actual event.
The treatment may include price, placement, advertising, a display, and retailer execution together.
If they occur together, their combined effect can be evaluated, but their individual effects cannot be separated without more variation or stronger assumptions.

Contribution is a flow measure, not a measure of satisfaction or total revenue.
It is revenue less the variable costs and event-specific costs that change between the alternatives.
An actual accounting profit figure may include allocated overhead that does not change with the decision.
Conversely, an economic comparison must include real incremental costs even if they post in another accounting line.

The boundary determines what the result means.
A promoted-stock-keeping-unit (SKU) result answers a narrower question than a supplier-portfolio result.
A one-week result differs from a four-week result when shoppers stock up.
A supplier result differs from a retailer result when a supplier funds a discount and the retailer controls the shelf price.
The same promotion can benefit one party and harm another.

Avoid subtracting the same effect twice.
If the event-week sibling sales and later promoted-SKU sales are already included in the actual-minus-baseline portfolio calculation, do not subtract separate estimates for cannibalization or pull-forward again.
Those labels can explain the bridge, but they are not additional charges.
The worked example keeps these buckets disjoint.

A forecast estimates future outcomes under specified conditions.
A plan is a chosen operational response, such as quantities and staffing.
A target is an aspirational performance threshold.
A commitment is a contractual or formally approved obligation.
None of these should silently replace the counterfactual estimate.

### Three important contrasts

| Contrast | Correct interpretation | Typical mistake |
|---|---|---|
| Incremental units versus promoted units | Incremental units are actual less no-event units in the same scope. | Count every unit sold during the event as new. |
| Contribution versus revenue | Contribution deducts relevant variable and event costs. | Call a revenue increase profit. |
| Settlement versus accrual | Settlement resolves an estimate against an eligible, documented claim. | Treat a booked accrual as proof the retailer performed. |
| Consumer demand versus orders | Consumer scans show sell-through; orders also reflect inventory decisions. | Treat a pre-event buy-in as incremental consumption. |
| Analysis versus proof | A model estimates the unobserved outcome under assumptions. | Present a point estimate as an observed fact. |
| Supplier versus retailer economics | Each party has distinct prices, costs, and objectives. | Apply the supplier's margin to the retailer's units. |

## 4. Definitions and vocabulary

The unit of analysis must be stated before using any of these terms.
For example, 1 case may contain 12 consumer units and must not be joined to 12 cases as if the units were identical.

| Term | Definition and unit | Common misuse |
|---|---|---|
| Promotion | Time-bounded commercial treatment such as a price reduction or display. | Assuming a discount is the only treatment. |
| Event | A specifically identified instance of an offer across eligible products, customers, stores, and dates. | Combining two overlapping agreements under one unidentified event. |
| Treatment | The intervention whose effect is being estimated. | Claiming a price-only effect from a combined price-and-display campaign. |
| Counterfactual | Outcome that would have occurred for the same population without the event, in units or USD. | Describing the unobserved counterfactual as recorded sales. |
| Baseline | Operational estimate of the counterfactual for a defined time and population, in units. | Using a sales target as the baseline. |
| Actual | Observed outcome for the defined eligible population and period, in units or USD. | Mixing data extracted at different cutoff dates without restatement. |
| Incremental units | Actual units minus baseline units over the same boundary. | Counting gross promotional units as incremental. |
| Gross lift | Event-period actual units minus baseline units, before other affected buckets. | Treating it as net portfolio gain. |
| Relative lift | Incremental units divided by baseline units, expressed as a percentage. | Comparing percentages without their different denominators. |
| Uplift factor | Actual units divided by baseline units, a ratio. | Calling a factor of 1.45 a 145% increase. |
| Incremental contribution | Actual contribution minus counterfactual contribution in the stated boundary, in USD. | Treating incremental revenue as contribution. |
| Variable contribution | Net realized revenue less costs that vary with the unit, in USD per unit. | Deducting the same scan allowance here and again as a separate cost. |
| Regular unit contribution | Unit contribution in the no-event state, in USD per unit. | Applying a temporary discounted margin to the counterfactual. |
| Event unit contribution | Unit contribution on promoted sales under the event terms, in USD per unit. | Applying it to noneligible stores or weeks. |
| Cannibalization | A treatment-caused reduction in contribution from another relevant product or channel. | Assuming every observed decline in another SKU was caused by the event. |
| Halo | A treatment-caused increase in contribution from another product or channel. | Counting an unrelated seasonal rise as a promotion benefit. |
| Pull-forward | Purchases moved from a later period into the event period because of the promotion. | Calling a permanent acquisition of customers pull-forward. |
| Post-event dip | Observed sales below an estimated later-period baseline. | Assuming every dip is causal without considering other changes. |
| Stockpiling | Customers or retailers accumulate product for later use or sale. | Assuming customer stockpiling and retailer forward buying are identical. |
| Forward buying | A retailer buys more inventory in advance to exploit terms or ensure supply. | Calling extra shipments final consumer demand. |
| Sell-in | Supplier shipments or invoices to the retailer, in cases or units. | Treating a shipment as a sale to a shopper. |
| Sell-through | Retailer sales to shoppers, commonly measured by point-of-sale scans. | Assuming all stores send complete scan data. |
| Point of sale (POS) | A checkout transaction recording product, quantity, price, and time. | Assuming POS alone measures shelf availability or causal lift. |
| On-invoice allowance | Discount credited when the supplier invoices the retailer, often USD per case or unit. | Counting a later deduction for the same discount as a second valid allowance. |
| Scan allowance | Supplier payment based on qualifying retailer consumer sales, often USD per scanned unit. | Paying on shipments when the agreement specifies scans. |
| Off-invoice allowance | Price reduction reflected on the supplier's invoice under agreed terms. | Assuming it necessarily reaches the shelf price. |
| Fixed fee | Agreed event payment not mechanically proportional to qualifying units, in USD per event. | Omitting it from the incremental event comparison. |
| Display fee | Payment for a specified merchandising placement or service, in USD. | Assuming payment proves the display ran. |
| Accrual | Estimated liability or expense recognized before final claim settlement, in USD. | Treating the estimate as settled cash. |
| Deduction | Amount withheld by a customer from a payment, often accompanied by a claim. | Treating every deduction as a valid, approved obligation. |
| Settlement | Reconciliation and resolution of an approved obligation against claim and payment, in USD. | Closing without matching the supporting evidence. |
| True-up | Adjustment between an estimate or prior settlement and an agreed final amount, in USD. | Silently changing the historical event without versioning. |
| Trade-spend return on investment (ROI) | A defined return divided by a defined trade-spend denominator. | Comparing ratios with different numerators or denominator conventions. |
| Break-even lift | Incremental event volume needed for zero incremental contribution under stated assumptions. | Treating an unconstrained algebraic threshold as a guaranteed forecast. |
| Control group | Comparable units not exposed to the treatment, used to estimate no-event outcomes. | Selecting stores after observing their outcomes. |
| Spillover | Treatment changes outcomes in nominal control or nearby populations. | Treating a contaminated control as untreated. |
| Stockout | Demand cannot be fulfilled because inventory is unavailable. | Interpreting constrained sales as unconstrained demand. |
| Pack conversion | Mapping of cases, packs, weights, and consumer units to a common unit. | Adding cases and units without conversion. |
| Plan | Chosen operational quantities and actions for an event. | Treating planned quantities as a demand forecast. |
| Target | Desired performance used to steer a team. | Using a target as observed demand. |
| Commitment | An approved or contractual obligation, subject to its terms. | Assuming an unapproved forecast is a binding obligation. |

## 5. The mechanics, step by step

### Step 1 — Frame the decision and the owner

**Input:** a proposed or completed event, decision deadline, accountable business owner, and available alternatives.
**Output:** a one-page decision brief with an explicit comparison.

1. Identify whether the question is pre-event selection, in-event intervention, settlement, or post-event learning.
2. Name the supplier, retailer, category, and brand perspectives separately.
3. Specify the eligible SKUs, customers, stores, channels, and geographic area.
4. State event start and end dates, time zone, and the later observation window.
5. Record whether the decision is one event versus no event or one event versus another feasible event.
6. State which financial result governs approval and which results are supporting evidence.
7. Name the decision maker and who can approve deviations or exceptions.

A no-event comparison shows whether a promotion creates value relative to doing nothing under the stated assumptions.
A choice between two positive promotions additionally depends on the best alternative use of money and scarce capacity.
An event can be profitable in isolation but inferior to a competing event.

Do not conflate the retailer's contribution with the manufacturer's.
A retailer may gain basket traffic while the supplier pays for a price cut on units that were already going to sell.
If a joint business plan values both parties, show each ledger before adding or negotiating transfers.
A transfer from supplier to retailer is a cost to one and revenue to the other; it cancels in a correctly consolidated joint view, but the actual consumer discount does not.

### Step 2 — Record the event treatment

**Input:** the signed promotion agreement, retailer calendar, item eligibility, and execution plan.
**Output:** a versioned event record with each treatment component and funding term.

Record the regular and planned shelf prices as USD per consumer unit.
Record the supplier's invoice price and any allowance in the same unit or show a conversion.
Specify the eligible date range and whether eligibility is based on shipment, invoice, scan, or another contractually defined event.
Note whether the payment is capped, tiered, subject to minimum performance, or conditional on proof.
Identify fixed advertising, placement, display, and operational costs separately.

Record the planned display location, advertisement, participating stores, expected stock, and media placement.
These elements matter because a displayed item can receive incremental demand even with a modest price change.
A pure price elasticity applied to a multi-component campaign is not enough to explain the combined effect.

If the retailer may independently change shelf prices, the supplier's funding is not proof of consumer pass-through.
Measure actual consumer price and placement where data permit.
If execution differs by store, capture the difference before looking at outcomes and report exposure coverage.

### Step 3 — Choose a defensible no-event baseline

**Input:** historical outcome data, event calendar, covariates, comparison units, and a causal design if available.
**Output:** expected no-event units by SKU, store, and week, with a version and uncertainty assessment.

A baseline can come from a randomized holdout, comparable non-promoted stores, a predictive model, or a carefully selected historical reference.
The best feasible method depends on identification, data, and decision consequence.
A historical average is easy to reproduce but can fail when the event coincides with a holiday or a distribution change.
A complex model can also fail if it learns from prior events that are not labeled.

Prefer a design that establishes a counterfactual before the event.
Randomize eligible stores or comparable clusters to treatment and control where feasible.
Define geographic clusters if shoppers can move between nearby stores.
Check that controls remain untreated and stocked during the measurement window.
Record the randomization unit, eligible population, sample size rationale, and planned analysis before outcome inspection.

Where randomization is not feasible, use matched comparisons or a transparent model with pre-event validation.
Check parallel pre-trends where a difference-in-differences design relies on them.
Parallel pre-trends support but do not prove the assumption of parallel counterfactual changes after treatment.
Assess simultaneous competitor offers, seasonal demand, distribution, price, holidays, and local disruptions.
Do not match on an outcome measured after the event begins.

For every baseline, distinguish the **forecast made before the event** from an ex post estimate using later information.
Freeze the original forecast for forecast-quality evaluation.
Version any later causal estimate and explain why its inputs or assumptions changed.
Do not revise the baseline to make a chosen event pass its approval threshold.

### Step 4 — Set the portfolio and time boundary

**Input:** customer purchase patterns, substitutes, complements, supply lead times, and stockpiling risk.
**Output:** a list of nonoverlapping product, channel, and time cells.

A cell might be one SKU, one store, and one week.
For consumer stockpiling, extend the horizon until a plausible dip can be observed.
For retailer forward buying, include inventory before and after the event and follow shipments beyond the promotional date.
For a portfolio question, include likely substitute and complementary items, not just the featured SKU.

The required window varies by shelf life, consumption rate, purchase frequency, and deal terms.
A four-week horizon is an example, not a universal rule.
A durable product may have effects that last much longer; a perishable product may have a shorter horizon.
If data stop before the likely effect ends, mark the estimate as truncated and show an alternative scenario.

Protect against overlap.
If a later sale is assigned to pull-forward, do not subtract that same unit again in a second post-event adjustment.
If the sibling SKU is already included in the portfolio actual-minus-baseline ledger, do not apply a separate cannibalization penalty.
If a later event overlaps the dip window, attribute results cautiously; the events cannot always be separated.

### Step 5 — Reconcile physical volumes

**Input:** supplier shipments, retailer receipts, beginning and ending inventory, POS units, returns, and known adjustments.
**Output:** a reconciled flow and an explicit list of missing or inconsistent data.

For a simple retailer inventory position, use:

```text
ending inventory = beginning inventory + receipts - sell-through
                   - returns to supplier - other net removals
```

All terms must use the same units and scope.
Other removals may include shrink, damage, transfers out, or write-offs; transfers in must be added.
Timing differences between shipment, receipt, and scan require a consistent cutoff.
The identity is a diagnostic, not a guarantee that all components were observed correctly.

When only shipments are available, a lift in shipments cannot identify lift in consumer sales unless inventory changes are known or defensibly bounded.
When only POS is available, it is still necessary to understand retailer inventory if the funding is tied to shipments or invoice deductions.
Report item availability and distribution coverage separately from sales.
A zero sale in an open, fully stocked store differs from a zero created by a missing file or a stockout.

### Step 6 — Calculate the promoted-item contribution bridge

**Input:** actual and baseline promoted-item units, regular unit contribution, event unit contribution, and event-specific costs.
**Output:** event-period incremental contribution before separately scoped portfolio and later-period effects.

Let `B` be event-period baseline units and `A` actual promoted units.
Let `m0` be regular contribution per baseline unit and `m1` event contribution per actual unit.
Both margins must be defined consistently and in USD per unit.
Then:

```text
promoted-item event contribution difference = A * m1 - B * m0
```

If the event margin differs only because a per-unit allowance `d` applies to every qualifying sale, then `m1 = m0 - d` and:

```text
same difference = (A - B) * m0 - A * d
```

The first term is regular contribution on incremental units.
The second is funding on **all** eligible actual units, including the baseline units.
Do not also subtract `A * d` from the first equation if it is already embedded in `m1`.
If the allowance is only on incremental units, is tiered, has a cap, or is applied to shipments rather than scans, substitute the actual contract rule.

Add incremental freight, labor, waste, or other variable changes when relevant.
Subtract incremental fixed event payments and execution costs once.
Do not include regular overhead that remains unchanged solely because it was allocated to the promotion.
If the promotion consumes genuinely scarce capacity, include the contribution of displaced feasible demand where it is not already in the portfolio ledger.

### Step 7 — Measure cannibalization, halo, and later effects

**Input:** actual and counterfactual contribution for affected non-promoted SKUs, channels, and later weeks.
**Output:** an incremental contribution bridge with mutually exclusive rows.

Compute each affected cell as actual contribution minus baseline contribution.
A negative sibling-SKU contribution difference can be described as cannibalization only to the extent the causal design supports that interpretation.
An observed decline may instead reflect a competitor's offer, an out-of-stock, or a distribution change.
A positive companion-item difference may be a halo, but the same causal caution applies.

For later promoted-item periods, compare actual with the estimated no-event baseline using the regular margin that applies in each later cell.
A post-event dip is consistent with pull-forward, but it is not proof that all missing units are the same shoppers who bought early.
A later inventory drawdown can also reflect retail forward buying rather than consumer stockpiling.
Investigate both channels before assigning the label.

A practical bridge can use separate rows for event-week sibling effects and later-week promoted-item effects.
A full cell-level ledger can instead add all affected cells directly.
These are two presentations of the same comparison, not additive methods.

### Step 8 — Price the funding and obligations

**Input:** contracted rates, eligible unit base, proof of performance, caps, invoice terms, and finance records.
**Output:** an expected obligation, approved accrual, and traceable settlement schedule.

For each term, ask four questions.
What triggers payment?
What quantity or service is eligible?
What evidence proves the trigger?
How and when will the amount be settled?

An on-invoice reduction is already reflected in the invoiced net price if the source uses net revenue.
A scan-back claim may arrive weeks after the event.
A fixed display fee may require proof that the display was present at a specified location and time.
An invoice deduction is a customer claim and must be matched to the underlying obligation; it is not automatically a new expense.

Build the expected obligation using the agreement's actual unit and date rules.
For a capped allowance, the obligation may be `min(rate * eligible_units, cap)` if that is what the contract specifies.
For tiered rates, explicitly state whether the rate applies marginally by tier or retrospectively to all units.
Different contract language produces different economics.

Set an accrual using the organization's accounting policy and the best supported estimate.
Reconcile accrual changes to claims, payments, and reversals.
Separate the operational event economics from the accounting recognition policy; the two should reconcile over time but need not post in the same week.
Escalate material disputed deductions rather than forcing a closure to meet a reporting deadline.

### Step 9 — Assess uncertainty and constraints

**Input:** baseline uncertainty, contribution assumptions, execution risk, available budget, inventory, and operational capacity.
**Output:** a decision range, break-even condition, and feasible plan.

Use scenarios or an uncertainty interval that reflects the causal design and sampling unit.
For cluster-randomized stores, do not calculate precision as if every transaction were an independent randomized observation.
Include uncertainty in baseline, substitution, and post-event effects where material.
A single point estimate is not proof that the true effect is positive.

Check whether additional demand can be supplied without unacceptable stockouts or displacement.
Consider replenishment lead time, minimum order quantities, shelf capacity, labor, and shelf-life constraints.
A financially attractive event forecast can be operationally infeasible.
Conversely, a positive event in a weak demand period may relieve perishable inventory risk, but only if that benefit is separately quantified.

Compare candidates using the same horizon, risk treatment, and contribution definition.
When budget is scarce, the highest percentage ROI need not maximize total contribution.
An event with a high ROI can be too small to use the available budget; a larger lower-ROI event may generate more USD within the constraints.
With multiple indivisible events, dependencies and capacity limits, rank by ratio alone is not guaranteed to produce the optimal portfolio.

### Step 10 — Approve, execute, learn, and settle

**Input:** decision brief, signed agreement, approved budget, plan, performance evidence, and validated data.
**Output:** approved or rejected event, closed claim record, and reusable learning.

Before launch, finance checks the economic bridge and funding cap.
Commercial and legal owners review agreement terms and applicable competition requirements.
Supply planning approves the inventory and replenishment plan, including contingency thresholds.
The data owner freezes the baseline and documents the analysis plan.

During the event, monitor realized price, display execution, availability, service, and spend against plan.
Use an explicit intervention policy; avoid changing a control group to rescue sales without recording it.
An intervention can protect the business but may change what can be learned from the original experiment.
Record the time, reason, affected stores, and resulting change in treatment.

After the event, close the data only when the agreed window and claim-lag policy permit.
Compare actual execution with the contracted treatment and planned quantities.
Reconcile scan counts, deductions, and fixed payments to contract and proof.
Investigate unmatched or duplicate claims and document the resolution.

Publish a post-event review with the decision, baseline method, actuals, economic bridge, uncertainty, exceptions, and next action.
Retain the original pre-event forecast alongside any revised estimate.
Assign one owner and date to each learning action.
A review is complete when it changes a decision, closes a justified exception, or records why no decision change is warranted.

## 6. Worked example

### Situation and decision

A fictional beverage supplier is considering whether to repeat a one-week promotion at a grocery chain.
The manufacturer perspective is used throughout.
The promoted item is a 500 mL bottle sold in consumer units.
The promotion also affects a related non-promoted bottle sold by the same supplier.
The evaluation covers the event week plus the following three weeks.
The decision is repeat versus no event under the same conditions, not whether the retailer's total store profit increased.

Assume retailer inventory is unchanged between the beginning and end of the evaluated horizon.
Assume event-week supplier shipments and eligible consumer scans both equal 14,500 units for the promoted item.
That equality is a simplifying assumption, not a general property of promotions.
The supplier pays a scan allowance on each eligible event-week consumer unit.
The allowance does not apply to the later three weeks.

The no-event baseline is an illustrative causal estimate.
It is not observed and is not established by the calculations below.
The case assumes that event-week sibling losses and later promoted-item losses are caused by the event and do not overlap.
In a real review, those assumptions require evidence and uncertainty bounds.

### Commercial terms

| Input | Amount | Meaning |
|---|---:|---|
| Regular supplier invoice revenue | USD 10.00 per unit | Before event funding. |
| Variable product and distribution cost | USD 6.00 per unit | Assumed unchanged by the event. |
| Regular contribution | USD 4.00 per unit | USD 10.00 minus USD 6.00. |
| Event scan allowance | USD 1.20 per eligible unit | Charged on every qualifying event-week scan. |
| Event contribution | USD 2.80 per unit | USD 10.00 minus USD 6.00 minus USD 1.20. |
| Fixed display fee | USD 1,500 per event | Incremental and contractually earned. |
| Supplier execution cost | USD 2,100 per event | Incremental labor and freight, not included in unit cost. |
| Sibling regular contribution | USD 3.00 per unit | Applied to the affected sibling units. |

The regular shelf price and retailer profit are not needed to calculate this supplier-only ledger.
They are nevertheless relevant to a joint commercial decision and should be documented in the actual event file.
The example assumes the event's scan allowance is incremental to the supplier's regular terms.
No additional on-invoice discount is hidden in the stated net revenue.

### Forecast, plan, target, and commitment

| Item | Case value | Interpretation |
|---|---:|---|
| No-event baseline | 10,000 promoted units in the event week | Estimated counterfactual, not an actual outcome. |
| Event forecast | 14,500 promoted units in the event week | Forecast under the proposed treatment. |
| Replenishment plan | 15,000 units available for the event week | Operational choice, not guaranteed demand. |
| Commercial target | At least USD 0 incremental contribution over four weeks | Decision threshold, not a forecast. |
| Funding commitment | USD 1.20 per qualifying scan plus USD 1,500 display fee | Assumed approved terms, subject to evidence. |
| Observed event scans | 14,500 promoted units | Actual event outcome in the hypothetical review. |

The 15,000-unit replenishment plan is adequate in this simplified case because observed eligible demand is 14,500 units.
The difference is not automatically wasted stock; its value depends on later sell-through and inventory costs.
No additional capacity opportunity cost is assumed.

### Counterfactual and actual volumes

| Scope | No-event baseline | Actual | Actual minus baseline |
|---|---:|---:|---:|
| Promoted item, event week | 10,000 units | 14,500 units | +4,500 units |
| Sibling item, event week | 7,200 units | 6,300 units | −900 units |
| Promoted item, following three weeks | 30,000 units | 29,400 units | −600 units |

The event-week gross lift is 4,500 units.
Relative lift is 4,500 / 10,000 = 45%.
The uplift factor is 14,500 / 10,000 = 1.45.
The 900-unit sibling decline and the 600-unit later decline are separate cells.
The net change over these cells is 4,500 − 900 − 600 = 3,000 units, but units with different margins cannot be treated as interchangeable profit.

The later baseline is the three-week total, not an assumed 10,000 units in each individual week.
The example does not infer which individual shopper moved a purchase.
It treats the aggregate later shortfall as a causal loss only because that is an explicit case assumption.

### Promoted-item bridge

Without the event, the promoted item generates:

```text
10,000 units * USD 4.00 per unit = USD 40,000
```

With the event, the same item generates:

```text
14,500 units * USD 2.80 per unit = USD 40,600
```

The event-week promoted-item contribution difference is therefore USD 600.
A check using regular contribution and total scan funding gives the same answer:

```text
4,500 incremental units * USD 4.00 per unit = USD 18,000
14,500 eligible units * USD 1.20 per unit   = USD 17,400
USD 18,000 - USD 17,400                     = USD 600
```

Funding on the 10,000 baseline units is USD 12,000.
Funding on the 4,500 additional units is USD 5,400.
Together they equal the USD 17,400 allowance.
Charging only USD 5,400 would incorrectly ignore the discount given on units that would have sold anyway.

### Portfolio and later-period bridge

The sibling item loses 900 units at USD 3.00 contribution per unit:

```text
-900 units * USD 3.00 per unit = -USD 2,700
```

The promoted item loses 600 units in the following three weeks at its regular USD 4.00 contribution per unit:

```text
-600 units * USD 4.00 per unit = -USD 2,400
```

Subtract the incremental display fee and execution cost once:

| Component | Incremental contribution |
|---|---:|
| Promoted item, event week, net of scan allowance | +USD 600 |
| Sibling item, event week | −USD 2,700 |
| Promoted item, following three weeks | −USD 2,400 |
| Fixed display fee | −USD 1,500 |
| Incremental execution cost | −USD 2,100 |
| **Four-week supplier total** | **−USD 8,100** |

The current decision under the stated threshold is **do not repeat on unchanged terms**.
The 45% gross lift and the positive USD 600 promoted-item event-week contribution are insufficient to cover the other effects.
This is an illustrative conditional decision, not a prediction about any real promotion.

### Break-even and sensitivity

Hold the 10,000-unit event baseline, USD 1.20 scan allowance, USD 4.00 regular margin, and all other effects fixed.
Let `L` be the extra event-week units over baseline.
Then the four-week contribution is:

```text
USD 4.00 * L - USD 1.20 * (10,000 + L)
- USD 2,700 - USD 2,400 - USD 1,500 - USD 2,100
= USD 2.80 * L - USD 20,700
```

The algebraic break-even lift is 20,700 / 2.80 = 7,392.857 units.
Because units are whole, the smallest nonnegative-contribution lift is 7,393 units, corresponding to 17,393 eligible event-week units.
That is a 73.93% lift on the 10,000-unit baseline, compared with the assumed 45% actual lift.
The threshold is conditional: sibling losses, later losses, execution costs, availability, and funding terms could change at higher volumes.
It is not a demand forecast or an operational commitment.

At the observed 14,500 units, a zero-result scan allowance with all other inputs fixed is:

```text
(USD 18,000 - USD 2,700 - USD 2,400 - USD 1,500 - USD 2,100)
/ 14,500 units = USD 0.641379... per unit
```

An allowance of USD 0.64 per unit would produce USD 20 of incremental contribution under those fixed assumptions.
That arithmetic does not establish that a retailer would accept the rate or deliver the same price, placement, or 14,500 units.
Renegotiation changes the treatment and needs a new forecast.

### Settlement and interpretation

At event close, the expected scan obligation is USD 17,400 and the assumed earned display fee is USD 1,500.
Total contracted trade funding is USD 18,900, excluding the supplier's USD 2,100 execution cost.
If a USD 18,900 accrual was booked and a substantiated claim later totals USD 18,900, the settlement matches the estimate.
If the retailer deducts USD 19,500, the USD 600 difference requires investigation, not automatic addition to event cost.
The financial posting depends on the actual claim, contract, and accounting policy.

One possible metric is incremental contribution divided by trade funding: −USD 8,100 / USD 18,900 = −42.86%.
This definition includes all event costs in the numerator but only contractual trade funding in the denominator.
Another organization might include execution cost in the denominator or use a different return numerator.
State the formula before comparing results; the USD contribution bridge is the primary decision evidence here.

## 7. Code, query, or calculation

The following standard-library Python 3 program reproduces the example.
It uses `Decimal` for monetary calculations and holds the stated effects fixed in the break-even sensitivity.
It does not estimate a baseline or prove causality.
Its inputs are assumed to have already been reconciled to eligible units and contract terms.
Run it as a standalone script; no files or packages are required.

```python
from decimal import Decimal, ROUND_CEILING


def money(value):
    """Format a signed Decimal as US dollars."""
    return f"${value:,.2f}"


# Units are consumer units; all monetary amounts are in USD.
baseline_units = 10_000
actual_units = 14_500
regular_margin = Decimal("4.00")
scan_allowance = Decimal("1.20")
sibling_delta_units = -900
sibling_margin = Decimal("3.00")
later_delta_units = -600
later_margin = Decimal("4.00")
display_fee = Decimal("1500")
execution_cost = Decimal("2100")

# The scan allowance is included here and is not deducted again below.
event_margin = regular_margin - scan_allowance
promoted_delta = (
    Decimal(actual_units) * event_margin
    - Decimal(baseline_units) * regular_margin
)
sibling_delta = Decimal(sibling_delta_units) * sibling_margin
later_delta = Decimal(later_delta_units) * later_margin
incremental_contribution = (
    promoted_delta
    + sibling_delta
    + later_delta
    - display_fee
    - execution_cost
)

# This alternative bridge is only a cross-check, not another cost line.
incremental_units = actual_units - baseline_units
scan_funding = Decimal(actual_units) * scan_allowance
cross_check = Decimal(incremental_units) * regular_margin - scan_funding
assert cross_check == promoted_delta

trade_funding = scan_funding + display_fee
roi = incremental_contribution / trade_funding * Decimal("100")

# Solve for event-week incremental units with all other effects fixed.
other_costs = -sibling_delta - later_delta + display_fee + execution_cost
contribution_per_lift_unit = regular_margin - scan_allowance
if contribution_per_lift_unit <= 0:
    raise ValueError("Break-even calculation requires a positive event margin")
required_lift = (
    Decimal(baseline_units) * scan_allowance + other_costs
) / contribution_per_lift_unit
whole_lift = required_lift.to_integral_value(rounding=ROUND_CEILING)
required_total = baseline_units + int(whole_lift)

print(f"Event lift: {incremental_units:,} units")
print(f"Scan funding: {money(scan_funding)}")
print(f"Promoted item difference: {money(promoted_delta)}")
print(f"Sibling difference: {money(sibling_delta)}")
print(f"Later difference: {money(later_delta)}")
print(f"Incremental contribution: {money(incremental_contribution)}")
print(f"Trade-funding ROI: {roi:.2f}%")
print(f"Break-even lift: {whole_lift:,.0f} units")
print(f"Break-even event total: {required_total:,} units")
```

Expected output:

```text
Event lift: 4,500 units
Scan funding: $17,400.00
Promoted item difference: $600.00
Sibling difference: $-2,700.00
Later difference: $-2,400.00
Incremental contribution: $-8,100.00
Trade-funding ROI: -42.86%
Break-even lift: 7,393 units
Break-even event total: 17,393 units
```

For a production workflow, replace the input constants with validated, versioned event data.
The production calculation must handle contract-specific caps, tiers, returns, missing stores, dates, and unit conversions.
Store the full cell-level actual and baseline records rather than only a final aggregate.
Keep the contract version and calculation version with every result.
Where actual event volume is negative because of returns, use the agreement's netting and eligibility rules rather than assuming this simple positive-volume formula applies.

A sensitivity analysis should vary causal quantities jointly when they are related.
For example, deeper discounts may increase both event volume and pull-forward.
Changing event volume alone is useful for an algebraic threshold, but it is not a realistic demand response model.
A model of competing promotion packages should predict each package's eligible volume, margin, substitution, later effects, and operational burden under comparable assumptions.

## 8. Data, standards, and regulatory context

### Minimum data contract

The analysis is only as credible as the data lineage from agreement to claim.
A useful event record includes the following fields.

| Field | Grain or unit | Why it matters |
|---|---|---|
| Event identifier | One stable ID per agreement version or linked event | Connects terms, executions, forecasts, and claims. |
| Customer identifier | Retailer or legal customer | Prevents an allowance being applied to the wrong customer. |
| Store identifier | Store and channel | Supports exposure and control definitions. |
| Product identifier | SKU and pack hierarchy | Separates featured products and substitutes. |
| Eligible dates | Start, end, time zone, and basis | Defines whether a scan, invoice, or receipt qualifies. |
| Unit conversion | Consumer units per case or pack | Aligns shipment, scan, and financial quantities. |
| Shelf price | USD per consumer unit and date | Distinguishes funding from actual consumer treatment. |
| Invoice price | USD per invoice unit | Establishes supplier revenue before applicable terms. |
| Allowance | Rate, currency, base, caps, tiers | Calculates the obligation under the agreement. |
| Fixed fees | USD and performance condition | Captures non-volume funding. |
| POS quantity | Consumer units per store-SKU-day | Measures observed consumer sell-through. |
| Shipment quantity | Cases or units per customer-SKU-day | Measures supplier sell-in. |
| Receipts and inventory | Units and cutoff timestamps | Reconciles order timing and physical flow. |
| Availability | In-stock status or shelf availability | Identifies constrained observed demand. |
| Returns | Units, date, and reason | Prevents overstatement of net activity. |
| Execution evidence | Display, advertisement, price, store, date | Supports the exposure and contractual claim. |
| Claim identifier | Deduction or invoice number | Links settlement to the right event. |
| Approval and revision | Owner, timestamp, version | Establishes authorization and audit trail. |
| Baseline version | Method, training cutoff, scenario | Prevents silent post hoc changes. |

Define whether a day refers to store-local time or a central reporting time.
Reconcile store openings, closures, and assortment changes before comparing week totals.
A mapping change can create artificial growth if the same SKU is counted under two identifiers.
Maintain an explicit crosswalk for pack changes instead of inferring units from product descriptions.

### Product and location identifiers

GS1 provides standards for product identification, including the Global Trade Item Number (GTIN), and for location identification, including the Global Location Number (GLN).
Those identifiers can support consistent matching across organizations when correctly assigned and maintained.
A GTIN is not proof that a transaction is eligible for a promotion.
The agreement still needs a dated list of eligible items, pack conversions, customers, and locations.
Not every retailer data feed contains all GS1 identifiers, and internal mappings require governance.

An electronic data interchange (EDI) document can communicate a transaction in an agreed electronic format.
The presence of an EDI message does not by itself establish that a deduction is valid or that a display ran.
Trading partners must agree on the document standard, implementation guide, business rules, and reconciliation process.
Do not assume a particular document type is universally used by all retailers.

### Financial reporting

For entities applying International Financial Reporting Standards, IFRS 15 addresses revenue from contracts with customers, including variable consideration and consideration payable to a customer.
Whether a particular payment reduces transaction price or purchases a distinct service depends on the facts and applicable requirements.
The internal contribution model is a decision tool; it is not a substitute for the accounting determination.
Organizations applying other accounting frameworks need their own policies and professional review.

An accrual estimate may change when eligible scan evidence, performance documentation, or contractual interpretation changes.
Preserve the original estimate and every material revision.
Use a consistent event identifier so an approved adjustment does not become an untraceable plug.
Claims should reconcile to the general ledger without counting a cash deduction and its accrued obligation as two separate costs.

### Competition and consumer protection

In the United States, the Federal Trade Commission (FTC) describes the Robinson-Patman Act's treatment of certain discriminatory prices and promotional allowances or services.
Applicability depends on facts, statutory elements, defenses, jurisdiction, and current law.
This lesson does not decide whether any particular allowance is lawful.
Do not assume that a commercially attractive term, an algorithm's recommendation, or a historical practice supplies legal clearance.
Escalate differences in customer terms and services to qualified counsel under the organization's policy.

Consumer-facing price and advertising claims may also be subject to applicable federal, state, and local rules.
Verify advertised prices, dates, quantities, and qualification conditions with the responsible legal and compliance teams.
Do not infer compliance with reference-price or advertising rules from an increase in sales.
Other jurisdictions have different rules; obtain advice for the relevant market and date.

Competitor data and conduct introduce separate competition-law issues.
Use independently obtained, lawful inputs and established information-sharing controls.
Do not coordinate current or future prices, promotional timing, or customer allocation with competitors.
Where an industry association or shared data provider is involved, legal review should define permissible aggregation and access.

### Privacy and access

Store-level POS aggregates often suffice for promotion economics.
Shopper-level transaction data may reveal sensitive behavior when linked to identifiers.
Collect only data needed for an approved purpose and apply relevant privacy law and contractual restrictions.
Define access, retention, aggregation, and deletion rules with the data owner.
Avoid exporting individual-level records into an unrestricted spreadsheet merely because doing so is convenient.

### Quality checks before release

1. Confirm the event ID joins to exactly the intended contract version.
2. Compare eligible stores with the executed store list and record exceptions.
3. Check missing daily feeds separately from zero-sales days.
4. Convert all cases and packs to consumer units using dated mappings.
5. Check the price and allowance currency and tax treatment.
6. Reconcile scan totals to the retailer's submitted claim.
7. Flag duplicate claim lines and overlapping events.
8. Compare beginning inventory, receipts, sales, and ending inventory.
9. Confirm that regular and promotional margins use the same cost scope.
10. Version the baseline and later revisions.
11. Separate payments from accruals and ledger adjustments.
12. Record unresolved material exceptions and their owners.

A failed check does not necessarily invalidate the entire analysis.
It does determine what can be claimed and whether the decision must wait.
For example, missing 5% of stores may be tolerable only if the missingness is understood and its potential effect is bounded.
A missing contract rate is not resolved by choosing whichever historical rate makes the promotion appear profitable.

## 9. Industry practice and case evidence

### What real evidence can and cannot establish

Peer-reviewed promotion research distinguishes the sales bump during an event from displacement across time, brands, or categories.
That distinction motivates a broader horizon and portfolio boundary.
The exact effect varies by product, channel, consumer behavior, and promotion mechanics.
No published average is a substitute for a local counterfactual.

Blattberg, Briesch, and Fox's review, *How Promotions Work*, organizes empirical findings about promotion response.
Van Heerde, Leeflang, and Wittink's work on decomposing a sales promotion bump examines the sources of extra promoted-item sales.
These are research sources for the mechanisms, not evidence for the fictional USD 8,100 loss above.
Their empirical settings should be checked before transferring any result to a new market.

### Named public case: PepsiCo and the FTC

On January 17, 2025, the FTC announced a lawsuit alleging that PepsiCo provided a large retailer with preferential pricing or promotional benefits in violation of the Robinson-Patman Act.
The announcement presents the agency's allegations, not a judicial finding that the allegations were true.
On May 22, 2025, the FTC announced dismissal of the lawsuit.
The agency's dismissal announcement and contemporaneous commissioner statements are part of the public record.
The dismissal means this example must not be presented as a proven violation or a settled rule for all trade terms.

**Business inference:** a funding register should make customer eligibility, promotional services, evidence, and approval traceable so that a company can evaluate its own terms and respond to legal questions.
That inference is a governance recommendation, not a factual finding about PepsiCo's systems or conduct.
The case does not reveal the event-level promotion ROI, baseline, or consumer response used in this lesson.

### Practice scenario A — Grocery multipack

A supplier funds a temporary discount on a multipack before a holiday weekend.
Retailer scans rise, but other pack sizes decline.
The event is also advertised in a flyer, so price and media effects occur together.
The relevant assessment includes the holiday baseline, other supplier packs, execution, and later weeks.
The retailer may value traffic or basket effects that the supplier cannot measure from its own invoices.
A commercial review should show both ledgers rather than presume that either party's benefit transfers automatically.

If comparable stores receive the flyer but not the price discount, their outcomes may help distinguish treatments, subject to assignment and spillover assumptions.
If every store receives both treatments, the combined package is the estimand.
A post hoc regression cannot automatically recover separate causal effects from perfectly overlapping activities.

### Practice scenario B — Beauty display and fixed payment

A cosmetics supplier pays a fixed placement fee and supplies display materials.
The retailer sells through only part of the incremental stock because some stores set up the display late.
The correct actual treatment includes measured exposure, not merely the contract's planned display dates.
Store-level photographs or compliance audits can support a claim that a display existed, but evidence quality and sampling still matter.
The review distinguishes a weak offer from a strong offer that was not executed as agreed.

The fixed payment may be partially earned, fully earned, or disputed depending on the written agreement.
The analyst should not decide the legal interpretation by changing the economic formula.
Record the contractual interpretation and show an alternative settlement scenario if the dispute is material.

### Practice scenario C — Distributor forward buying

A manufacturer offers a case allowance on shipments during a defined month.
The distributor orders heavily before the period ends and sells the product at a normal pace over subsequent months.
The manufacturer's near-term shipment revenue rises, but much of the increase may be inventory movement rather than a change in final demand.
If no inventory data are available, a shipment-based sales bump has a wide interpretation range.
The contract still governs the allowance owed, even if incremental consumption is unproven.

A better later review compares shipments, receipts, distributor inventory, and sell-through across a window long enough to observe drawdown.
If products expire or warehouses are constrained, include incremental waste and storage cost only in the perspective that bears them.
Distinguish the contractual cost of an eligible shipment from the causal business return on the event.

### Practice scenario D — Online retailer coupon

An online retailer issues a digital coupon that is funded by the supplier only when redeemed.
The eligible base is redemptions, not every product view or shipment, if the agreement says so.
A coupon may be offered selectively to shoppers already likely to buy.
An apparent difference between redeemers and non-redeemers is therefore not automatically causal.
A randomized offer or encouragement design can improve identification, provided noncompliance and spillovers are analyzed.

Keep coupon cost, retailer fee, shipping subsidy, and product margin separate.
If the coupon affects another channel or another pack size, include those effects in the stated boundary.
An online event may have little retailer inventory distortion but still create consumer pull-forward.

### Practice scenario E — Same spend, different payment bases

A fictional supplier is comparing two retail agreements for an established household product.
Both agreements advertise the same USD 1.00 per-unit nominal rate.
The regular supplier contribution is USD 4.00 per unit.
The no-event event-week baseline is 6,000 consumer units.
The forecast under either agreement is 8,000 consumer units.
The forecast is not a commitment and may change when the retailer's execution differs.

Agreement A pays USD 1.00 on every eligible consumer scan.
Agreement B pays USD 1.00 only on eligible scans above an agreed 6,000-unit threshold.
Assume, solely for this comparison, that the threshold and eligible scans are accurately documented and that the retailer accepts both alternatives.
Assume there are no fixed fees, other event costs, sibling effects, or later effects.

| Component | Agreement A | Agreement B |
|---|---:|---:|
| Forecast event scans | 8,000 units | 8,000 units |
| Estimated no-event scans | 6,000 units | 6,000 units |
| Incremental scans | 2,000 units | 2,000 units |
| Contribution before allowance | USD 8,000 | USD 8,000 |
| Contractual funding base | 8,000 units | 2,000 units |
| Funding at USD 1.00 per eligible unit | USD 8,000 | USD 2,000 |
| Incremental contribution | USD 0 | USD 6,000 |

The arithmetic illustrates why a rate without an eligibility base is incomplete.
Agreement B's contractual threshold happens to equal the analytical baseline here.
In practice, the contracted threshold and the estimated counterfactual can differ.
A contractual increment is not automatically a causal increment.
For example, a holiday may raise no-event sales above the contractual threshold.
Conversely, a threshold set above a credible no-event forecast may exclude some genuinely incremental sales from payment.

The example does not imply that retailers will offer the same shelf price or display for both funding structures.
Agreement B may transfer more demand risk to the retailer and therefore produce a different execution or forecast.
Evaluate the actual commercial package rather than assuming identical behavior.
Counsel and finance should review the specific terms and accounting treatment before approval.

### Practice scenario F — A dispute over the eligibility base

A fictional supplier agrees to pay a retailer USD 0.75 per eligible consumer unit during a seven-day event.
The signed schedule lists 100 stores, one product identifier, and store-local dates from Monday through Sunday.
The retailer submits a claim for 20,000 units, equivalent to USD 15,000.
The supplier's initial data match finds 18,400 eligible units, equivalent to USD 13,800.
The USD 1,200 gap is a reconciliation difference, not automatically a saving or an additional expense.

The investigation classifies the 1,600-unit difference as follows:

| Difference category | Units | Required action |
|---|---:|---|
| Store file arrived after the first extraction | 600 | Refresh the versioned data and verify the store was eligible. |
| Two stores not listed in the signed schedule | 400 | Check for an approved amendment or reject under the applicable terms. |
| Scans timestamped after the local closing date | 300 | Check time-zone conversion and the contractual cutoff. |
| Possible duplicated transaction records | 200 | Match transaction identifiers and retailer corrections. |
| Different product pack in the claim | 100 | Verify the eligible product list and dated pack mapping. |
| **Total difference** | **1,600** | Resolve each category with evidence. |

The first 600 units could increase the supported claim by USD 450 if the missing feed is valid.
The remaining categories cannot be resolved from the summary alone.
The claim owner should record the supporting evidence, decision, date, and approver for each adjustment.
The finance owner should update the estimate only in accordance with the organization's policy.
The retailer-facing owner should communicate the open items without promising payment before authorization.

If the contract specified an invoice-date rather than a scan-date basis, the required evidence and eligible quantity would be different.
The calculation must follow the signed terms, not the name of a report.
If a dispute remains at period close, label the resulting financial estimate and seek the appropriate accounting determination.
An open claim is not the same as an invalid claim.

### Practice scenario G — A baseline exposed to a holiday

A fictional retailer group normally sells about 2,000 units per week in a store cluster.
The supplier wants to evaluate a promotion in a holiday week.
The immediately preceding non-holiday week records 2,100 units.
Unpromoted comparable clusters sell 25% more during the holiday week than in their own preceding week.
If the treated cluster would have experienced the same proportional change without the event, a simple counterfactual is 2,100 × 1.25 = 2,625 units.

Suppose the treated cluster actually sells 3,000 units in the holiday week.
The simple adjusted lift is 3,000 − 2,625 = 375 units.
An analyst using the prior week alone would report 3,000 − 2,100 = 900 units.
The 525-unit difference is the assumed holiday adjustment, not observed treatment effect.
The 375-unit figure is also an estimate, not proof.

The assumption of a common proportional change could fail if the treated stores have a different shopper mix or opening schedule.
The unpromoted stores might run a different competing promotion.
Their inventory might be better or worse than the treated cluster's.
Comparing several earlier holiday periods and pre-event trends can reveal weaknesses, but cannot prove the counterfactual.
If random assignment is possible in a future holiday event, pre-specify the matched clusters and evaluate the new event separately.

### Practice scenario H — Retailer and supplier perspectives

Consider a fictional event with 1,000 additional consumer units under a common causal estimate.
Before the event, the supplier earns USD 3.00 contribution per incremental unit.
The supplier pays USD 2,500 in scan funding and USD 500 in a display fee.
Ignoring all other effects, supplier incremental contribution is USD 3,000 − USD 2,500 − USD 500 = USD 0.

The retailer's incremental ledger must be calculated from its own regular and event consumer prices, acquisition costs, labor, and funding receipts.
Suppose, illustratively, that its incremental consumer revenue is USD 8,000 and the incremental acquisition and operating costs are USD 7,000 before funding.
The retailer receives the USD 3,000 total supplier transfer, so its incremental contribution is USD 4,000 under this separate assumed ledger.
The retailer's USD 7,000 cost includes its incremental purchase cost under the example's invoice convention; it is not the supplier's variable production cost.
The illustration deliberately assumes the same eligible units and measurement horizon for both parties.

The combined incremental contribution is USD 0 + USD 4,000 = USD 4,000.
Equivalently, the USD 3,000 transfer cancels between the parties when the ledgers are consolidated.
The combined result is not USD 7,000: adding the retailer's funding receipt without also including the supplier's funding cost would count the transfer as value created.
The combined result also excludes any external effect not in the two ledgers, such as a third party's distribution cost.

This case does not say that the supplier should necessarily accept a zero return.
A repeat decision may depend on alternative uses of funds, longer-term effects, contractual strategy, and uncertainty.
Those considerations should be explicit rather than hidden inside the retailer's contribution.

### Practical scorecard for an operating review

A scorecard is useful only when each metric has an owner, data cutoff, and response rule.
The following measures are examples, not universal targets.

| Measure | Definition | Business question |
|---|---|---|
| Eligible exposure coverage | Executed eligible store-days divided by planned eligible store-days | Did the planned treatment reach the intended locations? |
| Price compliance | Store-days with the agreed observed price divided by measured eligible store-days | Did shoppers encounter the intended price? |
| Availability coverage | Eligible store-days with confirmed available stock divided by measured eligible store-days | Could observed demand be fulfilled? |
| POS feed completeness | Received expected store-days divided by expected store-days | Is the observed sales base sufficiently complete? |
| Baseline error in prior holdouts | Predicted no-event units minus observed control units, with stated aggregation | Does the method systematically over- or under-predict? |
| Gross event lift | Actual promoted units minus baseline promoted units | Did the featured item change during the event? |
| Net portfolio contribution | Sum of affected actual contribution less counterfactual contribution, less event costs | Did the scoped supplier business improve? |
| Funding variance | Settled or latest estimated eligible funding minus approved forecast funding | Why did the obligation differ from plan? |
| Claim exception rate | Unresolved claim lines divided by reviewed claim lines | Is the settlement process accumulating unresolved work? |
| Later-period difference | Later actual contribution minus later baseline contribution | Is there evidence of a post-event dip or halo? |
| Review coverage | Material closed events with completed review divided by all material closed events | Are failures and successes both being learned from? |

Report the numerator and denominator alongside each percentage.
A high price-compliance percentage based on a small, selectively sampled set of stores can be misleading.
If exposure is measured only in stores that supplied photographs, describe how the stores were selected.
Do not treat an unobserved display as a confirmed failure unless the evidence system supports that classification.

Review cadence should reflect the lifecycle of the event.
An in-event operational meeting can respond to availability or missing execution.
A later commercial review can estimate the event and post-event effects once the chosen window has elapsed.
A settlement review may occur later still when all relevant deductions and evidence are available.
Link these reviews through the same event ID rather than waiting for one meeting to serve every purpose.

### Business decision log

At minimum, retain the approved alternative and its no-event comparator.
Record the forecast, the inventory plan, the target, and contractual commitments in separate fields.
Document which quantities were observed and which were estimated.
Record the approved rate, eligible base, cap, and fixed payments.
State the counterfactual method and the pre-event freeze date.
Attach the actual execution evidence and explain missing coverage.
Show the affected portfolio and the length of the post-event window.
Provide the incremental contribution bridge in USD and the uncertainty assessment.
List unresolved claims, the owner of each exception, and the next settlement date.
Conclude with a specific decision: repeat unchanged, renegotiate, redesign, test, or stop.
If a strategic reason overrides a negative estimated contribution, name the reason and approver rather than altering the calculation.

### What to carry into a business review

- Ask whether the evidence identifies the complete commercial package or only a correlated sales pattern.
- Show actual execution coverage and any incomplete eligibility mapping.
- Separate supplier, retailer, and combined economics.
- Compare the incremental USD outcome with budget, inventory, and service constraints.
- Treat the public legal case as a prompt for documented review, not as a universal rule.
- Retain both successful and failed events to reduce selective reporting.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Gross units called incremental | Report shows total scans but no baseline. | Regular demand is credited to the event. | Inflated return and excess funding. | Show actual, baseline, and difference. |
| Funding on baseline ignored | Allowance multiplied only by lift units. | Every eligible sale may receive the discount. | Profit is overstated. | Apply contract rate to the contractual eligible base. |
| Allowance deducted twice | Net event margin and separate full allowance appear in the same bridge. | One payment is charged in two places. | Profit is understated. | Trace each cost to one bridge line. |
| Wrong baseline window | Holiday week compared with an ordinary week. | Seasonality is attributed to the treatment. | Biased lift. | Use seasonal controls or validated covariates. |
| Contaminated historical baseline | Prior promotional weeks remain in the training data without labels. | Expected regular demand contains past event effects. | Biased or unstable lift. | Label and handle prior interventions explicitly. |
| Controls receive treatment | Control stores run a nearby display or price change. | Untreated counterfactual is contaminated. | Effect is attenuated or otherwise biased. | Monitor control execution and assess spillovers. |
| Store selection after results | Treatment stores are chosen using observed sales lift. | Selection uses an outcome of interest. | Spurious success. | Define assignment and population before launch. |
| Cannibalization omitted | Only featured SKU is in the report. | Sales move from a sibling product. | Portfolio contribution is overstated. | Include plausible substitutes or bound the omission. |
| Pull-forward omitted | Review ends on the last event day. | Later sales are credited as new demand. | Short-term lift is mistaken for sustained growth. | Extend the horizon and report later cells. |
| Effects double-counted | Portfolio ledger already includes siblings and an extra cannibalization penalty. | Same loss is subtracted twice. | Profitable events may be rejected. | Use mutually exclusive cell definitions. |
| Shipment mistaken for consumption | Shipments spike and retailer inventory is unavailable. | Forward buying increases sell-in. | Apparent demand exceeds consumer response. | Reconcile inventory and obtain POS or bound uncertainty. |
| Stockout treated as weak demand | High-traffic stores show zero scans and low on-hand. | Demand is censored by availability. | True demand response is underestimated. | Track availability and analyze constrained exposure. |
| Missing feed treated as zero | Entire store-day files are absent. | Data loss appears as a sales drop. | Incorrect baseline and claim. | Separate missing from observed zero and investigate. |
| Pack conversion error | Case and unit totals disagree by pack size. | Different transaction units are combined. | Allowance or contribution is misstated. | Use dated SKU conversion records. |
| Terms misapplied | Claim uses a shipment rate when agreement specifies scans. | Wrong eligible base or dates. | Overpayment, underpayment, or dispute. | Match each claim to signed terms. |
| Unsupported display fee | Payment requested without proof of agreed execution. | Contract condition cannot be verified. | Invalid or disputed funding. | Define evidence and exception rules in advance. |
| Accrual treated as cash | Ledger expense is added to a later deduction. | Liability and settlement are counted separately. | Trade cost is overstated. | Reconcile accrual, claim, and cash ledger. |
| Unearned spend ignored | Post-event claim exceeds approved agreement. | Unmatched deduction is accepted silently. | Budget leakage and weak accountability. | Investigate and document true-up or dispute. |
| Attribution beyond evidence | Report assigns precise shares to price, display, and media that always coincided. | Components lack independent variation. | Misleading optimization recommendations. | Evaluate the combined package or design separate variation. |
| Biased treatment by stock | Better-stocked stores are selected into promotion. | Availability affects both assignment and sales. | Promotion effect is confounded. | Randomize or adjust with defensible pre-treatment data. |
| Ignored operational burden | Incremental freight and overtime appear after review. | Real costs post outside trade-spend accounts. | Reported value exceeds economic value. | Include incremental cross-functional costs. |
| Ratio optimized in isolation | Highest ROI event selected despite small absolute return. | Denominator differs from value objective. | Lower total contribution. | Optimize feasible contribution under constraints. |
| Unmodeled legal boundary | Customer terms vary with no documented review. | Potentially relevant law and facts are missed. | Regulatory or contractual exposure. | Escalate to qualified legal and compliance owners. |
| Scope changes after outcome | Baseline or store list is replaced only for poor events. | Analyst discretion follows the result. | Selection bias and weak trust. | Freeze plans and version justified revisions. |

A red flag is a prompt to investigate, not proof that the result is wrong.
A review should classify each material issue as resolved, bounded, or decision-blocking.
Document who accepted residual risk and why.

## 11. Assumptions and limits

The worked example is conditional on the following ledger.
A real event requires its own assumptions and evidence.

| Assumption | Why needed | What breaks it | Practical test |
|---|---|---|---|
| Baseline of 10,000 units is credible. | Establishes the no-event comparison. | Holiday or distribution changes differ from the model. | Check controls, pre-period fit, and alternative baselines. |
| Event scans equal relevant sell-in in this case. | Allows one unit count to represent the supplier result. | Retailer inventory rises or falls over the horizon. | Reconcile receipts, inventory, and scans. |
| All 14,500 scans qualify for funding. | Sets the USD 17,400 allowance. | Ineligible dates, stores, products, or returns. | Match scans to signed eligibility rules. |
| Supplier regular contribution is USD 4.00 per unit. | Values the counterfactual and later units. | Freight, costs, price, or mix change. | Reconcile dated net revenue and variable costs. |
| Event contribution is USD 2.80 per unit. | Values actual promoted sales. | Additional allowances or costs are omitted. | Trace the invoice and all related funding terms. |
| Sibling decline is causal. | Attributes USD 2,700 loss to the event. | Competitor activity or a sibling stockout causes the decline. | Compare suitable controls and availability. |
| Later shortfall is causal. | Attributes USD 2,400 loss to pull-forward. | Another promotion or supply interruption affects later weeks. | Extend controls and inspect later interventions. |
| Effects are disjoint. | Makes the bridge additive. | A unit is counted in multiple product or time cells. | Reconcile every ledger row to a unique cell. |
| Fixed fee is earned and incremental. | Requires USD 1,500 in the event comparison. | Service was not performed or fee would occur anyway. | Review contract, proof, and no-event alternative. |
| Execution cost is incremental. | Requires USD 2,100 in the event comparison. | Cost is fixed regardless of the decision. | Review labor, freight, and alternative deployment. |
| No material unmeasured halo exists. | Prevents missing a positive contribution. | Complementary products gain because of the event. | Expand the portfolio or bound plausible halo. |
| No material capacity displacement exists. | Avoids omitting foregone contribution. | Promotion uses scarce production or transport capacity. | Check capacity utilization and displaced orders. |
| Later margin is regular USD 4.00. | Values the 600-unit later shortfall. | Later price or variable cost differs. | Use dated cell-level margins. |
| Four weeks capture the important effect. | Makes the horizon usable for a decision. | Stockpiling or inventory drawdown continues beyond it. | Extend the window and report remaining exposure. |
| Other effects stay fixed in sensitivity. | Makes the algebraic threshold interpretable. | Higher volume changes substitution, fees, or capacity costs. | Model joint scenarios rather than a single-variable change. |
| Whole units and positive event margin apply. | Supports the stated rounding and break-even formula. | Returns exceed sales or event margin is nonpositive. | Check inputs and solve the actual contract case separately. |

The stated USD −8,100 is an illustrative estimate, not a confidence interval.
The true incremental contribution could differ materially if the assumed causal effects are wrong.
A positive lower-bound result under several defensible scenarios is stronger evidence than a positive point estimate under one favorable assumption.
Even a well-identified past effect may not transport to a different customer, price, placement, or season.

## 12. Exercises

### Exercise 1 — Define the decision

A beverage supplier is asked to repeat a two-week display for a single retailer.
The retailer also sells two substitute packs and one complementary snack from the supplier.
The proposed payment is USD 0.50 per qualifying scan plus a USD 3,000 fixed display fee.

**Deliverable:** a one-page decision brief.
**Acceptance criteria:** identify the economic owner, comparison, affected items, time horizon, unit, funding base, and an operational constraint.
Distinguish the retailer view from the supplier view and label any unobserved counterfactual.
**Time:** 12 minutes.

### Exercise 2 — Baseline and lift

A store group sells 12,400 units during a promoted week.
A pre-specified comparison estimates 10,000 units would have sold without the event.
The group had no known stockouts and no simultaneous other promotion in the case data.

**Deliverable:** calculate absolute lift, relative lift, and uplift factor; state one remaining baseline risk.
**Acceptance criteria:** show the denominator and explain why the baseline is not observed.
Do not claim that absence of known stockouts proves causality.
**Time:** 8 minutes.

### Exercise 3 — Funding on all eligible units

An event has a baseline of 8,000 units and actual eligible scans of 11,000 units.
Regular contribution is USD 5.00 per unit and the scan allowance is USD 1.00 per actual eligible unit.
A fixed event fee of USD 1,500 applies; ignore all other effects and costs for this exercise.

**Deliverable:** calculate incremental contribution with both the actual-minus-baseline and incremental-units bridges.
**Acceptance criteria:** both methods agree; the scan allowance applies to all 11,000 qualifying units exactly once.
**Time:** 12 minutes.

### Exercise 4 — Portfolio and later-period effects

Continue Exercise 3.
Assume a sibling loses 400 units at USD 2.50 contribution per unit and the promoted item loses 300 later units at USD 5.00 contribution per unit.
Assume both losses are caused by the event and neither is already included in the initial bridge.

**Deliverable:** calculate the revised total and explain the decision against a zero-contribution threshold.
**Acceptance criteria:** show two separate loss lines and avoid charging the event fee twice.
**Time:** 10 minutes.

### Exercise 5 — Critique a control design

A retailer selects its 20 highest-growth stores for a display and uses its 20 lowest-growth stores as controls.
The selection is based on growth measured during the display period.
The analyst reports a 25% treatment effect without an uncertainty interval.

**Deliverable:** write a short critique and propose a feasible improved design.
**Acceptance criteria:** identify outcome-based selection, describe pre-event assignment or randomization, address spillovers, and specify the unit of analysis.
State at least one reason an interval may be wide.
**Time:** 15 minutes.

### Exercise 6 — Funding and compliance

A supplier agrees to USD 0.80 per qualifying consumer scan and a USD 2,000 display fee if the display is documented.
The retailer submits 9,000 qualifying scans, a USD 2,000 fee with no proof, and a separate USD 1,000 deduction that has no contract reference.
The supplier has already accrued USD 9,200.

**Deliverable:** prepare an initial reconciliation table and identify the issues for commercial, finance, and legal review.
**Acceptance criteria:** compute the scan obligation, do not assume the fee is earned, do not count the accrual as an extra cost, and treat the unmatched deduction as unresolved.
Explain that different customer terms may require legal review without declaring the arrangement lawful or unlawful.
**Time:** 15 minutes.

### Exercise 7 — Flow and timing

A retailer begins the month with 2,000 units, receives 9,500 units, sells 7,000 units, returns 200 units to the supplier, and reports ending inventory of 4,000 units.
Assume no transfers or other removals were recorded.

**Deliverable:** calculate the implied ending inventory, the unexplained difference, and two possible investigations.
**Acceptance criteria:** identify whether reported inventory is above or below the identity and explain why shipment growth is not automatically consumption growth.
**Time:** 10 minutes.

### Exercise 8 — Choose under a budget constraint

An analyst estimates three independent candidate events, each with no other capacity constraints.
Event A needs USD 4,000 of budget and creates USD 3,000 of incremental contribution.
Event B needs USD 6,000 and creates USD 4,000.
Event C needs USD 10,000 and creates USD 6,500.
The total budget is USD 10,000, and each event is indivisible.

**Deliverable:** choose a feasible set that maximizes estimated contribution and compare it with ranking by individual ROI.
**Acceptance criteria:** enumerate feasible combinations, report the total USD contribution, and state one uncertainty that could change the choice.
**Time:** 12 minutes.

### Exercise 9 — Post-event governance design

A company has fragmented event calendars, customer deductions, and scan data.
Different teams use different event IDs, and successful events are reviewed more often than failures.

**Deliverable:** design a minimum event-to-settlement workflow and a post-event scorecard.
**Acceptance criteria:** include a versioned agreement, baseline freeze, execution evidence, claims reconciliation, exception owner, review timing, uncertainty, and a learning decision.
Address both the positive and negative events rather than selecting only winners.
**Time:** 20 minutes.

## 13. Self-check answers

### Exercise 1 — Scope rubric

A strong brief names the supplier as the initial economic owner and the retailer as a separate stakeholder.
It compares the proposed display with the no-display, no-promotion alternative over the same stores and period.
It includes both substitute packs and the complementary snack, with a justified later window.
It records USD per eligible scan, the fixed fee, eligible dates, and proof of display.
It distinguishes forecast, replenishment plan, target, and contractual commitment.
It identifies an inventory or labor constraint and a named approval owner.
A weak brief reports only promoted-item sales or assumes the retailer's outcome equals the supplier's.

### Exercise 2 — Lift calculation

Absolute lift is 12,400 − 10,000 = 2,400 units.
Relative lift is 2,400 / 10,000 = 24%.
The uplift factor is 12,400 / 10,000 = 1.24.
The 10,000-unit baseline is a counterfactual estimate, not a recorded sale.
For example, a local holiday or a competitor's price change could affect the comparison.
A sound answer explains how the comparison method addresses or fails to address that risk.

### Exercise 3 — Two equivalent bridges

Event margin is USD 5.00 − USD 1.00 = USD 4.00 per unit.
Actual event contribution is 11,000 × USD 4.00 = USD 44,000.
Baseline contribution is 8,000 × USD 5.00 = USD 40,000.
The difference is USD 4,000 before the fixed fee and USD 2,500 after the USD 1,500 fee.

The cross-check is 3,000 incremental units × USD 5.00 = USD 15,000.
Funding on all 11,000 scans is USD 11,000.
USD 15,000 − USD 11,000 − USD 1,500 = USD 2,500.
Applying the allowance only to the 3,000 incremental units would incorrectly overstate contribution by USD 8,000.

### Exercise 4 — Net affected portfolio

The sibling loss is 400 × USD 2.50 = USD 1,000.
The later promoted-item loss is 300 × USD 5.00 = USD 1,500.
Starting with USD 2,500 from Exercise 3, the total is USD 2,500 − USD 1,000 − USD 1,500 = USD 0.
The event exactly meets a zero threshold under the exercise's assumptions.
An organization may still reject it if uncertainty, strategic alternatives, or scarce resources make the opportunity unattractive.
The zero point estimate does not establish that the causal return is precisely zero.

### Exercise 5 — Design rubric

Selecting stores using growth during the event uses the outcome to create the groups.
The treated group is mechanically enriched for high growth, so the 25% difference is not an identified treatment effect.
A stronger design assigns comparable stores or geographic clusters before the event, ideally randomly.
It freezes the eligible population and estimand, monitors controls for contamination, and records actual execution.
Analysis should reflect the assigned cluster or store as the unit of randomization.
An interval may be wide because there are only 40 stores, store outcomes vary substantially, or treatment is unevenly executed.
The design should also evaluate whether shoppers travel between treated and control stores.

### Exercise 6 — Reconciliation

| Line | Initial amount | Status |
|---|---:|---|
| Eligible scan obligation | 9,000 × USD 0.80 = USD 7,200 | Supported if scans and contract eligibility are validated. |
| Fixed display fee | Up to USD 2,000 | Unresolved until the required evidence and terms are reviewed. |
| Unreferenced deduction | USD 1,000 claimed | Unresolved; seek contract reference and supporting details. |
| Accrual | USD 9,200 booked | Estimate to reconcile, not an additional USD 9,200 event cost. |

The supported scan amount alone is USD 2,000 below the accrual.
If the display fee is earned, scan funding plus the fee equals USD 9,200 before any valid additional deduction.
The USD 1,000 claim may be valid, duplicate, or unsupported; the provided facts do not decide it.
Commercial obtains evidence and agreement details, finance reconciles the liability and deduction, and legal reviews material contractual or competition questions as appropriate.

### Exercise 7 — Inventory identity

Implied ending inventory is 2,000 + 9,500 − 7,000 − 200 = 4,300 units.
Reported inventory is 4,000 units, or 300 units below the implied balance.
Possible explanations include unrecorded shrink, damage, transfers, a timing difference, or a unit-mapping error.
Investigate transaction cutoffs and physical inventory adjustments before assigning a cause.
The 9,500-unit receipt is not the 7,000-unit consumer sell-through; some receipts may remain in inventory.

### Exercise 8 — Feasible portfolio

The feasible nonempty choices are A, B, C, and A+B.
A costs USD 4,000 and contributes USD 3,000.
B costs USD 6,000 and contributes USD 4,000.
C costs USD 10,000 and contributes USD 6,500.
A+B costs USD 10,000 and contributes USD 7,000.
A+C, B+C, and A+B+C exceed the USD 10,000 budget.
Choose A+B on the stated point estimates, with USD 7,000 contribution.
Individual ROIs are 75% for A, approximately 66.67% for B, and 65% for C; selecting A then B also happens to work here.
That coincidence does not prove ratio ranking works for all indivisible portfolios.
If A and B share a cannibalized customer base, their combined contribution may be less than the sum; the independence assumption must be checked.

### Exercise 9 — Governance rubric

A strong workflow creates a stable event ID and stores approved contract terms, eligible items, and the required proof.
It freezes the baseline and planned analysis before launch and logs execution changes.
It reconciles scans, shipments, claims, accruals, and deductions with explicit exception ownership.
It schedules a review after the later-effect window and a separate settlement close when claims arrive.
The scorecard reports actual and baseline cells, contribution bridge, uncertainty, execution coverage, and the next decision.
It includes all material events, including failures, and preserves original forecasts alongside revisions.
A weak design closes a claim solely because an accrual exists or reports only sales lift for selected winners.

## 14. Interview questions

1. **How would you determine whether a promotion was incremental?**  
   Strong: define a no-event counterfactual, affected scope, time horizon, and causal design.  
   Weak: compare event sales with zero or assume the last week is always comparable.
2. **Why can a 45% sales lift still lose money?**  
   Strong: explain funding on baseline units, changed margin, substitution, later demand, and fixed costs.  
   Weak: equate higher revenue with higher profit.
3. **How does sell-in differ from sell-through?**  
   Strong: discuss orders, inventory changes, consumer scans, and a flow reconciliation.  
   Weak: treat a retailer purchase as proof of end-consumer consumption.
4. **What is the right denominator for promotion ROI?**  
   Strong: specify the numerator, funding denominator, time horizon, and comparability limits.  
   Weak: quote a percentage without defining cost or return.
5. **How would you measure cannibalization?**  
   Strong: compare affected product contribution with a credible no-event estimate and avoid double counting.  
   Weak: label every decline in a sibling item as causal.
6. **How would you separate price from display effects?**  
   Strong: require independent variation or a factorial design and acknowledge execution differences.  
   Weak: claim separate effects when all stores receive both at the same time.
7. **What would you do with a deduction larger than the accrual?**  
   Strong: match contract, eligible base, proof, prior payments, and claim; resolve the difference with owners.  
   Weak: book the entire difference without investigation or add the full deduction to the existing accrual.
8. **What changes if the promotion is capacity-constrained?**  
   Strong: check availability and value displaced feasible demand without counting it twice.  
   Weak: treat observed sales as unconstrained demand.
9. **How do you prevent post hoc baseline manipulation?**  
   Strong: freeze the original analysis, log revisions, document evidence, and review all material events.  
   Weak: choose the baseline that best supports the account team's target.
10. **How would you handle trade terms that differ across customers?**  
    Strong: document terms and facts, follow policy, and involve qualified legal counsel.  
    Weak: infer legal compliance from commercial popularity or from a single historical case.
11. **When is a post-event dip not proof of pull-forward?**  
    Strong: identify competing events, stockouts, seasonality, inventory movement, and baseline uncertainty.  
    Weak: assign every later missing unit to the earlier promotion.
12. **Would you choose the highest-ROI promotion?**  
    Strong: optimize incremental contribution subject to budget, capacity, dependencies, and risk.  
    Weak: rank incomparable percentages and ignore total value.

## 15. Key terms

Each term is defined in [Definitions and vocabulary](#4-definitions-and-vocabulary) and applied in the linked section.

- **Accrual:** estimated obligation before settlement; see [funding mechanics](#step-8--price-the-funding-and-obligations).
- **Actual:** observed event outcome for the defined scope; see [worked volumes](#counterfactual-and-actual-volumes).
- **Baseline:** estimate of the no-event outcome; see [baseline selection](#step-3--choose-a-defensible-no-event-baseline).
- **Break-even lift:** lift at zero incremental contribution under assumptions; see [sensitivity](#break-even-and-sensitivity).
- **Cannibalization:** event-caused loss elsewhere in the portfolio; see [affected effects](#step-7--measure-cannibalization-halo-and-later-effects).
- **Claim:** request or deduction for a contractual payment; see [settlement](#settlement-and-interpretation).
- **Commitment:** approved or contractual obligation; see [decision framing](#forecast-plan-target-and-commitment).
- **Control group:** untreated comparison population; see [baseline selection](#step-3--choose-a-defensible-no-event-baseline).
- **Contribution:** relevant revenue less variable and event-specific costs; see [bridge mechanics](#step-6--calculate-the-promoted-item-contribution-bridge).
- **Counterfactual:** outcome that would have occurred without treatment; see [mental model](#3-the-core-mental-model).
- **Deduction:** amount withheld from a payment, potentially linked to a claim; see [funding mechanics](#step-8--price-the-funding-and-obligations).
- **Display fee:** payment for specified placement or service; see [commercial terms](#commercial-terms).
- **Event:** identified occurrence of an offer; see [treatment record](#step-2--record-the-event-treatment).
- **Execution:** actual delivery of the agreed promotion; see [industry practice](#practice-scenario-b--beauty-display-and-fixed-payment).
- **Fixed fee:** payment not mechanically proportional to eligible units; see [commercial terms](#commercial-terms).
- **Forecast:** estimate of future outcomes under specified conditions; see [decision framing](#forecast-plan-target-and-commitment).
- **Forward buying:** retailer purchasing stock ahead of consumption; see [distributor scenario](#practice-scenario-c--distributor-forward-buying).
- **Gross lift:** event-period difference before other affected cells; see [worked volumes](#counterfactual-and-actual-volumes).
- **Halo:** event-caused benefit to another item or channel; see [affected effects](#step-7--measure-cannibalization-halo-and-later-effects).
- **Incremental contribution:** actual less no-event contribution; see [portfolio bridge](#portfolio-and-later-period-bridge).
- **Incremental units:** actual less no-event units; see [worked volumes](#counterfactual-and-actual-volumes).
- **Off-invoice allowance:** agreed invoice price reduction; see [funding mechanics](#step-8--price-the-funding-and-obligations).
- **Plan:** chosen operational action; see [decision framing](#forecast-plan-target-and-commitment).
- **Point of sale:** checkout record of consumer transactions; see [minimum data](#minimum-data-contract).
- **Post-event dip:** later sales below baseline; see [affected effects](#step-7--measure-cannibalization-halo-and-later-effects).
- **Pull-forward:** purchase moved from a later period; see [time boundary](#step-4--set-the-portfolio-and-time-boundary).
- **Scan allowance:** funding per eligible consumer sale; see [commercial terms](#commercial-terms).
- **Sell-in:** supplier shipment or invoice to a retailer; see [physical reconciliation](#step-5--reconcile-physical-volumes).
- **Sell-through:** sales from retailer to consumer; see [physical reconciliation](#step-5--reconcile-physical-volumes).
- **Settlement:** resolution of an obligation against a claim and payment; see [settlement](#settlement-and-interpretation).
- **Spillover:** treatment effect on nominally untreated units; see [baseline selection](#step-3--choose-a-defensible-no-event-baseline).
- **Stockout:** lack of availability that constrains sales; see [failure modes](#10-failure-modes-and-red-flags).
- **Target:** desired performance threshold; see [decision framing](#forecast-plan-target-and-commitment).
- **Trade-spend ROI:** defined return divided by defined funding; see [settlement](#settlement-and-interpretation).
- **Treatment:** intervention being assessed; see [treatment record](#step-2--record-the-event-treatment).
- **True-up:** documented adjustment to an estimate or settlement; see [funding mechanics](#step-8--price-the-funding-and-obligations).

## 16. Sources

Sources support the concepts and legal context; the numerical case and exercises are fictional.
Accessed for this lesson in 2026 unless a specific publication date is stated.

1. Blattberg, Robert C., Richard Briesch, and Edward J. Fox. “How Promotions Work.” *Marketing Science* 14, no. 3, supplement (1995): G122–G132. [INFORMS journal](https://pubsonline.informs.org/doi/10.1287/mksc.14.3.G122). A research review of promotion effects and mechanisms.
2. van Heerde, Harald J., Peter S. H. Leeflang, and Dick R. Wittink. “Decomposing the Sales Promotion Bump with Store Data.” *Marketing Science* 23, no. 3 (2004): 317–334. [INFORMS journal](https://pubsonline.informs.org/doi/10.1287/mksc.1040.0061). Research on the sources of promotion-period sales changes; check setting before generalizing.
3. U.S. Federal Trade Commission. [Price Discrimination: Robinson-Patman Violations](https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws/price-discrimination-robinson-patman-violations). Official general guidance; fact-specific legal advice is still required.
4. U.S. Federal Trade Commission. [FTC Sues PepsiCo for Rigging Soft Drink Competition](https://www.ftc.gov/news-events/news/press-releases/2025/01/ftc-sues-pepsico-rigging-soft-drink-competition), January 17, 2025. Agency allegations, not findings of liability.
5. U.S. Federal Trade Commission. [FTC Dismisses Lawsuit Against PepsiCo](https://www.ftc.gov/news-events/news/press-releases/2025/05/ftc-dismisses-lawsuit-against-pepsico), May 22, 2025. Procedural update to the preceding case.
6. U.S. Federal Trade Commission. [Advertising and Marketing Basics](https://www.ftc.gov/business-guidance/advertising-marketing/advertising-marketing-basics). Official overview; applicable state, local, and non-U.S. rules may differ.
7. IFRS Foundation. [IFRS 15 Revenue from Contracts with Customers](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-15-revenue-from-contracts-with-customers/). Authoritative standard information for entities applying IFRS; accounting treatment requires fact-specific review.
8. GS1. [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin). Standard identifier information, not a promotion eligibility determination.
9. GS1. [Global Location Number (GLN)](https://www.gs1.org/standards/id-keys/gln). Standard location identifier information, not proof of event execution.
10. GS1. [Electronic Data Interchange](https://www.gs1.org/standards/edi). Overview of trading-partner data exchange; implementation depends on partner agreements.

## 17. Links

- [Module 16 topic index](_index.md): topic sequence and course position.
- [Module 16 survey](../../../course/16_pricing_promotions_revenue_management.md): overview and course context.
- [Lab 16.3 — Trade promotion incrementality](../../../course/16_pricing_promotions_revenue_management.md#lab-163--trade-promotion-incrementality): apply baseline and post-promotion analysis.
- [Own- and cross-price elasticity](03_elasticity-identification.md): identify price response and substitution.
- [Markdown optimization with salvage](05_markdown-optimization.md): evaluate markdown timing and remaining inventory.
- [Pricing decision stack](01_pricing-decision-stack.md): locate promotion terms among commercial decisions.
