# Fashion and apparel: buying, sourcing, markdown, and returns

> Module 29 — Industry Electives: Sector Primers  
> Topic 06 of 08  
> Estimated study time: 4 hours plus lab  
> Prerequisites: Modules 14, 15, and 28  
> Version 2026.5

## 1. Why this topic exists

A fashion buyer must commit money before knowing whether shoppers will like a style.
A supplier often needs a fabric reservation, approved sample, and production slot months before the first selling week.
The selling season may be shorter than the replenishment lead time.
The buyer therefore cannot assume that observing strong sales in week one makes a new order useful in week three.
Nor can the buyer assume a weak opening means the product has no potential: it may not have arrived, been photographed, or been offered in the right size.

The central decision is how many units of each style, color, and size to buy, from which qualified suppliers, and when to change price.
The decisions are linked: a larger initial order can reduce missed demand but increase end-of-season markdown and leftover exposure.
A shorter, more expensive supply route can preserve the option to respond to actual sales.
Returns delay the point at which a sale becomes a retained sale and can return stock too late to sell at full price.
The buyer needs an economic view of the whole life cycle, not merely the lowest quoted factory price.

The commercial buyer owns assortment and financial exposure.
The merchandise planner frames inventory, sales, margin, and open-to-buy limits.
The sourcing team negotiates capacity, cost, lead time, and supplier terms.
The supplier converts fabric and trims into finished garments and may subcontract parts of production.
The allocation team places sizes across stores and e-commerce fulfillment nodes.
The pricing team proposes markdowns; the returns operation decides whether each garment can be resold.
Quality, legal, and responsible-sourcing teams constrain which options are permitted.
The customer ultimately chooses a product, size, channel, and whether to retain it.
These roles may be combined in a small business, but their decisions should remain distinct.

A fashion forecast is an estimate of uncertain demand, not a purchase order.
A plan specifies proposed quantities and timing; a target is an aspiration, such as retained full-price sell-through.
A supplier commitment is an accepted order or capacity reservation subject to its contract.
The firm must not report an unapproved forecast as committed supplier capacity.
This chapter uses synthetic numbers unless a public source is expressly named.
Amounts are U.S. dollars (USD), product quantities are individual garments, and time is in weeks unless stated otherwise.
Legal material is educational, dated where relevant, and is not legal advice.

## 2. Learning outcomes

After this topic you can:

1. Map the buyer, sourcing, supplier, pricing, and returns decisions and identify their deadlines; see Exercise 1.
2. Calculate retained-unit contribution including refunds, markdowns, salvage, fulfillment, and return handling; see Exercise 2.
3. Compare an offshore initial commitment with a responsive replenishment option using explicit scenarios; see Exercise 3.
4. Identify stockout censoring, incomplete return cohorts, size substitution, and data leakage; see Exercise 4.
5. Specify a stock-flow and data contract at style-color-size-location-week grain; see Exercise 5.
6. Design a markdown test and distinguish causal lift from a before-and-after sales change; see Exercise 6.
7. Prepare a responsible sourcing and compliance gate without treating certification as proof of compliance; see Exercise 7.
8. Modify the runnable model and communicate its limits in a decision brief; see Exercise 8.

A good answer identifies an owner, decision date, uncertainty, feasible actions, financial objective, and operational constraints.
It also states which facts are observed, which are assumed, and which are inferences.

## 3. The core mental model

```text
 customer research / comparable styles       supplier qualification / material availability
                 |                                             |
                 v                                             v
       unconstrained demand scenarios --> pre-season buy --> make / ship / receive
                 |                             |                    |
                 |                             v                    v
                 |                       cash and capacity      allocate sizes
                 |                                                  |
                 v                                                  v
          week-by-week signals <--------- available inventory + selling price
                 |                                                  |
                 +--> response option: reorder, transfer, or mark down
                                                                    |
                                         sales --> returns --> inspect and grade
                                                                    |
                                full-price resale / markdown resale / salvage
                                                                    |
                                     retained contribution and learning loop
```

The buyer is purchasing a distribution of outcomes rather than a single average outcome.
A garment that sells at full price, is returned, and is later resold on markdown produces two sales transactions but only one retained unit.
The merchandise ledger must count its purchase cost once and its fulfillment and return-handling events as they occur.
A responsive supply option is valuable only when it arrives early enough, in the right sizes, at a cost justified by the demand it serves.
A markdown moves inventory through time and changes demand; it is not free revenue recovery.
A forecast built on sales alone can mistake unavailable inventory for low customer interest.

The information loop has delays.
Point-of-sale data can arrive daily, but returns may arrive several weeks after purchase.
A product may have high gross sales and poor retained sales when an online return cohort matures.
Waiting for complete returns information protects against overreacting but can make a replenishment decision too late.
Use a nowcast based on mature comparable cohorts and show uncertainty rather than treating unreceived returns as impossible.
The decision clock starts with the last date on which a supplier can accept a change, not with the next scheduled report.

A style is not one interchangeable pool.
A size M stockout cannot automatically be fixed with an excess of size XS.
Different colors can attract different shoppers, and online inventory may not be available to a store customer.
The economically relevant state is inventory by sellable variant, location, condition, and time.
The product hierarchy supports aggregation, but a size-level feasibility check must precede a commitment.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Style | Design or garment model; identifier | Treating all variants as interchangeable. |
| Colorway | Color or print variant within a style; identifier | Assuming demand shares remain constant. |
| Size curve | Distribution of demand or allocation across sizes; percentage | Using received quantities as an unconstrained demand curve. |
| SKU | Stock keeping unit, typically a style-color-size identifier | Assuming a merchant SKU is a globally unique product code. |
| Assortment | Chosen set of products for a channel, store, and season | Confusing listed products with physically available products. |
| Buy quantity | Units ordered for the assortment | Calling an unapproved forecast a purchase commitment. |
| Open-to-buy | Remaining budget or unit capacity available to commit for a period | Treating it as cash on hand without checking accounting rules. |
| Minimum order quantity | Supplier's minimum accepted units per specified grouping | Assuming a style minimum applies separately to each size. |
| Lead time | Elapsed time between defined start and end events, in days or weeks | Combining order acceptance, manufacture, and transit without dates. |
| Critical path | Sequence of dependent events controlling earliest receipt | Ignoring sample approval or material release. |
| In-season replenishment | Additional supply ordered after observing selling signals | Assuming it arrives before the selling window ends. |
| Available-to-promise | Inventory or supply available for new customer commitments under defined rules | Counting already reserved or damaged stock. |
| Allocation | Assignment of limited supply across channels or locations | Allocating by sales where availability differs. |
| Full-price sell-through | Full-price units sold divided by a defined receipt or available-unit base | Comparing rates with different denominators or return treatments. |
| Markdown | Reduction from a stated prior or reference price | Treating markdown percentage as incremental demand. |
| Gross sales | Sales before refunds under a specified accounting convention | Treating gross sales as retained revenue. |
| Return rate | Returned units or value divided by a matched sales cohort | Dividing this week's returns by this week's sales. |
| Return lag | Time from sale to return receipt or disposition | Assuming every unreturned order will be kept. |
| Net sales | Sales after applicable discounts and returns; exact accounting policy matters | Subtracting the same return twice. |
| Contribution | Revenue less explicitly scoped variable costs | Calling contribution accounting profit. |
| Landed cost | Product and included inbound costs per unit | Assuming all firms include duty, freight, and inspection identically. |
| Salvage | Recovery from goods not sold through normal channels, net or gross as specified | Assuming every recovered dollar is profit. |
| Stockout censoring | Observed sales are limited by inventory, hiding potential demand | Training a model on zeros as if availability were complete. |
| Lost sales | Demand not fulfilled or recaptured through substitution | Treating every out-of-stock page view as a lost sale. |
| Substitution | Demand shifting to another size, color, style, time, or channel | Adding full unmet demand to an alternative SKU. |
| Cohort | Transactions sharing a purchase period or comparable attribute | Mixing immature and mature sale periods. |
| Return disposition | Inspection outcome such as resell, repair, recycle, or destroy | Assuming every returned garment is pristine. |
| Cost of goods sold | Accounting expense recognized for goods sold under applicable policy | Equating a planning cash outlay with recognized expense. |
| RFID | Radio-frequency identification used to identify tagged objects | Treating a tag scan as proof that an item is sellable. |
| GTIN | GS1 Global Trade Item Number identifying a trade item | Using a location identifier as a product identifier. |
| Causal lift | Difference between observed outcome and credible counterfactual | Calling an unadjusted week-to-week change causal. |

## 5. The mechanics, step by step

### Step 1 — Define the actual decision

Write the category, season, customer, channel, and final sale date.
Identify the buyer who can approve the buy, markdown, or supplier option.
Record the last date on which each action can affect sellable inventory.
Define the objective as expected contribution subject to financial, service, capacity, and responsible-sourcing constraints.
Input: calendar, budget, assortment draft, channel plan, and supplier commitments.
Output: one decision statement and a list of alternatives the buyer can authorize.
Branch: if the supplier must confirm fabric before the assortment is finalized, model the fabric reservation as a separate earlier decision.
Branch: if the item is a replenishable basic rather than seasonal fashion, extend the horizon and include ongoing ordering and holding costs.
A target of 80% sell-through is not itself a profit function.
A target can be a guardrail, but the buyer still needs the economic value and downside of each feasible action.

### Step 2 — Map the product and network

Represent product as style, color, size, and season with effective dates for each attribute.
Map suppliers, fabric mills, cut-and-sew facilities, finishing sites, ports, distribution centers, stores, and return centers.
Identify ownership and custody at each handoff.
Document whether suppliers may subcontract and what approval is required.
Map customer channels and the inventory pools they can actually access.
Input: item master, approved facility list, purchase orders, network map, and stock ownership rules.
Output: a versioned product-network table with route and supplier identifiers.
Branch: if a style shares constrained fabric with another style, link their decisions through that common material capacity.
Branch: if online and stores share inventory, model reservations and transfer time rather than counting the pool twice.
A supplier group name is not necessarily the production facility where a garment is made.
Traceability needs the relevant facility and, where material, upstream supplier information.

### Step 3 — Build the critical path

Work backward from the latest useful receipt date.
Separate design approval, material purchase, fabric production, garment manufacture, quality release, export, transport, import, receiving, and allocation.
Use actual event timestamps and calendars, not one undifferentiated lead-time assumption.
Account for local holidays, port congestion, booking cutoffs, inspection delays, and supplier shutdowns.
Input: historical milestone dates and current supplier-confirmed production slots.
Output: a distribution of receipt dates by route and a last feasible decision date.
Branch: if a supplier offers a nominal six-week route but the receiving and photography process adds two weeks, use eight weeks to customer availability.
Branch: if expedited transit shortens only the transport leg, do not subtract it from fabric production or approval time.
Record the source and uncertainty of each lead-time estimate.
The critical path can change when fabric is already reserved but sewing capacity is not.

### Step 4 — Construct demand scenarios

Choose analog styles by customer, price, silhouette, channel, launch timing, and distribution.
Make any expert overrides explicit and date them before actual sales become known.
Estimate total category demand and mix by variant, location, and week.
Use launch, weather, promotional calendar, and product visibility as contextual variables where known at the decision time.
Correct or bound demand where previous items stocked out or were never offered.
Input: point-of-sale transactions, inventory availability, product attributes, site traffic, campaign dates, and analog decisions.
Output: low, central, and high demand paths, size distributions, and scenario probabilities if defensible.
Branch: if there is no reliable analog, widen the range and increase the value assigned to a reversible commitment.
Branch: if model inputs become available only after the historical forecast date, exclude them from backtesting.
Observed sales are an outcome under a price and availability regime, not a direct observation of all demand.
Forecasts must include returns separately rather than quietly assuming gross orders equal kept units.

### Step 5 — Translate demand into an initial buy

For each variant, compare the marginal value of another garment if demand is high with the marginal loss if demand is low.
Under a simplified one-period model, the critical fractile is underage cost divided by underage plus overage cost.
Underage cost is the contribution lost when a demand unit cannot be served, not necessarily the selling price.
Overage cost is the unrecovered acquisition and disposition cost of a leftover unit, net of salvage.
The formula assumes a demand distribution and independent unit economics; it is a starting point, not a complete fashion model.
Apply minimum order quantities, supplier packs, size breaks, budget, capacity, and channel rules before committing.
Input: demand distribution, cost and price assumptions, supplier pack rules, and planned salvage.
Output: feasible initial order and residual demand scenarios.
Branch: if the budget binds, compare marginal expected contribution per budget dollar across styles, not just forecast volume.
Branch: if the minimum order forces an unattractive tail of sizes, negotiate packs or decline the option rather than hide the constraint.
A purchase order becomes a commitment only after the relevant acceptance and contract terms are confirmed.

### Step 6 — Value speed as an option

Split the feasible buy into an early commitment and a later option only if capacity and material are genuinely reserved.
Identify the decision signal observed before the late order deadline.
Model fast-route unit cost, reservation fee, minimum order, freight, duty, receipt date, and cancellation terms.
Estimate the demand that remains after receipt, not total season demand.
Avoid counting a fast-order unit as incremental if the same demand would have been served from an existing unit.
Input: early sales signal, available stock, route calendar, confirmed capacity, and price assumptions.
Output: state-dependent order quantities and expected option value net of fees.
Branch: if the signal is not informative by the cutoff, late flexibility may not justify its premium.
Branch: if material is unavailable, a quoted fast sewing lead time is not a complete replenishment promise.
If no feasible route reaches the customer before markdown, compare a transfer or no reorder instead.

### Step 7 — Allocate and protect size availability

Allocate initial receipts to locations using demand estimates, known space, channel commitments, and minimum presentation quantities.
Maintain style-color-size balances at each node and record the status of each unit.
Separate physically on hand, reserved, in transit, damaged, and sellable available inventory.
Use a store-to-store transfer only when transfer cost and lead time are less than the expected benefit.
Input: size-level receipt plan, location demand, store capacity, and allocation rules.
Output: a feasible distribution plan and exceptions for missing sizes or capacity.
Branch: if the selling size curve differs from the planned curve, check availability before interpreting the difference as preference.
Branch: if a store receives no large sizes, its low large-size sales are not evidence of zero demand.
Protect inventory integrity through periodic counts and reconciliation between physical, financial, and digital records.

### Step 8 — Review weekly performance

Review availability, full-price and markdown sales, conversion, returns, stock, and future receipts at an agreed cutoff.
Compare actual outcomes with forecasts made before the week began.
Separate a forecast revision from a decision to alter committed orders or prices.
Show available sizes beside sell-through so high sales do not hide a broken size run.
Track return cohorts by purchase date and report maturity age in days.
Input: frozen historical forecasts, current stock, price history, gross sales, refunds, and supplier milestones.
Output: an exception list with owner, action deadline, and expected effect.
Branch: if gross sales are strong but the immature return rate is unknown, use a cohort-based range before ordering more.
Branch: if inventory is late, compare the expected selling window after receipt with the cost of expediting or cancellation.
Avoid a weekly dashboard that mixes prices or hides whether a unit is actually available for sale.

### Step 9 — Set and evaluate markdowns

Define the eligible inventory, date, price, expected demand response, and planned end-of-season disposition.
Compare the immediate contribution of a sale at the new price with the expected value of waiting.
Include cannibalization of full-price sales, stock position, fulfillment cost, and likely returns.
Ensure the proposed price and promotional presentation pass applicable legal review.
Input: remaining stock by variant, demand response evidence, competitor context where lawful, calendar, and cost assumptions.
Output: approved markdown schedule and measurement plan.
Branch: if size availability is fragmented, segment by variant where customer-facing price rules and operational systems permit.
Branch: if a controlled test is not feasible, report the result as observational and show plausible confounding.
A markdown sell-through increase is not necessarily incremental profit: some customers may have paid the original price.

### Step 10 — Incorporate returns and disposition

Connect each returned garment to the original order line, refund, receipt date, inspection decision, and final disposition.
Distinguish customer return initiation from physical receipt and final credit.
Estimate resale eligibility and remaining time to sell by condition and size.
Avoid assuming the same returned unit can be resold twice without additional inventory events.
Input: order and return events, refund amounts, inspection results, processing costs, and resale dates.
Output: net revenue, retained units, disposition yield, and expected future returns for immature cohorts.
Branch: if returns arrive after the season ends, value them at the appropriate recovery or carryover value, not current full price.
Branch: if an item is unsellable, use its actual recovery net of disposal or recycling expense.
A return rate by SKU can reflect fit, quality, imagery, customer mix, and return policy; it does not by itself identify the cause.

### Step 11 — Apply sourcing gates and decide

Check facility approval, labor and human-rights due diligence, restricted-substance requirements, customs records, and contractual obligations.
Obtain the relevant evidence before allocating new volume to a source.
Escalate unresolved issues to the accountable business and compliance owners.
Compare expected contribution, downside exposure, working capital, lead-time reliability, and documented supplier capacity.
Input: scenario economics and the responsible-sourcing and legal gate results.
Output: approved decision, rejected alternatives, confidence, owners, and next review date.
Branch: if the economically best option fails a mandatory gate, it is infeasible rather than merely less attractive.
Branch: if two options are close, report the break-even values rather than manufacturing false precision.
Log the forecast, plan, target, and accepted commitment as separate artifacts.

### Step 12 — Learn after the season

Freeze the original decision information, then compare it with realized receipts, sales, markdowns, and returns after cohorts mature.
Reconcile unit and financial ledgers before scoring performance.
Ask which variance arose from demand, timing, mix, price, returns, supplier performance, or data quality.
Use a consistent counterfactual; the unchosen sourcing path cannot be observed directly.
Input: versioned plans, transaction history, stock movements, final costs, and qualitative buyer notes.
Output: a decision review and specific updates to next season's assumptions.
Branch: if the product changed after the buy decision, do not attribute all forecast error to the original model.
Branch: if a supplier failed a commitment, investigate root cause before changing the demand forecast.
Learning is strongest when teams record what was knowable at the time rather than judging with hindsight.

### Decision detail A — Build an assortment architecture

Start by separating permanent basics, repeatable seasonal shapes, and highly novel fashion.
Each group has a different evidence base and a different exposure to style obsolescence.
A basic T-shirt can have longer history but still change fabric, fit, price, or distribution.
A novel evening dress may have no direct prior sales and a short selling period.
For each group identify the role in the assortment: volume, margin, customer acquisition, or range completeness.
Then define the objective and constraints for that role rather than applying one target to every item.
A breadth decision asks how many distinct styles to offer.
A depth decision asks how many units to buy per selected style.
More breadth can diversify taste risk but can also fragment minimum orders and dilute available size depth.
The buyer should show this trade-off explicitly.
An assortment that offers every color in only one size may have high nominal breadth and poor customer usefulness.
Output: an approved assortment list with a decision rationale and a constrained unit and cost envelope.

### Decision detail B — Establish a buyer's calendar

Create an event calendar before estimating the sales forecast.
Mark line review, design freeze, sample approval, material reservation, production slot, shipping departure, receipt, launch, markdown, and exit.
For each event record who can change it, the notice required, and the penalty or cost of changing it.
For example, a fabric commitment can precede the final color split.
If the contract permits an undyed fabric reservation, the later color choice is a real option only within the contract's capacity and date limits.
If the contract fixes color at reservation, the apparent flexibility does not exist.
Track the last actionable date for each decision on the same timeline as information availability.
A forecast improvement arriving after the order deadline cannot change that order.
It might still improve a later allocation or markdown decision.
Output: a decision calendar with cutoffs and documented degrees of freedom.

### Decision detail C — Quantify fabric and trim exposure

A garment's production may require fabric, buttons, zippers, labels, thread, and packaging.
A single specialized zipper can hold up the whole style even when fabric and sewing slots are available.
Map each component to supplier, approval status, lead time, minimum, and substitutability.
For a synthetic jacket, suppose each finished unit requires 1.5 meters of approved fabric.
Allowing 5% process loss, 1,000 finished units require 1,500 / 0.95 = approximately 1,579 meters of input fabric.
This assumes the stated loss is measured as a fraction of input fabric and applies uniformly.
If 1,500 meters is the gross input rather than the net requirement, this formula would overstate need.
For multi-style shared fabric, reserve quantities against each accepted order rather than summing nominal supplier capacity.
An alternate fabric may change fit, cost, customer expectations, and compliance evidence.
Approve substitutes through the product and compliance process before treating them as executable.
Output: a component-constrained finished-unit supply plan.

### Decision detail D — Calculate usable selling time

For each potential receipt, count the actual days during which the product can be offered at the planned price.
Subtract receiving, quality inspection, content creation, site activation, and distribution time if not already in the route.
Distinguish physical receipt at a port from customer-facing availability.
Suppose the season ends at the end of week 12 and inventory becomes available at the start of week 9.
There are four calendar selling weeks remaining: weeks 9, 10, 11, and 12.
If markdown starts at the beginning of week 11, only two of those weeks remain at the original price.
Compare demand expected in those two weeks with existing inventory and any already committed receipts.
Do not allocate all historical week-1 launch demand to an item that only becomes available in week 9.
If a late arrival triggers a contractual charge or cancellation right, model that term only if it is actually in the agreement.
Output: price-specific selling windows and feasible receipt dates.

### Decision detail E — Develop an analog portfolio

Select more than one comparable where possible.
Use a stable set of attributes known before the decision: silhouette, customer segment, price band, fabric, and season.
Record material differences such as distribution footprint and campaign support.
Normalize for time on sale and availability before comparing weekly rates.
A prior item with 20 stores cannot be scaled directly to 200 stores if the customer mix differs.
An analog that sold out on day two supplies a lower bound on demand, not a complete demand observation.
An analog sold only after a steep markdown is not a clean full-price demand signal.
Use a range across credible analogs rather than selecting the one that supports a preferred buy.
Expert judgment can adjust the range if the reason and date are recorded.
Backtest selection rules on prior launches with the information actually available then.
Output: analog selection, adjustment log, and an uncertainty range.

### Decision detail F — Make a size-level buy feasible

Translate total style demand into size-color variants before submitting an order.
For a synthetic 1,000-unit buy, a 10% / 25% / 35% / 20% / 10% curve yields 100, 250, 350, 200, and 100 units.
The shares sum to 100%, but that alone does not prove that the curve fits each channel.
If a factory carton contains ten units of one size, all quantities in this illustration satisfy the pack rule.
If a factory requires mixed cartons in a fixed ratio, the feasible quantity set changes.
Map size labels across regional standards only through an approved product-specific mapping.
A label change does not necessarily mean the physical dimensions or fit changed.
For a limited budget, compare the expected value of a full size run against a partial presentation.
Do not assume a customer who cannot find size L will purchase size M.
Input from returns and fit specialists can inform the curve, but customer-reported reasons need validation.
Output: variant-level order quantities and a list of rounding differences from the unconstrained plan.

### Decision detail G — Represent uncertainty honestly

The central forecast is one possible outcome, not a guaranteed sales quantity.
Separate uncertainty about total demand from uncertainty about the distribution across sizes and channels.
A high total-demand scenario may still have excess XS and a shortage of L.
Likewise, an aggregate forecast can be correct while every location's allocation is wrong.
Represent correlated demand where appropriate: a successful launch can raise several sizes together.
Represent substitution cautiously; moving all unmet demand to another variant overstates recapture.
Capture uncertain delivery dates separately from uncertain customer demand.
If delays and high demand are correlated because peak periods constrain transport, independent simulation may understate downside.
Explain whether probabilities are estimated from history, expert judgment, or purely illustrative.
When data is weak, compare plausible bounds instead of presenting a spurious decimal probability.
Output: documented scenarios that reflect both demand and supply uncertainty.

### Decision detail H — Compare cost on like terms

Start with factory price and specify the currency and exchange-rate date.
Add only relevant freight, duty, insurance, handling, inspection, and other incremental costs.
Identify who pays each charge under the negotiated delivery terms.
Do not add a charge already included in the quoted price a second time.
If a fast supplier's unit cost is $34 versus $24 for an early order, the premium is $10 per purchased unit.
For 200 purchased units, that premium is $2,000 before any differences in receipts, returns, or recovery.
The faster route must create at least $2,000 of additional expected value to offset that premium in a like-for-like comparison.
It may create value by avoiding overbuying as well as by serving otherwise lost demand.
Consider payment timing separately if the analysis includes financing or a working-capital charge.
Document tariff classification and legal determinations with the appropriate specialists; a model estimate is not a customs ruling.
Output: a cost bridge that can be reconciled to contractual terms.

### Decision detail I — Account for reserved capacity

A supplier may offer a reservation in exchange for a fee, minimum exercise volume, or cancellation payment.
Write down whether the fee applies once, per unit, or only when an order is exercised.
Suppose capacity for up to 200 units costs $600 and may be used after a demand signal.
That fee is $600 whether the buyer exercises 50 or 200 units if the contract says it is a fixed reservation.
It is not legitimate to deduct $600 from each scenario and then deduct it again from the expected value.
Before reservation, compare expected value net of the fee against alternatives.
After the nonrefundable fee has been paid, the exercise decision generally compares incremental future revenues and costs.
The past fee still matters for evaluating the original reservation decision.
If the fee is refundable on nonexercise, the decision tree must show that refund.
Output: a dated decision tree with each payment or refund occurring exactly once.

### Decision detail J — Plan inventory across nodes

At the initial allocation, distinguish central reserve from inventory placed in a store.
An unallocated reserve can support a later response but may delay the first customer purchase.
Suppose a store has 20 units and expects 30 units of demand before new stock can arrive.
The potential 10-unit shortfall is a planning signal, not proven lost sales.
Check whether another location has transferable units in the same color and size.
Subtract the transfer's transport, handling, delay, and any loss of expected sales at the origin.
If origin demand would consume those units, the transfer may move the shortage rather than eliminate it.
Apply shipping cutoffs and receiving capacity before counting the transfer as available.
If inventory is reserved for an online order, it is not free stock for the store.
Output: a time-phased, size-level allocation and transfer proposal.

### Decision detail K — Evaluate early selling signals

Use the first available sales weeks to revise demand only after checking launch execution.
A missing product image, delayed listing, broken size run, or incorrect price can depress observed sales without proving low preference.
Compare units per available day rather than raw weekly totals when launch dates differ.
Include site visibility or store traffic where measured consistently and legally.
Check whether a high conversion rate comes from a small, selectively exposed audience.
Separate organic demand from a promotion or influencer campaign that may not repeat.
Revise uncertainty as well as the point forecast when the new data is noisy.
Freeze a copy of the pre-update forecast to prevent hindsight leakage.
If the production decision closes before enough evidence accrues, report the remaining uncertainty.
Output: a dated forecast update and decision note explaining the signal and caveats.

### Decision detail L — Decide whether to expedite

Expediting can shorten a transport leg, switch a production source, or prioritize inspection.
Each intervention has its own cost, capacity limit, and attainable time saving.
Calculate the extra full-price or avoided-shortage contribution created by the earlier availability.
Suppose expediting costs $1,200 and advances 300 jackets by one week.
If that earlier week creates 30 additional retained sales at $50 contribution each, incremental benefit is $1,500.
The simplified net gain is $300, before any additional emissions cost, returns, or displaced sales.
If those 30 customers would otherwise purchase the following week, incremental benefit is lower.
If inventory arrives before a customer-facing launch can be activated, extra speed may have no value.
Only compare a realistic expedited date against the realistic baseline date.
Output: a dated expedite approval with a break-even incremental sales count.

### Decision detail M — Model a markdown alternative

Define the inventory eligible for markdown and the reference full price.
Suppose 100 units remain and can be sold immediately at $40 or held for a chance of full-price sales.
At $40 less $5 fulfillment, an immediate retained sale contributes $35 before sunk acquisition cost.
If holding yields a 50% chance of an $80 sale and otherwise an $8 recovery, expected inflow after fulfillment is 0.5 × $75 + 0.5 × $8 = $41.50.
On those assumptions holding is worth $6.50 more per unit than the certain markdown sale.
This comparison assumes both outcomes occur within the decision horizon and ignores returns and capital costs.
It also assumes markdown demand is certain and holding does not affect other items.
Alter the decision if the markdown changes sell-through, return rates, or later recovery.
Do not subtract the already-incurred garment cost from one alternative but not the other.
Output: a comparable incremental-value calculation for each feasible pricing action.

### Decision detail N — Control experimentation

Identify the unit randomized: store, region, product, or eligible customer exposure.
Choose units before seeing the outcomes used for analysis.
Confirm the intervention can actually be delivered and recorded as assigned.
Use contribution per eligible unit or customer exposure when that matches the business decision.
Revenue alone can favor deep discounts even when they destroy contribution.
Track stockouts because an untreated store with no stock cannot express demand at the control price.
Keep the planned analysis window and return-maturity treatment in the test record.
Cross-store shopping, online price visibility, and transfers can contaminate control groups.
Report confidence intervals and practical relevance, not only statistical significance.
If a test is underpowered, do not present a null result as proof of no effect.
Output: a decision-ready evaluation with design, implementation checks, and limitations.

### Decision detail O — Manage reverse logistics capacity

Map the return path from initiation through carrier transit, receipt, inspection, refund, and resale or recovery.
Financial refund timing and physical item receipt can be different.
Measure units awaiting inspection by age and expected value, not just total units.
An item received in week 5 but inspected in week 11 may miss the original-price opportunity.
Prioritize processing based on remaining sellable value and contractual or consumer-law obligations.
Do not delay legally required refunds merely to optimize merchandise recovery.
Estimate labor capacity using observed minutes per inspection and shift availability.
For example, 240 garments at four minutes each require 960 minutes, or 16 labor-hours, before breaks and variability.
Record whether handling costs already include overhead to avoid counting them twice.
Output: a return-processing queue with aging, capacity, and estimated value at risk.

### Decision detail P — Separate risk from performance

A supplier can meet its accepted dates and still produce slow sales because the market disliked the design.
A buyer can make a reasonable decision under uncertainty and still see a poor outcome.
Conversely, a late supplier or poor forecast can appear successful if demand is unexpectedly strong.
Compare supplier delivery against the accepted schedule and agreed measurement convention.
Compare demand forecasts against unconstrained demand estimates where credible, not merely constrained sales.
Compare the decision's realized value with a feasible counterfactual rather than an impossible perfect-information plan.
Record relevant exceptions such as an approval held by the brand or a late design change.
Share corrective actions with owners and due dates while protecting sensitive supplier information.
The review should improve future decisions, not rewrite the information available at the original cutoff.
Output: an attribution ledger with evidence and explicit residual uncertainty.

## 6. Worked example

### Decision and evidence

A synthetic direct-to-consumer brand must commit to a seasonal jacket before a 12-week selling season.
The buyer can place a confirmed initial order of 1,000 garments with a qualified supplier.
The planned landed cost is $24 per garment, paid for all 1,000 units.
The planned initial retail price is $80 and the planned markdown price is $40 per garment.
The planned end-of-season net recovery is $8 per unsold garment.
A transaction costs $5 to fulfill, including a resale transaction.
Each returned garment costs $6 to inspect and process, excluding the original and resale fulfillment costs.
Taxes, fixed overhead, working-capital cost, inbound variability, and customer acquisition are excluded.
All values are synthetic assumptions, not a statement about apparel industry averages.

The 1,000-unit buy is a plan until the supplier accepts the purchase order.
The central demand forecast is a scenario, not an 80% full-price sell-through commitment.
The target is to maximize scoped contribution while meeting the approved budget and supplier gates.
For this arithmetic example, all garment counts and all costs are known ex post.
The lab below extends the analysis to uncertain customer demand and size mix.

### Event ledger

1. Receive 1,000 units; acquisition cost is 1,000 × $24 = $24,000.
2. Sell 800 units initially at $80; gross full-price sales are $64,000.
3. Refund 80 of those units in full: 80 × $80 = $6,400.
4. Inspect all 80 returned units; 50 are resellable in season and 30 are not.
5. Sell the 50 resellable returns plus 150 initially unsold units at $40: 200 × $40 = $8,000.
6. Recover $8 each for the remaining 50 initially unsold units plus 30 returned units: 80 × $8 = $640.
7. Count 800 full-price transactions and 200 markdown transactions, or 1,000 fulfillment events.
8. Count 80 return-processing events.

The physical stock reconciliation is 1,000 = 720 retained full-price units + 200 markdown units + 80 recovered units.
The calculation counts a garment only in its final disposition; the 80 refunded units are not retained full-price sales.
Gross full-price transactions and markdown transactions add to 1,000, but that does not mean 1,000 distinct garments were retained.
The 50 returned-and-resold garments appear in both transaction counts, correctly producing two fulfillment events each.

Net sales revenue is $64,000 − $6,400 + $8,000 = $65,600.
Adding recovery gives $65,600 + $640 = $66,240 of total inflow in this simplified contribution view.
Fulfillment cost is 1,000 × $5 = $5,000.
Return handling is 80 × $6 = $480.
Contribution is $66,240 − $24,000 − $5,000 − $480 = **$36,760**.
This is not accounting net profit and should not be presented as a financial statement.

Gross initial full-price sell-through on receipts is 800 / 1,000 = 80%.
Retained full-price sell-through after these returns is 720 / 1,000 = 72%.
Final retained unit sell-through including markdown is (720 + 200) / 1,000 = 92%.
Return rate for the initial full-price cohort is 80 / 800 = 10%; that calculation assumes the cohort has fully matured.
The 8% recovery fraction is not the return rate: it is 80 recovered units divided by 1,000 receipts.
A report should label all denominators and whether rates are gross or net of returns.

### Sensitivity and choice

If the same 200 markdown transactions could instead be completed at $45 without changing demand or returns, revenue rises by 200 × $5 = $1,000.
The contribution would be $37,760 under that deliberately restrictive assumption.
It is not a forecast that customers will buy the same quantity at $45.
If 20 more of the 80 returns become unsellable, while markdown demand remains sufficient and all else stays constant, recovery replaces markdown resale for those 20 units.
The contribution change is 20 × ($8 − $40 + $5) = −$540: the business loses the markdown revenue, receives recovery, and avoids the resale fulfillment cost.
The revised contribution is $36,220.
This sensitivity isolates disposition yield, not any causal effect of changing inspection standards.

For a simple time-feasibility check, suppose supplier production takes six weeks, transport takes three weeks, and receiving and allocation take one week.
A reorder placed at the beginning of selling week 3 arrives at the beginning of week 13, after a 12-week season.
It cannot serve in-season demand under these assumptions.
If a qualified fast route takes two weeks in total, the same decision can arrive at the beginning of week 5.
The fast route still requires a comparison of its additional cost with demand remaining from week 5 onward.
A shorter nominal lead time is not evidence that a supplier can produce the needed size curve.

### A size-specific allocation check

Suppose that at a review date one location has 30 size-S jackets and 5 size-L jackets.
The remaining demand forecast is 10 size-S units and 20 size-L units.
The aggregate forecast is 30 units and aggregate stock is 35 units.
That aggregate comparison conceals an expected shortage of 15 size-L units and an expected surplus of 20 size-S units.
Those are forecast imbalances, not observed lost sales or guaranteed leftovers.
If another location has genuine excess size-L stock, a feasible transfer could reduce the local imbalance.
Its value depends on remaining demand after the transfer arrives and on any demand displaced at the origin.
A size-S surplus cannot be assumed to satisfy size-L demand.
If customers substitute sizes, the analyst needs evidence of that behavior and must avoid double counting recaptured demand.
This example does not change the 1,000-unit case ledger; it illustrates a separate synthetic allocation state.

### Decision brief

The buyer should approve the 1,000-unit initial buy only if the budget, capacity, and sourcing gates are met and the downside scenarios are acceptable.
The numbers above are an ex-post scenario, not proof that 1,000 units is the optimal order.
An in-season order from the ten-week route is infeasible for this season at the stated cutoff.
A two-week route merits evaluation if capacity, price, and material availability are contractually credible.
The pricing owner should review size-level inventory and mature return estimates before making a markdown decision.
The returns owner should track inspection yield and the date each unit becomes sellable again.
The analyst should preserve the original forecast and assumptions to support a fair post-season review.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the event-ledger arithmetic above and compares two synthetic fast-reorder choices under three demand scenarios.
The model's demand scenarios represent incremental demand remaining after the fast route arrives, not total season demand.
The early inventory available to serve those scenarios is assumed to be zero; this isolates the option and is not a complete optimization.
The scenario probabilities are assumed, not estimated, and the route is assumed to arrive on time in every scenario.
The calculation excludes returns from the fast-route comparison; use the earlier ledger for the return example.
An unfilled request has no separate penalty; the lost contribution is the opportunity cost of not making that sale.
The fast route's unused units receive the stated end-of-season recovery.
No capacity reservation fee is assumed in the initial comparison.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Ledger:
    bought: int = 1000
    cost: float = 24.0
    full_transactions: int = 800
    full_price: float = 80.0
    returned: int = 80
    returned_resold: int = 50
    markdown_from_initial: int = 150
    markdown_price: float = 40.0
    recovery: float = 8.0
    fulfillment: float = 5.0
    return_handling: float = 6.0

    def evaluate(self):
        assert 0 <= self.returned_resold <= self.returned <= self.full_transactions
        assert self.full_transactions + self.markdown_from_initial <= self.bought
        markdown = self.markdown_from_initial + self.returned_resold
        retained_full = self.full_transactions - self.returned
        recovered = self.bought - retained_full - markdown
        assert retained_full + markdown + recovered == self.bought
        net_sales = (self.full_transactions - self.returned) * self.full_price
        net_sales += markdown * self.markdown_price
        transactions = self.full_transactions + markdown
        total = net_sales + recovered * self.recovery
        total -= self.bought * self.cost
        total -= transactions * self.fulfillment
        total -= self.returned * self.return_handling
        return retained_full, markdown, recovered, net_sales, total

@dataclass(frozen=True)
class Scenario:
    name: str
    probability: float
    remaining_demand: int

scenarios = (
    Scenario('low', 0.25, 50),
    Scenario('central', 0.50, 150),
    Scenario('high', 0.25, 250),
)

def fast_contribution(quantity, demand, price=80.0, cost=34.0,
                      fulfillment=5.0, recovery=8.0):
    sold = min(quantity, demand)
    leftover = quantity - sold
    return sold * (price - fulfillment) + leftover * recovery - quantity * cost

def expected_fast(quantity):
    return sum(s.probability * fast_contribution(quantity, s.remaining_demand)
               for s in scenarios)

assert abs(sum(s.probability for s in scenarios) - 1.0) < 1e-12
full, markdown, recovery_units, net_sales, contribution = Ledger().evaluate()
print(f'Final units: full={full}, markdown={markdown}, recovery={recovery_units}')
print(f'Net sales: ${net_sales:,.0f}')
print(f'Ledger contribution: ${contribution:,.0f}')
for q in (0, 100, 200):
    print(f'Fast reorder {q}: expected contribution ${expected_fast(q):,.0f}')
```

Expected output:

```text
Final units: full=720, markdown=200, recovery=80
Net sales: $65,600
Ledger contribution: $36,760
Fast reorder 0: expected contribution $0
Fast reorder 100: expected contribution $3,262
Fast reorder 200: expected contribution $4,012
```

The printed amounts are rounded to the nearest whole dollar.
For 100 fast units, sold quantities are 50, 100, and 100 in the three scenarios.
Their contributions are $750, $4,100, and $4,100, respectively.
Their probability-weighted value is $3,262.50.
For 200 fast units, contributions are −$1,850, $4,850, and $8,200, respectively.
Their probability-weighted value is $4,012.50.
Python's displayed formatting rounds these half-dollar values to the nearest even whole dollar.
The comparison is incremental to no fast order only under the stated zero-existing-stock assumption.
A reservation fee would be deducted from each option that incurs it; if paid irrespective of exercise, distinguish sunk and future decision costs.
To use actual data, replace scenarios with properly dated demand paths, include existing stock and inbound commitments, and model returns and size feasibility.
A model that merely maximizes these three scenarios does not establish that 200 units is the global optimum.

## 8. Data, standards, and regulatory context

### Data contract and lineage

A useful analytical grain is style-color-size-location-week, with transaction events retained at order-line level.
The aggregate must preserve both the original sale date and the return receipt date.
Keep immutable transaction identifiers and event times alongside ingestion times.
Time-zone and week-boundary conventions must be agreed across stores, suppliers, and e-commerce systems.
The following fields are a minimum starting contract, not a claim that every field exists in every business.

| Dataset | Key and time | Required content | Control |
|---|---|---|---|
| Product master | SKU and effective date | Style, color, size, season, GTIN where assigned | Reject overlapping mappings. |
| Supplier facility | Facility identifier and validity dates | Approved site, function, country, parent supplier | Do not substitute supplier headquarters for factory. |
| Purchase order | PO line and accepted version | Units, unit cost, route, milestones, terms | Compare amendments with acknowledgments. |
| Inventory | SKU, location, timestamp, status | On hand, reserved, damaged, in transit | Reconcile movements and physical counts. |
| Sales | Order line and event timestamp | Gross price, discount, channel, units, fulfillment | Ensure refunds are linked rather than duplicated. |
| Returns | Return line and timestamp | Original order line, refund, condition, disposition | Track unmatched and pending returns. |
| Price | SKU, channel, effective interval | Ticket, actual price, promotion, approval | Prevent overlapping price intervals. |
| Availability | SKU, location, interval | Sellable and visible status | Measure stockout duration, not only end-of-day balance. |
| Forecast | SKU or higher grain, issue timestamp | Horizon, units, assumptions, model version | Freeze the information known at issue time. |
| Sourcing evidence | Facility and evidence date | Approval, expiration, scope, remediation status | Escalate expired or out-of-scope evidence. |

Stock-flow reconciliation begins with opening stock plus receipts plus approved returns and transfers in.
Subtract gross shipments, transfers out, recovery dispositions, and other write-offs to obtain closing physical stock.
Use the actual posting convention; a refund without physical receipt is a financial event, not necessarily a stock movement.
If a return is recorded in a separate quarantine bucket, do not also add it to sellable stock.
A return can be counted in both gross transactions and later resale transactions without duplicating the physical unit.

Use cohort aging to estimate outstanding returns.
For example, calculate return probability by days since sale using mature comparable orders and show uncertainty for recent cohorts.
Keep reason codes as customer-reported observations rather than definitive root causes.
When a size has been unavailable, treat recorded sales as censored and examine other evidence before concluding demand is low.
If customer-level data is used, minimize collection, restrict access, and follow applicable privacy and retention requirements.

### Event-level illustration: one returned jacket

The following events are synthetic and describe one uniquely traceable jacket.
They are a teaching record, not a mandated accounting treatment.
The initial purchase order records a 100-unit commitment for a particular style-color-size.
At receipt, 100 physical units enter the central sellable inventory bucket.
One unit is reserved for an online order; the available quantity declines even before physical shipment.
The warehouse ships the item, converting one reserved unit into a customer shipment.
The sale is recorded at $80 and one fulfillment event is recorded.
The customer requests a return five days later.
The request creates an expected return, but the item is not yet physical inventory.
The customer receives a full $80 refund according to the merchant's policy.
The parcel arrives at the return center two days later.
The unit enters quarantine pending inspection, not immediately the sellable bucket.
An inspector finds the garment unworn and complete.
The garment moves from quarantine to sellable inventory at the return center.
The planning system assigns it to a store with sufficient time left to sell it.
The transfer creates a shipment from the return center and a receipt at the store.
The store sells it at $40 after the approved markdown.
The retailer has recorded $120 of gross sales transactions and an $80 refund.
Net sales for these two linked transactions are $40.
There is one final retained garment, two customer fulfillment events, and one return-processing event.
If the store sale is incorrectly joined to both original sale and return by SKU alone, the same records may multiply.
Join by stable event and line identifiers; use physical unit identity when available and appropriate.
An order-line identifier can establish the refund relationship even where individual units are not serialized.
If two identical units were purchased on one line, the return record must include quantities and allocation rules.
When a returned garment cannot be traced uniquely, report the uncertainty rather than inventing a serial match.

### Inventory reconciliation illustration

Suppose a store begins a week with 120 sellable units of a variant.
It receives 40 units from the distribution center and 8 approved customer returns.
It sells 70 units, transfers 12 to another store, and writes off 3 damaged units.
The expected closing balance is 120 + 40 + 8 − 70 − 12 − 3 = 83 sellable units.
A physical count finds 80 units; the unexplained variance is −3 units.
The difference is an exception for investigation, not automatically shrinkage or customer demand.
Potential causes include timing differences, scan omissions, incorrect transfer status, and misclassified damage.
Check whether the eight returns were physically inspected before placing them in sellable stock.
Check whether a transfer receipt posted at the destination before its departure posted at the origin.
If a financial refund was included as a stock receipt, remove that unsupported assumption.
Only after reconciliation should the store revise reorder or transfer decisions based on the balance.
Preserve both the original count and subsequent adjustment with owner and timestamp.
Repeated variance by location or event type can guide process improvement.
A count at one moment does not by itself establish how long a variant was unavailable during the week.

### Example of return-cohort maturity

Suppose 1,000 synthetic orders from a January purchase cohort have been observed for 60 days.
At day 14, 60 units had been returned; by day 60, 100 units had been returned.
In this illustrative cohort, 60 / 100 = 60% of the eventual 60-day returns had appeared by day 14.
Now consider 500 comparable February orders observed for only 14 days with 24 returns received.
If the January lag pattern is genuinely applicable, a simple 60-day estimate is 24 / 0.60 = 40 returned units.
The implied 60-day return rate is 40 / 500 = 8%, versus the observed day-14 rate of 4.8%.
This is an estimate, not a known outcome or a causal finding.
It assumes consistent return policies, shipping speed, product mix, channels, and observation definitions.
Small cohorts or new fits can make the lag curve unreliable.
Give the estimate a range and compare the same age across cohorts before taking action.
If an order contains multiple garments, define whether the rate uses orders, lines, or units.
If a refund occurs before physical receipt, specify which event defines return maturity.
Some jurisdictions and policies permit longer return windows; the observation period should match the decision.
Later complete outcomes should be used to assess whether the maturity adjustment was calibrated.

### Data-quality checks before deployment

Check that every transaction line references a product mapping valid at the transaction timestamp.
Check that an active product has a size, color, season, and price where the business requires them.
Check that no accepted purchase-order line has a negative quantity or ambiguous revision.
Check that expected receipts do not exceed supplier-confirmed quantities without an explicit exception.
Check that inbound milestone timestamps do not precede their dependent approvals without explanation.
Check that each price interval has a start and end, with a clear policy for overlapping promotions.
Check that discounts and refunds are expressed in the transaction currency and converted using the stated rate.
Check that a return quantity does not exceed the original eligible quantity after previous returns.
Check that return receipt and refund events are not treated as identical merely because their timestamps match.
Check that a return disposition cannot be both destroyed and resold without a correcting event.
Check that reserved stock is not simultaneously offered to multiple channels.
Check that reported stockout hours fit within the period in which the item was offered.
Check that inventory counts, sales, returns, and transfers satisfy a stock-flow reconciliation.
Check that missing values are distinguished from measured zeros.
Check that negative inventory balances are investigated rather than clipped silently.
Check that location changes and store closures are reflected in valid-time master data.
Check that canceled sales do not enter a retained-sales measure.
Check that sales made before the launch date are preorders or corrections rather than date errors.
Check that size conversion rules are versioned when product or market definitions change.
Check that a forecast's issue timestamp precedes all outcomes used to evaluate it.
Each check needs an owner, exception threshold, resolution path, and audit trail.
Do not automatically exclude failed rows without measuring the resulting bias in the analytical population.

### Worked data dictionary for decision-critical fields

The **product key** identifies a specific commercial variant under a documented internal convention.
It must not be reused for a materially different garment without an effective-dated version.
The **season key** indicates a planning season, not necessarily the transaction date.
Carryover garments can be sold in a later season while retaining their original design association.
The **facility key** identifies a physical production or processing site.
The brand should distinguish a legal supplier, a facility, and a subcontractor where the relationship matters.
The **order-line key** ties ordered units, amendments, acknowledgments, receipts, and invoices together.
An amendment should retain a history rather than overwrite the previously accepted delivery date.
The **event timestamp** records when the real-world event occurred.
The **ingestion timestamp** records when the analytical system learned about that event.
A historical forecast backtest must use the data actually available by the decision cutoff.
The **inventory status** records whether units are sellable, reserved, damaged, quarantined, or otherwise restricted.
A status transition should have an initiating event and an accountable system.
The **gross transaction price** is the amount before applicable refund events under the stated policy.
The **refund amount** is tied to an original sale line and may include components beyond merchandise.
Keep shipping, tax, and merchandise refunds separate if contribution calculations treat them differently.
The **return reason** is a recorded reason, often selected by a customer or agent.
It is evidence for investigation, not necessarily a verified defect.
The **disposition** indicates what actually happened after inspection.
A proposed disposition and a completed resale are distinct events.
The **forecast issue date** anchors what information was knowable.
The **forecast horizon** specifies the week or weeks to which the estimate applies.
The **constraint version** records pack sizes, capacity, or budget assumptions used in the decision.
The **approval status** records whether a site, design, or order is permitted under the applicable process.
Expiry or change of scope can make a previously valid approval unsuitable for a new order.
The **recovery value** must distinguish gross proceeds from proceeds net of disposal and handling costs.
These distinctions are needed to make a financial calculation reproducible.

### Privacy, access, and retention considerations

An analyst can often calculate product-level return rates without retaining names or addresses.
Use a stable pseudonymous key where linking transactions is necessary and permitted.
Pseudonymization reduces direct exposure but does not automatically make information anonymous.
Separate access to customer contact details from access to aggregate planning reports.
Limit who can see return comments, which may contain sensitive information beyond the reason code.
Define retention and deletion rules with the organization's privacy and legal owners.
When sharing data with a supplier, provide only the fields required for the agreed purpose.
Protect commercially sensitive forecasts, negotiated prices, and facility evidence as well as personal data.
Document data lineage and permitted use before using external consumer data in a model.
Privacy obligations vary by jurisdiction; this section is not a substitute for a legal assessment.

### Identifiers and exchange

GS1's GTIN identifies a trade item; its Global Location Number (GLN) identifies a party or location where assigned.
Identifier assignment and barcoding follow the relevant GS1 rules and trading-partner agreement.
A GTIN does not itself guarantee item quality, physical availability, facility approval, or legal compliance.
A merchant's internal SKU can map to a GTIN, but mappings need effective dates and variant-level controls.
Electronic data interchange (EDI) documents can communicate orders, acknowledgments, advance shipping notices, and invoices under agreed message standards.
An acknowledgment is useful evidence of a supplier response, but the contract determines its legal effect.
RFID can improve item-level visibility where deployed; tags and reads still require master-data and exception controls.
A missed read, incorrect association, or unprocessed return can leave a digital record inconsistent with physical stock.

### Legal and due-diligence boundaries

Rules vary by product, origin, market, and transaction date; the following are starting points for legal review, not exhaustive requirements.
U.S. Federal Trade Commission guidance on textile and wool labeling describes fiber content, country-of-origin, and responsible-business identification requirements under relevant statutes and rules, subject to exceptions.
The same guidance should be checked for products and channels within scope; labeling data must reflect the actual garment, not merely its design brief.
U.S. Customs and Border Protection describes the Uyghur Forced Labor Prevention Act's rebuttable presumption for covered goods and enforcement process.
Origin and supply-chain documentation therefore matter, but the needed evidence depends on the goods and applicable law.
A supplier assertion or audit certificate alone should not be treated as an automatic clearance determination.

The EU Textile Labelling Regulation (EU) No 1007/2011 establishes rules on textile fiber names and related labeling and marking of textile products.
The EU Ecodesign for Sustainable Products Regulation (EU) 2024/1781 establishes a framework for product-specific measures and digital product passports.
The framework does not mean that every apparel product already has identical passport data duties on the date of this chapter.
Check subsequent delegated acts, product scope, dates, exemptions, and competent authority guidance before specifying a binding implementation.
EU Directive (EU) 2024/825 addresses consumer protection against unfair practices and sustainability claims; applicability and national implementation must be checked.
Do not infer that a generic sustainability label establishes compliance with every claim requirement.

The OECD Due Diligence Guidance for Responsible Supply Chains in the Garment and Footwear Sector is guidance, not itself a universal statute.
It supports risk-based due diligence, prevention, mitigation, tracking, communication, and remediation processes.
An audit is one evidence source, not proof that all labor risks have been eliminated.
Consult current local labor law, applicable trade restrictions, customs requirements, and qualified counsel for the specific sourcing decision.

## 9. Industry practice and case evidence

### Public case: Inditex and visibility

Inditex describes its business model and integrated store and online operations in its annual reporting.
Its public materials also discuss RFID deployment in stores and logistics.
These are company-reported descriptions; they are not an independent estimate of causal profit uplift from any specific technology.
The relevant operational distinction is that an item-level view can support stock accuracy and cross-channel fulfillment if physical and system records agree.
[Inference] Greater visibility may improve allocation decisions, but benefits depend on process adoption, read accuracy, and how inventory is made available.
Do not transfer an observed company's outcome to a different retailer without comparable economics and an evaluation design.

### Public case: H&M and return information

H&M Group annual reporting discusses inventory management and the commercial context of its fashion business.
Company reporting does not establish a universal return rate, the profitability of every markdown, or the effect of one supply route.
[Inference] A retailer operating both stores and e-commerce needs to reconcile sales, transfers, and returns across channels before evaluating a style.
A store return of an online sale should remain linked to the original sales cohort even if its resale occurs in a different channel.
A public company report provides context, not the transactional data needed to reproduce the synthetic calculation in section 6.

### Evidence and decision transfer

Fisher and Raman's published work on a fashion retailer examines using early sales information to improve assortment decisions.
Its setting demonstrates the relevance of early information, not a guaranteed benefit for every product or supplier.
The application requires a signal that arrives before the supplier's decision cutoff and a feasible way to change supply.
Research on the newsvendor problem provides a useful underage-overage framework, but fixed size packs, uncertain returns, and substitution complicate direct application.
Treat company reports as company claims, regulatory sources as descriptions of their specified rules, and peer-reviewed work as evidence within its study setting.
Synthetic examples teach arithmetic; they do not measure the financial results of a named company.

### Operational playbook

Before a season, the buyer approves an assortment and a financial envelope, while sourcing confirms qualifying facilities and realistic capacity.
During production, planners follow sample, material, sewing, shipping, and receiving milestones, not just the promised delivery date.
At launch, the commercial team checks whether the item is actually visible, priced correctly, and available in the planned sizes.
During the season, the team evaluates sales and returns against the frozen forecast and remaining supply opportunities.
At season end, the team reconciles disposition and reviews whether assumptions held.
This sequence is a proposed operating method, not a claim that the named companies use this exact process.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Late replenishment | Receipt follows last full-price week | Route ignores post-production steps | Units arrive after useful demand | Backward-plan from sellable receipt. |
| Size distortion | Strong size sales coexist with missing sizes | Sales censored by stock availability | Wrong future size curve | Pair sales with time in stock. |
| Gross-net confusion | High sales and rising return backlog | Refunds attributed to a different week | Overstated performance | Match returns to original cohort. |
| Double-counted returns | Resold units exceed physical returns | Transaction and unit ledgers conflated | Inflated sell-through | Reconcile final disposition. |
| Markdown attribution | Sales rise during concurrent campaign | Price and advertising change together | False causal lift | Randomize or disclose confounding. |
| Inventory pooling error | Same unit offered by two channels | Reservations not deducted | Cancellations and poor service | Maintain atomic availability rules. |
| Hidden minimums | Factory rejects proposed size mix | MOQ or pack applies below style level | Infeasible buy | Obtain written pack constraints. |
| Fabric bottleneck | Fast sewing quote without fabric release | Upstream lead time omitted | Reorder misses season | Reserve or confirm material. |
| Stale master data | One SKU maps to multiple colors | Unversioned product changes | Wrong allocation and reporting | Effective-date mappings and validate. |
| Return lag bias | Recent cohorts appear unusually good | Unreceived returns treated as kept | Premature reorder | Compare cohorts at equal age. |
| Unqualified sourcing | Site not on approved facility list | Low quote bypasses due diligence | Legal and worker-harm exposure | Gate purchase approval by site. |
| Unmeasured salvage | Recovery recorded without disposition costs | Gross recovery mistaken for net value | Overstated contribution | Record recovery and costs separately. |
| Demand double count | Transfer and reorder each claim same demand | Alternatives evaluated independently | Excess inventory | Simulate a common inventory state. |
| Misleading price claim | Reference-price history missing | Promotion lacks substantiation | Compliance and reputational risk | Preserve price history and legal review. |
| Return disposition delay | Quarantine stock accumulates | Inspection capacity omitted | Missed resale window | Track aging and staffing. |
| False precision | One optimal unit quantity shown without ranges | Assumptions treated as certain | Brittle commitment | Report sensitivity and break-even. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| Scenario probabilities sum to one | Expected value needs a distribution | Overlapping or missing cases | Validate weights and stress alternatives. |
| Fast route arrives on time | Makes simulated demand reachable | Port or capacity delay | Model receipt-date scenarios. |
| Fast-route stock starts at zero | Isolates incremental option | Existing units serve demand | Include actual stock and receipts. |
| All fast sales are retained | Simplifies option calculation | Returns differ by channel or size | Model mature cohort return probabilities. |
| Constant unit price | Makes arithmetic transparent | Markdown or promotion changes price | Use dated price and response paths. |
| Demand is independent of quantity | Enables simple scenario comparison | Availability changes conversion or substitution | Model size availability and lost demand. |
| Net recovery is $8 | Values unsold units | Condition or disposal costs vary | Use disposition-specific evidence. |
| Fulfillment is $5 per transaction | Captures repeat shipments | Channel or parcel costs vary | Use channel and return-route costs. |
| Returns can be identified | Prevents duplicate units and refunds | Unmatched order lines | Measure linkage failure and reconcile. |
| Supplier capacity is real | Makes the option feasible | Capacity offered to several buyers | Obtain accepted reservation terms. |
| No fixed overhead in contribution | Isolates variable economics | Decision changes fixed staffing or facilities | Add incremental fixed costs explicitly. |
| Synthetic unit counts are final | Allows exact stock reconciliation | Pending refunds or missing shipments | Close and reconcile cohorts first. |
| One season horizon is sufficient | Values terminal units simply | Carryover has a next-season market | Model carryover, storage, and obsolescence. |
| Responsible-sourcing gate is binary | Ensures prohibited routes are excluded | Evidence is incomplete or risk is evolving | Escalate uncertainty and track remediation. |

Expected value hides downside and concentration of exposure.
A decision maker may prefer lower expected contribution if a high-loss scenario breaches a cash or inventory constraint.
The example omits discount timing, fit changes, demand cannibalization, and the effect of stock availability on conversion.
The lab results are sensitivity illustrations, not a production decision or a causal estimate.

## 12. Exercises

### Exercise 1 — Map the decisions

Time: 15 minutes.
A brand launches dresses in week 1; an offshore supplier needs ten weeks to make the goods sellable and a regional supplier needs three weeks.
The full-price selling window ends after week 10.
Deliverable: a one-page timeline naming buyer, sourcing, allocation, pricing, returns, and compliance owners.
Acceptance: identify the latest useful reorder cutoff for each route, distinguish forecast from commitment, and state which team approves the option.
Assume decisions occur at the start of a week and goods become sellable at the start of the arrival week.

### Exercise 2 — Reconcile the garment ledger

Time: 20 minutes.
Use the case in section 6, but suppose 60 rather than 50 returned garments are resold at markdown and all other counts remain unchanged.
Deliverable: final unit reconciliation, net sales, fulfillment cost, and scoped contribution.
Acceptance: count original and resale transactions separately, refund all 80 returns, and show the recovery units explicitly.

### Exercise 3 — Evaluate a responsive order

Time: 20 minutes.
Use section 7's 100-unit fast-order scenarios and add a $600 reservation fee incurred only if the order is placed.
Deliverable: expected incremental contribution and a comparison against no order.
Acceptance: weight scenario outcomes correctly and subtract the fee once, not once per unit.
Explain one reason the chosen route might still be infeasible.

### Exercise 4 — Diagnose the data

Time: 20 minutes.
A manager compares this week's 50 returns against this week's 200 sales and calls the result the current cohort's 25% return rate.
The product was unavailable in size L for half the selling week.
Deliverable: two concise diagnoses and a corrected measurement plan.
Acceptance: connect returns to purchase cohorts, describe maturity, and explain why size-L sales can understate demand.

### Exercise 5 — Write a data contract

Time: 25 minutes.
A returned online garment is refunded in week 4, received in week 5, and resold in a store in week 7.
Deliverable: a table of event keys, event times, inventory statuses, and links between events.
Acceptance: distinguish refund from receipt, link resale to the garment or return record, and prevent double counting across channels.
State how an unmatched return is escalated.

### Exercise 6 — Design a markdown evaluation

Time: 25 minutes.
A merchant wants to reduce selected jackets from $80 to $40 during a paid advertising campaign.
Deliverable: a test design, primary metric, eligibility rule, guardrail, and analysis of interference.
Acceptance: define a credible counterfactual, account for inventory availability and returns, and avoid equating observed uplift with causal profit.
If randomization is infeasible, identify at least two confounders and label the result observational.

### Exercise 7 — Review a sourcing proposal

Time: 20 minutes.
A factory offers a two-week route at a higher unit cost, but fabric availability is unconfirmed and the factory is not in the approved facility register.
Deliverable: a gate decision and a list of evidence owners and deadlines.
Acceptance: do not treat the quote as capacity, do not approve an unqualified site, and specify material and compliance evidence.
Do not claim a certificate guarantees legal compliance.

### Exercise 8 — Extend the lab

Time: 30 minutes.
Change the fast-route landed cost from $34 to $44 per garment and compare 0, 100, and 200 units.
Deliverable: revised expected values and a short recommendation conditional on assumptions.
Acceptance: deduct the cost on every purchased garment, including leftovers; identify excluded return risk and compare no order.
State whether the exercise proves the chosen quantity is optimal over all possible quantities.

## 13. Self-check answers

### Answer 1

A ten-week route ordered at the start of week 1 arrives at the start of week 11, after the stated window.
A three-week route ordered at the start of week 7 arrives at the start of week 10 and can serve the final week.
A start-of-week-8 order arrives at the start of week 11 and is too late under the exercise convention.
Real decisions must also account for receipt uncertainty, available size mix, and processing after arrival.
Award credit for clear assumptions and accountable owners, not for treating a quote as a commitment.

### Answer 2

Retained full-price units remain 800 − 80 = 720.
Markdown units become 150 + 60 = 210.
Recovered units are 1,000 − 720 − 210 = 70.
Net sales are 720 × $80 + 210 × $40 = $66,000.
Recovery is 70 × $8 = $560.
Fulfillment events are 800 + 210 = 1,010, costing $5,050.
Return handling is 80 × $6 = $480 and acquisition is $24,000.
Contribution is $66,000 + $560 − $5,050 − $480 − $24,000 = **$37,030**.
The $270 increase over the base case is 10 × ($40 − $8 − $5).

### Answer 3

The 100-unit scenario contributions before the fee are $750, $4,100, and $4,100.
Their expected value is 0.25 × $750 + 0.50 × $4,100 + 0.25 × $4,100 = $3,262.50.
After the $600 reservation fee, expected incremental contribution is **$2,662.50** versus $0 for no order.
This is conditional on demand, capacity, timing, price, and no returns being exactly as modeled.
Unconfirmed fabric or a late receipt could make the order infeasible.

### Answer 4

This week's return events generally relate to several earlier sales cohorts.
Divide returns by original sales in the same cohort and compare cohorts at a common elapsed age.
Estimate future returns for immature cohorts using suitable mature analogs and report uncertainty.
The missing size-L inventory suppresses observable sales; record availability intervals and use bounds or an appropriate censored-demand approach.
Do not assert the exact lost demand without additional evidence.

### Answer 5

A strong event contract includes the original order-line key, refund event, physical receipt event, inspection disposition, and subsequent resale event.
It retains event date and recorded date, channel and location, SKU, quantity, and condition/status.
The refund changes the financial ledger in week 4; receipt changes physical stock in week 5 under the specified convention.
Resale removes the unit from sellable stock in week 7 and creates a new sale transaction.
Unmatched events enter a monitored exception queue rather than being silently dropped or automatically matched on weak identifiers.

### Answer 6

A strong design randomizes eligible comparable stores or customer-exposure units where operationally and legally feasible.
It defines contribution after discounts, fulfillment, and matured or estimated returns as the primary outcome.
It checks stock availability, spillovers, customer substitution, and unequal advertising exposure.
Store or customer interference can dilute the apparent difference; a cluster design or explicit limitation may be necessary.
An observational comparison must discuss advertising, seasonality, changing sizes, and selection into markdown.
A before-and-after increase is descriptive and is not sufficient evidence of causality.

### Answer 7

The quoted option is not approved until the responsible team verifies factory eligibility and required due diligence.
Sourcing should obtain material availability, supplier acceptance, route milestones, capacity, and a credible sellable receipt date.
Compliance should assess the site and product-specific evidence against applicable internal policies and law.
The buyer should compare the feasible alternatives, financial risk, and response deadline.
Escalate if the evidence will not arrive before the order deadline; do not silently bypass the gate.

### Answer 8

Raising fast cost by $10 reduces each 100-unit scenario contribution by $1,000 and each 200-unit scenario contribution by $2,000.
Expected contributions become $0, $2,262.50, and $2,012.50 for 0, 100, and 200 units, respectively.
Among those three candidates, 100 has the highest expected contribution under the stipulated model.
This does not establish the best quantity among all integers, account for returns, or prove the route is operationally feasible.
A robust answer also reports sensitivity to price, demand, recovery, capacity, and arrival time.

## 14. Interview questions

1. **Why is a seasonal apparel buy different from routine replenishment?**
   Strong: connects long commitments, short selling windows, uncertainty, and end-of-season recovery.
   Weak: assumes each observed stockout can be replenished immediately.
2. **What is the buyer's decision and objective?**
   Strong: defines assortment, size, sourcing, budget, service constraints, and contribution horizon.
   Weak: offers a forecast accuracy score without an actionable choice.
3. **How do you estimate a size curve when sizes stock out?**
   Strong: pairs availability with sales and bounds or models censored demand and substitution.
   Weak: assumes sales share equals unconstrained demand share.
4. **How do you value a fast supplier?**
   Strong: models the late decision, information, capacity, cost premium, arrival date, and demand remaining.
   Weak: compares only quoted unit prices or total season demand.
5. **How do you calculate sell-through after returns?**
   Strong: states the denominator, cohort, timing, and gross versus retained definition.
   Weak: divides arbitrary weekly sales and returns without matching them.
6. **Can a markdown increase revenue but reduce contribution?**
   Strong: considers price dilution, incremental units, returns, and cost to serve.
   Weak: assumes every markdown sale is incremental and costless.
7. **How would you test a markdown?**
   Strong: specifies counterfactual, randomization or confounders, availability, interference, and return maturity.
   Weak: calls a before-and-after chart a causal estimate.
8. **What makes an apparel return sellable?**
   Strong: identifies receipt, inspection, condition, timing, channel, and disposition.
   Weak: assumes a refund immediately adds pristine stock.
9. **How do you prevent inventory double counting across channels?**
   Strong: explains status, reservation, event identity, physical reconciliation, and shared availability.
   Weak: sums store and online reports without checking whether they share a pool.
10. **What sourcing evidence is needed before ordering?**
    Strong: asks about facility approval, material availability, labor and customs risks, scope, and legal review.
    Weak: treats a price quote or a single certificate as complete assurance.
11. **When can you use a newsvendor calculation?**
    Strong: states cost definitions and demand assumptions, then discusses pack sizes, timing, and substitution.
    Weak: applies a critical fractile to a point forecast with no uncertainty.
12. **How would you explain a model recommendation to a merchant?**
    Strong: gives alternatives, assumptions, downside, break-even, decision owner, and next trigger.
    Weak: presents a precise quantity without feasibility or uncertainty.

## 15. Key terms

- **Allocation:** assignment of available units to locations; see section 5.
- **Assortment:** selected products offered to customers; see section 4.
- **Available-to-promise:** stock or supply eligible for new orders under defined rules; see section 4.
- **Buy quantity:** proposed or committed units for a season; see section 5.
- **Censored demand:** demand partially hidden by unavailable stock; see section 8.
- **Cohort:** group of sales sharing a purchase period or attribute; see section 8.
- **Contribution:** scoped revenue and recovery less stated variable costs; see section 6.
- **Critical fractile:** underage cost divided by total underage and overage costs under a simple model; see section 5.
- **Critical path:** sequence controlling earliest availability; see section 5.
- **Disposition:** final handling of a garment after sale, return, or season; see section 6.
- **Fast route:** responsive supply option with explicitly modeled cost and timing; see section 7.
- **Full-price sell-through:** full-price sales relative to a stated unit base; see section 6.
- **GTIN:** GS1 trade-item identifier; see section 8.
- **Landed cost:** product cost plus specified inbound costs; see section 4.
- **Lead time:** elapsed time between defined supply events; see section 5.
- **Markdown:** a reduction from a stated earlier or reference price; see section 5.
- **Minimum order quantity:** minimum accepted supplier order under defined grouping rules; see section 5.
- **Net sales:** sales after applicable discounts and returns; see section 6.
- **Open-to-buy:** remaining planned purchasing capacity; see section 4.
- **Overage cost:** loss on an additional unit that is not needed at the intended price; see section 5.
- **Recovery:** proceeds from remaining or otherwise disposed units under stated cost treatment; see section 6.
- **Return lag:** elapsed time from sale to return event; see section 8.
- **RFID:** radio-frequency identification for tagged objects; see section 8.
- **Scenario:** coherent possible future with assumptions and, where justified, a probability; see section 7.
- **Size curve:** distribution of demand or allocation across sizes; see section 4.
- **SKU:** stock keeping unit identifying a product variant; see section 4.
- **Stockout censoring:** missing demand because availability constrained sales; see section 10.
- **Substitution:** demand migration to another choice when the preferred one changes or is unavailable; see section 5.
- **Underage cost:** opportunity cost of one unit of unmet demand; see section 5.
- **Working capital:** cash tied up in operational assets net of relevant liabilities; see section 11.

## 16. Sources

1. [GS1, Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin) — identifier definition and scope.
2. [GS1, Global Location Number](https://www.gs1.org/standards/id-keys/gln) — party and location identifiers.
3. [U.S. Federal Trade Commission, Threading Your Way Through the Labeling Requirements Under the Textile and Wool Acts](https://www.ftc.gov/business-guidance/resources/threading-your-way-through-labeling-requirements-under-textile-wool-acts) — U.S. labeling guidance and exceptions.
4. [U.S. Customs and Border Protection, Uyghur Forced Labor Prevention Act](https://www.cbp.gov/trade/forced-labor/UFLPA) — U.S. enforcement information and guidance.
5. [Regulation (EU) No 1007/2011 on textile fibre names and related labelling and marking](https://eur-lex.europa.eu/eli/reg/2011/1007/oj) — official EU legal text.
6. [Regulation (EU) 2024/1781 establishing a framework for ecodesign requirements for sustainable products](https://eur-lex.europa.eu/eli/reg/2024/1781/oj) — framework and product passport provisions; consult later acts for product-specific obligations.
7. [Directive (EU) 2024/825 on empowering consumers for the green transition](https://eur-lex.europa.eu/eli/dir/2024/825/oj) — official directive; check national implementation and applicable dates.
8. [OECD, Due Diligence Guidance for Responsible Supply Chains in the Garment and Footwear Sector](https://www.oecd.org/en/publications/oecd-due-diligence-guidance-for-responsible-supply-chains-in-the-garment-and-footwear-sector_9789264290587-en.html) — risk-based due-diligence guidance.
9. [Inditex, annual reports](https://www.inditex.com/itxcomweb/en/investors/investor-relations/annual-reports) — company-reported business and operating context.
10. [H&M Group, annual and sustainability reports](https://hmgroup.com/investors/annual-and-sustainability-report/) — company-reported financial and operating context.
11. Fisher, M. and Raman, A., “Reducing the Cost of Demand Uncertainty through Accurate Response to Early Sales,” *Operations Research* 44(1), 1996, pp. 87–99. [DOI: 10.1287/opre.44.1.87](https://doi.org/10.1287/opre.44.1.87).
12. Fisher, M. and Raman, A., *The New Science of Retailing*, Harvard Business Review Press, 2010 — retail decision context; not evidence for the synthetic values.

Regulatory descriptions above refer to the cited texts and guidance; verify amendments, current application dates, and local implementation before operational use.
Company annual reports are self-reported evidence and should not be read as causal evaluations of a specific intervention.

## 17. Links

- [Module 29 topic map](_index.md).
- [CPG and retail: trade spend, shelf, and replenishment](01_cpg-retail.md).
- [Automotive sector primer](03_automotive.md).
- [Module 29 survey](../../../course/29_industry_electives.md).
- [Simulation, optimization, and production labs index](../30_simulation_optimization_production_labs/_index.md).
