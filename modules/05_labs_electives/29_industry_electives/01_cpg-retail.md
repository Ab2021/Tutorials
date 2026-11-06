# CPG and retail: trade spend, shelf, and replenishment

> Module 29 — Industry Electives: Sector Primers  
> Topic 01 of 08  
> Estimated study time: 4 hours plus the optional lab extension  
> Prerequisites: Modules 14, 15, and 28  
> Version 2026.5

## 1. Why this topic exists

A consumer packaged goods (CPG) supplier can ship every case on the purchase order and still disappoint the shopper.
The retailer can have inventory recorded in a store while the shelf is empty.
A promotion can increase measured units while reducing the category's total contribution.
A private-label product can improve the retailer's gross margin while changing the supplier's bargaining position.
These outcomes are connected, but they are not interchangeable measurements.

The commercial buyer decides which items receive assortment, shelf space, and promotional support.
The buyer usually owns a category profit or sales objective, not the supplier's factory utilization.
The supplier's account manager negotiates pricing, allowances, display commitments, and service terms.
The retailer's replenishment planner translates demand and inventory information into orders.
The store team receives, stocks, rotates, and counts products.
The shopper decides whether to buy, substitute, defer, or leave.
The analyst's job is to connect these decisions without claiming that one party controls the entire chain.

The immediate decision in this chapter is whether a retailer and a branded supplier should run a four-week price promotion and how they should replenish it.
A related decision is whether to allocate more shelf capacity to a private-label alternative.
The parties need a common decision window, evidence of incremental demand, a feasible order plan, and a clear commercial settlement.
They also need separate economics: a retailer margin gain does not automatically imply a supplier gain.

A wrong decision is costly in several distinct ways:
- A baseline that ignores seasonality credits normal demand to the promotion.
- A forecast that ignores stockouts treats constrained sales as true demand.
- A case-pack error causes an order to exceed backroom or shelf capacity.
- A trade-spend calculation that omits bill-backs overstates supplier contribution.
- A private-label comparison that ignores substitution overstates category growth.
- A contract dispute over eligibility delays or reverses expected settlement.
- An availability measure that uses a daily store total can miss an empty shelf during the evening peak.
- A claimed partnership result may reflect a concurrent distribution change rather than collaboration.

The examples below are **synthetic** unless explicitly identified as a documented public case.
Currency is U.S. dollars (USD), demand is in consumer units, and the principal planning bucket is one week unless otherwise stated.
Educational discussion of regulation is not legal advice; rules depend on jurisdiction and current facts.

## 2. Learning outcomes

After working through this topic you can:

1. **Map** the shopper-to-supplier decision chain and identify the buyer, planner, store, and supplier owners; demonstrate this in Exercise 1.
2. **Compute** retailer and supplier incremental promotion contribution without counting transfer payments twice; demonstrate this in Exercise 2.
3. **Design** a replenishment plan with a lead time, review period, stock target, case-pack rounding, and capacity check; demonstrate this in Exercise 3.
4. **Distinguish** distribution, shelf availability, sales, and underlying demand; demonstrate this in Exercise 4.
5. **Critique** a private-label allocation using category incrementality, substitution, capacity, and separate party economics; demonstrate this in Exercise 5.
6. **Specify** a usable retail data contract and a promotion evaluation with an appropriate counterfactual; demonstrate this in Exercise 6.
7. **Write** an accountable retailer-supplier decision brief with controls for settlement, privacy, and legal review; demonstrate this in Exercise 7.
8. **Adapt** the runnable lab to a sensitivity case and explain what its output cannot establish; demonstrate this in Exercise 8.

A successful response is not simply a forecast.
It states the decision, decision owner, horizon, objective, constraints, evidence, and review trigger.
It distinguishes a **forecast** of expected demand from a **plan** for operations.
It also distinguishes a **target**, such as a desired service level, from a **commitment**, such as an accepted purchase order.

## 3. The core mental model

```text
                      commercial agreement
 Supplier account  <-------------------------->  Retail category buyer
       |                       |                          |
       | forecast / supply     | allowance / terms        | price / space
       v                       v                          v
 Supplier plant --> retailer DC --> store backroom --> shelf --> shopper
       ^                ^                ^              |          |
       |                |                |              |          +--> POS sales
       |                |                |              +--> availability
       |                |                +--> receipts / inventory
       |                +--> orders / inventory
       +--> production / shipment / invoice / claims

       Evaluation: compare observed outcomes with a defensible counterfactual;
       reconcile retailer and supplier economics, then decide the next action.
```

The physical flow moves from supplier through a retailer distribution center (DC) to the store and shopper.
Some businesses use direct-store delivery or an intermediary; the diagram is a common pattern, not a universal one.
The information flow moves in both directions and may arrive late or at different levels of aggregation.
The contractual flow controls prices, allowances, evidence requirements, and settlement, not the physical availability of every unit.

The shopper sees a shelf price and the products actually available at the time of purchase.
Point-of-sale (POS) records show completed transactions, not necessarily the purchases lost when the shelf is empty.
A promotion changes exposure, price, expected demand, and often competitor behavior at the same time.
The retailer typically decides the final retail price, subject to its contracts and applicable law.
The supplier can propose terms, but an allowance is not equivalent to a guaranteed shopper price reduction.

The retailer and supplier each have an objective.
The retailer may seek category contribution, revenue, customer loyalty, availability, or a balance of these.
The supplier may seek contribution after trade spend, sustainable volume, distribution, and service.
They may share some gains and disagree over others.
The analytical task is to make the disagreement measurable rather than to assume a single joint profit target.

### Three separate lenses

1. **Shopper lens:** Was the desired product findable, available, and worth buying at the offered price?
2. **Retail lens:** Did the total category improve after considering margin, labor, space, and substitution?
3. **Supplier lens:** Did incremental contribution exceed discounts, funding, production, and service costs?

All three lenses must use the same time window and unit definition.
They do not have to use the same objective or share commercially sensitive data.
An agreement can specify a minimum common dataset and a neutral calculation without revealing every party's costs.

### The counterfactual is a model, not a ledger row

The baseline asks what would have happened without the action.
It is not directly observed for a promoted store during the same period.
A control group, historical seasonal baseline, or causal model can estimate it with different assumptions.
Any resulting incremental effect is an estimate, not a physical count visible on an invoice.
The settlement ledger, by contrast, records transactions and contractual claims.
Do not substitute one for the other.

## 4. Definitions and vocabulary

| Term | Working definition and unit | Common misuse |
|---|---|---|
| CPG | Frequently purchased packaged consumer products; product category | Assuming every item has identical shelf life or demand |
| Retailer | Firm selling to shoppers; legal entity | Treating the buyer, store, and DC as one decision maker |
| Supplier | Firm selling products to the retailer; legal entity | Assuming it controls the final shelf price |
| Category | Group managed together, such as cereal; retailer-defined scope | Restricting analysis to one promoted brand |
| Stock keeping unit (SKU) | An operationally distinct item, normally a specific size and variant | Treating a family of pack sizes as one unit |
| Global Trade Item Number (GTIN) | GS1 identifier for a trade item at a defined packaging level | Using one GTIN interchangeably for a case and each consumer unit |
| Global Location Number (GLN) | GS1 identifier for a party or location | Assuming every internal store code is a GLN |
| POS | Record of completed consumer sales, typically units and currency by time and location | Equating sales to unconstrained demand |
| Sell-in | Supplier shipments or sales to a channel, in cases or units | Calling it shopper consumption |
| Sell-through | Sales onward through the channel, often POS units | Assuming every system uses the same denominator |
| Baseline | Estimated outcome without the decision, in units per defined period | Treating last week's sales as automatically causal |
| Lift | Change relative to baseline, in units or percent | Using gross promoted sales as incremental sales |
| Incrementality | Portion of change caused by an action compared with a counterfactual | Counting shifted or cannibalized sales as new category demand |
| Cannibalization | Sales displaced from another item or brand, in units or contribution | Assuming each displaced unit has the same margin |
| Pantry loading | Purchases moved earlier for later consumption | Calling all promotion-period sales permanent growth |
| Forward buying | Channel purchases made early to benefit from a temporary commercial term | Confusing retailer purchasing with shopper demand |
| Trade spend | Supplier-funded commercial payments, discounts, or allowances; USD | Counting a claim and its underlying allowance twice |
| Bill-back | Claim for reimbursement under an agreement; USD | Assuming a submitted claim has been validated and paid |
| Off-invoice allowance | Discount applied directly to an invoice; USD per agreed eligible quantity | Applying it again as a bill-back |
| Scan allowance | Funding tied to eligible consumer sales, often USD per unit | Funding shipments when the contract specifies POS sales |
| Display allowance | Funding linked to agreed merchandising execution | Assuming the display was installed without evidence |
| Promotional price | Shopper-facing price during a defined period; USD per unit | Assuming a supplier-funded amount necessarily passes through |
| Retail gross margin | Retail sales less acquisition cost, before specified operating costs; USD or percent | Mixing percent of sales with percent markup on cost |
| Supplier contribution | Net supplier revenue less specified variable costs and funding; USD | Calling it net income or including fixed cost inconsistently |
| Private label | Retailer-owned or retailer-controlled brand sold under its specifications | Assuming the retailer manufactures the product |
| Branded item | Product sold under a supplier-controlled brand | Assuming brand loyalty eliminates substitution |
| Assortment | Items authorized for sale at a store or channel | Treating authorized items as physically stocked |
| Planogram | Store-specific or cluster-specific shelf placement and space plan | Assuming paper compliance guarantees on-shelf stock |
| Facing | One product width presented at the shelf front | Treating facings as a direct measure of all backroom inventory |
| Shelf capacity | Maximum units fitting in the assigned shelf location | Assuming a daily replenishment can exceed actual capacity |
| Backroom inventory | Physical store stock outside the customer-facing shelf | Calling it immediately shopper-available |
| On-shelf availability | Defined share of item-store-time observations with product available to shoppers | Using recorded store inventory as a direct shelf observation |
| Distribution | Defined share of stores, sales weight, or relevant market carrying an item | Mixing numeric and sales-weighted distribution |
| Inventory position | On hand plus on order minus backorders, under a stated convention | Adding the same in-transit receipt twice |
| Lead time | Time from order release to usable receipt, in days or weeks | Excluding receiving delay when stock cannot yet be sold |
| Review period | Interval between ordering decisions, in days or weeks | Assuming continuously reviewed stock in a weekly ordering system |
| Safety stock | Extra stock held against modeled uncertainty, in units | Using it to repair an unmodeled systematic bias |
| Order-up-to level | Target inventory position before a replenishment order, in units | Confusing the target with a commitment or actual order |
| Case pack | Number of consumer units in an orderable case | Rounding down when service depends on full cases |
| Fill rate | Fraction of ordered quantity supplied under a defined measurement rule | Calling it the probability every shopper finds a product |
| Shrink | Inventory lost through damage, theft, expiry, or other causes | Treating unexplained inventory differences as proven theft |
| Days of supply | Inventory divided by a specified demand rate, in days | Using a zero or stale demand rate without a rule |
| EDI | Electronic data interchange using agreed business message standards | Assuming a transmitted order was accepted or received |
| Forecast | Estimate of future demand with uncertainty and assumptions | Treating it as a purchase order |
| Plan | Intended operational response given constraints and forecast | Treating it as executed fact |
| Target | Desired outcome or policy threshold | Treating it as a measured result |
| Commitment | Binding or accepted obligation as defined in the agreement | Assuming every draft forecast is a binding obligation |

## 5. The mechanics, step by step

### Step 1: Define the decision and the people who can make it

**Input:** a trading-period proposal, category strategy, and the decision calendar.
**Output:** a decision statement with named owners and a review date.

Write the question in a form that admits an action.
For example: should the retailer run a four-week 12% price reduction on a specific cereal SKU, with supplier funding, in 120 stores?
Identify who may approve the shopper price, who may approve supplier funding, and who may change orders or space.
Record the planned start and end in local store time and the ordering cutoff in the relevant system time zone.

The category buyer controls assortment and usually approves retail merchandising.
The supplier account owner proposes commercial terms and confirms its funding authority.
The replenishment planner owns the operational order recommendation, subject to actual system permissions.
The supply planner confirms factory, transport, and inventory feasibility.
The store operations owner confirms merchandising and labor feasibility.
The finance or revenue-management owner validates the settlement and the contribution definition.

**Branch:** if no one can make the decision before the order cutoff, move the start date or reject the proposal.
A sophisticated estimate delivered after the allocation window is not decision support.

### Step 2: Define the objective and boundaries

**Input:** retailer and supplier objectives, approved accounting definitions, and the category scope.
**Output:** a metric dictionary and a comparison window.

Retailer contribution can be defined as shopper revenue minus net acquisition cost, promotional execution cost, and other agreed variable costs.
Supplier contribution can be defined as net sales to the retailer minus variable product and logistics cost and supplier-funded allowances.
Both definitions must specify taxes, returns, breakage, and whether fixed costs are excluded.
If the supplier allowance is deducted from retailer acquisition cost, it must also be deducted from supplier economics.
Do not add both party contributions and then add the allowance again: it is a transfer between the parties.

Select a category scope broad enough to observe important switching.
Select a pre-period and a post-period long enough to detect stockpiling and delayed purchase where practical.
Define whether the primary outcome is sales units, category contribution, availability, or another measure.
Use a common unit, such as consumer units, and preserve original case quantities for settlement.

**Branch:** if the retailer cares about loyalty or basket effects but cannot measure them reliably, state that the quantified result excludes those effects.
Do not label an unmeasured benefit as zero or include it as an invented monetary gain.

### Step 3: Build the product and location spine

**Input:** item master, packaging hierarchy, store master, assortment authorization, and time-effective mappings.
**Output:** a dated SKU-store-week analysis grid.

Map each consumer unit GTIN to the case GTIN and the effective case pack.
Map a supplier product code to the retailer item number using effective dates.
Retain pack size and unit of measure so that a 12-ounce package is not treated as a 12-unit case.
Resolve store openings, closures, remodels, and channel differences for each week.
Keep an authorization indicator separate from the first observed sale.

Document whether sales refer to checkout date, accounting date, or shipment date.
Map the promotion calendar to the same store-local time definition.
Identify missing mappings and quarantine ambiguous records for review rather than silently dropping them.
A change in pack size or formulation may require a new identifier and must not be spliced into a homogeneous series without an explicit mapping decision.

**Branch:** if a store was not authorized to sell the item, exclude it from item-level on-shelf availability but consider it when assessing distribution changes.
If an authorized store has no data, distinguish missing observation from observed zero sales.

### Step 4: Establish a credible baseline

**Input:** historical POS, prices, distribution, availability, holiday calendar, and comparable unpromoted observations.
**Output:** baseline unit demand and an uncertainty range by item-store-week or an agreed aggregation.

Start with a transparent seasonal or comparable-week baseline.
Remove or model prior promotions and other known unusual events.
Control for store openings, price changes, competitor actions where observed, and distribution changes.
When a store was out of stock, observed sales may be lower than unconstrained demand.
Flag those observations and use a documented correction or exclude them, depending on data quality.

A simple historical baseline is useful for planning but is not automatically an unbiased causal effect.
For evaluation, randomized store assignment is preferable when operationally and commercially feasible.
A matched control or difference-in-differences design can be useful if its assumptions are credible.
Check pre-period comparability and trends, record concurrent changes, and inspect spillovers between nearby stores.
A control store may be contaminated when a shopper changes stores to obtain the offer.

**Branch:** if promotion assignment follows expected local demand, a naive promoted-versus-unpromoted comparison can be biased.
Use a design that addresses the assignment mechanism, or report the result as descriptive rather than causal.

### Step 5: Estimate demand and substitution

**Input:** baseline, promotion mechanics, price history, elasticity evidence, and availability assumptions.
**Output:** a forecast distribution for promoted and related items.

Calculate the relative price change against the actual regular price and identify its effective days.
Estimate uplift from comparable events or a model whose training conditions resemble the planned event.
For a single-item forecast, separate baseline units from promotion-increment units.
For category evaluation, estimate changes to private label, other brands, and pack sizes.

Treat a same-period reduction in another SKU as observed displacement only after accounting for its own baseline and availability.
Also examine the period after the promotion for pantry loading.
A household that buys twice as much today and half as much next month may not create the apparent four-week increase in category demand.
Sales at a promoted store may also shift from a nearby store or from another retail channel.

Use low, central, and high scenarios, not a single pseudo-precise number.
The scenarios should vary coherent drivers such as uplift, cannibalization, and service rather than independently changing every number without explanation.
Record whether the forecast represents demand conditional on perfect availability or expected sales under the planned service level.

**Branch:** if price elasticity is estimated outside the observed price range, flag extrapolation and use wider uncertainty.
If the item has little comparable history, use a conservative analogy and plan an experiment.

### Step 6: Check shelf and store execution

**Input:** planogram, capacity by store cluster, shelf refill frequency, promotional display specification, and labor plan.
**Output:** a feasible merchandising plan and an on-shelf monitoring design.

Translate facings, depth, and stack height into physical capacity using the actual package dimensions.
Check shelf capacity separately from backroom capacity.
A store can have enough backroom stock and still lose sales if no one refills the shelf during the demand peak.
A display may expand shopper-facing capacity, but only if the store actually installs and maintains it.

Estimate the peak demand between shelf replenishments rather than relying only on weekly averages.
For example, 84 units per week averages 12 units per day, but a Friday evening peak may exceed that average.
If a shelf holds 18 units and the peak interval requires 25 units, arrange a refill, add capacity, or accept a service risk.
Include display assembly and replenishment labor in the economics when they are incremental.

Specify the observation method: store associate audit, image analysis with validation, shelf sensor, or a qualified sales-pattern proxy.
Record item, store, timestamp, and whether the shopper-facing location was actually checked.
A zero-sale interval is not conclusive evidence of a stockout for a slow-moving product.

**Branch:** if the available shopper-facing capacity cannot support the planned demand and labor cannot refill it, revise the promotion or reduce the store set.
A warehouse order alone does not solve an in-store execution constraint.

### Step 7: Translate demand into an order plan

**Input:** demand forecast, lead time, review period, inventory position, safety stock, case pack, and capacity limits.
**Output:** recommended order in full cases, with exceptions.

For a periodic review policy, the protection period is the lead time plus the review period.
A simple order-up-to target equals forecast demand during that protection period plus safety stock.
Inventory position equals on hand plus eligible on-order stock minus backorders, using consistent timing.
The unconstrained order is the positive difference between target and inventory position.
Round up to a full case when the ordering rules require it.

For independent daily demand with a fixed lead time, a simple safety-stock approximation is `z × daily demand standard deviation × sqrt(protection days)`.
The `z` value is a chosen normal quantile for a cycle-service target, not a universal fill-rate conversion.
Promotion demand, correlated days, uncertain lead times, and intermittent demand can invalidate this approximation.
For those conditions, use an empirical demand and lead-time distribution or a simulation.

Apply shelf, backroom, DC, transport, supplier capacity, expiry, and budget constraints after calculating the unconstrained recommendation.
If a capacity cap reduces the order, report the shortfall and decide whether to split delivery, adjust space, reduce promotion, or accept the risk.
Do not silently replace the target with a constrained order and then report the original service target as achieved.

**Branch:** if inventory records are unreliable, a physical count or cycle count may be more valuable than additional forecast precision.
If an order is already committed, distinguish changeable supply from the accepted commitment.

### Step 8: Reconcile retailer and supplier economics

**Input:** expected units, shopper prices, invoice cost, allowances, variable costs, and execution expenses.
**Output:** separate incremental contribution bridges and a clearly labeled combined view.

For each scenario compute retailer revenue as consumer units sold times shopper price.
Compute retailer acquisition cost using the agreed invoice cost and any eligible allowance.
Subtract incremental execution costs and the contribution that would have occurred in the baseline.
For the supplier compute eligible sell-in or sell-through according to the contract, not whichever gives a preferred answer.
Subtract unit variable cost and all relevant funding once.

Separate promotion effect from distribution effect and changes in pipeline inventory.
A retailer can buy product during the promotional window that is sold to shoppers later.
To compare parties over an event, either extend the horizon until inventory normalizes or include a transparent inventory adjustment.
A shipment-based allowance on 10,000 units is not the same as a scan allowance on 8,000 shopper units.

Compute both the branded-item contribution and the wider category effect when data allow.
A supplier may gain even when the retailer loses after private-label displacement; the reverse is also possible.
A positive combined contribution does not specify how the gain should be divided.
The commercial agreement determines the allocation, subject to applicable law.

**Branch:** if the event is negative under the central case but strategically justified by an unquantified objective, document the decision maker and limit of exposure.
Do not relabel an unmeasured strategic benefit as measured profit.

### Step 9: Compare private-label and branded space

**Input:** current category demand, candidate assortment, shelf capacity, item contribution, substitutions, and supply constraints.
**Output:** a space and assortment recommendation with an uncertainty range.

Compare products on the same unit basis, or use a common volume or serving basis when pack sizes differ.
Measure the difference between retailer margin per unit and contribution per unit of scarce shelf space.
Assess shopper switching when an item loses a facing or is delisted.
Some shoppers may switch within the category, while others may leave the store or buy at a different time.

Private label may be sourced from a contract manufacturer and can expose the retailer to specification, quality, recall, and supply risks.
A branded supplier may provide demand generation or innovation, but the value of those activities must be evidenced rather than assumed.
Retailer bargaining strength depends on alternatives, customer preference, information, shelf access, and contract terms.
It is not a fixed property of every negotiation.

**Branch:** if demand for the added private-label space is mostly diverted from a higher-contribution item, an apparent unit gain can reduce category profit.
If private-label supply is constrained, the nominal margin advantage may not be realizable.

### Step 10: Agree execution and settlement controls

**Input:** commercial terms, eligibility definitions, data access rules, and execution requirements.
**Output:** signed terms or an explicitly uncommitted proposal, plus an audit trail.

Specify product identifiers, eligible stores, start and end timestamps, funding rate, quantity basis, and maximum funding if applicable.
Specify whether returns reverse eligibility and how duplicate claims are resolved.
Define what evidence proves a display or feature placement occurred.
Name the person who approves exceptions and the deadline for submitting a claim.

Reconcile purchase orders, shipments, receipts, invoices, POS records, and submitted claims at their appropriate grains.
An EDI acknowledgment can confirm that a message was received; it does not by itself establish that product was received or a claim is valid.
Track claim status as submitted, validated, disputed, approved, paid, or reversed.
Retain source versions so a corrected POS file does not silently overwrite the calculation used in settlement.

**Branch:** if parties disagree on the eligibility definition, resolve it before the event rather than encoding competing assumptions as one number.
Escalate legal and competition-sensitive questions through the designated review process.

### Step 11: Monitor, intervene, and learn

**Input:** daily sales and availability, inventory and inbound status, store execution, and actual claims.
**Output:** exception queue, operational actions, and a post-event decision record.

Monitor at a cadence fast enough to act but slow enough to avoid noise-driven interventions.
Separate a late inbound shipment from a store shelf-refill problem; their remedies differ.
Set a clear trigger, such as a confirmed shelf stockout at a high-volume store, and an owner for each action.
Preserve the original forecast and record every intervention so the evaluation can account for treatment changes.

After the event, reconcile actual prices, store participation, claims, and product availability.
Compare results with the pre-agreed counterfactual and inspect a post-period for shifted demand.
Report the point estimate, uncertainty, data coverage, and limitations together.
Close with an action: repeat, modify, test again, or stop.

## 6. Worked example

### Situation and decision

**Synthetic example:** A retailer and supplier are considering a four-week promotion for a branded 400-gram cereal item across 120 similar stores.
All figures below describe the total participating store group over four weeks unless a different unit or window is stated.
The retailer owns the shopper price and shelf decision.
The supplier decides whether to offer a scan allowance and commits supply only after accepting the purchase order.
The plan assumes participating stores have enough shelf and backroom capacity after a display and refill plan.

The central baseline forecast is 10,000 branded consumer units at a regular shelf price of USD 4.00 per unit.
The proposed promotional price is USD 3.50 per unit.
The retailer pays an invoice cost of USD 2.40 per branded unit.
The supplier's variable cost, including the defined variable distribution cost, is USD 1.50 per branded unit.
The supplier offers a USD 0.30 scan allowance per eligible promoted unit, paid to the retailer.
The central demand forecast under the promotion is 14,000 branded units.
The retailer expects an incremental display and store labor cost of USD 800 for the event.

A private-label cereal is in the same category.
The baseline forecast for that item is 8,000 units at a retail contribution of USD 1.10 per unit.
The promotion scenario forecasts 7,000 private-label units, all else equal.
The displaced 1,000 private-label units are a scenario assumption, not a measured causal fact.
No other category effects are quantified in the base case.

The baseline, uplift, and displacement are forecasts, not commitments.
The plan is to offer the promotion and arrange supply if approved.
The retailer's service target is stated separately in the replenishment example below.
A purchase order becomes a commitment only under the parties' applicable acceptance terms.

### Retailer baseline

Branded baseline revenue is:

`10,000 units × USD 4.00/unit = USD 40,000`.

Branded baseline acquisition cost is:

`10,000 units × USD 2.40/unit = USD 24,000`.

Branded baseline contribution before event-specific operating costs is:

`USD 40,000 − USD 24,000 = USD 16,000`.

Private-label baseline contribution is:

`8,000 units × USD 1.10/unit = USD 8,800`.

The modeled baseline category contribution is therefore USD 24,800.
This is not the retailer's entire profit and excludes unrelated category items and fixed costs.

### Retailer promotion scenario

Promoted branded revenue is:

`14,000 units × USD 3.50/unit = USD 49,000`.

Gross invoice acquisition cost is:

`14,000 units × USD 2.40/unit = USD 33,600`.

The eligible scan allowance is:

`14,000 eligible units × USD 0.30/unit = USD 4,200`.

Net branded acquisition cost after the allowance is USD 29,400.
Branded contribution before event labor is USD 49,000 − USD 29,400 = USD 19,600.
Private-label contribution is `7,000 × USD 1.10 = USD 7,700`.
After USD 800 incremental execution cost, modeled category contribution is:

`USD 19,600 + USD 7,700 − USD 800 = USD 26,500`.

The retailer's modeled incremental category contribution is:

`USD 26,500 − USD 24,800 = USD 1,700`.

The branded item alone appears to improve by USD 3,600 before execution cost.
The category view deducts USD 1,100 of modeled private-label displacement and USD 800 of event cost.
The arithmetic is not a causal estimate; the demand inputs must be validated after the event.

### Supplier economics

For this example, assume the supplier's sell-in equals shopper sell-through within the analysis horizon and no residual inventory changes.
This assumption is restrictive and is included only to make the two-party bridge transparent.
Supplier baseline contribution is:

`10,000 units × (USD 2.40 − USD 1.50)/unit = USD 9,000`.

Supplier promotion contribution after the scan allowance is:

`14,000 units × (USD 2.40 − USD 1.50 − USD 0.30)/unit = USD 8,400`.

The supplier's modeled incremental contribution is USD 8,400 − USD 9,000 = **minus USD 600**.
The supplier gains 4,000 units of volume but loses contribution at the stated funding rate.
The retailer's modeled gain and supplier's modeled loss are different perspectives, not a contradiction.

The combined modeled change is USD 1,700 − USD 600 = **USD 1,100**.
The scan allowance is a transfer: it increases the retailer calculation and decreases the supplier calculation by the same USD 4,200.
It must not be added again to the combined result.
This combined number still excludes unmeasured items, taxes, fixed costs, and any private-label manufacturer's economics.

### Break-even and decision

If all other scenario quantities remain fixed, the supplier needs at least 15,000 units to match the USD 9,000 baseline contribution:

`USD 9,000 / (USD 2.40 − USD 1.50 − USD 0.30) = 15,000 units`.

That threshold is not an estimate of achievable demand.
It also assumes every additional unit incurs the same variable cost and allowance and ignores potential capacity or stockout effects.
The supplier could negotiate a lower funding rate, request a different event design, or decline.
The retailer could evaluate whether another design produces adequate category contribution and service.
Any negotiated retail price remains the retailer's decision subject to the applicable agreement and law.

### Store replenishment illustration

Consider one representative store with a forecast of 84 promoted units per week, or 12 units per day on average.
The store reviews orders every 7 days and the usable receipt lead time is 3 days.
The protection period is 10 days.
Expected protection-period demand is `12 units/day × 10 days = 120 units`.
For this simplified policy, use daily demand standard deviation of 4 units, independent daily demand, and a fixed lead time.
For a normal approximation with a chosen cycle-service quantile `z = 1.645`, safety stock is:

`1.645 × 4 units/day × sqrt(10 days) = 20.81 units`.

Round the target upward: `ceil(120 + 20.81) = 141 units`.
Suppose physical on-hand is 65 units, accepted on-order stock is 20 units, and backorders are zero.
The inventory position is `65 + 20 − 0 = 85 units`.
The unconstrained order is `141 − 85 = 56 units`.
With 12 consumer units per case, order `ceil(56/12) = 5 cases`, or 60 units.
The post-order inventory position is 145 units, four above the target because of case rounding.

This is a planning calculation, not a guarantee of 95% fill rate.
The `z` value approximates a 95% cycle-service quantile only under the stated distributional assumptions.
If store capacity permits only 48 more units, the planner must order at most four cases or change the delivery arrangement.
That constraint reduces the protection available; record the shortfall rather than claiming the unconstrained service target.
The average representative store is not a substitute for a store-level plan across 120 stores.

## 7. Code, query, or calculation

The following standalone Python 3 lab uses only the standard library.
It reproduces the central contribution bridge and the representative-store order.
It treats all values as synthetic and assumes sell-in equals sell-through during the horizon.
It is a deterministic scenario calculator, not a causal model or production ordering system.
Save the block as `cpg_lab.py` and run `python cpg_lab.py`.

```python
from math import ceil, sqrt

# Four-week group-level scenario, in USD and consumer units.
base_brand_units = 10_000
promo_brand_units = 14_000
base_private_units = 8_000
promo_private_units = 7_000
regular_price = 4.00
promo_price = 3.50
invoice_cost = 2.40
supplier_variable_cost = 1.50
scan_allowance = 0.30
private_contribution = 1.10
execution_cost = 800.00

# The allowance is paid once per eligible promoted POS unit.
retailer_base = (
    base_brand_units * (regular_price - invoice_cost)
    + base_private_units * private_contribution
)
retailer_promo = (
    promo_brand_units * (promo_price - invoice_cost + scan_allowance)
    + promo_private_units * private_contribution
    - execution_cost
)
supplier_base = base_brand_units * (invoice_cost - supplier_variable_cost)
supplier_promo = promo_brand_units * (
    invoice_cost - supplier_variable_cost - scan_allowance
)
retailer_delta = retailer_promo - retailer_base
supplier_delta = supplier_promo - supplier_base
combined_delta = retailer_delta + supplier_delta
break_even_units = supplier_base / (
    invoice_cost - supplier_variable_cost - scan_allowance
)

# One store's periodic-review plan, in days and consumer units.
daily_mean = 12.0
daily_sd = 4.0
lead_days = 3
review_days = 7
z = 1.645
on_hand = 65
on_order = 20
backorders = 0
case_pack = 12
protection_days = lead_days + review_days
safety_stock = z * daily_sd * sqrt(protection_days)
target = ceil(daily_mean * protection_days + safety_stock)
inventory_position = on_hand + on_order - backorders
unconstrained_units = max(0, target - inventory_position)
order_cases = ceil(unconstrained_units / case_pack)
order_units = order_cases * case_pack

print(f"Retailer baseline contribution: USD {retailer_base:,.2f}")
print(f"Retailer promotion contribution: USD {retailer_promo:,.2f}")
print(f"Retailer incremental contribution: USD {retailer_delta:,.2f}")
print(f"Supplier baseline contribution: USD {supplier_base:,.2f}")
print(f"Supplier promotion contribution: USD {supplier_promo:,.2f}")
print(f"Supplier incremental contribution: USD {supplier_delta:,.2f}")
print(f"Combined incremental contribution: USD {combined_delta:,.2f}")
print(f"Supplier break-even volume: {break_even_units:,.0f} units")
print(f"Safety stock: {safety_stock:.2f} units")
print(f"Order-up-to target: {target} units")
print(f"Inventory position: {inventory_position} units")
print(f"Order: {order_cases} cases / {order_units} units")
```

Expected output:

```text
Retailer baseline contribution: USD 24,800.00
Retailer promotion contribution: USD 26,500.00
Retailer incremental contribution: USD 1,700.00
Supplier baseline contribution: USD 9,000.00
Supplier promotion contribution: USD 8,400.00
Supplier incremental contribution: USD -600.00
Combined incremental contribution: USD 1,100.00
Supplier break-even volume: 15,000 units
Safety stock: 20.81 units
Order-up-to target: 141 units
Inventory position: 85 units
Order: 5 cases / 60 units
```

### Lab procedure

1. Run the unchanged scenario and compare each line with the worked bridge.
2. Change `promo_brand_units` to 15,000 units while holding the other inputs fixed.
3. Record the revised supplier contribution and the retailer contribution.
4. Explain why holding private-label displacement fixed may be unrealistic.
5. Restore the central values and set `on_order` to zero units.
6. Calculate the new order in cases, explaining case-pack rounding.
7. Restore the inputs and change the review period from 7 to 3 days.
8. Explain how the shorter protection period changes the target under the same distributional assumptions.
9. Add an explicit capacity constraint only after specifying whether it applies to order units or eventual physical stock.
10. Record which party owns each assumption and which source would validate it.

The code does not validate contractual eligibility, model returns, optimize price, infer lost sales, or simulate stockout probability.
It also assumes a positive supplier contribution per promoted unit when calculating break-even.
Before adapting it to actual decisions, use approved currency rounding rules and a decimal representation suitable for financial settlement.
Do not expose confidential supplier costs or individual shopper records in a shared notebook without authorization.

## 8. Data, standards, and regulatory context

### Data contract

A minimum event dataset should have an explicit grain and a time-effective product and location key.
The following table is an example specification, not a claim that every organization stores these fields identically.

| Dataset | Minimum grain | Important fields | Checks and limitations |
|---|---|---|---|
| Item master | Item and effective date | Retail item, supplier item, consumer GTIN, case GTIN, pack size | Check overlapping mappings and pack changes |
| Store master | Store and effective date | Store ID, location, channel, time zone, status | Distinguish closure from missing data |
| Assortment | Item, store, effective interval | Authorized status, start, end | Authorized does not mean on shelf |
| POS | Item, store, date or transaction | Units, net revenue, returns, selling price | Check returns and time-zone boundaries |
| Promotion | Item, store group, interval | Offer type, scheduled price, funding, display | Actual execution may differ from plan |
| Inventory | Item, location, timestamp | On hand, on order, in transit, damaged | Snapshot timing and ownership matter |
| Orders | Order line | Order quantity, unit, acceptance, due date | A transmitted order is not always accepted |
| Receipts | Receipt line | Received cases, units, timestamp, discrepancies | Record actual usable receipt separately |
| Shelf audit | Item, store, timestamp | Availability, facing, display, observer or method | Sampling can miss peak-period stockouts |
| Trade claims | Claim line | Eligibility basis, amount, status, evidence | Avoid duplicate off-invoice and bill-back credit |
| Costs | Party, item, effective interval | Invoice cost, allowance, variable cost | Restrict access and document excluded costs |
| Calendar | Date and geography | Week, holiday, event, time zone | Align fiscal and calendar week definitions |

Record completeness, latency, revision policy, access rights, and owner for each dataset.
Retain both event-time and ingestion-time fields where late arrivals matter.
Use a single versioned definition of the four-week event window.
For store-level causal studies, preserve assignment and actual participation separately.
If 120 stores were assigned but 110 executed, report both populations and the chosen estimand.

### Identifiers and messaging

GS1 describes GTINs as identifiers for trade items and GLNs as identifiers for parties and locations.
A company should follow current GS1 allocation and product-change rules for its market and packaging level.
A consumer unit and an orderable case may have distinct identifiers; the item master must preserve the hierarchy.
The identifier identifies the item; it does not, by itself, prove current shelf availability or item quality.
GS1 standards are industry standards and become binding only where incorporated into law, a contract, or a trading partner's requirements.

Trading partners commonly exchange business documents using electronic data interchange.
In the U.S., ANSI ASC X12 maintains standards including purchase order, invoice, and advance ship notice transaction sets.
Trading partner implementation guides specify the fields and business rules actually required in a particular connection.
A purchase order message, an acknowledgment, an advance ship notice, and a physical receipt describe different events.
They should be reconciled rather than treated as interchangeable proof of delivery.

### Labeling, food safety, and traceability

For U.S. packaged food, the Food and Drug Administration (FDA) provides labeling guidance and information on the Food Safety Modernization Act (FSMA).
Applicability depends on the product, facility, activity, and current regulation.
As of 6 November 2026, check the FDA's current Food Traceability Rule page for the applicable compliance date and any updates rather than assuming an older published date is still controlling.
The rule concerns foods on the Food Traceability List and relevant persons in the supply chain, subject to its terms and exemptions.
A GTIN or retailer SKU by itself is not evidence that all required traceability records exist.
For a recall, the operational data contract should be able to link affected lots, shipments, locations, and remaining stock where applicable.

A private-label retailer and its manufacturer must establish responsibilities for specifications, labeling, quality assurance, incident escalation, and recall execution.
The contractual allocation of tasks does not by itself determine every legal responsibility.
Seek qualified legal and food-safety advice for the actual product and jurisdiction.
Other CPG categories, including cosmetics and household chemicals, have different product-specific requirements.

### Privacy, pricing, and competition

Aggregate store-item data may avoid direct shopper identifiers but can still be commercially sensitive.
Loyalty-card or household-level data may be personal information under applicable privacy laws.
Limit collection and access to the purpose, use approved retention periods, and involve privacy counsel where required.
A hashed identifier is not automatically anonymous.

Retail pricing, supplier allowances, resale restrictions, and competitor information exchange can raise competition or unfair-practice issues depending on facts and jurisdiction.
The U.S. Federal Trade Commission (FTC) provides guidance on pricing and antitrust, including the Robinson-Patman Act.
Do not infer that every differential allowance or private-label negotiation is lawful or unlawful from its name alone.
Use legal review for proposed terms, data sharing, and collaborations between competitors.
In particular, do not share a competitor's confidential future pricing as part of an analytical exercise.

### Measurement conventions

A stockout rate needs an explicit denominator: item-store-day, item-store-hour, shopper visit, or another unit.
A fill rate needs an explicit quantity basis, order population, and measurement point.
A retail margin percentage divides contribution by sales; markup percentage typically divides by cost.
A promotion return metric must define whether its numerator is incremental gross profit, contribution, or revenue and whether its denominator is all spending or incremental spending.
Without these definitions, two apparently comparable metrics can disagree correctly.

## 9. Industry practice and case evidence

### Documented case: Walmart and Procter & Gamble collaboration

**Documented fact:** Procter & Gamble (P&G) describes its long-running relationship with Walmart in its corporate history and discusses collaboration with retail partners in its annual reports.
Those sources support the existence of a substantial supplier-retailer relationship and P&G's stated approach to working with customers.
They do not establish a quantified causal effect for the synthetic promotion in this chapter.

**Inference:** A relationship of this scale creates opportunities to coordinate demand information, logistics, and in-store execution.
It also creates governance challenges around data quality, incentives, information access, and commercial independence.
The source does not establish that every store followed the same replenishment policy or that every collaboration produced positive incremental profit.

**Decision application:** Before proposing joint planning, specify which common facts both parties can validate.
Agree how a forecast becomes an order, which party owns the order, and what constitutes a supply commitment.
Measure the result with the same category and time boundary used at approval.
Do not cite the partnership's existence as evidence that a particular algorithm works.

### Documented case: Walmart's private-brand portfolio

**Documented fact:** Walmart publicly describes its Great Value private brand and its assortment of grocery products.
The cited company page supports the existence and positioning of that portfolio.
It is a company source, not independent proof of a specific margin, switching rate, or causal effect.

**Inference:** A retailer with both national brands and private-label alternatives must decide how to allocate space and manage substitution.
The appropriate decision depends on item-level economics, shopper response, availability, and contractual conditions.
It cannot be inferred from the mere existence of a private label that every branded item should lose a facing.

**Decision application:** Compare candidate allocations using category contribution and a credible displacement range.
Check whether changes in assortment affect store choice, product quality expectations, or supply resilience.
Report what the observed data do not identify.

### Synthetic mini-case: strong sell-in, weak sell-through

A supplier ships 20,000 cases in the week before a promotion, compared with 12,000 cases in a typical week.
Shopper sales rise only 10% during the event, and the retailer ends the period with six extra weeks of inventory for the promoted item.
Those figures are invented for teaching.

A sell-in-only report would show a large shipment increase.
It would not establish that shoppers bought the incremental cases or that the event created category growth.
First reconcile DC and store inventory and extend the analysis through the subsequent sell-through period.
Then inspect terms for forward buying and compare actual inventory with the intended service buffer.
A replenishment action may be to defer future orders rather than push more product.

### Synthetic mini-case: shelf full on average, empty at peak

A store has 48 units on the shelf at opening and 120 units in the backroom.
During a weekend promotion it sells 60 units between opening and the first scheduled refill.
The shelf can be empty before the refill even though the inventory system reports positive store stock.
The lost demand cannot be calculated from the 60 units alone.
An operational response is an earlier refill or a larger approved shopper-facing display.
Evaluate the labor and space cost against the estimated benefit, and audit execution at the relevant time.

### Synthetic mini-case: private-label space decision

A retailer is considering moving one facing from a brand to a private-label cereal.
The private-label item's contribution is USD 1.20 per unit and the branded item's contribution is USD 1.50 per unit.
A scenario forecasts 100 additional private-label units and 60 fewer branded units per week.
The direct modeled change is `100 × USD 1.20 − 60 × USD 1.50 = USD 30` per week before incremental costs.
This is not automatically a causal result or total category profit.
If 90 branded units are displaced instead, the direct modeled change is minus USD 15 per week.
The break-even displacement is 80 branded units, holding all other assumptions fixed.
A test should also measure other item substitution, stockouts, and whether customers leave the category or store.

### Evidence grading

Regulatory and standards-body sources support descriptions of their own rules and standards, subject to current applicability.
Company publications support what the company reports, not independently verified incremental effects.
Synthetic numbers teach calculations; they are not observations of the named companies.
A causal claim requires an appropriate design and evidence beyond a company relationship or a before-and-after sales chart.

### Field decision: renegotiating a promotion without hiding the trade-off

The Section 6 supplier loses USD 600 in the central scenario.
The retailer gains USD 1,700 under the same scenario.
Neither number proves what happened in a real market.
The parties still need to decide whether the proposed allocation of funding is acceptable.

Consider three candidate decisions before discussing a final agreement.
All numerical values in this subsection are synthetic.
The candidates are decision alternatives, not a recommendation to coordinate retail prices.

**Alternative A: run the original proposal.**
The supplier accepts the projected USD 600 contribution loss.
The retailer expects USD 1,700 additional category contribution.
The parties may choose this only if their authorized decision makers accept the risk.
Any strategic justification must be recorded separately from quantified contribution.
The analyst should not convert a hoped-for future benefit into a claimed present return.

**Alternative B: reduce supplier funding.**
At USD 0.25 per eligible unit, the supplier's scenario outcome improves by USD 700.
The retailer's scenario outcome falls by USD 700 if price, units, and all other costs remain fixed.
The result is a USD 100 supplier gain and a USD 1,000 retailer gain.
This fixed-demand bridge isolates the transfer, not the actual result of renegotiation.
The retailer may choose a different event design after the funding change.
That response would require a new demand and execution scenario.

**Alternative C: shorten or narrow the event.**
The buyer could consider fewer stores or a shorter event window.
The supplier could offer funding only for eligible stores and dates in the signed agreement.
Narrowing may reduce exposure but can also increase average execution cost per unit.
A smaller program is not automatically more profitable.
Store selection may improve operational feasibility but complicate causal evaluation.
Compare candidates on expected contribution, uncertainty, shelf feasibility, and settlement burden.

Record a decision log before execution.
For each alternative include the scenario version, decision owner, action, and date.
List what the retailer will actually do and what the supplier has agreed to fund.
Keep proposed terms separate from the final executed agreement.
Attach an exception path for a missing display or a late store opening.
Specify whether a store that executes for only part of the event remains eligible.
The finance owners should be able to reproduce the settlement without relying on a verbal interpretation.

### Field decision: shelf capacity versus replenishment labor

Consider a synthetic store selling an average of 12 promoted units per day.
Its shelf can hold 24 units at opening.
The average is not a forecast of the busiest two-hour interval.
Suppose the store expects 30 units of demand before the next planned shelf refill.
If the shelf starts full and no refill occurs, at most 24 units can be purchased from that shelf during the interval.
The difference of six units is unmet demand only under the stated demand scenario.
Actual lost sales could be lower if shoppers return later or substitute.
They could be higher if the demand forecast is too low.

The store could add one mid-interval refill.
That action requires available backroom stock and staff time.
It may improve availability without changing the retailer's purchase order.
Alternatively, the store could install an approved secondary display.
The display requires space, setup, maintenance, and possibly incremental promotional funding.
If neither action is feasible, reduce the promotion intensity or accept a documented risk.
Do not compare the alternatives on units alone.

For a numerical illustration, suppose a refill costs USD 8 per interval.
Suppose the event covers 20 comparable high-demand intervals.
The incremental labor cost is `20 × USD 8 = USD 160`.
If the refill preserves 150 otherwise-lost category units at USD 1.20 contribution per unit, the direct contribution gain is USD 180.
The modeled net benefit is USD 20 before any other costs.
The 150 units are an assumed counterfactual, not a count established by a stockout audit.
If only 100 units are preserved, the net result is minus USD 40.
The decision is sensitive to the lost-sales estimate and should be monitored.

Use timestamped shelf checks to understand when the problem occurs.
Compare similar stores or intervals where practical.
Record whether staff actually completed the extra refill.
Track backroom stock at the same time as shelf status.
Without execution data, a failed test cannot distinguish ineffective refilling from refilling that never occurred.

### Field decision: choosing a private-label supplier

A private-label specification is an operational contract as well as a commercial product decision.
The retailer or brand owner must agree on the product, pack, labeling, quality controls, and usable shelf life.
The manufacturer must confirm capacity, lead time, production minimums, and change-control procedures.
The buyer must compare landed economics rather than manufacturing price alone.
The comparison should include freight, expected waste, testing, and the cost of a disruption.
Any claimed value of brand differentiation or customer loyalty needs evidence.

Suppose synthetic supplier A quotes USD 1.65 per consumer unit delivered.
Supplier B quotes USD 1.58 per consumer unit delivered.
The forecast annual volume is 200,000 units.
At equal quality and service, the simple quoted saving from B is `200,000 × USD 0.07 = USD 14,000` per year.
This is a price difference, not a total-cost or risk-adjusted saving.
Suppose qualification of B requires a one-time USD 9,000 testing cost.
The first-year arithmetic saving before other differences becomes USD 5,000.
If the required testing fails, the proposed source may be unavailable regardless of price.

Capacity and lead time can change the decision.
If B's minimum order is too large for the available shelf life, expected waste may exceed the price saving.
If A is the only approved source, switching also requires a validated transition and contingency plan.
If both suppliers are approved, splitting volume may increase resilience but reduce volume discounts.
Model the actual terms rather than assuming that dual sourcing is always cheaper.
Identify who authorizes specification changes and who receives early warnings of a supply interruption.
Confirm recall contacts and the records needed to trace affected lots.
Do not assume the retailer's ownership of a label means it operates the manufacturing plant.

The retailer and supplier may have different access to cost and performance data.
Agree on minimum necessary evidence rather than requesting every confidential record.
For example, a scorecard can show on-time usable receipt, confirmed quality incidents, and lead-time variance.
Each metric needs a denominator, source, date range, and dispute process.
The scorecard should separate an accepted order from a nonbinding forecast.
Before sharing commercially sensitive information, check the agreement and applicable law.

### Field decision: evaluate an executed event

Suppose 120 stores were assigned to a synthetic promotion and 120 comparable stores were assigned to control.
Ten promoted stores did not install the display.
Five control stores accidentally displayed promotional material.
The initial business question is the effect of assigning the full program, not merely the effect among stores that complied.
An intention-to-treat comparison preserves the original assignment groups.
It estimates the effect of offering the program as actually implemented, subject to the design and analysis assumptions.

A comparison of only the 110 compliant promoted stores with all control stores can be biased.
The stores able to install a display might also have better staffing or stronger sales.
Record compliance for diagnosis, but do not automatically interpret a compliant-only comparison as causal.
If the buyer's question is instead the effect of installing a display, that is a different estimand.
Answering it requires additional assumptions or an appropriate experimental design.

The analysis should use actual store participation and actual shopper prices.
Aggregate the outcome at the randomization unit when that is needed for valid uncertainty estimates.
If nearby stores were randomized as clusters, account for clustering in the analysis.
Check for pre-event differences and missing observations without selectively removing inconvenient stores.
Report the number assigned, observed, and missing in both groups.
Inspect category contribution during the event and in a prespecified post-period.
Record simultaneous changes such as a competitor's offer or a store remodel.

For an illustrative arithmetic comparison, suppose promoted stores average 1,100 category units per store during the event.
Suppose control stores average 1,040 category units per store during the same event.
The unadjusted difference is 60 units per store.
Suppose the promoted-minus-control difference was already 20 units per store in the comparable pre-period.
The simple difference-in-differences calculation is `60 − 20 = 40 units per store`.
This is an illustrative estimator, not proof that the effect is 40 units.
Its causal interpretation requires appropriate assignment or a credible parallel-trends assumption and no important differential shocks.
An uncertainty interval and checks for spillovers are necessary before using the estimate in a decision.

Finally, reconcile the measured category change with the trade settlement.
An event may perform differently from forecast while still producing a valid contractual allowance.
Conversely, an event may produce sales lift while a claim fails an agreed eligibility condition.
The causal evaluation and the settlement audit answer different questions.
Close the loop by revising the next forecast, policy, or contract only after documenting what was learned.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Baseline includes prior promotion | Baseline weeks have abnormal price or display | The comparator already contains treatment | Incrementality is understated or misallocated | Tag and model prior events |
| Baseline ignores holiday | Similar stores rise in the same week | Seasonality is mistaken for promotion lift | Funding is credited for ordinary demand | Use calendar controls and comparable periods |
| Out-of-stock demand ignored | Zero sales while audits show empty shelf | Observed sales are censored | Future demand and lost sales are understated | Flag censored periods and validate correction |
| Phantom inventory | System shows stock while counts do not | Recording, shrink, or timing discrepancy | Replenishment is suppressed | Cycle count and investigate causes |
| Backroom stock mistaken for shelf availability | Positive on-hand with repeated shelf gaps | Refill process fails | Shopper sales are lost | Audit shelf and assign refill ownership |
| Case pack mismatch | Ordered cases convert to unexpected units | Item hierarchy or effective date is wrong | Overstock or shortage | Validate dated case-to-unit mapping |
| Shelf capacity ignored | Target inventory exceeds physical display | Order model has no space constraint | Backroom congestion or empty shelf | Model shelf and backroom separately |
| Allowance double counted | Same amount appears on invoice and claim | Off-invoice discount is also billed back | Supplier contribution is understated | Reconcile unique claim and invoice lines |
| Funding basis confused | Claims use shipped units but contract says scan units | Sell-in and sell-through timing differ | Settlement disputes and misleading profit | Specify eligible unit and evidence |
| Forward buying mistaken for lift | Shipments spike then fall; inventory rises | Retailer buys early | Supplier demand signal is distorted | Reconcile inventory and post-period sales |
| Pantry loading ignored | Post-promotion sales fall | Shoppers shift purchases earlier | Event gain is overstated | Include a post-event measurement window |
| Private-label displacement omitted | Promoted brand rises while category is flat | Shoppers switch within category | Retailer category gain is overstated | Evaluate all relevant category items |
| Store control contamination | Control-store traffic falls near promoted stores | Shoppers cross store boundaries | Estimated lift is biased | Randomize clusters or model spillovers |
| Unequal execution | Assigned stores lack displays or correct prices | Intended treatment differs from received treatment | Average result is hard to interpret | Capture assignment and actual execution |
| Nonrandom selection | High-growth stores were selected for promotion | Assignment correlates with expected demand | Naive comparison is biased | Randomize or justify comparison design |
| Returns omitted | Promotional sales have delayed returns | Revenue and eligible units are overstated | Claims and contribution are inflated | Define return treatment and finalization window |
| Timing mismatch | POS week and invoice week do not align | Different event and accounting dates | Apparent volume or margin discrepancy | Define event-time and accounting views |
| Cost boundary mismatch | Finance and account teams report different profit | Variable and fixed costs are mixed | Incorrect go/no-go decision | Approve a shared metric dictionary |
| Price assumption mistaken for reality | Actual shelf price differs by store | Planned terms were not executed uniformly | Forecast and measured economics diverge | Capture actual price and participation |
| Unapproved data sharing | Dataset contains household IDs or competitor plans | Access controls and purpose are missing | Privacy or competition exposure | Minimize data and obtain legal review |
| Allocation favors noisy metrics | Small-store availability rates swing daily | Small denominators and sampling noise | Unnecessary intervention | Report denominator and uncertainty |
| Service target misread as fill rate | A normal quantile is labeled 95% fill | Different service definitions are conflated | Overconfidence in shopper availability | State metric and validate empirically |

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | How to test or respond |
|---|---|---|---|
| Baseline 10,000 branded units | Defines the no-promotion comparison | Seasonality, distribution, or other offers change | Compare matched controls and pre-period trends |
| Promotion forecast 14,000 units | Supports the central scenario | Price response or execution differs | Compare actual price, distribution, and unit response |
| Private-label forecast 7,000 units | Represents category displacement | Shoppers switch differently | Measure related SKUs and a post-period |
| Scan allowance applies to all promoted units | Makes the bridge reproducible | Contract excludes returns or stores | Join eligible POS to approved terms |
| No pipeline inventory change | Aligns supplier sell-in with shopper sell-through | Forward buying or delayed receipt | Reconcile opening and closing inventory |
| Invoice price is USD 2.40 per unit | Computes both party bridges | Tiered prices or other deductions apply | Reconcile invoices and approved agreements |
| Variable cost is USD 1.50 per unit | Computes supplier contribution | Overtime or transport cost changes | Use current marginal cost by scenario |
| Private-label contribution is USD 1.10 per unit | Values displaced units | Cost or selling price changes | Recompute from dated item economics |
| Execution cost is USD 800 | Values the display and labor action | Actual labor, display, or fees differ | Collect invoices and store labor estimates |
| No other category effects | Keeps the worked bridge bounded | Other brands, pack sizes, or basket items change | Expand the category model and report coverage |
| Demand is independent by day | Supports the simple square-root safety-stock rule | Day-of-week or promotion demand is correlated | Inspect residual correlation or simulate blocks |
| Lead time is fixed at 3 days | Defines the protection period | Transport and receiving vary | Analyze usable-receipt lead-time distribution |
| Normal approximation is adequate | Connects the quantile to a service target | Demand is skewed, intermittent, or bounded | Compare empirical distribution and simulation |
| Inventory position is accurate | Converts target to order | Phantom stock or duplicate on-order records | Cycle count and reconcile open orders |
| Case pack is 12 units | Converts units to cases | Packaging changes over time | Validate effective item-master records |
| Store has adequate capacity | Allows the proposed order to be received and stocked | Shelf or backroom is constrained | Check planogram, refill, and capacity by store |
| Price is executed as planned | Makes scenario units and revenue coherent | Store execution or competitor response differs | Compare actual POS prices with plan |
| Parties can share required data | Supports common measurement | Contracts or privacy rules limit sharing | Use minimum necessary aggregates and approvals |

An assumption ledger is a decision tool, not a substitute for uncertainty analysis.
Where an assumption cannot be tested before approval, give an owner, an exposure limit, and a review trigger.
A useful scenario changes assumptions coherently; higher demand may also increase labor, inventory requirements, and stockout risk.

## 12. Exercises

### Exercise 1 — Map the decision chain

A beverage supplier proposes a two-week price offer in a retailer's 80 stores.
The supplier account manager says the promotion is approved because a demand forecast is available.
The retailer buyer has not approved the shelf price, and the store operations team has not confirmed display labor.

**Deliverable:** Draw the physical, information, and contractual flows and list five decision owners.
**Acceptance criteria:** Distinguish forecast, plan, target, and commitment; identify two missing approvals and one escalation deadline.
**Time:** 20 minutes.

### Exercise 2 — Reconcile party economics

Use the worked example's baseline and central promotion scenario.
The supplier proposes reducing the scan allowance from USD 0.30 to USD 0.25 per promoted unit.
Assume the retailer maintains the same shopper price and all unit forecasts and execution costs remain unchanged.

**Deliverable:** Calculate the retailer, supplier, and combined incremental contribution in USD.
**Acceptance criteria:** Show the change in allowance; apply the transfer once on each party's side; explain why the combined result is unchanged under the stated assumptions.
**Time:** 20 minutes.

### Exercise 3 — Order under a capacity constraint

Use the representative-store target of 141 units and inventory position of 85 units.
The case pack is 12 units, but the store may receive no more than 48 additional units before its next review.

**Deliverable:** Give the unconstrained and constrained order in cases and units, and the resulting inventory position.
**Acceptance criteria:** Explain the eight-unit gap between the constrained position and target, and propose one operational response without claiming the original service level remains assured.
**Time:** 15 minutes.

### Exercise 4 — Diagnose availability

At 18:00 a system reports 30 units on hand for an authorized item.
An associate observes an empty shelf and finds 30 saleable units in the backroom.
The item sold zero units from 17:00 to 18:00.

**Deliverable:** State what is known, what is unknown, and one immediate action.
**Acceptance criteria:** Do not infer the number of lost sales from the zero-sale interval; separate store inventory from shopper-facing availability.
**Time:** 10 minutes.

### Exercise 5 — Compare private-label space

Use the synthetic space mini-case with private-label contribution USD 1.20 per unit and branded contribution USD 1.50 per unit.
Forecast 100 additional private-label units and 75 fewer branded units per week.
Incremental shelf labor is USD 12 per week.

**Deliverable:** Calculate the direct weekly contribution change after labor and recommend a measurement design.
**Acceptance criteria:** Include both displaced branded contribution and labor; identify at least two unmeasured effects and a decision criterion.
**Time:** 20 minutes.

### Exercise 6 — Specify an evaluation and data contract

The retailer wants to evaluate a four-week promotion in 120 of 240 comparable stores.
Stores differ in size and location; some shoppers can travel between nearby stores.
The retailer has daily POS, planned promotion records, a store master, and weekly inventory snapshots.

**Deliverable:** Write a one-page evaluation plan and a compact data contract.
**Acceptance criteria:** Define the estimand, assignment or comparison strategy, grain, item and location keys, data-quality checks, execution measures, uncertainty, and spillover risk.
**Time:** 35 minutes.

### Exercise 7 — Write a retailer-supplier decision brief

A category buyer wants category contribution to increase.
The supplier wants a nonnegative incremental contribution.
The proposed event has the central outcomes in Section 6.
A contract allows a scan allowance but the parties have not agreed how returns affect eligible units.

**Deliverable:** Write a one-page decision brief with recommendation, alternatives, assumptions, responsibilities, and settlement controls.
**Acceptance criteria:** Report both party outcomes; flag the returns issue; identify one feasible commercial alternative and the approvals needed; avoid claiming the forecast is observed evidence.
**Time:** 30 minutes.

### Exercise 8 — Sensitivity lab

Run the Section 7 code, then change promoted branded units to 15,000 while keeping all other inputs unchanged.
Return to the original scenario and set on-order inventory to zero.

**Deliverable:** Record both changed outputs and write a short limitation statement.
**Acceptance criteria:** Reconcile the supplier break-even result and the new case order; explain why the two changes are separate scenarios and why fixed substitution may be unrealistic.
**Time:** 20 minutes.

## 13. Self-check answers

### Exercise 1

The physical flow is supplier to retailer DC to store to shopper, unless the actual network uses another route.
The information flow includes POS, stock, replenishment orders, shipment notices, and execution evidence.
The contractual flow includes price and allowance terms and the evidence needed for settlement.
Owners include the category buyer, supplier account owner, retailer replenishment planner, supplier supply planner, and store operations owner.
The forecast is an estimate; the proposed event is a plan; service is a target; an accepted order under applicable terms is a commitment.
Buyer approval of the shelf decision and store operations confirmation of labor are missing.
Escalate before the display and supply order cutoffs.

### Exercise 2

The funding reduction is `USD 0.05 × 14,000 units = USD 700`.
The retailer's incremental contribution falls from USD 1,700 to USD 1,000.
The supplier's incremental contribution rises from minus USD 600 to USD 100.
Combined incremental contribution remains USD 1,100 because the USD 700 is a transfer.
This result holds only because price, demand, costs, and execution are fixed by the exercise.
In reality, a different funding rate may change the negotiation and the resulting event design.

### Exercise 3

The unconstrained requirement is 56 units, rounded up to 5 cases or 60 units.
The capacity-constrained order is 4 cases or 48 units.
The resulting inventory position is `85 + 48 = 133 units`.
That position is 8 units below the 141-unit target.
The case-rounded unconstrained order would have placed the position 4 units above target; do not confuse that 4-unit excess with the constrained shortfall.
A split delivery, additional approved capacity, or revised promotion may address the gap.
The capacity limit means the unconstrained service target is not established.

### Exercise 4

Known facts: the system shows 30 units, the associate found 30 saleable backroom units, and the observed shelf was empty at 18:00.
The zero-sale interval does not reveal how many shoppers wanted the item or when the shelf first became empty.
The immediate action is to refill the shelf, subject to store policy, and record the observation.
Investigate whether the refill schedule and peak demand make recurrence likely.

### Exercise 5

Incremental private-label contribution is `100 × USD 1.20 = USD 120` per week.
Lost branded contribution is `75 × USD 1.50 = USD 112.50` per week.
After USD 12 of labor, the direct change is `USD 120 − USD 112.50 − USD 12 = minus USD 4.50` per week.
Unmeasured effects include other category substitution, stockouts, store choice, and changes in basket sales.
A good design uses randomized or credibly matched comparable stores, measures actual space and availability, and predefines a category contribution threshold.

### Exercise 6 rubric

A strong answer defines the effect of the assigned event on category contribution over the event and post-period.
It considers cluster randomization where neighboring stores can share shoppers.
It stores assignment separately from actual execution and uses store-item-day data with dated product mappings.
It checks missing POS, price, assortment, inventory timing, and store openings.
It specifies an uncertainty interval and a plan for contamination and incomplete execution.
A weak answer compares raw promoted and unpromoted sales without addressing store selection or available inventory.

### Exercise 7 rubric

A strong brief recommends renegotiation, redesign, or an explicitly approved exception because the central supplier contribution change is negative.
It reports the retailer's USD 1,700 gain, supplier's USD 600 loss, and combined USD 1,100 gain as scenario results.
It considers a lower allowance or another feasible event design without assuming unchanged demand is guaranteed.
It assigns buyer, supplier, operations, and finance owners and resolves returns eligibility before execution.
It states the evaluation plan and the boundary of the quantified economics.

### Exercise 8

At 15,000 promoted branded units, supplier promotion contribution is `15,000 × USD 0.60 = USD 9,000`, equal to baseline.
Retailer promotion contribution is `15,000 × USD 1.40 + 7,000 × USD 1.10 − USD 800 = USD 27,900`.
Retailer incremental contribution is USD 3,100 under the unchanged other assumptions.
With original units and zero on-order inventory, the inventory position is 65 units.
The unconstrained requirement is `141 − 65 = 76 units`, so the case-rounded order is 7 cases or 84 units.
These are separate changes; combining them would be a different scenario.
Holding private-label volume fixed while changing branded volume is a simplifying assumption, not established shopper behavior.

## 14. Interview questions

1. **How do sell-in and sell-through differ?**
   A strong answer identifies shipment or channel purchase versus shopper sales, timing, and pipeline inventory.
   A weak answer calls a shipment spike proven consumer demand.
2. **What is the counterfactual for a promotion?**
   A strong answer describes the unobserved no-promotion outcome, possible controls, and assignment assumptions.
   A weak answer treats gross promoted sales as incremental.
3. **Why might an item gain while its category loses?**
   A strong answer discusses substitution, different margins, labor, and the post-period.
   A weak answer assumes every promoted unit is a new category unit.
4. **How do you calculate trade-spend return?**
   A strong answer first defines incremental contribution, funding, timing, and eligibility.
   A weak answer divides total revenue by only one visible allowance.
5. **What is the distinction between a fill rate and on-shelf availability?**
   A strong answer defines each denominator and measurement point.
   A weak answer treats DC shipment completion as proof that a shopper found the item.
6. **How would you plan an order for a weekly-reviewed store?**
   A strong answer covers protection period, demand uncertainty, inventory position, case packs, and capacity.
   A weak answer orders exactly the forecast without considering lead time or existing stock.
7. **What happens if the inventory system says stock is present but the shelf is empty?**
   A strong answer checks backroom, timing, counts, and refill execution before diagnosing the cause.
   A weak answer immediately adds another purchase order.
8. **How do you evaluate a private-label space expansion?**
   A strong answer compares category contribution, substitution, shopper response, supply, and space constraints.
   A weak answer ranks items solely by percentage margin.
9. **How should a scan allowance be settled?**
   A strong answer identifies eligible POS units, returns, timing, evidence, claim status, and duplicate controls.
   A weak answer applies the allowance to every shipment without checking the agreement.
10. **What would make a store test misleading?**
    A strong answer discusses selection, contamination, unequal execution, stockouts, and uncertainty.
    A weak answer claims that any pre/post comparison establishes causality.
11. **How do retailer and supplier objectives differ?**
    A strong answer calculates each perspective separately and recognizes that funding transfers do not create combined profit by themselves.
    A weak answer assumes a retailer gain proves a supplier gain.
12. **What do standards and regulation contribute?**
    A strong answer distinguishes identifiers and messaging conventions from applicable legal obligations and actual execution.
    A weak answer claims a GTIN alone proves traceability compliance.

## 15. Key terms

- [Assortment](#4-definitions-and-vocabulary) — items authorized for a defined selling location.
- [Baseline](#4-definitions-and-vocabulary) — estimated outcome absent the action.
- [Bill-back](#4-definitions-and-vocabulary) — a claim for contractual reimbursement.
- [Cannibalization](#4-definitions-and-vocabulary) — displacement of sales from another item.
- [Case pack](#4-definitions-and-vocabulary) — consumer units in one orderable case.
- [Category](#4-definitions-and-vocabulary) — items managed as a business group.
- [Commitment](#4-definitions-and-vocabulary) — accepted obligation under applicable terms.
- [Counterfactual](#3-the-core-mental-model) — outcome that would occur without the action.
- [Cycle service](#step-7-translate-demand-into-an-order-plan) — a defined probability of avoiding a stockout in a replenishment cycle.
- [Distribution](#4-definitions-and-vocabulary) — coverage under a stated store or sales-weighted denominator.
- [Electronic data interchange](#identifiers-and-messaging) — standardized business document exchange.
- [Facing](#4-definitions-and-vocabulary) — one item width at the shelf front.
- [Fill rate](#4-definitions-and-vocabulary) — supplied share of a defined ordered quantity.
- [Forecast](#4-definitions-and-vocabulary) — estimated future demand or outcome.
- [Forward buying](#4-definitions-and-vocabulary) — early channel purchases under temporary terms.
- [GLN](#identifiers-and-messaging) — GS1 identifier for a party or location.
- [GTIN](#identifiers-and-messaging) — GS1 identifier for a trade item.
- [Incrementality](#4-definitions-and-vocabulary) — change attributable to an action versus a counterfactual.
- [Inventory position](#4-definitions-and-vocabulary) — on hand plus on order minus backorders under a stated convention.
- [Lead time](#4-definitions-and-vocabulary) — elapsed time to usable receipt.
- [On-shelf availability](#4-definitions-and-vocabulary) — observed shopper-facing availability using a defined denominator.
- [Order-up-to level](#4-definitions-and-vocabulary) — target inventory position before ordering.
- [Pantry loading](#4-definitions-and-vocabulary) — shifting purchases earlier for future use.
- [Planogram](#4-definitions-and-vocabulary) — shelf placement and space plan.
- [Point of sale](#4-definitions-and-vocabulary) — completed shopper transaction records.
- [Private label](#4-definitions-and-vocabulary) — retailer-owned or retailer-controlled brand.
- [Safety stock](#4-definitions-and-vocabulary) — extra stock for modeled uncertainty.
- [Scan allowance](#4-definitions-and-vocabulary) — funding tied to eligible shopper sales.
- [Sell-in](#4-definitions-and-vocabulary) — supplier sales or shipments into a channel.
- [Sell-through](#4-definitions-and-vocabulary) — onward channel sales, often shopper sales.
- [Shelf capacity](#4-definitions-and-vocabulary) — units fitting in a shopper-facing location.
- [Trade spend](#4-definitions-and-vocabulary) — supplier-funded commercial discounts or payments.

## 16. Sources

Sources below support the specific claims attributed above; they do not validate the synthetic numbers.
These public references are starting points; verify current versions and applicability before operational or legal use.

1. GS1, [GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/). Identifier allocation and change rules; industry standard, not a universal legal mandate.
2. GS1, [Global Location Number (GLN)](https://www.gs1.org/standards/id-keys/gln). Scope and use of location and party identifiers.
3. GS1, [General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications). Current technical reference for GS1 identification and data capture.
4. ASC X12, [Transaction Sets](https://x12.org/products/transaction-sets). Official catalog of X12 business transaction sets; trading partner guides determine implementation details.
5. U.S. FDA, [Food Labeling Guide](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/guidance-industry-food-labeling-guide). U.S. guidance; assess product and jurisdiction applicability.
6. U.S. FDA, [Food Traceability Final Rule](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods). Rule, scope, updates, and compliance information; consult the current page and legal text.
7. U.S. FDA, [Food Safety Modernization Act](https://www.fda.gov/food/guidance-regulation-food-and-dietary-supplements/food-safety-modernization-act-fsma). Regulatory context and official resources.
8. U.S. FTC, [The Antitrust Laws](https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws/antitrust-laws). General U.S. competition-law guidance; not a determination about a particular agreement.
9. U.S. FTC, [Price Discrimination: Robinson-Patman Violations](https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws/price-discrimination-robinson-patman-violations). Official guidance on a specific U.S. statute and its conditions.
10. Procter & Gamble, [Our History](https://us.pg.com/history/). Company account of its history; use as company evidence, not independent causal evaluation.
11. Procter & Gamble, [Annual Reports](https://www.pginvestor.com/financial-reporting/annual-reports-and-proxies/default.aspx). Company disclosures on business and customer relationships; consult the relevant year's report.
12. Walmart, [Great Value](https://www.walmart.com/cp/great-value/1078664). Retailer description of its private-brand product assortment; not evidence of a particular margin or substitution rate.
13. Paul H. Zipkin, *Foundations of Inventory Management*, McGraw-Hill, 2000. Inventory policy and uncertainty background; formulas here use the explicitly simplified assumptions stated above.
14. Scott Cunningham, *Causal Inference: The Mixtape*, Yale University Press, 2021, [online edition](https://mixtape.scunning.com/). Comparison designs, assumptions, and causal interpretation.

## 17. Links

- [Module 29 index](./_index.md).
- [Module 29 survey](../../../course/29_industry_electives.md).
- [Pharma and cold chain sibling](./02_pharma-cold-chain.md).
- [Cross-sector comparison sibling](./07_cross-sector-comparison.md).
- [Sector playbook lab sibling](./08_sector-playbook-lab.md).
- [Simulation, optimization, and production labs index](../30_simulation_optimization_production_labs/_index.md).
