# The Assortment Decision Stack

> Module 17 — Assortment, Category, and Retail Analytics  
> Topic 01 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 04 and 16  
> Version 2026.5

## 1. Why this topic exists

A retailer cannot place every available product in every store.
A category manager must choose which shopper needs to serve, which items to carry, how much shelf and inventory each receives, and where the offer should differ.
These are linked choices, not independent league tables.
A product may sell well only because it has three facings, a prominent location, and frequent promotions.
Another may look weak because it was out of stock during its best selling hours.

The operating decision is expensive because a reset affects more than the deleted item's sales.
A shopper may switch to another item, leave the category, change stores, or postpone the purchase.
A substitution may improve margin but weaken a promise to offer an accessible price.
A new item can win shelf space from an incumbent and create operational work without creating incremental demand.
Meanwhile the shelf remains finite, replenishment capacity remains finite, and a purchase commitment may outlast the evidence supporting it.

The category manager, space planner, store operator, supply planner, merchant, and supplier see different parts of this problem.
Without a common decision stack, a buyer may negotiate a launch before the shelf planner has room, or an analyst may recommend delisting a product whose availability data are unreliable.
The stack makes the sequence and handoffs explicit.
It also distinguishes a forecast from an approved plan, a target from a contractual commitment, and a supplier's claim from measured category incrementality.

This topic builds a practical category reset and a new-item gate.
The synthetic example is deliberately small enough to audit by hand.
It is not evidence that any real retailer should choose these products or thresholds.
Its purpose is to show how the business decision, the model, and the review connect.

### The decision a reader should be able to make

At a stated date, approve, pilot, defer, or reject a candidate assortment for a defined store group.
Record which items occupy which space, what demand and cost assumptions support the recommendation, who may override it, and what evidence will trigger a change.
A decision is incomplete if a plan cannot be executed or if no one owns the post-launch measurement.

## 2. Learning outcomes

After working through this topic, you can:

1. **Describe** the decisions from shopper need and category role through range, depth, shelf space, localization, acceptance, and review.
   Exercise 1 checks the sequence and the dependence between levels.
2. **Define** range, facing, shelf capacity, availability, transfer, and incremental contribution with units and denominators.
   Exercises 1 and 2 check the vocabulary and measurements.
3. **Compute** category-level incremental contribution for a new item while accounting for estimated transfers from incumbent items.
   Exercise 3 checks the arithmetic and the direction of the transfer adjustment.
4. **Choose** a feasible shelf allocation and identify the operational constraint that determines the choice.
   Exercise 4 checks width and replenishment capacity.
5. **Critique** a localization claim using store exposure, stockouts, and a suitable comparison.
   Exercise 5 asks for a decision rather than an attractive cluster label.
6. **Design** an acceptance gate and post-launch review with explicit owners and stop rules.
   Exercise 6 asks for a decision-rights record.
7. **Interpret** a sensitivity range without treating a scenario as causal evidence or a forecast as a commitment.
   Exercises 7 and 8 test uncertainty and experimental design.

## 3. The core mental model

```text
Shopper mission and category role
             |
             v
Coverage constraints and eligible products
             |
             v
Range: which distinct items, in which stores?
             |
             v
Depth: how much stock and how many facings?
             |
             v
Space: shelf width, location, capacity, labor
             |
             v
Localization: which justified store differences?
             |
             v
New-item gate: incremental value and feasibility
             |
             v
Approval -> store execution -> measurement -> review
             ^                                  |
             +----------- learn and revise -----+
```

The vertical arrangement is a decision hierarchy, not a claim that each stage happens only once.
For example, a proposed range may not fit the allocated shelf, requiring a return to the coverage decision.
A local item may be valuable in a subset of stores but too expensive to replenish independently.
A launch may be accepted as a limited test instead of a chain-wide range change.

### Six coupled choices

**Range** is the list of distinct sellable items offered at a specified store and time.
Its size is not the number of brands, and it is not the number of units on a shelf.
The assortment must cover intended needs, not merely maximize the item count.

**Depth** is the quantity available for an included item.
It can mean number of facings, shelf units behind the front row, backroom stock, or inventory cover.
These measures are related but not interchangeable.
A depth decision affects replenishment workload and the risk of an empty shelf.

**Space** assigns a scarce physical or digital exposure resource.
For a shelf, width, height, depth, position, fixtures, and case-pack handling all matter.
A feasible width sum is necessary but is not a complete planogram.

**Localization** permits a controlled difference between store groups or individual stores.
A local assortment should respond to persistent, observable differences in missions or operating conditions.
It should not merely reflect random variation in a short period of noisy sales.

**Acceptance** decides whether an item is eligible and, if so, whether it should replace an incumbent, receive more space, or enter a pilot.
A supplier's projected item sales are not the same as incremental category sales.
An acceptance decision must consider displaced contribution, execution costs, shopper coverage, and measurement quality.

**Ownership** assigns the authority to approve each trade-off and the responsibility to execute and learn.
A model can recommend a choice; it cannot approve a supplier contract or certify that a store reset happened.

### Why ordering matters

A sales-per-item ranking omits what would happen after deletion.
A space-only optimization can protect all items while failing to cover required needs.
A local plan can create a delivery frequency that a depot cannot support.
A successful pilot can still be inappropriate to roll out if its cost profile changes at scale.
Each level therefore supplies constraints and evidence to the others.

A useful unit of analysis is the **store–item–week**: one product, one store, one calendar week.
The decision unit may instead be a store cluster and reset period.
Maintain the mapping explicitly: a weekly observation does not automatically justify a permanent chain-wide decision.

## 4. Definitions and vocabulary

The definitions below apply throughout the example.
Different organizations use some of these terms differently; an approved data dictionary should settle local usage.

| Term | Operational definition and unit | Common misuse |
|---|---|---|
| Category | Managed group of related products serving defined missions; boundary is a list of items and stores. | Assuming a supplier's product family is automatically the retailer's category. |
| Category role | Intended contribution to the retailer's strategy, such as routine or destination; qualitative. | Treating a role label as a numeric demand forecast. |
| Mission | Shopper's reason for purchasing, such as a quick breakfast; described with a need and context. | Equating a demographic label with a demonstrated need. |
| Range | Distinct eligible stock keeping units offered at a store during a period; items per store. | Counting facings or warehouse stock as distinct items. |
| Stock keeping unit (SKU) | A retailer's sellable inventory item; internal item code. | Assuming one SKU always equals one universal trade-item identifier. |
| Depth | Quantity or space assigned per included item, specified as facings, shelf units, or stock units. | Using the word without naming the measure. |
| Facing | One front-facing presentation of an item on a shelf; count. | Assuming one facing has unlimited stock behind it. |
| Linear shelf width | Horizontal shelf capacity; centimeters per shelf segment. | Ignoring unusable gaps, fixture boundaries, or height. |
| Planogram | Store-specific or template visual arrangement of item positions and facings. | Calling a width-only table an executable planogram. |
| Shelf capacity | Maximum physical sellable units in an item position under specified fixture and pack dimensions. | Treating front-row facings alone as total capacity. |
| Replenishment | Movement of units from backroom or delivery to shelf; units per day or replenishment events per week. | Treating on-hand stock as proof that the shelf is full. |
| On-shelf availability | Share of eligible observation opportunities when an item is purchasable on shelf; percent. | Using inventory-record availability as a direct observation. |
| Rate of sale | Sales units divided by a specified selling exposure, such as available store-weeks. | Comparing raw units across stores with different exposure or availability. |
| Gross margin | Net revenue less merchandise cost, before specified other costs; currency or percent of revenue. | Confusing margin percentage with markup on cost. |
| Contribution | Revenue less specified variable costs attributable to the decision; currency per unit or period. | Assuming all organizations include the same costs. |
| Incrementality | Difference in category outcome between an action and its credible no-action counterfactual; currency or units per period. | Calling every sale of a new item incremental. |
| Cannibalization | Sales shifted from another item because of the change; units or currency over a specified period. | Treating all correlation between item sales as causal transfer. |
| Transfer rate | Share of one item's change attributed to substitution from another item, under a stated unit basis and counterfactual. | Mixing revenue shares and unit shares. |
| Outside-category demand | Purchases gained from, or lost to, other categories, channels, competitors, or no purchase. | Assuming every displaced purchase stays in the measured category. |
| Store cluster | Set of stores assigned a common decision template using a stated rule. | Assuming demographic similarity proves similar demand. |
| Localization | Deliberate modification of the common offer for a store group or store. | Allowing unlimited exceptions without accounting for cost. |
| Core range | Items required across a defined population of stores, subject to documented exceptions. | Assuming an item is mandatory in every possible format. |
| Local slot | Shelf and operating capacity reserved for approved local variation. | Treating local items as free additions to a full shelf. |
| Listing | Authorization to offer an item in a stated channel, store set, and period. | Assuming authorization means the item is physically available. |
| Launch | First planned period of execution and measurement for a listed item. | Defining launch as the date a supplier shipped its first case. |
| Forecast | Estimate of an uncertain future outcome under stated conditions. | Presenting it as a contractual promise. |
| Plan | Approved intended actions and resources for a stated horizon. | Assuming approval guarantees execution. |
| Target | Desired outcome used for management or evaluation. | Calling a target an unbiased forecast. |
| Commitment | Binding or formally authorized obligation, such as an approved purchase quantity. | Treating an unapproved recommendation as an order. |
| Counterfactual | Outcome that would have occurred for the same decision unit without the change. | Using the same store's prior sales without accounting for other changes. |
| Pilot | Limited deployment designed to resolve an uncertainty before scaling. | Calling an uncontrolled rollout a causal experiment. |
| Case pack | Number of retail units in a supplier or distribution case; units per case. | Assuming fractional cases can always be ordered. |
| Lead time | Elapsed time from an agreed order event to usable receipt; days or weeks. | Mixing order-to-dispatch and order-to-shelf measures. |
| Global Trade Item Number (GTIN) | GS1 identification key for a trade item, represented as a numeric identifier. | Treating the identifier itself as a guarantee of accurate dimensions. |
| Reset | Coordinated change to the shelf layout and range on an effective date. | Assuming the planned date equals completed execution. |

## 5. The mechanics, step by step

The procedure is designed for a recurring category review.
The output of each step is a decision artifact, not just a chart.
A retailer may solve several steps jointly, but should still document the branch points.

### Step 1 — Write the decision brief

**Input:** merchant objective, eligible stores, decision date, and reset window.
**Action:** specify the category boundary and business question in one sentence.
**Output:** a dated brief with decision owner and scope.

A useful brief says: "Choose the breakfast-bar range for 40 neighborhood stores for the next 13 weeks, within each store's allocated shelf and labor capacity."
It states whether the decision concerns all items or only a proposed launch.
It also states whether the baseline is the current executed shelf, the last approved planogram, or a proposed future plan.
Those baselines may differ substantially.

Record the objective in business terms.
Examples include category contribution, household penetration, price accessibility, waste reduction, or a combination.
If objectives conflict, state the priority or the decision rule for trade-offs.
Do not hide a required entry-price item inside an unexplained score.

Name the decision authority.
The category manager may own the recommendation, while the merchandising director approves material resets and procurement signs the supplier agreement.
The actual authority varies by organization and should be confirmed before analysis.

**Branch:** if no one can approve the scope or provide a reset window, obtain that decision before calculating an optimized range.
A precise calculation cannot compensate for an undefined decision.

### Step 2 — Map shopper needs and category role

**Input:** shopper research, category strategy, transaction history, complaints, and current range.
**Action:** identify the missions the category is expected to serve.
**Output:** a concise needs map and mandatory coverage rules.

For breakfast bars, plausible needs include familiar low-price products, higher-protein products, dietary alternatives, and portable multipacks.
These are hypotheses until the retailer has evidence for its customers and stores.
The business may require at least one accessible-price item or a dietary alternative.
A minimum coverage rule is a business constraint, not proof that a particular brand is indispensable.

Separate an item attribute from a shopper need.
"Chocolate" is an item attribute; "an affordable familiar snack" is a possible mission.
A mission may be served by multiple attributes and products.
Check whether an apparent gap matters to shoppers rather than merely to a supplier's portfolio.

Use customer decision-tree research when available, but do not assume a universal order of attributes.
The way shoppers choose can differ by mission and channel.
Document whether the role is routine, destination, convenience, or seasonal, and what that role implies for service and economics.

**Branch:** if the role requires broad coverage, protect key needs even if one item has a lower short-run contribution.
If the role is convenience in a small format, fewer carefully selected items may be appropriate.
Either way, quantify the cost of the constraint when possible.

### Step 3 — Establish the current offer and exposure

**Input:** item master, store authorizations, planograms, point-of-sale transactions, inventory, promotion calendar, and shelf observations.
**Action:** reconstruct what could actually be purchased by store and week.
**Output:** a validated store–item–week panel and an exception list.

Keep separate fields for authorized, ordered, delivered, on hand, on shelf, and sold.
A product with no sales may have had no authorized exposure.
A product with positive on-hand inventory may still have been absent from the shelf.
If on-shelf observations are unavailable, label the availability measure as a proxy.

Check that product identifiers map to one product definition during each effective period.
Item replacements, pack-size changes, multipacks, and unit-of-measure changes can otherwise produce false growth or false decline.
Keep historical mappings rather than overwriting them with today's item master.

Join prices, promotions, and shelf layout using effective dates.
A unit sold during a deep discount should not be interpreted as evidence of ordinary-price demand without adjustment.
Distinguish net sales after discounts and returns from gross register revenue.

**Branch:** if critical exposure or cost data are missing, choose a bounded pilot or a data-repair action rather than pretending the ranking is reliable.
An item with frequent stockouts may require a replenishment test before a delist decision.

### Step 4 — Construct eligible candidates

**Input:** current range, supplier proposals, product compliance records, and shopper-needs map.
**Action:** identify the set of current and candidate products that could reasonably be offered.
**Output:** candidate list with eligibility status and documented exclusions.

Eligibility is not the same as economic desirability.
A product may meet the shopper need but fail a mandatory safety, labeling, contractual, or operational requirement.
Conversely, an eligible product may not be worth its scarce space.
Use separate fields for compliance clearance, supplier readiness, and commercial recommendation.

Collect pack dimensions, case pack, minimum order quantity, shelf life, handling requirements, lead time, and reliable cost.
If the proposed item is a replacement for a discontinued product, retain both identities and effective dates in the decision record.
Ask whether the candidate is available to all stores, a distribution region, or a single locality.

**Branch:** if a mandatory gate is not satisfied by the decision date, do not list the item merely because its projected margin is attractive.
If a pending clearance can be obtained before ordering, document the dependency and decision deadline.

### Step 5 — Form candidate ranges

**Input:** eligible items, baseline range, missions, and hard constraints.
**Action:** construct a small number of interpretable alternatives.
**Output:** baseline, candidate ranges, and coverage comparison.

Start with the executed baseline, not a theoretical ideal.
A useful alternative may replace one incumbent with one new item, while another may keep the baseline but adjust facings.
A third may use a local slot in a subset of stores.
Each alternative must identify what leaves, what enters, and which shopper needs remain covered.

Assess redundancy using more than product similarity.
Two similar products may serve different price tiers or have distinct loyal shoppers.
Two dissimilar products may compete for the same purchase occasion.
Use actual substitution evidence where credible, and explicitly label assumptions otherwise.

Avoid comparing a candidate with no action if the status quo is not feasible.
For example, if a supplier has discontinued the incumbent, the relevant baseline may be a different available replacement.
Similarly, compare a local offer with the best feasible core offer, not with an empty shelf unless that is genuinely the alternative.

**Branch:** reject a candidate range that violates a non-negotiable coverage rule or cannot be sourced.
Return to earlier steps if all feasible options violate an approved requirement.

### Step 6 — Estimate demand and substitution

**Input:** cleaned history, candidate launch assumptions, availability, prices, promotion plans, and relevant comparison evidence.
**Action:** forecast sales under each feasible alternative and estimate where the new sales originate.
**Output:** scenario table in units per store per week with evidence and uncertainty.

For a new item, partition its forecast sales into demand transferred from named incumbents and demand incremental to the measured category.
For a deletion, estimate how much demand transfers to surviving products and how much leaves the category.
Transfers need not be symmetric: the choices available after a deletion differ from those before an addition.

Where feasible, estimate the counterfactual using a randomized store test or a defensible comparison group.
Match on pre-period trends and relevant conditions, not only on store size.
Include the same calendar period for treatment and comparison stores.
Observed differences remain suspect if promotions, shelf position, or availability change differently.

When no suitable test exists, use transparent scenarios based on comparable launches, shopper research, and merchant judgment.
Do not present a transfer assumption as a measured causal effect.
Apply the same unit basis when multiplying forecast units by transfer shares.
If pack sizes differ, convert to a consistent consumption or transaction basis before inferring physical substitution.

**Branch:** if uncertainty reverses the decision, design a pilot or buy an option for a smaller reversible launch.
If even an optimistic scenario fails a hard economic threshold, defer or reject and explain the premise.

### Step 7 — Translate sales changes into category economics

**Input:** sales scenarios, net selling prices, variable costs, transfer estimates, and incremental execution costs.
**Action:** calculate incremental contribution relative to the feasible baseline.
**Output:** a bridge from baseline to alternative, in currency per store per week and for the decision horizon.

A simple bridge is:

```text
Incremental category contribution
  = candidate-item units × candidate-item contribution per unit
  - sum of incumbent units displaced × incumbent contribution per unit
  - incremental operating cost for the same period
```

This is a partial contribution model.
State exactly which cost components are included and which are excluded.
For example, product cost and a per-unit handling charge may be included while fixed head-office salaries are not.
Do not deduct an existing fixed cost again unless the decision changes it.

For a replacement, include sales and contribution of the removed incumbent that are not transferred to the candidate.
If a discontinued incumbent would disappear anyway, calculate against the best feasible no-launch alternative instead.
Never subtract the same transferred units twice.

Separately show one-time reset expenditure and recurring weekly costs.
A one-time cost can be evaluated over a stated horizon; it should not be described as a weekly recurring expense.
Avoid simply spreading it over an arbitrary horizon without also showing the cash timing.

A positive category contribution is not automatically a chain-profit increase.
The item may shift purchases from another category, require extra distribution infrastructure, or affect store traffic.
State the measured boundary and investigate material effects outside it.

### Step 8 — Allocate facings and check shelf feasibility

**Input:** selected range, actual usable fixture dimensions, product dimensions, and replenishment rules.
**Action:** allocate facings and test both geometry and operational capacity.
**Output:** feasible space table and a request for a detailed planogram.

For a simple single shelf, multiply each item's width by its facings and sum across items.
Use usable shelf width, not the manufacturer's nominal fixture width.
Allow for dividers and constraints on item height, depth, orientation, and adjacency.
A width calculation is a screening check, not an executable shelf layout.

Estimate shelf capacity from facings and units behind each facing.
Compare forecast demand with capacity and the interval between feasible replenishment events.
Safety stock, pack handling, intraday peaks, and variation matter.
Average daily demand alone cannot guarantee service during a peak hour.

A wide product may occupy twice the space of a narrow one.
Compare incremental contribution against the relevant bottleneck, such as centimeters or labor minutes, but do not use contribution per centimeter as the only decision criterion.
Coverage constraints, availability, and the response to changing facings still matter.

**Branch:** if the shelf is too narrow, remove or resize a feasible allocation and recalculate demand and economics.
If it fits geometrically but cannot be replenished, consider more facings, a different item, a different route, or a shorter delivery interval.
Do not approve a plan whose operational capacity exists only on paper.

### Step 9 — Decide the localization rule

**Input:** comparable store demand, persistent shopper needs, format constraints, delivery cost, and evidence quality.
**Action:** compare a common range with a controlled local variant.
**Output:** cluster assignment, local slots, exceptions, and a review date.

Begin with a core range when common execution has value.
Create a local variant only if the expected benefit exceeds the added costs and uncertainty, or if a documented local requirement makes it necessary.
The unit of localization might be format, region, or individual store; justify the level.

Use multiple comparable periods when possible.
A single high-sales week can reflect a temporary promotion, a competitor stockout, or an event rather than lasting demand.
Normalize for exposure and availability before comparing stores.
Protect against small-store noise by pooling evidence or using conservative thresholds.

A cluster is an operating tool, not a causal explanation.
Validate whether the selected assortment performs better for the cluster than a common alternative.
Include costs for additional planograms, distribution complexity, minimum orders, and store labor.
A product can be a local favorite and still be economically infeasible to supply.

**Branch:** if the expected benefit is close to the complexity cost or depends on unreliable evidence, use a pilot or the common offer pending measurement.
If a store's physical constraints prevent the standard plan, document an operational exception rather than inventing a demand-based cluster.

### Step 10 — Run the new-item acceptance gate

**Input:** candidate economics, coverage, space, supply readiness, compliance, and measurement plan.
**Action:** make one of four decisions: approve, pilot, defer, or reject.
**Output:** dated gate record with owner, conditions, and review trigger.

Approval requires a feasible shelf and a viable supply route, not simply a supplier's attractive item forecast.
A pilot is appropriate when the value may be material but the uncertainty is decision-relevant.
Deferral is appropriate when a prerequisite such as verified pack dimensions is not ready.
Rejection is appropriate when the item fails a mandatory gate or is unattractive across credible scenarios.

Separate the commercial launch decision from procurement commitment.
An approved pilot can still require an authorized purchase order and contractual review.
Record who owns the test design, store selection, installation, replenishment, and results.
Include the decision date and the conditions under which the gate will be reopened.

### Step 11 — Assign decision rights and handoffs

**Input:** approved alternative, operating calendar, and local authority matrix.
**Action:** assign one accountable owner for each material decision and named execution owners.
**Output:** decision-rights table and implementation checklist.

| Decision or deliverable | Illustrative accountable owner | Contributors | Evidence or handoff |
|---|---|---|---|
| Category role and coverage | Category director | Insights, merchant | Signed brief and coverage rules |
| Range recommendation | Category manager | Analyst, buyer | Candidate comparison and assumptions |
| Shelf fit and planogram | Space planning lead | Store operations | Measured fixture and executable layout |
| Local exception | Regional merchant under policy | Store manager, supply planner | Local evidence, cost, expiry date |
| Supply feasibility | Supply planning lead | Supplier, distribution | Lead time, case pack, allocation plan |
| Regulatory and product clearance | Designated compliance owner | Supplier, legal | Applicable clearance and unresolved issues |
| Commercial approval | Authorized merchandising executive | Finance, procurement | Economics, risk, delegated authority |
| Purchase commitment | Authorized procurement owner | Finance, supplier | Approved order and contract terms |
| Store reset | Store operations lead | Space planner, field team | Completion date and exception record |
| Measurement and review | Analytics lead | Category manager, finance | Pre-agreed metrics and decision date |

These are illustrative roles, not a universal corporate responsibility matrix.
A retailer should map them to its own delegated authorities.
The accountable owner is responsible for obtaining a decision, not necessarily for performing every task.
An override should name the authorizer, reason, financial exposure, and expiry or review date.

### Step 12 — Execute, observe, and decide again

**Input:** signed decision, approved planogram, supplier readiness, and measurement design.
**Action:** implement the reset and monitor actual execution before judging the outcome.
**Output:** post-launch review that continues, changes, scales, or stops the offer.

Before the reset, confirm store list, effective date, item setup, order parameters, barcode scanning, price, and shelf labels.
During execution, record actual installation and stock availability by store.
A store that never received the product is not equivalent to an exposed store with low demand.

After launch, compare outcomes to the pre-specified counterfactual and decision horizon.
Report item units, category contribution, availability, transfer, labor, and shopper coverage.
Distinguish an execution failure from a demand failure.
If the item was absent from shelves for half the pilot, a weak observed result may not answer the intended demand question.

Record the decision and learning even when the result is inconclusive.
A credible review can extend a test, repair availability, revise the plan, or stop the launch.
Do not silently change the target after seeing the results.

### Applied decision drill A — a delist with partial transfer

Consider a different fictional category with a discontinued 500 g cereal.
The cereal sells 40 packs per store-week and earns USD 1.20 per pack of contribution.
The retailer must choose a replacement because the supplier will no longer deliver the incumbent.
The feasible no-launch baseline is therefore not the discontinued item remaining forever.

The merchant identifies an existing substitute, S, that can receive some of the displaced demand.
Assume 50% of incumbent purchases would move to S without the proposed launch.
S earns USD 0.90 per pack of contribution.
Assume the remaining incumbent purchases leave the measured category.
The no-launch baseline thus receives 20 transferred packs and USD 18.00 per store-week of contribution from those packs.
This baseline excludes unaffected items whose contribution is identical across alternatives.

The proposed replacement, R, is forecast to sell 28 packs per store-week.
It earns USD 1.10 per pack.
Under the R alternative, assume S retains 10 of the 20 packs it would otherwise have gained.
The other 10 S packs are transferred to R.
The remaining 18 R packs are incremental relative to the no-launch baseline within this category boundary.
The scenario assumes no other changes and an extra USD 2.00 per store-week of handling.

The relevant calculation is:

```text
R contribution:             28 × USD 1.10 =  USD 30.80/week
S contribution displaced:  10 × USD 0.90 = -USD  9.00/week
Extra handling:                              -USD  2.00/week
Increment against no launch:                  USD 19.80/store-week
```

Subtracting all 40 discontinued incumbent packs from R would answer a different question.
The retailer cannot choose indefinite incumbent availability, so that calculation would not compare the two feasible future choices.
Likewise, subtracting both the 10 lost S packs and the discontinued incumbent contribution would double-count parts of the same transition.

This drill illustrates why a counterfactual is a business choice, not a convenient historical average.
The discontinued item's historic performance is still useful for estimating shopper behavior.
It is not the relevant future baseline once its discontinuation is a fixed fact.

Before approval, the merchant should ask:

1. Is the discontinuation date confirmed and does remaining stock create a transitional overlap?
2. Can S actually absorb 20 extra packs per week without stockouts?
3. Does R reach the same price tier and dietary need as the discontinued product?
4. Are the 28 R packs forecast under the same shelf position and promotional support as the comparison?
5. Is the USD 2.00 handling estimate separate from the per-pack contribution?
6. Are the 18 incremental packs incremental only to cereal, or also to the retailer?

If S cannot absorb the extra demand, update both the baseline and candidate forecasts.
If R needs more shelf width, add the contribution and coverage implications of the item or facing displaced.
If the retailer can negotiate continued supply of the discontinued item, that is another feasible alternative and deserves its own scenario.

### Applied decision drill B — an operationally constrained local item

A fictional chain considers a regional sauce for 12 of its 60 stores.
The expected category contribution increase before local handling is USD 8.50 per selected store-week.
The offer requires a separate depot handling run costing USD 72 per week in total.
It also requires one extra store task costing USD 1.00 per selected store-week.
Assume the depot cost is fixed for this 12-store decision and that no other costs change.

The weekly economics are:

```text
Gross contribution gain: 12 stores × USD 8.50 = USD 102.00/week
Store task cost:          12 stores × USD 1.00 = USD  12.00/week
Depot handling cost:                               USD  72.00/week
Net incremental contribution:                      USD  18.00/week
```

The item has a positive modeled result of USD 18.00 per week across the selected stores.
That is USD 1.50 per selected store-week on average.
The result is sensitive to the fixed delivery cost and to whether all 12 stores actually receive the item.
It says nothing about a one-time setup cost, spoilage, or effect beyond the category because those have not been supplied.

If only eight stores qualify at the same per-store contribution and store task cost, the calculation becomes:

```text
8 × (USD 8.50 - USD 1.00) - USD 72.00 = -USD 12.00/week
```

The item is not automatically attractive just because each store has positive gross contribution.
Its operating route changes the answer.
The minimum number of stores required to cover the stated weekly costs is ten:

```text
USD 72.00 / USD 7.50 per store-week = 9.6 stores
```

Because a store cannot be fractional, at least ten stores are needed under these assumptions.
At ten stores, the result is USD 3.00 per week before any omitted setup cost.
That small modeled surplus may not justify irreversible investment or uncertainty.

The localization gate should therefore ask the supply planner for actual route capacity and the merchant for an evidence-based store list.
If a regular route can carry the sauce without a separate USD 72.00 run, recalculate rather than retaining the old cost.
If shelf life forces waste, include expected incremental waste at the same boundary.
If only eight stores have evidence of demand, do not add two weak-fit stores merely to make the arithmetic pass.
Instead compare a different delivery arrangement, a smaller pilot, a different item, or no local offer.

The regional merchant owns the local recommendation under the chain's policy.
The supply planning lead confirms the route and minimum delivery rules.
The authorized commercial owner accepts the financial exposure.
Store operations confirms the task can be completed.
Analytics checks whether the forecast difference persists after controlling for promotion and availability.

### Applied decision drill C — depth is not the range

A store currently sells one brand of bottled drink in two sizes.
The 500 ml size has two facings and the 1 L size has one facing.
A proposal increases the 500 ml size to three facings while removing the 1 L size.
The store still carries the same brand, but the range falls from two SKUs to one SKU.
Depth rises for one remaining SKU, while variety by package size falls.

Assume the 1 L size sells 20 bottles per week at USD 0.80 contribution per bottle.
The store forecasts that eight of those purchases transfer to the 500 ml SKU.
The 500 ml SKU earns USD 0.45 per bottle.
Assume one 1 L purchase transfers to one 500 ml bottle in this deliberately simplified customer-purchase scenario.
That assumption does not preserve volume: 1 L and 500 ml are different quantities.
The remaining 12 purchases are assumed lost from the measured category.

Ignoring all other effects, the contribution change is:

```text
1 L contribution removed: 20 × USD 0.80 = -USD 16.00/week
500 ml transfer gained:     8 × USD 0.45 = +USD  3.60/week
Net before space benefit:                    -USD 12.40/store-week
```

The new facing might improve availability for existing 500 ml shoppers.
That benefit must be estimated separately and must not include the eight transferred purchases a second time.
The freed width might support another product; the opportunity should be compared with the best feasible use of that width.
Neither the same brand name nor the higher facing count proves that size choice is preserved.

If the category promises a family-size option, the range may fail a coverage rule even if another use of space is profitable.
If the 1 L item was frequently unavailable, observed sales may understate its underlying demand.
If the size is discontinued, use a feasible future replacement as the baseline rather than assuming it stays available.
The space planner and merchant should approve the revised layout together.

### Practical review: how to make a forecast decision-ready

For each candidate, record four separate values rather than a single unlabelled number.
The **forecast** is the model's expected or scenario demand under documented conditions.
The **plan** is the offer and execution the organization has decided to pursue.
The **target** is a desired result for management evaluation.
The **commitment** is the formally authorized quantity or contractual obligation.

For example:

| Field | Illustrative statement | Owner or control |
|---|---|---|
| Forecast | 24 units per exposed store-week in the base scenario. | Analyst records assumptions and uncertainty. |
| Plan | Pilot a two-facing layout in a selected store group. | Merchant and operations approve scope. |
| Target | Meet the pre-agreed category contribution and availability criteria. | Commercial owner signs criteria before results. |
| Commitment | Place an approved case-pack order after supplier and budget checks. | Procurement authority issues the order. |

A target can exceed the forecast when management wants a stretch goal.
It must not be substituted for the forecast in procurement or economic calculations.
An approved plan can be conditional on clearance.
It must not be treated as an unconditional order.
An order can exceed expected sales to cover minimum case quantities or lead time.
It must then explicitly account for inventory, shelf life, and waste risk.

### Practical review: select the comparison boundary

An item's unit sales describe what the item did.
They do not describe the category outcome without the item.
Select the boundary that matches the business question before gathering the result.

| Decision question | Minimum outcome boundary | Additional concern |
|---|---|---|
| Does P attract sales? | P units per exposed store-week. | Exposure and promotion can explain differences. |
| Does P grow its category? | Category units or contribution against a counterfactual. | Different item margins and pack sizes matter. |
| Does P grow store contribution? | Relevant store contribution across potentially affected categories. | Cross-category shifts and store-level costs matter. |
| Is a local route worthwhile? | Local store benefit less incremental route and handling cost. | Fixed costs and store count matter. |
| Should a pilot scale? | Incremental benefit at expected rollout conditions. | Supply, execution, and cost may change at scale. |

The largest boundary is not automatically the best measurement design.
A very broad outcome may be noisy and hard to attribute.
A very narrow outcome may omit material substitution.
Use the smallest boundary that captures economically material effects, and disclose what remains outside it.

### Practical review: decision thresholds and uncertainty

A business threshold must specify the metric, unit, horizon, and authority.
For example, “positive contribution” is incomplete if the calculation excludes the reset cost but the decision requires a one-season payoff.
Likewise, “95% availability” is incomplete without an observation method and eligible denominator.

Set guardrails before seeing results.
Examples include mandatory product clearance, an agreed minimum availability, and a maximum one-time investment.
Set an economic rule that names the contribution definition and comparison period.
Specify whether the rule applies to the point estimate, a conservative scenario, or another decision statistic.
The appropriate threshold depends on the organization's risk appetite and evidence quality.

Distinguish uncertainty in the size of an effect from uncertainty in execution.
If a pilot shows low sales because the product was never delivered, repair the execution problem before concluding that shoppers rejected it.
If delivery was reliable but category contribution is uncertain because the sample is small, decide whether more evidence is worth its cost.
If the item violates a non-negotiable requirement, a favorable forecast does not remove that gate.

### Practical review: change control after approval

After approval, the business may learn that pack dimensions, cost, supplier capacity, or the store list changed.
Each material change can invalidate part of the approved decision.
Use a change log that names the original assumption, revised value, effect, and approving owner.

| Change | First question | Likely owner |
|---|---|---|
| Pack width rises | Does the allocation still fit? | Space planner. |
| Cost rises | Does the contribution bridge still pass? | Merchant and finance. |
| Lead time doubles | Can stores maintain availability without excessive stock? | Supply planner. |
| Store list shrinks | Does the local route still cover its fixed cost? | Regional merchant and supply planner. |
| Launch date moves | Does the comparison still cover the intended seasonal window? | Analytics and category manager. |
| Claim or label changes | Is additional product clearance required? | Compliance owner. |

Do not assume every change requires a full chain-wide review.
Define delegated tolerances and escalation rules in advance.
A small permitted cost change may be handled by the merchant; a failed mandatory gate must be escalated regardless of projected sales.
The actual thresholds and authorities are organizational policy, not universal facts.

## 6. Worked example

### Case and decision

A fictional retailer, Harbor Market, is reviewing breakfast bars in 40 neighborhood stores.
Each participating store has one 120 cm usable shelf segment for the four items considered below.
The horizon is 13 weeks after a scheduled reset.
All amounts are US dollars (USD), per store per week unless otherwise stated.
The case is synthetic; the figures are assumptions, not a report of a real retailer's performance.

The category manager wants to test a new high-protein bar, P.
The current three-item range has an entry-price bar A, a familiar family bar B, and an oat bar C.
A and B must remain to cover approved price and familiar-choice needs.
C remains in the candidate range because it covers an oat-based need.
The retailer is not assuming that C is universally required.

The decision is whether to add P using spare width and by moving one facing from B to P.
No item is deleted in the base case.
The category director approves the pilot if the physical, commercial, and product-clearance gates pass.

### Baseline data

| Item | Shopper role | Width per facing | Baseline facings | Baseline sales | Contribution per unit |
|---|---|---:|---:|---:|---:|
| A | Entry price | 8 cm | 4 | 56 units/week | USD 0.40 |
| B | Familiar family | 10 cm | 4 | 45 units/week | USD 0.65 |
| C | Oat-based | 9 cm | 3 | 28 units/week | USD 0.75 |
| P | Proposed protein | 11 cm | 0 | 0 units/week | USD 1.10 |

The contribution figures equal net revenue less merchandise cost and the agreed per-unit variable handling cost.
They exclude the separately listed weekly operating cost and the one-time reset cost.
The same contribution definition applies to all items.
Baseline sales are assumed representative of a normal week at the planned prices and availability.
That assumption requires validation before a real decision.

The current width is:

```text
A: 4 facings × 8 cm  = 32 cm
B: 4 facings × 10 cm = 40 cm
C: 3 facings × 9 cm  = 27 cm
Total                  99 cm
Available             120 cm
Unused                 21 cm
```

The baseline category contribution is:

```text
A: 56 units × USD 0.40/unit = USD 22.40/week
B: 45 units × USD 0.65/unit = USD 29.25/week
C: 28 units × USD 0.75/unit = USD 21.00/week
Total                         USD 72.65/week
```

This is a forecast baseline for the decision, not a guarantee that those sales will occur.
It is not an approved purchase quantity or a contractual commitment.

### Candidate physical allocation

The candidate retains four A facings and three C facings, reduces B from four to three facings, and gives P two facings.

| Item | Proposed facings | Width used | Change in facings |
|---|---:|---:|---:|
| A | 4 | 32 cm | 0 |
| B | 3 | 30 cm | -1 |
| C | 3 | 27 cm | 0 |
| P | 2 | 22 cm | +2 |
| Total | 12 | 111 cm | +1 |

The proposed layout uses 111 cm of the 120 cm segment, leaving 9 cm for the stated allowance.
It passes the simple width screen.
The space planner must still verify fixture height, depth, dividers, placement, adjacency, and actual product dimensions.

Assume B has six units behind each facing, for 18 shelf units at three facings.
Assume a forecast of 38 B units per week after the launch.
Its average is approximately 5.43 units per day in a seven-day trading week.
If the store can replenish daily, this arithmetic suggests average capacity is adequate, but it does not prove peak-hour availability.
A field check of replenishment frequency and demand peaks is still required.

### Demand and transfer assumptions

The merchant forecasts 24 P units per store per week in the proposed layout.
The working assumption is that 25% of P units transfer from B, 25% from C, and 50% are incremental to this category.
The 50% category increment can include purchases shifted from elsewhere in the store; it is not necessarily incremental to the whole retailer.
These are scenarios, not measured causal transfer rates.

| Source of P units | Share of P forecast | Units per store-week |
|---|---:|---:|
| Transferred from B | 25% | 6 |
| Transferred from C | 25% | 6 |
| Incremental to measured category | 50% | 12 |
| Total P forecast | 100% | 24 |

A remains at 56 units per week in the scenario.
B also loses one unit per week from the reduced facing, independently of the six units transferred to P.
This separate space effect is an assumption; it must not already be included in the transfer estimate.
C loses six units per week through transfer.
The forecast alternative is therefore A 56, B 38, C 22, and P 24 units per store-week.

Category units increase from 129 to 140 per store-week.
The 11-unit increase is 12 incremental P units less the assumed one-unit B space effect.
Different pack sizes and consumption quantities could make unit totals misleading; this example assumes comparable retail units for the bridge.

### Contribution bridge

Assume incremental operating cost of USD 3.20 per store-week for extra replenishment and distribution complexity.
This cost is incremental to the baseline and is not included in the item contribution values.
The calculation is:

```text
P contribution:                    24 × USD 1.10 = +USD 26.40/week
B transfer loss:                    6 × USD 0.65 =  -USD 3.90/week
C transfer loss:                    6 × USD 0.75 =  -USD 4.50/week
B separate facing loss:             1 × USD 0.65 =  -USD 0.65/week
Incremental operating cost:                         -USD 3.20/week
Incremental category contribution:                  +USD 14.15/week
```

The alternative's category contribution after the extra recurring cost is USD 86.80 per store-week.
That equals the baseline USD 72.65 plus USD 14.15.
The result does not include effects outside the category, tax, financing, or unlisted fixed costs.
It is an expected scenario under the stated assumptions, not a causal result.

A separate one-time reset costs USD 42 per store.
At the same weekly scenario, the 13-week incremental contribution is:

```text
13 weeks × USD 14.15/week - USD 42 = USD 141.95 per store
40 stores × USD 141.95/store = USD 5,678 for the 13-week horizon
```

This simple horizon calculation ignores discounting and assumes all 40 stores execute on time and sustain the same weekly effect.
Actual orders should reflect case packs, lead times, and uncertainty rather than mechanically multiplying forecast sales.
The USD 5,678 is neither a guaranteed benefit nor a supplier commitment.

### Sensitivity and decision

Suppose the same 24 P units each week instead come 45% from B, 45% from C, and 10% from outside the measured category.
The estimated transfer is 10.8 units from each incumbent, with the same separate one-unit B facing loss and USD 3.20 weekly operating cost.

```text
USD 26.40 - (10.8 × USD 0.65) - (10.8 × USD 0.75)
          - USD 0.65 - USD 3.20 = USD 7.43 per store-week
13 × USD 7.43 - USD 42 = USD 54.59 per store over 13 weeks
```

Fractional units represent an average forecast, not an instruction to stock fractions of a unit.
The more cannibalistic case remains positive in this narrow contribution model, but the cushion is much smaller.
Other errors, such as lower P demand or higher operating cost, could change the decision.

The recommendation is a controlled pilot, not a 40-store rollout commitment.
Randomly assign eligible stores to the proposed reset or a comparable baseline where operationally feasible.
The merchant must first verify clearance, fixture fit, supply, and measurement readiness.
Set a review after sufficient exposed weeks; interpret any result using actual execution and a pre-specified analysis.
The category director decides scale, modification, extension, or stop after the review.

### Decision record

| Field | Case value |
|---|---|
| Forecast | P sells 24 units per exposed store-week under the candidate layout. |
| Plan | Pilot the candidate layout in selected eligible stores after approval. |
| Target | Positive incremental category contribution and acceptable availability, with thresholds agreed before launch. |
| Commitment | Only the purchase order authorized after pilot approval; quantity is not specified by this example. |
| Primary owner | Category director for the commercial gate. |
| Execution owner | Store operations lead for the reset. |
| Measurement owner | Analytics lead for the pre-agreed comparison. |
| Stop or revise | A compliance failure stops listing; otherwise review against agreed economics and availability rules. |

## 7. Code, query, or calculation

The following Python 3 example reproduces the economic bridge and width screen using only the standard library.
It compares the base and high-transfer scenarios.
It does not estimate demand, infer causality, optimize a planogram, or issue an order.
The input values are the case assumptions and units above.

```python
from decimal import Decimal

D = Decimal

# Counts are facings; widths are centimeters per facing.
width_cm = {"A": D("8"), "B": D("10"), "C": D("9"), "P": D("11")}
baseline_facings = {"A": 4, "B": 4, "C": 3, "P": 0}
proposed_facings = {"A": 4, "B": 3, "C": 3, "P": 2}
usable_width_cm = D("120")

# Net contribution is USD per retail unit, excluding the separate weekly cost.
unit_contribution = {"B": D("0.65"), "C": D("0.75"), "P": D("1.10")}
p_units_per_week = D("24")
b_facing_loss_units = D("1")
weekly_extra_cost = D("3.20")
reset_cost = D("42")
horizon_weeks = 13
stores = 40


def width_used(facings):
    return sum(width_cm[item] * count for item, count in facings.items())


def incremental_contribution(b_share, c_share):
    # Shares describe P units transferred from B and C, respectively.
    assert D("0") <= b_share <= D("1")
    assert D("0") <= c_share <= D("1")
    assert b_share + c_share <= D("1")
    b_transfer = p_units_per_week * b_share
    c_transfer = p_units_per_week * c_share
    return (
        p_units_per_week * unit_contribution["P"]
        - b_transfer * unit_contribution["B"]
        - c_transfer * unit_contribution["C"]
        - b_facing_loss_units * unit_contribution["B"]
        - weekly_extra_cost
    )


base_width = width_used(baseline_facings)
new_width = width_used(proposed_facings)
print(f"Baseline width: {base_width} cm")
print(f"Proposed width: {new_width} cm; spare: {usable_width_cm - new_width} cm")
assert new_width <= usable_width_cm

for name, b_share, c_share in (
    ("Base", D("0.25"), D("0.25")),
    ("High transfer", D("0.45"), D("0.45")),
):
    weekly = incremental_contribution(b_share, c_share)
    per_store = weekly * horizon_weeks - reset_cost
    total = per_store * stores
    print(f"{name}: USD {weekly:.2f}/store-week; "
          f"USD {per_store:.2f}/store over {horizon_weeks} weeks; "
          f"USD {total:.2f} for {stores} stores")
```

Expected output:

```text
Baseline width: 99 cm
Proposed width: 111 cm; spare: 9 cm
Base: USD 14.15/store-week; USD 141.95/store over 13 weeks; USD 5678.00 for 40 stores
High transfer: USD 7.43/store-week; USD 54.59/store over 13 weeks; USD 2183.60 for 40 stores
```

`Decimal` keeps these illustrative currency calculations exact at the displayed precision.
In a real system, input validation must also cover effective dates, missing costs, unit conversions, and store-specific constraints.
The `assert` statements here are teaching checks, not a substitute for production validation or a full feasibility model.
The width screen intentionally omits shelf height, depth, placement, and case-handling constraints.

## 8. Data, standards, and regulatory context

### Minimum decision dataset

| Data domain | Grain and example fields | Decision use | Important control |
|---|---|---|---|
| Item master | SKU and effective date; brand, size, dimensions, GTIN | Eligibility and shelf fit | Retain historical versions and units. |
| Store master | Store and effective date; format, region, fixture | Store scope and feasible localization | Avoid using today's format for an earlier period. |
| Authorization | Store–SKU–date; listed and delisted dates | Eligible exposure | Listing is not proof of shelf presence. |
| Transactions | Store–SKU–transaction or day; units, net sales, returns | Rate of sale and contribution | Resolve returns and multipack units consistently. |
| Price and promotion | Store–SKU–date; regular price, net price, event | Comparable demand | Record overlapping or local promotions. |
| Inventory | Store–SKU–time; on hand, receipts, adjustments | Availability proxy and supply readiness | Book inventory may differ from shelf stock. |
| Shelf observations | Store–SKU–timestamp; present, facing, position | Actual exposure and execution | Sampling times must represent selling hours. |
| Planogram | Store or cluster–SKU–effective date; facings, position | Physical design and reset | Separate intended from observed execution. |
| Supply | SKU–source–effective date; case pack, lead time, minimum order | Feasible procurement | Confirm actual route and restrictions. |
| Cost | SKU–supplier–effective date; cost, allowances, variable handling | Incremental economics | Avoid counting the same allowance twice. |
| Compliance | SKU–market–effective date; applicable clearance and owner | Mandatory gate | Version and retain evidence of approval. |
| Experiment | Store–assignment–period; treatment and comparison | Counterfactual estimation | Preserve original assignment and deviations. |

Create a data dictionary with time zone, currency, unit, effective-date convention, owner, and known gaps for each field.
Choose whether a week begins on Monday, Sunday, or a retail-calendar boundary, and apply the same calendar to treatment and comparison.
A 53-week retail year can otherwise distort year-over-year comparisons.

### Product identification and exchange

GS1's GTIN standard provides a way to identify trade items in supply-chain and retail processes.
A GTIN can support matching a scanned product to a master record, but it does not itself verify pack dimensions, nutritional claims, or stock presence.
A retailer's internal SKU may map differently across time or channels.
Pack changes can require a different trade-item identity under applicable GS1 rules; consult the current rules rather than inventing a universal mapping.

Electronic data interchange (EDI) is a family of structured business-document exchanges, not a single assurance of data quality.
A purchase order or advance shipping notice can help reconcile expected receipts with actual receipts.
It does not prove the item was placed on a shelf.
Trading partners must agree on message standards, versions, identifiers, and error handling.

### Regulatory boundaries

For US food products, the Food and Drug Administration (FDA) describes food-labeling requirements and related guidance.
Which requirements apply depends on the product, claim, jurisdiction, and circumstances.
A category analyst should route claims, allergens, and labeling questions to the designated compliance owner rather than infer approval from a supplier spreadsheet.
Some foods fall under other agencies' jurisdiction.
This educational discussion is not legal advice.

Competition law and supplier arrangements also depend on jurisdiction and facts.
A supplier proposal or category-management recommendation is not a license to share competitors' confidential information or coordinate prices.
Use the organization's legal and information-sharing controls.
The sources below are official guidance and standards references; verify current requirements at the decision date.

### Evidence quality

A shelf photograph can establish that a product was visible at the photographed time, not that it was available all week.
A point-of-sale record establishes a recorded transaction, not the demand of shoppers who found an empty shelf.
A supplier forecast is a commercial input, not independent evidence of incrementality.
A randomized comparison can strengthen attribution, but only if assignment, exposure, and analysis are handled properly.

## 9. Industry practice and case evidence

### What public evidence does and does not establish

Assortment decisions are often made with a mixture of shopper research, transaction data, category strategy, supplier input, and operational constraints.
The precise decision rights and economic definitions vary by retailer.
Published examples frequently describe a program or intention rather than reveal all data, costs, and counterfactual outcomes.
The following cases are used to illustrate decision questions, not to validate the numerical Harbor Market scenario.

### Walmart and store-specific assortment

Walmart's public annual reports describe its retail formats, merchandising, supply-chain operations, and customer-facing strategy.
They establish that Walmart operates multiple formats and markets; they do not establish an optimal local assortment for any particular store.
See the company's investor annual reports in the sources.

**Inference:** different store formats and customer contexts make a single physical assortment template potentially restrictive.
That inference does not prove that a particular local item adds profit, nor that a specific clustering algorithm is used.
A practitioner would still need store-level exposure, costs, and a controlled comparison before recommending local expansion.

### Tesco and supplier collaboration

The UK Competition and Markets Authority maintains information on the Groceries Supply Code of Practice and the Groceries Code Adjudicator enforces the code for designated retailers.
These sources document a governance context for certain retailer–supplier dealings in the United Kingdom.
They are not evidence that every assortment proposal is prohibited or that every retailer is subject to the same code.
Check the applicable designation and current rules with legal counsel.

**Inference:** even when a supplier contributes useful category evidence, a retailer needs clear ownership of the recommendation, confidential information, and trading decisions.
This protects the quality of the business decision as well as supporting the organization's compliance process.
A supplier's forecast and a retailer's independent incremental analysis should be separately identified.

### Evidence from assortment research

Kök, Fisher, and Vaidyanathan survey assortment planning models and discuss the interaction of demand, substitution, and inventory.
Their review provides a research foundation for treating assortment as a constrained portfolio rather than a ranking of stand-alone products.
It does not supply a universal transfer rate for a particular retailer.

Broniarczyk, Hoyer, and McAlister examine consumers' perceptions of assortment.
Their work illustrates why an observed item count is not identical to perceived choice.
A retailer should not infer that every reduction improves the shopper experience or that every addition causes overload.
The relevant shopper population, presentation, and mission matter.

### Practical interpretation

Use public cases to frame mechanisms and questions.
Use local evidence to decide store scope, thresholds, and expected value.
A credible business case distinguishes documented public facts, internal measurements, assumptions, and inferences.
If any of these cannot be separated, the decision memo is not ready for approval.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Ranking by raw sales | Low-ranked items have frequent stockouts. | Exposure differs by item. | A viable item is delisted. | Measure availability and compare available exposure. |
| Counting new sales as incremental | Proposal reports only new SKU revenue. | Incumbent purchases transfer. | Category benefit is overstated. | Build an explicit transfer and contribution bridge. |
| Double-counting transfer | Facing loss already appears in the demand forecast. | The same lost units are subtracted twice. | A useful change appears unattractive. | Define independent effects and reconcile the unit bridge. |
| Ignoring outside-category effects | Category grows while a neighboring category declines. | Demand shifts across the measurement boundary. | Retailer profit may not grow. | Expand the outcome boundary where material and measurable. |
| Treating listing as execution | Authorized stores have no receipts or shelf evidence. | Setup and physical execution differ. | Apparent weak demand is misdiagnosed. | Record actual reset, receipts, and on-shelf exposure. |
| Assuming width proves fit | Width fits but packages exceed shelf height. | Geometry has several dimensions. | Store cannot execute the layout. | Validate fixture, orientation, dimensions, and planogram. |
| Ignoring labor | Stores refill the candidate several times daily. | Shelf capacity is too small for demand. | Stockouts or excess labor erase value. | Test replenishment against peaks and labor capacity. |
| Overlocalizing | Many one-store exceptions have no owner or expiry. | Each exception increases operational complexity. | Higher cost and inconsistent execution. | Set an evidence threshold, cost the exception, and review it. |
| Misreading a promotion | Local sales peak only during a discount. | Price and promotion confound demand. | A permanent local slot is assigned on temporary evidence. | Align comparison periods and promotion exposure. |
| No common counterfactual | A launch is compared only to its own pre-period. | Seasonality and other changes affect outcomes. | Launch impact is overstated or understated. | Use randomized or defensible concurrent comparison stores. |
| Ignoring mandatory coverage | A low-selling entry-price item disappears. | Pure contribution objective omits shopper promise. | Price access or loyalty suffers. | Make coverage explicit and quantify its opportunity cost. |
| Confusing forecast and commitment | A supplier treats a scenario as a purchase order. | Approval and ordering authorities are unclear. | Excess inventory or contractual exposure. | Separate forecast, plan, target, and authorized order. |
| Incorrect item mapping | Pack-size changes appear as sudden unit growth. | Old and new identifiers or units are merged incorrectly. | Demand and substitution estimates are distorted. | Version mappings and normalize units. |
| Unowned review | Launch continues past its review date. | No one is accountable for a scale-or-stop decision. | Weak economics persist. | Assign a review owner, date, evidence, and authority. |
| Supplier claim treated as fact | Transfer assumptions lack a source or range. | Commercial incentives and evidence are conflated. | Gate approval rests on optimistic assumptions. | Record provenance and test independent scenarios. |
| Selective pilot analysis | Only high-performing stores are analyzed. | Exclusions are chosen after outcomes are known. | Biased estimate drives rollout. | Pre-specify assignment, analysis population, and deviations. |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test or respond |
|---|---|---|---|
| Baseline weeks are representative. | They anchor the no-change forecast. | Seasonal events, competitor changes, or unusual weather. | Compare multiple comparable periods and concurrent controls. |
| P sells 24 units per store-week. | The bridge needs a demand scenario. | Low awareness, price resistance, or supply interruptions. | Pilot with actual exposure and examine uncertainty. |
| Transfer shares sum to at most 100% of P units. | This simple unit bridge partitions P's sales. | Pack-size differences or complex multi-unit substitution. | Convert to a consistent basis and model richer behavior if needed. |
| B's separate facing loss is not already in transfer. | Prevents double-counting. | Both estimates come from the same bundled scenario. | Reconcile the counterfactual and individual effects. |
| Contribution per unit stays constant. | Enables a linear bridge. | Tiered rebates, promotional discounts, or different handling costs. | Use effective net prices and cost schedules by scenario. |
| The weekly operating cost is incremental. | It should be deducted only if caused by the change. | Labor is fixed or the same cost is included per unit. | Trace the cost to a resource and remove duplication. |
| All eligible stores fit the same width. | Permits a common shelf calculation. | Fixture dimensions vary by store. | Measure each format and generate exceptions. |
| Daily replenishment is feasible. | Supports the depth check. | Labor shortages, delivery delays, or intraday peaks. | Observe replenishment and shelf availability at peak times. |
| Category boundary is sufficient for the decision. | Makes a tractable contribution calculation. | Material substitution across categories or channels. | Extend the outcome boundary and report uncertainty. |
| The one-time reset costs USD 42 per store. | Needed for horizon economics. | Labor rates or fixture requirements vary. | Obtain store-specific cost estimates and actuals. |
| Effects are stable across the horizon. | Enables multiplication by 13 weeks. | Trial fades, repeat purchase changes, or seasonality. | Track weekly trajectories and model ramp-up or decay. |
| All 40 stores execute on time. | Supports the illustrative scale calculation. | Delayed shipments, reset failures, or missing labels. | Track actual exposed weeks; avoid claiming unexecuted benefit. |
| Units are comparable for the transfer bridge. | Allows shares to be expressed in retail units. | A new pack contains a different number or size of portions. | Re-express demand in comparable consumption or transaction units. |
| Pilot assignment is respected. | Supports a clean treatment comparison. | Contamination, unplanned promotions, or noncompliance. | Preserve assignments and analyze deviations explicitly. |

The case is a decision aid, not a guarantee or a demand model.
Its numeric thresholds, cost definitions, and authority matrix must be replaced with the retailer's own approved values.

## 12. Exercises

### Exercise 1 — Map the stack

**Time:** 15 minutes.
A merchant wants to add six snacks to every store because each has a higher item margin than the current average.
The shelf is full, and several stores have a different fixture.

**Deliverable:** a six-stage decision map and a list of three missing inputs.
**Acceptance criteria:** distinguish range, depth, space, localization, acceptance, and ownership; identify at least one feedback loop and distinguish item margin from category incrementality.

### Exercise 2 — Exposure and rate of sale

**Time:** 15 minutes.
An item sells 84 units over 12 calendar store-weeks.
Reliable observations indicate it was available for 9 equivalent store-weeks.
A comparable item sells 90 units over 12 available store-weeks.

**Deliverable:** both available-exposure rates and a short interpretation.
**Acceptance criteria:** show units per available store-week; explain why the first item's lost demand is not established by this arithmetic alone.

### Exercise 3 — New-item contribution

**Time:** 20 minutes.
A candidate sells a forecast 30 units per store-week at USD 0.90 contribution per unit.
Of those units, 40% transfer from an incumbent earning USD 0.50 per unit, 20% from an incumbent earning USD 0.80 per unit, and the remainder is incremental to the category.
Extra operating cost is USD 4 per store-week.
There is no additional facing effect.

**Deliverable:** a unit-origin table and weekly contribution bridge.
**Acceptance criteria:** calculate transferred units, avoid deducting incremental units, and report the result in USD per store-week.

### Exercise 4 — Shelf and depth

**Time:** 20 minutes.
A shelf has 96 cm usable width.
Three items require widths of 8 cm, 10 cm, and 12 cm per facing.
The proposed facings are 3, 3, and 4 respectively.
The third item holds five units behind each facing and forecasts 42 units per week.
Stores replenish it twice per week, with evenly spaced visits assumed.

**Deliverable:** a width feasibility calculation and a depth assessment.
**Acceptance criteria:** calculate width, excess or spare space, and average demand between visits; explain why average demand does not establish service at peak times.

### Exercise 5 — Critique localization

**Time:** 20 minutes.
A local product sold 60 units in Store X and 30 units in Store Y last month.
X had a four-week promotion and full availability; Y had two weeks of shelf stock and no promotion.
A manager proposes assigning X and Y to separate demand clusters using those totals.

**Deliverable:** a short recommendation and three additional data requirements.
**Acceptance criteria:** identify the unequal exposure and promotion confounding, avoid inferring a persistent preference, and propose a comparable test or holdout.

### Exercise 6 — Design the acceptance gate

**Time:** 25 minutes.
A supplier offers a high-margin chilled item with a three-day lead time and a minimum order of 200 units per depot delivery.
The candidate covers a dietary need but requires a new temperature-controlled handling process.

**Deliverable:** a gate record with four decision options, owners, evidence, and review trigger.
**Acceptance criteria:** include compliance, shelf and cold-chain feasibility, incremental economics, purchase authority, and actual execution; distinguish deferral from rejection and pilot from full approval.

### Exercise 7 — Sensitivity and horizon

**Time:** 20 minutes.
Return to Exercise 3.
Suppose the first incumbent transfer share rises to 70% and the second remains 20%.
The candidate sales, costs, and all other assumptions are unchanged.
A one-time reset costs USD 35 per store and the horizon is 10 weeks.

**Deliverable:** revised weekly and horizon contribution, plus one decision caveat.
**Acceptance criteria:** use the specified transfer shares, deduct reset once, and explain why a positive scenario is not a measured causal effect.

### Exercise 8 — Design a pilot

**Time:** 30 minutes.
Harbor Market can run the candidate reset in 20 of 40 comparable stores; the other 20 can retain the baseline.
The supplier requests an immediate chain-wide commitment based on first-week P sales.

**Deliverable:** one-page pilot plan with assignment, outcomes, exposure checks, review date, and scale/stop decision rights.
**Acceptance criteria:** include concurrent comparison, category contribution and availability, pre-specified handling of deviations, and a distinction between first-week item sales and causal category effect.
Do not invent a statistical power guarantee without variability data.

## 13. Self-check answers

### Exercise 1

A strong sequence begins with the shopper need and category role.
It selects eligible products and a range, allocates depth and space, determines justified local variants, applies the new-item gate, and assigns approval and execution ownership.
The shelf fit may send the team back to range selection.
Required inputs include actual usable fixture dimensions, incumbent displacement assumptions, and supply or replenishment capacity.
A high item margin alone says nothing about lost incumbent contribution or incremental operating cost.

### Exercise 2

The first item's available-exposure rate is 84 / 9 = 9.33 units per available store-week.
The second item's rate is 90 / 12 = 7.50 units per available store-week.
These are descriptive averages.
The missing three equivalent weeks for the first item could have different demand, promotions, or seasonal conditions.
One cannot simply multiply 9.33 by three and call the result observed lost sales.

### Exercise 3

The first incumbent contributes 30 × 40% = 12 transferred units per week.
The second contributes 30 × 20% = 6 transferred units per week.
The remaining 12 units per week are incremental to the defined category under the assumption.

```text
Candidate: 30 × USD 0.90 =  USD 27.00/week
First loss: 12 × USD 0.50 = -USD 6.00/week
Second loss: 6 × USD 0.80 = -USD 4.80/week
Extra operating cost:       -USD 4.00/week
Incremental contribution:    USD 12.20/store-week
```

The result is conditional on the transfer and cost assumptions.
It is not a measured experiment outcome.

### Exercise 4

Width is 3 × 8 + 3 × 10 + 4 × 12 = 102 cm.
The proposal exceeds the 96 cm shelf by 6 cm and is not feasible as stated.
The third item holds 4 × 5 = 20 units on the shelf.
At 42 units per week with two evenly spaced replenishments, average demand between visits is 21 units.
That exceeds the stated shelf capacity even before considering demand variation.
Revise the layout or replenishment plan and check resulting coverage and economics.
The average does not describe the peak interval or guarantee on-shelf availability.

### Exercise 5

Do not assign demand clusters from these totals.
Store X's promotion and Store Y's limited availability change their opportunities to sell.
Obtain exposure by day, promotion and price history, and comparable nonpromotion observations across enough stores and periods.
A concurrent pilot or holdout can test whether a local allocation improves category outcomes after execution costs.
A well-argued answer also considers fixture and delivery differences.

### Exercise 6 rubric

A complete gate identifies the authorized commercial approver, compliance owner, supply or cold-chain owner, store execution owner, and measurement owner.
It checks the 200-unit depot minimum against likely demand, shelf life, and delivery schedule.
It distinguishes **defer** for missing clearance or cold-chain evidence from **reject** when a mandatory condition cannot be met or economics are unacceptable.
A **pilot** is a bounded test with an approved order and review; **approve** must identify the scope and authority.
The review uses category contribution, waste, availability, temperature integrity, and actual execution, not only candidate margin.

### Exercise 7

The first transfer is 30 × 70% = 21 units per store-week.
The second transfer remains 30 × 20% = 6 units per store-week.
Ten percent, or three units per store-week, is assumed incremental to the category.

```text
USD 27.00 - (21 × USD 0.50) - (6 × USD 0.80) - USD 4.00
= USD 7.70 per store-week
10 × USD 7.70 - USD 35 = USD 42.00 per store over 10 weeks
```

The scenario is positive under the specified assumptions.
Actual transfer, execution, and operating cost can differ, and the calculation alone does not establish causality.

### Exercise 8 rubric

Randomize eligible stores when feasible and retain the original assignments for analysis.
Use a concurrent comparison for the same calendar weeks and document any differences in promotions, price, and supply.
Pre-specify the review horizon, outcome definitions, exclusion rules, and treatment of stores that did not execute.
Report category contribution, item units, availability, and deviations, with appropriate uncertainty.
The decision maker should be authorized before results are known and should decide whether to scale, change, extend, or stop.
First-week candidate sales are neither a mature demand estimate nor a category-level causal effect.

## 14. Interview questions

1. **How do range and depth differ?**
   A strong answer separates distinct items from facings or units per item and names the unit.
   A weak answer treats an extra facing as a new SKU.
2. **Why is a high-margin launch not automatically attractive?**
   A strong answer accounts for displaced contribution, operating cost, and the relevant counterfactual.
   A weak answer compares only candidate and incumbent margin percentages.
3. **How would you distinguish listing from availability?**
   A strong answer separates authorization, receipt, on-hand inventory, and shelf observation.
   A weak answer treats an item-master flag as proof that shoppers could buy it.
4. **What makes a space allocation executable?**
   A strong answer discusses fixture dimensions, pack orientation, location, capacity, and replenishment.
   A weak answer checks only the sum of nominal product widths.
5. **When would you localize a range?**
   A strong answer weighs persistent demand differences against incremental operating complexity and tests a common alternative.
   A weak answer assumes a demographic label is a proven demand segment.
6. **What is a transfer rate and what is its denominator?**
   A strong answer states which units and counterfactual the share refers to and distinguishes estimates from evidence.
   A weak answer mixes shares of revenue, units, and customers without conversion.
7. **What would you do with a low-selling, frequently out-of-stock SKU?**
   A strong answer investigates exposure and replenishment before using raw sales as a delist criterion.
   A weak answer extrapolates observed demand into stockout periods without qualification.
8. **How do you evaluate a new item in a pilot?**
   A strong answer uses a concurrent comparison, measures actual exposure and category contribution, and pre-specifies decisions.
   A weak answer treats first-week SKU sales as causal incrementality.
9. **Who should approve the final range?**
   A strong answer identifies an authorized commercial owner and distinct compliance, supply, execution, and measurement owners.
   A weak answer delegates a commercial commitment to an analyst's model.
10. **How do you handle uncertainty that reverses the recommendation?**
    A strong answer presents scenarios, checks decision sensitivity, and considers a reversible test.
    A weak answer selects the most favorable supplier forecast without disclosure.

## 15. Key terms

Each term points to the fuller operational definition in [Definitions and vocabulary](#4-definitions-and-vocabulary).

- **Assortment:** the items offered in a defined store and period; see range.
- **Availability:** whether a shopper could buy an item at the observed time.
- **Cannibalization:** demand shifted from another item because of a change.
- **Case pack:** retail units in a supplier or distribution case.
- **Category:** a managed group of products serving defined missions.
- **Category role:** the intended strategic contribution of the category.
- **Commitment:** an authorized obligation, distinct from a forecast.
- **Contribution:** revenue less the specified variable costs.
- **Core range:** items required across a defined store population.
- **Counterfactual:** the outcome that would occur without the action.
- **Depth:** item quantity or facings, expressed with its measure.
- **Exposure:** the opportunity for an item to be bought or observed.
- **Facing:** one front-facing item presentation on a shelf.
- **Forecast:** an uncertain estimate under stated conditions.
- **GTIN:** the GS1 identifier for a trade item.
- **Incrementality:** the outcome difference against the no-action counterfactual.
- **Lead time:** elapsed time between specified order and receipt events.
- **Listing:** authorization to offer an item in a defined scope.
- **Local slot:** space and operating capacity reserved for local variation.
- **Localization:** a justified difference in assortment by store or group.
- **Mission:** the shopper's need or reason for buying.
- **On-shelf availability:** the share of eligible observations with a purchasable item on shelf.
- **Pilot:** limited deployment intended to resolve decision uncertainty.
- **Plan:** approved intended actions and resources.
- **Planogram:** a visual arrangement of items and facings.
- **Range:** distinct items offered at a stated location and time.
- **Rate of sale:** units sold per specified unit of exposure.
- **Reset:** coordinated execution of a changed shelf layout.
- **SKU:** a retailer's sellable inventory item identifier.
- **Store cluster:** stores assigned a common decision template by a stated rule.
- **Target:** a desired outcome, not necessarily an unbiased forecast.
- **Transfer rate:** the stated share of demand attributed to substitution.

## 16. Sources

The following references are checkable sources for concepts, standards, and the public context described above.
The fictional numerical example is not drawn from any of them.
Links and rules should be checked for current applicability before an operational decision.

1. Kök, A. G., Fisher, M. L., and Vaidyanathan, R. “Assortment Planning: Review of Literature and Industry Practice.” In *Retail Supply Chain Management*, Springer, 2008. [Publisher record](https://link.springer.com/chapter/10.1007/978-0-387-78903-3_4). Research review of assortment, substitution, and operational trade-offs.
2. Broniarczyk, S. M., Hoyer, W. D., and McAlister, L. “Consumers' Perceptions of the Assortment Offered in a Grocery Category: The Impact of Item Reduction.” *Journal of Marketing Research*, 35(2), 1998, 166–176. [DOI: 10.1177/002224379803500202](https://doi.org/10.1177/002224379803500202). Shopper perception of assortment reduction.
3. GS1. [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin). Identification-key overview; it does not certify item attributes or availability.
4. GS1. [GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/). Rules for assigning trade-item identities; consult the current edition and market context.
5. US Food and Drug Administration. [Food Labeling & Nutrition](https://www.fda.gov/food/food-labeling-nutrition). Regulatory resources for applicable US food labeling; not a substitute for product-specific review.
6. UK Groceries Code Adjudicator. [Groceries Code Adjudicator](https://www.gov.uk/government/organisations/groceries-code-adjudicator). Official information on the code and its enforcement for designated retailers.
7. UK Competition and Markets Authority. [Groceries market investigation: final order](https://www.gov.uk/cma-cases/groceries-market-investigation). Official background to the Groceries Supply Code of Practice and associated order.
8. Walmart Inc. [Annual reports and proxies](https://stock.walmart.com/financials/annual-reports-and-proxies/default.aspx). Company-reported context on formats and operations, not independent proof of assortment impact.

## 17. Links

- [Module 17 index and topic table](_index.md).
- [Module 17 survey and its lab sections](../../../course/17_assortment_category_retail_analytics.md).
- [Sibling topic: variety versus choice overload](_index.md#3-topics) — listed as planned in the module index.
- [Sibling topic: store clustering and localization](_index.md#3-topics) — listed as planned in the module index.
- [Sibling topic: new-item acceptance and cannibalization](_index.md#3-topics) — listed as planned in the module index.
- [Sibling topic: space elasticity and planograms](_index.md#3-topics) — listed as planned in the module index.
