# Assortment for Dark Stores

> Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics  
> Topic 3 of 10  
> Estimated study time: 4 hours  
> Prerequisites: inventory fundamentals and [promise tiers and dark-store design](01_promise-tiers-dark-stores.md)  
> Version 2026.5

## 1. Why this topic exists

A dark store can promise a short delivery window only for products that are actually pickable at the local site.
Unlike a conventional supermarket, its selling area is not designed for customer browsing.
The operator still needs safe storage, receiving, access to stock and enough pick capacity.
Every additional product competes for a finite amount of usable space.
A new item may earn sales, complete baskets and serve a need that alternatives do not.
It may also consume the last chilled location, increase picking time and expire unsold.
The decision is not simply how many products to list.
It is which products to carry, how much stock to hold and where to locate that stock.

The category manager decides which customer needs the range should cover.
The site manager confirms whether the resulting layout can be operated safely.
The inventory planner converts the selected range into replenishment and stock targets.
Finance checks the incremental contribution and cash requirements.
The product team controls how availability and substitutions appear to the customer.
Food safety and legal teams own restrictions appropriate to the goods and jurisdiction.
No single margin-per-unit ranking can replace those decisions.

Consider a small store serving a dense residential area.
Its evening orders contain breakfast items, baby products, ready-to-eat meals and emergency top-ups.
A slow-selling infant formula might be critical to a small group even if its contribution per shelf location is low.
A second flavor of a popular snack might earn margin but substitute almost entirely for an existing flavor.
The correct comparison is the incremental value of each range change, not each item's observed sales in isolation.

Breadth means carrying more distinct options.
Depth means carrying more units or more selling capacity for an option already offered.
Breadth can improve need coverage and reduce abandonment.
Depth can reduce stockouts and protect an existing demand stream.
Both use capacity, and both can be wrong if stock records are inaccurate.
Treat this topic as a decision process rather than a fixed universal assortment formula.

### Decision brief

| Question | Decision output | Accountable owner |
|---|---|---|
| Which needs must be met? | Required customer missions and protected categories | Category lead |
| Which items and stock levels fit? | Approved store-specific range and inventory policy | Category and inventory leads |
| Can the range be safely operated? | Validated layout and location capacities | Site manager |
| Does the change create value? | Incremental business case and downside | Finance partner |
| What happens when it fails? | Pause, rollback and escalation rules | Named launch owner |

The accountable owner signs the decision; other teams supply evidence and constraints.
In a small business one person may hold several roles, but all of the checks remain necessary.

## 2. Learning outcomes

After working through this unit, you can:

1. Distinguish breadth, depth, availability and displayed selection using a specific store and time window.
2. Translate physical space, temperature zones and customer needs into a feasible assortment problem.
3. Calculate incremental weekly contribution after substitution, waste and controllable operating costs.
4. Compare an extra item with extra stock of an existing item using the same scarce capacity.
5. Explain why observed sales under stockouts are not an unbiased measure of demand.
6. Design a store-level test with customer, safety, operational and financial guardrails.
7. Communicate a recommendation, its assumptions, owner, trigger and rollback rule.

Exercises 1–8 respectively test these skills, with the final exercise combining them.
The worked case and runnable calculation show one transparent way to make the trade-off.
The case inputs are synthetic; they are not estimates for a named operator.

## 3. The core mental model

```text
Customer missions and store catchment
             |
             v
Candidate products ----> customer need and substitution map
             |                         |
             v                         v
Demand when available ----> incremental orders, units and value
             |                         |
             v                         v
Stock, replenishment, expiry ----> contribution scenarios
             |                         |
             +------------+------------+
                          v
      Space by zone + handling + safety + policy constraints
                          |
                          v
            Feasible breadth/depth alternatives
                          |
                          v
        Test, measure, revise or roll back by store
```

A customer mission is the job that motivates an order, such as preparing breakfast tonight for tomorrow morning.
One mission may need several complementary items.
A single product can support multiple missions, and two products can be partial substitutes.
The value of a product therefore depends on the rest of the range.

A capacity constraint is physical, not just a financial penalty.
If the cold room is full, an attractive additional chilled item cannot be added by paying a small model penalty.
A proposed exception requires a physically validated change, such as removing another item or adding approved refrigeration.
The same principle applies to segregated allergens, restricted goods and required access routes.

Supply and demand interact with space.
More stock of an item protects against demand before the next replenishment.
It may also increase spoilage when remaining shelf life is short.
A broader range can increase the number of replenishment lines, break cases into smaller lots and complicate picking.
Stock depth therefore has a marginal value that changes with the inventory already held.
The first unit of a scarce essential may prevent a lost basket; the twentieth may sit until expiry.

Observed orders reflect what customers could see and buy.
When an item is unavailable, an order may switch to an alternative, omit the item, use another merchant or disappear entirely.
A ranking based only on historical sales can punish items that were often unavailable.
Before comparing options, reconstruct exposure and make the missing-demand assumptions explicit.

A sound recommendation has two boundaries.
The outer boundary is feasibility: the proposal must fit each hard constraint at an appropriate peak, not only on average.
Inside that boundary the operator compares expected value, customer outcomes and uncertainty.
Policy requirements, such as minimum coverage for a defined customer need, can be constraints even when the narrow financial optimum would remove the item.

### Competing uses of one location

Suppose a chilled location can hold either two extra milk units or the minimum facing of a plant-based drink.
Extra milk units could prevent a milk stockout during the morning rush.
The plant-based drink may make a dairy-free mission possible but also take some sales from existing milk.
The right comparison includes both alternatives' incremental customer and financial outcomes over the same week.
It also checks whether the plant-based product remains viable given its minimum order quantity and shelf life.
No comparison is complete until it includes the option to leave the location empty as a buffer.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or grain |
|---|---|---|---|
| Assortment | Set of products eligible for sale at a site in a time period | Equating a global catalog with a store range | Store–SKU–interval |
| Stock keeping unit (SKU) | Operationally distinct item and pack configuration | Treating different pack sizes as interchangeable identifiers | SKU |
| Breadth | Number or diversity of distinct choices carried | Counting unavailable listings as usable choice | SKUs or need groups per store |
| Depth | Stock or capacity allocated to a selected item | Assuming more variants means more stock depth | Units or locations per SKU |
| Range | Another term for an assortment in a stated scope | Omitting the store or date | Store and effective interval |
| Facing | Front-accessible presentation or pick position for an item | Assuming all backstock is directly pickable | Positions per SKU |
| On-hand | Recorded physical quantity at a moment | Treating a ledger balance as a confirmed pickable quantity | Units per SKU–site–time |
| Available to promise | Quantity eligible to be offered after reservations and rules | Using physical stock without subtracting allocated orders | Units at a decision timestamp |
| In-stock rate | Fraction of defined exposure time or opportunities with stock | Averaging item rates without saying how weighted | Percent of item-hours or views |
| Stockout | Period when demand cannot be filled from the required inventory | Counting a delisted item as an accidental stockout | Minutes or exposure events |
| Lost demand | Desired demand that was not fulfilled | Equating lost demand to observed zero sales | Units or baskets per interval |
| Substitution | Customer or operator choice of an alternative when desired item is unavailable | Assuming every substitute sale is incremental | Conditional probability or units |
| Cannibalization | Demand shifted from another offered item after a new choice is introduced | Treating all overlap as harmful regardless of customer benefit | Units or contribution per period |
| Complement | Item whose value or purchase probability rises with another item | Concluding causality from a frequent co-purchase | Basket or product pair |
| Mission coverage | Extent to which defined needs can be met with available products | Treating all SKUs as equally important | Covered needs or weighted opportunities |
| Incremental contribution | Change in revenue less relevant variable costs caused by a choice | Adding gross sales without displaced sales or costs | Currency per store–period |
| Gross margin | Sales revenue less cost of goods sold under a defined accounting convention | Treating it as operating profit | Currency or percent |
| Waste | Unsellable inventory and its associated cost or quantity | Counting all markdowns as physical disposal | Units and currency per period |
| Markdown | Reduction from normal selling price to encourage sale | Assuming a markdown is free | Currency per unit |
| Shelf life | Period under specified conditions in which product remains suitable or saleable | Using nominal life instead of remaining life at receipt | Days or hours |
| FEFO | First-expiry-first-out picking or rotation | Confusing receipt sequence with expiry sequence | Lot-level rule |
| Lead time | Time between replenishment trigger and usable receipt | Ignoring receiving and putaway | Hours or days |
| Order cycle | Time between ordering opportunities | Confusing it with supplier transit time | Hours or days |
| Minimum order quantity (MOQ) | Smallest quantity a supplier accepts under agreed terms | Assuming it is a physical shelf capacity | Units or cases per order |
| Case pack | Number of units supplied together in a case | Assuming a case can always be split upstream | Units per case |
| Cube | Physical volume occupied by stock or its packaging | Using product volume alone as usable location capacity | Cubic meters |
| Effective capacity | Usable storage after access, clearances and operating allowances | Using building floor area as sellable space | Locations, units or volume |
| Zone | Physically or logically distinct storage area with relevant conditions | Pooling ambient and refrigerated capacity | Location class |
| Slotting | Assignment of an item to a specific storage or pick location | Equating range selection with the final physical layout | SKU–location–interval |
| Replenishment | Movement or ordering that restores stock | Treating an approved range as a replenishment plan | Units per cycle |
| Pick time | Labor time to locate, collect and confirm an item | Counting travel once per unit without checking batching | Seconds per line or order |
| Basket | Set of lines or units in one customer order | Treating a line as an order | Order ID |
| Attach rate | Fraction of defined baskets or opportunities containing an item | Ignoring that the set of opportunities changed | Percent of eligible baskets |
| Service level | Defined probability or fraction of demand served | Switching between cycle service and unit fill rate | Percent with stated denominator |
| Fill rate | Fraction of ordered units fulfilled under specified rules | Calling it an order-completion rate | Fulfilled units / ordered units |
| Shadow value | Local improvement in optimal objective from one more unit of a binding resource, where defined | Claiming a stable price for a discrete nonconvex decision | Currency per capacity unit–period |
| Planogram | Specified spatial arrangement of items | Assuming a plan is identical to the actual layout | Versioned SKU–location map |
| Forecast | Estimate of future demand under stated conditions | Calling a sales target a forecast | Units per SKU–period |
| Plan | Chosen quantities and timing to execute | Treating it as the same object as the forecast | Units and dates |
| Target | Desired performance threshold | Calling a target a promise to every customer | Percent or currency per period |
| Commitment | Externally or internally accepted obligation | Assuming a forecast guarantees delivery | Quantity and deadline |

## 5. The mechanics, step by step

### Step 1 — Define the decision and the comparison

Input: store, decision owner, review period, current assortment and operating baseline.
Output: an explicit choice among feasible future policies.
Write the question as a change relative to the current site rather than as a request for a ranking.
For example: which two changes should this store make for the next four weeks without expanding refrigeration?
State whether the decision concerns a permanent range, a seasonal trial or a single promotional period.
A trial with a two-week supplier lead time requires a different launch plan from a permanent reset.

Choose the evaluation period before comparing money.
A weekly period works well when replenishment, demand and costs can be expressed on the same basis.
If a setup cost is incurred once, keep it separate and compare it to the expected stream of weekly benefits.
Do not add a one-time installation charge to a weekly contribution without specifying how it is annualized or recovered.

Define the counterfactual.
If a product is not listed, another product may occupy its location and receive some of its sales.
If it is listed but frequently unavailable, the practical alternative differs from reliable availability.
The baseline should specify both the current product list and its replenishment policy.

### Step 2 — Define customer missions and coverage

Input: customer research, local order history, search logs, service tickets and demographic context where appropriate.
Output: a needs map with evidence quality and accountable policy choices.
Start with missions such as breakfast, a quick meal, a party top-up, infant care and an urgent household replacement.
These are examples, not universal categories.
Ask whether a mission requires a product type, a dietary attribute, a pack size or several complementary lines.

Group similar items by the need they serve, not only by supplier category.
A customer who cannot consume dairy may not regard ordinary milk as a suitable substitute.
A customer needing a precise infant formula may not accept a different formula merely because both are in the same category.
Use explicit eligibility rules and customer-consented substitution preferences where available.
A substitute must be acceptable, available and safe; price similarity alone is insufficient.

Record gaps as hypotheses rather than facts when data are incomplete.
Search with no resulting purchase can mean unmet need, a typo, price rejection or curiosity.
A small number of complaints may reveal a serious coverage issue but does not alone establish total demand.
Interview evidence, opportunity counts and experiments can improve the estimate.

Assign a policy owner to any protected coverage rule.
For example, the business might require at least one plant-based milk choice at each eligible store.
Write down how that rule was selected, which customer need it addresses and when it will be reviewed.
A protection rule is not automatically profitable; its business rationale may include access, trust or strategic positioning.
Make its measured cost visible instead of hiding it in a score.

### Step 3 — Build an effective capacity map

Input: measured locations, storage rules, handling clearances and operating schedule.
Output: hard capacity by zone and an auditable mapping from product to required space.
Begin with physical dimensions and the safe operating layout.
Subtract aisles, egress, fire equipment, workstation clearance and other non-storage areas.
Do not count the same refrigerated location as both forward-pick space and reserve space.
A single site may have ambient, chilled, frozen, secure and other separately constrained zones.

Estimate the location footprint at the stock peak, not simply the mean inventory.
An incoming supplier case can temporarily require more room than the individual units displayed in the pick face.
A replenishment quantity that fits after half a day of sales may still fail at the moment of receipt.
If receiving and reserve stock use separate areas, model their capacities separately.
If the layout cannot accommodate concurrent deliveries, change the delivery schedule or the selected range.

Use integer locations when an item occupies an indivisible bin.
Volume alone misses unusable gaps, weight limits, shelf geometry and access needs.
A large liquid pack and a small carton can have equal nominal volume but very different pick-face requirements.
Validate a proposed slot with the site operator and, when necessary, a physical trial.

Include fixed location requirements and stock-dependent requirements separately when possible.
A product can require a label and minimum pick face even at low stock.
Extra depth may use an additional bin or a larger footprint only beyond a threshold.
Record which items may safely share a location; never assume arbitrary mixing.

### Step 4 — Clean the candidate product universe

Input: product master, current catalog, supplier terms, site restrictions and customer requirements.
Output: an eligible list with versioned product identifiers and reasons for exclusions.
Distinguish a global catalog item from a product that this particular site may actually sell.
Deduplicate pack variants only when their identifiers and dimensions prove they are the same sellable item.
Preserve meaningful differences such as allergen formulation, pack size, storage condition and legal restrictions.

Exclude discontinued or noncompliant candidates before optimization.
For uncertain supplier availability, create a supply-risk scenario instead of assuming perfect service.
A supplier agreement can change price, case pack or order frequency during the evaluation period.
Date each term and keep the contract source outside the model when it is confidential.

List operational prerequisites for a new product.
These can include master-data setup, label printing, slot assignment, receipt testing, training and app imagery.
Do not schedule the customer-facing launch before those prerequisites are complete.
An item is not genuinely available if the app lists it but the site cannot receive or pick it.

### Step 5 — Reconstruct demand under availability

Input: item availability intervals, catalog exposure, orders, searches, prices and promotion history.
Output: a forecast distribution of customer demand under each proposed availability policy.
Start by joining data at store, SKU and time interval.
Use effective timestamps for assortment, price and availability rather than a current snapshot joined to historical orders.
Align timestamps to the store's time zone and record daylight-saving transitions.

Separate intentional delisting, temporary stockout, store closure and missing telemetry.
Each has a different implication for the counterfactual.
A product cannot sell while hidden, but that does not prove a lack of desire for it.
If availability is unknown, label the interval unknown rather than automatically calling it in stock.

A simple first approximation uses sales per observed in-stock hour multiplied by planned selling hours.
It is only defensible if in-stock hours are representative of all hours after conditioning on known drivers.
This can fail badly because a popular item often sells out precisely during the highest-demand hours.
Estimate separately by relevant time slot and adjust for promotions, price and day of week where evidence supports it.
For uncertain items, present a range rather than a single precise forecast.

Account for lost baskets, not just lost units.
A customer denied an essential may switch the entire order to another merchant.
Conversely, a customer who substitutes within the store may preserve most of the basket value.
These outcomes cannot be inferred reliably from the item-level sales ledger alone.
Use app interaction data, stated preferences and controlled tests, respecting privacy rules.

### Step 6 — Estimate incrementality and relationships

Input: demand scenarios, price, variable cost, substitution evidence and basket relationships.
Output: value of each range change against the same baseline.
For each proposed new item estimate gross customer demand and sales diverted from existing items.
Estimate additional basket contribution only where the causal mechanism and measurement support it.
Subtract supplier cost, incremental picking, packaging, waste, payment and other avoidable costs that change with the decision.
Avoid subtracting fixed rent again if it is unchanged by the option.

A simple item-level expression is:

```text
Incremental contribution per week
  = contribution from new fulfilled units
  - contribution displaced from existing units
  + contribution from genuinely incremental companion purchases
  - incremental waste, handling and other avoidable costs.
```

All terms must use the same currency, store and week.
Do not add revenue from the new item and again count the full value of the same order as basket uplift.
If companion purchases are already included in a measured total-order treatment effect, do not add them a second time.
Use gross margin only as an intermediate term when other variable costs change.

Relationships make the full problem nonlinear.
Adding a second near-identical beverage may divert sales from the first.
Adding tortillas may make an existing filling more useful.
A practical model may approximate these interactions with groups or pair terms, then validate the approximation in a test.
Document any interaction omitted for lack of evidence.

### Step 7 — Choose stock depth candidates

Input: selected items, demand distributions, replenishment schedule, lead-time variability and shelf life.
Output: a small set of feasible depth policies for each SKU.
For an item, compare a minimum operating quantity, the current quantity and one or more deeper alternatives.
Include safety stock only after specifying the service metric and demand uncertainty.
For perishables, compare the cost of expected shortage with the cost of expiry and markdown.

In a simple periodic-review approximation, expected demand over the protection period includes lead time and time until the next order opportunity.
If demand and lead time are uncertain or correlated, use a joint scenario or a suitable simulation rather than a single average.
An item with a long case-based replenishment cycle may need more depth than its average daily sales suggest.
A very short shelf life may make that depth uneconomic or impossible.

A continuous stock target is not necessarily orderable.
Round to valid packs, account for current stock and already ordered units, and respect remaining-life rules.
A deeper option should state the peak location footprint caused by receiving and replenishment.
If supplier minimums imply unavoidable waste, show it as a cost and question whether the supplier can offer a different pack.

### Step 8 — Formulate hard constraints

Input: candidates, capacity map, needs policy and feasible depth options.
Output: a set of explicit constraints that a proposed range must satisfy.
A basic binary decision is whether SKU `i` is selected at a given depth `d`.
A separate choice per depth prevents a model from selecting multiple mutually exclusive depths for the same SKU.
For each zone, total peak space consumed cannot exceed its effective capacity.

Other constraints can include minimum mission coverage, maximum weight per shelf, legal eligibility and a realistic handling-time budget.
Model a required product as mandatory only when a policy owner has approved the requirement.
Express it as a hard rule rather than a giant arbitrary benefit.
Use soft penalties only for outcomes that are genuinely negotiable.

Check feasibility before maximizing contribution.
If mandatory products exceed chilled capacity, a solver cannot create a valid physical plan.
The decision owner must relax a policy, change the layout, reduce quantities where safe or obtain approved capacity.
The analyst should present the conflicting constraints rather than silently violating one.

### Step 9 — Compare feasible alternatives

Input: constraint set, contribution estimates and uncertainty scenarios.
Output: a ranked shortlist with clear trade-offs and a feasible baseline.
Small problems can be enumerated exactly.
Larger problems can use integer programming, constraint programming or a transparent heuristic followed by feasibility checks.
The optimization method does not repair biased inputs or unmodeled safety constraints.

Use marginal contribution per scarce location as a screening measure, not as proof of the optimum.
Several resources can bind at once, so a high value per total location may be poor value per chilled location.
Discrete bins and minimum coverage rules can make greedy rankings miss the best combination.
The worked example deliberately enumerates complete feasible combinations.

Record whether each solution is an estimate, an approved plan or a contractual commitment.
A model's preferred option is a recommendation until owners approve it.
A target such as 95% availability is a goal, not evidence that the operating system can deliver it.

### Step 10 — Stress the recommendation

Input: low, central and high assumptions for demand, substitution, waste, labor and supply reliability.
Output: a decision table and named triggers for reconsideration.
Vary inputs jointly where there is a credible common cause.
For example, a late delivery can simultaneously reduce availability and increase emergency replenishment cost.
Do not combine every worst case and label it a likely forecast; call it a stress scenario.

Find the break-even contribution for a candidate and compare it with the plausible range.
If a small demand change reverses the decision, consider a shorter test or a reversible placement.
If the proposal depends on an uncertain estimate of basket retention, plan to measure that outcome.
Where data are too sparse, the honest answer may be to retain the current range until a feasible experiment can reduce uncertainty.

### Step 11 — Make and execute the decision

Input: recommended option, comparison, risk register and implementation dependencies.
Output: signed plan with dates, owners and control limits.
Category owns the final range decision within the approved policy.
Inventory owns stock targets and replenishment settings.
Site operations validates the location and confirms pick readiness.
Finance signs off on the relevant economics, not on food-safety compliance.

The launch checklist should include product-master accuracy, supplier confirmation, initial stock, app listing and employee training.
The site should verify physical fit under the intended receipt pattern.
Customer support needs a documented substitution and complaint path.
Define who can pause a listing immediately if a safety or data issue occurs.

### Step 12 — Learn and reset

Input: measured availability, order outcomes, contribution and store feedback.
Output: continue, modify, scale or roll back, with the decision documented.
Compare results to a predeclared counterfactual, not only to the previous week.
Prior weeks may differ because of weather, promotion, holidays or the operating hours of the store.
Track both item outcomes and total store outcomes so that substitution is visible.

Refresh fast-moving stock settings more frequently than strategic mission coverage if the operating process supports it.
Avoid constant range churn that introduces setup cost and confuses customers.
Archive the decision, model version and assumptions so that a future analyst can reproduce the reason for the change.

## 6. Worked example

### 6.1 Context and scope

A fictional urban dark store has twelve usable discrete storage locations for the modeled products.
Seven of those locations are refrigerated and five are ambient.
The space unit is a validated bin-equivalent, not cubic meters.
A product's listed total footprint includes its cold and ambient portions; it is not added to both again.
The model excludes other store stock by assuming the twelve locations are a dedicated, already-separated zone allocation.
All values are in US dollars per store-week and are synthetic.

The category lead has approved milk, bread and eggs as mandatory for this exercise.
These are business policy choices, not claims that every market needs these exact items.
A plant-based drink, snack and yogurt are optional candidates.
The inventory planner can also choose a deeper milk stock policy.
The milk depth alternative uses one additional refrigerated location and is only available if milk is selected.

The central forecast is a scenario for the next comparable week, not a guaranteed sales outcome.
The plan is to select the best feasible combination and stock it according to the defined policy.
The management target is to improve contribution without reducing required coverage.
No customer delivery commitment is implied by the model.

### 6.2 Candidate data

| Option | Total locations | Refrigerated | Ambient | Incremental contribution, USD/week | Policy |
|---|---:|---:|---:|---:|---|
| Milk, base depth | 3 | 3 | 0 | 42 | Mandatory |
| Bread, base depth | 2 | 0 | 2 | 33 | Mandatory |
| Eggs, base depth | 2 | 2 | 0 | 30 | Mandatory |
| Plant-based drink | 3 | 2 | 1 | 24 | Optional |
| Snack | 2 | 0 | 2 | 19 | Optional |
| Yogurt | 3 | 3 | 0 | 22 | Optional |
| Milk, extra depth | 1 | 1 | 0 | 8 | Optional addition to milk |

Each contribution is already net of the stated per-option waste, handling and estimated substitution effects.
For illustration, the plant drink's USD 24/week consists of USD 38/week in new-item contribution,
minus USD 9/week in displaced contribution and USD 5/week in added handling and waste.
The snack's USD 19/week and yogurt's USD 22/week use the same net convention.
The milk depth value is the modeled extra contribution from avoided lost sales minus added expected waste and carrying cost.
For example, USD 12/week of expected recovered contribution less USD 4/week of additional costs equals USD 8/week.

These values are treated as additive only within this illustrative case.
If the snack and yogurt compete or complete the same basket, a real model needs interaction effects.
A real product's peak footprint also must account for case deliveries and reserve space.

### 6.3 Mandatory baseline

The three mandatory products use 3 + 2 + 2 = 7 total locations.
They use 3 + 0 + 2 = 5 refrigerated locations.
They use 0 + 2 + 0 = 2 ambient locations.
Their modeled contribution is USD 42 + USD 33 + USD 30 = USD 105/week.
Remaining capacity is 5 total, 2 refrigerated and 3 ambient locations.

The mandatory baseline is feasible: 7 ≤ 12 total, 5 ≤ 7 refrigerated and 2 ≤ 5 ambient.
Because total capacity equals the sum of the zone capacities in this example, the total constraint is redundant here.
It remains explicit to make the meaning of the footprint clear and to support other layouts.

### 6.4 Feasible additions

| Additional choice | Extra total | Extra cold | Extra ambient | Added USD/week | Feasible? |
|---|---:|---:|---:|---:|---|
| None | 0 | 0 | 0 | 0 | Yes |
| Plant drink | 3 | 2 | 1 | 24 | Yes |
| Snack | 2 | 0 | 2 | 19 | Yes |
| Milk depth | 1 | 1 | 0 | 8 | Yes |
| Yogurt | 3 | 3 | 0 | 22 | No: refrigerated capacity |
| Plant drink + snack | 5 | 2 | 3 | 43 | Yes |
| Plant drink + milk depth | 4 | 3 | 1 | 32 | No: refrigerated capacity |
| Snack + milk depth | 3 | 1 | 2 | 27 | Yes |
| Snack + yogurt | 5 | 3 | 2 | 41 | No: refrigerated capacity |
| Plant drink + yogurt | 6 | 5 | 1 | 46 | No: multiple capacities |
| Plant drink + snack + milk depth | 6 | 3 | 3 | 51 | No: total and refrigerated capacity |

Any combination containing yogurt is infeasible because the mandatory base leaves only two cold locations.
The table lists the principal comparisons; the code checks all sixteen optional combinations.
The best feasible addition is the plant drink plus snack, with USD 43/week of incremental contribution.
Total modeled contribution is USD 105 + USD 43 = USD 148/week.
Its footprint is 12 total, 7 refrigerated and 5 ambient locations.
All three limits are exactly met, so there is no modeled spare location.

### 6.5 Breadth versus depth

Under the chosen option, two additional products are offered and milk remains at base depth.
The best feasible option containing extra milk depth is snack plus milk depth.
It earns USD 105 + USD 19 + USD 8 = USD 132/week.
Its footprint is 10 total, 6 refrigerated and 4 ambient locations.
The breadth option therefore leads by USD 148 − USD 132 = USD 16/week in the central scenario.
This difference is a model comparison, not observed profit.

The extra milk depth and plant drink cannot coexist because together they need three cold locations after the mandatory baseline.
Thus a generic recommendation to increase milk safety stock would displace the plant drink in this layout.
The decision depends on the plant drink's incremental value after cannibalization, not its gross sales.
If that value falls from USD 24 to below USD 8/week while all else remains fixed,
snack plus milk depth beats snack plus plant drink by the same amount.
At exactly USD 8/week the two options tie, subject to any other unmodeled requirement.

### 6.6 Execution and sensitivity

The recommended central-case plan is milk, bread, eggs, plant drink and snack at the specified depths.
The category lead owns the range approval and the site manager validates the exact bin assignments.
The inventory planner checks that peak receipt quantities fit; finance checks the net contribution assumptions.
The product team must not display the plant drink until initial stock is received and reconciled.

Zero spare capacity is operationally fragile.
If a reserved buffer is mandatory, this recommended plan is infeasible under the revised rules.
For example, reserving one cold location reduces the cold limit from seven to six.
The plant drink then cannot fit alongside the required baseline.
Snack plus milk depth uses six cold locations and is the best remaining feasible option at USD 132/week.
That policy change sacrifices USD 16/week in modeled contribution but creates a one-location cold buffer.
The buffer's operational value is not priced in the base case and must not be described as zero.

If a store policy instead requires the plant-based drink, the cold buffer constraint conflicts with mandatory milk, eggs and plant drink.
Escalate the conflict; do not silently omit the required drink or overfill refrigeration.
The operator could revise a validated footprint, secure more compliant capacity or change the policy after review.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Save it as a local script and run it with `python assortment.py`.
It enumerates all combinations of four optional choices against the mandatory baseline.
It models separate cold, ambient and total limits and requires milk before extra milk depth.
All inputs are the synthetic weekly case values above.

The code reports the optimal central case and a second run with one cold location reserved.
It breaks equal-value ties deterministically by sorted option names.
The model is exhaustive for these four binary choices, not a general-purpose production optimizer.
For a real range, input validation, site-specific depth options and interaction estimates require additional work.

```python
from itertools import product

# All footprints are discrete bin-equivalent locations.
# Contribution is net incremental US dollars per store-week.
items = {
    "milk": {"total": 3, "cold": 3, "ambient": 0, "value": 42},
    "bread": {"total": 2, "cold": 0, "ambient": 2, "value": 33},
    "eggs": {"total": 2, "cold": 2, "ambient": 0, "value": 30},
    "plant": {"total": 3, "cold": 2, "ambient": 1, "value": 24},
    "snack": {"total": 2, "cold": 0, "ambient": 2, "value": 19},
    "yogurt": {"total": 3, "cold": 3, "ambient": 0, "value": 22},
    "milk_depth": {"total": 1, "cold": 1, "ambient": 0, "value": 8},
}
mandatory = ("milk", "bread", "eggs")
optional = ("plant", "snack", "yogurt", "milk_depth")


def evaluate(names):
    """Sum footprints and weekly contribution for one candidate set."""
    return {
        key: sum(items[name][key] for name in names)
        for key in ("total", "cold", "ambient", "value")
    }


def optimize(limits):
    """Enumerate feasible choices and return the highest-value candidate."""
    feasible = []
    for flags in product((False, True), repeat=len(optional)):
        chosen = mandatory + tuple(
            name for name, take in zip(optional, flags) if take
        )
        if "milk_depth" in chosen and "milk" not in chosen:
            continue
        metrics = evaluate(chosen)
        if all(metrics[zone] <= limit for zone, limit in limits.items()):
            feasible.append((metrics["value"], tuple(sorted(chosen)), metrics))
    if not feasible:
        raise ValueError("No feasible assortment under the specified limits")
    # Highest value first; sorted names make ties deterministic.
    return sorted(feasible, key=lambda row: (-row[0], row[1]))[0]


def report(label, limits):
    value, names, metrics = optimize(limits)
    print(label)
    print("  items:", ", ".join(names))
    print("  contribution: USD", value, "/week")
    print(
        "  locations: total={total}, cold={cold}, ambient={ambient}".format(
            **metrics
        )
    )


report("Base case", {"total": 12, "cold": 7, "ambient": 5})
report("One cold location reserved", {"total": 12, "cold": 6, "ambient": 5})
```

Expected output:

```text
Base case
  items: bread, eggs, milk, plant, snack
  contribution: USD 148 /week
  locations: total=12, cold=7, ambient=5
One cold location reserved
  items: bread, eggs, milk, milk_depth, snack
  contribution: USD 132 /week
  locations: total=10, cold=6, ambient=4
```

### How to adapt the calculation

First replace synthetic numbers with the same-store, same-period estimates in a versioned input table.
Check that every option's contribution uses the same counterfactual and cost convention.
Then replace the binary depth increment with mutually exclusive SKU–depth options if more than one level is allowed.
Add explicit constraints for any protected need, zone compatibility or receiving capacity.
A production model should also represent known interactions or bound the error introduced by ignoring them.

Do not use a calculated optimum as proof of physical feasibility without site validation.
A reserved cold location only creates a buffer if that location is actually usable and kept empty when needed.
The example treats contribution estimates as known, even though actual demand and supply are uncertain.
Run scenarios on approved inputs and report the alternatives, not just one winner.

## 8. Data, standards, and regulatory context

### Data contracts

A store-level assortment decision needs a product master with stable identifiers and effective dates.
Include the sellable pack, dimensions, weight, storage class and supplier case configuration.
A changed recipe or pack can affect both customer suitability and the footprint of inventory.
Store dimensions should identify the measured location, allowed product classes and usable peak capacity.
Keep the measurement date and person responsible for validating each capacity.

Sales data should include order ID, order time, item, quantity, paid price and order status.
Distinguish customer cancellations, unavailable items, substitutions, refunds and completed deliveries.
Use event timestamps to reconstruct what was visible when the customer made a choice.
A current catalog snapshot cannot reliably recreate a past listing.

Inventory events should record receipt, adjustment, reservation, pick, return and disposal where applicable.
Capture reason codes and the source system for each adjustment.
Separate physical on-hand, reserved, damaged, quarantined and sellable stock.
For perishable goods, lot and expiry information allow actual FEFO behavior to be checked.
An inventory balance without a timestamp and state is not enough to estimate availability.

Demand enrichment may include promotions, weather, local events and operating hours.
Each join must have an appropriate time and geography.
A weather observation for a large metropolitan area may not represent every store's catchment.
Document missingness and whether any feature was known at the time the decision would have been made.

### Identity and exchange standards

GS1 defines the Global Trade Item Number (GTIN) for identifying trade items.
A GTIN can help reconcile product data across trading partners when correctly assigned and maintained.
It does not itself guarantee that dimensions, allergen attributes or inventory balances are accurate.
Internal SKU codes can still distinguish store-specific operational treatments, provided the mapping is versioned.
See the GS1 identification-key guidance in section 16.

Lot or batch identifiers support traceability when recorded consistently through receipt and handling.
A product identifier is not a substitute for a batch identifier.
Electronic data interchange may transmit orders and shipping notices between partners,
but the message's existence does not prove the physical goods arrived in the expected condition.
Reconcile the message to receipt and exception records.

### Food safety and legal boundaries

Food safety rules are jurisdiction- and product-specific.
In the United States, the Food and Drug Administration's Food Safety Modernization Act resources describe requirements applying to covered facilities and activities.
In the European Union, Regulation (EC) No 852/2004 establishes food hygiene obligations for covered food business operators.
Neither source is a universal planogram or a single temperature rule for every product.
The operator must identify applicable local law, product instructions and competent-authority guidance.
This educational discussion is not legal advice.

Where applicable, follow documented temperature controls, sanitation, segregation and traceability procedures.
A model should not place incompatible goods together merely because their volume fits.
The site must investigate out-of-range temperatures and follow its safety procedures before product is offered.
An expiry date, a best-before date and an internal sell-by cutoff may have different meanings.
Use the applicable jurisdiction's rules and the business's approved handling policy.

Restricted products may also require age verification, licensing or special controls.
The exact obligations depend on the product, location and transaction model.
Legal or compliance staff decide eligibility; an optimizer consumes that decision as a constraint.
Do not infer that a product is legally sellable simply because a supplier catalog contains it.

### Privacy and responsible analysis

Search histories and order records can reveal sensitive household circumstances.
Collect and retain only data appropriate to the declared analysis and applicable policy.
Restrict access, aggregate results where practical and avoid exporting identifiable customer data into example notebooks.
A dietary preference inferred from purchases is uncertain and should not automatically become a sensitive customer label.
Consult privacy counsel or the designated data-protection owner on applicable obligations.

### Data quality checks

| Check | Signal | Decision consequence |
|---|---|---|
| Duplicate item mapping | One physical pack has multiple active identifiers | Sales and capacity can be double counted |
| Missing dimensions | Candidate has no approved footprint | Cannot be safely slotted from the model |
| Stale store capacity | Layout changed after measurement | Feasibility may be false |
| Negative available stock | Reservations exceed recorded sellable inventory | Investigate stock-state or timing errors |
| Unexplained availability gaps | No state event for a long interval | Demand estimate has unknown exposure |
| Receipt after expiration | Lot dates conflict with receipt timestamp | Quarantine and investigate data or product |
| Price interval overlap | More than one active price at an instant | Contribution calculation is ambiguous |
| Time-zone mismatch | Events appear outside actual operating hours | Slot demand is shifted |
| Cancellation mismatch | Cancelled orders counted as fulfilled | Contribution and fill rate are overstated |
| Supplier pack change | Master pack differs from received case | Peak space and waste estimates are wrong |

Assign every check an owner, threshold and exception process.
A low-frequency discrepancy can still be critical for a restricted or perishable product.
Do not fill missing dimensions with zero; that makes an unknown item appear to consume no space.

## 9. Industry practice and case evidence

### Evidence hierarchy

Public company material can establish that an operator described a strategy or service.
It does not automatically establish causality, current operational practice or store-level profitability.
Peer-reviewed research can support a mechanism under studied conditions but may not transfer unchanged to a quick-commerce site.
Regulatory and standards sources establish the scope of their own published rules or guidance, not every local requirement.

### Online grocery and fulfillment choices

Ocado Group describes automated fulfillment technology and its grocery operations in public corporate materials.
Those materials show that online grocery fulfillment can be organized around specialized infrastructure.
They do not establish that every dark store has the same economics or should use automation.
[Inference] A specialized fulfillment environment still requires product-location, inventory and availability decisions,
but the scale and automation constraints differ substantially from a small neighborhood facility.

Tesco's online grocery service describes substitutions and customer choices in its help materials.
This is public evidence that substitutions are part of a real online grocery customer experience.
The exact rules and availability of preference controls can change by market and time.
[Inference] An assortment analysis that ignores substitution can misstate both customer outcomes and incremental sales.
The existence of a substitution feature does not reveal Tesco's private substitution probabilities or contribution margins.

### Assortment research

The paper by Kök, Fisher and Vaidyanathan reviews assortment planning and substitution modeling.
Its literature-based discussion supports treating customer choice and constrained selection as linked problems.
It does not provide a ready-made dark-store assortment for a particular neighborhood.
Use its modeling framework as a starting point and estimate local parameters separately.

The paper by Anupindi, Dada and Gupta studies demand estimation when inventory availability censors sales.
Its relevance is the mechanism: sales can be constrained by stock and choice among available alternatives.
Its assumptions and setting should be reviewed before applying any particular estimator.
[Inference] Using raw sales alone to rank a frequently unavailable item can understate its opportunity.
The degree of bias in a specific store must be estimated from that store's data.

### Product identity and food handling

GS1's GTIN guidance documents a common trade-item identification system.
It is evidence for an interoperability convention, not evidence that any one retailer's master data are complete.
FDA and EU sources describe food-safety regulatory frameworks within their respective scopes.
They do not validate a specific site or certify that the examples in this unit meet all obligations.

### Applying cases responsibly

Before transferring a public case, compare promise, customer density, store size and supplier network.
A large automated center can use different replenishment and picking economics from a small dark store.
A densely populated catchment can support more local variety than a sparse one at the same service promise.
A strong national brand preference can alter substitution and mission coverage.
Check whether the source reports actual results or merely describes a planned initiative.

| Evidence | What it can support | What it cannot support alone |
|---|---|---|
| Operator product page | A stated feature or service | Causal profit impact |
| Annual report | Company-reported strategy and aggregate results | Store–SKU contribution |
| Peer-reviewed model | Mechanism and assumptions under study | Local parameter values |
| Standards body | Identifier definitions and use | Accuracy of an implementation |
| Regulator | Rules or guidance within scope | Advice for every jurisdiction |
| Site trial | Local measured change under trial conditions | Universal external validity |

### Three additional decision cases

The following short cases use hypothetical figures.
Each illustrates a different way that an assortment decision can fail if the business case is too narrow.
The amounts are US dollars per store-week unless otherwise stated.

#### Case A — A low-volume emergency item

A store carries one pack of an emergency household product.
Its direct contribution after waste and handling is USD 4/week.
A replacement snack would produce USD 11/week of direct contribution if all its sales were incremental.
The household product is nevertheless included in some baskets that customers would abandon without it.

The team estimates six orders/week contain the household product.
On average, the other lines in those orders contribute USD 8/order.
Customer interviews and historical unavailability suggest, as a scenario rather than an established causal estimate,
that one quarter of those orders would be lost if the product were removed.
The estimated companion contribution at risk is 6 × 0.25 × USD 8 = USD 12/week.
Adding direct contribution gives USD 4 + USD 12 = USD 16/week of modeled value.
Under this scenario, replacing the item with the snack sacrifices USD 5/week.

This calculation does not prove that the household item should always remain.
Some customers may place their orders later or choose another acceptable product.
The scenario also assumes the companion contribution is not already included in the USD 4 direct contribution.
If the snack also creates new baskets, that increment must be added on a comparable basis.
If the two items have different space requirements, the analysis must compare complete feasible combinations.

At what lost-order probability would the household product and snack tie?
Set USD 4 + 6 × probability × USD 8 equal to USD 11.
The break-even probability is 7 / 48, or approximately 14.6% of eligible orders.
If the credible probability range straddles 14.6%, the decision is sensitive to uncertain customer behavior.
The category lead can request a limited test or preserve coverage while better evidence is gathered.
The product team should measure whole-order outcomes without exposing customers to unsafe substitutions.

#### Case B — A high-selling duplicate

A new beverage is forecast to generate 80 units/week at USD 0.75/unit in direct contribution.
That produces USD 60/week before considering displacement.
The analyst expects 50 of those units to replace sales of an existing beverage earning USD 0.70/unit.
The lost contribution on those existing units is 50 × USD 0.70 = USD 35/week.
Additional handling and waste cost USD 8/week.
Net incrementality is USD 60 − USD 35 − USD 8 = USD 17/week.

An alternative item occupying the same locations is forecast to add USD 22/week net.
On the central assumptions and without a coverage requirement, the alternative leads by USD 5/week.
The new beverage may still be preferred if it serves a documented unmet need or has a better downside profile.
The model should show that trade-off explicitly rather than inflate its sales estimate.

The 50-unit displacement assumption is the pivotal quantity.
If only 35 units are displaced, the new beverage yields USD 60 − USD 24.50 − USD 8 = USD 27.50/week.
It would then lead the alternative by USD 5.50/week.
The decision reverses when displaced units equal approximately 42.9 units/week:
USD 60 − USD 8 − 0.70 × displaced units = USD 22.
That threshold helps decide whether better substitution evidence is worth collecting.

Track the combined beverage category, not only the new line, during a trial.
Also observe whether price or promotion changed during the trial.
A price reduction can produce apparent new-item demand without proving that the assortment change caused it.
Finance should reconcile the trial estimate with actual waste and handling invoices where possible.

#### Case C — A short-life product with a supplier minimum

A prepared product has an expected demand of 18 units/week when continuously available.
The supplier sells cases of 24 units, with one weekly delivery.
For this illustration, assume all 18 expected units sell before the internal selling cutoff,
the remaining six units are discarded, and there is no salvage value.
Each sold unit earns USD 2.00 before unsold inventory cost.
Each discarded unit costs USD 1.20 to acquire and dispose of.

Expected contribution before handling is 18 × USD 2.00 − 6 × USD 1.20 = USD 28.80/week.
If special handling costs USD 7.80/week, the resulting contribution is USD 21/week.
This is not a full stochastic waste estimate: demand variability can increase both lost sales and waste.
The assumption of exactly 18 sales is useful for arithmetic, not an operational forecast distribution.

Suppose a split-case agreement permits a 20-unit order without changing unit costs.
Under the same deterministic demand assumption, two units are discarded.
The contribution becomes USD 36 − USD 2.40 − USD 7.80 = USD 25.80/week.
The split case improves the simple scenario by USD 4.80/week.
If the supplier charges a split-case fee above USD 4.80/week, it erases that modeled benefit.
The true comparison should also include the risk of stockout when demand exceeds 20 units.

The inventory planner should test the peak space for the full incoming case, not the average stock.
The category lead should ask the supplier about remaining life at delivery and any return terms.
Food safety owns the approved selling cutoff and product-handling conditions.
Finance should distinguish the carrying cost of inventory from the cost of units actually written off.

### A practical experiment design

An assortment intervention changes more than the presence of a single item.
It can change what customers see, which orders they place, how staff pick and how suppliers deliver.
Define the intervention as a complete policy before measuring its effect.
For example, treatment could be a new product plus its approved stock target and replacement slot.
Control could retain the existing product and replenishment setting.
If the new product is unavailable for most of the test, report implementation exposure separately from assignment.

Choose the unit of assignment to avoid interference where practical.
If neighboring stores compete for the same customers, store-level treatment may shift demand between them.
Cluster nearby stores or measure customer switching when data and privacy rules allow.
Random assignment can reduce selection bias but does not automatically guarantee balance in a small sample.
Predeclare the analysis, duration and operational stop conditions.

A primary outcome might be total variable contribution per store-week.
The definition should state whether delivery, payment, picking, refunds and waste are included.
Measure the entire store when the hypothesis involves basket retention or cannibalization.
Report secondary outcomes such as need coverage, item availability and substitution acceptance.
Also report sample size, uncertainty and missing data rather than only a percentage uplift.

Use intention-to-treat analysis when comparing assigned stores, preserving the assigned groups.
Separately describe whether the planned listing and stock policy were actually implemented.
Do not treat an after-the-fact comparison of only successful launches as an unbiased treatment effect.
Investigate why a store could not implement the policy: supplier failures may be part of the real operating effect.

Safety and customer trust guardrails can trigger immediate review even before the planned analysis date.
A site can pause an unsafe product without waiting for statistical significance.
The designated safety owner determines how product is isolated and assessed.
Finance should not offset a safety incident against modeled contribution.
Operational triggers should specify the measure, owner, time window and allowed response.

At the end of the experiment, separate three possible conclusions.
First, the evidence may favor scaling within the conditions tested.
Second, the evidence may favor stopping because the downside or service harm is material.
Third, results may be too uncertain to distinguish useful alternatives.
In the third case, decide whether another experiment has sufficient decision value to justify its cost.

### Business-case worksheet

Before approval, finance and the decision owner can complete the following worksheet.
Each amount should be stated for the same store, currency and time period.
The worksheet is intentionally incremental rather than a full company profit-and-loss statement.

| Component | Question | Evidence or owner |
|---|---|---|
| New-item contribution | What margin is earned on fulfilled incremental units? | Category and finance |
| Displaced contribution | Which existing sales would disappear? | Analytics and category |
| Companion contribution | Which other lines are genuinely caused by the change? | Experiment owner |
| Waste and markdown | What is the change in expected unsellable stock and price reduction? | Inventory and finance |
| Picking and replenishment | Does the policy change variable labor or require another shift? | Site operations |
| Delivery and payment | Do order count, payment fees or delivery variable cost change? | Operations and finance |
| Supplier terms | Do case packs, fees, allowances or returns differ? | Procurement |
| One-time launch cost | What setup, relabeling or training is required? | Launch owner |
| Capacity opportunity | What approved option or buffer is displaced? | Site and category |
| Customer coverage | Which needs improve or deteriorate? | Category and customer team |
| Compliance | Are there unresolved safety or legal constraints? | Safety and legal |
| Uncertainty | What plausible input changes reverse the choice? | Analytics |

Subtract one-time cost once, not in every weekly period.
If a launch costs USD 240 once and is expected to create USD 30/week,
the simple undiscounted payback is eight weeks, assuming the benefit actually persists.
Payback is not a substitute for a full discounted cash-flow analysis when timing and duration matter.
If the expected benefit is uncertain or negative, present scenarios instead of a misleading single payback number.

### Operational review cadence

A daily site review can investigate stock discrepancies, temperature incidents and pick failures.
A weekly inventory review can adjust replenishment within the already approved assortment policy.
A periodic category review can consider range changes using accumulated customer and financial evidence.
The appropriate cadence depends on product life, supplier constraints and operational cost.
The review should not reset a product every time a noisy one-day metric changes.

Keep an explicit event log whenever a product is added, removed or moved.
Record the policy version, decision date, effective time, responsible owner and reason.
Record whether the change was planned, emergency, seasonal or supplier-driven.
This history allows analysts to reconstruct exposure and distinguish stockouts from intentional delisting.
When a trial ends, record the final decision and the operational changes needed to implement it.

An escalation record should include the actual constraint that failed.
For example, “cold receiving capacity exceeded on Tuesday” is more actionable than “assortment failed.”
Attach the measurement period, affected locations and any safety controls applied.
The owner should identify whether the cause is inaccurate dimensions, supplier timing or excess stock.
The corrective action may be a new delivery window rather than removal of a customer-critical item.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Rank raw sales only | Frequent stockouts among removed items | Sales censor demand | Remove a needed item | Model exposure and lost demand |
| Optimize gross revenue | Revenue rises while contribution falls | Costs and displacement omitted | Apparent growth destroys value | Use consistent incremental contribution |
| Pool all space | Chilled utilization exceeds physical capacity | Ambient bins treated as interchangeable | Plan cannot be executed | Constrain each validated zone |
| Ignore receiving peak | Deliveries block aisles despite average fit | Case receipt exceeds bin footprint | Unsafe congestion or missed receipts | Model peak receipt and reserve capacity |
| Overstate substitution | High refund or rejection rate | Claimed substitute is unacceptable | Lost trust and basket abandonment | Use customer preferences and measured outcomes |
| Ignore cannibalization | New SKU sells while category profit is flat | Demand shifts from existing SKU | Incremental benefit overstated | Compare category and store control outcomes |
| Double count basket uplift | Item and full order revenue both credited | Same transaction appears twice | Inflated business case | Define additive components and reconcile |
| Optimize mean stock | Frequent expiry alongside stockouts | Demand and lead time vary around mean | Waste and poor service coexist | Use distributions and feasible depth options |
| Ignore case pack | Small plan requires full supplier cases | Order constraints omitted | Overflow and write-offs | Model MOQ, pack size and delivery timing |
| Treat recorded as pickable | App shows stock but pickers cannot find it | Damaged, reserved or misplaced stock included | Customer cancellations | Reconcile stock states and cycle counts |
| Eliminate all buffer | Every zone is always modeled as full | No room for variability or safe handling | Fragile operation | Validate peak and consider explicit reserve |
| Apply stale dimensions | New pack does not fit its old bin | Product change not propagated | Unsafe stacking or pick failure | Version dimensions and trigger re-slotting |
| Mix incompatible products | Staff find unsuitable co-location | Storage rule absent from model | Safety or quality incident | Encode approved segregation and audit layout |
| Confuse target with forecast | Plan assumes target availability is certain | Desired outcome substituted for evidence | Overpromised sales and service | Report forecast, target and risk separately |
| Overfit a small trial | One store-week drives full rollout | Noise or unusual local event | Poor network-wide decision | Predefine duration, uncertainty and replication |
| Compare unlike weeks | A holiday week is used as baseline | Seasonality changes demand | False improvement or decline | Use comparable controls and exposure |
| Uncontrolled range churn | Frequent re-slotting and missing labels | Model is refreshed without change costs | Labor cost and customer confusion | Set review cadence and launch threshold |
| Missing accountability | Exceptions remain open across teams | No owner can approve or pause | Unsafe or delayed launch | Name decision and escalation owners |

A red flag is a reason to investigate, not automatic proof of harm.
The severity and required response depend on the specific product and operating context.
Any safety-critical uncertainty should be escalated through the site's established process.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Peak footprints are correct | Feasibility relies on location use | New pack, case receipt or inaccurate measurements | Physical slot and receipt trial |
| Zones are correctly assigned | Cold and ambient limits differ | Misclassified storage requirement | Product specification and site review |
| Contributions are incremental | Comparisons require a common baseline | Displacement or companion sales counted twice | Reconcile item and total-store economics |
| Values are additive in the example | Allows simple enumeration | Products substitute or complement each other | Estimate interaction scenarios or run a test |
| Demand estimate accounts for exposure | Raw sales can be censored | Stockouts cluster in peak hours | Compare slot-level exposure and sales |
| Supplier terms hold for the period | MOQ and lead time determine stock depth | Supplier changes pack, price or delivery frequency | Confirm terms before launch and monitor receipts |
| Mandatory rules are legitimate | The model treats them as hard constraints | A rule is outdated or undocumented | Owner approval and periodic review |
| Safety requirements are complete | Optimization cannot discover omitted rules | New product hazard or jurisdiction | Safety and legal review before listing |
| Product data identify the right pack | Forecast and footprint must match the same item | Duplicate or reused identifiers | GTIN and master-data reconciliation |
| Operational labor is adequately valued | Net contribution includes avoidable handling | Pick paths or workload change nonlinearly | Time study and site trial |
| Scenario period is comparable | Weekly values are compared fairly | Holiday, weather or campaign changes | Matched periods and sensitivity analysis |
| Inventory is sellable when recorded | Availability drives both demand and service | Damage, expiry, shrink or phantom stock | Cycle counts and pick-failure analysis |
| Available locations can be filled at once | The model allocates capacity simultaneously | Access or stacking interactions | Physical layout walkthrough |
| Fixed costs remain unchanged | Only avoidable costs belong in short-run contribution | Additional shift or refrigeration is required | Capacity and staffing approval |
| Customer response is stable enough | Historical estimates inform the next decision | Competitor entry or new customer mix | Monitor cohort and catchment shifts |

These assumptions are not equally important in every case.
Prioritize by the size of the decision reversal and the severity of a potential safety or customer outcome.
A useful sensitivity analysis says which assumption changes the recommendation and by how much.
No optimizer can compensate for an unrecorded legal prohibition or an unsafe physical arrangement.

## 12. Exercises

Use the synthetic case in section 6 unless an exercise says otherwise.
All currency amounts are US dollars per store-week unless a different unit is stated.

### Exercise 1 — Classify the decision

A store removes one snack flavor and uses its bin to hold more units of the remaining flavor.
Deliverable: identify the changes in breadth and depth, and name one possible customer and one operational consequence.
Acceptance: distinguish the number of distinct choices from units held and avoid assuming demand is unchanged.
Time: 8 minutes.

### Exercise 2 — Test feasibility

A candidate combination uses the mandatory baseline plus plant drink and milk depth.
Deliverable: calculate total, refrigerated and ambient locations and state whether it fits.
Acceptance: show each sum, compare to all three limits and identify every violated limit.
Time: 10 minutes.

### Exercise 3 — Calculate incrementality

A new item contributes USD 55/week before substitution.
It displaces USD 17/week of existing contribution and adds USD 9/week of waste and handling.
Deliverable: compute net incremental contribution and describe the counterfactual.
Acceptance: subtract both displaced contribution and incremental costs; keep the same store-week basis.
Time: 8 minutes.

### Exercise 4 — Compare breadth with depth

In the worked case, reduce plant-drink value to USD 6/week; leave all other assumptions unchanged.
Deliverable: identify the optimal feasible option and its weekly contribution.
Acceptance: consider the refrigerated conflict with milk depth and state the resulting gap against plant plus snack.
Time: 12 minutes.

### Exercise 5 — Diagnose censored sales

An item sells 12 units in six in-stock morning hours and 4 units in eight in-stock afternoon hours.
It is unavailable for two more morning hours and two more afternoon hours.
Assume within each time slot that in-stock demand rates apply to unavailable hours.
Deliverable: estimate missed units and explain one reason the assumption may fail.
Acceptance: calculate separate slot rates, multiply by unavailable hours and label the result an estimate.
Time: 12 minutes.

### Exercise 6 — Critique an unsafe proposal

An analyst proposes adding yogurt because it has positive weekly contribution.
They pool ambient and chilled locations and use average inventory instead of receipt peaks.
Deliverable: write a short rejection and specify what must be validated before reconsideration.
Acceptance: identify the cold constraint, peak-space issue, relevant food-safety review and a feasible alternative process.
Time: 12 minutes.

### Exercise 7 — Design a trial

A category lead wants to test one additional dietary-need product across six stores.
Deliverable: propose an allocation, measurement plan, minimum guardrails and a rollback owner.
Acceptance: define the unit and duration, comparable control, primary incremental outcome, availability, waste and safety measures;
address spillovers and uncertainty without asserting that six stores automatically provide statistical power.
Time: 25 minutes.

### Exercise 8 — Write a decision memo

The cold-buffer rule is being considered for the worked case.
Deliverable: write a one-page memo recommending either the base option or the buffer option.
Acceptance: report both modeled contributions and capacity use, state a nonfinancial rationale, name owners,
separate forecast from target and commitment, identify assumptions and define a review trigger.
Time: 20 minutes.

## 13. Self-check answers

### Answer 1

Breadth falls by one distinct snack flavor.
Depth of the remaining flavor increases if the freed bin actually holds additional units of it.
Some customers may lose a preferred option or abandon a basket.
The extra units may reduce stockouts, but only if replenishment, demand and shelf life support that outcome.

### Answer 2

Total locations are 7 + 3 + 1 = 11, within the limit of 12.
Refrigerated locations are 5 + 2 + 1 = 8, above the limit of 7.
Ambient locations are 2 + 1 + 0 = 3, within the limit of 5.
The combination is infeasible because refrigerated capacity is exceeded by one location.

### Answer 3

Net incremental contribution is USD 55 − USD 17 − USD 9 = USD 29/week.
The counterfactual includes the contribution from existing items that customers would otherwise have bought.
If a measured treatment effect already included displaced sales, subtracting them again would be double counting.

### Answer 4

Snack plus milk depth adds USD 19 + USD 8 = USD 27/week to the mandatory baseline.
The total is USD 105 + USD 27 = USD 132/week.
Plant plus snack adds USD 6 + USD 19 = USD 25/week, for USD 130/week.
They cannot be combined with milk depth because cold capacity would be exceeded.
The depth option therefore leads by USD 2/week under these revised inputs.

### Answer 5

Morning observed rate is 12 / 6 = 2 units/hour.
The two unavailable morning hours imply 2 × 2 = 4 estimated missed units.
Afternoon observed rate is 4 / 8 = 0.5 units/hour.
The two unavailable afternoon hours imply 0.5 × 2 = 1 estimated missed unit.
The total is 5 estimated missed units, not a directly observed count.
The unavailable morning hours may be the busiest hours, so the in-stock average could be biased downward.
Customers may also substitute or defer purchases, so missed item units are not necessarily lost orders.

### Answer 6

The mandatory baseline already uses five of seven refrigerated locations.
Yogurt needs three more cold locations, so it cannot fit even though its contribution is positive.
Pooling ambient capacity cannot change refrigeration or safe storage requirements.
Average stock also does not show the footprint at case receipt.
Reconsider only after site and food-safety owners validate a feasible cold layout, receipt peak and approved product handling.
Then compare the change against displaced products and their contribution.

### Answer 7 rubric

A strong answer specifies a store-level treatment and comparable control, ideally with a defensible assignment rule.
It defines a fixed observation window and measures whole-store incremental contribution per comparable exposure or store-week.
It tracks product availability, basket completion, substitution acceptance, waste and operational burden.
It explains contamination, such as customers shifting orders between nearby treated and control sites.
It includes a safety stop, a named owner and a feasible rollback procedure.
It acknowledges that six stores may yield wide uncertainty and does not promise a statistically conclusive result.

### Answer 8 rubric

A strong memo states USD 148/week for the base option and USD 132/week for the one-cold-location buffer option.
It reports base use of 12 total, 7 cold and 5 ambient locations and buffer use of 10 total, 6 cold and 4 ambient locations.
It explains that USD 16/week is modeled opportunity cost, not an observed cash loss.
It names a decision owner and the site, inventory, finance and safety reviewers.
It distinguishes the next-week demand forecast, selected plan, management target and any actual customer commitment.
It sets a review trigger, such as persistent receipt congestion or a material change in measured incrementality.

## 14. Interview questions

1. **How would you distinguish assortment breadth from depth?**
   A strong answer uses a concrete product example and identifies the common capacity constraint.
   A weak answer calls every additional unit a new SKU.
2. **Why can a top-selling SKU be a poor addition?**
   A strong answer discusses displacement, waste, handling and the counterfactual.
   A weak answer assumes gross revenue equals incremental profit.
3. **Why are observed sales not always demand?**
   A strong answer explains unavailable exposure and peak-hour censoring.
   A weak answer treats zero sales during a stockout as zero interest.
4. **How would you allocate a constrained cold location?**
   A strong answer compares feasible combinations, safety rules and the value of a buffer.
   A weak answer pools ambient and cold space or ranks only by gross margin.
5. **What does an optimizer need before it can select a range?**
   A strong answer names eligible items, capacity, economics, interactions and policy constraints.
   A weak answer assumes an algorithm can discover unrecorded safety requirements.
6. **How can you estimate substitution?**
   A strong answer distinguishes stated preferences, observed switching and causal trials, with selection bias.
   A weak answer assumes every alternative in the same category is acceptable.
7. **How would you handle a product with high customer importance but low direct contribution?**
   A strong answer quantifies available economics and makes a transparent owner-approved coverage rule.
   A weak answer hides the policy in an unexplained weighted score.
8. **Why can average inventory fit while the plan fails?**
   A strong answer describes case receipt peaks, reserve stock and physical geometry.
   A weak answer assumes mean cube is the maximum instantaneous requirement.
9. **What should a range experiment measure?**
   A strong answer includes store-level contribution, exposure, customer outcomes, waste and safety guardrails.
   A weak answer declares success from new-item sales alone.
10. **What happens when mandatory items do not fit?**
    A strong answer escalates the conflicting constraints and evaluates approved changes.
    A weak answer quietly exceeds capacity or drops a required item.

## 15. Key terms

- **Assortment:** selected store–SKU range; see [definitions](#4-definitions-and-vocabulary).
- **Available to promise:** eligible quantity after relevant reservations; see [data contracts](#data-contracts).
- **Basket:** lines in one customer order; see [incrementality](#step-6--estimate-incrementality-and-relationships).
- **Breadth:** distinct options offered; see [mental model](#3-the-core-mental-model).
- **Cannibalization:** shifted demand from an existing item; see [incrementality](#step-6--estimate-incrementality-and-relationships).
- **Case pack:** supplier's supplied grouping; see [depth candidates](#step-7--choose-stock-depth-candidates).
- **Cold zone:** separately constrained refrigerated space; see [capacity map](#step-3--build-an-effective-capacity-map).
- **Complement:** item whose value can rise with another; see [mental model](#3-the-core-mental-model).
- **Contribution:** relevant revenue less avoidable costs; see [worked example](#6-worked-example).
- **Counterfactual:** outcome without the proposed change; see [decision scope](#step-1--define-the-decision-and-the-comparison).
- **Coverage:** ability to serve a defined need; see [missions](#step-2--define-customer-missions-and-coverage).
- **Depth:** stock or location capacity for a chosen item; see [depth comparison](#65-breadth-versus-depth).
- **Effective capacity:** physically usable capacity after allowances; see [capacity map](#step-3--build-an-effective-capacity-map).
- **FEFO:** first-expiry-first-out rotation; see [data contracts](#data-contracts).
- **Forecast:** evidence-based estimate, not a commitment; see [worked context](#61-context-and-scope).
- **GTIN:** GS1 identifier for a trade item; see [identity standards](#identity-and-exchange-standards).
- **In-stock rate:** availability over a defined denominator; see [demand reconstruction](#step-5--reconstruct-demand-under-availability).
- **Incrementality:** value caused by a change relative to baseline; see [incrementality](#step-6--estimate-incrementality-and-relationships).
- **Lead time:** elapsed time from trigger to usable stock; see [depth candidates](#step-7--choose-stock-depth-candidates).
- **Lost demand:** unfulfilled desired demand; see [exercise 5](#exercise-5--diagnose-censored-sales).
- **Mission:** customer job to be accomplished; see [missions](#step-2--define-customer-missions-and-coverage).
- **MOQ:** minimum order quantity; see [candidate cleaning](#step-4--clean-the-candidate-product-universe).
- **Peak footprint:** location requirement at the operating maximum; see [capacity map](#step-3--build-an-effective-capacity-map).
- **Planogram:** intended SKU placement; see [slotting](#step-11--make-and-execute-the-decision).
- **Replenishment:** restoring stock through receipt or movement; see [depth candidates](#step-7--choose-stock-depth-candidates).
- **SKU:** operationally distinct sellable item; see [definitions](#4-definitions-and-vocabulary).
- **Stockout:** interval in which desired inventory is not available; see [demand reconstruction](#step-5--reconstruct-demand-under-availability).
- **Substitution:** acceptable alternative chosen in place of an item; see [industry evidence](#online-grocery-and-fulfillment-choices).
- **Waste:** unsellable product and associated cost; see [failure modes](#10-failure-modes-and-red-flags).
- **Zone:** separately constrained storage class; see [constraints](#step-8--formulate-hard-constraints).

## 16. Sources

1. GS1, [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin).
   Identifier definition and use; a standard does not certify local master-data accuracy.
2. U.S. Food and Drug Administration, [Food Safety Modernization Act](https://www.fda.gov/food/food-safety-modernization-act-fsma).
   Entry point to U.S. regulatory resources; scope and applicability require case-specific review.
3. European Parliament and Council, [Regulation (EC) No 852/2004 on the hygiene of foodstuffs](https://eur-lex.europa.eu/eli/reg/2004/852/oj).
   Primary EU legal text; consult current consolidated law and local implementation as applicable.
4. Kök, A. G., Fisher, M. L. and Vaidyanathan, R., “Assortment Planning: Review of Literature and Industry Practice,”
   in *Retail Supply Chain Management*, Springer, 2008, [DOI: 10.1007/978-0-387-78902-6_6](https://doi.org/10.1007/978-0-387-78902-6_6).
   Research review of assortment and substitution; not a local parameter source.
5. Anupindi, R., Dada, M. and Gupta, S., “Estimation of Consumer Demand with Stock-Out Based Substitution: An Application to Vending Machine Products,”
   *Marketing Science*, 17(4), 1998, [DOI: 10.1287/mksc.17.4.406](https://doi.org/10.1287/mksc.17.4.406).
   Empirical and methodological treatment of demand under stockouts; setting differs from dark stores.
6. Ocado Group, [Our technology](https://www.ocadogroup.com/technology).
   Company description of grocery technology; company claims should not be read as independent causal evidence.
7. Tesco, [Unwanted substitutions on electronic customer delivery notes](https://www.tesco.com/help/pages/online-grocery-faqs/ecdns-electronic-customer-delivery-notes/unwanted-substitutions-on-ecdns-and-why-we-have-to-deliver-them).
   Customer-facing service description; features and terms can vary or change.

Sources are cited for the propositions stated, not as endorsement of a particular operator.
Regulatory references are educational, not a determination of compliance for a specific site.

## 17. Links

- [Module 22 index](./_index.md).
- [Promise tiers and dark-store design](./01_promise-tiers-dark-stores.md).
- [Long-tail forecasting in e-commerce](../21_ecommerce_supply_chain_analytics/03_long-tail-forecasting.md).
- [Inventory placement for e-commerce](../21_ecommerce_supply_chain_analytics/04_inventory-placement.md).
- [Module 22 survey and its lab sections](../../../course/22_quick_commerce_on_demand_delivery.md).
- [Inventory optimization survey](../../../course/10_inventory_optimization_multi_echelon.md).
- [Assortment and category retail analytics survey](../../../course/17_assortment_category_retail_analytics.md).
- [Simulation and optimization labs](../../../course/30_simulation_optimization_production_labs.md).
