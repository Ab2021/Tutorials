# New Item Acceptance and Cannibalization

> Module 17 — Assortment, Category, and Retail Analytics  
> Topic 10 of 10  
> Estimated study time: 4 hours  
> Prerequisites: 01, 03, and basic experimental design  
> Version 2026.5

## 1. Why this topic exists

A buyer must decide whether to give a new product scarce shelf space.
The supplier presents a forecast of strong sales.
The category manager asks whether those sales will add to the business or displace products already sold.
The store operator asks what must leave the shelf and whether the new item can stay in stock.
Finance asks whether the incremental contribution covers the cost of launch.
The answers are related, but they are not interchangeable.

New-item sales measure what went through the till under the launch.
They do not reveal what the same shoppers would have purchased without the launch.
Some buyers would have purchased another item in the category.
Some would have visited another store, bought elsewhere online, delayed purchase, or bought nothing.
Some may now buy complementary products, while others may reduce purchases of products outside the category.
The relevant counterfactual depends on the decision being made.

A category can increase sales while earning less contribution.
A cheaper new item may move more units but displace a higher-margin incumbent.
A premium item can generate few units but improve contribution and serve an unmet need.
A product can meet a supplier's shipment target while sitting unsold in retailer inventory.
A trial can look successful simply because its stores were more likely to grow anyway.

Good acceptance practice therefore separates four questions.
First, is the product eligible and feasible to list?
Second, can the launch be executed consistently and safely?
Third, what changed relative to a credible no-launch comparison?
Fourth, is that change worth the resources required to sustain and scale it?

The stakes are practical.
Shelf capacity, working capital, replenishment attention, promotional support, and shopper trust are finite.
A weak launch can also cause lost sales by removing an incumbent shoppers still need.
A prematurely rejected item may forgo a genuinely incremental segment or a strategically important capability.
A post-launch review should expose both possibilities rather than defend the original forecast.

This unit follows a fictional grocery launch from proposal to pilot and decision.
The numbers are illustrative, not evidence about a real retailer or brand.
The comparison estimates a pilot-period effect; it is not automatically a prediction for every store.
The commercial examples use US dollars and store-weeks unless stated otherwise.
The method applies to other currencies when all prices and costs are converted consistently.

## 2. Learning outcomes

After working through this unit, you should be able to:

1. Distinguish an item's observed sales from the incremental category sales caused by listing it.
   Demonstrate the distinction in Exercise 1.
2. Specify an acceptance gate that includes shopper need, operational feasibility, and an explicit alternative.
   Produce the gate in Exercise 2.
3. Design a pilot with a defensible counterfactual, assignment rule, execution plan, and review date.
   Design that pilot in Exercise 3.
4. Calculate a difference-in-differences estimate, transferred sales, and a cannibalization ratio.
   Reproduce the arithmetic in Exercise 4.
5. Convert incremental sales into contribution and compare them with relevant launch and ongoing costs.
   Evaluate that business case in Exercise 5.
6. Diagnose threats from promotion, stockouts, spillovers, and selective reporting.
   Critique those threats in Exercise 6.
7. Make a scale, adapt, extend, or stop recommendation with uncertainty and a named owner.
   Write the recommendation in Exercise 7.
8. Interpret a post-launch review without confusing a forecast, a target, a plan, and a commitment.
   Classify those statements in Exercise 8.

You should be able to explain why product sales and category lift answer different questions.
You should also be able to identify a missing counterfactual before calculating a ratio.
The exercises include numerical answers where a single answer is justified.
Design tasks use a rubric because there can be several defensible solutions.

## 3. The core mental model

### The decision is a comparison between worlds

```text
Decision: list new item in an eligible store?
       |
       +--> WITH launch: new SKU + incumbent SKUs + execution costs
       |          |
       |          +--> observed category and business outcomes
       |
       +--> WITHOUT launch: existing assortment + normal operations
                  |
                  +--> counterfactual category and business outcomes

 WITH minus WITHOUT = incremental effect for the defined population and period
       |
       +--> sales effect --> contribution effect --> relevant cash flows
       |
       +--> scale / adapt / extend / stop, subject to uncertainty and constraints
```

The same store cannot simultaneously have and not have the item in the same week.
A randomized comparison group helps estimate the missing outcome.
It is not the missing outcome itself.
Sampling variation, operational differences, and spillovers can still affect the estimate.

The new stock-keeping unit (SKU) may attract demand that would otherwise have gone to incumbent SKUs.
That transferred demand is a real commercial phenomenon, not automatically a reason to reject the item.
A substitute may be less profitable, have supply risk, or fail to serve a strategic customer need.
Conversely, a launch with a low estimated transfer rate may still destroy value if its costs are high.

The unit of analysis must match the decision.
For a store-level listing decision, a store-week is often useful.
For a chain-wide decision, include affected channels and warehouses when the data allow it.
A store-only estimate can miss demand moving between stores or from e-commerce to physical retail.
A category-only estimate can miss complementary purchases or substitution across category boundaries.

### Separate three layers of evidence

The descriptive layer reports units, revenue, availability, and margin observed after launch.
The causal layer compares outcomes with an estimated no-launch counterfactual.
The decision layer applies relevant costs, constraints, risk preferences, and strategic considerations.
A strong descriptive report does not by itself establish causality.
A credible causal estimate does not by itself make a launch profitable.
A positive pilot profit does not establish that a nationwide rollout will reproduce it.

### Bound the question before you measure

Define the item, eligible stores, assignment date, measurement period, category boundary, and outcome.
Specify whether the outcome is revenue, units, gross profit, or contribution after variable costs.
State whether the rollout decision applies to all eligible stores or a specific segment.
State whether the launch replaces a named incumbent or uses empty capacity.
The treatment is the whole launch package, including its shelf space and support.
If price, display, and advertising change together, the pilot identifies their combined effect.
It cannot isolate the effect of the product independently without a different design.

### Interpret the bridge carefully

If the new item sells $500 per store-week and the category gains $190 per store-week,
then incumbent sales are $310 per store-week below their estimated no-launch level.
In that bridge, $310 divided by $500 is 62%.
This is an estimate of net incumbent revenue displacement relative to new-item revenue.
It is not the percentage of individual shoppers known to have switched.
It can be negative if incumbents grow, or exceed 100% if the launch harms incumbents by more than its own revenue.
Such values are signals to investigate, not values to clip automatically to zero or 100%.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Acceptance | Decision to admit a proposed item subject to specified conditions; decision status | Treating supplier shipment as proof of consumer acceptance |
| Listing | Authorization to sell a SKU in a defined store or channel; binary status per location | Assuming every authorized location is live and stocked |
| SKU | Retailer's distinct stock-keeping unit; identifier | Treating two pack sizes as interchangeable units |
| GTIN | Global Trade Item Number identifying a trade item; GS1 identifier | Using a store-specific SKU as a globally unique trade item |
| Trial | Shopper's first observed purchase under a stated identity rule; shoppers or purchases | Equating first observed loyalty-card purchase with true first purchase |
| Repeat | Subsequent purchase by an identified shopper within a defined window; shoppers or rate | Counting multiple units in one trip as separate repeat trips |
| Distribution | Share or number of eligible outlets carrying an item; stores or percent | Confusing authorization, on-shelf availability, and sales distribution |
| Availability | Ability to purchase an item at the intended time; percent of observations or hours | Inferring full availability merely because some sales occurred |
| Velocity | Sales per store per unit of time; units/store-week or dollars/store-week | Comparing velocities across unequal availability or promotion |
| Baseline | Observed pre-launch outcome or model reference, as explicitly defined; dollars/store-week | Calling pre-launch sales the causal counterfactual without adjustment |
| Counterfactual | Outcome that would have occurred for treated units without treatment; outcome units | Claiming it is directly observed for the same unit at the same time |
| Treatment | The specified launch package assigned to treated stores; condition | Attributing a bundle's effect solely to the product |
| Control | Comparison group assigned or selected not to receive the launch; stores | Assuming any untreated store is automatically comparable |
| Randomization | Chance-based assignment under a documented procedure; assignment probabilities | Randomizing after stores are chosen based on post-launch performance |
| Difference-in-differences | Treated change minus comparison change; outcome units/store-week | Treating it as causal regardless of trends and other interventions |
| Parallel trends | Assumption that group outcomes would change similarly without treatment; outcome trend | Requiring identical baseline levels instead of comparable untreated changes |
| Incrementality | Difference between launch outcome and no-launch outcome; units or currency | Calling all new-SKU sales incremental |
| Transfer | Estimated reduction in incumbent sales relative to their counterfactual; currency or units | Claiming it identifies individual shopper switches |
| Cannibalization ratio | Transfer divided by new-item sales on a consistent basis; percent | Mixing unit transfer with revenue sales in the denominator |
| Category boundary | Defined set of items whose outcomes are summed; item list and effective dates | Redrawing the boundary after inspecting results |
| Revenue | Net sales after stated discounts and returns; currency per period | Using shipped wholesale value as retailer consumer sales |
| Gross profit | Net sales less cost of goods sold under stated accounting conventions; currency | Assuming gross profit covers every incremental operating cost |
| Contribution | Revenue less explicitly specified variable costs; currency | Using an undefined margin percentage across items with different economics |
| Fixed launch cost | Cost incurred because the launch happens and not varying with observed units in the horizon; currency | Charging sunk product-development spending again |
| Opportunity cost | Value of the best feasible alternative use of a scarce resource; currency | Adding an incumbent loss twice when it is already included in category contribution |
| Treatment effect | Difference in outcomes under assigned launch and no-launch conditions; outcome units | Treating an observational difference as automatically causal |
| Intention-to-treat | Effect of assignment regardless of adherence, using all assigned units; outcome units | Removing poorly executed stores after assignment |
| Spillover | Effect of one unit's treatment on another unit's outcome; outcome units | Assuming nearby control stores cannot be affected by treated stores |
| Ramp | Evolution of sales and costs following launch; time profile | Extrapolating week-one trial to the full year |
| Target | Desired performance threshold; metric and date | Presenting a target as a forecast or guarantee |
| Forecast | Conditional estimate of a future outcome; metric, horizon, and assumptions | Calling a supplier's promotional ambition an unbiased forecast |
| Plan | Chosen actions, owners, and timing; operational commitments | Treating a plan as proof that actions occurred |
| Commitment | Authorized obligation or promise; amount and responsible party | Treating an unapproved scenario as a purchase commitment |
| Incremental return on investment | Defined incremental net benefit divided by specified incremental investment; ratio | Comparing ratios that use different cost or time boundaries |
| Net present value | Sum of dated incremental cash flows discounted to a valuation date; currency | Discounting accounting margin without converting it to cash flows |
| Post-launch review | Preplanned comparison of results with evidence, economics, and decision criteria; dated decision | A sales-only scorecard prepared to justify the launch |

## 5. The mechanics, step by step

### Step 1 — State the decision and the alternative

**Input:** a supplier proposal, category objective, and candidate store population.
**Action:** write the item, proposed launch date, stores, shelf change, and decision owner.
**Output:** one sentence defining the choice and the no-launch alternative.

For example, compare listing one refrigerated snack in 20 pilot stores with retaining the existing assortment.
If the shelf is full, specify which existing SKU loses space or is delisted.
The no-launch alternative should include the same season and normal category activity.
It need not mean the store stands still: existing promotions and assortment updates may continue if preplanned.

Ask what happens if the buyer declines the proposal.
Another item might get the space, or the incumbent might keep it.
If a different candidate is the real alternative, compare the candidates directly where practical.
A launch versus status quo test does not establish superiority over the untested candidate.

**Branch:** if legal, safety, master-data, or supply prerequisites fail, do not start the commercial pilot.
Resolve the prerequisite and document the revised decision date.
Passing a prerequisite is necessary, not proof of shopper demand.

### Step 2 — Describe the customer need and the category role

**Input:** category decision tree, unmet-need evidence, and candidate positioning.
**Action:** state who might buy, on what occasion, and why current options may not suffice.
**Output:** a falsifiable customer and purchase-occasion hypothesis.

A plausible hypothesis might be that shoppers seeking a high-protein single portion lack a convenient option.
That statement should identify what is distinctive about the item and who it is intended to serve.
Check whether existing items already satisfy the need at comparable price, format, and placement.
Consider the category's role: traffic building, routine replenishment, seasonal relevance, or convenience.
The role affects how much weight to place on contribution, coverage, price image, and availability.

**Branch:** when the item merely duplicates an existing item, require a stronger economic or supply-resilience rationale.
A duplicate can still be useful, but the need must be explicit.
Do not assume that a supplier's brand awareness translates into incremental category demand.

### Step 3 — Clear the operational acceptance gate

**Input:** product specifications, supplier data, cost terms, store capacity, and replenishment constraints.
**Action:** confirm identifiers, product attributes, shelf life, ordering units, receiving, and recall readiness.
**Output:** a documented pass, conditional pass, or rejection for each prerequisite.

Check pack dimensions against shelf and case dimensions against handling equipment.
Check storage temperature and remaining shelf life at delivery.
Check minimum order quantities, lead time, fill rate history when available, and substitution rules.
Confirm the price and tax treatment used by the retailer's systems.
Confirm which party funds introductory discounts and how settlement is recorded.
Document promotional display and shelf placement rather than leaving them implicit.

An item that cannot be replenished consistently may create misleading trial results.
An item that is repeatedly unavailable may disappoint shoppers and distort the apparent repeat rate.
Condition acceptance on a named corrective action and deadline if a deficiency is remediable.
Do not make a permanent rollout commitment while critical data or product safety checks remain open.

### Step 4 — Define the treatment package

**Input:** launch plan, standard category calendar, and proposed store execution.
**Action:** specify the SKU, price, shelf position, facings, promotional support, and replenishment policy.
**Output:** a treatment specification with effective dates and permissible deviations.

State whether a display is part of the proposed rollout or only part of the experiment.
Specify whether the incumbent loses a facing, is delisted, or is moved to another location.
State which promotions also apply to control stores.
Record changes in competitor SKUs and category signage.
Measure actual implementation; a planogram is a plan, not proof of compliance.

If display support is unusually generous in the pilot, the result estimates that generous package.
It should not be presented as the expected effect of an unsupported rollout.
If management needs separate item and display effects, design multiple arms and allocate adequate stores.
Do not split a small pilot into many arms without considering the resulting loss of precision.

### Step 5 — Choose the estimand and outcome hierarchy

**Input:** business decision, feasible data, and intended period.
**Action:** define population, assignment, comparator, outcome, and time horizon before launch.
**Output:** a written estimand and primary metric.

An example estimand is the average effect of assigning the launch package to eligible stores
on net category contribution per store-week during the first four full launch weeks.
The comparator is assignment to normal assortment during the same weeks.
This is an intention-to-treat estimand: all assigned stores remain in their assigned group.

Use category revenue and new-item revenue as secondary measures for a sales bridge.
Also report availability, unit velocity, returns, waste, and compliance.
Choose a longer follow-up if repeat purchase and maturation are central to the decision.
An early pilot measures early performance, not an established steady state.

Specify the expected sign and treatment of returns, discounts, and tax.
Use the same category membership and accounting basis for both groups and all periods.
Freeze the primary metric before looking at post-launch results.
Predefine how partial weeks, store closures, and missing data will be handled.

### Step 6 — Select a credible comparison

**Input:** eligible stores, historic category data, operating calendar, and constraints.
**Action:** randomize eligible stores when feasible; stratify or pair on relevant pre-launch attributes.
**Output:** assignment list, seed or audit record, and eligibility log.

Matching can improve precision, but it does not replace random assignment.
Useful pre-launch matching variables include category sales, store format, geography, and seasonal pattern.
Do not match on outcomes measured after launch.
If stores are paired, randomize one store per pair to each arm and preserve pair identifiers in analysis.
If stores are stratified, apply the documented allocation within each stratum.
Use a common calendar and comparable observation windows.

Consider excluding stores scheduled for closure or remodel before randomization.
Record exclusions before assignment and define the resulting eligible population.
If a store becomes unavailable after assignment, retain it in the primary analysis where outcomes are observable.
Explain any missing outcomes rather than silently dropping an inconvenient store.

**Branch:** if randomization is infeasible, use a justified observational comparison.
Document selection rules, historical trends, concurrent changes, and residual confounding.
An observational difference-in-differences estimate depends heavily on the no-launch trend assumption.
Do not describe it as randomized or guarantee that adjustment eliminates bias.

### Step 7 — Plan sample size and duration

**Input:** primary outcome variation, expected effect worth detecting, store count, and assignment design.
**Action:** calculate or simulate detectable effects using store-level correlation and the actual design.
**Output:** sample-size rationale, minimum duration, and uncertainty plan.

A store-week is not necessarily an independent observation.
The same store's weekly outcomes are correlated, and promotions may create common shocks.
Use store-level variation and clustering consistent with the assignment unit.
Account for pairing or stratification in estimation and uncertainty calculations.
If there are too few stores for reliable asymptotic approximations, consider randomization inference
consistent with the actual assignment scheme, or report the limitations prominently.

Estimate variation from historical data without selecting the pilot because its post-launch result looks good.
Specify the smallest contribution effect that could change the rollout decision.
A statistically detectable effect may still be commercially trivial.
A small pilot may leave the interval compatible with both valuable growth and meaningful loss.
Predefine the review date rather than repeatedly checking until a favorable result appears.

Allow time for distribution, initial trial, and repeat purchasing where feasible.
Consider the buying cycle, not merely the length of the reporting calendar.
If launch timing is seasonal, a short pilot may not represent other seasons.
If capacity limits the number of pilot stores, admit the precision constraint before launch.

### Step 8 — Set thresholds without confusing evidence and targets

**Input:** economics, strategic objectives, risk limits, and estimated precision.
**Action:** define primary outcome, guardrails, and possible decisions before results arrive.
**Output:** a decision rule with accountable owners and escalation paths.

A guardrail might require acceptable availability and no product safety incident.
Another might limit waste or require that a strategic incumbent remain available.
A financial threshold can require positive expected contribution after relevant launch costs.
Specify what happens when the estimate is positive but too imprecise for a confident rollout.
Predefine whether the decision is to extend the pilot, adapt execution, stop, or accept managed risk.

A target is a desired outcome, such as $500 of new-item sales per store-week.
It is not evidence that the category gained $500 per store-week.
A forecast is a conditional prediction with an explicit horizon and uncertainty.
A plan specifies actions and owners; a commitment creates an authorized obligation.
Keep these labels separate in the business case and the review.

### Step 9 — Capture execution and commercial data

**Input:** point-of-sale records, inventory, planogram audits, promotion calendar, and financial terms.
**Action:** build a dated store-SKU-week panel for both assigned groups.
**Output:** a reconciled analysis dataset and documented exceptions.

Keep sales and returns on the same stated net basis.
Store item attributes with effective dates when packs, prices, or identifiers change.
Identify whether inventory is on hand, in transit, or available on the shelf.
A system inventory balance is not the same as physical shelf availability.
Collect actual prices and discounts, not only planned prices.
Separate supplier funding received from funding merely expected.

Record store closures, missing feeds, planogram compliance, and out-of-stock periods.
Record deviations for controls as well as treated stores.
A control store receiving the item is a treatment contamination event, not a reason to hide it.
Prevent post-treatment filtering based on observed sales or compliance in the primary analysis.

### Step 10 — Estimate the category counterfactual

**Input:** pre- and post-launch group means, or a store-level model consistent with assignment.
**Action:** subtract the control change from the treated change.
**Output:** estimated incremental category outcome per store-week.

Let T_pre and T_post denote treated category means before and after assignment.
Let C_pre and C_post denote the corresponding control means.
The difference-in-differences estimate is (T_post - T_pre) - (C_post - C_pre).
The implied treated no-launch post outcome is T_pre + (C_post - C_pre).
This additive form uses a common absolute change assumption.
A percentage-growth counterfactual is a different model and should not be silently substituted.

In randomized pilots, comparing post-period outcomes, possibly adjusted using pre-period outcomes,
can also estimate the assignment effect and may improve precision.
The choice of estimator should be stated in advance and match the design.
Difference-in-differences does not cure an incorrectly assigned or contaminated control group.
Inspect historical pre-period patterns for warnings, while recognizing that pre-trends cannot prove parallel trends.

### Step 11 — Bridge new-item sales to incremental sales

**Input:** treated new-item sales and estimated incremental category sales on the same basis.
**Action:** subtract the category effect from new-item sales to estimate net incumbent displacement.
**Output:** a sales bridge and clearly labeled ratio.

When the new item did not exist in the no-launch category, the accounting identity is:
new-item sales plus the change in incumbent sales equals the total category effect.
Estimated incumbent displacement equals new-item sales minus the estimated category effect.
Divide displacement by new-item sales only if the denominator is nonzero.
Use dollars with dollars, or units with comparable units; do not mix the two.

The ratio is an aggregate estimate and inherits the uncertainty of the counterfactual.
It does not identify which incumbent items were switched by which shoppers.
Item-level movements can suggest hypotheses but are not automatically causal.
A broader boundary can show category gains that a narrow definition misses, or vice versa.
Predefine a primary boundary and use broader measures as clearly labeled secondary analyses.

### Step 12 — Calculate incremental contribution and relevant cash flows

**Input:** net revenue, item cost, variable selling and handling costs, and launch costs.
**Action:** compare observed contribution with estimated no-launch contribution, then deduct incremental costs.
**Output:** pilot and rollout business cases with explicit accounting and timing.

Calculate each item's contribution using its own price and cost where possible.
Do not apply the new item's margin rate to displaced incumbent revenue by default.
Compute category contribution for treated and control groups using the same definition.
Then estimate the incremental contribution with the same comparison design used for revenue.

Add launch-specific costs not already in contribution, such as incremental installation labor.
Exclude sunk expenses that cannot change with the current decision.
Include additional working-capital and waste effects if relevant and not already captured.
Avoid counting a lost incumbent margin twice: it may already appear in category contribution.
If an alternative use of space is outside the measured category, explicitly consider its opportunity cost.

For a longer rollout, project each period's incremental contribution and incremental cash costs.
Separate one-time costs, ongoing variable costs, and recurring fixed costs.
Discount dated cash flows when the decision horizon and finance policy require it.
Show a base case and scenarios for plausible demand, margin, availability, and cost variation.
A scenario is a conditional calculation, not a confidence interval unless derived as one.

### Step 13 — Quantify uncertainty and examine heterogeneity

**Input:** store-level outcomes, assignment record, and prespecified analysis plan.
**Action:** estimate uncertainty at the assignment unit and inspect planned subgroups cautiously.
**Output:** effect estimate, interval or design-based uncertainty, and sensitivity checks.

Report the number of stores and observation periods behind each estimate.
An average can hide a mix of high-performing and loss-making store formats.
Predefine important segments such as refrigerated capacity or store format.
A subgroup with few stores can produce unstable differences by chance.
Do not declare a reliable segment effect simply because one subgroup is significant and another is not.
Compare subgroup effects directly when the design supports that comparison.

Check the sensitivity of the decision to a lower plausible contribution effect.
Consider whether conclusions change under an alternative defensible category boundary.
Inspect stockouts and execution as diagnostic context, not as permission to remove weak stores after the fact.
If no uncertainty calculation is feasible, label the result descriptive and avoid a certainty claim.

### Step 14 — Conduct the post-launch review and decide

**Input:** reconciled scorecard, causal estimate, economics, operational evidence, and risks.
**Action:** compare evidence with the predeclared decision rule and document deviations.
**Output:** scale, adapt, extend, or stop decision with owner, date, and triggers.

Scale only to the population for which a rollout case is credible.
Adapt when a correctable execution or positioning problem has a plausible remedy.
Extend when uncertainty matters and more data can reasonably resolve it.
Stop when the evidence and constraints make continuation unattractive.
A stopped launch should still produce a learning record and inventory disposition plan.

Document who approved any change from the original rule and why.
Assign owners for supplier notification, inventory, shelf reset, and financial reconciliation.
Set a later review because pilot outcomes can decay as introductory support ends.
Archive the initial forecast alongside actuals so that forecast accuracy can be evaluated honestly.

### Decision note A — Choose a counterfactual that matches the alternative

An attractive new SKU can still be inferior to the next-best use of the same shelf space.
The status-quo counterfactual answers whether the proposed package beats keeping the existing assortment.
It does not answer whether the package beats another launch, a different price, or an extra incumbent facing.
Write the alternative in the approval document before allocating inventory.
If there are two credible candidates and sufficient stores, consider comparing both with a common control.
That comparison needs enough observations for each decision-relevant contrast.
Otherwise prioritize the most important decision rather than spreading a small sample across many arms.

An incumbent may already be scheduled for delisting for an independent safety or supply reason.
In that situation, retaining it may be an infeasible counterfactual.
Compare with the feasible replacement or empty-space policy instead.
Document why the alternative is feasible and how it would be executed.
Changing the comparator after results are available makes the business case difficult to interpret.

A control store can retain the incumbent while treated stores replace it.
That is appropriate when replacement is the actual decision.
The measured effect then belongs to the combined replacement, not to a hypothetical addition with unlimited space.
If an incumbent loses facings rather than its listing, record the before and after facings.
The substitution effect may depend on the severity of the space reduction.
Likewise, moving an incumbent to a less visible shelf can change its sales without removing it.

### Decision note B — Map the assortment and demand boundaries

Create a list of all items in the primary category at the start of the analysis.
Record each item's effective dates, pack size, brand, supplier, and price basis.
Add the new SKU to the launch-period category without changing the treatment of incumbents.
Keep discontinued items in the pre-period so their earlier sales do not disappear from the comparison.
Maintain a documented mapping for barcode changes and corrected master data.
Review the mapping with the category owner and the data steward before examining the result.

The primary category should be broad enough to include plausible close substitutes.
If a new chilled snack substitutes for a shelf-stable snack outside the selected category,
the primary category lift may overstate the gain to the whole retailer.
A wider food-department outcome can reveal that risk, although it may be noisier.
Conversely, a narrow category can miss genuine complementary sales in a different department.
Both wider and narrower views can be useful if labeled as secondary.
Do not repeatedly redraw the category until a preferred result emerges.

Physical stores and online channels need consistent treatment.
If an online order is fulfilled from a treated store, decide in advance which store receives the sale.
If the item is unavailable online, specify whether online purchases belong to the comparator.
Store transfers, click-and-collect, and delivery zones can cause spillovers between assigned locations.
Record the geographic relationship between test and control stores.
When shoppers routinely use multiple nearby stores, randomizing geographic clusters may be more credible.
The effective number of independent assignments is then the number of clusters, not the number of stores.

### Decision note C — Align the pilot with the commercial calendar

Select pre- and post-periods that share a consistent calendar and operational definition.
A full store-week may run Monday through Sunday, but the exact boundary must be stated.
Exclude a partial installation week from both arms only if that exclusion was specified in advance.
Otherwise describe how partial exposure enters the estimand.
Do not allow treated stores to start on different dates without recording the actual timing.
Staggered starts require an analysis appropriate to the assignment schedule.
The simple two-period calculation in this unit assumes a common treatment window.

Consider holidays, pay cycles, weather, and advertised category promotions.
These events can affect all stores, but their effects need not be equal across formats or regions.
The control group helps account for common events only to the extent that it is comparable.
A pilot spanning a seasonal event may be appropriate if the planned rollout also spans that event.
It may be unrepresentative if the decision concerns an ordinary nonseasonal period.
Write the seasonality risk into the rollout assumptions rather than removing inconvenient weeks.

If the item is purchased monthly, a four-week pilot may contain mostly first purchases.
The team may need a longer cohort window before interpreting repeat behavior.
Conversely, a perishable high-frequency item may reveal operational waste quickly.
The appropriate horizon depends on buying cadence, shelf life, and the decision deadline.
Specify when data arrive, when costs are reconciled, and when the review can actually occur.
A review date before returns or supplier-funding settlement can produce an incomplete business case.

### Decision note D — Assign stores without selecting winners

Construct the eligible store list using criteria fixed before the random draw.
Examples include suitable refrigeration, sufficient category history, and no planned closure.
Record the number of excluded stores and the reason for each exclusion.
Store management can identify feasibility constraints before assignment.
Management should not choose the highest-growth stores after seeing the treatment allocation.

For paired randomization, group similar stores into pairs using pre-launch characteristics.
Randomly assign one store in each pair to treatment and the other to control.
Keep the pair identifier and assignment probability in the analysis file.
The assignment mechanism must be reproducible from the retained allocation record.
If an intervention is assigned by region, analyze the regional assignment structure.
Do not pretend stores within the same treated region were independently randomized.

Randomization does not guarantee perfectly balanced outcomes in a small pilot.
Review pre-assignment characteristics to understand the realized sample.
Do not rerandomize repeatedly until the post-launch effect becomes attractive.
If a rerandomization rule is used, specify its balance criteria before allocation.
Report the randomized population, the actual exposure, and the analysis population separately.
That audit trail lets reviewers distinguish assignment problems from operational noncompliance.

### Decision note E — Treat availability as execution evidence

Availability has several distinct meanings in a retail launch.
An authorized item may be listed in a system but absent from the warehouse.
Warehouse inventory may exist while the store shelf is empty.
A shelf audit can show a gap without revealing how long the gap lasted.
Point-of-sale transactions prove that some units sold; they do not prove continuous availability.
Use the most direct feasible measure and document its frequency.

Predefine an availability guardrail, such as observed shelf presence on scheduled audit visits.
Specify the numerator, denominator, timing, and treatment of missed visits.
Do not call an audit-based percentage an exact percentage of hours in stock.
Compare execution across locations and across the launch period.
If treated stores have repeated gaps, investigate ordering, lead time, shelf capacity, and delivery.
The business decision may be to fix execution rather than to reject the shopper proposition.

However, removing unavailable stores from the primary analysis changes the question.
Stores that struggle to execute may also differ in unobserved demand or operational capacity.
The intention-to-treat result describes the effect of assigning the actual launch process.
An analysis restricted to well-stocked stores is descriptive unless a defensible causal method is used.
If management wants the effect under reliably high availability, design and test that operational package.

### Decision note F — Define trial and repeat without overstating identity

The first observed transaction in a loyalty dataset need not be a customer's first purchase.
The shopper may have purchased without a card, at another chain, or before the available history begins.
Define a lookback period and call the measure first observed purchase within that period.
Specify whether the identity is a person, account, card, or household.
Include the share of transactions that cannot be linked to an identifier.

A useful repeat denominator is identified first-time buyers with enough follow-up time.
The numerator is those buyers with a qualifying later purchase in the stated window.
For example, two units on the same receipt may be one purchase occasion rather than repeat.
A four-week window and an eight-week window can produce different repeat rates.
Report both the window and the cohort date when comparing launches.
Account for shoppers who have not yet had the full opportunity to repeat.

Repeat purchase does not prove incrementality.
A loyal buyer of the new item may have previously purchased an incumbent every week.
Likewise, a low repeat rate does not necessarily make a seasonal trial unsuccessful.
Use repeat as evidence about product acceptance and demand maturation.
Combine it with the category comparison and the financial decision rather than substituting it for them.

### Decision note G — Interpret the difference-in-differences arithmetic

The additive estimator assumes that, without treatment, the treated group's expected change
would equal the comparison group's expected change on the selected outcome scale.
It does not assume that the two groups had the same starting sales level.
It also does not assume that sales would have remained constant over time.
The control group's change supplies the estimated common movement.

Consider treated and control stores with different baseline levels.
An equal absolute change may be plausible in one setting and implausible in another.
An equal percentage change is a different counterfactual assumption.
For example, if both groups grow proportionally because of a broad price change,
an additive comparison and a proportional comparison can produce different results.
Specify the model before seeing the preferred outcome and assess its business plausibility.
Do not choose a scale merely because it produces the largest lift.

Historic trend checks can reveal obvious mismatches.
Plot several pre-launch periods using the same outcome definition and calendar alignment.
Investigate different slopes, missing observations, or a promotion that affects only one group.
Failure to find a visible difference does not prove the future counterfactual assumption.
Randomization reduces reliance on matching observed historical trends, but it does not eliminate sampling variation.
Explain the actual assignment and analysis rather than invoking a method name as a guarantee.

### Decision note H — Convert accounting results to a cash decision

Start with revenue net of the explicitly chosen discounts, returns, and taxes.
Subtract item acquisition cost and other variable costs to obtain the defined contribution.
Make supplier rebates and trade funding visible as separate components when material.
Only include funding once and use the same accounting basis for treatment and comparison.
If funding is contingent on a future condition, show the condition and risk of nonpayment.

One-time setup can include planogram labor, shelf labels, and incremental system configuration.
Do not allocate ordinary fixed overhead to the decision unless it actually changes because of the launch.
On the other hand, a new recurring service fee is relevant if it will be incurred only with the launch.
Waste, markdowns, and returns must be included once on the same horizon.
If they already reduce net revenue or contribution, do not subtract the same loss again.

Inventory purchases create cash outflows before all units are sold.
Unsold inventory has a remaining value only to the extent that it can be sold, returned, or reused.
When modeling cash flow, include the purchase timing and the recoverable ending inventory consistently.
Do not deduct the same inventory cost both as cost of goods sold and as a separate cash outflow without reconciliation.
For a longer horizon, discount dated net cash flows using an approved rate and valuation date.
The short example in this unit deliberately omits discounting and working-capital timing.

### Decision note I — Set a decision rule before results

A practical rule identifies the primary financial threshold and operational guardrails.
For example, the team may require positive expected net contribution at the proposed rollout scale.
It may also require an acceptable supply service level and completion of safety checks.
These thresholds should follow the retailer's actual risk tolerance and decision rights.
They are not universal industry standards.

Specify what happens if the point estimate exceeds the threshold but uncertainty remains wide.
One option is a limited rollout with a capped financial exposure and a defined reversal trigger.
Another is an extension designed to answer a specific unresolved question.
The extension should state its added cost, duration, and expected decision value.
An extension that cannot materially change the decision may be wasteful.
Conversely, scaling a reversible, low-cost launch may be rational despite residual uncertainty.

Avoid a rule based only on a significance label.
A precise small effect can be commercially unattractive.
An uncertain large effect can be attractive or unacceptable depending on the downside.
Consider the interval, plausible costs, capacity constraints, and whether further information is obtainable.
Document any override and the decision-maker responsible for accepting the risk.

### Decision note J — Review performance over time

Separate the initial introduction period from later trading when the decision requires both.
An opening promotion can produce a one-time stock-up that depresses subsequent purchases.
An initially slow item may improve after shoppers discover it, if support and availability are sustained.
Both patterns are hypotheses until supported by a comparison with sufficient follow-up.
Plot weekly treated and control outcomes and note changes in execution.
Avoid interpreting every weekly fluctuation as an independent causal effect.

At review, compare actual price, distribution, display, and supply with the plan.
Explain material deviations and their likely implications for rollout.
Show the original forecast next to observed item sales, but separately show estimated category lift.
Compare the business case using the original assumptions and using reconciled actual economics.
Record which assumptions changed and why.
The objective is learning for the next decision, not defending the original proposal.

The review should end with a named owner and dated action.
For a scale decision, specify the store population, package, cost cap, and next monitoring date.
For an adaptation, specify the changed package and whether another controlled test is required.
For an extension, state the additional evidence sought and the predefined stopping date.
For a stop, specify stock disposition, supplier communication, and restoration of shelf space.
Keep the review record accessible to future buyers and analysts.

## 6. Worked example

### Decision and setting

A fictional regional grocery chain is considering a single-serve refrigerated protein snack.
The chain proposes to list the item in 20 eligible stores for four full weeks.
Twenty comparable eligible stores retain the existing assortment during the same four weeks.
Stores are paired on pre-launch category sales and format, then one store in each pair is randomized to launch.
The illustrative calculation below uses group means; it does not calculate uncertainty from store-level data.
The category includes all refrigerated snacks in the predeclared item list, including the new item after launch.

The treatment package includes one facing, a standard shelf label, and no special display.
The facing is taken from an existing item that remains listed.
Both groups follow the ordinary category promotion calendar.
The retailer's primary decision is whether to extend the package to similar eligible stores.
A decision about other formats requires additional evidence or an explicitly justified extrapolation.

All sales figures are net of discounts and returns, exclude sales tax, and are in US dollars per store-week.
Contribution equals net sales less item acquisition and specified variable selling and handling costs.
Contribution figures do not include the one-time launch cost described below.
The pre-period is four comparable weeks, and the post-period is four full weeks.
Assume equal observed store-weeks and no missing values for this illustration.

### Predeclared commercial statements

| Statement | Classification | Meaning |
|---|---|---|
| New item sells $500/store-week in the pilot | Forecast | Conditional prediction, not a guarantee |
| New item should reach $500/store-week | Target | Desired performance, not evidence of incrementality |
| Install one facing in each treated store | Plan | Operational action to execute and audit |
| Purchase initial inventory under an approved order | Commitment | Authorized obligation under the actual order terms |

The forecast and target have the same numerical value here only for teaching convenience.
A real forecast and target need not be equal.
The pilot's observed new-item sales happen to equal the forecast; that does not validate the whole business case.

### Category revenue comparison

| Group | Pre-period | Post-period | Change |
|---|---:|---:|---:|
| Launch stores | $2,000/store-week | $2,300/store-week | +$300/store-week |
| Control stores | $1,900/store-week | $2,010/store-week | +$110/store-week |

The treated stores increase by $300 per store-week.
The control stores increase by $110 per store-week over the same calendar period.
The additive difference-in-differences estimate is $300 - $110 = $190 per store-week.
The implied treated no-launch category revenue is $2,000 + $110 = $2,110 per store-week.
Observed treated category revenue of $2,300 exceeds that counterfactual by $190 per store-week.

This estimate relies on the comparison being appropriate for the no-launch change.
Random assignment supports comparability in expectation, but the realized sample can still differ by chance.
The table alone cannot establish statistical precision or prove absence of spillovers.
An actual review would show store-level variation and an interval consistent with the paired assignment.

### Transfer bridge

The new SKU sells $500 per treated store-week after launch.
Observed incumbent category revenue is therefore $2,300 - $500 = $1,800 per store-week.
Because the item would not be listed without the launch, the estimated no-launch incumbent total is $2,110 per store-week.
Estimated incumbent displacement is $2,110 - $1,800 = $310 per store-week.
The bridge is $500 of new-item sales minus $310 of incumbent displacement equals $190 of category lift.
The estimated revenue cannibalization ratio is $310 / $500 = 62%.
The complementary incremental revenue share is $190 / $500 = 38%.
These shares describe the estimated aggregate bridge, not observed customer switching.

If shoppers switch from another department, this category bridge does not measure that department's loss.
If the launch attracts a complementary purchase elsewhere, it does not measure that gain either.
Both effects require a broader business boundary and a credible comparison.
The estimate can also reflect changes in price and mix, not just unit substitution.

### Contribution comparison

| Group | Pre-period contribution | Post-period contribution | Change |
|---|---:|---:|---:|
| Launch stores | $600/store-week | $710/store-week | +$110/store-week |
| Control stores | $570/store-week | $603/store-week | +$33/store-week |

Estimated incremental contribution is $110 - $33 = $77 per store-week.
The implied treated no-launch contribution is $600 + $33 = $633 per store-week.
The observed $710 exceeds the implied no-launch $633 by $77 per store-week.
This calculation uses measured contribution, not a single assumed margin applied to revenue lift.

Suppose the new item contributes $150 per store-week after its specified variable costs.
Then incumbent contribution after launch is $710 - $150 = $560 per store-week.
The estimated no-launch incumbent contribution is $633 per store-week.
Incumbent contribution is therefore $73 per store-week below its counterfactual.
The contribution bridge is $150 - $73 = $77 per store-week.
The $73 contribution displacement is already in the $77 result and must not be deducted again.

### Pilot economics

Twenty treated stores observed for four weeks produce 80 treated store-weeks.
At $77 incremental contribution per store-week, estimated gross incremental contribution is $6,160.
Assume $1,000 of incremental pilot setup cost, not included in the contribution data.
Estimated pilot net benefit is $6,160 - $1,000 = $5,160 over the four-week post-period.
The setup cost averages $1,000 / 80 = $12.50 per treated store-week over this horizon.
The corresponding net benefit averages $77 - $12.50 = $64.50 per treated store-week.

The $1,000 is an assumed incremental setup cost, not a supplier invoice verified by the course.
The figure excludes any unmeasured effect on other departments, channels, or store traffic.
It also excludes future costs or benefits beyond the four-week evaluation period.
A positive point estimate is not proof that the true effect is positive.
Without store-level data, no defensible confidence interval can be calculated from the table alone.

### Illustrative rollout scenarios

Suppose 100 similar stores could launch for 12 weeks with $8,000 incremental setup cost.
At the pilot point estimate of $77 per store-week, contribution before setup would be $92,400.
After setup, the 12-week benefit would be $84,400, assuming no additional recurring costs.
At $20 per store-week, the corresponding net benefit would be $16,000.
At $0 per store-week, the corresponding net benefit would be a loss of $8,000.
These are arithmetic scenarios, not statistically estimated confidence bounds.
They assume the 100 stores can execute the same package and sustain the same effect for 12 weeks.

The break-even incremental contribution is $8,000 / (100 stores × 12 weeks), or $6.67 per store-week.
That threshold excludes other costs and any time value of money.
If the range of plausible effects includes values below the threshold, decision risk remains material.
A full cash-flow model would add inventory timing, recurring costs, and any relevant discounting.

### Review recommendation

The point estimate supports further consideration because the category and contribution bridges are positive.
It does not justify an unconditional chain-wide rollout without uncertainty and execution evidence.
Review the paired store data, availability, planogram audits, supplier funding, and spillover exposure.
If the interval and execution evidence support the predeclared decision rule, scale to similar stores.
If uncertainty is too wide and more observations are practical, extend the pilot under a precommitted design.
If the launch depends on support unavailable at scale, adapt the package and test the adapted package.
The owner should document the final choice, its evidence, its limitations, and the next review date.

## 7. Code, query, or calculation

The following Python 3 script uses only the standard library.
It reproduces the illustrative group-level bridge and the 12-week scenario.
Save it as `new_item_example.py` and run `python new_item_example.py`.
Inputs are assumed to have been reconciled and expressed on the same per-store-week basis.
The script does not perform randomization, calculate uncertainty, or establish causality.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class GroupMeans:
    treated_pre: float
    treated_post: float
    control_pre: float
    control_post: float


def difference_in_differences(means: GroupMeans) -> float:
    treated_change = means.treated_post - means.treated_pre
    control_change = means.control_post - means.control_pre
    return treated_change - control_change


def counterfactual(means: GroupMeans) -> float:
    return means.treated_pre + (means.control_post - means.control_pre)


def main() -> None:
    sales = GroupMeans(2000.0, 2300.0, 1900.0, 2010.0)
    contribution = GroupMeans(600.0, 710.0, 570.0, 603.0)
    new_item_sales = 500.0
    new_item_contribution = 150.0
    pilot_stores = 20
    pilot_weeks = 4
    pilot_setup_cost = 1000.0

    incremental_sales = difference_in_differences(sales)
    incremental_contribution = difference_in_differences(contribution)
    incumbent_sales = sales.treated_post - new_item_sales
    incumbent_displacement = counterfactual(sales) - incumbent_sales
    incumbent_contribution = contribution.treated_post - new_item_contribution
    contribution_displacement = counterfactual(contribution) - incumbent_contribution
    cannibalization = incumbent_displacement / new_item_sales
    pilot_benefit = (
        incremental_contribution * pilot_stores * pilot_weeks - pilot_setup_cost
    )

    rollout_stores = 100
    rollout_weeks = 12
    rollout_setup_cost = 8000.0
    rollout_benefit = (
        incremental_contribution * rollout_stores * rollout_weeks
        - rollout_setup_cost
    )
    break_even = rollout_setup_cost / (rollout_stores * rollout_weeks)

    print(f"Counterfactual revenue: ${counterfactual(sales):,.2f}/store-week")
    print(f"Incremental revenue: ${incremental_sales:,.2f}/store-week")
    print(f"Incumbent displacement: ${incumbent_displacement:,.2f}/store-week")
    print(f"Cannibalization: {cannibalization:.1%}")
    print(f"Contribution displacement: ${contribution_displacement:,.2f}/store-week")
    print(f"Incremental contribution: ${incremental_contribution:,.2f}/store-week")
    print(f"Pilot net benefit: ${pilot_benefit:,.2f}")
    print(f"Rollout scenario net benefit: ${rollout_benefit:,.2f}")
    print(f"Rollout break-even: ${break_even:,.2f}/store-week")


if __name__ == "__main__":
    main()
```

Expected output:

```text
Counterfactual revenue: $2,110.00/store-week
Incremental revenue: $190.00/store-week
Incumbent displacement: $310.00/store-week
Cannibalization: 62.0%
Contribution displacement: $73.00/store-week
Incremental contribution: $77.00/store-week
Pilot net benefit: $5,160.00
Rollout scenario net benefit: $84,400.00
Rollout break-even: $6.67/store-week
```

The denominator in this example is positive and fixed at $500 per store-week.
In a production implementation, handle a zero new-item denominator explicitly rather than reporting a percentage.
A production analysis should use individual store outcomes and the actual assignment record.
It should also reconcile item-level revenue and cost data to the financial ledger.
The example does not model uncertainty, taxes, cash timing, cross-category effects, or inventory losses.
Changing the inputs changes the estimate; it does not validate the counterfactual assumption.

## 8. Data, standards, and regulatory context

### Data contract

A practical analysis table needs one row per store, SKU, and week or a more granular documented grain.
Include a store identifier, SKU identifier, week start, net sales, units, returns, and item contribution components.
Keep assignment group and randomization stratum in a separate controlled assignment table.
Join on identifiers and effective dates; document changes in product definitions over time.
Include store opening, closure, and remodel dates.
Include promotion, price, stock availability, and planogram audit data when available.

The primary outcome is a category total, so reconcile the SKU list to the category hierarchy.
Freeze the hierarchy for the main analysis or document a consistent historical restatement.
A hierarchy change that puts a competing product outside the category can manufacture apparent growth.
Pack-size changes can distort unit comparisons, even when the sales-revenue bridge is internally consistent.
Consider standardized physical quantity as a secondary measure where the products are comparable.

Data quality checks should identify duplicate transactions and missing stores.
Distinguish a true zero sale from a missing point-of-sale feed.
Identify returns posted after the original purchase and define the cutoff.
Check that the same tax and discount conventions apply in every arm and period.
Ensure cost changes are effective-dated rather than overwritten with a current cost.
Do not use the retail price as a substitute for actual net revenue when discounts vary.

### Product identification

GS1 defines the Global Trade Item Number (GTIN) for identifying trade items.
Its GTIN management standard describes when changes to a trade item require a new identifier.
For the current standard, consult the official GS1 resources in Section 16.
A retailer can additionally assign its own internal SKU and location identifiers.
Neither a GTIN nor an internal SKU establishes that an item was actually available on the shelf.
Identifier standards support consistent data exchange; they do not prescribe a cannibalization estimator.

### Consumer product and food obligations

Requirements depend on product, market, and the legal entity's role.
For US packaged foods, the Food and Drug Administration (FDA) provides food-labeling resources.
Allergen labeling and other requirements may apply to the specific product and distribution channel.
The FDA's Food Safety Modernization Act resources describe preventive-control frameworks for covered facilities.
A pilot is not exempt from applicable product-safety and labeling requirements merely because it is small.
The retailer and supplier should obtain current specialist review before launch.
The course does not provide a product-specific legal determination.

### Customer data and consent

Shopper-level repeat measures can require loyalty identifiers or other personal data.
Apply the privacy law and consent requirements relevant to the jurisdiction and use.
Collect only what is needed for the stated analysis and define retention and access controls.
A household identifier is not necessarily a person, and a missing loyalty identifier is not a new customer.
Use aggregated store outcomes when customer-level identification is not necessary.
Do not infer individual switching from aggregate sales changes.

### Commercial and accounting conventions

Document whether supplier rebates, promotional funding, freight, waste, and handling are included in contribution.
Use recognized, measurable amounts and reconcile accruals with finance.
A supplier-funded introductory discount can increase retailer contribution while reducing supplier economics.
The appropriate perspective depends on whether the decision is retailer, supplier, or joint business value.
Report those perspectives separately rather than silently mixing them.
Accounting profit, operating contribution, and cash flow are related but different measures.
The business case should name its perspective, cost boundaries, horizon, and currency.

## 9. Industry practice and case evidence

### Evidence from retail field experiments

Anderson and Simester's research on field experiments discusses how firms can use controlled tests
to learn about marketing decisions in actual business settings.
The central transferable lesson is that a real-world comparison can reveal responses that simple before-and-after reporting misses.
The cited paper is not evidence that the fictional snack launch will succeed.
Implementation quality and the fit between experimental conditions and rollout conditions remain essential.

Bronnenberg, Kruger, and Mela describe the IRI marketing data set, which includes retail scanner information.
Such data can support analysis of prices, promotions, and category purchases over time.
Scanner observations alone do not randomly assign an item to stores.
A causal claim still requires a defensible assignment or comparison design.
The dataset's existence does not validate any particular transfer estimate in this unit.

### Evidence from product assortment research

Broniarczyk, Hoyer, and McAlister studied consumer perceptions of assortment and the role of a favorite brand.
Their work highlights that item count alone does not capture shoppers' perceptions of variety.
A delisting decision may matter more when it removes a shopper's preferred option.
**Inference:** a new-item pilot that records only category totals may miss dissatisfaction among a small affected segment.
That inference motivates a guardrail; it does not establish a measured effect for the fictional example.

### Evidence from causal methods

Bertrand, Duflo, and Mullainathan discuss inference problems in difference-in-differences settings
when outcomes are serially correlated.
Their analysis is a reason to respect the assignment unit and dependence structure when estimating uncertainty.
It does not mean every store-week can be counted as an independent experiment.
The exact inference method must fit the pilot's assignment and sample size.

### Retailer and supplier perspectives

Retailers often evaluate the contribution of an item in the context of a category and constrained shelf space.
Suppliers may track their own brand sales and distribution as well as retailer outcomes.
Those perspectives can conflict when an item shifts sales from another supplier without growing the category.
A joint review can show both perspectives while retaining separate definitions and ledgers.
A payment between the parties can change the division of value without creating shopper demand.
The appropriate decision depends on the contracting parties and the costs each actually bears.

### What the example does and does not establish

The fictional results demonstrate the arithmetic of a category bridge and contribution review.
They are not a named company's actual pilot results.
The 62% ratio is not a benchmark for all launches or all categories.
No universal acceptable cannibalization threshold follows from the example.
A product with substantial transfer can still be attractive if incremental contribution is positive.
A product with little transfer can be unattractive if launch costs, waste, or opportunity costs are too high.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Sales treated as lift | Review reports only new-SKU sales | No counterfactual is estimated | Transfer is mistaken for growth | Estimate category outcome against a comparison |
| Incompatible controls | Control stores have different pre-launch trajectory | Untreated changes differ | Biased effect | Randomize or justify selection and inspect history |
| Promotion confounding | Launch stores get a unique display | Treatment contains several changes | Product-only claim is unsupported | Define the package or use separate arms |
| Stockout blindness | Pilot stores have gaps in availability | Demand is censored by supply | Weak sales are misinterpreted | Audit shelf availability and report execution |
| Control contamination | New SKU appears in comparison stores | Control receives treatment | Contrast is diluted or distorted | Track assignment and exposure; retain primary assignment analysis |
| Spillovers | Nearby controls lose shoppers to launch stores | One store's treatment affects another | Estimated effect misses total market change | Design geographic clusters or measure wider area |
| Boundary manipulation | Category membership changes after results | Incumbent losses are omitted | Inflated lift | Freeze primary category definition |
| Margin averaging | One margin rate is applied to all revenue | Items have different economics | Profit estimate is biased | Calculate item-level contribution consistently |
| Double counting | Incumbent margin loss deducted twice | Category contribution already includes loss | Benefit understated | Reconcile the contribution bridge |
| Sunk-cost inclusion | Prior development spend drives continue decision | Historical cost cannot change | Future choice distorted | Include only decision-relevant future costs |
| Missing launch costs | Setup labor absent from financial case | Contribution excludes one-time expense | Rollout looks too attractive | Add incremental costs once |
| Shipped units mistaken for sales | Supplier invoice is the only measure | Inventory may remain unsold | Consumer acceptance overstated | Use net point-of-sale and inventory data |
| Selective store removal | Low-velocity stores disappear after launch | Post-treatment selection changes sample | Biased performance | Analyze all assigned stores and disclose missing data |
| Premature repeat claim | Review occurs before the purchase cycle | Buyers lack time to return | Repeat is understated or undefined | Set a cohort and adequate follow-up window |
| False precision | Store-weeks treated as independent | Within-store observations correlate | Intervals too narrow | Use design-consistent store-level inference |
| Supplier funding mismatch | Funding is promised but not booked | Expected income is treated as certain | Contribution overstated | Reconcile accrual and contract terms |
| Rollout extrapolation | Pilot had unusual support or selected stores | Treatment or population changes | Forecast fails at scale | Model the actual rollout and test transportability |
| Unplanned peeking | Team stops on the first favorable week | Many looks increase false positives | Overconfident approval | Predefine duration or a valid sequential rule |
| Pack-size distortion | Unit bridge mixes different pack sizes | A unit represents different quantity | Transfer ratio is misleading | Use revenue or standardized comparable quantity |
| Missing downstream costs | Waste and returns arrive after cutoff | Short horizon omits liabilities | Profit is overstated | Extend follow-up and accrue measurable costs |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or manage |
|---|---|---|---|
| Comparable untreated changes | Supports additive difference-in-differences | Differential local shock or promotion | Inspect pre-period patterns and documented events; randomize where feasible |
| Stable category definition | Keeps the outcome comparable | SKU hierarchy changes by period | Freeze or consistently restate the item list |
| Correct assignment record | Defines the intended contrast | Manual overrides or lost allocation file | Audit assignment against store master and launch records |
| Limited spillover | Makes store-level contrast interpretable | Shoppers switch between nearby arms | Use geographic clusters or broader outcomes |
| Consistent treatment package | Makes the effect actionable | Different price or display by store | Audit compliance and report deviations |
| Accurate contribution data | Converts sales to economics | Missing costs or incorrect funding | Reconcile with finance and effective-dated costs |
| Comparable observation windows | Avoids calendar-driven contrast | Partial weeks or different seasons | Align calendar periods and document exceptions |
| No selective outcome removal | Preserves assigned populations | Dropping low-performing stores | Predefine missing-data handling and report counts |
| New item absent under control | Supports the simple displacement bridge | Item is sold in controls or another channel | Track contamination and redefine the estimand if needed |
| Relevant costs are incremental | Makes the decision prospective | Sunk costs included or future costs omitted | Review cost causality and timing with finance |
| Pilot is transportable | Supports rollout forecast | Different stores, support, competitors, or supply | Compare populations and run sensitivity scenarios |
| Follow-up represents decision horizon | Makes pilot economics relevant | Trial spike or delayed repeat and waste | Extend follow-up and model time profile |

These assumptions do not become facts because they appear in a table.
Pre-period agreement can increase confidence but cannot prove the unobserved no-launch outcome.
Randomization supports causal inference for assignment, but finite samples still have uncertainty.
A business case can use scenarios when evidence is limited, provided they are labeled honestly.

## 12. Exercises

### Exercise 1 — Separate new sales and incrementality

**Time:** 10 minutes.
A new item sells $420 per store-week.
Estimated category lift is $105 per store-week on the same net-revenue basis.
**Deliverable:** calculate incumbent displacement and the revenue cannibalization ratio.
**Acceptance criteria:** show the bridge, units, and one limitation of the ratio.

### Exercise 2 — Write an acceptance gate

**Time:** 20 minutes.
A supplier proposes a chilled multipack with a short shelf life and minimum order quantities.
The category manager believes it serves a family occasion, but the shelf is full.
**Deliverable:** write a one-page acceptance gate and identify the alternative use of space.
**Acceptance criteria:** include customer need, operational prerequisites, economics, ownership, and a stop condition.
Distinguish a conditional pass from commercial approval.

### Exercise 3 — Design a pilot

**Time:** 30 minutes.
A chain has 60 eligible stores across three formats and can supply 30 stores initially.
A two-week introductory display is proposed but the eventual rollout may not use a display.
**Deliverable:** outline assignment, comparator, treatment, metric, duration rationale, and review rule.
**Acceptance criteria:** address format, contamination, display interpretation, uncertainty, and an intention-to-treat analysis.
Do not claim that the store count alone proves adequate power.

### Exercise 4 — Calculate a difference-in-differences bridge

**Time:** 15 minutes.
Treated revenue is $1,500 before and $1,740 after per store-week.
Control revenue is $1,400 before and $1,480 after per store-week.
The new item contributes $390 of post-period revenue per treated store-week.
**Deliverable:** calculate the counterfactual, category lift, displacement, and ratio.
**Acceptance criteria:** show each subtraction and state the additive trend assumption.

### Exercise 5 — Build a short-horizon business case

**Time:** 15 minutes.
Estimated incremental contribution is $45 per store-week for 25 stores over eight weeks.
One-time incremental setup is $2,500, not included in contribution.
**Deliverable:** calculate incremental contribution, net benefit, and break-even contribution per store-week.
**Acceptance criteria:** use the correct 200 store-weeks, identify the horizon, and avoid deducting setup twice.

### Exercise 6 — Critique a review

**Time:** 20 minutes.
A launch report drops stores with poor shelf compliance, uses shipped units as consumer sales,
and compares launch stores' post-period sales with their own pre-period sales.
**Deliverable:** identify at least four threats and recommend a remedy for each.
**Acceptance criteria:** cover selection, measurement, counterfactual, and uncertainty.
Explain why a compliance-only result cannot replace the primary assignment analysis.

### Exercise 7 — Recommend a decision under uncertainty

**Time:** 25 minutes.
A pilot estimates $12 incremental contribution per store-week.
The planned rollout breaks even at $10 per store-week, but the interval includes negative effects.
Availability is high, and the pilot package matches the planned rollout.
**Deliverable:** write a short scale, adapt, extend, or stop recommendation with a decision owner and review date.
**Acceptance criteria:** distinguish point estimate from uncertainty, weigh the value of additional information,
and name the conditions under which a different decision would be reasonable.

### Exercise 8 — Classify business statements

**Time:** 10 minutes.
Classify each statement as forecast, target, plan, or commitment:
(a) expected sales are $300/store-week under a specified price;
(b) the team aims for $350/store-week;
(c) operations will reset the shelf on 3 March;
(d) an authorized purchase order obligates the buyer to purchase 1,000 cases.
**Deliverable:** label each statement and explain one risk of confusing the labels.
**Acceptance criteria:** all four labels and the explanation are correct.

## 13. Self-check answers

### Exercise 1

Estimated displacement is $420 - $105 = $315 per store-week.
The revenue cannibalization ratio is $315 / $420 = 75%.
The bridge is $420 - $315 = $105 per store-week.
The ratio is based on a counterfactual estimate and does not identify individual shopper switches.
It is not a profit measure and can change with the category boundary.

### Exercise 2 rubric

A strong gate names the family occasion and the evidence needed to support it.
It specifies which existing product or alternative use gives up shelf space.
It checks shelf life at arrival, case quantity, replenishment, capacity, identifiers, and product safety.
It estimates incremental contribution and waste on a defined time horizon.
It identifies accountable commercial, operational, and finance owners.
It distinguishes a remediable conditional pass from approval to roll out.
A weak answer approves the item solely because its supplier forecasts high sales.

### Exercise 3 rubric

A strong design randomizes within format or otherwise documents the assignment procedure.
It retains 30 control stores under a defined no-launch package.
It specifies whether the display is part of the treatment and limits the inference accordingly.
It defines a primary category contribution outcome and a predeclared review date.
It plans for store-level uncertainty, compliance measurement, and contamination.
It recognizes that power depends on variation, effect size, design, and duration.
A strong answer either tests a no-display package or explicitly explains why the display-supported result may not transport.

### Exercise 4

Treated change is $1,740 - $1,500 = $240 per store-week.
Control change is $1,480 - $1,400 = $80 per store-week.
Estimated category lift is $240 - $80 = $160 per store-week.
The treated no-launch counterfactual is $1,500 + $80 = $1,580 per store-week.
Post-launch incumbent revenue is $1,740 - $390 = $1,350 per store-week.
Estimated displacement is $1,580 - $1,350 = $230 per store-week.
The ratio is $230 / $390, approximately 59.0%.
The result assumes the untreated treated-group change would have matched the control's $80 per store-week change.

### Exercise 5

The horizon contains 25 stores × 8 weeks = 200 store-weeks.
Estimated incremental contribution is 200 × $45 = $9,000.
Net benefit after setup is $9,000 - $2,500 = $6,500 over eight weeks.
Break-even contribution is $2,500 / 200 = $12.50 per store-week.
The result assumes no additional omitted costs or benefits and no discounting over the short horizon.

### Exercise 6 rubric

Dropping low-compliance stores selects on a post-assignment condition; retain assigned stores in the primary analysis.
Shipped units are not point-of-sale purchases; reconcile consumer sales and inventory.
A pre/post comparison lacks a contemporaneous counterfactual; add a credible control.
A point estimate without assignment-consistent uncertainty can misstate confidence; report an appropriate interval.
Further threats include promotion differences, stockouts, category changes, and supplier funding errors.
Execution-specific analyses can be diagnostic, but they do not automatically recover a causal effect among compliers.

### Exercise 7 rubric

A strong recommendation acknowledges that $12 exceeds the $10 break-even point estimate by $2 per store-week.
It also acknowledges that the interval includes losses, so the estimate does not establish profitable rollout.
An extension can be reasonable if additional data are likely to narrow decision-relevant uncertainty at acceptable cost.
A limited rollout can also be reasonable if management explicitly accepts the downside and can reverse the decision.
Stopping can be reasonable if the downside or cost of learning exceeds the expected opportunity.
The recommendation needs an owner, deadline, stopping rule, and rationale for the chosen risk.

### Exercise 8

(a) is a forecast because it predicts a conditional future outcome.
(b) is a target because it states an aspiration.
(c) is a plan because it specifies an intended operational action.
(d) is a commitment because the approved purchase order creates an obligation under its terms.
Treating the target as a forecast could cause an order based on aspiration rather than expected demand.

## 14. Interview questions

1. **Why is a new SKU's revenue not its incremental revenue?**
   A strong answer states the missing no-launch counterfactual and the possibility of substitution.
   A weak answer assumes every sale would otherwise have been lost to the retailer.
2. **How would you estimate a category effect?**
   A strong answer defines the population, treatment, comparator, outcome, and time window before choosing an estimator.
   A weak answer compares the new item's sales with zero without examining incumbent outcomes.
3. **What does the cannibalization ratio mean?**
   A strong answer gives a consistent-unit bridge and explains its dependence on the category and counterfactual.
   A weak answer calls it the observed percentage of shoppers who switched.
4. **Why can the ratio exceed 100%?**
   A strong answer notes that estimated incumbent losses can exceed new-item sales, producing negative category lift.
   A weak answer clips the result without investigating the estimate or data.
5. **How would you handle a pilot stockout?**
   A strong answer retains assigned stores in the primary analysis and reports availability as execution context.
   A weak answer silently removes every store with poor availability.
6. **What is parallel trends?**
   A strong answer describes the unobserved no-treatment change and limitations of historical trend checks.
   A weak answer says the groups must have identical baseline levels.
7. **What changes when a display is bundled with the SKU?**
   A strong answer says the estimated effect belongs to the combined package and considers rollout compatibility.
   A weak answer attributes the entire effect uniquely to product quality.
8. **How do you move from lift to value?**
   A strong answer uses item-specific contribution, relevant incremental costs, timing, and a consistent perspective.
   A weak answer multiplies all category lift by the new SKU's margin and ignores displaced margin.
9. **What would you do with a positive but imprecise result?**
   A strong answer compares the decision threshold and downside with the value and cost of more information.
   A weak answer treats the point estimate as certain.
10. **Why might rollout results differ from pilot results?**
    A strong answer discusses store mix, support, supply, seasonality, and demand maturation.
    A weak answer assumes a pilot average is a guaranteed nationwide outcome.

## 15. Key terms

- [Acceptance](#4-definitions-and-vocabulary): conditional decision to admit a proposed item.
- [Availability](#4-definitions-and-vocabulary): whether the item can be purchased at the intended time.
- [Baseline](#4-definitions-and-vocabulary): explicitly defined reference measurement.
- [Cannibalization ratio](#4-definitions-and-vocabulary): estimated incumbent displacement divided by new-item sales.
- [Category boundary](#4-definitions-and-vocabulary): defined item set included in the outcome.
- [Commitment](#4-definitions-and-vocabulary): authorized obligation, not merely an aspiration.
- [Contribution](#4-definitions-and-vocabulary): revenue less specified variable costs.
- [Control](#4-definitions-and-vocabulary): group assigned or selected not to receive the launch.
- [Counterfactual](#4-definitions-and-vocabulary): outcome that would occur without the launch.
- [Difference-in-differences](#4-definitions-and-vocabulary): treated change less comparison change.
- [Distribution](#4-definitions-and-vocabulary): defined measure of outlets carrying the item.
- [Forecast](#4-definitions-and-vocabulary): conditional prediction for a specified horizon.
- [GTIN](#4-definitions-and-vocabulary): GS1 identifier for a trade item.
- [Incrementality](#4-definitions-and-vocabulary): difference between launch and no-launch outcomes.
- [Intention-to-treat](#4-definitions-and-vocabulary): effect of assignment regardless of adherence.
- [Listing](#4-definitions-and-vocabulary): authorization to sell an item in a location or channel.
- [Opportunity cost](#4-definitions-and-vocabulary): value of the best feasible alternative.
- [Parallel trends](#4-definitions-and-vocabulary): comparable untreated changes in group outcomes.
- [Plan](#4-definitions-and-vocabulary): chosen actions, owners, and timing.
- [Post-launch review](#4-definitions-and-vocabulary): documented review against evidence and decision criteria.
- [Randomization](#4-definitions-and-vocabulary): chance-based assignment under a defined procedure.
- [Repeat](#4-definitions-and-vocabulary): subsequent purchase within a specified observation window.
- [SKU](#4-definitions-and-vocabulary): retailer stock-keeping identifier.
- [Spillover](#4-definitions-and-vocabulary): impact of one unit's treatment on another unit.
- [Target](#4-definitions-and-vocabulary): desired metric and date.
- [Transfer](#4-definitions-and-vocabulary): estimated incumbent outcome reduction relative to its counterfactual.
- [Treatment](#4-definitions-and-vocabulary): specified launch package assigned to a group.
- [Trial](#4-definitions-and-vocabulary): first observed purchase under a stated identity rule.
- [Velocity](#4-definitions-and-vocabulary): sales per store per unit of time.

## 16. Sources

1. GS1, [GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/).
   Official trade-item identifier rules; consult the current edition for product changes.
2. US Food and Drug Administration, [Food Labeling & Nutrition](https://www.fda.gov/food/food-labeling-nutrition).
   Official US food-labeling resources; applicability depends on product and circumstances.
3. US Food and Drug Administration, [Food Safety Modernization Act](https://www.fda.gov/food/food-safety-modernization-act-fsma).
   Official US food-safety framework and implementation resources.
4. Marianne Bertrand, Esther Duflo, and Sendhil Mullainathan (2004),
   ["How Much Should We Trust Differences-in-Differences Estimates?"](https://doi.org/10.1162/003355304772839588),
   *Quarterly Journal of Economics*, 119(1), 249–275.
   Discusses serial correlation and inference in difference-in-differences settings.
5. Eric T. Anderson and Duncan I. Simester (2011),
   ["A Step-by-Step Guide to Smart Business Experiments"](https://hbr.org/2011/03/a-step-by-step-guide-to-smart-business-experiments),
   *Harvard Business Review*.
   Practitioner discussion of business experiments; not evidence for the fictional example's effect.
6. Bart J. Bronnenberg, Michael W. Kruger, and Carl F. Mela (2008),
   ["Database Paper—The IRI Marketing Data Set"](https://doi.org/10.1287/mksc.1080.0419),
   *Marketing Science*, 27(4), 745–748.
   Describes a retail scanner dataset and its research uses.
7. Susan M. Broniarczyk, Wayne D. Hoyer, and Leigh McAlister (1998),
   ["Consumers' Perceptions of the Assortment Offered in a Grocery Category: The Impact of Item Reduction"](https://doi.org/10.1509/jmkr.35.2.166),
   *Journal of Marketing Research*, 35(2), 166–176.
   Evidence about assortment perceptions, not a universal launch-effect estimate.
8. Paul R. Rosenbaum and Donald B. Rubin (1983),
   ["The Central Role of the Propensity Score in Observational Studies for Causal Effects"](https://doi.org/10.1093/biomet/70.1.41),
   *Biometrika*, 70(1), 41–55.
   Observational adjustment requires assumptions and does not solve unmeasured confounding automatically.

## 17. Links

- [Module 17 index](_index.md).
- [Assortment decision stack](01_assortment-decision-stack.md).
- [Category roles and decision trees](03_category-roles.md).
- [Space elasticity and planograms](06_space-elasticity.md).
- [Module 17 survey and lab sections](../../../course/17_assortment_category_retail_analytics.md).
