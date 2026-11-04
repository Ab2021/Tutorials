# Space Elasticity and Planograms

> Module 17 — Assortment, Category, and Retail Analytics  
> Topic 06 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Module 04, Module 16, topics 01 and 03  
> Version 2026.5

## 1. Why this topic exists

A shelf is a scarce physical asset and a customer-facing service system.
A category manager must decide how much of it each item receives, where it sits, and whether the return justifies the labor and inventory required to keep it filled.
The allocation affects more than visibility.
It changes the number of units that fit on the shelf, the frequency of replenishment, the probability of an empty shelf, and the substitutes a shopper sees nearby.
A visually appealing drawing that cannot be replenished or physically built is not an executable plan.

The expensive mistake is to give more space to a fast-selling item merely because it is fast-selling and then attribute its continued success to that extra space.
High demand may have caused the allocation in the first place.
The expensive opposite mistake is to give every item the same number of facings when some will empty between deliveries while others tie up cash for weeks.
Both errors confuse a descriptive sales-per-foot metric with the incremental value of a particular allocation.

The commercial decision is usually comparative: move one facing from item B to item A, preserve the current plan, or pay for a larger reset.
The relevant question is what changes across the affected category and shopping trips, not how many extra units A sells in isolation.
A more prominent A can take sales from B, reduce empty-shelf losses, or change the size of the total category.
It may also impose additional store labor, stock, waste, and implementation costs.

This topic treats the planogram as an operational specification rather than a picture.
It separates a forecast of demand response from a proposed allocation, an economic target, and an approved implementation commitment.
It explains how to estimate and test response without pretending that observational correlations prove causation.
The numerical business case is illustrative, not a claim about any named retailer or a universal elasticity.

A category manager owns the proposition and desired commercial outcome.
The space planner owns the physical layout and versioned files.
The store team validates access, capacity, replenishment, and execution.
The analyst designs measurement and checks the economics.
Finance validates contribution and cost definitions.
Merchandising leadership decides whether the uncertain return merits a pilot or rollout.

## 2. Learning outcomes

After studying this topic you can:

1. **Calculate** facings, linear space, shelf capacity, days of supply, and the trade-off between two products; exercises 1 and 2 assess the calculations.
2. **Interpret** a local space elasticity without treating it as a constant causal law; exercises 3 and 4 assess interpretation.
3. **Design** a feasible planogram that respects physical, operational, and shopper constraints; exercise 5 assesses the design.
4. **Compare** incremental category contribution with reset, inventory, labor, and waste costs; exercise 6 assesses decision economics.
5. **Specify** a randomized or otherwise defensible measurement design with execution, stockout, and substitution measures; exercise 7 assesses the design.
6. **Critique** a vendor space-response claim and distinguish a forecast from an approved commitment; exercise 8 assesses the critique.
7. **Prepare** a practical approval, field verification, and rollback process; exercise 9 assesses governance.

A correct recommendation may be to preserve the existing layout, run a pilot, or reject an infeasible proposal.
Showing a positive forecast alone does not establish that a rollout is justified.

## 3. The core mental model

```text
Shopper mission + category role + store/fixture constraints
                            |
                            v
       Candidate: product, facing count, position, adjacency
                            |
              +-------------+-------------+
              v                           v
      Visibility / choice             Shelf capacity
              |                           |
              v                           v
      Consideration / switching    Availability / replenishment
              +-------------+-------------+
                            v
               Item and category demand
                            |
                            v
 Contribution - handling - waste - reset - inventory carrying cost
                            |
                            v
       Pilot -> verify execution -> estimate incremental effect
                            |
                 approve, revise, or roll back
```

Read the upper half as a set of mechanisms, not a guarantee.
Extra facings can improve visibility, make a product easier to find, and hold more units on the shelf.
Additional capacity can prevent shelf outages between replenishment visits.
However, extra exposure can also simply move customers from another item or have little effect once the item is easy to find.
A facing placed at a different height changes both space and position, so the effects cannot automatically be separated.

Read the lower half as a business decision.
The correct comparison is the proposed allocation against a specific feasible counterfactual, over a defined time horizon.
The category-level result includes displaced items and stockout-related substitution where those effects can be measured.
A broader retailer outcome may matter when products shift sales to another category or influence store choice; do not assert those benefits without evidence.

A planogram is the intended arrangement.
Executed shelf state is what shoppers actually encounter.
The two differ when a reset is late, labels are missing, an item is out of stock, or a store substitutes products without authorization.
Record both so that a weak observed response can be interpreted.
A compliance measure is a process measure, not proof of incremental profit.

A marginal decision concerns the next feasible unit of space, not the item's average productivity.
A facing added to a crowded premium row may cost more valuable space than one added to an underused bottom row.
For a fixed fixture, a facing is rarely free: identify exactly what is displaced.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Stock-keeping unit (SKU) | Retailer-defined item or sellable variant; identifier | Treating pack sizes or store assortments as interchangeable |
| Facing | One visible front presentation of an item; count | Assuming all facings have the same width or capacity |
| Linear shelf space | Horizontal width allocated to an item; centimeters (cm) | Counting fixture width as fully usable without dividers and clearances |
| Shelf capacity | Units that physically fit in assigned shelf positions; units | Equating theoretical capacity with sellable, accessible stock |
| Depth | Number of units stored behind a front unit; units per facing | Assuming depth is the same for every package orientation |
| Shelf height | Vertical position and clear usable height; cm | Treating eye level as identical for all shoppers or fixtures |
| Planogram | Versioned specification of the intended product layout | Assuming the file is proof of implementation |
| Bay | A defined fixture section, often bounded by uprights; identifier and width | Assuming all bays within a store have identical usable dimensions |
| Reset | Physical change from one layout to another; event and labor hours | Ignoring disposal, labeling, store disruption, or follow-up |
| Space elasticity | Percentage change in demand associated with a 1% change in space, locally and under stated conditions; dimensionless | Treating a correlation as a causal and portable coefficient |
| Arc elasticity | Ratio of midpoint percentage changes across two observations; dimensionless | Applying it when price, position, stock, and other conditions changed |
| Own-space effect | Change in an item's demand caused by changing its own space | Calling the increase incremental category demand |
| Cross-space effect | Change in one item's demand caused by changing another item's space | Assuming every displaced unit becomes a lost category sale |
| Cannibalization | Sales transferred from another item because of the intervention; units or currency | Estimating it from item trends alone |
| Availability | Whether a shopper can buy the item at the relevant location and time; proportion | Using a daily inventory balance to prove shelf presence throughout the day |
| On-shelf availability (OSA) | Product available on the shelf when a shopper seeks it; proportion under a defined sampling rule | Equating backroom inventory with shelf availability |
| Stockout | An item unavailable at a defined point and time | Ignoring partial-day outages or substituting store inventory for shelf condition |
| Days of supply | Shelf units divided by expected daily units, under stated assumptions; days | Calling it a service guarantee when demand and replenishment vary |
| Replenishment interval | Time between effective shelf fills; hours or days | Using delivery frequency when the backroom is not promptly worked |
| Compliance | Share of audited features matching the approved layout; defined percentage | Reporting only stores that passed inspection |
| Contribution | Revenue less explicitly included variable costs; currency per unit or period | Treating gross revenue or an undefined margin as incremental profit |
| Incremental effect | Outcome under a change minus the outcome under its counterfactual; units or currency | Comparing only before and after during a promotion |
| Counterfactual | Outcome that would have occurred for the same unit without treatment | Treating it as directly observable |
| Treatment | The layout or policy assigned to a test unit | Calling a mixed price, promotion, and space change a pure space test |
| Control | Comparison unit assigned or selected not to receive the intervention | Allowing unrecorded control resets while claiming an unchanged comparison |
| Intention to treat (ITT) | Effect of assignment regardless of actual execution | Excluding noncompliant stores after assignment without reporting selection |
| Payback | Time required for accumulated net benefit to cover initial investment; periods | Ignoring recurring costs, uncertainty, or the decision horizon |
| Inventory carrying cost | Period cost of holding additional average inventory, under an explicit rate | Charging the full stock purchase cost as a recurring expense and again as working-capital cost |
| Global Trade Item Number (GTIN) | GS1 identifier for a trade item, where applicable | Assuming a GTIN identifies a specific shelf position |
| Global Location Number (GLN) | GS1 identifier for a location or party, where used | Assuming a location identifier specifies a fixture geometry |

## 5. The mechanics, step by step

### Step 1: Define the decision and the alternative

Write the decision in a form that can be approved or rejected.
For example: move one facing from B to A in a specified set of stores for a 12-week pilot, with price and promotion unchanged.
The alternative might be the existing layout, a different facing exchange, or a fixture expansion.
A no-change alternative must still include expected replenishment and stockout conditions.

Inputs include category objectives, store list, fixture constraints, financial hurdle, and decision date.
Output is a decision brief naming the two feasible states being compared.
Do not write “maximize sales” without specifying whether category, store, or item sales are the outcome.
A destination category may justify a service guardrail even where immediate category contribution is not the sole goal.
Document who can authorize a commercial trade-off and who can reject a layout as unsafe or unworkable.

### Step 2: Define products, stores, and time

Identify each SKU, package orientation, unit of sale, and valid date range.
Use stable item identifiers and maintain a mapping when packs or codes change.
Define the store population, fixture variant, baseline period, proposed reset date, and measurement horizon.
Separate seasonal stores, promotional displays, and recently remodeled sites where their exposure differs materially.

Inputs are item master, store master, fixture inventory, transaction history, promotion calendar, and reset records.
Output is a panel at a stated grain, such as store × SKU × week, plus a separate planogram-effective-date table.
Week-level sales can be adequate for a randomized outcome comparison but too coarse to diagnose hour-level shelf outages.
Document the aggregation and time-zone convention.
Avoid joining a current product hierarchy to historical sales without recording when classification changed.

### Step 3: Measure actual usable space

Measure the inside width of each shelf after accounting for uprights, dividers, label strips, and safety clearances.
Measure usable height and depth, not just the exterior fixture dimensions.
Record whether shelves are straight, angled, refrigerated, locked, or subject to other operational restrictions.
Check whether every store in a proposed cluster really has the same fixture.

Measure each sellable package in its intended orientation.
Use the widest practical dimension, including any shelf-ready packaging lip or divider allowance that consumes space.
For one row, verify that the sum of item widths times facing counts, plus occupied hardware and tolerances, does not exceed usable width.
A physical fit check is necessary even if a planning application reports that the drawing fits.
Packaging variability, fixture damage, and local store practices can make the nominal geometry unreliable.

### Step 4: Calculate capacity and replenishment exposure

A simple capacity calculation is facings × units deep × stackable units high.
Only multiply by height when stacking is safe, allowed, and actually used.
Cap the calculated quantity at a practical fill level if staff cannot access the back units or the product must not be stacked.
Record capacity in individual selling units, not cases, unless the entire calculation uses cases consistently.

At an expected 8 units per day and 24 shelf units, nominal shelf coverage is 3 days.
That figure assumes demand is stable, the shelf starts full, and all units are accessible.
If the store sells 12 units during a weekend peak and replenishes only twice per week, an average-based coverage figure can conceal a Saturday outage.
Compare shelf capacity with demand over the actual interval between shelf fills, allowing for day-of-week and delivery variability.
Backroom inventory prevents neither a shelf outage nor a lost sale if no one replenishes it in time.

For perishables, test whether expected sell-through occurs within remaining shelf life.
For heavy goods, validate lifting limits, shelf loading, and access with store operations.
For very slow items, a full shelf may create unnecessary inventory or expiry risk.
Do not set a universal minimum facing without checking both shopper coverage and operational consequences.

### Step 5: Establish the observed baseline

Calculate item and category units, revenue, contribution, in-stock measures, and sales per allocated unit of shelf length.
Show numerator, denominator, period, and store population for each metric.
For example, weekly sales per meter is weekly sales divided by allocated meters, not total store sales divided by total category fixture width unless that is the declared measure.
If an item is missing for half the week, its observed sales per meter does not reveal its unconstrained demand.
Use availability data to flag, rather than automatically repair, biased observations.

Record prices, promotions, distribution, competitor conditions if observable, fixture location, and display exposure.
Identify changes around the baseline: product launches, price changes, store closures, and delivery failures.
A baseline describes what happened under existing conditions.
It is not a forecast of the causal effect of changing the shelf.

### Step 6: Form the causal question

Specify treatment precisely: which item gains which facing, which item loses space, and whether height, adjacency, price, stock, or replenishment also change.
A policy that simultaneously increases facings and reduces shelf outages can be useful even if the individual pathways cannot be separated.
Label it an effect of the combined policy, not a pure visual exposure effect.
A study designed to isolate the exposure mechanism must hold or independently vary inventory and replenishment conditions.

The primary estimand might be average incremental category contribution per assigned store-week over the pilot.
Secondary estimands can include item sales, OSA, waste, labor, and customer complaints.
State whether the estimate targets all eligible stores, a particular fixture cluster, or only compliant stores.
The appropriate unit of assignment is often the store because employees and shoppers share one shelf.
Where neighboring stores share customers or distribution changes spill over, account for that in the design.

### Step 7: Design an experiment when feasible

Select eligible stores before observing outcomes and stratify by relevant baseline features, such as fixture type, category volume, and region.
Randomly assign comparable stores to the proposed layout or current layout within strata.
Document the randomization seed, allocation, start date, and any exclusions before implementation.
Use the same measurement window for both arms and retain assigned stores in the primary ITT analysis.

Keep prices, promotions, range, and other merchandising policies comparable when the question concerns space.
If they cannot be held fixed, record them and interpret the test as the combined intervention or prespecify a valid factorial design.
Avoid allocating the treatment only to stores with cooperative managers; that would confound layout with execution capability.
Avoid changing the test after inspecting early sales without a prespecified sequential procedure.

Estimate the difference in average category contribution between assigned arms over the defined period.
A baseline-adjusted analysis can improve precision when the adjustment is prespecified and appropriate.
Use store-level variation or a cluster-aware method for uncertainty; individual transactions from the same store are not independent replications of the layout.
Budget the sample and duration using historical store-level variability and a minimum economically important effect.
If the detectable effect is too large to inform the decision, say so before running the pilot.

### Step 8: Use observational evidence carefully

Randomization may be impractical during a chain-wide reset.
A controlled before-and-after comparison can use similar stores with unchanged layouts, provided the comparison was chosen without future outcome information.
Difference-in-differences compares the treatment group's before-to-after change with the control group's before-to-after change.
Its causal interpretation depends on a credible parallel-trends assumption and no important concurrent differential changes or spillovers.
Pre-period trends and knowledge of store operations help assess plausibility but cannot prove the assumption after the fact.

High-selling stores are often granted more facings, creating reverse causality.
A regression of sales on facings can therefore reflect managerial selection, seasonality, local demand, and stock availability rather than a response to space.
Store and time fixed effects may remove some stable differences and common shocks but cannot remove all time-varying confounding.
A natural experiment or instrument requires an independently defensible source of variation and an exclusion argument, not merely a convenient variable.
Present observational estimates as conditional evidence with sensitivity checks and a credible range.

### Step 9: Calculate and interpret elasticity

For small changes near a baseline, elasticity is approximately percentage change in demand divided by percentage change in allocated space.
For two discrete positive observations, the midpoint or arc form is:

```text
arc elasticity = [(Q2 - Q1) / ((Q1 + Q2) / 2)]
                 / [(S2 - S1) / ((S1 + S2) / 2)]
```

Here Q is demand in units per comparable period and S is the chosen space measure, such as cm or facing count for a single unchanged package.
The ratio is dimensionless.
The denominator cannot be zero; midpoint percentage change is also unsuitable where the relevant average is zero.
Use comparable periods and clearly identify whether the demand change is observed, modeled, or causally estimated.
An elasticity of 0.20 means a local 1% space increase is associated with approximately 0.20% higher demand under the model's conditions.
It does not mean that a 100% increase yields exactly 20% more sales, nor that every product shares the coefficient.

An arc computed from raw before-and-after sales is a descriptive response, not automatically causal.
Price, promotion, shelf height, season, availability, and competitive activity can all change the numerator.
When a facing is a large discrete change, calculate candidate-specific expected sales and costs instead of relying on infinitesimal approximations.
Report uncertainty and test alternative response assumptions, especially near a decision threshold.

### Step 10: Generate feasible candidates

Start with the current layout and generate a small set of understandable moves.
A move specifies the SKU gaining space, the SKU losing it, exact shelf position, and any adjacency change.
Include the no-change option and any necessary fixture-specific variants.
Do not assume that equal facing counts use equal width: a 12 cm package and a 9 cm package are different physical resources.

Enforce required range, minimum facings, usable shelf width, height, depth, weight, and replenishment constraints.
Consider accessibility, shelf labels, rotation, cold-chain requirements, merchandising agreements where applicable, and local safety rules.
Treat mandatory constraints separately from preferences such as brand blocks or color transitions.
An attractive but infeasible candidate is not a candidate for economic ranking.
Have store operations inspect edge cases, including mixed fixtures and shelf-ready packaging.

### Step 11: Estimate demand and substitution

For each candidate, estimate changes in units for the gaining item, displaced items, and other affected items.
Use randomized results where available, supplemented by product-specific and store-specific evidence.
Do not simply add the gaining item's increase to category sales.
If shoppers substitute within the category, subtract the contribution lost on the displaced products.
If a product disappears from shelf because its capacity falls below replenishment needs, model or measure that availability effect separately.

Track the outside option: shoppers may buy nothing in the category, return later, visit another store, or purchase a substitute elsewhere.
Transaction data alone may not reveal the outside option or the number of shoppers who searched unsuccessfully.
A customer survey, shelf observation, or controlled test may help, subject to its own selection and privacy limits.
Where evidence is too weak, bound the result with scenarios rather than inventing precision.

### Step 12: Convert volume into decision economics

Multiply each affected SKU's incremental units by its appropriate contribution per unit.
The contribution definition must exclude costs that are subsequently entered as separate incremental costs, avoiding double counting.
Add incremental labor, spoilage, replenishment, and other recurring effects for the same period.
Include the cost of additional average inventory using an agreed carrying rate over the same horizon.
Treat initial reset and label expenses as one-time cash costs.
If products are purchased for inventory, distinguish cash tied up in stock from the opportunity or financing cost of holding it.

Sum the period-level net effect and compare it with the initial investment.
Discount future effects when the decision horizon and finance policy warrant it.
For a short pilot, a simple undiscounted calculation may be appropriate if explicitly stated.
The result is a forecast based on input assumptions, not an earned benefit or an approved commitment.
Use a low, central, and high case and determine which assumptions would change the recommendation.

### Step 13: Approve and execute

Give approvers the candidate layout, economics, service and safety guardrails, and uncertainty.
Specify the decision owner, planned stores, installation window, budget, and conditions for a stop or rollback.
A target is the desired outcome; it is not a forecast merely because it appears in the business case.
A plan is the intended action; a commitment requires an authorized owner and approved resources.

Publish a versioned layout with fixture dimensions, SKU mapping, facing counts, orientation, shelf coordinates, labels, and effective dates.
Provide store teams with a change list, expected labor, escalation contact, and instructions for missing items or nonstandard fixtures.
Do not authorize local substitutions that change the treatment without a documented exception.
If a reset cannot be implemented safely, the store should escalate rather than force the product into the drawing.

### Step 14: Audit and learn

Verify installation promptly with store observations, photos where permitted, or an independent shelf audit.
Record audit time, assigned version, actual facing counts, positions, shelf stock, and deviations.
Sample audits on a prespecified basis; auditing only high-performing stores inflates apparent compliance.
Keep assignment, implementation, and audit data distinct so an ITT estimate remains possible.

Review outcomes at the agreed cadence and compare category results with the control and guardrails.
Investigate outages, price changes, delivery disruptions, and other events without deleting inconvenient observations.
Document exceptions and any analysis changes made after seeing results.
At the end, decide whether to roll out, modify, extend testing, or revert, naming a decision owner and date.
Archive the old and new planograms, measurement definitions, financial assumptions, and lessons for the next reset.

### Additional technique: compare feasible marginal moves

Suppose a planner has three possible moves, each taking one facing from an existing item.
The planner should list the exact change rather than rank items by current sales per meter.
The following synthetic one-store, one-week comparison illustrates why.

| Move | Gain in product contribution | Contribution lost on displaced products | Other recurring cost | Net recurring effect |
|---|---:|---:|---:|---:|
| A gains a facing from B | USD 7.00 | USD 3.30 | USD 0.55 | USD 3.15 |
| C gains a facing from D | USD 5.00 | USD 0.80 | USD 0.40 | USD 3.80 |
| E gains a facing from F | USD 8.00 | USD 6.20 | USD 0.30 | USD 1.50 |

The values are forecasts per store-week, not measured results.
On recurring benefit alone C's proposed exchange ranks first.
E's gross gain is largest, but its displaced contribution is also largest.
This table is not enough to choose a rollout because the one-time cost, horizon, and feasibility of each move are missing.
If C's reset costs USD 80 per store and A's costs USD 22, the preferred option can reverse over a short horizon.
At 12 weeks, C produces USD 3.80 × 12 − USD 80 = −USD 34.40 per store.
At the same horizon, A produces USD 3.15 × 12 − USD 22 = USD 15.80 per store.
The comparison assumes both changes start immediately and that their recurring effects persist for all 12 weeks.
If C can be implemented as part of a reset already funded for another purpose, count only the genuinely incremental cost.
Avoid setting that cost to zero merely because it is paid from another department's budget.
Store labor still has an opportunity cost when the same employee must defer other work.

When package widths differ, a one-facing exchange may not fit.
Suppose C is 12 cm wide and D is 9 cm wide.
Replacing one D facing with one C facing needs 3 additional cm.
If only 1 cm of genuine slack exists, the candidate is infeasible without another adjustment.
Removing two D facings would free 18 cm, but it is a different treatment with a different demand and service consequence.
In that case, recalculate D's capacity and expected sales rather than doubling the effect of the first facing mechanically.
Space-response functions can have thresholds: the last remaining facing may be much more valuable than the second.

### Additional technique: identify capacity-driven response

A larger shelf allocation can produce two separable operational observations.
First, the shelf might remain visible and buyable for more shopping hours.
Second, the product might attract more attention even when both layouts are fully stocked.
The first is an availability pathway; the second is an exposure pathway.
Both can have economic value, but they call for different operational remedies.

For example, assume a product sells 8 units per day on average.
Its existing shelf holds 12 units and the practical shelf-fill interval is two days.
The 16-unit expected interval demand exceeds capacity by 4 units.
The store may still have stock in the backroom, but shoppers can encounter an empty shelf.
A 2-facing layout holding 20 units might reduce that exposure if staff fill it as planned.
Alternatively, an additional shelf-fill visit might solve the problem without using more space.
Compare the cost and feasibility of both interventions.
Do not assume the second visit occurs merely because a procedure says it should.

The expected demand of 16 units per interval is not a guaranteed maximum.
On a peak day, demand can exceed the average and exhaust even the larger shelf.
If the expected fill interval varies, use the distribution of demand between actual fills where data permit.
If only averages are available, present a conservative scenario and an availability monitoring plan.
For a perishable item, test whether the larger shelf also increases expiry risk.
The choice depends on the balance between avoided outages, increased stock, and labor.

To estimate a visibility response separately, compare layouts with different visible facings while maintaining comparable on-shelf availability.
That may require extra replenishment for the smaller-space layout, and that replenishment itself has a cost.
The resulting effect answers a narrower question than the total effect of a normal operational rollout.
Alternatively, compare the complete policies and deliberately accept that the pathways are combined.
Choose the estimand before inspecting the result.
The decision maker may prefer the combined policy effect if the real decision includes both space and standard replenishment.

### Additional technique: identify what the test estimates

Consider 20 stores assigned to a new plan and 20 assigned to the existing plan.
Suppose 17 treatment stores complete the reset and 3 do not.
Suppose one control store implements the new plan by mistake.
The assignment contrast still compares all 20 assigned treatment stores with all 20 assigned control stores.
It estimates the effect of offering and attempting this rollout under the execution conditions of the test.
That effect can be smaller than the effect of perfect installation because implementation is incomplete.
It is nevertheless relevant to a rollout that is likely to face similar execution problems.

Comparing only the 17 treatment stores that completed the reset with the 19 compliant controls answers a different and potentially biased question.
The stores that execute may have better staffing, more reliable deliveries, or more engaged managers.
Those factors can affect contribution independently of the layout.
Keep compliance analysis as a descriptive operational diagnostic unless a stronger causal design justifies another interpretation.
Report the number assigned, installed, audited, missing, and rolled back in each arm.
Report reasons for missing audits and whether missingness differs between arms.

A test's treatment can be defined as a single layout or a rule for choosing a layout.
If a different facing exchange is used for each fixture type, the intervention is the set of assignments under that rule.
That may be the actual policy the retailer intends to scale.
However, a pooled policy effect does not automatically reveal the effect of every individual exchange.
Prespecify subgroup comparisons and acknowledge that they are less precise than the overall comparison.
Do not select only the best-looking subgroup after the test and present it as a confirmed finding.

### Additional technique: planogram coordinate and capacity contract

A shelf specification needs a consistent coordinate system.
For example, define bay number from left to right as the shopper faces the shelf.
Define shelf number from the floor upward, or choose another convention and document it.
Specify whether horizontal position is measured to the left edge, center, or another anchor of the item block.
Identify the unit of length and the origin.
Without these choices, two technically correct files can describe different physical locations.

For each SKU record the number of facings, facing width, orientation, depth, permitted stack height, and number of display units per case where relevant.
Record whether a facing is an individual pack or an entire shelf-ready tray.
If a tray holds multiple sellable units across its front, one tray and one sellable-unit facing are different counts.
The capacity formula must use consistent units for both front count and depth.
For irregular packages, use a conservative physical measurement rather than multiplying idealized catalog dimensions.
Where a product can be placed sideways, store that orientation explicitly instead of silently changing its width.

A planogram change request should include a before-and-after table.
List each affected item, existing and proposed location, facing count, and practical capacity.
Include the net width difference for each shelf.
List the items that disappear from the shelf and any new labels or shelf dividers required.
For a refrigerated shelf, identify the temperature zone and the operational approval needed before installation.
For a heavy product, identify the relevant fixture-load review.
If no valid plan fits a particular fixture, mark the store as an exception rather than altering the approved treatment silently.

### Additional technique: calculate a category-level experimental contrast

Assume, purely for illustration, that all 20 treatment stores and all 20 control stores report 10 complete weeks.
After exclusions specified before assignment, the mean weekly category contribution is USD 510 per assigned treatment store and USD 500 per assigned control store.
The unadjusted assignment contrast is USD 10 per store-week.
This is a difference in levels, not automatically a net profit figure.
If the outcome excludes incremental store labor and reset costs, those costs remain to be subtracted.
If costs are already included in the outcome, subtracting them again double counts them.

Suppose treatment stores averaged USD 485 per store-week in the baseline and controls averaged USD 480.
The treatment change is USD 25 per store-week and the control change is USD 20.
The difference in changes is USD 5 per store-week.
This descriptive difference-in-differences calculation adjusts for the observed baseline level difference.
It does not by itself prove parallel trends or repair an implementation error.
In a randomized design, a prespecified baseline-adjusted estimator can improve precision without relying on the same identification story as an observational comparison.
Use a method appropriate to the actual allocation and report uncertainty across independent stores.

If one treatment store has an unusually high result because a nearby competitor closed, inspect and document the event.
Do not delete it simply because it weakens or strengthens the desired conclusion.
An analysis plan can prespecify sensitivity to extreme observations, provided the primary analysis remains transparent.
Similarly, report an interval around the estimated effect and translate its bounds into business consequences.
An interval spanning both a large loss and a large gain supports further learning, not a confident rollout decision.

### Additional technique: handle space–inventory trade-offs over time

Space, inventory, and labor use different time bases.
Shelf space is a physical constraint at each instant.
Inventory carrying cost depends on average incremental value held over time.
Reset cost is incurred when the change is installed.
Contribution and recurring labor are usually measured over weeks or months.
Bring them to a common decision horizon before comparing alternatives.

An extra six-unit shelf capacity does not prove average inventory rises by six units.
The store may move units from the backroom to the shelf without changing total store inventory.
It may also change order quantities, delivery frequency, or safety stock.
Measure or forecast the change in total average inventory value rather than simply pricing the extra shelf capacity.
If the inventory increase is temporary during installation, use its actual expected duration.
If the business uses an annual carrying rate, convert it to the period under analysis consistently.

Working capital is a cash requirement; carrying cost is an expense or economic charge for holding capital and related risks.
Do not ignore cash constraints merely because the forecast includes a carrying cost.
Conversely, do not treat the recoverable stock balance as a permanent expense unless it is written off or otherwise lost.
At the horizon end, specify whether stock is sold, transferred, returned, or left as ongoing inventory.
When shrink or expiry is expected, include the incremental loss once under a documented convention.

### Additional technique: document a rollout decision

The approval package should state the proposed population and the reason it is eligible.
It should show the current and proposed layouts and a fixture-specific fit check.
It should state the outcome expected under no change and under the candidate.
It should show the expected incremental units and contribution for every material displaced item.
It should separate one-time installation cost from recurring operating cost.
It should disclose the average inventory value and rate used for carrying cost.
It should include a sensitivity range, the decision horizon, and an explicit no-change comparator.

The execution plan should name the central owner and the store contact.
It should identify product and label availability before the reset date.
It should say how a store reports a nonstandard fixture or a missing item.
It should specify the audit sample and the evidence needed to close the work order.
It should name who may pause or reverse an unsafe or commercially harmful change.
It should also say when the final evaluation will be available and who must make the next decision.

If a test fails a service guardrail, investigate whether the cause is the layout, stock supply, staff execution, or another event.
Do not automatically declare that shelf space has no effect.
If the economic effect is positive but the fixture cannot be built reliably, the present policy is not ready for rollout.
A redesigned candidate may be worth testing, but it is a new decision with a new expected cost and benefit.
If the measured result is close to break-even, ask whether the precision supports a decision at the required hurdle.
The analysis should preserve unfavorable outcomes as well as successful examples.

## 6. Worked example

### Decision and baseline

A regional grocery chain is considering a facing exchange in 40 eligible stores with the same 120 cm usable shelf width.
The category is shelf-stable breakfast cereal.
Products A and B are adjacent and their packages are each 10 cm wide in their normal orientation.
Other items occupy 70 cm, leaving 50 cm for A and B.
The current plan assigns A 2 facings and B 3 facings; the proposed plan assigns A 3 and B 2.
Both plans fit exactly: 2 × 10 + 3 × 10 + 70 = 120 cm, and 3 × 10 + 2 × 10 + 70 = 120 cm.
The calculation assumes the stated 120 cm already excludes hardware and tolerances.
There is no new fixture, shelf-height change, or planned price change.

The current planogram is the counterfactual.
A and B both fit 6 units deep per facing, with no stacking.
A's shelf capacity rises from 12 to 18 units; B's falls from 18 to 12 units.
The store team confirms that the products fit, labels remain legible, and no fixture loading limit is breached.
That confirmation is an assumed input for this teaching case, not a field audit.

| Measure per store | Current plan | Proposed plan | Units |
|---|---:|---:|---|
| A facings | 2 | 3 | facings |
| B facings | 3 | 2 | facings |
| A shelf capacity | 12 | 18 | units |
| B shelf capacity | 18 | 12 | units |
| A expected sales before change | 42 | 42 baseline | units/week |
| B expected sales before change | 28 | 28 baseline | units/week |
| A contribution per unit | 1.40 | 1.40 | USD/unit |
| B contribution per unit | 1.10 | 1.10 | USD/unit |

At the baseline average, A sells 6 units per day and B sells 4 units per day.
A's nominal shelf coverage increases from 2.0 to 3.0 days; B's decreases from 4.5 to 3.0 days.
The store fills the shelf at least once daily under the plan, but peak demand and execution variability still require an OSA guardrail.
These ratios are planning indicators, not a guarantee of service.

### Forecast and economic case

A previous comparable pilot is assumed to support the following central forecast per store-week under stable price, range, and promotion:
A gains 5 units, B loses 3 units, and other category units are unchanged.
This is a synthetic teaching assumption; it is not presented as a published retail result.
The category therefore gains 2 units per store-week, not 5.
A's additional contribution is 5 × USD 1.40 = USD 7.00 per store-week.
B's lost contribution is 3 × USD 1.10 = USD 3.30 per store-week.
The incremental product contribution is USD 3.70 per store-week.

Additional replenishment and waste cost together are forecast at USD 0.45 per store-week.
Additional average inventory value is USD 20 per store and the annual carrying rate is 26%; using 52 weeks, carrying cost is USD 20 × 0.26 / 52 = USD 0.10 per store-week.
Net recurring benefit is USD 3.70 − USD 0.45 − USD 0.10 = USD 3.15 per store-week.
Assume the costs are incremental, mutually exclusive, and valued consistently with the contribution figures.
The one-time reset and label cost is USD 22 per store.
The evaluation horizon is 12 weeks after installation, with no discounting for this short illustration.

| Component | Arithmetic | Per store, 12 weeks |
|---|---|---:|
| Incremental product contribution | USD 3.70/week × 12 weeks | USD 44.40 |
| Incremental handling and waste | USD 0.45/week × 12 weeks | −USD 5.40 |
| Inventory carrying cost | USD 0.10/week × 12 weeks | −USD 1.20 |
| Reset and label cost | One-time | −USD 22.00 |
| Net benefit | 44.40 − 5.40 − 1.20 − 22.00 | USD 15.80 |

For 40 stores all executing for 12 complete weeks, the forecast net benefit is 40 × USD 15.80 = USD 632.
Simple payback is USD 22 / USD 3.15 per week = approximately 7.0 weeks.
This calculation is not a promise that every store will achieve the average or that a 40-store rollout has already occurred.
It does not count any wider basket, traffic, supplier funding, taxes, or terminal inventory recovery.
The USD 20 stock value is disclosed as working capital, not subtracted as a permanent loss and also charged as carrying cost.
Finance should separately assess cash funding and end-of-horizon stock disposition.

### Decision threshold and sensitivity

The 12-week break-even recurring benefit is USD 22 / 12 = USD 1.833 per store-week.
After USD 0.55 per store-week of recurring costs, the required incremental product contribution is USD 2.383 per store-week.
If B still loses 3 units per week and A contributes USD 1.40 per unit, A must gain approximately (2.383 + 3.30) / 1.40 = 4.06 units per store-week to break even.
The threshold is a continuous expectation; actual unit counts are discrete in any individual store-week.

A low case with A +3 units and B −3 units produces USD 4.20 − USD 3.30 − USD 0.55 = USD 0.35 per store-week.
Its 12-week net is USD 4.20 − USD 22 = −USD 17.80 per store.
A high case with A +7 units and B −2 units produces USD 9.80 − USD 2.20 − USD 0.55 = USD 7.05 per store-week.
Its 12-week net is USD 84.60 − USD 22 = USD 62.60 per store.
Neither scenario is a statistical confidence interval.
Both change demand assumptions while holding the stated cost assumptions fixed.
A separate sensitivity analysis should vary costs, compliance, and availability if those are uncertain.

### Elasticity interpretation

A changes from 2 to 3 facings, a 40% midpoint increase: (3 − 2) / 2.5 = 0.40.
Its forecast rises from 42 to 47 units per week, an 11.24% midpoint increase: 5 / 44.5 ≈ 0.1124.
The forecast arc response is 0.1124 / 0.40 ≈ 0.281.
This is derived from the assumed forecast and is not independent evidence for it.
It describes A's own response under this particular exchange; it does not capture B's loss or prove a general elasticity.

### Proposed pilot and decision

Before a broader commitment, assign 20 of the 40 eligible stores to the new layout and 20 to the current layout, stratifying by baseline volume and fixture variant.
The business case above describes a hypothetical 40-store full implementation, not the cost or benefit already realized by that 20-store treatment pilot.
Budget the pilot separately, including measurement and any later rollback costs.
Prespecify category contribution per assigned store-week over 12 full weeks as the primary outcome.
Measure A and B units, OSA, compliance, labor, and waste as secondary outcomes.
A feasibility review must determine whether 40 stores and 12 weeks provide useful precision; the example supplies no variance estimate and therefore cannot establish statistical power.

The proposed service guardrail is that B's OSA must not fall by more than 2 percentage points relative to the control group's change, measured with the same audit method.
The proposed safety stop is any substantiated unsafe shelf loading or blocked access; operations can immediately pause the affected installation.
Commercial leadership should also define an action rule for a materially negative category effect and the minimum precision needed for rollout.
The recommendation is to pilot if the expected value of the information and test cost justify it, not to declare a chain-wide benefit from the synthetic calculation.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
Save it as `space_case.py` and run `python space_case.py`.
It reproduces the central, low, and high economics and the assumed forecast's A arc response.
It does not simulate uncertainty, estimate a causal effect, check store-level geometry, or optimize the layout.
The 40-store output is a hypothetical full implementation; the example's proposed experiment treats only 20 stores.
All money is USD, all unit changes are per store-week, and the calculation is undiscounted.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    a_change: float  # units per store-week
    b_change: float  # units per store-week; negative is a loss


A_CONTRIBUTION = 1.40  # USD per unit
B_CONTRIBUTION = 1.10  # USD per unit
HANDLING_AND_WASTE = 0.45  # USD per store-week
AVERAGE_EXTRA_INVENTORY = 20.00  # USD per store
ANNUAL_CARRY_RATE = 0.26  # fraction per year
WEEKS_PER_YEAR = 52
RESET_COST = 22.00  # USD per store, one time
HORIZON_WEEKS = 12
STORES = 40


def arc_elasticity(q1, q2, s1, s2):
    """Midpoint demand change divided by midpoint space change."""
    if min(q1, q2, s1, s2) <= 0 or s1 == s2:
        raise ValueError("Positive values and a space change are required")
    demand_change = (q2 - q1) / ((q1 + q2) / 2)
    space_change = (s2 - s1) / ((s1 + s2) / 2)
    return demand_change / space_change


def economics(scenario):
    product = (scenario.a_change * A_CONTRIBUTION
               + scenario.b_change * B_CONTRIBUTION)
    carry = AVERAGE_EXTRA_INVENTORY * ANNUAL_CARRY_RATE / WEEKS_PER_YEAR
    recurring = product - HANDLING_AND_WASTE - carry
    net = recurring * HORIZON_WEEKS - RESET_COST
    return product, recurring, net


scenarios = (
    Scenario("low", 3, -3),
    Scenario("central", 5, -3),
    Scenario("high", 7, -2),
)

for scenario in scenarios:
    product, recurring, net = economics(scenario)
    print(f"{scenario.name}: product=${product:.2f}/week, "
          f"recurring=${recurring:.2f}/week, net=${net:.2f}/store")

central_net = economics(scenarios[1])[2]
central_weekly = economics(scenarios[1])[1]
print(f"central 40-store net=${central_net * STORES:.2f}")
print(f"central payback={RESET_COST / central_weekly:.2f} weeks")
print(f"A forecast arc response={arc_elasticity(42, 47, 2, 3):.3f}")
```

Expected output, allowing for normal display rounding:

```text
low: product=$0.90/week, recurring=$0.35/week, net=$-17.80/store
central: product=$3.70/week, recurring=$3.15/week, net=$15.80/store
high: product=$7.60/week, recurring=$7.05/week, net=$62.60/store
central 40-store net=$632.00
central payback=6.98 weeks
A forecast arc response=0.281
```

The script intentionally raises an error for nonpositive inputs or an unchanged space denominator.
A production model would need validated inputs, store-level assignment and execution data, uncertainty estimates, and a documented finance definition.
Do not replace the assumed changes with raw before-and-after sales and call the resulting output a causal estimate.

### Adapting the calculation to a real decision

Before replacing the synthetic numbers, write the financial scope in plain language.
For each item, obtain expected incremental units under the candidate versus the same item under the comparator.
Use contribution per unit from the same period, price basis, and store population.
Do not calculate benefit using regular-price contribution if the test will occur entirely during a discount.
If promotions vary by store, estimate the effect under the actual planned promotion policy.
List which costs are included in contribution and which are separate.

Replace `HANDLING_AND_WASTE` only with costs that vary because of this candidate.
A store's total labor expense is not necessarily incremental to a single facing exchange.
However, a shift in work from another activity may have a real opportunity cost even if payroll is unchanged.
Finance and operations should agree the valuation before the result is used for approval.
Record the source and owner of each input alongside its numeric value.

The script uses a single annual rate and a 52-week year to keep the example transparent.
A retailer may apply a different finance convention or require discounted cash flow.
Use the approved convention consistently for all alternatives.
If recurring benefits vary over time, calculate each week's net cash flow rather than multiplying a constant.
If rollouts occur in waves, model the number of active stores and reset costs in each week.
Do not multiply a fully installed store-week effect by all stores for weeks before installation.

The script calculates a simple point estimate.
To represent uncertainty, obtain store-level or appropriately aggregated estimates from the actual study.
Propagate uncertainty in demand response and costs using a documented method.
Preserve correlation between A's gain and B's loss where the data support it.
Independent random draws for both may misstate risk if substitution makes their outcomes dependent.
Show the probability of clearing the financial hurdle only when the probability model is defensible.

The input checks on arc elasticity are deliberately conservative.
They avoid division by zero and undefined midpoint percentage changes in this example.
The function does not validate whether the inputs represent a causal treatment effect.
A correct arithmetic output can still be a poor business answer when the baseline and comparison are biased.
Use the accompanying decision and measurement checks before interpreting the result.

## 8. Data, standards, and regulatory context

### Data contract

A usable analytical dataset needs stable keys and effective dates.
At minimum, connect store, SKU, date, sales quantity, price, promotion, layout version, intended facing count, and observed availability where available.
A layout table should identify fixture and shelf coordinates as well as width, height, depth, orientation, and capacity assumptions.
A store may have more than one fixture or display for the same item; distinguish primary shelf, secondary placement, and backroom stock.
A transaction record alone rarely establishes where the shopper encountered the product.

| Dataset | Grain | Required checks | Interpretation limit |
|---|---|---|---|
| Sales | Store × SKU × day or week | Returns, unit conversions, missing days | Purchases are not all attempted demand |
| Item master | SKU × effective date | Pack change, dimensions, GTIN mapping | Nominal dimensions may differ from physical orientation |
| Layout | Store/fixture × SKU × effective date | Version overlap, geometry, duplicates | Intended rather than necessarily executed |
| Shelf audit | Store × SKU × timestamp | Sampling, auditor, missed visits | A point observation can miss intraday outages |
| Store inventory | Store × SKU × timestamp | Negative balances, delayed posting | Backroom and shelf are not interchangeable |
| Promotion and price | Store × SKU × date | Effective period and display flags | A price change can coincide with a reset |
| Replenishment | Store × SKU × event | Event timestamps, actual shelf fill | Scheduled visits do not establish execution |
| Labor and waste | Store × activity × period | Cost basis, attribution, completeness | Allocation rules can obscure incremental effort |

Maintain both planned and actual dates; do not backdate execution to the scheduled date.
Retain store exclusions and missing observations with reasons.
A missing audit is not proof of availability or noncompliance.
Document the exact denominator of OSA: for example, the number of scheduled store-SKU audit observations at which the item was on shelf divided by all completed valid observations.
That definition weights audit times, not necessarily shopper visits.
If audits occur only on weekday mornings, they may miss weekend outages.

### Identification and measurement standards

GS1 publishes standards for identifying trade items and locations, including GTINs and GLNs.
These identifiers support unambiguous joining when consistently maintained; they do not prescribe optimal facings or validate a shelf drawing.
GS1's package measurement rules provide a common basis for product dimensions, but a planogram still needs the actual presentation orientation, fixture clearances, and a physical fit check.
Use the retailer's item and location crosswalk when supplier and retailer codes differ.
Check that a packaging redesign has not silently changed dimensions under the same analytical grouping.

Currency, time bucket, unit of sale, and tax treatment must be consistent across sales and contribution data.
A category contribution definition is a management accounting convention, not a universal standard.
Record whether discounts, supplier funding, delivery costs, and shrink are included before comparing scenarios.
If contribution includes shrink already, do not subtract the same shrink again as a separate cost.
Store-level averages should be weighted according to the estimand, not simply according to whichever join produces more rows.

### Legal and policy context

The U.S. Food and Drug Administration (FDA) Food Code is a model code; adoption and specific requirements depend on the relevant jurisdiction and local authority.
Food handling and temperature requirements apply where adopted and relevant, but the model code does not prescribe an optimal cereal-facing count.
Refrigerated and perishable categories require local food-safety review in addition to space planning.
The U.S. Occupational Safety and Health Administration (OSHA) regulates workplace hazards under applicable standards; evaluate fixture loading, safe access, and lifting practices with the safety team rather than treating a layout calculation as a compliance certification.
Retailers operating elsewhere must identify their own jurisdiction's rules.
These references were consulted as framework sources for this educational material; they are not legal advice or proof of a site's compliance.

If customer-level loyalty or video data are added, follow applicable privacy law, retention rules, purpose limitations, and internal access controls.
Aggregated store-week data may avoid some individual-level analysis, but aggregation alone does not settle every legal obligation.
Supplier merchandising agreements and competition rules may constrain execution; legal and procurement should review actual contracts and jurisdiction-specific obligations.
Do not assume that every supplier-requested placement is mandatory or that all contractual terms override safety and local law.

### Data quality and lineage

Store the source, extraction timestamp, owner, transformation, and version for each input.
Keep a lineage from an approved planogram to the physical store and effective period.
Log corrections rather than overwriting prior assignments, particularly during experiments.
Reconcile sales totals to the finance reporting source at a specified grain.
Investigate join fanout: a duplicated planogram row can multiply sales without creating a real transaction.
Check that each SKU has no more than the intended number of active layout records for the same fixture position and date.

### Data collection design in a real store

An audit protocol should define the observation window before auditors enter stores.
Specify whether the auditor records the front shelf, all secondary displays, or both.
Specify whether a single visible unit counts as available even when the item will likely empty before the next visit.
Record a timestamp and, where appropriate and permitted, photographic evidence.
Define how auditors treat a blocked shelf, a misplaced item, and an incorrect label.
Use the same definitions across treatment and control stores.

Scheduled visits and completed visits should both appear in the data.
If 100 visits are scheduled and 80 are completed, a 95% pass rate among completed visits does not establish 95% compliance among all scheduled visits.
Report 76 passing visits out of 80 completed, with 20 missing visits, if those are the observed counts.
Investigate whether missed visits cluster by store, geography, or layout assignment.
Do not impute a pass or fail without an explicit and justified missing-data rule.

Sales and stock events also require aligned clocks.
A sale timestamp may be local store time while an inventory update is in coordinated universal time.
A transaction posted after midnight can be attributed to the wrong store day if conversion is inconsistent.
Define week boundaries before joining events to a weekly planogram.
If a reset occurs on Wednesday, choose whether the first partial week is excluded, separately labeled, or allocated by actual exposure.
The rule should be the same across arms or be built into a prespecified analysis.

Quality checks should be actionable, not a ceremonial list.
If a dimension is absent, route it to the item-data owner and hold the physical fit calculation.
If two layout versions overlap for the same store and shelf, resolve effective dates with the space planner.
If shelf audit says an item was present but store inventory is negative, retain both observations and investigate the source.
Do not silently overwrite one system with the other; they measure different things and can both contain errors.
Log the correction, date, and decision owner.

### Interpretation and access

Keep store-level test assignment restricted to people who need it when premature changes could contaminate the control.
Give store teams the instructions they need to execute their assigned plan.
Preserve the ability to audit decisions without unnecessarily exposing customer-level data.
If photos contain identifiable people, follow the retailer's privacy and retention policies.
The appropriate legal basis and consent requirements depend on jurisdiction and collection context.

Supplier-provided data can help identify pack dimensions and brand performance.
They do not automatically represent the retailer's complete category or costs.
Record whether supplier data are measured sell-in to the retailer or consumer sell-through at the register.
Sell-in changes can reflect inventory timing rather than shopper demand.
Reconcile periods and units before comparing the two.
Commercial confidentiality and contractual rights can limit sharing of store or competitor detail.

## 9. Industry practice and case evidence

### What evidence can and cannot establish

Space management combines store operations, merchandising, and measurement.
A retailer may use a planogram platform, a spreadsheet, or a drawing; the method does not become causal merely because software displays a precise layout.
Practitioners usually need to reconcile the central layout with local fixtures and actual store execution.
Category-level economics and service guardrails are necessary because a product's gain can come at a neighbor's expense.
These are decision principles, not claims that all retailers use the same workflow.

The published retail shelf-space literature provides evidence that space and position can matter while also showing variation across contexts.
Curhan's 1972 article studies the relationship between shelf space and unit sales in supermarkets.
Its empirical context and research design should be read before transporting its results to a modern chain, category, or e-commerce environment.
The paper is evidence that the relationship has been investigated, not a source of a universal coefficient for the cereal example.

Chandon and coauthors' 2009 *Journal of Marketing* article examines shelf position and in-store attention and evaluation using eye-tracking evidence.
Their study supports examining visual attention and shelf placement as mechanisms.
It does not by itself establish the financial return of adding a facing in a particular retailer's current fixture.
Eye tracking measures attention under its study conditions; it should not be equated mechanically with incremental profit or all future purchases.

### Named company evidence

Walmart describes its category-management and supplier collaboration processes in public supplier materials.
Those materials illustrate a real retailer–supplier context in which item data and merchandising coordination matter.
They do not disclose the causal impact of the fictional A/B cereal exchange in this lesson.
A supplier's presentation about expected sales should be scrutinized for the comparator, category displacement, promotion changes, and whether the data are independent.
A manufacturer's brand-level objective may legitimately differ from the retailer's total-category objective.

GS1's package measurement guidance is an industry standard, not a company outcome study.
The presence of a common identification or measurement standard can reduce ambiguity in exchanged data.
It does not demonstrate that a particular layout was built correctly or generated an incremental return.
Inferences about execution or return require retailer-specific measurements.

### Turning evidence into a local decision

Before using an outside case, compare product category, initial space, fixture geometry, shopper mission, baseline availability, price, and replenishment cadence.
A shelf-capacity benefit in a high-volume urban convenience store may not appear in a low-volume rural store.
A visual-attention effect in a laboratory may not survive a real shelf with different signage and competing displays.
A low-volume product may deserve a minimum facing to preserve a required dietary choice even if its short-run contribution per centimeter is low.
These are hypotheses to test or constraints to document, not claims about the named studies.

A practical evidence hierarchy is randomized local tests, carefully designed local quasi-experiments, descriptive local evidence, and external benchmarks.
That ordering concerns causal credibility for a specific decision, not the prestige of an author or the volume of data.
Any tier can be uninformative if treatment, outcome, and eligible population are poorly defined.
A randomized study with poor execution or an outcome too noisy for the decision can still fail to answer the business question.

### Reading a claim of shelf-space productivity

A supplier presentation might show that a brand has higher sales per centimeter than a competing item.
That statistic describes productivity under the existing mix of prices, placements, and availability.
It does not reveal how much the brand would gain from the next centimeter.
It also does not reveal which product must lose that centimeter.
Ask whether the figure measures revenue, units, or contribution.
Ask whether the denominator includes only the primary shelf or all displays.
Ask whether out-of-stock periods are included.

If the proposal cites another chain's gain, request its baseline, comparison group, and treatment definition.
Check whether the original change also altered price, promotion, distribution, or shelf height.
Check whether the reported gain is for the brand or the entire category.
Check whether store labor, reset, and incremental inventory were deducted.
If the evidence is a vendor case study, label its commercial interest and avoid implying independent verification.
Even an independently verified historical result may not transfer to a different category and fixture.

The retailer can respond constructively without rejecting supplier evidence.
Invite the supplier to provide product dimensions, supply constraints, and a testable response hypothesis.
Keep retailer-level financial assumptions and the control design under retailer governance.
Agree in advance on which results may be shared and at what aggregation.
The objective is a decision both parties can understand, not a predetermined winning brand.

### Field review after installation

An operating review should start with execution: which stores installed which version and when.
Next inspect whether the intended products and labels were present.
Then compare outcomes using the assigned treatment and control groups.
Finally reconcile the financial effect with the costs actually incurred.
This order helps distinguish an unimplemented intervention from an ineffective one.

If the treatment group missed stock deliveries, report the disruption and its timing.
Do not automatically remove it: supply reliability may be part of the deployment reality.
Consider a supplementary analysis of operational disruption while preserving the primary comparison.
If the disruption is unrelated to the candidate but unusually concentrated in one group, discuss the resulting uncertainty.
If a successful pilot depended on unusually intensive support, include that support in the scale-up plan and cost.
A rollout is not simply a copy of a drawing; it is a repeatable operating process.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Reverse causality | High sellers already have more space | Managers allocate space in response to demand | Correlation overstates response | Randomize or use a defensible design and disclose selection |
| Hidden displacement | Item A rises while category is flat | A takes purchases from B | Brand gain mistaken for category profit | Measure all affected category items and contribution |
| Availability confounding | Outages change with facings | Extra capacity changes availability as well as visibility | Mechanism misattributed | Measure OSA and state whether treatment is a combined policy |
| Promotion overlap | Treatment stores have a different display calendar | Price or display changes demand | Space credited for another intervention | Hold policies comparable or test the combined package |
| Infeasible geometry | Nominal widths sum to fixture width with no clearance | Hardware or package lips consume space | Store cannot build layout | Use usable dimensions and physical verification |
| Excess shelf stock | Shelf life is short relative to coverage | Added capacity raises aging | Waste erodes contribution | Limit practical fill and monitor expiry |
| Backroom mistaken for shelf | Inventory is positive while audits show empty shelf | Stock is not replenished promptly | Lost sales remain hidden | Track shelf availability and actual shelf-fill events |
| Wrong unit conversion | Case and each sales move inconsistently | Mixed pack sizes or dimensions | Invalid capacity and margin | Normalize units and version item master |
| Layout drift | Photos differ from assigned version | Late reset or unauthorized substitution | Estimated exposure is wrong | Audit actual state and retain assignment for ITT |
| Biased audits | Only compliant stores return photos | Missingness depends on execution | Compliance is overstated | Prespecify audit sampling and report missing visits |
| False precision | One elasticity used across every store | Differences in fixtures and demand ignored | Misallocation and misleading forecast | Segment, bound, and validate locally |
| Double-counted costs | Shrink appears in margin and cost table | Same cost subtracted twice | Good proposal rejected | Reconcile finance definitions line by line |
| Ignored labor | Reset is shown as free | Store staff absorb unbudgeted work | False payback | Cost installation and ongoing work |
| Simultaneous position change | Extra facing moves A to eye level | Exposure and location change together | Individual effects cannot be separated | Hold location fixed or label the combined intervention |
| Insufficient test precision | Confidence interval spans large gains and losses | Too few independent stores or short duration | Inconclusive rollout decision | Size test from store-level variation and decision threshold |
| Seasonal extrapolation | Pilot covers only a holiday | Demand pattern is not representative | Rollout forecast fails later | Test or adjust by season and state limits |
| Control contamination | Control stores receive the new layout | Operations apply a common reset | Treatment contrast shrinks or changes | Protect control execution and document deviations |
| Unsafe access | Product overhangs or blocks a label/access path | Geometry ignores store conditions | Injury or policy violation | Operations approval and immediate stop authority |
| Unclear owner | Layout is approved but no store receives work order | Accountability ends at the drawing | No actual change or audit | Name installation, audit, and decision owners |
| Overfitted assortment | Many moves chosen from the same noisy pilot | Winner selected partly by chance | Rollout underperforms | Hold out validation stores and report selection |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Usable fixture dimensions are correct | Candidate must fit | Uprights, dividers, or damaged shelves reduce space | Measure representative stores and verify exceptions |
| Package geometry is stable | Facings determine width and capacity | Pack redesign or orientation change | Date product dimensions and physically verify |
| Historical response transfers | Forecast needs an effect estimate | Different store, season, or shopper mission | Compare populations and run a local pilot |
| Category boundary captures substitution | Contribution sums affected products | Switching crosses category or store | Examine neighboring categories and state unmeasured effects |
| Contribution is incremental | Unit changes must translate to value | Included fixed costs or duplicated shrink | Reconcile cost definitions with finance |
| Shelf availability is measured comparably | Distinguish demand and execution | Sparse or biased audit timing | Sample at comparable times and quantify missingness |
| Assignment is respected in analysis | ITT estimates policy assignment | Selective exclusion of noncompliant stores | Retain all assigned stores and report actual exposure separately |
| Control is a credible counterfactual | Effect requires comparison | Different promotion, trends, or contamination | Randomize, monitor policies, and inspect pre-period patterns |
| Carrying rate fits horizon | Inventory cost requires time basis | Annual rate used as weekly rate | Convert the approved annual rate to weeks |
| Demand is stable enough for coverage estimate | Average sales approximate exposure | Weekend spikes or irregular deliveries | Examine interval-level demand and fill cadence |
| Test is informative | Estimate must distinguish decision-relevant effects | High store variation or short duration | Calculate precision using independent store variability |
| Costs do not change with scale | Pilot economics extrapolate to rollout | Travel, labor learning, or implementation bottlenecks | Model scale-specific unit costs and capacity |

None of these assumptions is guaranteed by a large transaction dataset.
When the assumption cannot be supported, report a bound, choose a smaller pilot, or withhold the rollout recommendation.
A statistically significant item effect can still be economically negative after displacement and costs.
Conversely, a noisy estimate is not proof that the true effect is zero.

## 12. Exercises

### Exercise 1 — Calculate space and capacity

A 90 cm usable shelf contains item C, 12 cm wide, with 3 facings, and item D, 9 cm wide, with 4 facings.
Other dividers consume 6 cm within the stated usable width.
C holds 5 units deep per facing; D holds 4 units deep per facing; neither stacks.
Deliverable: report occupied width, remaining width, and each item's capacity.
Acceptance: use cm and units, include dividers exactly once, and show arithmetic.
Time: 8 minutes.

### Exercise 2 — Interpret coverage

C holds 15 units and sells an average of 35 units per week.
The store fills the shelf every two days, but half the weekly demand occurs on Saturday and Sunday.
Deliverable: calculate average daily sales and nominal days of supply, then state one operational risk.
Acceptance: use a seven-day week and explain why average coverage does not guarantee weekend availability.
Time: 8 minutes.

### Exercise 3 — Calculate an arc response

An item increases from 4 to 5 facings and forecast sales rise from 80 to 86 units per week.
Deliverable: compute midpoint percentage changes and arc elasticity.
Acceptance: show both denominators, round to three decimals, and distinguish forecast response from proven causation.
Time: 10 minutes.

### Exercise 4 — Critique the causal claim

A vendor says, “Stores with 5 facings sell 25% more than stores with 3, so every store will sell 25% more after adding 2 facings.”
Deliverable: write a five-sentence critique.
Acceptance: identify selection, at least one concurrent factor, the counterfactual, extrapolation risk, and a stronger design.
Time: 12 minutes.

### Exercise 5 — Design a feasible plan

You may reallocate a 120 cm shelf among A, B, and other products as in the worked example.
Assume B must retain at least 2 facings and C, a dietary-need item within the other products, must retain its current position.
Deliverable: specify one feasible exchange, the measurements to verify, and two reasons not to infer profitability from fit alone.
Acceptance: show width arithmetic, protect both constraints, and name execution and financial checks.
Time: 15 minutes.

### Exercise 6 — Evaluate incremental economics

A candidate produces +6 A units and −4 B units per store-week.
A contributes USD 1.50 per unit and B contributes USD 0.90 per unit.
Incremental recurring costs total USD 0.60 per store-week, reset costs USD 30 per store, and the horizon is 10 weeks.
Deliverable: calculate weekly net benefit, total net benefit per store, and simple payback.
Acceptance: include the displaced item and all stated costs, and specify whether the undiscounted case breaks even within the horizon.
Time: 12 minutes.

### Exercise 7 — Design a pilot

A retailer has 60 eligible stores across two fixture types and expects simultaneous seasonal promotions.
Deliverable: describe assignment, primary metric, data, duration and sample-size decision, a guardrail, and an analysis approach.
Acceptance: define the comparison and independent assignment unit, address promotions and execution, preserve ITT, and avoid asserting power without variance data.
Time: 25 minutes.

### Exercise 8 — Critique and communicate

A proposed layout increases a brand's forecast contribution by USD 10 per store-week, but deletes a competing item whose contribution is USD 8 per store-week.
No substitution estimate or reset cost is supplied.
Deliverable: write a recommendation of at most 120 words, explicitly distinguishing forecast, target, plan, and commitment.
Acceptance: do not call USD 10 incremental category profit; name missing counterfactual and cost inputs.
Time: 15 minutes.

### Exercise 9 — Govern implementation

The space planner sends a drawing to 100 stores, but 15 have nonstandard fixtures and 10 report that the replacement product has not arrived.
Deliverable: draft an exception and rollback procedure with named roles and retained records.
Acceptance: cover physical safety, missing product, version control, audit, escalation, and the primary analysis population.
Time: 20 minutes.

## 13. Self-check answers

### Exercise 1

C occupies 3 × 12 = 36 cm and D occupies 4 × 9 = 36 cm.
Including 6 cm of dividers, total occupied width is 78 cm.
Remaining width is 90 − 78 = 12 cm.
C holds 3 × 5 = 15 units; D holds 4 × 4 = 16 units.
Do not subtract divider width twice if it was already excluded from usable width; here the question explicitly includes it within the stated 90 cm.

### Exercise 2

Average daily sales are 35 / 7 = 5 units per day.
Nominal coverage is 15 / 5 = 3 days from a full shelf.
Weekend demand is 17.5 units on average across the two days, so a full 15-unit shelf can empty before the next fill if no intermediate replenishment occurs.
The half-unit is an average across weeks, not a half package sold to one shopper.
Actual risk depends on timing, demand variability, and backroom-to-shelf execution.

### Exercise 3

Demand midpoint change is (86 − 80) / 83 = 0.072289, or 7.229%.
Space midpoint change is (5 − 4) / 4.5 = 0.222222, or 22.222%.
The arc response is approximately 0.072289 / 0.222222 = 0.325.
The figures are forecasts and do not isolate a causal effect.
The ratio is local to this discrete change and says nothing directly about category displacement.

### Exercise 4

A strong answer notes that high-demand stores may have received more facings before the comparison.
Different prices, promotions, demographics, fixtures, or availability could also explain the observed difference.
The missing counterfactual is what those same stores would have sold under the alternative allocation.
The 25% difference need not transfer to every store or to a two-facing change.
Randomizing comparable stores to clearly specified layouts while tracking category contribution and execution would provide stronger evidence.

### Exercise 5

One feasible exchange is A from 2 to 3 facings and B from 3 to 2, leaving the other products including C unchanged.
The proposed width is 3 × 10 + 2 × 10 + 70 = 120 cm, assuming 120 cm is genuinely usable.
Verify package dimensions, dividers, height, depth, actual fixtures, labels, and safe access.
Fit does not establish that the shelf remains stocked or that the added A contribution exceeds B displacement and costs.
A strong answer also names an exception path for nonstandard fixtures.

### Exercise 6

Product contribution changes by 6 × USD 1.50 − 4 × USD 0.90 = USD 5.40 per store-week.
After recurring costs, net benefit is USD 5.40 − USD 0.60 = USD 4.80 per store-week.
Over 10 weeks, net after reset is 10 × USD 4.80 − USD 30 = USD 18 per store.
Simple payback is USD 30 / USD 4.80 = 6.25 weeks.
The undiscounted central case breaks even within the 10-week horizon, conditional on its input assumptions.

### Exercise 7 rubric

A strong design assigns stores randomly within fixture and relevant baseline strata and records the seed and eligible list.
It states a category-level contribution outcome per assigned store-week, defines the comparison and evaluation window, and retains all assigned stores for ITT.
It either keeps promotion exposure comparable or interprets and models the treatment as a combined policy.
It captures planned and actual layouts, OSA, costs, and audit missingness, with a prespecified service or safety guardrail.
It uses historical store-level variance and the minimum economically important effect to assess feasible duration and precision.
No numerical sample-size claim is justified by the exercise's inputs alone.

### Exercise 8 rubric

A strong recommendation says that USD 10 per store-week is a brand forecast, not established incremental category profit.
It requests an estimate of retained versus lost competing-item sales and relevant contribution, reset, inventory, and operating costs.
It labels an aspirational category improvement as a target, the proposed reset as a plan, and an authorized budget and owner as a commitment only after approval.
It proposes a comparison or pilot if decision-relevant uncertainty remains.

### Exercise 9 rubric

A strong procedure prevents forced installation in unsafe or incompatible fixtures and routes exceptions to the space planner and operations owner.
It holds or reschedules stores without stock rather than silently substituting another item.
It records assignment, intended and actual dates, layout version, reason, and subsequent correction.
It specifies who authorizes rollback, how the prior plan is restored, and how the result is audited.
The primary ITT analysis retains assigned stores even if installation fails; an execution analysis is supplementary and may be selected.

## 14. Interview questions

1. **What is a facing, and why is it not equivalent to a fixed amount of space?**  
   Strong: identifies visible fronts, package width, fixture geometry, and orientation.  
   Weak: assumes one facing always occupies the same width and stores the same units.
2. **How would you calculate shelf capacity?**  
   Strong: uses facing count, practical depth, permitted stacking, and physical verification.  
   Weak: equates capacity with warehouse stock or ignores handling limits.
3. **What does an elasticity of 0.2 mean?**  
   Strong: describes a local percentage response, the space measure, conditions, and uncertainty.  
   Weak: promises a fixed gain for any product and any size change.
4. **Why can more facings correlate with higher sales without causing them?**  
   Strong: identifies allocation to high-demand stores and other confounders.  
   Weak: treats correlation as an experiment.
5. **How do you evaluate a facing exchange?**  
   Strong: specifies the displaced item, category contribution, costs, and feasible counterfactual.  
   Weak: counts only the gaining brand's revenue.
6. **How would you separate visibility from availability?**  
   Strong: defines treatments and measures OSA and replenishment, or acknowledges a combined policy effect.  
   Weak: assumes extra space changes only visibility.
7. **What is the role of randomization?**  
   Strong: balances observed and unobserved factors in expectation and estimates assignment effects.  
   Weak: says randomization guarantees perfect execution or eliminates sampling uncertainty.
8. **How do you handle stores that did not implement the reset?**  
   Strong: retains assignment for ITT, records deviations, and treats execution analyses cautiously.  
   Weak: drops all failures and reports the remaining average as the rollout effect.
9. **Why can nominal days of supply be misleading?**  
   Strong: discusses peaks, shelf-fill intervals, variability, and backroom stock.  
   Weak: treats average coverage as a service guarantee.
10. **What belongs in an implementation decision?**  
    Strong: names the owner, layout, cost, uncertainty, store constraints, guardrails, and rollback.  
    Weak: approves a drawing based only on sales per meter.

## 15. Key terms

- [Availability](#4-definitions-and-vocabulary) — ability to buy an item at the specified time and place.
- [Bay](#4-definitions-and-vocabulary) — identified fixture section.
- [Capacity](#4-definitions-and-vocabulary) — practical units held in assigned shelf space.
- [Cannibalization](#4-definitions-and-vocabulary) — sales transferred from another product.
- [Compliance](#4-definitions-and-vocabulary) — measured match to the intended layout.
- [Contribution](#4-definitions-and-vocabulary) — revenue less defined included variable costs.
- [Control](#4-definitions-and-vocabulary) — comparison assignment without the treatment.
- [Counterfactual](#4-definitions-and-vocabulary) — outcome without the proposed change.
- [Cross-space effect](#4-definitions-and-vocabulary) — response of one item to another's space.
- [Days of supply](#4-definitions-and-vocabulary) — capacity divided by average daily demand.
- [Depth](#4-definitions-and-vocabulary) — units behind a front presentation.
- [Elasticity](#step-9-calculate-and-interpret-elasticity) — ratio of percentage demand and space changes.
- [Facing](#4-definitions-and-vocabulary) — visible front presentation of an item.
- [Fixture](#step-3-measure-actual-usable-space) — physical shelving or display structure.
- [GLN](#identification-and-measurement-standards) — GS1 location or party identifier.
- [GTIN](#identification-and-measurement-standards) — GS1 trade item identifier.
- [Guardrail](#proposed-pilot-and-decision) — monitored constraint on an acceptable intervention.
- [Incremental effect](#4-definitions-and-vocabulary) — difference from the counterfactual outcome.
- [Intention to treat](#step-7-design-an-experiment-when-feasible) — analysis by assigned policy.
- [Inventory carrying cost](#4-definitions-and-vocabulary) — cost of holding average incremental stock.
- [Linear shelf space](#4-definitions-and-vocabulary) — allocated horizontal width.
- [On-shelf availability](#data-contract) — item available on the shelf under a defined sampling rule.
- [Payback](#decision-threshold-and-sensitivity) — time to recover the initial cost from recurring benefit.
- [Planogram](#4-definitions-and-vocabulary) — versioned intended shelf arrangement.
- [Replenishment interval](#step-4-calculate-capacity-and-replenishment-exposure) — time between effective shelf fills.
- [Reset](#step-13-approve-and-execute) — physical transition between layouts.
- [SKU](#4-definitions-and-vocabulary) — retailer-defined item or sellable variant.
- [Stockout](#4-definitions-and-vocabulary) — unavailability at a defined time and place.
- [Treatment](#step-6-form-the-causal-question) — assigned layout or operational policy.

## 16. Sources

1. Curhan, Ronald C. “The Relationship between Shelf Space and Unit Sales in Supermarkets.” *Journal of Marketing Research* 9, no. 4 (1972): 406–412. [DOI: 10.1177/002224377200900411](https://doi.org/10.1177/002224377200900411). Empirical shelf-space study; interpret in its original context.
2. Chandon, Pierre, J. Wesley Hutchinson, Eric T. Bradlow, and Scott H. Young. “Does In-Store Marketing Work? Effects of the Number and Position of Shelf Facings on Brand Attention and Evaluation at the Point of Purchase.” *Journal of Marketing* 73, no. 6 (2009): 1–17. [DOI: 10.1509/jmkg.73.6.1](https://doi.org/10.1509/jmkg.73.6.1). Evidence on attention and evaluation, not a universal profit response.
3. GS1. [GTIN allocation and identification](https://www.gs1.org/standards/id-keys/gtin). Official identifier overview; identifiers do not prescribe allocation.
4. GS1. [GLN identification](https://www.gs1.org/standards/id-keys/gln). Official location identifier overview.
5. GS1. [Package Measurement Rules Standard](https://www.gs1.org/standards/gs1-package-measurement-rules-standard/current-standard). Common package measurement conventions; check the current standard and physical presentation.
6. U.S. Food and Drug Administration. [FDA Food Code](https://www.fda.gov/food/fda-food-code/food-code-2022). The 2022 model code is not automatically binding without jurisdictional adoption.
7. U.S. Occupational Safety and Health Administration. [Warehousing](https://www.osha.gov/warehousing). Safety information; applicable requirements depend on the activity and jurisdiction.
8. Walmart. [Supplier resources](https://corporate.walmart.com/suppliers). Company information about supplier relationships, not evidence of the worked example's results.
9. Hernán, Miguel A., and James M. Robins. [*Causal Inference: What If*](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/). Chapman & Hall/CRC, 2020. Open textbook on causal questions, intervention, and observational limitations.

Sources support the general methods and context indicated above.
The cereal inputs, cost values, test design, and outputs are synthetic teaching material.

## 17. Links

- [Module 17 index](_index.md).
- [Module 17 survey](../../../course/17_assortment_category_retail_analytics.md).
- [Assortment decision stack](01_assortment-decision-stack.md).
- [Category roles and decision trees](03_category-roles.md).
- [Variety versus choice overload](02_variety-choice-overload.md).
- [Simulation and optimization production labs](../../05_labs_electives/30_simulation_optimization_production_labs/).
