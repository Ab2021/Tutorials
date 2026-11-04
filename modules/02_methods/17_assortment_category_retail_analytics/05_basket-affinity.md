# Basket Affinity and Cross-Selling

> Module 17 — Assortment, Category, and Retail Analytics  
> Topic 05 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Module 04, Module 16, topics 01 and 04  
> Version 2026.5

## 1. Why this topic exists

A retailer often knows which items sell together but not whether an intervention will make shoppers buy more.
A category manager may want to place pasta sauce beside pasta, recommend batteries with toys, or give a discount on salad dressing when a shopper buys salad.
Each action uses scarce space, margin, labor, customer attention, or digital impressions.
The apparent association is not itself the benefit of the action.

A basket is an observed collection of purchased items at a defined checkout or order.
An attach rate describes how often a second item appears in baskets containing an anchor item.
It does not reveal what the same shopper would have purchased without a recommendation, discount, or display.
A pasta buyer may have planned the sauce before entering the store.
A high attach rate can therefore coexist with zero incremental sauce sales.

Commercial decisions also change which purchases occur elsewhere.
A promoted dressing may replace another dressing instead of expanding the category.
A display can move products from the regular shelf to an endcap without adding store sales.
A discount can be paid on units that shoppers would have purchased anyway.
If a recommendation crowds out another profitable recommendation, its value is its incremental contribution after that displacement.

Supply-chain teams have a stake in the decision.
An intervention that succeeds can create local stockouts, overstocks at untreated stores, or extra handling.
A broken execution creates a misleading null result.
A store team needs a feasible placement, sufficient replenishment, a clear start and end date, and an owner.

This topic takes affinity as a descriptive starting point and develops a decision process.
You will specify the eligible population, calculate the correct denominator, translate associations into a plausible action, and test that action against a counterfactual.
You will also distinguish contribution from revenue and a test estimate from a forecast or commitment.
The goal is not to maximize the number of linked items in a basket.
The goal is to improve the retailer's defined outcome without violating customer, operational, or legal constraints.

## 2. Learning outcomes

After this topic you can:

1. **Define** a basket, anchor, attach rate, support, and lift using a specified population and time window; exercise 1 checks the definitions.
2. **Compute** directional attach rates and lift from a basket contingency table; exercise 2 checks the arithmetic.
3. **Diagnose** why observed affinity does not identify an intervention's causal effect; exercise 3 requires a causal explanation.
4. **Calculate** incremental contribution after discounts, substitution, implementation cost, and a stated opportunity cost; exercise 4 checks the calculation.
5. **Design** an appropriately randomized cross-sell test with an estimand, denominator, guardrails, and execution checks; exercise 5 evaluates the design.
6. **Interpret** an intention-to-treat estimate, uncertainty, and heterogeneous execution without conditioning on post-treatment behavior; exercise 6 checks interpretation.
7. **Recommend** an operational decision with inventory, customer experience, privacy, and rollback requirements; exercises 7 and 8 assess the recommendation.

A correct numerical answer must name its units and population.
A correct design must say what would happen to the comparison group.
A strong recommendation states what evidence would change the decision.

## 3. The core mental model

```text
Business decision and constraints
           |
           v
Define eligible trips, anchor, partner, time, and channel
           |
           v
Describe observed co-purchase and baseline availability
           |
           v
Generate a feasible cross-sell hypothesis, not a causal claim
           |
           v
Randomize a deployable offer or placement against business as usual
           |
           v
Measure incremental store or customer contribution and guardrails
           |
           v
Deploy, revise, stop, or gather more evidence
           |
           +----> Operations, inventory, privacy, and review loop
```

Read the diagram from top to bottom.
The action precedes the choice of metric because a digital recommendation, a coupon, and a shelf relocation change different things.
The denominator defines whose response matters.
The descriptive step identifies opportunities and exposes data problems.
Randomization creates a comparison that can estimate the consequence of the action under stated conditions.
The economic step translates the estimate into a decision.

Draw a separate causal sketch for the observational association:

```text
Mission / season / store format ----> purchase anchor
             |                       /
             +--------------------> purchase partner
Price, availability, and promotions -> both purchases
```

The common causes are examples, not an exhaustive causal graph.
For instance, a barbecue mission may cause both charcoal and burger purchases.
That co-purchase does not establish that showing burgers to charcoal buyers creates another burger sale.
Randomizing the placement changes the source of variation and makes the assigned groups comparable in expectation.

There are three different quantities to keep separate.
**Affinity** is a descriptive relationship in observed baskets.
**Incrementality** is the difference between outcomes under an intervention and under the specified alternative.
**Contribution** is the incremental financial value after the relevant incremental costs and displaced value.
The first can prioritize the second for testing; neither guarantees the third is positive.

An anchor can be selected before or after a shopper sees a recommendation.
That timing matters.
If treatment affects whether someone buys the anchor, conditioning analysis on observed anchor purchasers can select different populations across arms.
For causal comparisons, define eligibility before assignment or analyze all randomized units.
A pre-treatment anchor signal, such as an item already in a digital cart before randomization, can support an anchor-specific estimand if it is recorded consistently.
A final receipt anchor cannot automatically serve that role.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Basket | Deduplicated collection of items in a defined transaction or order. | Treating multiple checkout events as one customer mission without a linking rule. |
| Eligible basket | Basket meeting a prespecified scope such as store, channel, date, and completed sale. | Switching the denominator after seeing the result. |
| Anchor | Item or item group used to define the conditional population. | Calling the anchor the cause of the partner purchase. |
| Partner | Item or group proposed for attachment to the anchor. | Assuming every product in the group has the same economics. |
| Pair count | Number of eligible baskets containing both anchor and partner, in baskets. | Counting item units instead of distinct baskets. |
| Anchor count | Number of eligible baskets containing the anchor, in baskets. | Including returns or ineligible transactions only in one side. |
| Partner count | Number of eligible baskets containing the partner, in baskets. | Using units as a proxy for baskets. |
| Attach rate | Pair count divided by anchor count, as a percentage of anchor baskets. | Treating it as incremental conversion. |
| Reverse attach rate | Pair count divided by partner count, as a percentage of partner baskets. | Assuming it equals forward attach. |
| Support | Pair count divided by all eligible baskets, as a percentage. | Confusing prevalence with the conditional attach rate. |
| Baseline partner rate | Partner count divided by all eligible baskets, as a percentage. | Ignoring it when assessing whether attachment is unusual. |
| Lift | Attach rate divided by baseline partner rate, a dimensionless ratio. | Reading a lift above one as a causal uplift. |
| Confidence | Association-rule term for the conditional probability of the partner given the anchor. | Treating a confidence level as a confidence interval. |
| Incremental effect | Difference in potential outcomes under the action and its specified alternative, in the chosen unit. | Equating treated sales with incremental sales. |
| Counterfactual | Outcome the same unit would have had under the alternative treatment. | Assuming pre-period sales alone reveal it. |
| Randomization unit | Unit assigned to an intervention, such as customer, store, or store cluster. | Treating individual receipts as independently randomized when stores were assigned. |
| Analysis unit | Unit summarized and compared in the analysis. | Ignoring clustering when it differs from the randomization unit. |
| Intention to treat | Effect of assignment, including imperfect delivery, in all eligible randomized units. | Restricting to customers who clicked or stores that complied. |
| Exposure | Actual delivery or view of the intervention. | Assuming assignment guarantees exposure. |
| Conversion | Fraction of eligible units completing a stated purchase or other action. | Changing its denominator between arms. |
| Cannibalization | Sales or contribution displaced from other items because of the intervention. | Counting all partner units as new category demand. |
| Halo | Change in sales or contribution outside the directly targeted item. | Assuming every correlated basket item benefits causally. |
| Contribution margin | Net sales less the specified variable costs, in currency per unit or period. | Calling gross revenue profit. |
| Opportunity cost | Value of the displaced alternative use of a constrained resource. | Subtracting a cost already included in the observed control outcome. |
| Guardrail | Prespecified limit on an outcome such as availability or complaints. | Adding a threshold only after results are known. |
| Spillover | Treatment affecting units assigned to control or otherwise outside the treated unit. | Assuming every control is unaffected because its own label says control. |
| Stockout | Period or event in which an item is unavailable to the customer. | Treating missing purchases as evidence of no demand. |
| Incremental return on investment | Incremental net benefit divided by an explicitly defined incremental investment. | Using an undefined denominator or double-counting costs. |
| Estimand | Precisely defined population, treatment contrast, outcome, and time horizon. | Calling a generic metric an estimand without specifying the comparison. |
| Minimum detectable effect | Effect size the planned design has a specified probability of detecting at a specified significance level. | Treating it as a guaranteed business result. |
| Confidence interval | A procedure-derived interval for an unknown effect with a stated long-run coverage property. | Interpreting it as the probability that a fixed parameter lies inside this realized interval. |

## 5. The mechanics, step by step

### Step 1 — Name the decision and alternative

**Input:** a proposed cross-sell intervention and an accountable business owner.
**Output:** one written decision, its alternative, and an evaluation horizon.

Write the intervention as an operational instruction.
For example: place a shelf-edge suggestion for chilled salsa next to tortilla chips in selected stores for four weeks.
The comparison could be the current shelf presentation, not an imaginary world with no marketing.
State whether the decision concerns a single display, a standing recommendation, a coupon, or a permanent layout change.
Specify the stores, channels, dates, and customer journeys to which a decision would apply.

Ask what would happen if the proposal is rejected.
The space may remain empty, carry a different promotion, or return to a regular product.
The proper alternative matters because it determines the incremental benefit.
A high-performing salsa display may still be inferior to the display it displaces.
If the randomized control already retains the alternative display, its measured contribution incorporates that alternative.
Do not subtract the same displaced display value again.

State a financial objective and nonfinancial boundaries.
For example, maximize incremental store contribution over four weeks subject to minimum availability and no material increase in customer complaints.
A traffic-building category may accept a lower immediate margin, but its approved objective should state that trade-off.
Do not replace an agreed strategic objective with attach rate merely because attach is easy to measure.

**Branch:** if there is no implementable intervention, report affinity as exploratory and stop before making a deployment claim.
If the decision owner cannot specify the alternative use of the space, resolve it before approving a financial forecast.

### Step 2 — Define the observational population

**Input:** transaction records, product hierarchy, store list, calendar, and channel definitions.
**Output:** a documented eligible-basket table and a product membership table.

Choose a time window that represents the decision rather than the most favorable promotion.
Define the transaction key at the scope where it is unique, often store plus terminal plus receipt plus date.
For online orders, decide whether the unit is the original order, the fulfilled order, or a delivery.
Explain how edits, substitutions, cancellations, and split fulfillments are handled.
A shopper may have several baskets; a basket count is not a unique-customer count.

Separate completed sales from returns and voids.
A return can occur on another day and reference an earlier receipt.
A defensible initial descriptive extract may use completed positive sales and report returns separately.
If returns are material, link them back to the original sale when measuring realized contribution.
Do not treat a negative line as a second basket containing a positive purchase.

Map items into anchor and partner groups using a versioned hierarchy.
Keep product codes and group memberships effective-dated when packs or classifications change.
Decide whether variants such as mild and hot salsa are pooled.
Pooling can improve sample size but hide price, margin, and customer-need differences.
If a product can belong to both groups, document overlap or remove it; otherwise a single line can create a mechanical pair.

Define exposure to availability as carefully as the data allow.
A partner that is absent from the range cannot be attached in that store.
A partner that is listed but out of stock may be unavailable for part of the day.
A listing flag alone is not an in-stock measure.
Historical availability is often incomplete, so label any in-stock adjustment as an assumption rather than a fact.

**Branch:** if unique transaction keys cannot be reconstructed, repair the data before calculating basket probabilities.
If only loyalty-card baskets are available, report the coverage and avoid assuming they represent all shoppers.
If the population is restricted to stores stocking both groups, say so; the result does not describe the entire estate.

### Step 3 — Construct the 2-by-2 table

**Input:** one row per eligible basket, with Boolean anchor and partner flags.
**Output:** four mutually exclusive basket counts and a reconciliation.

| | Partner present | Partner absent | Total |
|---|---:|---:|---:|
| Anchor present | `n11` | `n10` | `n1.` |
| Anchor absent | `n01` | `n00` | `n0.` |
| Total | `n.1` | `n.0` | `N` |

Count each group once per basket, regardless of how many eligible stock-keeping units (SKUs) or units were purchased.
A basket with two chip bags and three salsa jars contributes one to `n11`.
A basket with chips and no salsa contributes one to `n10`.
An eligible basket with neither item contributes one to `n00`.
All four cells must add to `N`.

Calculate the measures as follows:

```text
attach(partner | anchor) = n11 / n1.
attach(anchor | partner) = n11 / n.1
support(anchor, partner) = n11 / N
baseline(partner) = n.1 / N
lift(anchor -> partner) = (n11 / n1.) / (n.1 / N)
```

Use a percent for rates and a dimensionless number for lift.
If the denominator is zero, the corresponding rate is undefined; do not silently report zero.
If partner prevalence is zero, lift is also undefined.
If the pair count is zero but both marginal counts are positive, the observed attach and lift are zero.

Forward and reverse attach rates have different denominators.
They answer different business questions: how often partner accompanies anchor, and how often anchor accompanies partner.
For the same 2-by-2 table, lift is symmetric even though the conditional rates usually differ.
A large base rate can produce a high attach rate with only modest lift.
A rare partner can produce a high lift from a small number of co-purchases.
Inspect the actual counts, business relevance, and uncertainty before selecting either pair.

**Branch:** if the partner is ubiquitous, consider whether the conditional rate is high merely because most shoppers buy it.
If the pair is rare, pool only genuinely substitutable products or collect more time; do not treat a large ratio as a stable commercial opportunity.

### Step 4 — Describe differences without calling them effects

**Input:** the contingency table and contextual slices.
**Output:** a candidate list with counts, descriptive rates, and explicit limitations.

Compare candidate pairs across store formats, channels, price periods, and mission proxies where those cuts were specified.
Show both the pooled result and a few decision-relevant slices.
A pooled association can arise because large stores sell more of both products, even if the association is weak within each format.
A combination may be frequent during a holiday but not during the proposed intervention period.
A promotion on either item may create a temporary relationship.

Use a minimum pair count or uncertainty screen before ranking candidates.
This avoids allocating shelf space to an unstable high lift from a handful of baskets.
A confidence interval around a descriptive rate summarizes sampling uncertainty under a suitable sampling model; it does not remove confounding.
If many possible pairs were searched, the highest observed rates will often be exaggerated by selection.
Use a later holdout period for descriptive replication and a randomized test for causal validation.

Document the customer need suggested by the pair.
Consider whether the pairing is helpful, redundant, unsuitable, or potentially intrusive.
A pairing may be statistically common because both items are basic staples, not because recommending one improves the trip.
A food allergy, dietary preference, or age restriction can make a generic suggestion inappropriate.
Require human merchandising review before a recommendation is displayed.

**Branch:** if there is no plausible customer benefit or no delivery channel, deprioritize the pair.
If baseline stock availability is unreliable, improve service before interpreting weak attachment as an information problem.

### Step 5 — Specify the intervention and estimand

**Input:** one candidate pair, a feasible intervention, and the business decision.
**Output:** a treatment protocol and a causal estimand.

Define the treatment completely enough that another team can implement it.
For a digital recommendation, state the eligible cart event, placement, number of impressions, message, price, and suppression rules.
For a store display, state its location, facing, signage, product allocation, replenishment, and removal date.
State whether a discount is part of the treatment or whether the recommendation is informational only.
A test that combines placement and discount estimates the bundle, not the isolated effect of either component.

Define the alternative with equal precision.
A control customer may see the existing recommendation rather than a blank space.
A control store may retain its existing endcap.
The estimate is relative to that actual business-as-usual condition.
If the business question concerns the best of several candidate messages, allocate distinct randomized arms and plan for comparisons before launch.

A useful estimand might be: the average difference in four-week store contribution per eligible store between stores assigned the salsa display and comparable stores assigned the current display.
That estimand includes effects on other products, the intended shopper response, and ordinary execution failures.
If stores vary greatly in size, also specify whether the decision values the average store effect or total estate impact.
An equal-weighted store average and a transaction-weighted average answer different questions.
Do not switch between them because one result looks better.

For a digital experiment, eligibility can be defined by a cart containing chips immediately before a recommendation is randomized.
The primary outcome could be incremental contribution per eligible cart within the same order.
If the intervention can change later shopping trips, a longer customer-level horizon may be needed.
Specify whether the outcome includes tax, delivery fees, returns, loyalty costs, and variable fulfillment costs.

**Branch:** if the intervention can alter whether a shopper qualifies for analysis, move eligibility earlier or use a broader randomized population.
If the treatment changes multiple operational features, either accept the bundled estimand or design separate arms.

### Step 6 — Choose a credible assignment and execution design

**Input:** the estimand, operational boundaries, and a list of eligible units.
**Output:** a reproducible assignment list, protocol, and balance checks.

Randomize at the level at which the intervention can actually be controlled.
An endcap visible to everyone in a store usually requires store-level assignment.
A logged-in recommendation may permit customer-level assignment.
If households share devices or shopping accounts, consider whether household-level assignment is necessary.
For repeated visits, keep assignment stable for the unit for the planned horizon.

Match or stratify stores on pre-treatment attributes relevant to the outcome, such as format, geography, and historical contribution.
Randomize within each pair or stratum using a recorded seed.
Pre-period information can improve precision but does not by itself create a causal comparison.
Account for the matched or stratified design in analysis and uncertainty estimation.

Consider spillovers before choosing the unit.
Adjacent stores may share customers, local advertising, or inventory.
A customer may encounter both treated and control stores.
A store-level display may change stock allocation at control stores.
If spillovers are substantial, cluster connected stores or redesign the operational boundary.
If treatment cannot be contained, interpret the estimate as the effect of the actual mixed environment rather than an ideal isolated intervention.

Size the experiment before assignment.
Use historical outcome variability, the number of independent randomization units, the expected intracluster correlation, the planned significance level, power, and the smallest economically meaningful effect.
A store test with thousands of receipts but only a few stores still has limited independent assignment information.
Where there are few stores, use methods appropriate to the randomization scheme and acknowledge limited precision.
Avoid presenting receipt-level standard errors as though each receipt were independently randomized.

Prespecify dates, treatment allocation, exclusions, the primary outcome, secondary outcomes, and a stopping rule.
Do not repeatedly inspect significance and stop at the first favorable result unless the sequential method was planned and calibrated.
Record the assignment before launch and restrict edits to an auditable process.
Separate rollout readiness checks from outcome-based decisions.

**Branch:** if the smallest economically meaningful effect is below the minimum detectable effect, enlarge the test, extend the horizon where appropriate, or accept that the test cannot resolve the decision.
Do not call a nonsignificant test proof of no effect.

### Step 7 — Define economics before looking at results

**Input:** margin definitions, treatment costs, expected displacement, and the relevant comparison.
**Output:** an incremental contribution equation and decision threshold.

For an item-level model, start with incremental net units rather than gross treated units.
Multiply incremental units by contribution per unit, then add or subtract measured effects on other items.
Subtract discounts on all units receiving them, including units that would have sold anyway, if contribution is calculated before discounts.
Subtract incremental labor, materials, spoilage, logistics, platform fees, and other decision-relevant variable costs.
Include the lost contribution of displaced alternatives only when it is not already captured in the measured outcome.

A store-wide contribution outcome can already include sales shifts among shelf locations and items.
If the measurement includes the displaced alternative's contribution, subtracting a modeled cannibalization cost again would double count.
Similarly, a margin field net of the coupon must not have the same coupon deducted a second time.
Write an accounting bridge showing what is included at each stage.

One general bridge for a four-week pilot is:

```text
incremental net contribution
  = incremental net sales across the chosen scope
  - incremental product and fulfillment variable costs
  - incremental treatment and operating costs not already in the outcome
  - uncaptured opportunity cost of displaced resources
```

The term **incremental** means relative to the specified control, not relative to zero sales.
Supplier funding can reduce the retailer's incremental cost only if the contract actually pays for the units or activity in question.
Separate a one-time setup cost from recurring cost when projecting a repeatable rollout.
A sunk analysis cost should not be included in the forward-looking decision unless it changes with the decision.

Set a minimum acceptable effect in currency per randomized unit and over a stated horizon.
Include a contingency for estimation uncertainty and operational risk if the decision process requires one.
A positive point estimate alone is not necessarily sufficient when the downside is costly.
A forecast is a predicted future outcome, a plan is an intended action, a target is a desired result, and a commitment is an authorized obligation.
Do not label a pilot estimate as a guaranteed rollout commitment.

### Step 8 — Run, monitor, and analyze the test

**Input:** the frozen protocol, assignment, execution logs, transactions, and cost records.
**Output:** an intention-to-treat estimate with uncertainty, diagnostics, and guardrail results.

Verify that the assigned treatment is actually deliverable at launch.
Check signage, prices, recommendation rendering, item listing, and inventory.
Record missing displays, late launches, and stockouts without deleting those assigned units from the primary analysis.
Execution metrics explain the treatment received, but a comparison restricted to compliant units can be biased because compliance may be related to store performance.

Follow the prespecified assignment in the primary intention-to-treat analysis.
For store randomization, aggregate or use a model that respects store-level assignment and the original strata.
Compare the primary outcome across assigned groups using the planned weights.
Report the point estimate, interval, number of randomized units, and duration.
Report baseline imbalance and missingness as diagnostics, not as permission to change the primary outcome after the fact.

Use pre-period outcomes for a prespecified precision adjustment if the assumptions and model are appropriate.
A difference-in-differences summary can be helpful, but randomization rather than a historical trend assumption is the source of identification in a randomized trial.
If assignment is not randomized, a difference-in-differences analysis additionally relies on a credible parallel-trends assumption and no relevant concurrent confounding.
Neither method repairs unmeasured interference automatically.

Calculate attach rates as secondary descriptive indicators.
If treatment changes the number or identity of anchor buyers, the post-treatment attach rate can change even without a corresponding customer-level causal response.
A digital cart that met anchor eligibility before assignment is a different situation: the rate among those pre-treatment eligible carts can be a causal outcome if assignment and measurement are valid.
Clearly distinguish the two populations.

Examine stock availability, substitutions, complaints, returns, and workload.
A higher partner conversion accompanied by lower overall contribution or severe stockouts may fail the decision rule.
If multiple secondary metrics and segments are inspected, label them exploratory or apply the planned multiplicity approach.
Do not select a single favorable subgroup after the fact as proof of a rollout opportunity.

### Step 9 — Decide and operationalize

**Input:** the effect estimate, uncertainty, economics, guardrails, and capacity constraints.
**Output:** an authorized decision, implementation plan, or a documented reason to gather more evidence.

Compare the plausible range of effects with the economic threshold.
If the interval contains both an unacceptable loss and a material gain, the evidence may be insufficient even if the point estimate is positive.
If a guardrail is breached, escalate according to the protocol instead of quietly changing the threshold.
If execution failed, distinguish a deployable real-world effect from a question about a fully compliant intervention.
The latter may require a new test rather than an unadjusted compliant-only comparison.

For rollout, specify a store or channel eligibility list, owner, launch date, training, stock cover, reorder logic, and customer safeguards.
Estimate unit demand by location rather than allocating the same quantity to all stores.
Ensure that the control-period alternative is not being displaced differently in the rollout.
Set a review date and a rollback trigger for negative contribution, sustained stockouts, or customer harm.
Record the version of the message, price, product mapping, and eligibility logic.

If the result is negative, preserve the learning.
An ineffective message may still reveal a promising pair for a different, separately tested intervention.
If the test is inconclusive, state which missing precision or execution evidence prevents a decision.
Do not turn inconclusive into a claim of success or failure.

### Implementation worksheet A — compare candidate interventions

The same observed pair can motivate several materially different actions.
Select a treatment because it addresses a plausible customer friction, not merely because the pair has high lift.
For each option below, record the business-as-usual alternative and the capacity it uses.

**Adjacent shelf placement.**
The proposed action moves the partner closer to the anchor.
The customer friction is the effort of finding a complementary product.
The relevant alternative is the existing shelf allocation, including products that currently occupy the space.
The practical constraint may be refrigeration, shelf load, or physical separation required by store policy.
Measure the effect across both product groups and any displaced assortment.
Do not assume that units picked from the new location represent new demand.

**Secondary display.**
The proposed action adds a small presentation while retaining the partner's normal shelf location.
The customer friction is visibility rather than the existence of an assortment.
The incremental inputs include fixture use, labor, replenishment, and any increased spoilage.
The alternative may be another display rather than an empty floor.
The inventory team must know whether the display draws from a common backroom stock pool.
A backroom shortage can empty the regular shelf even when the secondary fixture looks full.

**Digital recommendation.**
The proposed action presents a partner after a qualifying event.
The customer friction may be search effort or a forgotten complementary need.
The alternative could be another recommendation selected by the existing algorithm.
Record impression opportunity, message, position, eligibility, and availability at assignment.
A recommendation never rendered because the page failed is still part of the assigned-policy effect.
If a shopper adds and later removes the partner, decide whether the outcome is cart addition or completed purchase.
For contribution decisions, completed and fulfilled orders are generally more informative than clicks alone.

**Conditional coupon.**
The proposed action discounts a partner when the qualifying condition is met.
The customer friction may be price, but the discount can also simply reduce revenue on a planned purchase.
The control must specify whether it receives the normal price or an alternative offer.
The retailer must establish whether the coupon applies to one unit, all units, or a limited quantity.
The cost ledger must account for redemption, supplier funding, and applicable fulfillment cost.
If coupon eligibility is based on an anchor purchased after assignment, the analysis must not condition solely on eventual purchasers.
Randomizing the offer before the final basket preserves an interpretable assigned-policy comparison.

**Staff prompt.**
The proposed action asks a cashier or associate to suggest the partner.
The treatment includes training, timing, wording, and any restriction on who receives the prompt.
The operational capacity is employee time as well as the partner's availability.
An apparently free prompt may lengthen queues and reduce service quality.
Monitor queue time and customer complaints if those outcomes are material to the decision.
Do not measure only the transactions where staff chose to deliver the prompt.
Such transactions may differ systematically from transactions where staff did not ask.

### Implementation worksheet B — make the intervention comparable

Prepare the following information before assigning units.
Each item has an owner and an expected evidence artifact.

1. **Policy owner.**
   The category manager names the intervention and the normal alternative.
   The artifact is a signed treatment specification with a version number.
2. **Population owner.**
   The analytics team defines eligible stores or customers before outcome collection.
   The artifact is a frozen eligibility extract and a timestamp.
3. **Assignment owner.**
   The experiment team specifies randomization units, strata, and seed.
   The artifact is an assignment file protected from silent modification.
4. **Execution owner.**
   Store operations or the digital team translates the specification into an executable change.
   The artifact is an installation or release checklist, including the planned end date.
5. **Supply owner.**
   Replenishment confirms item listing, lead time, and minimum stock requirements.
   The artifact is a store-level availability and replenishment plan.
6. **Measurement owner.**
   Finance and analytics agree on the contribution definition and its source systems.
   The artifact is a cost and revenue reconciliation with a named reporting period.
7. **Customer safeguards owner.**
   The appropriate privacy, legal, safety, and customer-experience teams review the intended use.
   The artifact records restrictions and the authority to stop.
8. **Decision owner.**
   The sponsor approves a threshold, uncertainty rule, and review date.
   The artifact states what happens for success, failure, or an inconclusive estimate.

The checklist is a control against ambiguous handoffs, not a claim that governance establishes causality.
If assignment changes after launch, retain both the original and actual records with reasons and timestamps.
If a store is permanently closed during the trial, use the prespecified missing-data treatment rather than inventing its outcome.
Distinguish a truly missing measurement from an observed zero contribution.
A store with no sales is not automatically a store with no data.

### Implementation worksheet C — interpret a rate change

Consider a fictional digital checkout experiment with an anchor already in the cart before assignment.
Suppose 1,000 eligible carts are assigned to a partner recommendation and 1,000 to the existing recommendation.
Assume 250 treated carts and 200 control carts purchase the partner in the completed order.
The observed partner conversion rates are 25% and 20% of pre-assignment eligible carts, respectively.
The unadjusted difference is 25% - 20% = 5 percentage points.
Equivalently, the treated rate is 1.25 times the control rate in this sample.
Neither calculation alone gives the uncertainty interval or net contribution.

Under a valid randomized design, the five-percentage-point difference estimates the assigned-policy effect for the eligible-cart population.
It is not necessarily the effect of a displayed recommendation among people who actually saw it.
Some assigned recommendations may not render.
Some control customers may find the partner without a recommendation.
The intention-to-treat comparison includes both possibilities.

Now consider a different design in which the recommendation appears before the customer decides whether to buy the anchor.
Suppose the analyst calculates partner attachment only among final orders containing the anchor.
The numerator and denominator may both be affected by assignment.
For example, the recommendation could attract new anchor buyers who have different prior partner interest.
The resulting change in conditional attachment combines behavior change with a change in the composition of anchor buyers.
It is therefore not automatically the causal effect for a fixed group of anchor buyers.
One remedy is to define eligibility using a pre-assignment anchor signal when the journey permits it.
Another is to evaluate total contribution among all randomized eligible shoppers.

The chosen outcome must follow the decision.
If the decision is which recommendation occupies a slot, compare policies including the recommendation displaced in control.
If the decision is whether to make a product easier to find, consider availability and total category demand.
If the decision is whether to offer a coupon, include discounts paid on purchases that would have occurred without the coupon.
If the decision is whether to retain an assortment item, a recommendation test alone does not identify the effect of delisting it.

### Implementation worksheet D — separate the decision horizon

A cross-sell can shift the timing of a purchase without increasing cumulative demand.
For example, a shopper may purchase two packages of filters today and skip the usual purchase next month.
A one-week experiment that measures only the first order can overstate the longer-horizon gain.
Choose a follow-up period informed by the partner's purchase cycle and expected returns.
If a longer horizon is unavailable, explicitly describe the effect as a short-horizon result.

The unit of customer linkage also matters.
A shopper may buy the partner through a different channel after seeing the recommendation.
If those channels can be linked lawfully and accurately, include the agreed outcomes within the stated horizon.
If they cannot be linked, report the limitation rather than assuming the missing purchases are equal across arms.
Do not combine identifiable customer data from separate systems without an approved purpose and appropriate safeguards.

A four-week display may also create a temporary operational burden.
The first week can include installation and training.
The final week can include dismantling or markdowns.
If a later rollout reuses fixtures, its recurring cost differs from the pilot's first-use cost.
Document the basis for each scenario instead of extrapolating a pilot cost mechanically.
The scope of the measured outcome must remain consistent when comparing scenarios.

### Implementation worksheet E — reconcile contribution and capacity

Use the following ledger to avoid treating every attractive recommendation as independently scalable.
Suppose a recommendation slot can display one partner per eligible cart.
Candidate A has an estimated incremental net contribution of USD 0.08 per eligible cart against the current slot.
Candidate B has an estimated incremental net contribution of USD 0.05 per eligible cart against the same current slot.
These are hypothetical estimates over the same population, horizon, and accounting scope.
The slot cannot display both simultaneously without a new design.
Choosing A over the current slot has an estimated benefit of USD 0.08 per cart, not USD 0.13 per cart.
The comparison between A and B is an estimated USD 0.03 per cart only if the effects are sufficiently comparable and the relevant uncertainty is addressed.
An experiment directly randomizing A versus B can answer that comparison more efficiently than combining unrelated pilots.

Physical capacity creates a similar constraint.
A display can occupy a fixture that would otherwise host another promotion.
If the control retains that promotion, a store-wide contribution comparison captures its value.
If the control fixture is empty but rollout displaces a promotion, the pilot does not capture the rollout opportunity cost.
Estimate the additional cost or test against the actual deployment alternative.
If the candidate uses a scarce chilled fixture, availability of ambient space is not a substitute for that constraint.

Before multiplying a per-unit effect by an estate count, check whether the effect applies to the relevant units.
A pilot drawn from high-traffic stores may not represent smaller sites.
A product with finite supplier capacity may not sustain the pilot's stock availability at scale.
A supplier-funded discount may have a redemption cap that changes marginal economics.
A digital recommendation may produce diminishing returns if the same customer sees it repeatedly.
Document these risks in the rollout forecast and monitor them after release.

The forecast should include a central estimate and decision-relevant downside scenarios.
The plan should identify eligible sites and implementation dates.
The target should specify the desired economic or service outcome without relabeling it as evidence.
The commitment should identify approved spend and accountable authority.
These four artifacts can differ even when they are presented in the same business review.

## 6. Worked example

### Business setting

A regional grocer is considering a four-week salsa shelf-edge display beside tortilla chips.
The alternative is its current salsa shelf and existing endcap arrangement.
The proposed display uses a small shelf-edge fixture rather than a price discount.
The retailer wants incremental contribution, not simply more paired baskets.
Figures below are fictional and stated in US dollars (USD).

The category manager first studies 20,000 eligible completed baskets in comparable stores over four weeks.
Each basket is counted once, regardless of the number of chip bags or salsa jars.
Chip and salsa membership are disjoint product groups.
Returns are excluded from this descriptive table and are incorporated separately in the financial outcome.
The table covers only stores listing both groups; no claim about out-of-stock intervals is made.

| | Salsa present | Salsa absent | Total baskets |
|---|---:|---:|---:|
| Chips present | 1,260 | 3,240 | 4,500 |
| Chips absent | 1,740 | 13,760 | 15,500 |
| Total baskets | 3,000 | 17,000 | 20,000 |

The count reconciles: 1,260 + 3,240 + 1,740 + 13,760 = 20,000 baskets.
The forward attach rate is 1,260 / 4,500 = 28.0% of chip baskets.
The reverse attach rate is 1,260 / 3,000 = 42.0% of salsa baskets.
Pair support is 1,260 / 20,000 = 6.3% of all eligible baskets.
Baseline salsa incidence is 3,000 / 20,000 = 15.0% of all eligible baskets.
Lift is 0.28 / 0.15 = 1.867, or approximately 1.87 times baseline.

A shopper buying chips is therefore more likely to buy salsa in the observed data than an average eligible basket is.
That statement is descriptive.
It does not say that placing salsa beside chips causes a 13 percentage-point increase in salsa purchases.
Shared party occasions, existing displays, promotions, and store mix could affect both items.
The pair is a candidate for a test, not proof of incremental benefit.

### Trial design

The retailer selects 40 eligible stores with consistent reporting and sufficient fixture capacity.
It matches stores into 20 pairs using pre-period format and contribution, then randomly assigns one store in each pair to the fixture and one to business as usual.
The assignment unit is the store; all customers in a store encounter its assigned condition.
The four-week outcome is store contribution including product margin, substitutions, and returns but excluding fixture and incremental setup costs.
The decision compares the average paired store difference and later subtracts those excluded costs.

The primary estimand is the average four-week contribution difference per assigned store under fixture assignment versus current practice in this eligible store population.
The same stores remain in their assigned group even if a fixture arrives late.
The team records execution and partner availability separately.
Because the randomization is paired by store, uncertainty must use the 20 independent pair differences or an analysis consistent with that randomization.
The many receipts within a store do not become independent treatment assignments.

The trial is illustrative; the following results are assumed, not observed from a real company.
A valid experiment would use its actual paired differences to calculate the interval.
Assume the average treated-minus-control difference is USD 310 per store over four weeks.
Assume a design-consistent 95% confidence interval for that difference is USD 90 to USD 530 per store.
This interval is given for the teaching example; it cannot be reconstructed from the group average alone.
The outcome already includes the contribution of other items and their displacement under the control arrangement.

The fixture costs USD 75 per treated store for the four-week pilot, including installation and removal.
The fixture cost is not included in the contribution outcome above.
Therefore the estimated net effect is USD 310 - USD 75 = USD 235 per store over four weeks.
If fixture cost is known and fixed at USD 75, the corresponding interval after that cost is USD 15 to USD 455 per store.
Do not subtract a further estimate for displaced sales if those sales are already in the store-wide contribution outcome.
Do not claim the intervention causes every observed salsa attachment.

Suppose the retailer's prespecified financial threshold is at least USD 100 incremental net contribution per store over four weeks.
The point estimate of USD 235 exceeds the threshold, but the interval also includes USD 15, which is below it.
If the decision rule requires the lower interval bound to exceed the threshold, this trial does not meet that rule.
If the approved rule permits a risk-based decision using the point estimate and downside, the owner must document why the uncertainty is acceptable.
In either case, the result is not automatically a rollout mandate.

### Operational review

Assume fixture compliance was 18 of 20 treated stores at launch, with two late installations.
The primary estimate retains all 20 assigned treated stores.
A compliant-only result would be vulnerable to selection because stores with reliable installation may also perform differently.
The team can report the installation rate and reasons as diagnostics.
A separate estimate of a perfectly executed fixture effect would need additional assumptions or a new design.

Assume availability is within the prespecified guardrail in both arms and no customer safety issue is reported.
That does not prove availability will remain adequate at a larger scale.
The replenishment team should use store-level sales and lead-time information to plan a larger test or staged rollout.
The plan must cover fixture removal and restoration of the original shelf if the test stops.

The historical 28.0% attach rate is a baseline description, not the outcome used for the store-level causal estimate.
The USD 235 per-store estimate is a test result under the pilot conditions.
A future estate forecast would require assumptions about store mix, capacity, costs, and transferability.
A rollout plan would name stores and dates.
A target would state the desired contribution.
A commitment would be the authorized expenditure or contractual obligation, not the historical estimate.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It recreates the descriptive table and the contribution bridge above.
Each input row represents one group of distinct eligible baskets, not product units.
The code deliberately does not calculate a confidence interval because the example does not contain the 20 paired store differences.
Replace the synthetic counts and test estimate only after applying the population and accounting rules in sections 5 and 8.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BasketCounts:
    both: int
    anchor_only: int
    partner_only: int
    neither: int

    def __post_init__(self) -> None:
        values = (self.both, self.anchor_only, self.partner_only, self.neither)
        if any(value < 0 for value in values):
            raise ValueError("Basket counts must be nonnegative")

    @property
    def total(self) -> int:
        return self.both + self.anchor_only + self.partner_only + self.neither


def ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def summarize(counts: BasketCounts) -> dict[str, Optional[float]]:
    anchor = counts.both + counts.anchor_only
    partner = counts.both + counts.partner_only
    forward = ratio(counts.both, anchor)
    reverse = ratio(counts.both, partner)
    support = ratio(counts.both, counts.total)
    baseline = ratio(partner, counts.total)
    lift = (
        forward / baseline
        if forward is not None and baseline not in (None, 0)
        else None
    )
    return {
        "forward_attach": forward,
        "reverse_attach": reverse,
        "support": support,
        "partner_baseline": baseline,
        "lift": lift,
    }


def show_rate(value: Optional[float]) -> str:
    return "undefined" if value is None else f"{value:.1%}"


def main() -> None:
    counts = BasketCounts(
        both=1260,
        anchor_only=3240,
        partner_only=1740,
        neither=13760,
    )
    metrics = summarize(counts)
    print(f"Eligible baskets: {counts.total:,}")
    print(f"Salsa on chip baskets: {show_rate(metrics['forward_attach'])}")
    print(f"Chips on salsa baskets: {show_rate(metrics['reverse_attach'])}")
    print(f"Pair support: {show_rate(metrics['support'])}")
    print(f"Salsa baseline: {show_rate(metrics['partner_baseline'])}")
    lift = metrics["lift"]
    print(f"Lift: {'undefined' if lift is None else f'{lift:.2f}x'}")

    # These are assumed design-consistent store-level results in USD per store.
    effect = 310.0
    interval_low, interval_high = 90.0, 530.0
    fixture_cost = 75.0
    net_effect = effect - fixture_cost
    net_low = interval_low - fixture_cost
    net_high = interval_high - fixture_cost
    print(f"Net contribution: USD {net_effect:.0f} per store")
    print(f"Net 95% interval: USD {net_low:.0f} to {net_high:.0f} per store")


if __name__ == "__main__":
    main()
```

Expected output:

```text
Eligible baskets: 20,000
Salsa on chip baskets: 28.0%
Chips on salsa baskets: 42.0%
Pair support: 6.3%
Salsa baseline: 15.0%
Lift: 1.87x
Net contribution: USD 235 per store
Net 95% interval: USD 15 to 455 per store
```

The interval subtraction assumes the USD 75 cost is known and fixed.
If cost varies by store or is estimated, include its uncertainty in the analysis rather than simply shifting the endpoints.
The code does not perform randomization, ingest receipts, estimate a causal effect, or validate execution.
Production analysis should preserve the randomized assignment, stratification, and audit trail.
The sample counts are nonnegative integer basket counts; the class checks nonnegativity, while the caller is responsible for supplying integer, mutually exclusive counts.

## 8. Data, standards, and regulatory context

### Transaction grain and lineage

A useful basket dataset contains a transaction identifier, store, channel, transaction time, item identifier, quantity, net price, discounts, and return or void status.
A product master maps item identifiers to the versioned anchor and partner definitions.
A store master identifies format, operating dates, and range eligibility.
An inventory source records availability at a time resolution appropriate to the analysis.
A promotion calendar records price and placement changes that could affect the baseline.

Transaction identifiers may be unique only within a store, terminal, or day.
A compound key should be tested for collisions before building baskets.
Time zones and trading-day boundaries can change which receipt belongs to the study window.
Online orders may be edited after checkout; use a documented snapshot and link fulfillment events to the original order where possible.
A partial fulfillment can produce different ordered and delivered baskets.
Choose the one that matches the decision and state the choice.

Deduplicate item membership within basket before constructing a co-occurrence table.
Keep quantities and values in a separate fact table for economic calculations.
A many-to-many hierarchy join can otherwise multiply lines and create false pairs.
Test category membership on effective dates and investigate unmatched or duplicated product codes.
Do not join returns to a receipt solely by matching customer and date if the actual original sale reference is available.

### Product identifiers and standards

GS1 defines the Global Trade Item Number (GTIN) for identifying trade items.
A GTIN helps exchange and match product identities across organizations, but it is not itself a guarantee that a retailer's product hierarchy or transaction data is correct.
Different trade-item configurations can have different identifiers; pack changes and private-label substitutions require careful mapping.
Use the GS1 General Specifications and the relevant organizational implementation rules when designing identification processes.
GS1 is a standards organization; use of a GS1 identifier is not a universal legal requirement for every retail transaction.

A Universal Product Code (UPC) is a barcode-related identifier commonly encountered in North American retail; treat the stored code according to the applicable identifier scheme and data format.
Preserve leading zeros and validate the mapping rather than converting identifiers to floating-point numbers.
A store's internal SKU may map to one or more GTINs over time.
The analysis needs an effective-dated mapping, not an assumption that one identifier remains unchanged forever.

### Privacy and responsible use

Basket analysis can be performed with anonymous transaction identifiers when customer-level linkage is unnecessary.
Customer identifiers make longitudinal analysis possible but increase privacy and security obligations.
Limit access, retain identifiers only for the approved period, and separate identity resolution from analysis where practical.
Aggregate results can still expose sensitive information when groups are small; apply the organization's disclosure controls.
Hashing an identifier alone does not make data anonymous if it can still be linked to a person.

The European Union General Data Protection Regulation (GDPR) applies to processing of personal data within its scope.
Its principles include purpose limitation, data minimization, and storage limitation; it also provides rights and obligations subject to the regulation's conditions and exceptions.
A retailer must identify a valid legal basis and assess relevant obligations for its own processing.
A cross-sell analysis should not assume that an existing loyalty identifier automatically authorizes every new use.
Whether a particular purchase inference involves special-category data requires context-specific legal assessment.

In California, the California Consumer Privacy Act as amended by the California Privacy Rights Act gives covered consumers certain rights and imposes duties on covered businesses, subject to statutory definitions and exceptions.
The California Privacy Protection Agency provides regulations and guidance.
The applicability of notice, opt-out, or other requirements depends on the business, data use, and transaction.
These statements describe the frameworks as of November 2026, not a determination of compliance.
Consult current law and qualified counsel before deployment.

Marketing channels can impose additional consent, accessibility, and platform requirements.
Products subject to age restrictions or safety rules require relevant screening before recommendation.
A model should not recommend a substitute that defeats a shopper's stated allergy or dietary exclusion.
Sensitive inferences and vulnerable customers require particular care even where an aggregate pairing is statistically strong.

### Measurement controls

Store the assignment file separately from the outcome extract with access controls and version history.
Record the eligibility timestamp, assignment timestamp, treatment version, delivery logs, and outcome horizon.
An impression event is evidence that a system attempted to display a recommendation, not necessarily that a person saw it.
Measure missing logs by arm; differential missingness can bias comparisons.
Maintain separate identifiers for randomized unit, exposure event, transaction, and product.

Reconcile financial totals to the retailer's accounting definitions.
Specify whether tax, vendor funding, returns, discounts, cost of goods, delivery, and handling are included.
Decide whether a margin measure is realized or estimated and whether cost updates are applied retrospectively.
Record currency, exchange-rate convention if applicable, and time bucket.
A financial reconciliation does not establish causality, but it prevents an otherwise valid effect from being valued on inconsistent inputs.

## 9. Industry practice and case evidence

### Retail recommendation at scale

Linden, Smith, and York describe Amazon.com's item-to-item collaborative filtering approach in an IEEE Internet Computing article published in 2003.
The paper explains how item relationships can support recommendations and discusses scaling the computation for a large catalog.
It is a public description of a recommendation method, not evidence that a particular chip-and-salsa display is profitable.
The item relationships reflect customer activity and the platform's historical environment.
A retailer transferring the idea must still define eligibility, available inventory, customer relevance, and a causal evaluation.

**Inference:** item-to-item candidate generation can reduce the number of recommendations that need to be evaluated in a large assortment.
It does not remove selection bias, guarantee individual relevance, or estimate the counterfactual purchase without a recommendation.
A subsequent randomized comparison is needed for a causal claim about the proposed recommendation policy.

### Randomized online experimentation

Kohavi and coauthors describe controlled experiments on the web, including practical issues of randomization, metrics, and interpretation.
Their work documents experimentation practice across online settings rather than a universal effect size for cross-selling.
An online retailer can adapt the principles by assigning eligible users before the recommendation and keeping the unit stable across repeated visits.
It must still check interference, logging quality, and whether the chosen outcome captures substitution and cost.

**Inference:** an increase in clicks on a recommended item is insufficient if the recommendation displaces a better item or reduces overall contribution.
The experimental outcome should match the business decision rather than the most convenient interaction event.

### Association-rule evidence

Agrawal, Imieliński, and Swami introduced mining association rules between sets of items in large databases using a retail transaction motivation.
The contribution is an efficient way to discover patterns such as items appearing together.
The rule itself is an observational association.
It does not identify a display's effect, a customer's intent, or the margin generated by a cross-sell.

A buyer can use these rules to propose coherent merchandising opportunities.
A finance partner then defines the incremental economics.
An operations partner checks fixture and stock feasibility.
An experimentation partner designs a comparison with appropriate units and uncertainty.
These are complementary responsibilities rather than interchangeable metrics.

### External validity

A public case may use a different catalog, store format, time period, recommendation placement, or customer population.
Its result should not be inserted as a forecast for a new retailer without a transfer argument.
Even a successful internal pilot may not generalize to stores with lower availability or different shopper missions.
Specify which operational features must remain similar and which may change during rollout.
Collect follow-up evidence when the policy, supplier cost, price, or customer journey changes materially.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong basket key | Multiple unrelated receipts share an identifier. | Identifier is unique only within a store or date. | False co-purchases and misleading lift. | Construct and reconcile a unique composite key. |
| Unit counts treated as baskets | Pair counts exceed anchor basket counts. | Multiple units or lines are counted repeatedly. | Impossible or inflated rates. | Deduplicate item-group membership per transaction. |
| Returns treated as purchases | Negative-value receipts appear in the basket extract. | Return lines are included as positive incidence. | Misstated attachment and economics. | Separate returns and link them to original sales where feasible. |
| Direction reversed | The reported rate changes when the table is transposed. | Anchor and partner denominators are confused. | Wrong operational interpretation. | Label the conditional rate and show both marginal counts. |
| High lift on tiny counts | A top-ranked pair has only a few joint baskets. | A rare base rate magnifies sampling noise. | Wasted test or fixture allocation. | Require meaningful counts and replicate before testing. |
| Ubiquitous partner | High attach but lift is close to one. | Partner is common in nearly all baskets. | Recommendation targets normal behavior. | Show baseline partner incidence alongside attach. |
| Mission confounding | Pairing peaks on holidays or in a particular store format. | A common occasion causes both purchases. | Association is mistaken for incremental effect. | Randomize the intervention; slice baseline descriptively. |
| Price confounding | Both items were promoted during the observed window. | Promotion influences both purchases. | Inflated candidate strength. | Document price periods and use randomized comparisons. |
| Post-treatment selection | Analysis includes only customers who bought the anchor after exposure. | Treatment changes who enters the denominator. | Biased causal contrast. | Define eligibility before assignment or analyze all assigned units. |
| Stockout masking | Treated stores have lower partner availability. | Successful promotion depletes inventory. | Understates feasible demand and harms shoppers. | Monitor availability and plan replenishment; retain intention-to-treat analysis. |
| Store clustering ignored | Receipt count is large but few stores were randomized. | Outcomes within stores are correlated. | Intervals are too narrow. | Analyze at or account for the randomized store or cluster. |
| Spillover | Shoppers visit both treatment and control stores. | Control receives indirect treatment exposure. | Contrast is diluted or otherwise altered. | Cluster connected stores or explicitly define the mixed-policy estimand. |
| Compliance filtering | Only stores with correctly installed displays are analyzed. | Execution is related to store characteristics. | Selection bias. | Keep all assigned stores in the primary intention-to-treat analysis. |
| Cannibalization omitted | Partner sales rise while category margin falls. | The new item displaces other profitable items. | A loss-making policy appears successful. | Measure contribution across the relevant category or store. |
| Cost double counted | Margin is net of coupons and coupons are subtracted again. | Accounting scopes overlap. | Understates contribution. | Document the bridge and reconcile each cost exactly once. |
| Opportunity cost omitted | The proposed fixture replaces an established profitable display. | The control alternative is ignored. | Overstates incremental value. | Use the actual alternative in the control or explicitly price its displacement. |
| Repeated peeking | The test stops immediately after one favorable daily readout. | Repeated unadjusted significance checks inflate false positives. | Unreliable deployment. | Prespecify duration or use a calibrated sequential design. |
| Segment mining | Only one of many slices is favorable. | Multiple comparisons create apparent winners. | Unreliable localization. | Prespecify key segments and treat other cuts as exploratory. |
| Privacy overreach | Individual predictions are retained without a clear purpose. | Linkable behavior is reused beyond the approved scope. | Customer and regulatory harm. | Minimize data and review legal basis, retention, and access. |
| Unsupported rollout | Pilot effect is applied unchanged to all stores. | Store mix, supply, and execution differ. | Benefits fail to materialize. | Stage rollout and monitor transfer assumptions and guardrails. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or manage it |
|---|---|---|---|
| Basket keys uniquely identify eligible transactions. | Counts and probabilities require a coherent unit. | Reused receipt numbers, split orders, or duplicate ingestion. | Reconcile keys and sample transactions against source records. |
| Product grouping is accurate and effective-dated. | Anchor and partner flags must represent the intended items. | Stale hierarchy, pack changes, or overlapping groups. | Audit mappings and test unmatched and overlapping identifiers. |
| The historical population matches the described scope. | Descriptive rates refer only to included baskets. | Loyalty-only coverage, excluded channels, or selected stores. | Report coverage and compare observable included and excluded populations. |
| Assignment is random within the stated design. | Randomization supports comparable potential outcomes in expectation. | Manual reassignment, predictability, or allocation bugs. | Preserve the seed and assignment log; audit deviations and baseline balance. |
| Outcomes are measured consistently by arm. | A difference must represent behavior rather than data capture. | Differential logging failure, missing costs, or changed systems. | Reconcile data and compare missingness by assignment. |
| Interference is negligible or explicitly included in the estimand. | Unit outcomes should correspond to the assigned policy contrast. | Cross-store visits, shared advertising, or inventory transfers. | Cluster connected units or interpret a mixed-policy effect. |
| The outcome captures relevant substitution. | Cross-selling economics depend on displaced and halo contribution. | Measuring only the promoted item. | Use a wider category or store outcome and an accounting bridge. |
| Costs are complete and not duplicated. | Net contribution must match the decision boundary. | Coupons already netted in revenue or omitted fixture labor. | Reconcile finance fields and review the cost ledger. |
| The sample supports the planned precision. | A decision requires an interval informative relative to the threshold. | Too few randomized stores or high store variability. | Conduct design-based power planning and report the actual interval. |
| Operational conditions are transferable. | Rollout benefit depends on similar eligibility, execution, and supply. | Different formats, staff capacity, product cost, or shopper missions. | Stage deployment and monitor outcomes by prespecified context. |
| The evaluation horizon captures important effects. | Short-run sales may omit returns or delayed substitution. | Stockpiling, delayed returns, or longer customer response. | Extend follow-up where feasible and disclose remaining omissions. |
| Fixed-cost interval adjustment is appropriate. | Shifting interval endpoints assumes a known fixed cost. | Costs vary by store or are estimated with uncertainty. | Include measured cost variation in the store-level outcome or uncertainty analysis. |

These assumptions are not a checklist that makes an observational correlation causal.
Randomization, consistency of the intervention, measurement quality, and limited interference jointly support an interpretation of the assigned-policy effect.
The result remains specific to the tested population and period unless a defensible transport argument is made.

## 12. Exercises

### Exercise 1 — Define the denominator

A team reports that 30% of cereal baskets contain milk.
It describes this as “30% incremental milk sales from cereal.”

- **Deliverable:** Two sentences defining the actual observed quantity and a separate causal question.
- **Acceptance criteria:** Name cereal baskets as the denominator and explain why co-purchase does not establish incremental sales.
- **Time:** 8 minutes.

### Exercise 2 — Calculate directional attachment

In 10,000 eligible baskets, 2,000 contain coffee, 1,500 contain filters, and 600 contain both.
Each basket is counted once.

- **Deliverable:** The four contingency-table cells, both attach rates, pair support, baseline filter rate, and lift for coffee to filters.
- **Acceptance criteria:** The cells sum to 10,000 baskets; each rate has the correct denominator and lift is dimensionless.
- **Time:** 12 minutes.

### Exercise 3 — Critique a causal claim

A retailer observes high sunscreen and cold-drink affinity during summer and proposes a drink display beside sunscreen.
The data include several beach stores and an overlapping drink promotion.

- **Deliverable:** A causal sketch and three reasons the association may not identify the display's incremental effect.
- **Acceptance criteria:** Include at least one common cause, one measurement or selection issue, and a proposed randomized comparison.
- **Time:** 15 minutes.

### Exercise 4 — Build an economic bridge

A randomized test estimates 400 incremental partner units over four weeks.
Each unit contributes USD 2.50 before a treatment coupon.
The retailer pays a USD 0.40 coupon on 1,500 treated units, including baseline units.
Incremental contribution from displaced products is negative USD 150.
Fixture and labor cost USD 120 and are not included in the other figures.

- **Deliverable:** Net incremental contribution in USD over four weeks, with each arithmetic step.
- **Acceptance criteria:** Apply the coupon to all 1,500 treated units, count displaced contribution once, and subtract fixture and labor once.
- **Time:** 12 minutes.

### Exercise 5 — Design a store test

A chain can place a non-discounted soup display next to bread in 30 of 60 eligible stores.
Store traffic varies widely and some neighboring stores share shoppers.

- **Deliverable:** A one-page test protocol.
- **Acceptance criteria:** Define the control, store or cluster randomization, pre-treatment stratification, estimand, primary outcome, power inputs, spillover response, guardrails, and stop rule.
- **Time:** 25 minutes.

### Exercise 6 — Interpret assignment and exposure

A recommendation is randomly assigned to 5,000 eligible customers and control to another 5,000.
Only 3,500 assigned customers are logged as exposed.
The analyst proposes comparing those 3,500 customers with all 5,000 controls.

- **Deliverable:** A short critique and the appropriate primary comparison.
- **Acceptance criteria:** Explain post-assignment selection, retain all randomized eligible customers for intention to treat, and identify exposure logging as a diagnostic rather than the primary denominator.
- **Time:** 12 minutes.

### Exercise 7 — Make a threshold decision

A pilot estimates USD 180 net contribution per store over four weeks, with a 95% interval from negative USD 40 to positive USD 400.
The prespecified rule requires the lower interval bound to exceed USD 50 per store.
The stockout guardrail also fails.

- **Deliverable:** A decision memo of at most 150 words.
- **Acceptance criteria:** Apply the specified threshold and guardrail, state the uncertainty honestly, and propose a feasible next action without calling the result a proven loss.
- **Time:** 12 minutes.

### Exercise 8 — Prepare operations and data governance

The business wants to roll out a personalized cross-sell using loyalty identifiers and limited stock.

- **Deliverable:** A deployment checklist with owners, data fields, inventory controls, customer safeguards, monitoring, and rollback.
- **Acceptance criteria:** Include pre-assignment eligibility, identifier minimization, legal review, treatment-version logging, stock coverage, a measurable guardrail, and an accountable stop authority.
- **Time:** 20 minutes.

## 13. Self-check answers

### Exercise 1

The observed quantity is the number of eligible cereal baskets that contain milk divided by the number of eligible cereal baskets.
It is a conditional co-purchase rate, not an estimate of milk sales caused by cereal or by an intervention.
A causal question would compare milk contribution among eligible shoppers assigned a cereal-linked milk offer with the contribution among comparable shoppers assigned the specified alternative.
The relevant unit, horizon, and treatment must be stated.

### Exercise 2

The both cell is 600 baskets.
Coffee only is 2,000 - 600 = 1,400 baskets.
Filters only is 1,500 - 600 = 900 baskets.
Neither is 10,000 - 600 - 1,400 - 900 = 7,100 baskets.

Coffee-to-filter attach is 600 / 2,000 = 30% of coffee baskets.
Filter-to-coffee attach is 600 / 1,500 = 40% of filter baskets.
Pair support is 600 / 10,000 = 6% of eligible baskets.
The baseline filter rate is 1,500 / 10,000 = 15% of eligible baskets.
Lift is 0.30 / 0.15 = 2.0, a dimensionless ratio.
The lift describes the observed data and does not establish a causal effect.

### Exercise 3

A suitable sketch has hot weather or beach mission pointing to both sunscreen and drink purchase.
Store format can also affect assortment and customer mix.
The drink promotion directly affects drink purchases and may coincide with sunscreen promotion or summer traffic.
A store-level randomized display test against current practice, with promotion and inventory documented, would address the proposed intervention more directly.
If the display is visible to every shopper in a store, assigning individual receipts independently is not a feasible treatment design.

### Exercise 4

Incremental partner contribution before coupons is 400 units × USD 2.50 per unit = USD 1,000.
Coupon cost is 1,500 units × USD 0.40 per unit = USD 600.
Displaced contribution is negative USD 150.
Fixture and labor cost USD 120.
Net incremental contribution is USD 1,000 - USD 600 - USD 150 - USD 120 = USD 130 over four weeks.
The answer assumes these inputs are separately measured and do not overlap.
If the USD 2.50 margin already included the coupon, the separate USD 600 deduction would double count it.

### Exercise 5 rubric

A strong protocol defines business as usual, the precise display, eligible stores, four-week or other justified horizon, and a store-contribution estimand.
It randomizes stores within prespecified strata or clusters neighboring stores if shared shoppers make interference important.
It uses independent assignment units in power and interval calculations.
It records execution and stockouts while retaining assigned units in the primary analysis.
It states a threshold, guardrails, and a stopping process before outcomes are inspected.
A weak protocol compares only compliant treated stores with all controls or treats every receipt as an independent randomization.

### Exercise 6

The 3,500 exposed customers are selected after assignment.
Exposure may depend on device, attention, site behavior, or logging, which can also predict spending.
The primary intention-to-treat comparison includes all 5,000 customers assigned treatment and all 5,000 assigned control, using the prespecified eligible population and outcome horizon.
The exposure rate is 3,500 / 5,000 = 70% of assigned treatment customers, subject to logging completeness.
It is a delivery diagnostic, not proof of the causal effect among those exposed.

### Exercise 7

The lower bound of negative USD 40 per store does not exceed the required positive USD 50 per store.
The stockout guardrail also fails.
The approved rollout rule is therefore not met.
The interval includes both losses and gains, so the evidence does not establish that the true effect is negative.
A reasonable next step is to address inventory and consider a better-powered or otherwise redesigned test under a new approved protocol.

### Exercise 8 rubric

A strong checklist names a commercial owner, experiment owner, privacy or legal reviewer, inventory owner, and operational stop authority.
It defines eligibility before assignment and minimizes linkable identifiers and retention.
It logs assignment and treatment version independently of purchase.
It includes availability by location, suppresses unavailable and inappropriate recommendations, and sets a measurable threshold for customer or inventory harm.
It specifies how to restore the prior policy and preserve logs if rollout stops.
A weak checklist relies on aggregate lift as a substitute for privacy, causal evidence, or stock readiness.

## 14. Interview questions

1. **What is an attach rate?**  
   Strong: defines a directional conditional probability with basket numerator and anchor denominator.  
   Weak: calls it the percentage lift caused by a promotion.
2. **Why can the forward and reverse attach rates differ?**  
   Strong: explains the different marginal denominators with an example.  
   Weak: assumes conditional probabilities are symmetric.
3. **What does lift add to an attach rate?**  
   Strong: compares the conditional rate with baseline partner incidence and mentions rare-count instability.  
   Weak: says lift above one proves a causal relationship.
4. **Why might two products appear together without a cross-sell effect?**  
   Strong: identifies a shared mission, season, price promotion, or selection mechanism.  
   Weak: assumes co-occurrence reveals purchase intent.
5. **How would you test an in-store fixture?**  
   Strong: specifies a store-level randomized comparison, execution, independent units, and a business outcome.  
   Weak: compares receipts before and after without addressing other changes.
6. **Why retain stores that did not install the fixture on time?**  
   Strong: describes intention to treat and bias from post-assignment compliance selection.  
   Weak: removes inconvenient observations to improve the result.
7. **What if partner sales rise but total contribution falls?**  
   Strong: considers substitution, discounts, costs, and opportunity cost within the outcome scope.  
   Weak: calls the experiment a success on unit sales alone.
8. **How do you handle a stockout during a trial?**  
   Strong: records availability, protects shoppers, retains assigned units, and interprets execution limits.  
   Weak: deletes affected stores from the primary result.
9. **What does an interval spanning the decision threshold mean?**  
   Strong: distinguishes an inconclusive decision from proof of no effect and considers downside.  
   Weak: treats a positive point estimate as a guarantee.
10. **How would you deploy responsibly?**  
    Strong: names supply, data minimization, customer suitability, monitoring, and rollback ownership.  
    Weak: assumes a recommendation score is sufficient authorization.

## 15. Key terms

- **Anchor:** item group conditioning the forward rate; see [definitions](#4-definitions-and-vocabulary).
- **Analysis unit:** unit whose outcome is summarized; see [test design](#step-6--choose-a-credible-assignment-and-execution-design).
- **Attach rate:** partner incidence among anchor baskets; see [table mechanics](#step-3--construct-the-2-by-2-table).
- **Baseline incidence:** partner frequency across eligible baskets; see [definitions](#4-definitions-and-vocabulary).
- **Basket:** specified completed transaction or order; see [population](#step-2--define-the-observational-population).
- **Cannibalization:** displaced sales or contribution; see [economics](#step-7--define-economics-before-looking-at-results).
- **Confidence:** conditional probability in an association rule; see [definitions](#4-definitions-and-vocabulary).
- **Confidence interval:** uncertainty interval from a stated procedure; see [worked example](#6-worked-example).
- **Contribution:** sales net of specified variable costs; see [economics](#step-7--define-economics-before-looking-at-results).
- **Counterfactual:** outcome under the alternative policy; see [mental model](#3-the-core-mental-model).
- **Eligible population:** units satisfying prespecified inclusion rules; see [population](#step-2--define-the-observational-population).
- **Estimand:** defined treatment contrast, population, outcome, and horizon; see [intervention](#step-5--specify-the-intervention-and-estimand).
- **Exposure:** delivery or view of the assigned treatment; see [analysis](#step-8--run-monitor-and-analyze-the-test).
- **Guardrail:** prespecified boundary for an important outcome; see [decision](#step-9--decide-and-operationalize).
- **Halo:** change in outcomes outside the directly targeted item; see [definitions](#4-definitions-and-vocabulary).
- **Incrementality:** causal difference against a specified alternative; see [mental model](#3-the-core-mental-model).
- **Intention to treat:** comparison by assignment including imperfect delivery; see [analysis](#step-8--run-monitor-and-analyze-the-test).
- **Lift:** attach divided by baseline partner incidence; see [table mechanics](#step-3--construct-the-2-by-2-table).
- **Opportunity cost:** value of a displaced alternative; see [economics](#step-7--define-economics-before-looking-at-results).
- **Partner:** proposed companion product group; see [definitions](#4-definitions-and-vocabulary).
- **Randomization unit:** independently assigned customer, store, or cluster; see [test design](#step-6--choose-a-credible-assignment-and-execution-design).
- **Reverse attach:** anchor incidence among partner baskets; see [worked example](#6-worked-example).
- **Spillover:** treatment influence on another unit; see [assumptions](#11-assumptions-and-limits).
- **Stockout:** customer-facing unavailability; see [data context](#8-data-standards-and-regulatory-context).
- **Support:** joint basket frequency in the full eligible population; see [table mechanics](#step-3--construct-the-2-by-2-table).

## 16. Sources

1. Agrawal, R., Imieliński, T., and Swami, A. (1993). “Mining Association Rules between Sets of Items in Large Databases.” *Proceedings of the ACM SIGMOD International Conference on Management of Data*. [https://doi.org/10.1145/170035.170072](https://doi.org/10.1145/170035.170072). Foundational association-rule mining; not causal identification.
2. Linden, G., Smith, B., and York, J. (2003). “Amazon.com Recommendations: Item-to-Item Collaborative Filtering.” *IEEE Internet Computing*, 7(1), 76–80. [https://doi.org/10.1109/MIC.2003.1167344](https://doi.org/10.1109/MIC.2003.1167344). Public description of item-based recommendations.
3. Kohavi, R., Henne, R. M., and Sommerfield, D. (2007). “Practical Guide to Controlled Experiments on the Web.” *Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. [https://doi.org/10.1145/1281192.1281295](https://doi.org/10.1145/1281192.1281295). Practical experimental-design guidance.
4. Hernán, M. A., and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. [https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/). Potential outcomes, selection, and causal interpretation.
5. GS1. [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications). Trade-item identification and barcode standards; consult the current release.
6. European Union. Regulation (EU) 2016/679, General Data Protection Regulation. [Official EUR-Lex text](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Binding requirements and scope are determined by the regulation and applicable law.
7. California Privacy Protection Agency. [Regulations](https://cppa.ca.gov/regulations/) and [consumer privacy information](https://cppa.ca.gov/). Official California privacy resources; consult current text and counsel.

## 17. Links

- [Module 17 topic map](_index.md).
- [Module 17 survey and its lab sections](../../../course/17_assortment_category_retail_analytics.md).
- [Assortment decision stack](01_assortment-decision-stack.md).
- [Variety versus choice overload](02_variety-choice-overload.md).
- [Category roles and decision trees](03_category-roles.md).
- [Market basket analysis and association rules](04_market-basket.md).
- [Space elasticity and planograms](06_space-elasticity.md).
