# Variety versus Choice Overload

> Module 17 — Assortment, Category, and Retail Analytics  
> Topic 02 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Module 04, Module 16, and Module 17 topic 01  
> Version 2026.5

## 1. Why this topic exists

A category manager has been asked to make a crowded shelf easier to shop.
At the same time, the replenishment team wants fewer slow-moving items.
The finance team wants more contribution per metre of shelf.
A store manager warns that customers come specifically for a niche variant.
All four concerns can be true at once.

The decision is not simply whether the store carries too many stock keeping units (SKUs).
It is which needs to serve, which alternatives make those needs easy to satisfy, and how reliably those alternatives remain available.
The shopper experiences the displayed, available assortment, not the list in the item master.
A range that looks broad in a spreadsheet can be narrow on the shelf if several distinct needs are absent.
A shorter range can feel just as broad when it retains recognizable alternatives across needs and removes nearly identical duplicates.

More options can increase the probability that a shopper finds a suitable product.
They can also increase search and comparison effort.
Neither effect is guaranteed to dominate in every category, channel, or shopper segment.
Some customers arrive with a specific brand and size in mind.
Others use the display to discover a new flavor or compare ingredients.
A time-constrained repeat buyer and an exploratory gift shopper should not be treated as the same decision maker.

The cost of an error is wider than the sales of the item that is removed.
A customer may switch to another item, buy elsewhere, defer the purchase, or abandon the trip.
A surviving item may gain sales at a different margin.
Additional facings may improve availability but use labor and working capital.
A customer who cannot find a dietary requirement may lose access even if the average category conversion rate rises.
The relevant decision is therefore a portfolio and service decision, not an item-level rank ordering.

This topic gives you a practical decision sequence.
You will distinguish objective range breadth from perceived variety, estimate substitution without assuming all sales transfer, and evaluate the incremental economics of a proposed reduction.
You will also design a test that separates the effect of assortment changes from coincident changes in price, placement, and availability.
The worked example is synthetic; its numbers illustrate the method and are not a claim about a retailer.

## 2. Learning outcomes

After studying this topic, you can:

1. **Classify** breadth, depth, and perceived variety for a defined shopping mission; see Exercises 1 and 2.
2. **Explain** when extra options can improve matching and when comparison costs may inhibit a choice; see Exercise 1.
3. **Map** alternative shopper responses to a deletion and distinguish diversion, lost category demand, and lost store trips; see Exercises 2 and 3.
4. **Calculate** incremental contribution from a range change, including transferred demand and avoidable operating costs; see Exercises 3 and 4.
5. **Critique** sales and conversion evidence for stockout, denominator, selection, and treatment-bundling bias; see Exercises 5 and 6.
6. **Design** an operationally feasible randomized store experiment with guardrails and a decision rule; see Exercise 7.
7. **Recommend** an assortment decision that protects meaningful needs and communicates uncertainty; see Exercise 8.

You should be able to show your calculations, name the unit and time horizon, and explain what would change your recommendation.
A good answer distinguishes an observed fact from an estimate, a forecast, a plan, a target, and an operational commitment.

## 3. The core mental model

```text
Shopper mission, preferences, budget and time
                    |
                    v
Range offered --> perceived differentiation --> search and comparison
       |                      |                          |
       |                      v                          v
       |                suitable match              effort / deferral
       |                      \                          /
       |                       v                        v
       +------------------> choice or no choice <-------+
                                  |
                +-----------------+------------------+
                |                 |                  |
             own item        substitute         no category sale
                |                 |                  |
                +---------+-------+                  v
                          v                   other store / delay /
                units x unit contribution       abandoned basket
                          |
                          v
     category contribution + operational effects - change cost
                          |
                          v
       decision subject to coverage, service, and equity guardrails
```

The diagram is causal in intent, not proof that every arrow has been measured.
A decision maker proposes an offer; shoppers encounter an actual subset of the offer in a particular context.
Their response depends on both the fit of the options and the work required to compare them.
A shorter range can reduce comparison effort but can also remove a shopper's only acceptable match.
A larger range can improve fit but consume space needed to keep each option available.

### Three levels of variety

**Physical breadth** counts distinct need states or subcategories represented.
For example, a shelf can serve regular, decaffeinated, and caffeine-free needs.
The number of labels on a shelf is not itself the number of needs served.

**Depth** counts alternatives within a defined need state.
Six decaffeinated variants give more depth than two, but not necessarily six meaningfully different benefits.
Depth depends on where you draw the category boundary.

**Perceived variety** is the customer's impression of how many distinct, relevant choices exist.
A shopper may see four clearly labeled roast styles as more useful than eight nearly identical packs.
Perception can differ by expertise, accessibility, mission, and channel interface.
A counting rule alone cannot establish perceived variety.

### The matching-versus-effort trade-off

The matching benefit of an added alternative is highest when it serves an otherwise unmet need.
The benefit may be small when the new item is almost interchangeable with an existing one.
The comparison burden depends on presentation and prior knowledge, not just the number of items.
Good navigation may allow a large range to be manageable.
Poor labeling may make even a short range difficult to understand.

Choice overload is a possible behavioral response in which an expanded choice set impairs a relevant outcome under particular conditions.
Relevant outcomes can include choice deferral, decision difficulty, or satisfaction.
A reduction in purchase rate alone does not identify the mechanism: prices, availability, promotion, and composition may also have changed.
Evidence on the average effect is mixed and context-dependent, so treat overload as a hypothesis to test rather than a law that every additional item lowers sales.

### The operating feedback loop

An extra item can split demand across more facings and reduce units per facing.
If shelf capacity becomes insufficient, the store may experience more empty shelf time.
A stockout then changes what the shopper can buy and contaminates the observed substitution pattern.
Conversely, fewer items can release space for existing products, but that benefit is real only if the planogram, replenishment, and backroom processes actually use the space.
Measure availability as part of the intervention and describe it as a possible mechanism, not an incidental nuisance.

### A decision is a comparison of counterfactuals

The relevant question is not whether the removed item had positive sales.
It is what would happen over the same horizon if the assortment changed versus if it did not.
The difference must include the response of other products and the avoidable costs of operating the new assortment.
The estimate is uncertain until the counterfactual is supported by a credible design.

## 4. Definitions and vocabulary

The table defines terms as used in this lesson; other organizations may use different naming conventions.

| Term | Definition and unit | Common misuse |
|---|---|---|
| Category | Commercial grouping managed as a portfolio; named boundary | Assuming every retailer uses the same boundary |
| Subcategory | Meaningful division within a category; count | Equating it automatically with a supplier's hierarchy |
| Need state | Shopper's job or constraint, such as caffeine-free; qualitative label | Treating demographics alone as a need |
| Mission | Purpose of a trip or visit, such as routine replenishment | Assuming every visitor intends to purchase the category |
| SKU | A distinct operational stock keeping unit; count | Treating every SKU as a distinct consumer need |
| Breadth | Number of defined needs or subcategories covered; count | Calling the total SKU count breadth |
| Depth | Number of alternatives within a defined need; count | Comparing depths across inconsistent boundaries |
| Range | Items authorized for a store and time period; SKU count | Assuming authorized items are in stock |
| Displayed assortment | Items visible and purchasable at the decision point | Counting online hidden or unavailable products |
| Assortment availability | Fraction of relevant item-time or shopper exposures in stock; percentage | Using warehouse stock as shelf availability |
| Facing | One front-facing item position on a shelf; count | Treating facing count as total physical inventory |
| Perceived variety | Shopper's perception of distinct relevant options; survey scale | Assuming it equals SKU count |
| Choice overload | A context-dependent deterioration in a choice outcome attributable to the choice environment | Declaring it whenever a large assortment sells poorly |
| Search cost | Time or effort to locate candidate options; seconds or score | Valuing search time without defining whose time |
| Comparison cost | Effort to evaluate alternatives; time or score | Assuming comparisons are identical across shoppers |
| Conversion | Purchases divided by a defined exposure or eligible-visit denominator; percentage | Dividing category transactions by all store visitors without qualification |
| Deferral | A decision not to buy now, possibly followed by a later purchase | Automatically treating it as a permanent lost customer |
| Match | An alternative satisfying a shopper's relevant requirements | Assuming visual similarity proves acceptability |
| Duplicate | An item with little incremental need coverage relative to alternatives | Inferring duplication from slow sales alone |
| Substitution | Purchase or use of another option in response to a changed offer | Treating all remaining sales as incremental |
| Diversion ratio | Fraction of a removed item's otherwise expected demand that moves to a specified alternative; percentage | Using the fraction of observed switchers as the fraction of all displaced demand |
| Outside option | A response outside the measured choice set, such as no purchase or a competitor | Omitting it and forcing transfers to sum to 100% |
| Category leakage | Category demand that leaves the measured retailer or category because an option is unavailable | Counting every nonpurchase as a competitor purchase |
| Cannibalization | Sales transferred from another product within the relevant counterfactual | Calling every co-occurrence cannibalization |
| Halo | Causal change in another item or basket component due to the intervention | Assigning all correlated basket sales to one SKU |
| Unit contribution | Incremental selling price less defined variable costs per unit; currency/unit | Confusing gross margin with total profit |
| Incremental contribution | Change in contribution versus a specified counterfactual; currency/period | Reporting surviving-item sales without displaced sales |
| Avoidable cost | A cost that actually changes under the decision; currency/period | Counting allocated overhead as savings without evidence |
| One-time cost | Implementation expenditure incurred once; currency/change | Adding it again in every future period |
| Planogram | A specified arrangement of products and facings | Equating the approved plan with actual execution |
| Store cluster | A group of stores sharing decision-relevant attributes | Assuming a cluster is a causal treatment group |
| Randomization unit | Entity assigned to treatment, for example store | Treating many customers in one store as independent assignments |
| Counterfactual | Outcome for the same unit and period under the alternative policy | Treating last year's outcome as automatically equivalent |
| Guardrail | Predefined limit on an unacceptable side effect | Replacing the primary economic objective with a vanity metric |
| Intention to treat | Analysis by assigned intervention, regardless of implementation | Excluding noncompliant stores after assignment |
| Forecast | Conditional estimate of a future outcome | Treating a forecast as a guarantee |
| Plan | Intended actions and resources | Calling the plan an observed result |
| Target | Desired performance level | Treating a target as a prediction |
| Commitment | An agreed operational promise with an accountable owner | Assuming an analyst can commit store labor |

## 5. The mechanics, step by step

### Step 1 — Frame the decision and the counterfactual

**Input:** A specific assortment proposal, decision owner, store population, and decision date.
**Output:** A written comparison between an intervention and a credible alternative.

State the category and the channel.
Define whether the decision concerns physical shelf, online browse, or both.
Specify the assortment that would remain in place without the proposal.
Name the period over which benefits and costs will be measured.
Separate a one-time reset from recurring weekly effects.
Identify whether the decision is reversible and how quickly a removed item could be restored.

A usable decision statement is: compare removing two named cereal variants in 24 similar stores with keeping the existing range over 12 weeks, while holding regular prices and promotional commitments fixed.
That statement is better than “optimize cereal” because it identifies a treatment and a comparison.
If the proposal includes a new shelf layout, say so explicitly.
The resulting estimate will be the effect of the combined package, not a pure estimate of assortment size.

The business owner sets the objective, such as incremental contribution after avoidable operating costs.
The owner also sets constraints, such as minimum coverage for allergy-related needs.
Record who has authority to approve deletions, resets, procurement changes, and restoration.
The analyst does not turn a forecast into an operational commitment.

**Branch:** If a product is legally required or contractually protected, remove it from the discretionary deletion set until the appropriate owner resolves the constraint.
If a range change is already committed, the analysis can still inform execution and monitoring, but it is not an unbiased ex ante choice between all possible ranges.

### Step 2 — Define the shopper and the available choice set

**Input:** Transaction data, item attributes, store ranging records, shelf observations, and customer research where available.
**Output:** A time-specific choice set and a map from items to shopper needs.

Define whether the shopper is a category visitor, category buyer, loyalty customer, or all-store visitor.
Each denominator supports a different conclusion.
If category visitors are not counted, do not call category buyers divided by all store entries a shelf conversion rate.
For online analysis, define an impression, browse session, product view, and completed order separately.
Include only products that were actually visible, eligible, and available to the relevant customer at that time.

Create a store-by-week item panel with a product identifier, effective range dates, price, promotion, availability, facings, sales units, and contribution.
Add attributes that customers actually use, such as size, flavor, dietary suitability, quality tier, and format.
Use customer interviews, query logs, or structured observation to challenge the hierarchy.
Do not infer a need state solely from a supplier's reporting category.

A planogram says what should be visible; shelf checks say what was visible.
An item can be ranged but absent because of a delivery failure or an empty shelf.
A customer cannot choose an unavailable item, so a catalog-only count overstates the effective range.
Conversely, a temporary display can make an unplanned item visible.

**Branch:** If exposure data are missing, calculate transaction-based rates and label them accurately.
Do not invent browsing counts from transactions.
If availability is unreliable, repair measurement or bound its effect before interpreting low item sales as weak preference.

### Step 3 — Count breadth, depth, and differentiated coverage

**Input:** The need-state map and the actual available assortment.
**Output:** A coverage table for the existing and proposed ranges.

Count the number of need states with at least one acceptable option.
Count options within each need state.
Record the number of items that serve multiple needs, without double-counting them as separate SKUs.
Mark non-negotiable attributes, such as a medically relevant dietary restriction where the retailer claims coverage.
For each deletion, identify which customer needs lose their only option and which retain alternatives.

A coverage matrix is more informative than a single count.
It can show that a shelf with 18 items contains only one dairy-free option while five nearly identical regular options compete for the same facing space.
The matrix does not prove that the five options are interchangeable.
Price, taste, brand loyalty, and package size can make apparently similar products distinct to shoppers.
Investigate those differences before treating them as duplicates.

Measure perceived variety separately if it matters to the decision.
For example, show a representative set of shoppers two realistic shelf images in randomized order and ask about ease of finding a suitable choice and perceived differentiation.
A survey is a measure of perception, not an estimate of actual purchases.
Include accessible display formats and avoid questions that suggest a preferred answer.

**Branch:** If the proposed range fails a mandatory coverage constraint, revise the candidate set or explicitly seek an authorized exception.
A favorable average profit estimate does not silently override a hard constraint.

### Step 4 — Generate distinct, feasible alternatives

**Input:** Coverage constraints, shelf dimensions, costs, supplier obligations, and the baseline range.
**Output:** A small set of actionable assortment and presentation options.

Include the status quo.
Consider a deletion-only option, a presentation-only option, and, if feasible, a combined option.
A presentation-only option might group products by the first attribute shoppers use without removing any SKU.
A deletion-only option should specify how freed space will be used.
Another option may retain the long tail online while reducing store depth, but it is not equivalent for shoppers who need immediate in-store access.

Each option needs a shelf-feasible planogram.
Specify facings, pack dimensions, shelf height, minimum replenishment quantity, and local restrictions.
List supplier and distribution implications, including minimum order quantities and return rights where relevant.
Exclude designs that cannot be executed in the target stores.

Do not compare a fully funded redesign against an unfunded status quo and call the result the effect of fewer options.
If the question is the pure effect of a shorter range, hold layout and other features as comparable as practical.
If management will only implement a combined package, estimate and label the package effect.

### Step 5 — Build the shopper-response tree

**Input:** Candidate deletions, customer research, historical variation, or a designed experiment.
**Output:** A response tree with mutually exclusive outcomes and an explicit outside option.

For a removed item, begin with its expected demand under the unchanged assortment.
Some shoppers purchase a surviving item in the category.
Some buy a different category, postpone, or leave without the category.
Some may buy from a competitor or make a separate trip.
If the unit of analysis is the store rather than the retailer, a purchase at another branch is outside that store but not necessarily outside the retailer.
State the boundary before calling a response leakage.

A diversion ratio to survivor `j` is the incremental units of `j` caused by the deletion divided by baseline demand displaced from the removed product.
The numerator is not every sale of `j` after the change.
The denominator is counterfactual demand for the removed item, not just the shoppers who can be tracked.
For a 400-unit baseline and 240 incremental units sold by survivors, total within-category diversion is 60%.
The remaining 40% is unresolved outside the measured category unless additional evidence distinguishes deferral, competitor purchase, and no need.

If multiple survivors receive transfers, report a vector of diversion ratios.
The vector plus an outside-option share should sum to 100% under a one-unit-to-one-unit accounting assumption.
Different pack sizes can violate that assumption.
A shopper moving from two small packs to one large pack should be normalized to an appropriate consumption unit where possible.
Do not add ratios from overlapping definitions or interpret them as probabilities without a clear denominator.

Loyalty data can reveal repeat purchases but omit anonymous customers and can be selected toward more engaged shoppers.
A customer survey can ask stated intentions, but intended substitution may differ from behavior.
Stockout episodes provide natural variation, but inventory failures can coincide with promotions or demand spikes.
Each source supports a different strength of inference.

### Step 6 — Estimate substitution without mistaking correlation for causation

**Input:** A comparison design, item-level sales, prices, availability, and exposure data.
**Output:** Estimated incremental changes in survivor demand and uncertainty.

The preferred evidence for a consequential decision is a feasible randomized intervention.
Assign comparable stores or store clusters to the proposed and current assortments.
Randomize at the unit where the intervention is implemented.
Keep a record of the randomization scheme and treatment start date.
Measure outcome windows consistently in both arms.

If randomized assignment is not feasible, a carefully designed observational comparison may be useful.
For example, compare the change in treated stores with the change in comparable untreated stores.
This difference-in-differences approach requires an assumption that the groups would have followed parallel trends without the change.
Pre-intervention trends can challenge that assumption but cannot prove it.
Concurrent local competition, promotion, or store renovation can invalidate the comparison.

Account for stockouts before the change.
A removed item with frequent stockouts has observed sales below its unconstrained demand.
Using observed sales as displaced demand can understate the true impact of deletion.
At the same time, a stockout may produce apparent survivor sales that are not representative of a permanent deletion, because customers learn or change stores over time.
Do not treat temporary and permanent substitution as automatically identical.

Check heterogeneity across store formats, customer missions, and relevant need states.
Small samples can make segment estimates noisy.
Predefine priority segments and avoid claiming a precise effect for every slice of the data.
Where credible, show an interval or scenario range instead of a single diversion estimate.

### Step 7 — Convert the response into category economics

**Input:** Baseline displaced units, diversion quantities, unit contributions, space effects, and avoidable costs.
**Output:** Incremental contribution versus the baseline over a stated horizon.

For one removed item and several survivors, use:

```text
incremental contribution
  = - baseline removed-item units x removed-item unit contribution
    + sum(incremental survivor units x survivor unit contribution)
    + other causal contribution changes
    + avoidable operating savings
    - incremental operating and implementation costs
```

All terms must refer to the same stores, period, currency, and counterfactual.
Use contribution per unit, not revenue per unit, unless revenue is the explicitly chosen objective.
Define whether contribution includes supplier funding, spoilage, fulfillment, and variable labor.
Avoid double-counting an availability improvement if it is already captured in observed survivor sales.
Avoid double-counting a reduction in markdowns if it is already included in unit contribution.

An allocation of fixed headquarters overhead is not automatically avoidable.
A supplier fee is not a saving unless it changes under the decision and is permitted by the contract.
A one-time reset cost belongs in the first-period calculation or an explicitly stated investment appraisal.
For a longer horizon, use an agreed discount rate when material and avoid projecting the first few weeks indefinitely without evidence.

If the relevant objective includes a basket or traffic effect, estimate the **incremental** contribution of the additional or lost basket, not its whole observed value.
Do not add a basket effect if it is already included in the measured outcome.
Report category economics and any wider-store effect separately where attribution is uncertain.

### Step 8 — Test sensitivity and constraints

**Input:** Base-case estimate, plausible response ranges, and policy guardrails.
**Output:** A scenario table, break-even threshold, and constraint assessment.

Vary the parameters that matter most: diversion, margin of substitutes, avoidable labor, implementation cost, and space benefit.
Use scenarios grounded in evidence or explicitly label them as assumptions.
Do not mix a favorable diversion rate with an incompatible unfavorable baseline without explaining the scenario.
A break-even calculation identifies the value at which the recommendation changes; it is not evidence that the value will occur.

Check minimum assortment coverage for each protected need state.
Check store-level availability, customer complaints, and service levels.
Check whether the candidate hurts a small but important customer group even when the average effect is positive.
A policy can fail a guardrail and remain economically attractive; that does not make the guardrail disappear.
Escalate exceptions to the accountable decision maker.

### Step 9 — Design measurement and implementation together

**Input:** Feasible intervention, comparison group, operational calendar, and decision thresholds.
**Output:** A test protocol with owners, measurement, and a rollback plan.

Choose the primary outcome before looking at the outcome data.
Incremental contribution per store-week is often more decision-relevant than conversion alone.
Define secondary measures such as units, category conversion where exposure is measured, availability, shopper effort, and complaints.
Set guardrails and the minimum economically meaningful effect in advance.

Randomize stores within relevant strata, such as format and baseline category volume, when the sample supports it.
Use the store as the statistical unit when stores are randomized.
A large number of individual transactions does not create a large number of independent treatment assignments.
Account for clustering, the actual allocation, and pre-specified covariates in uncertainty estimates.
Calculate sample size using realistic store-level variation and available duration; do not promise a fixed power without the necessary inputs.

Record the exact set of delisted items, shelf layout, prices, promotional calendar, and replenishment rules.
Audit implementation, but preserve the assigned-group analysis as the primary intention-to-treat comparison.
A supplementary implementation analysis can diagnose noncompliance, but it may be biased if compliant stores differ systematically.
Set a stop or rollback procedure for safety, material service failures, and agreed commercial limits.

### Step 10 — Decide, communicate, and learn

**Input:** Economic estimate, uncertainty, constraint outcomes, and implementation evidence.
**Output:** A documented decision with an owner and review date.

Recommend scale, revise, hold, or stop.
State the business objective and the observed effect separately from the target.
Give the uncertainty interval and its economic interpretation where available.
Name any segment or store format where the result is too uncertain for rollout.
Explain which parts of the result are measured and which are assumptions.

A forecast extends the measured effect into a future context.
A plan describes how to implement the decision.
A target is the desired performance, not the observed effect.
A commitment is a promise made by an authorized operator after checking capacity and contracts.
Record all four separately in the decision memo.

Set a post-rollout review date long enough to capture learning and replenishment cycles.
Watch for supplier exit, changes in competitor offers, and customer migration after the initial test.
Keep an owner for restoring coverage if a guardrail fails.
The purpose of the process is not to establish that fewer items are always better; it is to choose a defensible range for a particular business context.

### Supplemental procedure A — Understand the shopper's decision sequence

A category hierarchy is an operating taxonomy; a decision tree describes how shoppers narrow choices.
The two can disagree without either being erroneous.
A retailer may organize tea by supplier while shoppers first distinguish caffeinated from caffeine-free.
One shopper may select brand before flavor; another may treat price as the first hard constraint.
The analyst should therefore document the population and mission before proposing a single decision tree.

Start with open-ended observation.
Ask shoppers what they came to find without first presenting the retailer's attribute list.
Record which constraints they identify unprompted.
Then observe whether the shelf enables those constraints to be checked quickly.
If participants agree, note the sequence of labels and packs they inspect.
Do not equate looking at an item with willingness to buy it.

Next, ask what they would do if the preferred item were unavailable.
Record the first acceptable replacement and the conditions under which it is acceptable.
Ask separately about price, ingredient suitability, and pack size.
Do not phrase the question as if buying a surviving item were mandatory.
Include “leave without the category” and “buy elsewhere” among available responses.
Treat these stated responses as directional evidence, not as measured diversion.

For an online category, inspect the first page and the result of common filters.
An item buried beyond the portion a shopper sees may be nominally ranged but effectively absent.
Check whether out-of-stock products remain visible and whether a substitute is explicitly offered.
Capture the order and state of search results at the time of the interaction.
If sponsored placement affects rank, record it as part of the choice environment.
Do not assume that a display order is neutral because the catalog contains all items.

Synthesize findings into a small number of decision-relevant needs.
Avoid creating a separate need for every possible attribute combination when sample sizes cannot support it.
At the same time, do not collapse distinct dietary or accessibility constraints into a generic preference.
Ask the category and customer research teams to review the classification.
Maintain a dated version so that future outcomes can be interpreted against the rule used at decision time.

### Supplemental procedure B — Audit substitution evidence

An estimate of substitution can come from more than one source.
The sources differ in what they measure and which selection problems they introduce.
Use a structured evidence table rather than combining estimates without qualification.

| Evidence source | What it can show | Main concern | Appropriate response |
|---|---|---|---|
| Randomized deletion | Effect of the assigned package in the test population | Noncompliance and limited duration | Analyze assigned groups and audit execution |
| Unplanned stockout | Behavior when an item is temporarily unavailable | Stockouts may coincide with demand shocks | Adjust design and compare comparable periods |
| Loyalty transitions | Recorded repeat purchases by linked customers | Anonymous and unlinked shoppers are missing | Compare linked and total sales coverage |
| Shopper interview | Stated requirements and perceived alternatives | Stated choices may differ from actual purchases | Use to formulate and interpret hypotheses |
| Shelf observation | Actual availability at sampled times | Sampling can miss empty periods | Define a schedule and uncertainty around coverage |
| Online clickstream | Search, rank, and observed funnel behavior | Visibility and consent differ by user | Define exposure and apply privacy controls |
| Supplier view | Commercial and supply constraints | Incentives may favor the supplier's items | Reconcile claims with retailer-level evidence |

For every source, state its population, period, unit, and missingness.
Ask whether the item was unavailable or merely not purchased.
Ask whether the alternative had the same price and availability during the observation.
Ask whether the comparison captures new shoppers as well as repeat shoppers.
Ask whether customers could easily substitute across channels or stores.
An estimate can be informative while still being unsuitable as a direct forecast.

Temporary stockout evidence requires particular care.
A customer may defer a routine purchase for a day rather than switch products.
A permanent deletion may lead that customer to change brands after repeated visits.
The opposite can occur if a temporary substitute is tolerated once but rejected over time.
The duration of unavailability is therefore part of the treatment.
Separate immediate substitution from repeat behavior whenever data permit.

Consider a 100-pack observed stockout episode.
Survivor sales rose by 30 packs versus a credible comparison.
The estimated short-run diversion is 30%, not 30 packs divided by the survivor's total sales.
The 70-pack remainder is not necessarily lost permanently.
Some customers may return when stock is restored.
The comparison must also account for any demand shock that caused the stockout.
Without a credible comparison, the 30-pack increase is descriptive, not a causal estimate.

### Supplemental procedure C — Plan a store-level experiment

Begin with the decision that the test will support.
For example, management may need to decide whether to adopt a specific range-and-layout package.
The effect of that package is not the same estimand as the effect of removing one SKU while keeping every other shelf feature fixed.
Write the treatment and comparison in terms that a store operator can execute.
Specify the dates, item lists, facings, signage, and replenishment instructions for each arm.

Before randomization, identify the eligible stores.
Exclude stores with scheduled closures or renovations only using criteria fixed before assignment.
List the stores and retain a dated record of their eligibility.
Where appropriate, form strata from pre-treatment format and sales information.
Randomly allocate within strata using a documented seed and allocation rule.
If nearby stores share shoppers, evaluate whether a geographic cluster is a more appropriate assignment unit.
Cluster assignment reduces cross-arm contamination but also reduces the number of independent units.

The primary estimand might be average incremental category contribution per assigned store-week.
Define contribution before the test, including whether product costs, variable labor, and supplier funding are included.
Decide how one-time reset costs will be represented in the commercial decision.
Use the same calendar weeks in both arms.
Retain all eligible assigned stores in the primary analysis.
Report noncompliance, late resets, and item availability separately.

Suppose a store is assigned the new range but completes the reset three weeks late.
Under intention to treat, the store remains in the new-range group.
The observed effect then reflects the assigned rollout including this implementation reality.
A secondary analysis can describe the association with actual execution.
That secondary association is not automatically causal because late stores may differ from timely stores.
Improving compliance is an operational action, not a reason to rewrite assignment history.

Record the primary outcome before inspecting treatment results.
Define whether early stopping is allowed and who authorizes it.
Safety or severe service failures may require an immediate stop.
Repeatedly examining an ordinary commercial outcome and stopping on the first favorable result can distort inference unless the monitoring design accounts for it.
Use a statistician or qualified experiment owner to select the inference method.
Report the point estimate together with an interval and the number of randomized units.

Plan for treatment fidelity without turning fidelity into the outcome.
A store may have the correct product list but incorrect shelf labels.
An audit may show the right layout while stock remains in the backroom.
Specify which execution failures are recorded and when they are corrected.
Use consistent audits in both arms so that measurement effort is not confounded with assignment.
Record material interventions made by the field team during the test.

Guardrails should be actionable.
“Customer experience must not deteriorate” is too vague to govern a rollback.
Instead specify the coverage check, measurement period, threshold, and responsible owner.
For instance, the owner could require at least one in-stock affordable option for a defined need at each sampled store visit.
That operational threshold is a business policy, not a universal statistical rule.
Where possible, combine a leading service measure with outcome measures such as complaints and repeat purchasing.

At the end of the test, compare the measured effect with the minimum economically meaningful effect.
A statistically uncertain result does not prove the effects are zero.
Similarly, a statistically detectable gain may be too small to fund a chain-wide reset.
Consider uncertainty, implementation expense, and guardrails together.
If the interval includes both a material loss and a material gain, the decision may be to collect more evidence or retain a reversible option.

### Supplemental procedure D — Move from average results to rollout decisions

The average result is an estimate for the test population under the tested conditions.
A rollout decision often concerns stores outside that population and a later period.
List the ways the rollout environment differs before extrapolating.
Differences may include store size, customer missions, supplier lead time, assortment availability, and local competition.
New stores may have a different mix of customer preferences or a different price tier.

Create an applicability table for each intended store group.
Record whether the group was represented in the test, whether its operating constraints match, and whether key outcome data exist.
Use pre-specified segment estimates where samples permit.
Avoid declaring a reliable segment winner from many exploratory comparisons.
Small segments can still be important for coverage even when their contribution estimate is imprecise.
Use a service audit and qualitative research alongside the aggregate outcome.

Translate the recommendation into an implementation sequence.
Confirm the end date for ordering the removed item and the treatment of remaining stock.
Confirm that surviving items have sufficient distribution capacity and shelf space.
Communicate price-label and online-catalog changes to the relevant owners.
Train store teams to handle customer questions and record recurring unmet needs.
Schedule a post-reset check for physical execution and an outcome review after sufficient time.

Make restoration practical before removing an item.
Identify the supplier, lead time, minimum order quantity, and available shelf location.
Define which guardrail failure triggers investigation and which triggers immediate restoration.
Do not promise same-day restoration if the distribution network cannot supply it.
The rollback plan is a supply commitment only after the relevant owner accepts it.

Finally, maintain a record of what was learned.
Save the assumptions and the version of the item-to-need mapping.
Record what actually changed in range, layout, pricing, and availability.
Separate test evidence from judgments used to extrapolate it.
At the next review, compare realized contribution and service with the original forecast.
Update the model when the operating context changes rather than treating a historic diversion rate as a permanent product attribute.

## 6. Worked example

### Decision and boundaries

A fictional neighborhood grocery chain is reviewing shelf-stable breakfast cereal.
The category manager proposes deleting one slow-selling 350 g low-sugar variant in 20 stores.
The proposed period is 12 weeks.
The baseline is the existing range in the same 20 stores for the same 12-week future period.
All amounts are in US dollars (USD).
The unit of demand in this example is one retail pack.
The decision excludes externalities beyond the category unless separately noted.

The existing range has 24 items across four need states: regular, low-sugar, high-fiber, and gluten-free.
Some items serve more than one need, so need-state counts do not sum to 24 items.
The proposed deletion reduces the low-sugar options from five to four.
It leaves at least one option in every need state under the agreed coverage map.
A customer may nevertheless reject all four survivors because of taste, price, package size, or brand.
Coverage is necessary for this proposal but is not proof of customer acceptance.

### Data and assumptions

The table below is an illustrative 12-week forecast, not an observed test result.
Values are totals for the 20 stores, not per-store weekly rates.

| Input | Base assumption | Interpretation |
|---|---:|---|
| Baseline demand for removed item | 400 packs/12 weeks | Counterfactual sales if kept |
| Removed-item unit contribution | USD 1.50/pack | Net of defined variable item costs |
| Incremental survivor demand | 240 packs/12 weeks | Caused by deletion, not total survivor sales |
| Average survivor unit contribution | USD 1.20/pack | Weighted over incremental survivor packs |
| Avoidable operating savings | USD 110/12 weeks | Verified reduction in variable handling |
| Capacity and availability benefit | USD 130/12 weeks | Incremental contribution not already in transfer estimate |
| One-time reset cost | USD 40/change | Incremental labor and materials |

The space benefit assumes the surviving facings are enlarged and replenishment uses the released space.
It is separately estimated here and must not already appear in the 240 incremental survivor packs.
If a real experiment measures total category contribution directly, adding this modeled benefit again would double-count it.
The assumed 110 USD saving is an actual avoidable cost, not an allocated overhead charge.
The reset cost is incurred once in the evaluated 12-week period.

### Response tree

Of the 400 packs of baseline demand, 240 packs are forecast to transfer to the survivors.
That is 240 / 400 = 0.60, or 60% diversion within the measured category.
The remaining 160 packs, or 40%, are not bought as cereal in the measured stores during the same period.
The analysis does not know whether those customers defer, buy elsewhere, or abandon the category entirely.
It does not assign a competitor share without evidence.
It also assumes one pack of survivor demand replaces one pack of displaced demand.
A different pack size would require a more careful common-unit analysis.

### Contribution calculation

The removed item would have generated:

```text
400 packs x USD 1.50/pack = USD 600 per 12 weeks
```

Its deletion removes that contribution from the baseline.
Surviving items gain:

```text
240 packs x USD 1.20/pack = USD 288 per 12 weeks
```

The direct item-and-transfer contribution change is:

```text
-USD 600 + USD 288 = -USD 312 per 12 weeks
```

Add the separately estimated capacity benefit and avoidable operating savings, then subtract reset cost:

```text
-USD 312 + USD 130 + USD 110 - USD 40 = -USD 112
```

The base-case incremental contribution is therefore **negative USD 112 over 12 weeks across 20 stores**.
That is an average of negative USD 5.60 per store over the period, not a claim that every store loses exactly that amount.
It is also approximately negative USD 0.47 per store-week when divided by 240 store-weeks.
The period contains a one-time reset cost; do not annualize this result without separating recurring and one-time components.

### Break-even diversion

Let `q` be incremental survivor packs over the 12-week period.
Set total incremental contribution to zero:

```text
-600 + 1.20q + 130 + 110 - 40 = 0
1.20q = 400
q = 333.33 packs
```

The break-even ratio is 333.33 / 400 = 83.33% of the displaced packs.
For whole packs, at least 334 incremental packs would be required under the other unchanged assumptions.
That is substantially above the assumed 60% diversion.
This threshold is not an estimate of actual shopper response.
If the survivor margin, capacity effect, or avoidable cost changes, the threshold changes too.

### Scenario analysis

Hold all inputs other than diversion constant.
The following scenarios are illustrative, not confidence limits.

| Diversion | Incremental survivor packs | Transfer contribution | Total incremental contribution |
|---:|---:|---:|---:|
| 40% | 160 packs | USD 192 | -USD 208/12 weeks |
| 60% | 240 packs | USD 288 | -USD 112/12 weeks |
| 80% | 320 packs | USD 384 | -USD 16/12 weeks |
| 90% | 360 packs | USD 432 | USD 32/12 weeks |

Even the 80% case falls slightly short in the first 12 weeks.
The 90% case is positive but small relative to uncertainty in space and handling effects.
The first-period threshold includes the reset cost; a repeat period with no new reset has a different threshold.
Under these assumptions, recurring break-even is 300 packs, or 75% diversion, because the one-time USD 40 is no longer charged.
This is a separate period calculation, not a promise that demand and savings remain constant.

### Shopper and service implications

The deletion retains four low-sugar items on paper.
A store-level shelf audit must confirm that those items are actually in stock and accessible.
If the removed item is the only affordable low-sugar option in a particular store, the chain-wide coverage count conceals a local loss.
The manager should inspect price tier, pack size, ingredient suitability, and store-specific availability.
A retained option at a substantially higher price may not be an acceptable substitute for a budget-constrained shopper.

Customer complaints may help identify service failures but do not represent all affected shoppers.
A quiet deletion can still lead to competitor purchases.
Likewise, one complaint does not establish the total causal effect.
Combine quantitative outcomes with structured customer feedback and store observations.

### Recommendation

Do not roll out the deletion based on the base-case forecast alone.
The central scenario is negative and the break-even diversion is high.
First check whether the space benefit or handling saving can be measured independently without double counting.
Then consider a randomized store test if the decision's expected value justifies the reset cost.
A presentation-only alternative may reduce search effort while retaining the low-sugar option.

The **forecast** is a 112 USD loss over 12 weeks under the stated assumptions.
The **plan** is to validate costs and test an alternative, subject to approval.
A **target** might be nonnegative incremental contribution with no loss of agreed need coverage; it is not the forecast.
No operational **commitment** exists until the store and supply owners approve labor, stock, and timing.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
Copy it into a file and run it with `python filename.py`.
It reproduces the synthetic example and its sensitivity table.
It assumes a common pack unit, constant unit contributions, a separately estimated capacity benefit, and a one-time reset cost.
It does not estimate diversion or establish causality.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Assumptions:
    baseline_packs: float = 400.0
    removed_margin: float = 1.50
    survivor_margin: float = 1.20
    operating_savings: float = 110.0
    capacity_benefit: float = 130.0
    reset_cost: float = 40.0


def incremental_contribution(a: Assumptions, diversion: float) -> float:
    """Return USD for the stated 12-week, 20-store horizon."""
    removed = a.baseline_packs * a.removed_margin
    transfer = a.baseline_packs * diversion * a.survivor_margin
    return -removed + transfer + a.operating_savings + a.capacity_benefit - a.reset_cost


def break_even_diversion(a: Assumptions) -> float:
    """Return the required fraction of displaced packs, not a forecast."""
    required_transfer = (
        a.baseline_packs * a.removed_margin
        - a.operating_savings
        - a.capacity_benefit
        + a.reset_cost
    )
    return required_transfer / (a.baseline_packs * a.survivor_margin)


a = Assumptions()
print(f"Base case: ${incremental_contribution(a, 0.60):.2f}")
print(f"Break-even diversion: {break_even_diversion(a):.2%}")
print("Diversion  Survivor packs  Incremental USD")
for diversion in (0.40, 0.60, 0.80, 0.90):
    packs = a.baseline_packs * diversion
    value = incremental_contribution(a, diversion)
    print(f"{diversion:>8.0%}  {packs:>14.0f}  {value:>15.2f}")
```

Expected output:

```text
Base case: $-112.00
Break-even diversion: 83.33%
Diversion  Survivor packs  Incremental USD
     40%             160          -208.00
     60%             240          -112.00
     80%             320           -16.00
     90%             360            32.00
```

The calculation uses the weighted average survivor contribution as a simplifying assumption.
If survivor products have different margins, replace the single transfer term with the sum of incremental packs multiplied by each survivor's contribution.
If pack sizes differ, decide whether the commercial unit is packs, kilograms, servings, or another comparable unit before calculating diversion.
The break-even formula is meaningful only when baseline packs and survivor contribution are positive and when the other terms do not themselves depend on the diversion rate.
A threshold above 100% indicates that one-for-one within-category diversion cannot reach break-even under the fixed assumptions.
A negative threshold indicates that other benefits exceed the removed item's contribution before any transfer; inspect those assumptions carefully.
Neither situation should be clipped silently because the extreme result is useful decision information.

To use the code with actual data, first establish the appropriate comparison period and estimate causal incremental packs.
Do not substitute observed post-change survivor sales for incremental packs.
Do not use a 12-week saving together with a 52-week item contribution.
Reconcile the cost ledger with finance and the space assumptions with store operations.
The example does not include uncertainty intervals; calculate those from the actual design rather than pretending the input assumptions are observed data.

## 8. Data, standards, and regulatory context

### Minimum data contract

The primary analytical grain is store, item, and week, supplemented by a transaction or exposure grain when available.
Every table needs a consistent store identifier and an effective date.
A transaction table should contain transaction ID, timestamp, store, item, units, net sales, and eligible discounts.
A product table should contain stable identifiers and dated attributes for size, ingredients, brand, and need coverage.
A range table should contain authorization start and end dates by store and channel.
A price table should distinguish regular price, promotion, loyalty price, and actual net selling price.

Availability requires its own measurement plan.
An inventory balance at midnight is not a direct measure of shelf availability at 5 p.m.
A shelf audit is a sample of moments, not continuous availability.
Online availability may vary by customer location, fulfillment mode, and substitute rules.
Define the numerator, denominator, and measurement frequency before comparing stores.

A planogram table should record intended facings and position with effective dates.
An execution audit should record actual shelf state and the time of inspection.
A cost table should distinguish variable product cost, vendor funding, handling, markdown, and one-time reset labor.
Customer complaints and interviews can supply context but require consent and appropriate access controls.

### Item identifiers and hierarchy

GS1 identifies the Global Trade Item Number (GTIN) as a way to identify trade items.
A GTIN helps join product records across trading partners, but a GTIN alone does not define the shopper's need state or establish interchangeability.
A retailer's SKU is an internal operational identifier and may not map one-to-one to a GTIN throughout time.
Changes in packaging or formulation can require identifier and attribute maintenance under the applicable GS1 rules.
Use effective-dated mappings and document any aggregation across pack versions.

The GS1 Global Product Classification standard supplies a common product classification framework.
It can improve interoperability, but it is not a causal model of customer choice.
A retailer may need a different, documented decision tree for its category and market.
Do not assume an industry hierarchy reflects local dietary requirements or the shopper's first comparison attribute.

### Privacy and lawful use

Loyalty identifiers, linked transactions, location, and browsing events can identify or make inferences about individuals.
Collect only the data needed for the decision and restrict access to the authorized analytical purpose.
Apply retention rules and avoid exposing individual purchase histories in a category review.
Small groups can be reidentified even when names are removed.
Aggregate reports and suppression thresholds should be chosen with the organization's privacy team.

In the European Union, the General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, applies to relevant processing of personal data under its territorial and material scope.
Article 5 sets principles including purpose limitation, data minimization, and storage limitation; Article 6 requires a lawful basis for processing.
Whether a particular analysis complies depends on the facts, local law, and the controller's implementation.
This educational discussion is not legal advice and a retail analytics label is not itself a lawful basis.
Consult qualified counsel before linking identifiable behavioral data across purposes or markets.

### Product claims and accessibility

Ingredient and nutrition claims must be taken from authoritative, current product information appropriate to the jurisdiction.
A modeled need-state flag is not a substitute for a verified label.
For US packaged foods, the Food and Drug Administration's food labeling resources describe applicable federal requirements, including nutrition labeling and allergen-related requirements where relevant.
Requirements vary by product and jurisdiction; verify current rules before presenting a dietary claim.
Do not infer medical suitability solely from a category name or customer review.

Online assortments also depend on interface accessibility.
The World Wide Web Consortium's Web Content Accessibility Guidelines (WCAG) provide technical guidance for accessible digital content; the guidelines are not automatically a law in every jurisdiction.
Relevant legal obligations depend on the jurisdiction and context.
A filter that hides a needed item from keyboard or screen-reader users changes the effective choice set for those shoppers.
Test actual navigation and availability, not only visual screenshots.

### Data quality checks

Check for duplicate item-store-week records and inconsistent units.
Reconcile unit sales and contribution to finance totals for the same period.
Flag returns and negative units rather than silently dropping them.
Use actual effective dates for item launch, delisting, price, and promotion.
Check that the treatment did not start in the comparison arm before the intended date.
Investigate missing shelf audits and differences in measurement frequency by arm.
Document anonymous versus loyalty-covered sales and report coverage.

A missing availability value is not necessarily an in-stock value.
A zero sale is not necessarily zero demand.
A zero stock balance may be a data-latency artifact rather than a confirmed shelf stockout.
Preserve the distinction between measured zero, missing, and imputed values.
The credibility of the substitution estimate depends on the quality of these distinctions.

## 9. Industry practice and case evidence

### The well-known jam demonstration

Iyengar and Lepper's 2000 paper reported studies of extensive versus limited choice, including a jam tasting display.
The research is often summarized as showing that a larger display attracted more interest while a smaller display produced a higher purchase rate among tasters.
The result illustrates a possible tension between attracting attention and completing a decision.
It does not establish a universal optimal number of grocery items.
The context, product, sample, and way the choice set was presented matter.
Nor does it establish that removing items from a permanent store range will necessarily increase category profit.

For a retail decision, distinguish the number of people attracted to a display from the fraction who purchase after interacting with it.
The product of traffic and conditional conversion determines the number of purchasers in that simplified setting.
A higher conditional conversion rate can coexist with fewer total purchases if exposure falls sufficiently.
A permanent assortment change can also affect return visits and availability, which a short display study may not measure.

### Mixed and conditional evidence

Scheibehenne, Greifeneder, and Todd's 2010 meta-analysis found substantial variation across studies and did not support a simple general claim that more choice is always demotivating.
Chernev, Böckenholt, and Goodman later synthesized moderators of choice overload, including characteristics of the choice set, decision task, and decision maker.
Together these sources support treating the effect as conditional rather than inevitable.
They do not supply an assortment size that can be transported directly to a new retailer.

A good application therefore specifies the outcome and mechanism.
Does the proposed range change reduce time to find an acceptable item?
Does it improve the probability of a purchase, or merely speed a decision not to buy?
Does it help novices but hurt customers with a specific preference?
A study of one of these outcomes cannot automatically answer all the others.

### Perceived variety and organization

Research by Kahn and Wansink examines how assortment organization affects perceived variety.
The implication for practice is that arrangement and labeling can change the way a fixed range is experienced.
This is a reason to compare a navigation intervention with a deletion, not proof that a particular retailer's shelf redesign will increase sales.
A retailer can group options by shopper-relevant attributes, use legible shelf labels, and make differences comparable.
The first attribute in the customer's decision tree should be investigated rather than imposed by the supplier hierarchy.

A useful store test might compare the same set of products with different grouping.
If the range remains constant, the comparison can isolate the presentation package more cleanly.
If the number of facings also changes, record that change as part of the treatment.
When the planned intervention changes both grouping and range, the test estimates their combined operational effect.

### Evidence from actual behavior

Broniarczyk, Hoyer, and McAlister studied consumers' perceptions of assortment and the role of favorite brands and shelf space.
Their work is relevant because perceived assortment need not track item count mechanically.
The lesson is not that favorite brands should always be protected irrespective of economics.
It is that a shopper can perceive the loss of a preferred option more strongly than a raw count suggests.
A carefully chosen reduction may preserve perceived variety, but that proposition must be tested in the proposed context.

For an operational deletion, triangulate across evidence types.
Transactions tell you what was bought, conditional on what was available.
Shelf observations show what was offered at observed moments.
Interviews reveal reasons and unmet needs, but stated intentions may not predict actual behavior.
A randomized range change can estimate an overall effect, provided implementation and measurement are credible.
These methods complement rather than replace each other.

### What not to infer

No published jam experiment proves that the synthetic cereal proposal in Section 6 will succeed or fail.
No cited study gives the 60% diversion assumption used in that example.
The 60% figure is a teaching assumption and should be replaced by local evidence.
A local test result is also not automatically transportable to all store formats or future market conditions.
Record the setting, treatment, comparison, period, and population whenever using an external case as evidence.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Count-first deletion | Target specifies a percentage of SKUs to remove before mapping needs | Raw count substitutes for coverage and value | A niche need loses its only viable product | Map needs and hard constraints before ranking |
| Overload asserted, not measured | Declining sales are attributed to too many options without a comparison | Alternative explanations are ignored | Useful range is cut for the wrong reason | Test a defined behavioral outcome and counterfactual |
| Confusing visits and category exposure | Conversion denominator is all store entries | Many visitors had no category mission | A false change in shopping efficiency | Name denominator and measure category exposure where feasible |
| Stockout treated as low demand | Candidate item has frequent empty-shelf periods | Observed sales are censored by availability | A valued item looks expendable | Measure item-time availability and investigate lost demand |
| All sales assumed to transfer | Diversion shares are forced to sum to 100% within the category | Outside options are excluded | Profit impact is overstated | Include no-purchase, deferral, and leakage explicitly |
| Transfer confused with survivor sales | All post-change units of a survivor are counted as incremental | Baseline survivor demand is ignored | The same sales are counted twice | Estimate change versus a counterfactual |
| Mixed pack units | One large pack replaces two small packs in a one-for-one ratio | Pack count is not comparable consumption | Diversion and margins are distorted | Reconcile packs, weight, servings, and contribution |
| Savings are allocated overhead | Finance cannot identify a cost that actually changes | Accounting allocation is mistaken for cash or avoidable cost | A negative decision appears positive | Verify the resource and period that change |
| Space benefit double counted | A modeled availability uplift is added to measured total category lift | The same mechanism appears in two terms | Incremental value is inflated | Maintain a mutually exclusive contribution bridge |
| Treatment bundle mislabeled | Range, price, signage, and facings all change at once | Multiple interventions affect the outcome | A package effect is falsely attributed to SKU count | Hold elements fixed or label and test the bundle |
| Local harm hidden by average | Chain result is positive while a format loses its affordable option | Aggregation conceals heterogeneous coverage | Customers in some stores lose access | Review predefined segment and store guardrails |
| Loyalty selection bias | Only identified members are analyzed | Members differ from anonymous shoppers | Diversion estimate is unrepresentative | Compare coverage and triangulate with total sales |
| Post-assignment exclusions | Noncompliant test stores are removed from the primary analysis | Compliance may be related to outcomes | Randomization protection is weakened | Analyze assigned groups and report execution separately |
| Supplier and price confounding | A supplier promotion starts with the reset | Treatment and market conditions move together | The estimated assortment effect is biased | Coordinate calendars and retain a comparable control |
| Short horizon | Initial weeks look favorable but repeat purchase is unmeasured | Shoppers need time to learn or leave | Long-run impact is missed | Measure repeat periods and predefine a later review |
| No rollback owner | A guardrail fails and nobody can restore the item | Decision rights and stock are absent | Service damage persists | Name owner, inventory source, and trigger in advance |

The earliest sign of a weak proposal is often an undefined comparison.
Another is a claimed saving that cannot be traced to an actual change in a resource or invoice.
Neither can be repaired by a more sophisticated model alone.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Baseline demand is credible | It defines sales displaced by deletion | Promotions, stockouts, or trend shifts alter demand | Compare matched periods and account for availability |
| Survivor packs are incremental | Transfer is the main offset to lost contribution | Sales would have occurred anyway | Estimate against a randomized or credible comparison |
| Pack units are comparable | A single diversion ratio needs a common denominator | Pack size or usage changes | Convert to relevant consumption units and retain price differences |
| Contribution per pack is stable | Simplifies the scenario calculation | Discounts, supplier funding, and costs vary | Calculate item-period margins and run sensitivity cases |
| Savings are avoidable | The decision must change real resources | Labor is fixed or merely reallocated without value | Confirm with operations and finance after implementation |
| Capacity benefit is separate | Prevents overlap with transferred demand | Measured survivor sales already include the uplift | Use a contribution bridge with nonoverlapping terms |
| Coverage map reflects shopper needs | A retained option must be acceptable | Taste, accessibility, affordability, or ingredients differ | Research actual missions and audit the shelf |
| Store randomization is respected | Assigned groups should be comparable on average | Contamination or selective exclusions occur | Monitor assignment, execution, and intention-to-treat outcomes |
| No important spillover across arms | Control stores represent the untreated outcome | Shoppers or stock move between nearby stores | Consider geographic clusters and measure cross-store flows |
| Test period represents the decision horizon | Short-run estimates inform rollout | Learning, seasonality, or competitive response changes behavior | Extend follow-up and test transportability |
| Guardrails are observable | A service failure needs a detectable signal | Small affected groups or inaccessible feedback are missed | Combine store checks, accessible feedback, and segment analysis |
| Cost and outcome windows align | All terms must describe the same comparison | Annual savings are combined with a short-term reset cost | Standardize units and separate one-time from recurring terms |

The example uses assumed point values, not an estimated probability distribution.
Its sensitivity table illustrates direction and break-even but is not a confidence interval.
The method is less reliable when product availability cannot be reconstructed or when the intervention is too small to measure against normal store variation.
Where evidence is weak, a reversible pilot or a decision to retain coverage can be more defensible than a precise but unsupported point estimate.

## 12. Exercises

### Exercise 1 — Distinguish forms of variety

A store sells 18 tea SKUs: 10 black, 4 green, 2 herbal, and 2 decaffeinated.
A proposal removes two black SKUs and retains all other items.
Assume the four listed groups are the agreed need-state classification and do not overlap for this exercise.

- **Deliverable:** State baseline and proposed breadth and the black-tea depth in each range; explain one reason perceived variety may change differently from the SKU count.
- **Acceptance criteria:** Use the defined groups, distinguish the two counts, and name a plausible presentation or preference mechanism without claiming it was observed.
- **Time:** 10 minutes.

### Exercise 2 — Build a response tree

An item would sell 500 packs over a period if retained.
You estimate 200 incremental packs for survivor A and 125 for survivor B after deletion.
All pack units are comparable and there are no other within-category survivors receiving incremental demand.

- **Deliverable:** Calculate the two diversion ratios and the outside-category share; list two possible behaviors within the outside share.
- **Acceptance criteria:** Shares use the 500-pack baseline denominator, sum to 100%, and do not assert a competitor purchase without evidence.
- **Time:** 12 minutes.

### Exercise 3 — Calculate a first-period decision

Use Exercise 2's 500-pack baseline and survivor transfers.
The removed item earns USD 1.40/pack; survivor A earns USD 1.10/pack; survivor B earns USD 1.60/pack.
Avoidable costs are USD 90 for the period and the one-time reset costs USD 35.
Assume no other effects.

- **Deliverable:** Produce a contribution bridge and a scale/hold recommendation based on the stated arithmetic.
- **Acceptance criteria:** Include the removed baseline contribution, both transfers, savings, reset cost, currency, and period; distinguish the calculation from certainty about the estimates.
- **Time:** 15 minutes.

### Exercise 4 — Find a break-even threshold

A proposed deletion displaces 300 packs at USD 2.00/pack.
Each transferred pack earns USD 1.50; separately verified savings and benefits total USD 120 for the period.
The reset costs USD 30 in that period.

- **Deliverable:** Calculate the break-even transferred packs and diversion percentage; state whether 70% diversion is sufficient.
- **Acceptance criteria:** Include the reset, show the equation, and distinguish the algebraic threshold from an estimated diversion rate.
- **Time:** 12 minutes.

### Exercise 5 — Critique a dashboard

A dashboard shows category transactions divided by all store entries rose from 12% to 13% after a deletion.
The store also ran a category promotion and did not record category visitors or shelf availability.
The analyst concludes that choice overload fell and every deleted sale transferred.

- **Deliverable:** Identify at least four distinct problems and propose one data or design remedy for each.
- **Acceptance criteria:** Address the denominator, attribution, availability, and substitution; do not infer a causal mechanism from the dashboard alone.
- **Time:** 20 minutes.

### Exercise 6 — Audit the cost bridge

A proposal adds observed post-change total category contribution, a modeled contribution from the same extra survivor sales, and an allocation of unchanged headquarters rent as savings.
The proposal also includes a one-time reset cost every week for 12 weeks even though the reset occurs once.

- **Deliverable:** Rewrite the bridge in words and identify each double count or inappropriate term.
- **Acceptance criteria:** Use a counterfactual difference, include only avoidable costs, and treat the reset according to its actual timing.
- **Time:** 15 minutes.

### Exercise 7 — Design a field experiment

A retailer has 40 stores and can implement a 12-week range-and-layout package in half of them.
Stores differ by format and baseline category contribution.
Some stores are close enough that shoppers may visit either location.

- **Deliverable:** Write a one-page protocol covering assignment, primary outcome, guardrails, data, analysis unit, spillover, and rollback.
- **Acceptance criteria:** Specify a counterfactual, randomization at the operational unit, an intention-to-treat analysis, a feasible measurement schedule, and a limitation on inference.
- **Time:** 30 minutes.

### Exercise 8 — Make a portfolio recommendation

A proposed deletion has a positive expected contribution but removes the only low-price, gluten-free option in six stores.
A more expensive alternative remains.
The retailer has a documented guardrail requiring an affordable option for each defined need in those stores.

- **Deliverable:** Write a decision memo recommending scale, revise, hold, or stop, and name an owner and evidence needed next.
- **Acceptance criteria:** Treat the guardrail as a constraint, discuss affordability and local availability, and distinguish forecast, plan, target, and commitment.
- **Time:** 20 minutes.

## 13. Self-check answers

### Exercise 1

Baseline breadth is four defined groups.
Proposed breadth remains four groups because at least one option remains in each.
Black-tea depth falls from 10 SKUs to 8 SKUs.
Total range falls from 18 to 16 SKUs.
Perceived variety might fall more than the count if a customer's preferred brand disappears, or might remain similar if the removed items were hard to distinguish.
That is a hypothesis requiring customer evidence, not a conclusion from the count.

### Exercise 2

Survivor A's diversion is 200 / 500 = 40%.
Survivor B's diversion is 125 / 500 = 25%.
The unaccounted outside-category share is 175 / 500 = 35%.
The shares sum to 100% under the stated one-pack assumption.
The remaining shoppers might buy at a competitor, defer, or decide not to buy; the data do not distinguish these outcomes.

### Exercise 3

The removed item's baseline contribution is 500 × USD 1.40 = USD 700.
Survivor A adds 200 × USD 1.10 = USD 220.
Survivor B adds 125 × USD 1.60 = USD 200.
Adding USD 90 of avoidable savings and subtracting the USD 35 reset gives:

```text
-700 + 220 + 200 + 90 - 35 = -225 USD for the period
```

The stated scenario loses USD 225, so hold or revise rather than scale on this calculation alone.
The recommendation is conditional on the assumed causal transfers and cost estimates.
It does not prove the actual rollout would lose exactly USD 225.

### Exercise 4

The removed contribution is 300 × USD 2.00 = USD 600.
Let `q` be transferred packs.
The first-period break-even equation is `-600 + 1.50q + 120 - 30 = 0`.
Thus `q = 510 / 1.50 = 340` packs, or 113.33% of the 300-pack baseline.
A one-for-one diversion of 70%, or 210 packs, is not sufficient.
At 70% the change is `-600 + 315 + 120 - 30 = -195 USD`.
A threshold above 100% signals that the assumed one-for-one transfers cannot break even without other changes.
It is not a prediction that diversion will exceed 100%.

### Exercise 5 rubric

A strong critique says all-store entries are not category exposures.
It identifies the simultaneous promotion as an attribution problem and recommends a control or randomization.
It notes that availability could change both sales and the effective choice set and recommends dated shelf or online availability measurement.
It explains that the dashboard contains no item-level counterfactual and therefore cannot establish transfer.
It also notes that a higher ratio does not by itself identify the psychological mechanism of overload.
A weak answer merely requests more data without saying how each piece addresses a specific bias.

### Exercise 6 rubric

Start with post-change contribution minus the contribution expected under the unchanged range for the same period.
Do not add the same survivor sales again as a modeled benefit.
Exclude unchanged allocated headquarters rent because it is not avoidable.
Count the reset cost once in the period in which it is incurred.
A strong answer also states whether any independently measured space effect is already captured in the total outcome.

### Exercise 7 rubric

A strong protocol randomizes stores, or geographic clusters if spillover makes individual-store assignment unsuitable, with an explicit allocation procedure.
It uses predeclared strata where feasible and acknowledges the limited number of independent units.
It compares incremental contribution per store-week by assigned arm, with uncertainty appropriate to the assignment unit.
It logs range, execution, prices, promotions, availability, and dates in both arms.
It defines coverage and service guardrails and identifies an owner and trigger for rollback.
It labels the result as the combined range-and-layout effect and states that spillovers and limited power constrain interpretation.
A precise sample-size claim without a variance estimate and minimum detectable effect does not satisfy the design requirement.

### Exercise 8 rubric

A strong recommendation holds or revises the proposed range because the documented affordability guardrail would fail in six stores.
It identifies an owner to validate local prices, stock, and a replacement option before approval.
It treats positive expected contribution as a forecast, not a realized result.
It defines the plan to restore coverage, the target of maintaining the guardrail, and any commitment only after the responsible operator agrees.
A weak recommendation averages across stores and assumes the more expensive alternative is acceptable to every shopper.

## 14. Interview questions

1. **How do breadth and depth differ?**
   - Strong: defines the need-state boundary and counts alternatives within it.
   - Weak: uses both words as synonyms for SKU count.
2. **What is choice overload?**
   - Strong: names an outcome, context, and counterfactual and explains that evidence is conditional.
   - Weak: says every extra product necessarily reduces sales.
3. **Why can a low-selling item be valuable?**
   - Strong: discusses unique needs, substitution, basket effects, and contribution without assuming any effect is causal.
   - Weak: ranks items by unit sales alone.
4. **How do you estimate a diversion ratio?**
   - Strong: divides causal incremental survivor demand by counterfactual displaced demand and includes an outside option.
   - Weak: divides total survivor sales by deleted sales.
5. **How do stockouts bias the analysis?**
   - Strong: explains demand censoring and the difference between temporary and permanent substitution.
   - Weak: treats zero sales as zero demand.
6. **What is the difference between category conversion and store penetration?**
   - Strong: states the denominator, exposure window, and interpretation of each measure.
   - Weak: uses all visitors as category shoppers without qualification.
7. **How would you compare a shorter range and better navigation?**
   - Strong: defines feasible arms, controls other changes, and identifies the package effect if components change together.
   - Weak: claims a bundled test isolates SKU count.
8. **How do you prevent double-counting space benefits?**
   - Strong: builds a mutually exclusive bridge between measured category contribution and separately modeled effects.
   - Weak: adds every favorable operational estimate to observed sales lift.
9. **What would you randomize?**
   - Strong: uses the operational assignment unit, considers spillovers, and analyzes uncertainty at the relevant cluster level.
   - Weak: treats every transaction as an independent randomized store.
10. **When would you reject a positive expected-value deletion?**
    - Strong: cites a documented coverage, affordability, service, legal, or contractual constraint and identifies the decision owner.
    - Weak: assumes average contribution overrides every constraint.
11. **How would you transport a result to another format?**
    - Strong: checks shopper missions, available space, price mix, execution, and uncertainty before extrapolating.
    - Weak: assumes the same diversion ratio applies everywhere.
12. **What belongs in a decision memo?**
    - Strong: separates observed evidence, forecast, plan, target, commitment, uncertainty, and rollback.
    - Weak: presents a target as if it were a measured effect.

## 15. Key terms

- **Assortment availability:** fraction of relevant exposure in stock; see [data context](#8-data-standards-and-regulatory-context).
- **Avoidable cost:** expense changed by the decision; see [economics](#step-7--convert-the-response-into-category-economics).
- **Breadth:** need states covered; see [mental model](#three-levels-of-variety).
- **Category:** portfolio boundary for the decision; see [definitions](#4-definitions-and-vocabulary).
- **Choice overload:** context-dependent impairment of a choice outcome; see [evidence](#mixed-and-conditional-evidence).
- **Comparison cost:** effort needed to evaluate options; see [mental model](#the-matching-versus-effort-trade-off).
- **Contribution:** price less specified variable costs; see [worked example](#contribution-calculation).
- **Conversion:** purchases divided by a defined exposure denominator; see [failure modes](#10-failure-modes-and-red-flags).
- **Counterfactual:** outcome under the alternative policy; see [decision framing](#step-1--frame-the-decision-and-the-counterfactual).
- **Coverage:** presence of acceptable options for a defined need; see [coverage mechanics](#step-3--count-breadth-depth-and-differentiated-coverage).
- **Deferral:** a choice not to buy now; see [response tree](#step-5--build-the-shopper-response-tree).
- **Depth:** alternatives within one need state; see [mental model](#three-levels-of-variety).
- **Diversion ratio:** incremental survivor demand divided by displaced demand; see [response tree](#step-5--build-the-shopper-response-tree).
- **Facing:** front-facing shelf position; see [operating loop](#the-operating-feedback-loop).
- **Forecast:** conditional future estimate; see [recommendation](#recommendation).
- **Guardrail:** predefined limit on an unacceptable outcome; see [test design](#step-9--design-measurement-and-implementation-together).
- **Incremental contribution:** difference in contribution versus baseline; see [calculation](#contribution-calculation).
- **Intention to treat:** analysis by assigned intervention; see [test design](#step-9--design-measurement-and-implementation-together).
- **Need state:** shopper job or constraint; see [coverage mechanics](#step-3--count-breadth-depth-and-differentiated-coverage).
- **Outside option:** response outside the measured choice set; see [response tree](#response-tree).
- **Perceived variety:** shopper's impression of differentiation; see [industry evidence](#perceived-variety-and-organization).
- **Planogram:** intended shelf arrangement; see [data contract](#minimum-data-contract).
- **Search cost:** effort required to locate options; see [mental model](#the-matching-versus-effort-trade-off).
- **SKU:** stock keeping unit; see [definitions](#4-definitions-and-vocabulary).
- **Substitution:** movement to an alternative when the offer changes; see [estimation](#step-6--estimate-substitution-without-mistaking-correlation-for-causation).
- **Treatment bundle:** multiple changes implemented together; see [failure modes](#10-failure-modes-and-red-flags).

## 16. Sources

1. Iyengar, S. S., and Lepper, M. R. (2000). “When Choice Is Demotivating: Can One Desire Too Much of a Good Thing?” *Journal of Personality and Social Psychology*, 79(6), 995–1006. [https://doi.org/10.1037/0022-3514.79.6.995](https://doi.org/10.1037/0022-3514.79.6.995). Primary experimental evidence; not a universal retail rule.
2. Scheibehenne, B., Greifeneder, R., and Todd, P. M. (2010). “Can There Ever Be Too Many Options? A Meta-Analytic Review of Choice Overload.” *Journal of Consumer Research*, 37(3), 409–425. [https://doi.org/10.1086/651235](https://doi.org/10.1086/651235). Meta-analysis and limits of an unconditional claim.
3. Chernev, A., Böckenholt, U., and Goodman, J. (2015). “Choice Overload: A Conceptual Review and Meta-Analysis.” *Journal of Consumer Psychology*, 25(2), 333–358. [https://doi.org/10.1016/j.jcps.2014.08.002](https://doi.org/10.1016/j.jcps.2014.08.002). Moderators and synthesis.
4. Kahn, B. E., and Wansink, B. (2004). “The Influence of Assortment Structure on Perceived Variety and Consumption Quantities.” *Journal of Consumer Research*, 30(4), 519–533. [https://doi.org/10.1086/380286](https://doi.org/10.1086/380286). Assortment presentation and perceived variety.
5. Broniarczyk, S. M., Hoyer, W. D., and McAlister, L. (1998). “Consumers' Perceptions of the Assortment Offered in a Grocery Category: The Impact of Item Reduction.” *Journal of Marketing Research*, 35(2), 166–176. [https://doi.org/10.1177/002224379803500203](https://doi.org/10.1177/002224379803500203). Perceived assortment and item reduction.
6. GS1. [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin). Identifier guidance; does not establish product substitutability.
7. GS1. [Global Product Classification](https://www.gs1.org/standards/gpc). Interoperable product classification guidance.
8. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Articles 5 and 6. Binding law where applicable; jurisdiction and facts matter.
9. US Food and Drug Administration. [Food Labeling & Nutrition](https://www.fda.gov/food/food-labeling-nutrition). Official regulatory resources; check product-specific and current requirements.
10. World Wide Web Consortium. [Web Content Accessibility Guidelines (WCAG) 2.2](https://www.w3.org/TR/WCAG22/). Technical guidance; legal applicability varies.

## 17. Links

- [Module 17 index](./_index.md).
- [Topic 01: The assortment decision stack](./01_assortment-decision-stack.md).
- [Topic 03: Category roles and decision trees](./03_category-roles.md).
- [Module 17 survey and its lab exercises](../../../course/17_assortment_category_retail_analytics.md).
- [Module 04: Statistics and causal inference](../../../course/04_statistics_causal_inference.md).
- [Module 05: Experimentation and pilots](../../../course/05_experimentation_pilots_evidence.md).
- [Module 27: Datasets, tools, and benchmarks](../../../course/27_datasets_tools_benchmarks.md).
