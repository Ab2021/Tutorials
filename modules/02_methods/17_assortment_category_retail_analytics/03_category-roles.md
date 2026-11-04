# Category Roles and Shopper Decision Trees

> Module 17 — Assortment, Category, and Retail Analytics  
> Topic 03 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Module 04, Module 16, topics 01 and 02  
> Version 2026.5

## 1. Why this topic exists

A grocery chain cannot make every category the reason to choose its stores.
The store has limited shelf space, inventory, labor, advertising, and management attention.
A category role makes that allocation explicit.
It says what a category is expected to contribute to the retailer's offer for a defined customer and shopping mission.
It is not a permanent label attached to a product.

A category manager may see a slow-moving specialty coffee and propose deleting it.
The finance team may prefer the higher-margin replacement.
The store manager may report that the specialty item attracts customers who otherwise shop elsewhere.
The analyst's job is not to settle this dispute by calling the coffee a destination item.
The job is to specify what would have to be true, assemble evidence, and propose a decision with a measurable downside limit.

The same chain may use coffee as a destination in a large urban store and as a routine fill-in purchase in a commuter store.
Holiday baking may be a seasonal mission, while a single-serve pain reliever may be convenience.
A different business, such as a specialist pharmacy, can assign those categories different roles.
A role is therefore a choice about the retailer's proposition, not a natural property of an item.

A shopper decision tree describes the choices or constraints shoppers consider when finding a product.
For coffee, a shopper may first need decaffeinated coffee, then select a format, then choose a brand.
Another shopper may start with a specific brand and substitute format only within that brand.
Treating one hypothetical tree as universal can erase the very need that brings a customer to the store.

This lesson connects role, decision tree, assortment, operations, and governance.
It distinguishes a customer's stated choice from observed purchase and a basket association from a causal effect.
It also makes the operational consequences visible: a role must change an assortment, service, pricing, space, or review decision.
A role that changes nothing is a presentation label rather than a management tool.

## 2. Learning outcomes

After completing this lesson, you can:

1. **Classify** a category as destination, routine, seasonal, or convenience for a stated format and mission; exercise 1 checks this distinction.
2. **Draw** a testable shopper decision tree with branches, substitutes, and an outside option; exercise 2 checks the tree.
3. **Calculate** role-specific performance measures using explicit denominators; exercises 3 and 4 check the arithmetic.
4. **Translate** a role and tree into assortment, shelf, inventory, and promotion choices; exercise 5 requires an implementable plan.
5. **Critique** association, availability, and selection biases in a proposed deletion; exercise 6 tests the critique.
6. **Design** a controlled evaluation with guardrails and a stop rule; exercise 7 tests the design.
7. **Write** an accountable category review and exception process; exercise 8 tests the governance design.

The desired result is a defendable business decision, not a particular classification algorithm.
State when the evidence is insufficient to select a single tree or establish incremental impact.

## 3. The core mental model

```text
Retail proposition + store format + shopper mission
                       |
                       v
          Choose a role and decision rights
                       |
                       v
     Observe needs, constraints, choices, and gaps
                       |
                       v
   Build and challenge shopper decision tree(s)
                       |
                       v
 Role-specific scorecard + operational constraints
                       |
                       v
 Assortment / price / promotion / shelf / service plan
                       |
                       v
 Approval -> execution -> measurement -> revision
                       ^                         |
                       +---- evidence and feedback+
```

Read the diagram from the top.
The retailer's strategy constrains what the category is meant to do.
The shopper's mission determines which attributes and substitutes matter.
The tree translates those needs into practical coverage requirements.
The role scorecard describes the benefits the retailer hopes to deliver and the costs it accepts.
The plan is then approved, executed, and measured.
Feedback can change either the tactical plan or the role itself.

A destination role gives particular weight to trip choice and credible differentiation.
That does not imply infinite variety, free inventory, or a guaranteed causal traffic effect.
A routine role emphasizes dependable mission completion and commercial efficiency.
Seasonal roles require readiness before demand peaks and disciplined exit afterward.
Convenience roles emphasize availability and speed for a narrow set of relevant needs.
All four can have financial and service guardrails.

There are two distinct trees in this lesson.
A **shopper decision tree** represents how people distinguish or substitute products.
A **management decision tree** is a procedure for deciding what to list, protect, test, or remove.
They inform each other but are not interchangeable.
A manager's preferred brand hierarchy is not proof of a shopper's hierarchy.
Likewise, a shopper tree alone does not decide whether an item meets a profitability constraint.

The tree is a model of a decision, not a claim that everyone literally thinks in a fixed order.
A customer can filter on price and dietary requirement simultaneously.
A digital shopper can revise a search after seeing availability.
Different missions may require multiple trees or a network of substitutions rather than one strict hierarchy.
The practical test is whether the representation improves a specific decision compared with a simpler alternative.

The role also needs a counterfactual.
A large basket containing coffee does not prove coffee caused the visit or generated the rest of that basket.
To estimate incremental value, compare outcomes with and without a credible change, including substitution and displaced sales.
Until such evidence exists, treat a basket relationship as a diagnostic signal, not a profit credit.

## 4. Definitions and vocabulary

| Term | Meaning | Common misuse or unit |
|---|---|---|
| Category | A managed group of products that serve related shopper needs. | Assuming the supplier's product taxonomy is automatically the shopper's category. |
| Subcategory | A more specific group inside the category. | Confusing a reporting subdivision with a shopper's first choice. |
| Category role | The intended contribution of a category to a retailer's proposition. | Treating a role as an item-level causal finding. |
| Destination | A role intended to make the retailer a preferred choice for a mission. | Calling a high-volume category destination solely because it is large. |
| Routine | A role intended to support frequent, dependable shopping missions. | Assuming routine categories never attract trips. |
| Seasonal | A role whose importance changes materially with an occasion or season. | Treating all demand peaks as identical or assuming no off-season need. |
| Convenience | A role intended to meet a quick, immediate or supplementary need. | Confusing a convenience role with a convenience-store format. |
| Mission | The customer's reason for the trip or order. | Equating a basket's contents with a verified reason for visiting. |
| Occasion | The event or use context associated with a need. | Using calendar month as the only evidence of an occasion. |
| Shopper decision tree | A representation of how needs and attributes organize a choice. | Assuming a retailer hierarchy must equal the shopper's hierarchy. |
| Branch | A set of alternatives sharing a relevant decision condition. | Treating a branch label as a unique item identifier. |
| Leaf | An end segment of the model containing one or more possible items. | Assuming every leaf must contain exactly one stock keeping unit. |
| Stock keeping unit (SKU) | An inventory-managed item variant. | Using SKU counts without distinguishing pack size or active status. |
| Need state | A use requirement or preference relevant to choice. | Inferring all needs from purchased products alone. |
| Hard constraint | A condition a shopper or business cannot accept violating. | Treating a strong preference as a medical or legal requirement. |
| Preference | An attribute valued by a shopper that may be traded off. | Treating a preference as universal across missions. |
| Outside option | The shopper does not buy from the offered set, including shopping elsewhere. | Forcing every lost sale into another item in the category. |
| Substitution | A change to an alternative when the preferred item is unavailable or removed. | Assuming all alternatives preserve the same margin or trip. |
| Transfer rate | The fraction of affected demand redirected to a specified alternative or set. | Reporting a percentage without its denominator, horizon, or outside option. |
| Coverage | Extent to which defined need branches have an available acceptable option. | Calling every branch equally important without a stated weighting. |
| Availability | Whether the item is offered and in stock at the relevant time. | Counting a ranged but empty item as a viable option. |
| In-stock rate | In-stock observations divided by eligible observations, expressed as a percentage. | Using end-of-day snapshots as proof of all-day availability. |
| Facing | One product-width position visible at the front of a shelf. | Equating facings with total shelf depth or actual sales. |
| Planogram | An instruction specifying item placement and shelf allocation. | Assuming the instruction reflects execution in every store. |
| Penetration | The share of an explicitly defined population or trips with a purchase. | Comparing household penetration with trip penetration as if equivalent. |
| Basket | Products purchased in one transaction or order. | Interpreting products in a basket as incremental profit. |
| Attach rate | Baskets containing both a focal item and an attached item divided by focal-item baskets. | Reversing the denominator or claiming causality. |
| Contribution | Revenue less explicitly specified variable costs. | Calling gross sales or gross margin the full economic contribution. |
| Gross margin | Net sales minus cost of goods sold, under a stated accounting policy. | Including or excluding supplier funding inconsistently. |
| Incrementality | Change in an outcome relative to the credible no-change counterfactual. | Treating before-and-after change as automatically incremental. |
| Guardrail | A boundary on an outcome that limits an otherwise attractive plan. | A target with no owner, measurement, or consequence. |
| Category captain | A supplier providing category insights or recommendations to a retailer. | Delegating the retailer's independent assortment decision to the supplier. |
| Decision rights | Named authority to recommend, approve, execute, and review. | Assuming participation in a meeting means approval authority. |
| Forecast | An estimate of a future outcome under specified assumptions. | Calling a forecast a commitment. |
| Plan | A selected set of intended actions and resources. | Assuming a plan necessarily achieves the forecast. |
| Target | A desired performance level. | Treating a target as a predicted outcome. |
| Commitment | A binding or formally agreed delivery obligation. | Treating an unapproved target as a supplier commitment. |

## 5. The mechanics, step by step

### Step 1 — Define the decision and its boundary

**Input:** strategy, store format, channel, review date, and the decision requested.
**Action:** name the category, affected stores, shoppers, period, and available levers.
**Output:** a one-page decision brief with owner and approval date.

A useful brief says, for example: decide the coffee range and service standards for urban supermarkets for the next 26 weeks.
A weak brief says: optimize coffee.
The first names where, when, and what can change.
It still needs a budget and a statement of what success means.

List the alternatives before examining a ranking.
Possible options include retaining the current range, replacing a duplicate, expanding a missing need branch, changing facings, or running a pilot.
The no-change option is a real alternative, not an analytical failure.
Also record what is not in scope, such as a store-wide price repositioning or a supplier contract renegotiation.

Define whether the decision is national, cluster-level, store-level, or channel-specific.
A click-and-collect customer may value searchable filters and accurate substitution settings.
An in-store customer may value shelf navigation and visible product attributes.
A category role may be shared across channels while execution differs.
Record the choice rather than assuming either uniformity or complete localization.

### Step 2 — Establish the retailer proposition and missions

**Input:** retailer strategy, customer research, store format, and trading history.
**Action:** write the role the category should play for the intended customer.
**Output:** a role hypothesis, alternative roles, and evidence gaps.

Ask what failure would matter most to the retailer.
If customers select the store for a distinctive range, the destination hypothesis deserves examination.
If customers expect to complete a regular shop without special effort, consider routine.
If demand is tied to an identifiable event, consider seasonal.
If the need is urgent, supplementary, or impulse-led, consider convenience.

These are management archetypes, not a mutually exclusive empirical truth.
A retailer should choose a primary role for a defined scope to resolve trade-offs.
It can document a secondary role or a different role in a store cluster.
A destination designation needs stronger evidence than an executive's preference.
Relevant evidence can include mission surveys, switching studies, market position, and performance of controlled changes.

Record the proposed role in a falsifiable sentence.
For example: urban coffee should help shoppers choose this store for their weekly coffee mission, subject to a minimum contribution and service budget.
Then write what would change your mind.
A finding that customers consistently buy their preferred coffee elsewhere despite the range, or that the role cannot be supported economically, should trigger review.

### Step 3 — Choose role-specific objectives and guardrails

**Input:** the role hypothesis, finance policy, service constraints, and measurement capacity.
**Action:** distinguish the primary objective from secondary measures and hard guardrails.
**Output:** a scorecard with formulas, baseline, target, owner, and review frequency.

For destination, consider mission-relevant penetration, choice of store, key-branch availability, and share where representative market data exist.
Retain a contribution and inventory guardrail.
For routine, consider mission completion, repeat availability, category contribution, and price credibility.
Do not reward high fill rate if basic needs remain uncovered.

For seasonal, measure readiness before demand, event-period sales or contribution, availability during the event, and residual stock afterward.
Specify the event window before looking at the result.
For convenience, measure availability of essential options, transaction conversion where observable, time to find, and contribution per constrained unit of space.
Do not infer that a high-margin product belongs on a shelf if shoppers cannot find the expected basic option.

For each metric, specify its numerator and denominator.
For example, trip penetration is category-purchase transactions divided by all eligible transactions.
It is not household penetration, and neither measure directly identifies trip causation.
A scorecard also needs a time bucket, currency where applicable, exclusions, and a person who can correct its data.

### Step 4 — Map the category and available alternatives

**Input:** item master, supplier information, product attributes, historical listings, inventory, and shelf records.
**Action:** assemble the category universe, including currently delisted items and meaningful alternatives outside the category.
**Output:** an item-to-attribute table with effective dates and unresolved mappings.

Identify item variants using the retailer SKU and, where supplied, the Global Trade Item Number (GTIN).
Map packs, sizes, brand, format, dietary attributes, price tier, and store eligibility.
Keep the original identifiers and date ranges so that a pack change does not silently merge distinct offers.
Check whether private-label and national-brand variants are encoded consistently.

Do not build the choice set from sales alone.
An item with no observed sales may be out of stock, unlisted, misclassified, or genuinely unwanted.
An item that has never been listed cannot appear as a purchased substitute.
Add other-store, other-channel, and no-purchase options to the conceptual set even when their shares are unknown.

Reconcile range and availability by store and day when data permit.
Separate planned assortment from actual shelf execution.
An item coded as active in the master but absent from a particular store is not an observed available alternative there.
Record known data gaps rather than assigning false zero demand.

### Step 5 — Gather direct and behavioral evidence

**Input:** interviews, surveys, transaction data, availability records, and observed substitutions.
**Action:** triangulate what shoppers say, what they do, and what they could actually buy.
**Output:** an evidence table for each candidate branch and substitution path.

Ask shoppers about the mission before asking them to rank product features.
Probe what they would do if their first choice were unavailable.
Separate a hard requirement from a preference and from a convenience preference that depends on price.
Recruit beyond the retailer's loyal shoppers when feasible: shoppers who already left are especially important for destination hypotheses.

Behavioral records provide scale but only for observable choices.
For online orders, distinguish customer-selected substitutions from retailer-selected substitutions and rejected substitutes.
For in-store purchases, a nearby product bought during an outage is suggestive but may not represent the absent shopper's choice.
A natural experiment or randomized availability change can improve inference when it is feasible and ethical.

Use basket patterns to identify possible related missions, not to assign all attached margin to the focal category.
An observed co-purchase can reflect store size, customer affluence, promotion, or the weekly shopping mission.
Document confidence and source for each insight.
Contradictory research is a reason to segment or test, not a reason to hide a result.

### Step 6 — Build and challenge shopper decision trees

**Input:** product attributes and evidence by mission or segment.
**Action:** draw branches that represent discriminating needs, then test alternatives.
**Output:** one or more versioned trees with coverage and confidence annotations.

A practical draft begins with an entry point: what is the shopper trying to accomplish?
For coffee, candidate top-level splits may be caffeine requirement, brewing format, brand, or price.
Do not decide the order by looking only at the product hierarchy.
Ask whether an item across the proposed split is an acceptable substitute when the preferred item is absent.

A tree for a dietary constraint may start with decaffeinated versus regular, then format and price tier.
A brand-loyal segment may start with brand, then format.
Both can be true for different shoppers; the business needs to know how important each segment is.
A tree can record a hard constraint above soft preferences without implying all shoppers process choices sequentially.

At each leaf, list observed offers, viable substitutes, and evidence quality.
Add an outside option for every leaf.
Record branches with no acceptable current offer as unmet needs, not zero-value segments.
If a proposed split does not change listing, shelf, search, or substitution decisions, remove it or explain why it remains useful.

Challenge the tree against disconfirming cases.
Can a customer cross the branch at a small price difference?
Does a brand-first customer accept private label when their brand is unavailable?
Does online search behavior differ from in-store navigation?
Are apparent preferences an artifact of unequal availability or promotion?

### Step 7 — Translate the tree into assortment requirements

**Input:** role, tree, available space, eligible item economics, and operational constraints.
**Action:** define essential branch coverage and only then compare depth within branches.
**Output:** a proposed assortment with retain, replace, add, test, or remove decisions.

First protect mandatory business or customer requirements.
A dietary claim must be supported by reliable product data; it is not inferred from branding.
A branch with one genuinely acceptable item has a single-point-of-failure risk.
Retain it or find and validate a like-for-like replacement before removal if the role requires that need to be served.

Next distinguish breadth from depth.
Breadth is coverage across relevant need branches.
Depth is the number of alternatives within a branch.
A destination role may justify depth where shoppers value genuine differences, but duplicate depth can crowd out a distinct need.
Routine and convenience roles may have less depth while still requiring complete coverage of their chosen mission.

For each candidate deletion, estimate own contribution lost, credible transferred contribution, operating savings, and any measurable incremental effects elsewhere.
Do not add a full basket's margin to every item in the basket.
If a causal estimate is unavailable, show a scenario range and leave unmeasured effects uncredited.
Ask whether the decision remains acceptable at a conservative transfer assumption.

### Step 8 — Convert the assortment into space and service

**Input:** selected items, dimensions, demand forecast, replenishment capacity, and store layout.
**Action:** set facings, placement, replenishment parameters, and online navigation.
**Output:** an executable planogram and service plan with named owners.

A listed SKU without shelf capacity or replenishment is not a reliable offer.
Estimate demand during replenishment lead time and check whether available shelf and backroom capacity can support it.
Use an explicit service goal for mission-critical branches.
Place related items so shoppers can compare alternatives without hiding genuinely different needs.

A destination plan may allocate visible space to distinctive alternatives and protect availability on important branches.
A routine plan may emphasize fast replenishment and clear everyday price communication.
A seasonal plan needs staging dates, inbound timing, event display, and a markdown or return decision.
A convenience plan may prioritize a few immediately recognizable products in a high-access location.
All these actions remain constrained by safety, contracts, shelf geometry, and store labor.

For digital channels, map the same attributes to filters, search synonyms, images, and substitution rules.
A well-designed physical decision tree is ineffective if online shoppers cannot discover the relevant leaf.
Test whether the actual fulfillment store carries the displayed items.
Avoid promoting an unavailable SKU as the solution to a customer need.

### Step 9 — Design the decision and evaluation

**Input:** options, cost and capacity estimates, evidence confidence, and risk appetite.
**Action:** compare alternatives against role objectives, constraints, and the no-change counterfactual.
**Output:** a recommendation, scenario table, decision owner, and evaluation design.

Distinguish observed baseline from forecast and target.
The forecast is the outcome estimated under stated assumptions.
The plan is the chosen range and actions.
The target is the desired outcome, and a commitment exists only where an authorized party agrees to deliver something specific.
None is interchangeable with actual results.

Where feasible, randomize matched stores or clusters to a change and a control, including enough time for replenishment and shopper adaptation.
Choose the primary metric before launch and specify the time window.
Estimate the treatment-minus-control change relative to each group's baseline.
Account for clustering, seasonality, concurrent promotions, and the possibility that shoppers cross between stores.
A pilot too small to resolve the decision can still establish feasibility, but it does not establish a precise effect.

State the stop rule and response before launch.
For example, pause a deletion if a critical need branch has no acceptable in-stock offer for a specified number of days.
An exception is a documented decision with an owner and expiry, not a silent override.
Predefine how to act if economic results improve but service deteriorates.

### Step 10 — Govern execution and review

**Input:** approved plan, supplier and store constraints, baseline, and evaluation schedule.
**Action:** assign decision rights, implement, monitor, and capture exceptions.
**Output:** a decision log, execution dashboard, and dated post-implementation review.

The category manager recommends; the commercial owner approves within delegated authority.
Finance validates the contribution definition and assumptions.
Operations confirms shelf, replenishment, and labor feasibility.
Data owners validate item, transaction, and availability records.
Legal and compliance review claims and supplier conduct when relevant.
Store teams report execution and customer friction.

A supplier or category captain can provide insights and product information.
The retailer should independently challenge assumptions, protect competing suppliers' confidential information, and retain its own decision rights.
Do not exchange competitors' commercially sensitive terms through a common supplier.
The exact legal obligations depend on jurisdiction and facts; seek qualified advice.

Review operational indicators early and commercial outcomes after enough time to observe them.
Separate an execution failure from a failed strategy: a range cannot be fairly evaluated if half the pilot stores never installed it.
At the review, choose scale, modify, extend, reverse, or retire.
Record the evidence, uncertainty, exception owners, and next review date.

### Role playbook A — Destination

**Decision question:** what must the retailer do to be a credible first choice for this mission?
The answer starts with the mission and the intended customer, not with an automatic instruction to maximize SKU count.
Write down the competing store or channel against which the proposition will be judged.
Where market share is unavailable, use an explicitly narrower measure rather than calling internal sales share market share.

The range should cover the distinctive needs that justify the destination claim.
Some branches may need several alternatives because shoppers perceive meaningful differences.
Others may need only one reliable offer.
The review should show which branches received depth and why.
Compare each additional alternative with the cost of the space and inventory it displaces.

The service plan should specify which branches are essential and what happens if their sole item is unavailable.
A frequent stockout of a highly differentiated item can undermine the claimed proposition.
Conversely, a destination category does not require a 100% service target for every long-tail item.
The economically appropriate service level depends on demand, supply variability, customer consequences, and capacity.
Document the rationale instead of giving every SKU an identical target.

Price also matters to a destination proposition.
Some customers may come for a distinctive premium range; others may expect an attractive entry price.
Check whether the proposed pricing architecture supports the chosen mission.
Do not assert that a premium price or a low price alone creates a destination.
The relevant test is whether the combined range, price, service, and experience are credible for the intended shopper.

Potential leading measures include availability of protected branches and the share of target needs with a validated offer.
Potential lagging measures include contribution, customer retention where observable, and survey-based store preference.
Each measure needs a population, time window, and owner.
When a result worsens, first check whether the plan was implemented as designed.
Then distinguish proposition failure from a competitor action, a local supply interruption, or a measurement change.

**Branch point:** if the evidence supports distinct needs but not the value of additional depth, protect coverage and test depth.
If coverage is strong but shoppers cannot identify the offer, test navigation or communication.
If the required service cost exceeds the approved guardrail, escalate the role and resource trade-off.
Do not silently reclassify the category merely to improve a dashboard.

### Role playbook B — Routine

**Decision question:** can a shopper reliably complete a recurring mission without disproportionate effort or cost?
Routine does not mean unimportant.
A frequently needed category can have a large effect on a shopper's overall experience even when it is not the reason for choosing the store.
Its role favors consistent availability and a predictable offer.

Define the essential branches for the relevant format.
For milk, those branches might include pack size, fat content, and a customer-relevant dietary option.
The exact choices require evidence and should not be copied from another retailer without validation.
Compare category breadth against the missions expected in the store.
Then assess depth within each branch for duplication.

Set replenishment and shelf capacity using demand patterns and store operating constraints.
If a common product sells out before replenishment, an additional facing or a schedule change may matter more than another variant.
If demand is low and intermittent, more shelf space may increase waste rather than service.
Use shelf and inventory observations together; a stock record does not guarantee an accessible product.

Review everyday price credibility and relevant pack-price comparisons.
Use a consistent unit, such as USD per liter, where products can meaningfully be compared.
A lower shelf price for a smaller pack is not necessarily a lower unit price.
Unit price is useful but does not replace the shopper's affordability constraint at checkout.
The cheapest price per liter may still require an unaffordable package outlay.

Use repeatable measurements such as in-stock observations, contribution per store-week, and eligible-trip penetration.
If the category has meaningful spoilage, include waste under a clearly defined cost treatment.
Review performance by store cluster so that a chain average does not conceal a consistently underserved group.
Investigate whether missing sales coincide with stockouts or changes in store hours.

**Branch point:** if service is weak, fix supply and execution before interpreting low sales as lack of demand.
If the need is well served and several items are genuinely interchangeable, a controlled depth reduction may be reasonable.
If a proposed deletion removes the only acceptable option for a material need, seek a replacement or explicitly approve the service loss.
Document the decision even when the category is operationally routine.

### Role playbook C — Seasonal

**Decision question:** can the retailer meet the event mission at the right time without unacceptable residual exposure?
An event can have a short selling window and a long procurement lead time.
The relevant decision calendar therefore begins well before the first customer display.
It continues through post-event stock disposition and learning.

Define the event start, peak, and end using the relevant market and shopper behavior.
For a movable holiday, calendar-week comparisons can be misleading.
Align historical observations to event-relative weeks where appropriate.
Identify whether products also have meaningful off-season uses.
Avoid deleting an everyday-use branch merely because the event has ended.

Estimate demand for baseline products, event-specific products, and promotional displays separately.
Describe the uncertainty around each estimate.
For novelty items, limited history may require analogs or scenarios rather than a precise statistical model.
An analog is a stated assumption, not evidence that the new item will behave identically.
Consider supplier minimum orders, transport capacity, and shelf installation lead time.

Create three dates: purchase approval, shelf-ready deadline, and exit decision.
The purchase approval precedes the supplier lead time plus any required internal processing.
The shelf-ready deadline precedes the shopper's event preparation period.
The exit decision specifies whether to retain, transfer, mark down, donate where appropriate, or otherwise dispose of stock.
Any return to supplier requires an actual contractual right or negotiated agreement.

Measure readiness as a leading indicator.
For example, divide the number of planned item-store combinations delivered by the agreed deadline by the total planned item-store combinations.
Measure peak availability with a time-stamped observation and explicit eligibility.
Measure residual units and realized contribution after the event.
Explain whether markdown funding, spoilage, and disposal costs are included.

Compare a lost-sale risk with an excess-stock risk rather than minimizing either in isolation.
A conservative order can reduce residuals while sacrificing the event mission.
An aggressive order can protect service while consuming cash and requiring markdowns.
The right trade-off depends on role, shelf capacity, supplier lead time, and the economics of the specific products.

**Branch point:** if an essential item is late, activate the agreed substitute or distribution escalation.
If early sell-through is below plan, check whether the display is installed and the forecast horizon remains valid.
If stock is high near the exit date, execute the authorized disposition rather than waiting for the next annual review.
Capture the reason for any override so the next event does not repeat the same error.

### Role playbook D — Convenience

**Decision question:** can the shopper find an acceptable answer quickly when the need arises?
Convenience is a role relative to a mission, not a synonym for a small store.
A supermarket can use a convenience role for a narrow urgent-need assortment.
A specialist store may treat the same products as a core destination offer.

Identify the immediate need and the location or digital journey in which it appears.
For an urgent personal-care item, the shopper may care more about recognizable function and availability than extensive brand choice.
For a snack near checkout, the purchase may be supplementary rather than the reason for the trip.
Neither statement applies automatically to every shopper.
Use observation and customer research to distinguish the relevant mission.

Select a narrow range that covers the essential needs.
Avoid interpreting narrowness as permission to omit product safety, age restrictions, or other applicable requirements.
The item's placement and signage should help the customer identify the correct option.
Do not use a high margin as a substitute for suitability.
Where legal or clinical questions arise, refer to the applicable policy and qualified experts.

Evaluate contribution per unit of constrained space with a clearly defined period.
Include replenishment effort and waste if material.
If an item is often unavailable at the decision point, a high margin per sold unit can conceal lost demand.
Check the rate at which the selected offer is physically accessible during store operating hours.
An end-of-day inventory snapshot may not represent checkout availability.

For digital orders, convenience can mean discoverability, accurate availability, and a simple substitution choice.
The customer may not see the physical adjacency that creates an in-store impulse.
Do not copy physical shelf logic to a search interface without a usability test.
Treat online and in-store results as potentially different populations and missions.

**Branch point:** if customers cannot find the item, test location or navigation before increasing assortment.
If a single option frequently fails to meet the urgent need, evaluate a second distinct option.
If many variants compete for attention without meeting distinct needs, consider a controlled simplification.
Always evaluate the impact on service as well as the space released.

### Governance artifacts and review cadence

A working category charter fits on one page but has several non-negotiable fields.
It names the category boundary and effective dates.
It states the primary role, the intended customer or mission, and any approved local exceptions.
It identifies the decision owner and the person authorized to approve the budget.
It specifies the primary objective and at least one service and one economic guardrail.
It links to the evidence and sets a review date.

The review pack then expands the evidence behind the charter.
It contains the current and proposed trees, with the source and confidence of each material branch.
It shows the current assortment and the proposed changes by store cluster and channel.
It includes the baseline, forecast, target, and any separately authorized commitments.
It explains how units, prices, costs, and time windows were aligned.
It includes a no-change option and the consequence of each material alternative.

Record uncertainty beside the decision it affects.
If transfer is the critical unknown, show sensitivity to transfer.
If lead time is critical, show the outcome of a late delivery.
If customer need coverage is disputed, document the missing evidence and interim protection.
Avoid a generic disclaimer that does not identify which assumption could reverse the decision.

The approval record should distinguish recommendation, approval, consultation, and implementation.
The category manager may recommend a range and own the review pack.
The commercial director or delegated owner approves the role and resource trade-off.
Finance confirms the economic bridge and funding treatment.
Supply planning confirms orders and replenishment capacity.
Operations confirms that stores can execute the plan.
Data stewardship confirms the definitions and quality of the input data.
Legal, privacy, quality, and compliance teams participate when their domains are affected.
The precise titles vary; the accountable authority must nevertheless be unambiguous.

An exception register makes local knowledge usable without losing control of the plan.
Each entry identifies the store or cluster, affected SKU or branch, requested deviation, reason, and expected duration.
It records the impact on the role objective and relevant guardrails.
It names an approver and an expiry date.
It also names the evidence that will trigger early review.
An expired exception must be renewed or closed, not allowed to become an invisible permanent assortment.

An operating cadence should match the speed of the decision.
Review an event launch frequently enough to detect a missed delivery before the peak.
Review a long-term destination proposition over a window long enough to observe shopper response.
Check data and execution early, then evaluate business outcomes after the agreed observation period.
Use the same definitions for the baseline and evaluation windows.
Where they cannot be made comparable, document the break and the effect on interpretation.

Escalate when an economic objective and a protected customer need conflict.
For example, a replacement may improve expected contribution while removing the only affordable decaffeinated option.
The category manager should not silently treat the service loss as zero.
The commercial owner can approve a different trade-off, fund an alternative, or request more evidence.
The resulting decision should describe who is affected and when it will be revisited.

### Worked governance handoff checklist

Before approving a range change, the category owner should be able to answer these questions:

1. Which shopper mission and role does the change support?
2. Which tree branches gain or lose an acceptable option?
3. Which stores and channels will execute the change, and when?
4. Are item attributes, physical dimensions, and costs verified?
5. Which forecast assumptions drive the recommendation?
6. What does the no-change option cost or risk?
7. Who has approved the economic and service guardrails?
8. How will the team know that the change was actually implemented?
9. Who can pause or reverse the decision if a guardrail is breached?
10. What evidence and date will trigger the final review?

If an answer is unknown, record the uncertainty and decide whether the risk is acceptable before approval.
The checklist is not a replacement for evidence or the retailer's formal authorization process.

## 6. Worked example

### Business situation

The following retailer, data, and estimates are fictional and designed for instruction.
A regional chain operates 24 urban supermarkets.
It must decide whether to delete one decaffeinated ground-coffee SKU from its coffee category for a 12-week trial.
Coffee is proposed as a **destination** role for these stores.
The retailer believes a credible coffee range can influence store choice, but has not yet established the incremental trip effect.

The decision concerns one SKU, not the role of coffee in every format.
All amounts are US dollars (USD), and the analysis horizon is 12 weeks across the 24 stores.
Net contribution means selling revenue less product cost and directly attributable variable handling costs, before fixed overhead.
Customer identifiers are available for some shoppers, but the primary calculations use transactions.

The current plan has 40 coffee SKUs and 96 shelf facings per store.
The proposed SKU uses two facings per store.
It is the only listed decaffeinated ground coffee in the value price tier.
A premium decaffeinated ground alternative remains, as do regular ground and decaffeinated pods.
Store interviews suggest some shoppers will accept the premium alternative, while others may defer or buy elsewhere.

### Role and mission hypotheses

The category team's hypothesis is that coffee helps customers choose the store for a weekly coffee mission.
Its operational implication is to protect important formats and caffeine requirements, not to retain every existing brand.
The scorecard includes eligible-trip penetration, in-stock rate for protected branches, and category contribution.
The financial guardrail requires any pilot to avoid a forecast contribution reduction greater than USD 1,000 over 12 weeks.
The service guardrail requires an acceptable in-stock option in each protected need branch.
The contribution figure is a management limit in this example, not an externally established standard.

A candidate tree for one mission is:

```text
Coffee for home brewing
|-- caffeine requirement
|   |-- decaffeinated
|   |   |-- brewing format
|   |       |-- ground
|   |       |   |-- value tier: SKU D1
|   |       |   |-- premium tier: SKU D2
|   |       |-- pods: several SKUs
|   |-- regular
|       |-- ground / beans / pods
|-- outside option: defer, buy elsewhere, or abandon mission
```

The tree makes price tier a potential differentiator, not proof that every customer refuses to cross it.
A separate brand-first tree may describe another segment.
If D1 is removed, D2 can cover the physical decaffeinated-ground branch, but it may not meet a value-constrained need.
The team needs evidence about that distinction before calling the removal harmless.

### Baseline measures

During the 12-week baseline, the 24 stores recorded 192,000 eligible transactions.
Coffee appeared in 27,840 transactions.
Coffee trip penetration is 27,840 / 192,000 = 0.145, or **14.5% of eligible transactions**.
This is an association with a purchase, not the percentage of trips caused by coffee.

The D1 SKU sold 2,400 units at USD 2.10 contribution per unit.
Its own contribution was 2,400 × USD 2.10 = **USD 5,040** for 12 weeks.
The 2,400 units are not necessarily 2,400 different people or trips.
The SKU was in stock in 1,092 of 1,200 eligible store-day observations, or 91.0%.
A daily observation is a coarse snapshot and can miss an intraday stockout.
Low availability may suppress measured demand, so the 2,400-unit baseline is not automatically unconstrained demand.

Of 2,000 observed D1-purchase baskets with a complete set of category tags, 680 also contained bakery.
The bakery attach rate is 680 / 2,000 = **34.0% of D1 baskets**.
This does not mean D1 caused 680 bakery purchases.
The attach rate's denominator differs from the 2,400-unit sales count.
Multiple units can occur in a single basket, and the tagged basket sample is incomplete.

### Deletion scenarios

The team proposes a forecast for the same 12-week horizon, assuming affected demand is 2,400 units.
Each transferred unit is assumed to generate USD 2.40 contribution in the remaining range.
The assumptions are illustrative; the transferred margin could differ by alternative.
A deletion is expected to save USD 720 in direct handling and shelf maintenance over 12 weeks.
The contribution definition excludes that separately stated saving to avoid double counting.
No downstream trip or bakery effect is credited because no causal estimate is available.

| Scenario | Transfer rate of affected units | Transferred units | Transferred contribution | Handling saving | Change vs current |
|---|---:|---:|---:|---:|---:|
| Low | 40% | 960 units | USD 2,304 | USD 720 | −USD 2,016 |
| Base | 60% | 1,440 units | USD 3,456 | USD 720 | −USD 864 |
| High | 80% | 1,920 units | USD 4,608 | USD 720 | +USD 288 |

The change is transferred contribution plus handling saving minus the current USD 5,040 contribution.
For the base case, USD 3,456 + USD 720 − USD 5,040 = **−USD 864**.
The low case fails the USD 1,000 loss guardrail; the base case passes it narrowly.
The high case is positive before any unmeasured downstream effects.
These scenarios are forecasts conditional on transfer assumptions, not results or commitments.

The break-even transfer rate solves:

```text
2,400 affected units × transfer rate × USD 2.40/unit
    + USD 720 savings − USD 5,040 current contribution = USD 0
transfer rate = USD 4,320 / USD 5,760 = 75%
```

The rate needed to stay within a USD 1,000 loss is:

```text
2,400 × transfer rate × USD 2.40 + USD 720 − USD 5,040 >= −USD 1,000
transfer rate >= USD 3,320 / USD 5,760 = 57.64%
```

These thresholds exclude the opportunity value of the freed facings and any incremental trip effect.
Include either only if measured or estimated consistently, with its own uncertainty.
The two-facings-per-store saving has already been represented only by the stated USD 720 handling saving; do not count that saving twice.

### Decision and implementation

The category team recommends retaining D1 until it can establish whether the value-tier branch has an acceptable alternative.
The manager may instead test a validated like-for-like replacement in a subset of stores.
This is a plan recommendation, not proof that D1 creates destination traffic.
An alternative supplier item must meet the relevant product, quality, cost, and supply requirements before approval.

A 12-store replacement pilot and 12-store control could compare outcomes if stores are matched and randomly assigned within pairs.
The pilot must be sized for the effect and variability of interest; 12 stores per arm may be too small for a precise trip effect.
The primary decision measure is the difference in changes in coffee contribution per store-week.
Secondary measures include value-tier decaf availability, total store contribution, and documented customer complaints.
Use transaction penetration as a diagnostic, not a causal estimate of store choice by itself.

Before launch, record baseline dates, store matching, promotion plans, eligible transactions, and the treatment start date.
The operations owner checks that the replacement is physically present and correctly labeled.
Pause in an affected store if its acceptable value-tier decaf option is unavailable for two consecutive daily checks.
The two-day trigger is an illustrative operating rule, not a statistical significance threshold.
A review after 12 weeks can scale, modify, or reverse the change based on the agreed guardrails and uncertainty.

### Interpret the decision

A margin-only rank would flag D1 because it consumes space and has modest own contribution.
The tree reveals a potentially distinct value-constrained need.
The role elevates service and trip risk, but does not quantify either.
The scenario table shows that the result depends on transfer and that a 60% transfer assumption is close to the financial limit.
The pilot provides a route to learn without asserting that a basket association is a traffic effect.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the deletion scenarios and the two transfer thresholds in the worked example.
Save it as `category_roles_example.py` and run `python category_roles_example.py`.
It is a deterministic scenario calculator, not an estimation model or a recommendation engine.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class DeletionCase:
    affected_units: int
    old_contribution_per_unit: float
    transferred_contribution_per_unit: float
    handling_saving: float
    allowed_loss: float

    @property
    def current_contribution(self) -> float:
        return self.affected_units * self.old_contribution_per_unit

    def change(self, transfer_rate: float) -> float:
        if not 0.0 <= transfer_rate <= 1.0:
            raise ValueError("transfer rate must be between zero and one")
        transferred = (
            self.affected_units
            * transfer_rate
            * self.transferred_contribution_per_unit
        )
        return transferred + self.handling_saving - self.current_contribution

    def threshold(self, permitted_loss: float) -> float:
        denominator = (
            self.affected_units * self.transferred_contribution_per_unit
        )
        if denominator <= 0:
            raise ValueError("positive transferred contribution is required")
        return (
            self.current_contribution - self.handling_saving - permitted_loss
        ) / denominator


case = DeletionCase(
    affected_units=2400,
    old_contribution_per_unit=2.10,
    transferred_contribution_per_unit=2.40,
    handling_saving=720.0,
    allowed_loss=1000.0,
)

print(f"Current contribution: USD {case.current_contribution:,.2f}")
for label, rate in (("Low", 0.40), ("Base", 0.60), ("High", 0.80)):
    print(f"{label}: transfer {rate:.0%}, change USD {case.change(rate):,.2f}")
print(f"Break-even transfer: {case.threshold(0.0):.2%}")
print(f"Transfer for loss guardrail: {case.threshold(case.allowed_loss):.2%}")
```

Expected output:

```text
Current contribution: USD 5,040.00
Low: transfer 40%, change USD -2,016.00
Base: transfer 60%, change USD -864.00
High: transfer 80%, change USD 288.00
Break-even transfer: 75.00%
Transfer for loss guardrail: 57.64%
```

The input is 2,400 affected units over 12 weeks in 24 stores.
The code assumes all transferred units earn the same contribution and that the saving is incremental to the per-unit contribution definition.
It holds demand fixed and does not estimate unmet need, cannibalization of other categories, or effects on trip frequency.
It also treats transfer as a proportion of units rather than households or baskets.
Those distinctions matter if pack sizes or number of units per purchase differ.

A threshold above 100% means the specified condition cannot be achieved under these assumptions.
A threshold below 0% means the specified condition is already met even with no transfer, under these assumptions.
Neither result validates the inputs or proves that a service guardrail is satisfied.
Use decimal arithmetic rather than binary floating point if exact currency settlement is required; this illustration rounds only for display.

## 8. Data, standards, and regulatory context

### Minimum data contract

A decision-ready dataset needs a category definition with effective dates.
It needs an item master with retailer SKU, GTIN where available, pack, attributes, unit of measure, and supplier.
It needs a store and channel hierarchy with store opening and closure dates.
Transaction lines need transaction ID, timestamp, store or channel, SKU, units, net sales, discounts, and returns handling.
Inventory and availability need an observation timestamp and a clear definition of an eligible item-store observation.

Keep the raw and normalized attributes together with source and date.
For example, a caffeine classification may come from verified product information rather than a text match on the description.
An incomplete attribute is **unknown**, not automatically regular coffee.
A changed pack can affect unit economics, price comparisons, and substitutability.
Record mappings over time rather than overwriting history.

A planogram describes intended shelf positions.
A shelf audit or image can indicate execution, but its own coverage and timing must be documented.
An inventory record is not always proof that a shopper could locate an item.
Distinguish ranged, ordered, delivered, shelf-present, and available-to-promise states.

### Identifiers and product information

GS1's GTIN identifies trade items under GS1 allocation rules; it is not a category-role code or a statement of interchangeability.
GS1 Global Data Synchronisation Network materials describe sharing standardized product data among trading partners.
A GTIN does not replace a retailer's internal SKU, store-level availability history, or verified shopper need mapping.
Consult the current GS1 specifications for the relevant market and implementation.

Standard product identifiers improve joins, but identification errors still require data stewardship.
A product's physical dimensions and package quantity may change at a different time from its commercial attributes.
Validate joins at the appropriate item and effective-date grain.
Do not infer that two items with the same brand are acceptable substitutes.

### Privacy and competition

Shopper, loyalty, and location data can be personal data where they identify or relate to an identifiable person.
For processing covered by the European Union General Data Protection Regulation (GDPR), principles in Article 5 include purpose limitation, data minimisation, and storage limitation.
Applicable legal bases and obligations depend on the facts and jurisdiction.
The regulation does not prescribe a particular category tree or guarantee that a de-identified extract is anonymous.
Use approved access controls, retention rules, and disclosure review; consult the organization’s privacy team.

Supplier participation requires competition-law controls appropriate to the jurisdiction.
The US Federal Trade Commission and Department of Justice provide general antitrust guidance for businesses.
A supplier's category recommendation is not, by itself, evidence of an infringement.
Nevertheless, sharing rival suppliers' sensitive terms or coordinating competing retailers' decisions can create legal risk depending on circumstances.
The retailer should control access and approvals and seek qualified advice rather than relying on this lesson as legal guidance.

As of November 2026, these sources are cited as legal and standards reference points, not as a complete inventory of current obligations.
Check amendments, local rules, contracts, and product-specific requirements before implementing a decision.
In particular, dietary, health, and price claims need evidence and review under the rules applicable to the market and product.
A tree label is not a compliant product claim.

### Measurement integrity

Align sales and inventory at store, SKU, and time grain before calculating availability-adjusted outcomes.
Decide how to handle canceled orders, substitutions, refunds, employee purchases, and multi-unit packs.
A transaction-based penetration denominator excludes customers who entered but did not buy; count visitors separately only if a reliable, privacy-reviewed source exists.
When comparing stores, account for opening hours, store closures, channel eligibility, and population differences.

Maintain a data dictionary with field owner, definition, unit, refresh interval, known gaps, and version.
The metric owner signs off on changes to definitions.
A dashboard showing a current rate without its denominator or coverage can hide a material data outage.
Publish missingness and eligibility along with the result.

## 9. Industry practice and case evidence

### Category management as a process

The Category Management Association describes category management as a collaborative process that treats categories as strategic business units.
The description supports a role for structured review and cross-functional collaboration.
It does not establish that any particular retailer's role assignment caused a sales increase.
The four roles in this lesson are useful operating archetypes rather than a mandatory industry coding standard.

A supplier can contribute market data, product attributes, consumer research, and execution capacity.
The retailer still needs independent evidence and a clear approval process.
A category captain's view may reflect its portfolio and the data available to it.
The commercial team should test whether recommendations protect relevant needs across brands and price tiers.

### Public evidence on choice

Broniarczyk, Hoyer, and McAlister's research on assortment perceptions examines how consumers perceive variety and the effect of preferred-brand availability and shelf space.
Its setting and measures do not establish a universal tree for coffee or any specific store.
The practical implication is to test perceived coverage and preferred-option availability rather than count SKUs alone.
That implication is a management interpretation of the research, not a reported retailer outcome.

Boatwright and Nunes studied the effect of reducing product assortment in a retail setting.
Their work demonstrates that assortment reductions can have consequences beyond a simple item count.
It does not imply that every reduction improves sales or that every slow item should be removed.
The store, category, substitutes, and execution context determine whether a finding transports.

### Public evidence on stockouts and substitution

Fitzsimons studied consumer response to stockouts, including reactions beyond a simple immediate substitution.
Campo, Gijsbrechts, and Nisol examined consumer responses to out-of-stock situations in a grocery context.
These studies motivate explicitly tracking the outside option and distinguishing product, store, and purchase-timing responses.
They do not provide a transferable fixed transfer rate for the fictional D1 SKU.

A retailer should combine published evidence with its own data and a prospective evaluation.
In particular, availability-driven behavior can vary with brand loyalty, urgency, price, channel, and the alternatives visible at the moment of choice.
A single average transfer rate can conceal a group for whom no acceptable alternative remains.
The fictional 40%, 60%, and 80% scenarios in this lesson are assumptions, not published estimates.

### Practical governance pattern

A common review cycle starts with a strategy and role proposal, then examines shopper, market, category, and retailer evidence.
The team chooses objectives and tactics, obtains approval, implements, and reviews results.
This is a general management pattern, not a claim about a named retailer's proprietary process.
Its value is the link between a named role and specific actions with accountable owners.

A destination category is not automatically entitled to the most space.
Its contribution to store choice must be weighed against its cost and the needs displaced elsewhere.
A convenience category is not automatically the highest-margin category.
Its offer must be appropriate to the mission and satisfy relevant customer and legal requirements.
A seasonal category needs a calendar and exit discipline; it should not be judged only on its weakest off-season week.
A routine category needs dependable service, but should not be protected from evidence-based assortment improvement.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Role inflation | Most categories are called destination. | No prioritization of scarce resources. | Service and space promises exceed capacity. | Limit role assignments by proposition, budget, and review. |
| Label without action | The same scorecard and plan apply to all roles. | The role never affects a decision. | Governance adds reporting but no value. | Require a role-to-action and guardrail mapping. |
| One universal tree | Strong segment contradictions are dismissed. | Heterogeneous needs are forced into one ordering. | Critical needs disappear in some stores. | Build mission or segment variants and test their importance. |
| Retail taxonomy mistaken for choice | Tree matches supplier reporting exactly. | Operational codes substitute for shopper evidence. | Shoppers cannot find acceptable alternatives. | Challenge branches with interviews and substitution observations. |
| Price constraint ignored | Only premium alternative remains after a deletion. | Attribute coverage is counted without affordability. | Value-constrained shoppers leave or defer. | Include price tier and outside option in the need test. |
| Basket causality | Entire basket margin is assigned to each focal SKU. | Co-purchase is treated as an incremental effect. | Deletion or retention economics are overstated. | Use controlled incremental analysis or uncredited scenarios. |
| Availability bias | Slow SKU is frequently out of stock. | Observed sales are censored by supply. | High-demand item appears unimportant. | Join availability and sales; investigate unconstrained demand. |
| False substitution | Category sales hold while trips fall. | Transfer is inferred only among observed purchasers. | Lost households and outside purchases are missed. | Include customer and store outcomes where lawfully measurable. |
| Reversed denominator | Attach rate changes with unexplained sample size. | Analyst divides by attached baskets instead of focal baskets. | Teams misread cross-category relevance. | Publish numerator, denominator, and eligibility. |
| Seasonal timing error | Event display arrives after the peak. | Forecast and replenishment ignore lead time. | Lost sales and excess residual stock. | Set order, readiness, review, and exit dates. |
| Empty shelf | Listing file says active while shelf audits fail. | Range approval is confused with execution. | Customer experiences a missing offer. | Monitor actual presence and replenishment by store. |
| Supplier capture | Recommendations systematically remove rival options. | One supplier controls evidence or decisions. | Biased range and potential legal exposure. | Independent retailer review and information controls. |
| Invalid comparison | Pilot stores receive a simultaneous promotion. | Treatment and promotion effects are confounded. | Rollout based on a misleading lift. | Randomize or document and adjust co-interventions. |
| Margin double count | Handling saving appears in item margin and savings line. | Same cost is credited twice. | Overstated deletion benefit. | Define contribution bridge and reconcile finance inputs. |
| Role drift | Old scorecard persists after strategy changes. | No expiry or ownership for a role. | Resources support an obsolete mission. | Set a review date and escalation trigger. |
| Privacy overreach | Analyst exports named customer histories unnecessarily. | Data collection exceeds the decision need. | Regulatory, security, and trust risks. | Minimize data, restrict access, and involve privacy owners. |

For each warning, distinguish a process symptom from an outcome.
A rising complaint rate may reflect genuine need loss, poor labeling, or a temporary execution issue.
Investigate before assigning a cause.
The mitigation must have an owner and a date, not merely appear in a risk register.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| The defined mission is commercially meaningful. | It connects the role to a customer decision. | Customers buy for unrelated purposes. | Use mission interviews and compare segments. |
| Category boundaries are stable in the review window. | Like-for-like metrics require a common scope. | Items are reclassified during the pilot. | Version the hierarchy and restate both periods. |
| A tree captures relevant choices. | The range needs a tractable representation. | Shoppers use simultaneous filters or multiple paths. | Test alternative trees and represent multiple segments. |
| Essential attributes are accurate. | Coverage depends on product characteristics. | Item descriptions or dietary data are wrong. | Validate against authoritative product information. |
| Eligible availability is observed. | Missing sales may reflect stockouts. | Daily snapshots miss intraday gaps. | Compare shelf audits, inventory events, and transactions. |
| Scenario affected units equal 2,400. | It provides a common deletion denominator. | Baseline sales were censored or demand changes. | Vary affected units and measure availability-adjusted demand. |
| Transferred contribution is USD 2.40 per unit. | It converts transferred units into money. | Shoppers choose different items or pack sizes. | Model transfer by destination item and contribution. |
| The USD 720 saving is incremental. | It belongs in the economic bridge. | It is already included in per-unit costs or not realized. | Finance reconciles cost definitions and verifies labor changes. |
| Transfer rates apply to the same 12-week period. | It makes the scenarios comparable. | Delayed purchases occur after the window. | Extend follow-up and report timing separately. |
| The control represents the no-change counterfactual. | A causal pilot needs a credible comparison. | Differential promotions, trends, or spillovers occur. | Randomize, monitor co-interventions, and assess uncertainty. |
| A 12-store-per-arm trial is informative. | It is the proposed practical design. | Variance or a small effect makes precision inadequate. | Perform a power or precision assessment before launch. |
| Guardrails reflect business priorities. | They turn trade-offs into actionable limits. | Limits are unapproved or inconsistent with capacity. | Obtain documented owner approval and revise before launch. |

The scenario arithmetic is exact under the stated inputs, but the inputs are uncertain.
A precise break-even rate is not a precise estimate of actual transfer.
The role hypothesis is a strategic choice awaiting evidence, not a finding that coffee creates incremental trips.
The tree is a decision aid and does not represent every shopper's psychological sequence.

## 12. Exercises

### Exercise 1 — Assign a role

A chain has a specialist cheese counter in its flagship stores, basic milk in every store, holiday baking displays for six weeks, and a small urgent-need travel-size toiletries rack near checkout.
Choose a primary role for each offer for the stated scope.
**Deliverable:** a four-row table with one reason and one role-specific measure per row.
**Acceptance:** distinguish the role from the physical format and state the denominator for each measure.
**Time:** 12 minutes.

### Exercise 2 — Draw and challenge a tree

Design a shopper tree for plant-based milk in a supermarket.
Consider dietary requirements, intended use, flavor, pack size, price, and brand.
Include a shopper who will not substitute across a dietary constraint and a shopper who is brand-first.
**Deliverable:** a tree, an alternative path, and an outside option.
**Acceptance:** identify at least one hard constraint, one preference, and two pieces of evidence needed to validate the ordering.
**Time:** 20 minutes.

### Exercise 3 — Compute penetration and attach

A store records 18,000 eligible transactions in four weeks.
Coffee appears in 2,700 transactions; 810 of those transactions also contain bakery.
**Deliverable:** calculate coffee trip penetration and bakery attach rate, with formulas.
**Acceptance:** use the appropriate denominator and state one reason neither metric proves incremental traffic.
**Time:** 10 minutes.

### Exercise 4 — Calculate deletion economics

A candidate SKU sells 1,500 units in 12 weeks at USD 1.80 contribution per unit.
A proposed deletion transfers 55% of affected units to alternatives earning USD 2.00 per unit.
Direct handling savings are USD 300 for the same period and are excluded from unit contribution.
**Deliverable:** calculate the contribution change and the break-even transfer rate.
**Acceptance:** show the full bridge and identify one omitted effect.
**Time:** 15 minutes.

### Exercise 5 — Design a seasonal plan

A retailer plans a six-week baking event in 30 stores with an eight-week supplier lead time.
Last year it had frequent peak-week stockouts and substantial residual specialty decorations.
**Deliverable:** a dated action sequence covering range, ordering, shelf readiness, availability, and exit.
**Acceptance:** include an owner, one leading measure, one lagging measure, and a decision for excess inventory without assuming a supplier return right.
**Time:** 20 minutes.

### Exercise 6 — Critique a destination claim

A manager states: “The premium coffee SKU is in baskets averaging USD 65, so deleting it loses USD 65 per purchase.”
The manager also reports that the SKU was out of stock for one-third of observations.
**Deliverable:** a short critique and an evidence plan.
**Acceptance:** identify the causal error, stockout bias, at least two plausible substitutes or outside outcomes, and a safer decision rule.
**Time:** 15 minutes.

### Exercise 7 — Design an evaluation

A chain wants to replace a unique value-tier item in 40 stores.
A national price promotion begins during the proposed pilot.
**Deliverable:** a one-page evaluation plan with treatment, control, primary metric, timing, guardrail, and interpretation.
**Acceptance:** address promotion confounding, store-level randomization or a justified alternative, implementation checks, and uncertainty.
**Time:** 25 minutes.

### Exercise 8 — Write governance

A supplier acting as category captain recommends removing two competitors' products and offers a sales analysis.
The recommendation would eliminate one dietary need branch.
**Deliverable:** a decision-rights matrix and exception log entry.
**Acceptance:** identify retailer approval authority, independent evidence, confidential-information controls, a service decision, an expiry, and a review trigger.
**Time:** 20 minutes.

## 13. Self-check answers

### Exercise 1

A defensible mapping is specialist cheese to destination in flagship stores, basic milk to routine, holiday baking to seasonal, and urgent travel-size toiletries to convenience.
Other assignments can be defensible if the retailer's proposition and customer evidence differ.
For cheese, a measure might be the share of surveyed cheese-mission shoppers naming the flagship as their preferred store: the denominator is surveyed cheese-mission shoppers.
For milk, use milk-purchase transactions divided by eligible transactions, alongside an in-stock rate.
For holiday baking, use in-stock item-store-day observations divided by eligible event item-store-day observations.
For toiletries, use transactions with a selected toiletries SKU divided by all eligible transactions near the relevant checkout or store, according to the data design.
None of these measures alone proves the assigned role is causally achieved.

### Exercise 2

One possible tree starts with a verified dietary or ingredient constraint, then intended use, then flavor and pack, with price and brand considered within acceptable leaves.
A brand-first shopper may follow a separate path and switch format within the same brand.
The outside option includes no purchase or purchase elsewhere.
A strong answer asks customers what they would reject and studies choices when alternatives are actually available.
It recognizes that an attribute such as low price can be a constraint for one shopper and a preference for another.
A weak answer uses the retailer's item hierarchy as proof of shopper behavior.

### Exercise 3

Coffee trip penetration is 2,700 / 18,000 = **15% of eligible transactions**.
Bakery attach among coffee transactions is 810 / 2,700 = **30% of coffee transactions**.
The first denominator is all eligible transactions; the second is coffee transactions.
A weekly shopping mission can produce both coffee and bakery in the same basket without coffee causing the visit or bakery purchase.
Unpurchased visits are also absent from a transaction denominator.

### Exercise 4

Current contribution is 1,500 × USD 1.80 = **USD 2,700**.
Transferred units are 1,500 × 55% = **825 units**.
Transferred contribution is 825 × USD 2.00 = **USD 1,650**.
The change is USD 1,650 + USD 300 − USD 2,700 = **−USD 750 over 12 weeks**.
Break-even solves 1,500 × rate × USD 2.00 + USD 300 − USD 2,700 = 0.
The rate is USD 2,400 / USD 3,000 = **80% of affected units**.
The calculation excludes, for example, changes in other-category contribution and any value of redeployed shelf space.
Do not assume that either omitted effect is zero in reality.

### Exercise 5

A strong plan starts before the eight-week supplier lead time, with item decisions and order quantities approved before purchase commitments.
It assigns merchandising to set the range, supply planning to order, stores to install displays, and finance to review residual exposure.
A leading measure could be the share of planned event item-store combinations delivered before the launch date.
A lagging measure could be residual inventory units at the agreed exit date or event contribution in USD.
The plan should include a markdown, transfer, or other approved disposition decision without inventing a return agreement.
A weak plan reports total event sales after the event but has no readiness or exit action.

### Exercise 6

The USD 65 is an average basket value, not incremental margin and not proof that the SKU caused the trip.
It would also double count if credited to every product in the same basket.
The one-third stockout rate may suppress observed sales and distort apparent demand.
Shoppers may buy a premium alternative, choose a different format, defer, or shop elsewhere.
A defensible next step is to protect the unique need while measuring availability and testing a credible replacement or controlled change.
Use a scenario range when transfer and trip effects are unknown.

### Exercise 7

Randomize comparable stores to replacement or control within appropriate matched groups if operationally feasible.
Apply the national promotion consistently to both groups, or postpone the pilot if the promotion interacts differently with treatment.
Define the primary outcome as a store-week contribution difference in changes, with a pre-period and a fixed follow-up period.
Check that the replacement was actually listed, delivered, and on shelf.
Specify a protected-branch availability guardrail and a stop owner.
Report an effect estimate with uncertainty and avoid interpreting an underpowered null result as proof of no harm.

### Exercise 8

The retailer category owner approves the range; the supplier provides a documented recommendation and data under controlled access.
Finance validates economics, operations validates service, and legal or compliance reviews information handling as needed.
The exception entry identifies the missing dietary branch, an interim acceptable replacement or a decision not to remove the final option, the accountable owner, expiry, and review date.
Independent shopper evidence and competitor-neutral product data challenge the supplier analysis.
Do not share a competitor's confidential terms with the captain without an authorized legal basis.
A weak answer gives the captain final authority or treats all supplier participation as automatically unlawful.

## 14. Interview questions

1. **What makes a category destination rather than routine?**  
   Strong: ties the role to a scoped retailer proposition and evidence about mission and store choice.  
   Weak: equates destination with the highest sales category.
2. **Can one category have different roles across formats?**  
   Strong: explains why missions, constraints, and customer expectations vary and names governance for exceptions.  
   Weak: assumes every role must be globally identical.
3. **What is the difference between a shopper tree and a retailer decision rule?**  
   Strong: separates a representation of needs and substitutes from the business process for choosing actions.  
   Weak: treats a listing rule as observed customer behavior.
4. **How would you validate the first split of a decision tree?**  
   Strong: triangulates stated needs, observed choices, availability, and alternative segment trees.  
   Weak: uses the internal product hierarchy alone.
5. **What happens when a unique low-volume SKU is deleted?**  
   Strong: considers transfer, outside options, service and uncertainty before recommending a test.  
   Weak: assumes its own sales are the entire effect.
6. **Why is a large associated basket not the item's incremental value?**  
   Strong: explains confounding, other causes of the trip, and double counting.  
   Weak: multiplies average basket revenue by item transactions.
7. **How would you manage a seasonal role?**  
   Strong: connects the event calendar to lead time, readiness, peak service, and residual stock.  
   Weak: optimizes using an annual average without event timing.
8. **How do you compare a deletion with no change?**  
   Strong: builds a contribution bridge, specifies the counterfactual, and varies uncertain transfer.  
   Weak: counts only saved shelf space or only lost item revenue.
9. **What is a category captain's proper role?**  
   Strong: recognizes supplier expertise while retaining retailer decision rights and information controls.  
   Weak: gives the supplier final authority or assumes all collaboration is prohibited.
10. **When would you reverse a pilot?**  
    Strong: cites pre-agreed service or economic guardrails, execution evidence, and a named stop owner.  
    Weak: changes the threshold after seeing the result.

## 15. Key terms

- [Category](#4-definitions-and-vocabulary): a managed set of related shopper needs.
- [Category role](#4-definitions-and-vocabulary): intended contribution to a retailer proposition.
- [Destination](#4-definitions-and-vocabulary): role focused on becoming a preferred mission choice.
- [Routine](#4-definitions-and-vocabulary): role supporting dependable regular shopping.
- [Seasonal](#4-definitions-and-vocabulary): role concentrated around an event or season.
- [Convenience](#4-definitions-and-vocabulary): role serving a quick or supplementary need.
- [Mission](#4-definitions-and-vocabulary): reason for a trip or order.
- [Shopper decision tree](#4-definitions-and-vocabulary): model of how needs organize choice.
- [Branch](#4-definitions-and-vocabulary): alternatives under a decision condition.
- [Leaf](#4-definitions-and-vocabulary): end segment of the tree.
- [Need state](#4-definitions-and-vocabulary): relevant use requirement or preference.
- [Hard constraint](#4-definitions-and-vocabulary): condition that cannot be violated for the decision.
- [Preference](#4-definitions-and-vocabulary): attribute with potential trade-offs.
- [Outside option](#4-definitions-and-vocabulary): no purchase from the offered set.
- [Substitution](#4-definitions-and-vocabulary): choice of an alternative after a change.
- [Transfer rate](#4-definitions-and-vocabulary): affected demand redirected to alternatives.
- [Coverage](#4-definitions-and-vocabulary): service of defined need branches.
- [Availability](#4-definitions-and-vocabulary): actual opportunity to buy at the relevant time.
- [Facing](#4-definitions-and-vocabulary): product-width shelf position.
- [Penetration](#4-definitions-and-vocabulary): defined share of trips or population buying.
- [Attach rate](#4-definitions-and-vocabulary): co-purchase share conditional on a focal basket.
- [Contribution](#4-definitions-and-vocabulary): revenue less specified variable costs.
- [Incrementality](#4-definitions-and-vocabulary): difference from a credible counterfactual.
- [Guardrail](#4-definitions-and-vocabulary): boundary that limits a plan.
- [Category captain](#4-definitions-and-vocabulary): supplier providing category insight.
- [Decision rights](#4-definitions-and-vocabulary): assigned authority over the decision process.
- [Forecast](#4-definitions-and-vocabulary): conditional estimate of a future outcome.
- [Plan](#4-definitions-and-vocabulary): selected actions and resources.
- [Target](#4-definitions-and-vocabulary): desired performance level.
- [Commitment](#4-definitions-and-vocabulary): authorized delivery obligation.

## 16. Sources

1. GS1, [GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/). Identifier allocation guidance; consult the current version for the market.
2. GS1, [Global Data Synchronisation Network](https://www.gs1.org/services/gdsn). Product-data sharing context, not a shopper-choice model.
3. European Union, [Regulation (EU) 2016/679, GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Article 5. Primary legal text; applicability depends on facts.
4. US Federal Trade Commission, [Guide to Antitrust Laws](https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws). General US competition guidance; not legal advice.
5. Category Management Association, [Category Management](https://www.categorymanagement.com/). Industry association and process context; not evidence of a particular intervention's causal outcome.
6. Broniarczyk, S. M., Hoyer, W. D., and McAlister, L. (1998), “Consumers' Perceptions of the Assortment Offered in a Grocery Category: The Impact of Item Reduction,” *Journal of Marketing Research*, 35(2), 166–176. [DOI: 10.2307/3151845](https://doi.org/10.2307/3151845).
7. Boatwright, P., and Nunes, J. C. (2001), “Reducing Assortment: An Attribute-Based Approach,” *Journal of Marketing*, 65(3), 50–63. [DOI: 10.1509/jmkg.65.3.50.18329](https://doi.org/10.1509/jmkg.65.3.50.18329).
8. Fitzsimons, G. J. (2000), “Consumer Response to Stockouts,” *Journal of Consumer Research*, 27(2), 249–266. [DOI: 10.1086/314323](https://doi.org/10.1086/314323).
9. Campo, K., Gijsbrechts, E., and Nisol, P. (2000), “Towards Understanding Consumer Response to Stock-outs,” *Journal of Retailing*, 76(2), 219–242. [DOI: 10.1016/S0022-4359(00)00026-9](https://doi.org/10.1016/S0022-4359(00)00026-9).

Published studies motivate the questions and mechanisms; their estimates should not be transplanted without considering setting and identification.
The retailer, operational thresholds, and numerical scenarios in this lesson are fictional.

## 17. Links

- [Module 17 index](_index.md)
- [Module 17 survey and lab sections](../../../course/17_assortment_category_retail_analytics.md)
- [Assortment decision stack](01_assortment-decision-stack.md)
- [Variety versus choice overload](02_variety-choice-overload.md)
- [Market basket analysis](04_market-basket.md)
- [Space elasticity and planograms](06_space-elasticity.md)
- [Store clustering and localisation](09_store-clustering.md)
- [New item acceptance and cannibalization](10_new-item-cannibalization.md)
