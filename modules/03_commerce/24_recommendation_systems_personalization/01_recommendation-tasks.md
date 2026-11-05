# Recommendation tasks: deciding what to recommend and why

> Module 24 — Recommendation Systems and Personalization  
> Topic 01 of 11  
> Estimated study time: 4 hours  
> Prerequisites: Modules 06, 07, and 17; basic probability and contribution accounting  
> Version 2026.5

## 1. Why this topic exists

A recommendation is a proposed action in a particular decision context, not merely a product with a high score.
A merchant deciding what to show beside a viewed item faces a different problem from a picker replacing an unavailable grocery item.
The former can suggest a related item without promising equivalence.
The latter must protect the customer's original intent and obtain the appropriate acceptance before fulfillment.
A marketer deciding whether to send an offer faces a third problem: the best action may be to send nothing.

These decisions share catalog and customer data, but their success criteria differ.
Related-item discovery can improve navigation.
A substitution can recover an otherwise lost order.
A complement can make the initial product more useful.
A next-best-offer can change a future decision, but only if its incremental benefit exceeds its costs.
Calling all four tasks “people also bought” hides these differences.

A high click-through rate can coexist with customer harm.
A substitute may be clicked because its image looks familiar while containing an allergen or fitting the wrong device.
An attractive add-on may be out of stock at the delivery location.
A discount may induce a purchase that would have happened at full price anyway.
These are not merely modeling errors; they are errors in the business decision definition.

Supply chain constraints make the distinction especially important.
Availability is location- and time-specific.
Inventory shown in a catalog is not necessarily inventory available to promise.
Reservations, safety stock, cut-off times, delivery capacity, and inventory feed latency change what can actually be fulfilled.
A model trained on historical orders may learn what was exposed, not what customers would prefer among all feasible alternatives.

The product manager owns the customer-facing decision and experience.
Merchandising owns assortment and commercial policies.
Operations owns fulfillment feasibility and service commitments.
Legal, privacy, and safety owners decide applicable policy constraints.
Analytics and experimentation teams define evidence, measurement, and uncertainty.
Engineering owns reliable implementation and observability.
No single model score transfers these responsibilities to a data scientist.

This topic gives you a contract for each recommendation task before choosing an algorithm.
You will identify the trigger, candidate, objective, customer constraints, supply constraints, decision owner, baseline, and measurement plan.
You will also calculate a small business case that separates expected contribution from causal incremental value.
The result is a decision specification that can survive a change of model.

## 2. Learning outcomes

After studying this topic you can:

1. **Classify** a recommendation as related discovery, substitution, complement, or next-best-offer from its trigger and customer intent.
   Exercise 1 checks ambiguous placements.
2. **Write** a task contract with an eligible population, action, decision time, success metric, and owner.
   Exercise 2 requires a complete contract.
3. **Apply** hard customer and supply constraints before ranking candidates.
   Exercise 3 checks the eligible set.
4. **Compute** expected contribution and compare a proposed action with an appropriate baseline.
   Exercise 4 checks arithmetic and the no-offer alternative.
5. **Distinguish** a forecast of response from a causal estimate of incremental response.
   Exercise 5 critiques an observational comparison.
6. **Design** an experiment with a primary metric, guardrails, assignment unit, and operational monitoring.
   Exercise 6 asks for an experiment brief.
7. **Identify** privacy, fairness, safety, and fulfillment failure modes and name accountable owners.
   Exercise 7 asks for a response plan.
8. **Evaluate** a recommendation business case under uncertainty and capacity limits.
   Exercise 8 requires a sensitivity analysis.

The outputs are deliberately observable: a reader should be able to inspect your contract, calculation, and decision rule.
A well-defined task is useful even when a simple rule remains the best implementation.
The goal is not to justify personalization in every placement.

## 3. The core mental model

A recommendation system is a constrained decision pipeline inside a larger customer and fulfillment process.

```text
Customer need + trigger + context
              |
              v
     Define task and action
              |
              v
  Candidate sources and identity
              |
              v
  Customer, safety, policy gates -----> no safe candidate -> fallback or abstain
              |
              v
  Supply and promise feasibility -----> stale / unknown -> refresh or abstain
              |
              v
  Rank feasible actions against objective and baseline
              |
              v
  Show, explain, record, and obtain required choice
              |
              v
  Reserve / fulfill / deliver or handle failure
              |
              v
  Measure incremental outcomes and guardrails
              |
              +-----------------> revise policy with decision owners
```

The trigger defines the decision moment.
A product-detail view, an out-of-stock event, a cart addition, and an offer-eligible customer visit are different triggers.
The same pair of products can have different meanings at each trigger.
Coffee and filters may be complements when a machine is in the cart, but not substitutes if the machine is unavailable.
Two flavors of the same drink might be related items, acceptable substitutes, or neither for a particular customer.

Candidate generation answers “what could we consider?”
Eligibility answers “what may we show or offer here?”
Ranking answers “in what order should we present the eligible options?”
A final customer choice or offer acceptance is distinct from model ranking.
Fulfillment confirms whether the promised action could actually be completed.
Each stage has a different failure signal and a different owner.

A hard constraint excludes an option irrespective of its predicted value.
For example, a known incompatibility with the customer's device cannot be traded against margin in a safe substitute policy.
A soft preference affects ordering among eligible options; an acceptable color preference may be soft if the customer has said so.
A policy must say who sets the boundary and how missing information is handled.
An unknown allergen status is not proof of safety.

Objectives should be expressed at the customer or order decision level, not only at the impression level.
An impression metric can diagnose a placement, but it does not establish incremental profit or customer benefit.
In a controlled experiment, compare outcomes for assignment groups over a prespecified horizon, including users who do not click.
Count substitution acceptance, delivered orders, returns, refunds, and customer-service contacts where appropriate.

The no-action alternative belongs in the action set for offers and often for recommendations.
Showing nothing can be better than a misleading item, an unprofitable discount, or a low-confidence substitution.
A default list is an action too: replacing it with a model requires evidence against that actual alternative.
Do not confuse the predicted value of a selected action with the value added by using the system.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Recommendation task | Decision defined by trigger, actor, candidate action, objective, and constraints | Treating all placements as the same prediction problem |
| Related item | Item connected to a viewed or selected item by an interpretable similarity or relationship | Claiming it is interchangeable |
| Substitute | Candidate intended to meet the original need when the requested item is unavailable or rejected | Assuming similarity guarantees customer acceptance |
| Complement | Item whose utility depends partly on use with another item | Equating any co-purchase with compatibility |
| Next-best-offer (NBO) | Choice of a commercial action, potentially including incentive, channel, and timing | Selecting the item with highest purchase probability regardless of cost |
| No offer | Action of not presenting an incentive or message | Omitting it from the comparison |
| Trigger | Event or state that creates the decision opportunity | Treating a batch prediction as a customer decision |
| Context | Facts available and permitted at decision time, such as location and cart | Using information recorded after the decision |
| Candidate | Item or action proposed for possible selection | Calling every retrieved item eligible |
| Retrieval | Selection of a manageable candidate set from a larger universe | Treating retrieved results as the final policy |
| Ranking | Ordering eligible actions using a stated objective or proxy | Assuming a score is a probability or causal effect |
| Re-ranking | Applying additional constraints or preferences to an initial order | Fixing safety by merely lowering an unsafe item's rank |
| Hard constraint | Rule that must be satisfied for an action to be eligible | Treating it as a small score penalty |
| Soft preference | Preference that may alter order among permissible candidates | Overriding a stated customer exclusion |
| Eligibility | Result of applying customer, policy, and operational gates | Equating eligibility with availability alone |
| Compatibility | Evidence that an item works with a specified product or use | Inferring fit from co-purchase alone |
| Availability | Quantity or status reported for an item at a location and time | Treating an old on-hand count as a promise |
| Available to promise (ATP) | Policy-defined quantity that can be committed for a specific fulfillment context | Assuming a universal formula or a physical count |
| Reservation | Temporary or confirmed allocation of stock to an order under an operational rule | Assuming recommendation exposure reserves inventory |
| Promise | Customer-facing commitment to a delivery or pickup outcome | Treating a statistical forecast as a commitment |
| Forecast | Estimate of a future quantity with uncertainty | Treating it as an approved plan |
| Plan | Chosen operational course based on forecasts and constraints | Treating it as a guaranteed outcome |
| Target | Desired performance level | Treating an aspirational target as observed performance |
| Impression | A recommendation actually rendered or otherwise exposed under a defined logging rule | Counting candidates returned by an API as views |
| Click-through rate (CTR) | Clicks divided by eligible recorded impressions for a specified placement and period | Using CTR as proof of incremental profit |
| Conversion | Defined desired action divided by a stated denominator and time window | Changing the denominator between variants |
| Contribution | Revenue less specified variable costs, in currency per order or customer | Calling gross sales profit |
| Incrementality | Difference in an outcome caused by an action relative to its counterfactual | Treating predicted purchase probability as uplift |
| Uplift | Difference between potential outcomes under two actions for a population or modeled segment | Assuming it is observed for an individual |
| Counterfactual | Outcome that would have occurred under an alternative action for the same decision | Claiming both outcomes are observed for one customer |
| Guardrail | Metric or constraint that bounds unacceptable side effects | Replacing a safety rule with a dashboard alone |
| Fallback | Predefined safe behavior when normal ranking cannot operate | Showing an arbitrary item when eligibility fails |
| Abstention | Deliberate choice not to recommend or offer | Counting it automatically as an error |
| Exposure bias | Dependence of observed feedback on what was shown and where | Treating unexposed items as rejected |
| Positivity | Relevant actions have nonzero probability in contexts where their effects are estimated | Estimating counterfactual outcomes for never-offered actions without assumptions |
| Attribution window | Prespecified period in which an outcome is attributed for measurement | Choosing a window after seeing results |
| Cannibalization | Sales or contribution shifted from another item, channel, or period rather than created | Counting every recommended-item sale as incremental |
| Incremental net value | Change in contribution less incremental implementation and operating costs | Subtracting a discount twice |
| Decision owner | Person accountable for accepting the decision and its consequences | Assuming the model developer owns commercial policy |

## 5. The mechanics, step by step

### Step 1: State the decision and its timing

**Input:** placement, operational trigger, customer journey, and responsible business team.

**Action:** write one sentence of the form: “When [trigger], choose [action] for [eligible population] before [deadline] to improve [outcome], subject to [constraints].”
Record whether the system recommends to a customer, a service agent, a picker, or a marketer.
The actor matters because a picker may have a narrow fulfillment deadline while a browsing customer can compare alternatives.

**Output:** one decision sentence and a named product or operations owner.

**Branch:** if no one can state what decision changes, stop before building a ranking model.
If two decisions have different triggers or consequences, write separate contracts even if they share catalog features.

A placement is not itself an objective.
“Personalize the home page” names a surface.
“Help a returning customer discover an available product relevant to a stated interest” identifies a customer job.
“Automatically substitute a grocery item” additionally requires acceptance, safety, and fulfillment rules.
The last example can be unacceptable even if its predicted click probability is high.

### Step 2: Classify the task

**Input:** decision sentence and original customer intent.

**Action:** select the primary task by the role of the proposed action.
Use related-item discovery when the relationship helps browsing but no equivalence is asserted.
Use substitution when the proposed item stands in for a requested item.
Use complement when an additional item is useful with a selected item.
Use next-best-offer when choosing whether and how to make a commercial intervention.

**Output:** task label, reason, and boundaries.

**Branch:** a surface can support multiple tasks, but label each slot and measure it accordingly.
A “similar items” panel may include substitutes only if the system has evidence and permission to make that claim.
A next-best-offer can involve a complement, but the offer decision still needs an incremental-value comparison.

| Task | Trigger | Typical action | Primary customer question | Critical error |
|---|---|---|---|---|
| Related | Product view | Show similar or connected items | What else might interest me? | Misrepresenting similarity as interchangeability |
| Substitute | Requested item unavailable | Offer an acceptable replacement | Will this meet the same need? | Wrong specification or unapproved replacement |
| Complement | Item added or purchased | Suggest compatible accessory | What else makes this work? | Recommending an incompatible or unnecessary item |
| NBO | Eligible contact or visit | Choose offer or no offer | Is this relevant and worthwhile? | Discounting a purchase that would happen anyway |

### Step 3: Define the unit, horizon, and population

**Input:** task, customer journey, and source-system identifiers.

**Action:** choose the unit of decision and the unit of measurement.
A decision may be a product-detail-page view; the experiment may randomize at customer level to reduce cross-variant contamination.
A substitution decision may be an unavailable order line, while customer-level outcomes require clustering or appropriate aggregation.
State the geographic market, channels, fulfillment method, eligibility date, and observation period.

**Output:** population definition, decision unit, randomization unit, and measurement window.

**Branch:** if identity cannot be joined lawfully and reliably, use an anonymous session or contextual policy rather than inventing person-level history.
If customers encounter multiple variants, define a stable assignment or explicitly analyze the interference.
If one order contains multiple unavailable lines, do not treat the resulting customer experiences as independent observations without justification.

The denominator must include everyone eligible for the decision, not only customers who clicked.
For a substitution experiment, a sensible primary denominator can be eligible unavailable order lines, with delivered contribution measured per eligible line.
For an offer, contribution per eligible customer over a fixed period includes nonresponders and no-offer customers.
Exposure rates and clicks are useful secondary diagnostics.
Changing denominators after assignment can introduce selection bias.

### Step 4: Establish customer intent and permissions

**Input:** explicit selection, account settings where permitted, session context, and customer-facing policy.

**Action:** record what is known and what is unknown.
For substitution, capture customer exclusions and replacement preference where available.
Separate a known restriction from an inferred preference.
An explicit “do not substitute” instruction is not a low preference score.
A compatibility requirement for a medical device or infant product may require a specialized, approved process rather than a general recommender.

**Output:** permitted context, mandatory exclusions, and treatment of unknowns.

**Branch:** if the critical attribute is missing or stale, refresh it, route to human review, or abstain according to policy.
Do not infer a sensitive attribute from purchase behavior simply because it is predictive.
The privacy owner determines whether the purpose, data, retention, and permissions are appropriate for the applicable jurisdiction.

A customer may prefer a cheaper replacement, require the same brand, prohibit a particular ingredient, or decline all substitutions.
Some preferences can be inferred only weakly from behavior.
Ask for explicit choices when a wrong inference has material consequences.
Record the version and time of the preference so downstream decisions can be reconstructed.
A preference can change after exposure but before fulfillment; define the authoritative state at commitment.

### Step 5: Define the candidate universe

**Input:** catalog, task-specific relationships, market assortment, and approved action definitions.

**Action:** identify sources of candidates and a fallback source.
Related items may come from category and content similarity.
Substitutes may come from curated product families and specification rules.
Complements require a compatible relationship, not merely a common order.
Offers include a product or service, incentive amount, channel, eligibility conditions, and expiration.

**Output:** a versioned candidate universe and source lineage for each candidate.

**Branch:** if candidate generation produces no relevant results, invoke the task-specific fallback.
If a new item has no historical interactions, use approved content or merchandising rules rather than treating missing history as evidence of low quality.

Do not confuse candidate coverage with eligibility.
A candidate source can nominate a product that subsequently fails an allergen, safety, market, or supply check.
Log why it failed in a privacy-appropriate way.
A curated list can be valuable, but its maintenance owner and refresh cadence must be named.

### Step 6: Apply customer and policy gates

**Input:** candidates, product attributes, customer instructions, and applicable policy.

**Action:** apply hard rules before optimizing a commercial objective.
Examples include product compatibility, explicit customer exclusions, age restrictions where applicable, prohibited claims, and channel permissions.
For substitution, compare the requested and proposed item on the attributes material to the need: function, size, formulation, dietary attributes, and acceptable price change.
For a complement, verify that the accessory fits the exact product variant.
For an offer, check the promotion's approved eligibility and contact policy.

**Output:** eligible candidates and reason codes for excluded candidates.

**Branch:** distinguish `false` from `unknown` in attribute checks.
A known incompatible candidate fails.
An unknown candidate follows the task's missing-data policy; where safety is involved, that often means abstention or review rather than a permissive default.
Policy owners decide which requirements are hard, not the ranking team alone.

A model score cannot turn an unverified compatibility claim into proof.
A disclaimer does not automatically repair a misleading label or an unsafe recommendation.
Explain material differences to the customer in a way that supports a choice.
For example, a replacement pack with a different unit size requires a clear quantity and price comparison.

### Step 7: Apply supply and promise gates

**Input:** location-specific inventory and reservations, fulfillment method, delivery or pickup slot, catalog sellability, and service rules.

**Action:** evaluate whether each candidate can be offered for the actual destination and time.
Use the organization's ATP calculation rather than inventing a universal inventory formula.
A simple illustrative policy could subtract existing reservations and a protected buffer from on-hand units, but real systems may include inbound supply, allocation, and other rules.
Check stock status freshness and whether fulfillment capacity exists for the promised window.

**Output:** feasible candidate set with inventory snapshot time, location, and promise context.

**Branch:** if stock is unknown or the feed is too old for policy, refresh or abstain.
If stock disappears between display and commitment, revalidate and offer a recovery path.
If a recommendation is only an unreserved suggestion, do not describe it as guaranteed availability.

A positive ATP is not always a successful delivery.
It may be based on inaccurate counts or be consumed by another customer before checkout.
A reservation can reduce that risk, but it consumes scarce supply and has an expiration and release policy.
For substitution, the operational commitment may occur after the customer placed the original order.
A customer-facing promise must follow the organization's actual commitment and confirmation rules.

### Step 8: Choose the objective and baseline

**Input:** business outcome, customer job, candidate set, and current policy.

**Action:** state a primary objective at a defined horizon and the baseline action.
For an offer, expected incremental contribution relative to no offer is generally more decision-relevant than gross revenue from recipients.
For a substitute, delivered acceptable-order contribution may be useful, subject to safety and customer acceptance.
For a related-item panel, a primary outcome could be contribution per eligible session rather than clicks per impression.

**Output:** metric definition, numerator, denominator, unit, time window, and baseline.

**Branch:** if the objective conflicts with hard policy, the hard policy wins.
If expected value is negative or uncertain beyond the decision maker's risk tolerance, retain the baseline or abstain.
If a short-term revenue objective causes returns or customer dissatisfaction, include those outcomes in the horizon or guardrails.

An objective is not a permission to use any available data.
Nor is it necessarily a single model score.
An organization can first enforce non-negotiable constraints, then maximize expected contribution, while monitoring fairness, cancellations, and experience.
A product owner must resolve the commercial trade-off; analytics should make it explicit rather than hiding it in an arbitrary weight.

### Step 9: Define cost accounting and causality

**Input:** action-level revenue, variable cost, discount, fulfillment cost, expected returns, and baseline outcomes.

**Action:** define a contribution equation and a separate causal comparison.
For a purchased item, contribution may be net revenue after discount less product cost, payment and fulfillment variable cost, and expected return cost.
State whether tax, shipping revenue, fixed overhead, and implementation costs are included.
Do not subtract the discount again if revenue is already net of discount.

Expected contribution under an offer is the sum over outcomes of probability times contribution, including nonresponse and other downstream effects where material.
Incremental contribution is expected contribution under the offer minus expected contribution under the counterfactual policy.
Implementation and operating costs are then deducted on a compatible time and population basis.

**Output:** calculation sheet with units and explicit assumptions.

**Branch:** if only historical correlations are available, label the result a forecast or scenario, not a causal estimate.
If the same customer can purchase a full-price item without the offer, include that possibility in the baseline.
If a recommended item displaces another product or shifts timing, measure the total customer or order contribution over a suitable horizon.

### Step 10: Specify experience and decision rights

**Input:** chosen task, eligible actions, and customer and operational workflows.

**Action:** specify what is displayed, what is claimed, and what the customer can accept, reject, or change.
For a substitution, show material differences and follow the customer-selected substitution rules.
For a complement, distinguish required components from optional accessories.
For an offer, show conditions and expiry consistently with the approved promotion.

**Output:** interaction design, approval record, and escalation route.

**Branch:** if the product is safety critical or the system is uncertain, use the approved human workflow or abstain.
If the customer rejects a substitute, do not silently replace it with another item outside the agreed policy.

| Decision | Accountable owner | Important contributors |
|---|---|---|
| Customer job, placement, and fallback | Product manager | Design, service, analytics |
| Assortment, price, and promotion terms | Merchandising or commercial owner | Finance, legal, product |
| Substitution and fulfillment policy | Operations owner | Customer service, product, safety |
| Product compatibility and attributes | Catalog or category owner | Supplier, quality, engineering |
| Data use and retention | Privacy/legal owner | Security, product, data governance |
| Experiment analysis | Experimentation or analytics owner | Finance, product, operations |
| Serving reliability and logs | Engineering owner | Operations, data platform |
| Launch and rollback decision | Named business owner | All affected control owners |

Names and job titles vary by organization.
The important feature is a single accountable decision maker per decision and an explicit escalation path.
Model developers can recommend thresholds but should not unilaterally approve legal, safety, or commercial policy.

### Step 11: Instrument and evaluate

**Input:** task contract, event schema, and evaluation plan.

**Action:** log eligibility, candidate-set version, assignment, rendered impressions, position, customer response, subsequent fulfillment, and relevant policy versions.
Use pseudonymous identifiers and retention controls appropriate to the approved purpose.
Record when inventory was observed and whether a reservation was made.
Separate an item returned by a service from an item actually rendered to the customer.

**Output:** metric specification, event definitions, audit trail, and monitoring dashboard requirements.

**Branch:** if rendered impressions are not observable, do not label service responses as impressions.
If outcomes can arrive after the initial measurement window, prespecify a maturation or follow-up period.
If assignment or logging fails, investigate before interpreting differences.

Use an offline replay to catch obvious errors, but recognize that historical outcomes were generated under the historical exposure policy.
A held-out click metric does not establish the effect of a new recommendation on sales or customer welfare.
Randomized assignment can estimate an average policy effect when the experiment is properly designed and implemented.
Examine delivery failures, cancellations, refunds, and customer complaints as well as primary outcomes.

### Step 12: Decide, monitor, and retire

**Input:** experiment estimate, uncertainty, guardrails, capacity, and operating costs.

**Action:** compare the estimated incremental value and its uncertainty with the decision threshold agreed before launch.
Review whether effects differ across fulfillment locations, new and returning customers, and relevant customer groups where lawful and statistically supportable.
Check that no hard constraint was violated even if average economics are positive.
Name a rollback owner, threshold, and response time.

**Output:** launch, revise, or stop decision with a dated record.

**Branch:** if evidence is inconclusive, the owner can retain the current policy, gather more evidence, or make a documented risk-based decision.
If a supplier changes specifications, revalidate compatibility and task rules.
If inventory feed latency breaches the agreed limit, use the safe fallback rather than continuing on stale data.
If a promotion expires, remove the action from the eligible set.

Monitor both model quality and business process health.
A sudden fall in eligible candidates can indicate a catalog defect, not a change in taste.
A rise in accepted substitutions followed by a rise in refunds can signal that the proposed replacements looked acceptable but did not work in practice.
A promotion that increases response while decreasing total contribution may be cannibalizing full-price orders.
The task contract determines which observation matters and who acts.

### Task-specific decision walkthroughs

The following short walkthroughs show how the same procedure changes with the customer job.
They are decision designs, not claims about particular companies.
The named values and rules are illustrative and must be approved for the actual business.

#### Related items on a product page

Suppose a customer views a three-person backpacking tent.
The placement is a panel labeled “Explore similar tents.”
The trigger is a rendered product-detail page, not an out-of-stock event.
The candidate is another tent that can be sold in the customer's market.
The intended benefit is easier discovery of comparable choices.
The task does not promise that all shown tents have identical capacity, weight, or weather protection.

Start with category and product attributes to retrieve possible items.
Exclude recalled, unsellable, and market-restricted items according to policy.
Evaluate inventory in the customer's chosen channel when the placement claims local availability.
If the channel is not yet known, avoid presenting a local availability promise.
Show material differences such as stated capacity and packed weight.
Do not describe a tent as an exact replacement unless the claim has separate support.

The product owner can use contribution per eligible product-page session as a primary outcome.
The analyst must define whether the session ends at purchase, after a fixed idle period, or at a fixed elapsed time.
The denominator includes sessions where the panel does not attract a click.
A click-through increase can help diagnose discovery, but it is not the causal business objective by itself.
Compare against the existing curated list or no panel, whichever is the actual policy.

Monitor product-page latency, out-of-stock clicks, and returns.
Check that the panel does not conceal the customer's originally viewed item.
If no candidate passes the rules, omit the panel or show a clearly labeled approved fallback.
Merchandising owns product claims; product owns the experience; engineering owns rendering and logging.

#### Substitution during picking

Suppose a shopper's order specifies a particular 1 L bottle of unsweetened soy drink.
The picker discovers that the store has no sellable unit at the time of picking.
The customer's order settings permit a replacement but exclude added sugar.
The decision must be made before the delivery route cutoff.
The actor might be a picker using a recommendation, not the end customer directly.

First check the current order instruction and the applicable substitution policy.
Next identify candidates in the same functional product family.
Then verify formulation, size, packaging, customer exclusions, price rules, and any other material requirements.
An oat drink may be related in the catalog but is not automatically an authorized soy-drink replacement.
A sweetened soy drink fails the stated customer restriction even if its package looks almost identical.
If the ingredient information is unknown, follow the approved missing-data rule rather than assuming no sugar.

Check availability in the actual picking location and delivery slot.
A unit available in a different store cannot automatically satisfy this order.
Recheck or reserve according to the fulfillment system's policy before confirmation.
Show material differences to the customer when the experience and order terms require it.
The unfilled line remains the fallback when no approved candidate exists.

The primary measure can be delivered contribution per eligible unavailable line.
Customer rejection, refunds, wrong-item complaints, and explicit policy violations are separate guardrails.
Operations approves the time-sensitive workflow, while the catalog owner maintains formulation data.
Do not infer a causal improvement from the fact that accepted substitutions are more profitable than rejected ones.
Acceptance is an outcome selected after the decision, not a randomized baseline.

#### Complement for a selected product

Suppose a customer adds a cordless drill to a cart and sees suggested batteries.
The decision is whether to suggest an additional item, not whether to replace the drill.
The customer needs to know whether the drill already includes a battery and whether the suggested battery fits the exact model.
The product listing and compatibility relationship must therefore be checked at variant level.

Candidate generation can use manufacturer part relationships and approved catalog mappings.
A purchase association can nominate a candidate for review but is insufficient proof of fit.
Exclude batteries incompatible with the drill's connector, voltage, or approved product family.
Where a manufacturer changes a model, version the compatibility evidence and its effective date.
If the mapping is missing, do not say that an accessory fits merely because it belongs to the same brand.

The display should distinguish “required to operate” from “optional spare.”
If the cart already contains a compatible battery, a duplicate recommendation may be unhelpful.
If a battery is part of the drill bundle, avoid implying that the customer must buy it separately.
Availability and shipping restrictions must be evaluated for the selected fulfillment method.
The product and safety owners approve the wording and compatibility requirements.

Measure incremental order contribution per eligible cart, not just accessory attachment among customers who clicked.
Monitor accessory returns, compatibility contacts, and cancellations.
Compare with the actual existing accessory policy.
If the accessory generates contribution but also drives avoidable returns, a short measurement window can produce the wrong launch decision.

#### Next-best-offer during a customer visit

Suppose a retailer can show either a $5 coupon or no coupon to an eligible customer.
The trigger is a visit where the approved promotion is active and channel permissions are satisfied.
The system chooses an action before observing whether that customer will purchase.
The offer definition includes the discount, minimum basket value, expiration, product exclusions, and redemption channel.
No offer is an explicit candidate rather than a failure to personalize.

Estimate the outcome under each permitted action on the same eligible-customer basis.
Include the chance of purchase without a coupon, the discounted order's contribution, and operational costs.
Consider whether the coupon changes other purchases, accelerates a planned purchase, or creates a later return.
A customer with a high purchase probability may have low offer uplift if they would buy anyway.
A customer with uncertain response may still be worth offering a coupon if the incremental expected contribution is positive and the estimate is reliable.

Apply contact-frequency limits and approved promotion rules before selecting an action.
If predicted net incrementality does not justify the offer, choose no offer.
If the estimate is too uncertain for the policy's risk tolerance, retain the current policy or use an approved experiment.
The commercial owner owns the discount budget and promotion terms.
Privacy and channel owners review data use and contact permissions.
Analytics owns the estimation plan, not unilateral approval of a campaign.

Track total contribution per assigned eligible customer over the agreed horizon.
Measure redemption as a diagnostic, not as the denominator for causal value.
Monitor unsubscribes, complaints, discount spend, and effects on customers who do not redeem.
If inventory is scarce, ensure the promotion does not create a promise the supply plan cannot support.
A coupon that is economically attractive on paper may be operationally infeasible for a constrained item.

### Decision record: minimum fields and review questions

Use the following record before implementation and update it when the policy changes.
Each field has an owner and a review question; an empty answer is a decision gap.

| Field | Review question | Suggested owner |
|---|---|---|
| Trigger and deadline | What event opens the decision, and when is it too late to act? | Product or operations |
| Customer job | What need does this action serve, and what is not being promised? | Product |
| Decision actor | Is the action shown to a customer, a picker, or an agent? | Operations |
| Candidate definition | What exactly constitutes an item or offer, including variant and terms? | Merchandising |
| Data as-of time | What information was available and permitted before the decision? | Data governance |
| Customer gates | Which restrictions and instructions must never be overridden? | Product and safety |
| Missing-data policy | Which unknown fields cause review or abstention? | Catalog and safety |
| Supply gate | Which location, slot, inventory age, and reservation rules apply? | Fulfillment |
| Action and baseline | What does the system select and what would happen otherwise? | Product and commercial |
| Objective | What contribution or customer outcome is measured per eligible unit? | Finance and analytics |
| Guardrails | Which side effects make a positive average result unacceptable? | Product and control owners |
| Measurement | How are assignment, exposure, outcome, and late events joined? | Experimentation |
| Fallback | What happens on empty candidates, stale data, and service failure? | Engineering and operations |
| Review and rollback | Who can stop the policy and how quickly must they act? | Named launch owner |

Review the record with the teams that will execute it, not only with the modeling team.
A substitution policy that cannot be executed before the picking cutoff is not operationally complete.
An offer policy without a no-offer outcome and cost definition is not an incremental business case.
A related-items panel without a clear label can unintentionally imply equivalence.
A complement policy without evidence of compatibility can transfer the work and risk to the customer.

## 6. Worked example

### Case and decision

A grocery retailer receives an online order for a 750 g bag of plain oats.
At picking time the requested item is unavailable at the store serving the customer's selected delivery slot.
The customer has allowed substitutions and requested no added sugar.
The decision is whether to offer an available replacement or leave the line unfilled.
The operations owner approves the substitution policy; merchandising owns the product attributes and price data.
The customer retains the choice described by the retailer's order settings.

The following numbers are illustrative, not evidence about a real retailer.
Currency is US dollars (USD).
Quantities are per eligible unavailable order line unless otherwise stated.
The delivery slot is the same afternoon.
The stated delivery promise is the retailer's existing commitment, not a forecast produced by the recommender.

### Candidate data

| Candidate | Product and size | Added sugar | Price | Product cost | Fulfillment cost | Local ATP | Status |
|---|---|---:|---:|---:|---:|---:|---|
| A | Plain oats, 700 g | No | $3.70 | $2.05 | $0.35 | 4 units | Candidate |
| B | Sweetened oats, 750 g | Yes | $3.40 | $1.85 | $0.35 | 9 units | Excluded by customer requirement |
| C | Plain oats, 1,000 g | No | $4.60 | $2.55 | $0.40 | 0 units | Excluded by supply |
| D | Plain oats, 800 g | Unknown | $3.95 | $2.15 | $0.35 | 5 units | Held until attribute verified |

ATP is the store's illustrative available-to-promise output after its reservations and buffer policy.
It is a snapshot, not a reservation.
An unknown added-sugar field for D does not establish that the product satisfies the customer's restriction.
B is not eligible even though it has the most units.
C is not currently feasible even though it is plain.
A is the only candidate remaining under this contract.
A is smaller than the requested 750 g; the customer must see the size difference under the assumed policy.

Suppose A's expected refund and customer-service variable cost, averaged over all accepted replacements, is $0.15 per accepted replacement.
Its contribution conditional on acceptance is:

```text
$3.70 revenue - $2.05 product cost - $0.35 fulfillment - $0.15 expected post-order cost
= $1.15 per accepted replacement
```

Assume the customer accepts A with forecast probability 0.64 per offered line.
When A is rejected, the line is not fulfilled and contributes $0 in this simplified line-level accounting.
The expected contribution under offering A is therefore:

```text
0.64 accepted/offer × $1.15/accepted + 0.36 rejected/offer × $0/rejected
= $0.736 per offer
```

The 0.64 is an illustrative forecast, not a commitment and not a causal estimate.
The probability applies to the defined customer and operational context; it is not a universal property of the item.
The model or analyst must provide evidence and uncertainty for this number in a real decision.

### Compare with the real baseline

Suppose the current policy is a manually selected, permitted replacement with expected contribution of $0.55 per eligible line.
That baseline includes its own acceptance, costs, and unfilled outcomes.
The expected difference in the scenario is:

```text
$0.736 per eligible line - $0.55 per eligible line = $0.186 per eligible line
```

This is a scenario difference between two forecasts.
It is not yet an experimentally established incremental effect.
A comparison with an empty line would overstate the proposed system's value if manual substitution is the actual alternative.
If the manual policy contains unacceptable products, fix its compliance separately; do not make unsafe treatment a required control.

Assume 20,000 eligible unavailable lines per month and $2,400 per month in incremental operating and amortized implementation cost.
If all 20,000 lines can be served with economics like this example, the scenario is:

```text
20,000 lines/month × $0.186/line = $3,720/month gross incremental contribution
$3,720/month - $2,400/month = $1,320/month net incremental value
```

This is a planning scenario, not a forecast of realized profit.
It assumes no capacity bottleneck, no cross-item cannibalization, stable acceptance, and identical value across lines.
Those assumptions are unlikely to hold exactly and must be checked.
A plan to deploy is not a commitment to achieve $1,320 per month.
A target, if management sets one, must be labeled separately from this scenario.

### Sensitivity and supply

At 0.50 acceptance, A's expected contribution is $0.575 per offer.
The difference from the $0.55 baseline is $0.025 per line, or $500 per month before the $2,400 monthly cost.
Net value is then negative $1,900 per month under the same volume assumptions.
At 0.75 acceptance, expected contribution is $0.8625 per offer.
The difference is $0.3125 per line, or $6,250 per month before cost and $3,850 per month after cost.

The break-even acceptance probability, holding the other assumptions fixed, solves:

```text
p × $1.15 - $0.55 = $2,400 / 20,000
p = ($0.55 + $0.12) / $1.15 ≈ 0.583 accepted/offer
```

A point forecast of 0.64 leaves limited room above the approximately 0.583 break-even threshold.
Uncertainty in either acceptance or the baseline matters.
The true choice is not simply “deploy because predicted acceptance exceeds 50%.”

A's 4 ATP units cannot support unlimited simultaneous offers.
If ten pickers see the same unreserved snapshot, more than four may attempt the replacement.
The operations team must choose whether to reserve, revalidate at commitment, cap the offer rate, or use another approved process.
If no feasible replacement remains, the fallback is the customer's authorized unfilled-line workflow, not B or C.

### Experiment and decision

Randomize eligible customers, or another defensible stable unit, to the proposed policy or the existing compliant manual policy.
Use delivered contribution per eligible unavailable line over a prespecified maturation window as the primary outcome.
Include unfilled lines, refunds, and the same cost definitions in both arms.
Guardrails include customer-reported substitution dissatisfaction, refund rate, delivery failures, and explicit-policy violations.
Inspect results by store and major product category, with appropriate uncertainty and multiple-comparison caution.

The experiment estimates the average effect of the complete policy, including how often it has a feasible candidate and how often customers accept.
It does not automatically establish that A is best for every individual.
A statistically uncertain positive point estimate is not proof of the monthly scenario.
The named launch owner compares the estimate and uncertainty against the cost threshold and the agreed risk policy.

## 7. Code, query, or calculation

The following Python 3 example implements the illustrative gates and line-level accounting.
It uses only the standard library and does not contact a live inventory, customer, or promotion system.
Run it as a standalone script.
The `None` attribute is deliberately treated as ineligible because the example's customer restriction requires positive verification.
A real system must replace the fixed data with authorized, current sources and enforce reservation, pricing, and customer-choice policies at the actual commitment point.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Candidate:
    name: str
    added_sugar: Optional[bool]
    atp_units: int
    price: float
    product_cost: float
    fulfillment_cost: float
    expected_post_order_cost: float
    acceptance_probability: float


def exclusion_reasons(item: Candidate) -> list[str]:
    reasons = []
    if item.added_sugar is not False:
        reasons.append("sugar restriction not verified")
    if item.atp_units <= 0:
        reasons.append("no available-to-promise units")
    return reasons


def expected_contribution(item: Candidate) -> float:
    accepted_contribution = (
        item.price
        - item.product_cost
        - item.fulfillment_cost
        - item.expected_post_order_cost
    )
    return item.acceptance_probability * accepted_contribution


items = [
    Candidate("A", False, 4, 3.70, 2.05, 0.35, 0.15, 0.64),
    Candidate("B", True, 9, 3.40, 1.85, 0.35, 0.15, 0.70),
    Candidate("C", False, 0, 4.60, 2.55, 0.40, 0.15, 0.60),
    Candidate("D", None, 5, 3.95, 2.15, 0.35, 0.15, 0.65),
]

eligible = []
for item in items:
    reasons = exclusion_reasons(item)
    if reasons:
        print(f"{item.name}: excluded ({'; '.join(reasons)})")
    else:
        eligible.append(item)
        print(f"{item.name}: eligible")

baseline_per_line = 0.55
monthly_lines = 20_000
monthly_cost = 2_400.0
if eligible:
    best = max(eligible, key=expected_contribution)
    proposed = expected_contribution(best)
    difference = proposed - baseline_per_line
    net = monthly_lines * difference - monthly_cost
    print(f"Selected: {best.name}")
    print(f"Expected contribution: ${proposed:.3f}/line")
    print(f"Scenario difference: ${difference:.3f}/line")
    print(f"Scenario monthly net: ${net:,.2f}/month")
else:
    print("No eligible candidate: use authorized fallback")
```

Expected output:

```text
A: eligible
B: excluded (sugar restriction not verified)
C: excluded (no available-to-promise units)
D: excluded (sugar restriction not verified)
Selected: A
Expected contribution: $0.736/line
Scenario difference: $0.186/line
Scenario monthly net: $1,320.00/month
```

The code intentionally does not simulate customer choice, inventory contention, or causal treatment effects.
It assumes each listed probability is valid, all prices are in USD, and the customer has authorized substitutions.
It also assumes rejected lines have zero contribution and that no other basket-level effect exists.
The arithmetic uses floating-point numbers for an educational scenario; financial production systems should use an approved decimal and rounding policy.
A real policy would include compatibility, location, freshness, slot, preference version, and other gates absent from this short illustration.
The baseline is an input assumption; the program cannot establish whether it is correct.

## 8. Data, standards, and regulatory context

### Data needed for the decision

The minimum data depends on the task.
Related-item discovery needs an item identifier, catalog attributes, placement context, and a way to assess whether an item can be sold in the market.
A substitution additionally needs the original item, material product attributes, customer instructions, local fulfillment context, and current stock or ATP.
A complement requires a maintained compatibility relationship between precise product variants.
An offer needs the action's price and terms, audience eligibility, channel permissions, and a no-offer baseline.

Keep the identity of a product distinct from the identity of a relationship.
A product may have a stable identifier while its ingredients, package size, compatibility, or availability changes.
Store the attribute value, source, effective time, and last verification time.
Do not assume a supplier feed is correct simply because it has a valid identifier.
A product identifier supports matching; it does not certify safety or fitness for a particular use.

A decision log should be sufficient to reconstruct what was eligible and why, subject to approved data minimization and retention rules.
Useful fields include decision timestamp, pseudonymous decision identifier, task type, placement, market, candidate and policy versions, availability timestamp, action, and reason codes.
Record actual rendering separately from retrieval and rank.
Link later acceptance, delivery, refund, and return events using controlled identifiers.
Avoid writing raw sensitive preferences into general application logs.

### Standards and identifiers

GS1 describes the Global Trade Item Number (GTIN) as an identifier for trade items.
A GTIN can help join catalog and supply-chain records where correctly assigned and maintained.
It does not encode all compatibility, dietary, local availability, or customer preference information.
A retailer may also need internal stock-keeping unit identifiers and variant-specific relationship data.
The existence of a GS1 identifier is not a recommendation-quality label.
See the official GS1 source in Section 16.

Inventory and availability semantics are organization-specific.
An on-hand count is not identical to ATP, and ATP is not automatically a delivery promise.
Document the source system, units, location, refresh cadence, allocation rules, and reservation state.
A message with a timestamp is useful only if the consuming service understands the timestamp's meaning and acceptable age.
Document timezone and event-time versus processing-time semantics.

### Privacy and consumer protection

The EU General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, establishes rules for processing personal data where applicable.
Its principles include purpose limitation and data minimization; the appropriate lawful basis and other obligations depend on the processing context.
Article 22 addresses certain solely automated decisions with legal or similarly significant effects, subject to its conditions and exceptions.
Not every product recommendation is automatically such a decision.
A privacy specialist must assess the particular use, jurisdiction, profiling, data, and rights; this educational discussion is not legal advice.

In the United States, Federal Trade Commission (FTC) guidance discusses disclosures and deceptive advertising.
Whether a particular recommendation, paid placement, discount claim, or endorsement requires a disclosure depends on facts and applicable law.
A commercial ranking should not falsely claim neutrality or invent product benefits.
An offer's price, restrictions, and expiration should be presented consistently with the approved terms.
Review the current rules and relevant state and sector requirements with counsel.

Customer instructions can reveal sensitive information even when represented as ordinary product filters.
A dietary exclusion, inferred health need, or household purchase history may deserve stricter handling than generic page views.
Use only data needed for the documented purpose, control access, and implement the approved retention schedule.
Where personal data is not necessary, a session-level or context-only policy may be adequate.
Do not interpret an ability to predict a sensitive attribute as permission to infer or use it.

### Quality and operational controls

Catalog quality should be measured on attributes that are decision critical, not only on total field completeness.
For a substitute, an unknown formulation or size can matter more than a missing marketing description.
A compatibility table should identify the exact variants and evidence source.
A stock feed should expose age and failure state rather than silently returning the last known value indefinitely.
Reconcile a sample of recommendations against actual fulfilled products and customer instructions.

Keep training features separate from post-decision outcomes.
For example, a refund recorded after an order cannot be a feature known when the original recommendation was made.
It can be an outcome or a historical feature for a later decision if available and permitted at that later time.
Point-in-time joins and decision timestamps help avoid leakage.
They do not by themselves remove exposure bias or establish causality.

## 9. Industry practice and case evidence

### Amazon: item-to-item relationships

Linden, Smith, and York describe Amazon.com's item-to-item collaborative filtering in an IEEE Internet Computing article.
The paper discusses constructing similar-item relationships from customer behavior and using them for recommendations.
This is documented evidence that item relationships can support a scalable recommendation experience.
It does not establish that every pair of co-purchased items is a safe substitute or a compatible accessory.
The paper also does not provide evidence for the grocery example in this topic.

**Inference:** an item-to-item signal can be useful for related-item retrieval, while a substitution workflow still needs separate customer and supply gates.
That inference follows from the difference between observed association and a claim of functional equivalence.
It is not a claim that Amazon uses the specific policy proposed here.

### Instacart: substitutions as a customer workflow

Instacart's public help material describes options for managing replacements when an item is unavailable, including customer replacement preferences and communication with a shopper.
The details of the service and interface can change by market and date; consult the live official help page.
The public material is evidence that replacement choice is an explicit customer workflow, not evidence of the acceptance probability or economic outcome in the worked example.

**Inference:** capturing a customer's preference before or during fulfillment can reduce ambiguity in a replacement decision.
Whether a particular design improves satisfaction or contribution requires measurement.
A retailer with different customer agreements or fulfillment operations must not assume the same permissions.

### Netflix: evaluation beyond a single metric

Gomez-Uribe and Hunt describe Netflix's recommender system and its use of multiple algorithms and evaluation approaches.
Their account illustrates that the business placement and customer experience influence how recommendations are evaluated.
The source concerns entertainment, not grocery substitution or product safety.
It should not be treated as proof that the same metric or architecture is appropriate for a physical supply chain.

**Inference:** a recommendation task should be evaluated in the context of the action it enables and the outcome it seeks.
For a physical product, availability, fulfillment, and returns add constraints not present in the same form for a streamed title.

### Experiments and counterfactuals

Kohavi and colleagues describe controlled experiments on the web, including practical issues in online evaluation.
Randomization can make treatment and control comparable in expectation when implemented correctly.
It does not prevent logging failures, interference, inventory competition, or incorrect metric definitions.
In a scarce-stock setting, customers assigned to one policy may change availability for the other policy.
The experiment owner must assess whether customer-level randomization is adequate or whether a store, time, or other cluster design is needed.

An experiment can estimate a policy's average effect without proving the mechanism behind every change.
For example, higher delivered contribution might arise from better substitutes, different prices, or a changed rate of offering a replacement.
Predefined diagnostics help investigate mechanisms, but post-hoc subgroup differences are not automatically causal claims.
Report uncertainty and operational constraints with the estimate.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation and owner |
|---|---|---|---|---|
| Related item presented as substitute | Similarity label used in replacement flow | Association is mistaken for equivalence | Wrong product reaches customer | Task-specific equivalence rules; product and operations |
| Complement incompatible | Accessory returns or fit complaints rise | Co-purchase treated as fit evidence | Refunds and loss of trust | Variant-level compatibility source; catalog owner |
| Explicit customer exclusion ignored | Excluded items appear in decision logs | Preference treated as a soft score | Customer harm and complaints | Hard gate and audit; product and safety |
| Unknown attribute treated as safe | Missing ingredient fields pass rules | Null interpreted as false | Restriction violated | Three-state logic and fallback; data owner |
| Stale inventory | Offer-to-order stock failures rise | Snapshot older than policy allows | Failed promise or repeated replacement | Freshness gate and revalidation; operations |
| Oversold scarce stock | Many offers use same positive ATP | Exposure is mistaken for reservation | Cancellations and picker rework | Reservation or controlled recheck; fulfillment |
| Discount cannibalizes full-price demand | Conversion rises while contribution falls | Baseline purchases receive discount | Negative incrementality | No-offer control and cost accounting; commercial owner |
| Wrong denominator | CTR improves but eligible-session value drops | Non-exposed or non-clicking customers omitted | Biased launch decision | Prespecified eligible-population metric; analytics |
| Exposure bias | Popular exposed products dominate training | Unshown products lack feedback | Narrow assortment and self-reinforcement | Randomized exploration where appropriate; analytics |
| Wrong unit of analysis | Many order lines from one customer treated as independent | Correlated outcomes understate uncertainty | Overconfident result | Match analysis to assignment and clustering; experiment owner |
| Inventory interference | Control stockouts rise when treatment launches | Treatment consumes shared scarce supply | Biased policy comparison | Cluster design or inventory-aware analysis; experiment and operations |
| Leakage | Offline score unusually high | Post-decision outcomes enter features | Online failure | Point-in-time feature controls; data engineering |
| Misleading offer terms | Complaints about price or expiry | Ranking uses outdated promotion definition | Refunds or regulatory risk | Versioned terms and review; merchandising and legal |
| Latency or service outage | Fallback rate and response time spike | Candidate or stock service unavailable | Missing or unsafe experience | Safe fallback, alert, and rollback; engineering |
| Unobserved returns | Short-window value rises then refunds increase | Outcome window closes too early | Overstated contribution | Matured cohort and longer follow-up; finance |
| Excessive contact | Unsubscribes and complaints increase | Each offer optimized independently | Customer fatigue | Contact limits and no-offer option; marketing |
| Unsupported subgroup claims | Large percentage swings in tiny groups | Sampling noise and multiple comparisons | Misallocated interventions | Prespecify groups and intervals; analytics |
| No accountable owner | Defects persist across teams | Policy is split without escalation | Slow remediation | Named owner and response time; product lead |

A high model score is not a mitigation for any of the hard-policy failures above.
A dashboard is also insufficient if no one has authority to stop or correct the policy.
Define the trigger for intervention before launch and rehearse the fallback with the teams that operate it.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or monitor |
|---|---|---|---|
| Customer substitution permission is current | Establishes whether an offer or replacement is allowed | Preference changed after order placement | Compare authoritative preference version at commitment |
| Product attributes are accurate | Supports restriction and compatibility gates | Supplier changes formulation or data is miscoded | Audit against source and sample physical labels |
| ATP is sufficiently fresh | Supports local feasibility | Feed delay, phantom stock, concurrent demand | Monitor snapshot age and stockout after offer |
| Acceptance forecast applies to this context | Converts conditional contribution into expected value | New store, product, season, or policy | Calibrate by cohort and time with uncertainty |
| Rejected line contributes zero in the example | Simplifies line-level arithmetic | Service cost, basket change, or later purchase | Measure rejection-path costs and downstream orders |
| Manual baseline is $0.55 per line | Defines the comparison | Different mix or stale cost data | Estimate baseline over the same eligible population |
| 20,000 lines per month are addressable | Scales the scenario | Eligibility and capacity reduce coverage | Count eligible opportunities and feasible offers |
| Costs are incremental and correctly allocated | Prevents misleading net value | Shared costs or omitted operating labor | Finance review with time horizon and cost ledger |
| No cross-item displacement | Makes line-level value approximate total value | Customers buy another product or cancel basket | Measure total order or customer contribution |
| Experiment assignment remains stable | Supports interpretable comparison | Customer crosses devices or assignments | Audit assignment joins and contamination |
| No material treatment interference | Supports ordinary customer-level comparison | Policies compete for common inventory | Check shared-stock exposure; consider cluster design |
| Outcome window captures returns | Prevents overstated value | Delayed refunds or service contacts | Compare mature cohorts over longer windows |
| Applicable data use is authorized | Enables lawful and trusted processing | Purpose or permission changes | Privacy review and access controls |
| Objective reflects customer experience | Aligns measured success with intended benefit | Short-term contribution hides long-term harm | Monitor complaints, retention, and customer research |

These assumptions are not universal requirements for every task.
They identify what the example needs and what a production owner must replace with local evidence.
An assumption ledger is a living artifact: record the owner, last validation date, and decision triggered by failure.

## 12. Exercises

### Exercise 1: Classify four decisions

**Time:** 15 minutes.  
**Deliverable:** a four-row table with task label, trigger, and one critical constraint.

1. A customer views a tent and sees other tents with similar capacity.
2. A picker cannot find the selected milk and proposes a different milk.
3. A customer adds a printer and sees a cartridge claimed to fit it.
4. A customer is eligible for a $5 coupon but might purchase without it.

**Acceptance criteria:** distinguish related discovery from substitution, identify compatibility for the accessory, and include no offer in the coupon decision.
Explain why an item relationship alone cannot establish the critical constraint.

### Exercise 2: Write a task contract

**Time:** 25 minutes.  
**Deliverable:** a one-page contract for an online grocery substitution decision.

Include trigger, decision actor, eligible population, candidate sources, hard customer gates, supply context, action and fallback, primary metric, and accountable owner.
Specify when the customer can accept or reject the replacement.

**Acceptance criteria:** the contract distinguishes a forecast from a promise, specifies the actual baseline, and treats unknown critical attributes explicitly.
A generic instruction to “maximize sales” is not sufficient.

### Exercise 3: Filter a candidate set

**Time:** 15 minutes.  
**Deliverable:** the eligible candidates and a reason for each exclusion.

A customer permits replacement but excludes nuts.
Item P has verified no nuts and ATP 2 units.
Item Q contains nuts and ATP 7 units.
Item R has unknown nut status and ATP 3 units.
Item S has verified no nuts and ATP 0 units.
Under this exercise's policy, unknown nut status fails the safety gate.

**Acceptance criteria:** apply both the customer gate and the supply gate; do not use a ranking score to rescue a failed candidate.
State whether an ATP snapshot reserves an item.

### Exercise 4: Calculate offer value

**Time:** 20 minutes.  
**Deliverable:** expected contribution under each action, incremental contribution, and the better action under the given assumptions.

An offer produces an order with probability 0.30 and contribution $8 per order after the discount and variable costs.
Without the offer, an order occurs with probability 0.25 and contributes $10 per order.
Assume zero contribution when no order occurs and no other effects.
The costs of running the offer are $0.05 per eligible customer, in addition to the stated order contribution.

**Acceptance criteria:** use the eligible-customer denominator and do not subtract the discount twice.
State whether response probability alone identifies the better action.

### Exercise 5: Critique an observational claim

**Time:** 20 minutes.  
**Deliverable:** a short critique and a replacement measurement proposal.

A marketer says, “Customers who received a coupon purchased 20% more often, so the coupon caused a 20% increase.”
The recipients were chosen because they had recently viewed the product.

**Acceptance criteria:** identify selection and counterfactual problems, distinguish absolute from relative change when reporting a difference, and propose a defensible control.
Do not claim a causal lift from the stated information.

### Exercise 6: Design a controlled experiment

**Time:** 30 minutes.  
**Deliverable:** an experiment brief for a substitute recommendation policy compared with the existing compliant workflow.

Specify eligible population, randomization unit, primary outcome and window, at least three guardrails, shared-inventory risk, and a launch decision owner.

**Acceptance criteria:** include all assigned eligible decisions in the primary analysis, describe how late refunds are handled, and explain how assignment and inventory interference will be checked.
Do not invent a sample size without the required variance and effect assumptions.

### Exercise 7: Respond to a policy failure

**Time:** 20 minutes.  
**Deliverable:** a triage plan for a stale inventory feed and a separate plan for a missing allergen attribute.

For each, identify the customer risk, immediate fallback, diagnostic evidence, owner, and condition for resuming the normal policy.

**Acceptance criteria:** distinguish unknown from safe, avoid promising unreserved stock, and specify a human escalation route.

### Exercise 8: Test the business case

**Time:** 25 minutes.  
**Deliverable:** a sensitivity table for the Section 6 substitution case at acceptance probabilities 0.50, 0.64, and 0.75.

Use $1.15 contribution per accepted replacement, $0.55 per line for the baseline, 20,000 lines per month, and $2,400 monthly incremental costs.
State the break-even acceptance probability and two reasons actual value might differ.

**Acceptance criteria:** calculate monthly net value at all three probabilities and distinguish the resulting scenario from experimentally estimated causality.

## 13. Self-check answers

### Answer 1

| Situation | Task | Critical constraint |
|---|---|---|
| Similar tents | Related-item discovery | Relevant specifications and sellability; do not claim interchangeability |
| Replacement milk | Substitution | Customer restrictions, acceptance rules, and local availability |
| Printer cartridge | Complement | Verified compatibility with the exact printer variant |
| Coupon | NBO | Offer eligibility, incremental economics, and no-offer alternative |

A relationship may be based on a shared category or purchase pattern.
Neither establishes that a milk meets the same dietary need or a cartridge fits a specific printer.

### Answer 2 rubric

A strong contract names the actor and decision point, for example the picker at the out-of-stock event before fulfillment.
It identifies customers who have permitted substitution and names a current-policy baseline.
It excludes known violations and has an explicit rule for unknown safety-critical fields.
It evaluates stock for the actual location and slot, with a recheck or reservation rule.
It describes customer acceptance, the unfilled-line fallback, delivered contribution per eligible line, and an operations owner.
A weak contract uses clicks as the sole outcome, assumes all similar products are safe, or claims that a stock forecast guarantees fulfillment.

### Answer 3

P is eligible under the stated rules.
Q fails the explicit nut exclusion.
R fails because its nut status is unknown and the exercise requires verification.
S fails because ATP is zero.
A positive ATP snapshot does not itself reserve an item; availability must be confirmed under the relevant operating policy.

### Answer 4

Offer expected contribution before operating cost is 0.30 × $8 = $2.40 per eligible customer.
After the $0.05 operating cost it is $2.35 per eligible customer.
No-offer expected contribution is 0.25 × $10 = $2.50 per eligible customer.
The offer's incremental net contribution is $2.35 − $2.50 = −$0.15 per eligible customer.
Under these assumptions, choose no offer.
The offer has the higher response probability but lower expected contribution because it reduces contribution on orders that might occur anyway.
The discount is already included in the $8, so it must not be subtracted again.

### Answer 5

Recipients were selected using recent product views, so they may already have had higher purchase intent.
The observed difference mixes the effect of the offer with differences in who received it.
The counterfactual purchase outcome for those same recipients without the coupon is not observed.
Randomly assign eligible customers to coupon or no coupon and compare purchase and contribution per assigned customer over a fixed window.
If rates were, for example, 24% versus 20%, the absolute difference would be 4 percentage points and the relative difference 20%; the prompt does not provide those rates.

### Answer 6 rubric

A strong design randomizes a stable, appropriate unit within a clearly defined eligible population.
It compares the new policy with the existing compliant policy and retains all assigned decisions in the primary denominator.
It defines delivered contribution and a mature returns window, plus safety violations, dissatisfaction, and fulfillment failure guardrails.
It checks assignment balance, event logging, and whether both groups compete for scarce stock.
It specifies an operations-aware cluster or other design if interference is material.
The owner uses the estimate, uncertainty, cost threshold, and guardrails to decide; a positive click metric alone is insufficient.

### Answer 7 rubric

For a stale feed, stop treating the old snapshot as a current promise, refresh or use a safe fallback, alert operations and engineering, and resume only after the agreed freshness and health checks pass.
For an unknown allergen, exclude or route for approved review rather than treating missing as safe.
Catalog and safety owners verify the authoritative attribute before the candidate returns to the eligible set.
Record affected decisions and follow the organization's customer and incident procedures.

### Answer 8

| Acceptance | Expected contribution per line | Difference from baseline per line | Monthly net |
|---:|---:|---:|---:|
| 0.50 | $0.5750 | $0.0250 | −$1,900 |
| 0.64 | $0.7360 | $0.1860 | $1,320 |
| 0.75 | $0.8625 | $0.3125 | $3,850 |

Break-even is ($0.55 + $2,400 / 20,000) / $1.15 ≈ 0.583 accepted per offer.
Actual value can differ because only some lines have feasible stock, because the baseline varies, or because refunds and basket effects differ.
The table uses assumed probabilities and is not a causal estimate.

## 14. Interview questions

1. **How do related items differ from substitutes?**
   A strong answer distinguishes discovery from replacing an original need and names compatibility and customer permission.
   A weak answer says both are simply similar products.
2. **When is a complement unsafe or misleading?**
   A strong answer gives an exact-variant compatibility example and distinguishes optional from required accessories.
   A weak answer relies exclusively on co-purchase counts.
3. **Why include no offer in NBO?**
   A strong answer compares incremental contribution against purchases that would happen anyway.
   A weak answer assumes every response to a discount is incremental.
4. **What is the difference between ATP and on-hand stock?**
   A strong answer describes policy-defined allocation, location, timing, and reservations.
   A weak answer treats a warehouse count as a delivery guarantee.
5. **How do hard and soft constraints differ?**
   A strong answer excludes violations before ranking and handles unknown attributes explicitly.
   A weak answer applies a small score penalty to an unsafe item.
6. **What is an appropriate primary metric?**
   A strong answer defines population, denominator, horizon, costs, and baseline for the task.
   A weak answer selects CTR without relating it to customer or business outcomes.
7. **Why can an offline improvement fail online?**
   A strong answer mentions historical exposure, position, supply, changed behavior, and causal evaluation.
   A weak answer assumes a higher held-out score proves incremental profit.
8. **How would you handle a stock feed outage?**
   A strong answer specifies freshness limits, fallback, ownership, and a recovery condition.
   A weak answer serves the last value indefinitely without labeling uncertainty.
9. **How would you account for shared inventory in an experiment?**
   A strong answer recognizes interference between arms and considers cluster or inventory-aware design.
   A weak answer assumes independent customers regardless of stock contention.
10. **Who decides whether a recommendation policy launches?**
    A strong answer names a business decision owner and relevant operations, safety, privacy, finance, and engineering approvals.
    A weak answer assigns every policy decision to the model author.

## 15. Key terms

- **Action:** a candidate recommendation, offer, or abstention; see Section 5, Step 1.
- **Assignment unit:** entity randomized in an experiment; see Section 5, Step 3.
- **ATP:** policy-defined available-to-promise quantity; see Section 5, Step 7.
- **Baseline:** actual alternative policy for a comparison; see Section 5, Step 8.
- **Candidate:** item or action considered for a task; see Section 5, Step 5.
- **Cannibalization:** shifted rather than created contribution; see Section 5, Step 9.
- **Compatibility:** verified fit with a particular item or use; see Section 5, Step 6.
- **Complement:** item useful with another item; see Section 5, Step 2.
- **Constraint:** condition on permissible actions; see Section 3.
- **Contribution:** revenue less specified variable costs; see Section 5, Step 9.
- **Counterfactual:** outcome under an alternative action; see Section 5, Step 9.
- **Decision owner:** person accountable for the policy; see Section 5, Step 10.
- **Eligibility:** result of applying the task's gates; see Section 5, Steps 6–7.
- **Exposure bias:** missing or distorted feedback due to what was shown; see Section 10.
- **Fallback:** safe behavior when a normal action is unavailable; see Section 5, Step 12.
- **Forecast:** uncertain estimate rather than a commitment; see Section 6.
- **Guardrail:** metric or rule limiting harmful effects; see Section 5, Step 11.
- **Hard gate:** exclusion that cannot be outweighed by score; see Section 5, Step 6.
- **Impression:** defined actual exposure, not merely a returned candidate; see Section 5, Step 11.
- **Incrementality:** causal difference from a counterfactual; see Section 5, Step 9.
- **Interference:** one assignment affects another unit's outcome; see Section 9.
- **NBO:** choice of the next commercial offer or no offer; see Section 5, Step 2.
- **Plan:** selected course of action, not a guaranteed outcome; see Section 6.
- **Promise:** customer-facing commitment; see Section 5, Step 7.
- **Ranking:** ordering eligible candidates; see Section 3.
- **Related item:** item connected for discovery, not necessarily equivalent; see Section 5, Step 2.
- **Reservation:** allocation under an operational policy; see Section 5, Step 7.
- **Substitution:** replacement of a requested item; see Section 5, Step 2.
- **Target:** desired outcome, distinct from evidence or forecast; see Section 6.
- **Trigger:** event or state that creates a decision; see Section 5, Step 1.

## 16. Sources

Sources are public and checkable; the worked numbers in this topic are illustrative and are not quoted from them.
Access and product interfaces can change; consult the current versions when applying a policy.

1. Greg Linden, Brent Smith, and Jeremy York, “Amazon.com Recommendations: Item-to-Item Collaborative Filtering,” *IEEE Internet Computing*, 2003. [DOI: 10.1109/MIC.2003.1167344](https://doi.org/10.1109/MIC.2003.1167344). Describes item-to-item collaborative filtering, not a guarantee of substitution safety.
2. Carlos A. Gomez-Uribe and Neil Hunt, “The Netflix Recommender System: Algorithms, Business Value, and Innovation,” *ACM Transactions on Management Information Systems*, 2015. [DOI: 10.1145/2843948](https://doi.org/10.1145/2843948). Describes a recommendation system in an entertainment setting.
3. Ron Kohavi, Randal M. Henne, and Dan Sommerfield, “Practical Guide to Controlled Experiments on the Web: Listen to Your Customers not to the HiPPO,” *Proceedings of KDD*, 2007. [DOI: 10.1145/1281192.1281295](https://doi.org/10.1145/1281192.1281295). Practical experimental design and interpretation.
4. GS1, [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin). Official description of the trade-item identification key; the key is not a compatibility certification.
5. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Articles 5 and 22. Binding legal text where applicable; application requires contextual legal analysis.
6. U.S. Federal Trade Commission, [Advertising and Marketing](https://www.ftc.gov/business-guidance/advertising-marketing). Official business guidance and links to applicable rules; not a universal statement that every recommendation is an advertisement.
7. Instacart, [Replacing items](https://www.instacart.com/help/section/360007902831/360039166252). Public customer help describing replacement choices; service features may change.

## 17. Links

- [Module 24 index and complete topic map](_index.md).
- [Module 24 survey and its lab sections](../../../course/24_recommendation_systems_personalization.md).
- [Sibling topic 02: popularity and content-based baselines](_index.md#3-topics), listed in the index; consult its topic file when available.
- [Sibling topic 07: offline and online evaluation](_index.md#3-topics), listed in the index; consult its topic file when available.
- [Sibling topic 09: business constraints and re-ranking](_index.md#3-topics), listed in the index; consult its topic file when available.
- [Sibling topic 10: inventory-aware recommendation and substitution](_index.md#3-topics), listed in the index; consult its topic file when available.
