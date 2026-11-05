# Inventory-Aware Recommendation and Substitution

> Module 24 — Recommendation Systems and Personalization  
> Topic 10 of 11  
> Estimated study time: 4 hours  
> Prerequisites: Topics 01, 07, and 09; basic probability and contribution accounting  
> Version 2026.5

## 1. Why this topic exists

A recommendation is not useful merely because a customer likes it.
It must also be obtainable, appropriate for the customer's purpose, and offered under terms the customer can understand.
An online grocery basket, a maintenance order, and a business-to-business replenishment contract all encounter the same structural problem: the requested item may not be available when fulfillment actually occurs.
The appropriate substitute depends on why the first choice was selected and what the customer permits.

Consider a shopper who selects lactose-free milk for tomorrow's delivery.
A ranking model may put ordinary milk near the top because its brand, package, and price are similar.
That apparent similarity is insufficient: lactose-free status may be a hard requirement.
Even an appropriate alternative may sell out before the picker arrives.
The merchant needs a decision that accounts for both uncertain availability and the customer's choice.

The decision is not simply whether to show an item.
A team must decide whether to reserve it, ask permission, offer a price adjustment, send a picker a suggestion, or leave the order unfilled.
The correct action may be to offer no substitute.
Customer care must be able to reconstruct the decision after a dispute.

The cost of a poor decision has several owners.
The shopper bears lost time, unwanted goods, potential safety risk, and possible loss of trust.
The store bears picking and handling work, credits, returns, and waste.
The merchant bears lost contribution and customer-service cost.
The data team bears the operational burden of uncertain stock, late events, and misleading labels.
A product owner cannot optimize any one of these costs in isolation.

Substitution is particularly sensitive because customer intent is contextual.
A customer who accepts a different brand of flour may reject a different package size when storage space is limited.
A hospital's equivalent part must satisfy clinical and procurement authorization; a ranking score cannot authorize it.
In retail, the shopper's recorded substitution setting may have been made for a previous order or a different item.
That setting is evidence of permission only within its documented scope and validity period.

This unit treats inventory-aware recommendation as a controlled decision system.
It separates candidate discovery, compatibility, consent, stock assessment, ranking, execution, and learning.
It also separates predicted outcomes from the actions and promises made to customers.

### Decisions and accountable owners

| Decision | Accountable owner | Input from | Evidence to retain |
|---|---|---|---|
| Define allowed product equivalence | Category or product owner | Quality, catalog, safety, customer research | Versioned rules and attribute sources |
| Define customer permission | Product and privacy owners | Legal and customer care | Consent language, scope, and event history |
| Decide stock freshness and reservation policy | Fulfillment operations | Inventory engineering and store teams | Feed latency, stock counts, reservation outcomes |
| Choose economic objective and thresholds | Commercial owner | Finance and analytics | Incremental contribution model and sensitivity |
| Deploy and stop the service | Service owner | Operations, security, experiment owner | Release record, alerts, rollback trigger |
| Approve regulated or safety-critical equivalence | Authorized specialist | Quality and legal | Explicit approval and applicable evidence |

The accountable owner can delegate analysis but not outsource a safety decision to a model score.
A service-level agreement is a proposed operational commitment only when an owner accepts it; a forecast of success is not a commitment.

## 2. Learning outcomes

After this unit you can:

1. **Draw** the decision path from a stockout signal to customer-confirmed fulfillment, including the no-substitute branch.
   Exercise 1 asks you to label each decision and owner.
2. **Distinguish** book inventory, available-to-promise inventory, and the probability of fulfillment at the intended pick time.
   Exercise 2 requires a numerical calculation and a warning about dependence.
3. **Apply** hard compatibility and permission rules before ranking eligible alternatives.
   Exercise 3 requires you to reject a high-scoring but incompatible product.
4. **Calculate** expected incremental contribution against an explicit no-substitution counterfactual.
   Exercise 4 requires a per-offer decision and sensitivity check.
5. **Design** an auditable consent and execution state machine with a reservation-expiry and failed-pick path.
   Exercise 5 asks for a state diagram and event fields.
6. **Evaluate** a substitution policy using randomized assignment and customer, operational, and financial guardrails.
   Exercise 6 asks for estimands and analysis choices.
7. **Critique** misleading stock and acceptance metrics, including exposure and selection bias.
   Exercise 7 asks you to interpret a biased comparison.
8. **Propose** a safe rollout and an exception-handling plan for a new category.
   Exercise 8 requires an owner, an intervention, and a stop condition.

Completion means that you can explain why an appealing alternative must sometimes be withheld, not only compute which score is largest.

## 3. The core mental model

A substitution system makes a conditional offer under uncertainty.
It does not turn a prediction into a promise.

```text
Customer requests item A for a location and fulfillment window
                 |
                 v
     Is A reliably fulfillable at pick time?
        | yes                     | no / uncertain
        v                         v
  Retain original        Retrieve candidate alternatives
  and recheck later                |
                                   v
                   Hard compatibility + customer restrictions
                                   |
                       no eligible candidates? ---- yes ---> offer none
                                   |
                                   no
                                   v
                Is customer authorization valid for this action?
                    | no or unclear               | yes
                    v                             v
             Request permission or          Estimate pick-time stock,
             offer no substitute            acceptance, and economics
                    |                             |
                    +--------------+--------------+
                                   v
                        Present ranked offer
                                   |
                  explicit accept / reject / no response
                                   |
                  check or reserve stock near fulfillment
                                   |
            pick succeeds ----> fulfill and notify with actual terms
                  |
                  +---- pick fails / hold expires ----> re-evaluate or refund
                                   |
                                   v
                    Log assignment, exposure, decision,
                    pick, charge, return and feedback
```

The diagram is deliberately not a single prediction arrow.
Stock data can change after ranking; so can the customer's instructions.
Each transition must check the facts relevant at that moment.
A previously displayed offer is not a reservation unless the system actually created one.
A reservation is not successful fulfillment until the item is picked or otherwise secured under the business's process.

### Five separations to preserve

**Similarity versus interchangeability.** Similarity helps retrieve options.
Interchangeability requires the attributes and use conditions that matter for this customer and order.
A lower-ranked eligible product is preferable to a more similar prohibited product.

**Recorded count versus obtainable stock.** A count is an observation made at a time.
The relevant quantity is stock available to this order at its fulfillment location and decision horizon, after existing commitments and uncertainty.
A count of three is not automatically three units that can be promised.

**Offer versus authorization.** Showing an alternative, receiving customer approval, and making a substitution are distinct events.
The organization must know whether an opt-in applies to this product, this order, and the proposed price.
Silence should follow the disclosed default rather than be treated as a newly obtained approval.

**Forecast versus plan versus target versus commitment.** A forecast is a probability or expected outcome, such as a 70% chance of successful pick.
A plan is the proposed action, such as presenting an alternative.
A target is a desired operational result, such as a 90% fill rate.
A commitment is a promise communicated or contractually made to a customer.
A forecast alone cannot authorize a commitment.

**Recommendation versus outcome.** An offered alternative can be rejected, run out, be picked incorrectly, be refunded, or later be returned.
Measure the entire chain rather than counting only clicks or initial acceptance.

### Two clocks and two uncertainty sources

The inventory clock measures the age of a location-specific observation.
The customer clock measures the time between authorization and fulfillment.
Both may expire before a picker acts.
Inventory uncertainty concerns whether the candidate can be picked.
Preference uncertainty concerns whether the candidate is acceptable.
The two probabilities need not be independent: for example, a popular offer can both appeal to shoppers and deplete stock faster.

### A practical policy hierarchy

1. Follow applicable law, contract, safety policy, and customer-specific prohibitions.
2. Apply mandatory product compatibility and price constraints.
3. Verify the scope and timing of permission for the intended action.
4. Assess location- and time-specific availability.
5. Rank eligible, authorized options by the agreed objective.
6. Execute with a fresh check or reservation and communicate the actual result.
7. Log enough evidence for customer care, operations, and evaluation.

No profitability threshold overrides a hard restriction.
Where data is missing, the policy must explicitly decide whether to exclude, ask the customer, or escalate to a qualified person.
A model should not silently invent a missing attribute.

## 4. Definitions and vocabulary

The table defines terms in the order they enter the decision.
The common misuse column gives a reason to keep each distinction explicit.

| Term | Meaning and relevant unit | Common misuse |
|---|---|---|
| Original item | Requested product and quantity on an order line, usually an item identifier and units | Treating any neighboring product as the same request |
| Stockout | Original cannot be fulfilled at the relevant location and time | Assuming a global count describes the local shelf |
| Stockkeeping unit (SKU) | Merchant's identifier for a specific sellable variant | Assuming the SKU itself proves ingredients or equivalence |
| Global Trade Item Number (GTIN) | GS1 identifier for a trade item, when assigned | Treating an identifier as a complete compatibility specification |
| On-hand inventory | Recorded physical units at a location and timestamp | Equating book quantity with verified shelf quantity |
| Committed inventory | Units already allocated or reserved for other demand | Subtracting the same reservation twice |
| Safety stock | Deliberate buffer, in units, against uncertainty or operational needs | Treating a policy buffer as physically missing stock |
| Available to promise (ATP) | Quantity offered after specified commitments and buffers under a defined policy | Assuming all organizations calculate ATP identically |
| Inventory observation age | Decision time minus source observation time, in seconds or minutes | Using ingestion time as if it were physical count time |
| Pick horizon | Time between the decision and planned physical pick, in minutes or hours | Evaluating availability only at page render |
| Fulfillment probability | Calibrated probability of successful pick for the requested quantity and window | Interpreting an uncalibrated model score as a probability |
| Reservation | Temporary or durable allocation that prevents competing use under an explicit contract | Assuming a user-interface offer automatically creates a hold |
| Time to live (TTL) | Duration before a reservation or authorization expires | Failing to release a hold after cancellation |
| Candidate | Product retrieved for consideration, before rules and ranking | Calling every retrieved item a valid substitute |
| Compatibility | Satisfaction of purpose, product, quality, and policy requirements | Measuring compatibility only through clicks |
| Hard constraint | Condition whose violation makes a candidate ineligible | Compensating for an allergy violation with higher margin |
| Soft preference | Desirable attribute that may be traded off within permission | Treating a customer prohibition as a weak preference |
| Customer authorization | Evidence that a specified substitution action is permitted within a scope and time | Treating acceptance of one SKU as unlimited consent |
| Consent event | Recorded permission or withdrawal, with subject, scope, version, and timestamp | Storing only the latest boolean without its history |
| Customer acceptance | Customer's agreement to a presented offer under stated terms | Assuming acceptance guarantees a successful pick |
| Exposure | An offer actually rendered to a customer or delivered to a picker | Treating all generated candidates as viewed |
| Conditional probability | Probability given stated preceding events | Multiplying incompatible or unconditional probabilities |
| Contribution | Revenue less explicitly included variable costs, in currency per order line | Calling gross revenue incremental profit |
| Counterfactual | Outcome that would have occurred under an alternative policy for the same decision | Comparing unrelated shoppers without addressing selection |
| Incremental value | Expected outcome under the proposed policy minus the specified baseline | Crediting the substitution with revenue that would have happened anyway |
| Fill rate | Fulfilled quantity divided by a defined requested quantity or lines | Mixing unit fill and line fill without labeling |
| Calibration | Agreement between predicted probabilities and observed frequencies over a defined population | Equating discrimination or ranking accuracy with calibration |
| Guardrail | Metric or constraint that can limit or stop rollout | Maximizing fill while ignoring complaints or safety |
| Propensity | Probability of an action under a logging or assignment policy | Assuming deterministic logs provide overlap for all actions |
| Idempotency key | Stable identifier that prevents a retried command from applying twice | Treating duplicate delivery as a new customer decision |
| Audit trail | Ordered evidence of inputs, policy versions, decisions, and outcomes | Logging personal data indefinitely without a purpose |

## 5. The mechanics, step by step

### Step 1 — Define the decision and the baseline

**Input:** original item, requested quantity, fulfillment location, planned pick window, customer setting, and current inventory signal.
**Output:** a decision record with a defined trigger and baseline action.

Identify whether the decision occurs on the product page, after checkout, during picking, or after a failed pick.
These moments have different information and permissible actions.
At browsing time a customer can compare alternatives before committing to purchase.
During picking the system may have less time and must respect pre-existing instructions.
A post-pick replacement may require a different approval or refund workflow.

Define the unit of analysis as one affected order line unless a basket-level interaction requires otherwise.
A substitute that changes delivery fees or causes another line to fail needs basket-level accounting.
Define the comparison: for example, no substitute, refund the unavailable line, and retain the rest of the order.
The comparison must include any support or cancellation cost that occurs even without an offer.
Do not subtract costs common to both arms twice.

The owner agrees the policy objective before ranking begins.
For example, maximize expected incremental contribution per eligible stockout while protecting customer choice and safety.
The objective must specify the time horizon for returns and credits.
Long-term retention can be a separate measured outcome rather than an invented per-order currency value.

Branch: if the original item remains available with sufficiently fresh evidence, keep it and recheck at the agreed execution point.
If its availability is uncertain, apply the stated threshold or query the inventory service rather than asserting a confirmed stockout.
If the feed is unavailable or too stale, choose the documented safe fallback.

### Step 2 — Obtain and interpret inventory evidence

**Input:** location-level inventory snapshot, event timestamp, requested quantity, existing allocations, and expected replenishment.
**Output:** a point-in-time availability estimate, with age and confidence.

The serving key must include location, item, fulfillment channel, and relevant time window.
Warehouse inventory is not a substitute for the store's shelf count unless the delivery route and promised timing permit transfer.
Distinguish a source event's occurrence time from the time a downstream service processed it.
Late events can arrive in order of ingestion but not in order of occurrence.
Use sequence numbers or an authoritative ordering contract where available.

One simple bookkeeping estimate is:

```text
ATP = max(0, on_hand - committed - policy_buffer)
```

All quantities must refer to the same item, unit of measure, location, and snapshot.
This expression is an illustrative policy, not a universal definition.
If on-hand already excludes reservations, subtracting committed quantity again is wrong.
If committed units include the current order, adjust the calculation consistently.
Inventory in transit is not available for a pick unless receipt and put-away before the pick are sufficiently supported.

Book ATP does not establish the chance of finding the item.
Shelf errors, shrinkage, damage, concurrent shoppers, delayed scans, and picker substitutions can make physical outcomes differ.
Estimate the probability of fulfilling the requested quantity at the pick horizon using appropriately timestamped historical decisions and outcomes.
Use only information that would have been available at decision time.
Stratify or model by location, category, stock count, quantity, feed age, time of day, and horizon where data support it.

Branch: if a probability is not calibrated for the relevant segment, use a conservative documented fallback, a fresh inventory check, or a manual workflow.
Do not display a precise numerical probability merely because a model produces one.
Revalidate after significant stock-process changes, category expansion, or inventory outages.

### Step 3 — Retrieve a broad candidate set

**Input:** original product, catalog attributes, customer context, location assortment, and retrieval indexes.
**Output:** a set of candidate item identifiers with source and retrieval version.

Retrieve candidates using explicit catalog relationships, category attributes, approved equivalence lists, text search, or collaborative signals.
A retrieval model may favor similar descriptions or previous co-purchases, but co-purchase can indicate complementarity rather than substitution.
A bottle of sauce purchased with pasta is not a replacement for pasta.
Preserve the reason and source of each candidate for inspection.

Filter out products not sold in the applicable location or channel.
Normalize pack size and unit of measure before comparing prices and quantities.
A two-pack of 500 milliliters is not a single 500-milliliter pack merely because the product name contains the same number.
Where product variants share an identifier at one system boundary, resolve to the sellable variant before comparing attributes.

Branch: if retrieval returns no plausible candidate, offer no substitute or route to a human under the product policy.
An empty result is a valid decision, not an error that justifies relaxing a safety restriction.

### Step 4 — Enforce hard compatibility

**Input:** candidates, trusted product attributes, customer restrictions, order purpose, and versioned eligibility rules.
**Output:** eligible candidates plus a reason code for each exclusion.

A category owner defines necessary and prohibited attributes with domain specialists.
Examples include dietary requirements, allergen declarations, compatible equipment models, operating temperature, formulation, age restrictions, and delivery requirements.
The specific constraints depend on the product and jurisdiction.
A model's claim that two products are equivalent does not establish safety or legal compliance.

Treat a missing critical attribute separately from a known safe value.
If allergen data are missing, do not infer absence from a product name or a blank field.
A qualified human may resolve the missing attribute from an authoritative source if the workflow permits it.
Otherwise exclude the candidate or ask the customer where the question is appropriate and safe.

Keep hard restrictions distinct from preferences.
A preference for a particular brand may be traded against a modest price increase if the customer permits it.
A prohibition on a specified ingredient is not a weighted feature to trade away.
Product rules should specify package equivalence and allowed quantity transformations.
Offering two smaller packs may change total price, picking effort, and packaging; it is not automatically a one-to-one substitution.

Branch: if no candidate remains after hard rules, record the exclusions and offer none.
Do not promote a rejected candidate simply to increase fill rate.
If rules conflict or a high-risk attribute is ambiguous, route to the designated specialist.

### Step 5 — Resolve permission and customer terms

**Input:** customer setting, order-level instructions, candidate, proposed price, policy version, and event time.
**Output:** allowed action, required customer prompt, or no-substitution decision.

A customer may select no substitutions, choose a specific replacement, allow alternatives within defined bounds, or request approval for every change.
Represent these as distinct states, not one broad Boolean.
A specific customer-selected replacement still requires the stock and safety checks relevant at execution.
Record the scope, source interface, version of the text shown, time, and any later withdrawal.
The latest valid instruction for the specific order and item should take precedence according to documented business rules.

The proposed offer should show what changes: product, quantity, relevant attributes, actual or bounded price, and any delivery implications.
If a merchant offers a price cap or absorbs a price difference, show the terms and implement them in the charge and credit process.
Do not infer permission for a higher price from acceptance of a different brand.
A setting stored months ago may not cover a new product category or fulfillment channel.

Branch: if permission is absent, expired, ambiguous, or withdrawn, request authorization if there is time and a reliable channel.
If no response arrives, follow the disclosed default; in this worked policy that default is no substitution and a refund for the unfilled line.
A different organization may have a different lawful and clearly disclosed contract; the system must implement that actual agreement rather than guess.

### Step 6 — Estimate customer and inventory outcomes

**Input:** eligible candidate, customer context, proposed terms, location, time horizon, and historical outcomes.
**Output:** probabilities with conditioning and uncertainty appropriate to the decision.

Define `p_accept` as the probability the customer accepts the displayed offer under the stated terms.
Define `p_pick_given_accept` as the probability the candidate can be picked for the requested quantity after acceptance.
Their product is the joint probability of acceptance and successful pick under this sequence:

```text
P(accept and pick | offer) = P(accept | offer) × P(pick | accept, offer)
```

The multiplication does not require acceptance and pickability to be independent because the second probability is conditional on acceptance.
It would be invalid to use an unconditional stock probability without a justified independence assumption or a joint model.
If the workflow reserves before asking, the probability tree changes and the offer cost may include hold contention.
Use the sequence that actually runs.

Past acceptance labels are observed only for customers who saw an offer.
A policy may systematically expose a candidate to shoppers most likely to accept it.
Consequently, historical acceptance by itself does not identify the effect of offering that candidate to all eligible shoppers.
Use randomized exploration with appropriate consent and safety boundaries when estimating causal effects.

Calibrate probabilities on a temporally held-out period and evaluate by location, category, stock age, and relevant customer cohorts where sample sizes and privacy allow.
A probability of 0.70 should correspond to roughly 70 successful outcomes per 100 comparable observations over sufficient data, not necessarily each small group.
Monitor confidence intervals and changes in mixture.

### Step 7 — Compare incremental economics

**Input:** conditional event probabilities, realized contribution in each branch, and the baseline outcome.
**Output:** expected incremental value per offer and sensitivity to uncertain inputs.

List every branch of the actual sequence.
In the example below, the customer can reject, accept but encounter a failed pick, or accept and receive the item.
Each branch has a conditional probability and net incremental outcome relative to refunding the original without an offer.
Include offer-handling cost for every offer if it is incurred regardless of outcome.

```text
Expected incremental value =
    P(accept) × P(pick | accept) × incremental value if picked
  + P(accept) × P(failed pick | accept) × incremental value if failed
  + P(reject) × incremental value if rejected
  - cost incurred on every offer
```

Each value is in currency per affected order line, and each probability refers to the same population, offer, and time horizon.
Avoid counting the same customer credit inside both a margin figure and a separate cost line.
If customer acceptance leads to different quantities or prices, compute the branch economics for those actual terms.
If an original item can later become available, the baseline should reflect that possibility instead of assuming zero value.

Rank only after eligibility and authorization checks.
Expected contribution can be one component of the score; customer relevance, diversity, and operational objectives may also matter.
If the business uses a weighted score rather than direct monetary value, document units and trade-offs, and preserve hard constraints.
A customer-facing list need not be ordered purely by margin.

Branch: if the best option has negative expected incremental value, the policy may offer none unless a separate service commitment justifies it.
A negative monetary estimate does not prove that a service-oriented offer is wrong; it means the decision owner must document the nonfinancial reason and guardrails.

### Step 8 — Reserve, offer, and execute

**Input:** selected candidate, authorized terms, available inventory service, and fulfillment time.
**Output:** offer and, if accepted, a pickable order instruction or a safe failure path.

A reservation must be created by the inventory authority, not implied by a ranker's cache.
Specify whether the hold is exclusive, its quantity, expiry, refresh policy, and release conditions.
Check that the same unit cannot be committed simultaneously to several orders.
Use an atomic inventory operation or a service contract with equivalent concurrency protection.

If reservation occurs after customer acceptance, explicitly communicate that availability is subject to confirmation where that is the actual process.
If reservation fails, do not silently substitute a different item under an acceptance that applied only to the first offer.
Re-rank and ask again only if the customer's policy and timing allow it.
Otherwise refund or follow the authorized fallback.

At pick time, verify the actual item, quantity, condition, and any applicable critical attributes.
A picker override needs a reason and must remain inside the customer's and organization's hard rules.
A manual decision is not an exemption from safety or authorization.
Publish the final substitution, actual price, and any credit through the normal customer communication channel.

Make commands idempotent so a retry cannot reserve twice or generate duplicate charges.
Handle lost acknowledgments by querying the authoritative state before retrying.
A cancelled order must release its hold, and an expired hold must not be treated as live.
Where events arrive out of order, reconcile against source-of-truth state rather than assuming last arrival is last occurrence.

### Step 9 — Measure the entire funnel

**Input:** assignment, eligibility, offer, customer response, inventory, pick, charge, return, and support events.
**Output:** operational metrics and an experimental estimate of incremental effects.

Useful counts include eligible stockouts, candidates passing hard rules, offers rendered, accepts, successful picks, actual substitutions, refunds, returns, complaints, and safety escalations.
Define each denominator explicitly.
Acceptance per shown offer differs from successful substitution per eligible stockout.
A high acceptance rate can coexist with a low fulfillment rate if offers frequently run out before pick.

Measure customer outcomes and cost on the randomized assignment population, including shoppers who never saw an offer because of downstream system failure.
This intention-to-treat comparison estimates the effect of assigning the policy under the actual delivery process.
A conditional comparison among accepted offers answers a different question and is susceptible to selection bias.
Measure unit fill and line fill separately.

Link refunds and returns to the original order line within a stated attribution window.
Monitor uneven error rates by location, time, customer accessibility needs, and relevant categories where lawful and adequately sampled.
A lift in fill rate accompanied by a rise in unwanted substitutions is not automatically success.

### Step 10 — Operate, govern, and learn

**Input:** metric dashboards, exceptions, customer feedback, catalog changes, and inventory incidents.
**Output:** maintained policy, corrective action, and auditable ownership.

The service owner sets an incident response for stale feeds, unavailable permission records, and failed reservation calls.
The category owner reviews exclusion reasons and product attribute quality.
Finance reconciles estimated branch contribution to realized charges, credits, and returns.
Customer care must see a comprehensible history without receiving unnecessary sensitive information.

Set a rollback switch that can disable automated substitutions while preserving the normal refund flow.
Choose stop conditions before launch: for example, confirmed prohibited substitutions, a material increase in complaints, or sustained stale inventory beyond the agreed freshness threshold.
A safety incident can require immediate suspension independent of statistical significance.
Review models and rules after operational changes, not only on a calendar.

## 6. Worked example

### Business setting and scope

A grocery retailer receives 1,000 eligible order lines per week in which a 1-liter lactose-free milk item is unavailable at the assigned store.
The retailer considers an alternative 1-liter lactose-free product for customers whose order settings permit a specific offer.
The customer is asked to accept the displayed item and actual price before the picker substitutes it.
The customer can decline; silence results in no substitution and a refund of the unavailable line.

The quantity is one unit per order line.
The example is fictional and the values are assumptions, not reported company results.
Currency is U.S. dollars (USD).
The calculation covers a 30-day return and credit window, with a fixed fulfillment workflow.
There are no other basket effects in the base case.

### Check eligibility first

The original and alternative are both documented as lactose-free 1-liter refrigerated milk products.
The customer has not prohibited the alternative's ingredients or brand.
The product owner has validated the attribute source and approves this comparison for this category.
A conventional milk product is rejected regardless of its lower price or higher margin because it fails the customer's hard requirement.
If the lactose-free attribute is missing, the candidate is not automatically treated as compliant.

The example considers a customer-facing offer rather than an automatic substitution.
The order-specific setting permits the retailer to ask.
The approval, if obtained, applies to the displayed item and price only.
It does not authorize a different item if the offered one later becomes unavailable.

### Interpret the inventory snapshot

At the offer decision time, the store reports four units on hand, one unit committed to other orders, and a one-unit policy buffer.
Under this example's bookkeeping convention the on-hand count includes those other committed units.
The current order is not yet reserved.

```text
Book ATP = max(0, 4 - 1 - 1) = 2 units
```

The book ATP indicates that a one-unit offer passes the simple quantity screen.
It does not guarantee a pick.
Based on comparable historical offers and the relevant pick horizon, the estimated probability of successful pick **conditional on acceptance** is 0.72.
That estimate already reflects the modeled depletion and inventory discrepancies for this cohort.
It must not be multiplied by a second generic availability factor for the same risk.

The expected pick is four hours after the offer.
The forecast is a 0.72 probability, not a promise that this particular unit will be found.
The plan is to request acceptance and attempt a pick.
The operational target is set separately by the retailer; no target is assumed achieved here.
The customer commitment is limited to the disclosed offer and refund process.

### State the probability tree

Historical evidence for comparable customers and the same offer terms gives an assumed acceptance probability of 0.65.
Conditional on accepting, the pick succeeds with probability 0.72 and fails with probability 0.28.
Rejecting includes an explicit decline or a timeout under the stated policy.

| Branch | Probability | Interpretation |
|---|---:|---|
| Accept and successful pick | 0.65 × 0.72 = 0.468 | Customer approves and item is obtained |
| Accept and failed pick | 0.65 × 0.28 = 0.182 | Customer approves but stock or pick fails |
| Reject or time out | 1 − 0.65 = 0.350 | No substitution is attempted |
| Total | 0.468 + 0.182 + 0.350 = 1.000 | All outcomes for one offer |

The success rate per offered line is 46.8%, not 72%.
The latter is conditional on acceptance.
The acceptance rate is 65.0%, not the final substitution rate.

### Define the counterfactual and branch values

Without the offer, the retailer refunds the unavailable line and retains the remaining basket.
The incremental value of that baseline is defined as USD 0.00 per affected line.
A successful substitution produces USD 2.40 incremental contribution after merchandise cost and variable picking cost, before any offer-level cost.
That USD 2.40 already includes the expected customer credit and return expense for successful substitutions within the 30-day window.
A failed pick after acceptance costs an additional USD 0.80 in handling and support compared with the no-offer baseline.
A rejection adds no branch-specific cost beyond the universal offer cost.
Creating and delivering an offer costs USD 0.05 per eligible line, including offers that are rejected.

| Branch | Incremental value before offer cost | Reason |
|---|---:|---|
| Successful pick | +USD 2.40 per line | Net contribution after included variable costs and expected credits |
| Failed pick after acceptance | −USD 0.80 per line | Additional handling and support |
| Reject or timeout | USD 0.00 per line | Same refund path as baseline |
| Any offer | −USD 0.05 per line | Universal offer and messaging cost |

The expected incremental value per offer is:

```text
0.468 × USD 2.40 + 0.182 × (−USD 0.80) + 0.350 × USD 0.00 − USD 0.05
= USD 1.1232 − USD 0.1456 − USD 0.05
= USD 0.9276 per offer
```

For 1,000 comparable eligible offers per week, the forecast is USD 927.60 incremental contribution per week.
This is not a budget commitment or a measured causal lift.
The forecast assumes that 1,000 offers can be made under the same probabilities and costs, and that contention does not change them.
Expected weekly outcomes are 468 successful substitutions, 182 failed picks after acceptance, and 350 rejections or timeouts.
They are expectations, not promised counts.

### Sensitivity to fulfillment probability

Hold the 0.65 acceptance probability and branch values fixed while changing `p_pick_given_accept`.
This is a one-factor sensitivity exercise; in reality acceptance and stock may change together.

| Conditional pick probability | Expected successful substitutions per 1,000 offers | Expected incremental value per offer |
|---:|---:|---:|
| 0.40 | 260 | USD 0.2620 |
| 0.60 | 390 | USD 0.6780 |
| 0.72 | 468 | USD 0.9276 |
| 0.80 | 520 | USD 1.0940 |
| 0.90 | 585 | USD 1.3020 |

For example, at 0.40 the arithmetic is `0.65 × (0.40 × 2.40 − 0.60 × 0.80) − 0.05 = USD 0.2620`.
The break-even conditional pick probability `p` satisfies:

```text
0.65 × (p × 2.40 − (1 − p) × 0.80) − 0.05 = 0
p = (0.05 / 0.65 + 0.80) / 3.20 ≈ 0.2740
```

This threshold depends on the acceptance rate and assumed costs.
It is not a universal inventory rule and does not override compatibility or consent.
Below approximately 27.4% conditional pick probability, the offer is negative on the specified short-term economics.
A confidence interval crossing that threshold warrants additional caution or data, not a claim of certainty.

### Decision and ownership

Under the stated assumptions, the commercial owner may approve a controlled experiment because the forecast is positive.
The category owner must separately approve the compatibility rule.
The privacy and product owners must approve the scope and wording of the customer interaction.
The inventory owner must confirm the source and freshness of the stock signal and the fallback when it is unavailable.
The experiment owner must compare the policy with the no-offer baseline on all eligible stockout lines.
A real deployment requires measured outcomes and operating costs; the example alone cannot justify launch.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the probability tree and expected contribution in the worked example.
It also applies basic hard eligibility and permission checks before returning an offer decision.
It is a teaching calculation, not production inventory, payment, or consent infrastructure.

The code assumes a single one-unit candidate and an explicit customer-facing offer.
The supplied `pick_given_accept` is already conditional on customer acceptance.
The `successful_value` already contains expected credit and return costs; it must not be reduced by those costs again.
The data structure contains only the minimum fields needed for this example, not every possible compatibility rule.
A missing hard attribute is rejected, not treated as safe.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    sku: str
    lactose_free: bool | None
    size_liters: float
    on_hand: int
    committed: int
    buffer: int
    ask_permission: bool
    acceptance: float
    pick_given_accept: float
    successful_value_usd: float
    failed_pick_value_usd: float
    offer_cost_usd: float


def probability(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def evaluate(candidate: Candidate) -> dict:
    """Return a decision and values per offered order line."""
    if candidate.lactose_free is not True:
        return {"decision": "exclude", "reason": "critical_attribute"}
    if candidate.size_liters != 1.0:
        return {"decision": "exclude", "reason": "size_mismatch"}
    if not candidate.ask_permission:
        return {"decision": "exclude", "reason": "permission_to_ask"}

    atp = max(0, candidate.on_hand - candidate.committed - candidate.buffer)
    if atp < 1:
        return {"decision": "exclude", "reason": "insufficient_book_atp"}

    accept = probability(candidate.acceptance, "acceptance")
    pick = probability(candidate.pick_given_accept, "pick_given_accept")
    success = accept * pick
    failed_pick = accept * (1.0 - pick)
    rejected = 1.0 - accept
    value = (
        success * candidate.successful_value_usd
        + failed_pick * candidate.failed_pick_value_usd
        - candidate.offer_cost_usd
    )
    return {
        "decision": "offer" if value > 0.0 else "no_offer",
        "atp_units": atp,
        "success_probability": success,
        "failed_pick_probability": failed_pick,
        "reject_probability": rejected,
        "incremental_value_usd": value,
    }


candidate = Candidate(
    sku="ALT-LF-1L",
    lactose_free=True,
    size_liters=1.0,
    on_hand=4,
    committed=1,
    buffer=1,
    ask_permission=True,
    acceptance=0.65,
    pick_given_accept=0.72,
    successful_value_usd=2.40,
    failed_pick_value_usd=-0.80,
    offer_cost_usd=0.05,
)
result = evaluate(candidate)
print("decision:", result["decision"])
print("book ATP units:", result["atp_units"])
print("success probability:", f"{result['success_probability']:.3f}")
print("failed-pick probability:", f"{result['failed_pick_probability']:.3f}")
print("reject probability:", f"{result['reject_probability']:.3f}")
print("incremental USD/offer:", f"{result['incremental_value_usd']:.4f}")
print("incremental USD/1000 offers:", f"{1000 * result['incremental_value_usd']:.2f}")
```

Save the block as `substitution_example.py` and run it with Python 3.10 or later:

```bash
python substitution_example.py
```

Expected output:

```text
decision: offer
book ATP units: 2
success probability: 0.468
failed-pick probability: 0.182
reject probability: 0.350
incremental USD/offer: 0.9276
incremental USD/1000 offers: 927.60
```

The `offer` result means only that this simplified candidate passes the illustrated screens and has positive modeled incremental value.
It is not customer acceptance, a stock reservation, an authorization to substitute, or a guarantee of positive realized value.
A real service must check current customer instructions and inventory at the relevant transitions.
For currency settlement, use decimal arithmetic and the organization's rounding policy rather than binary floating-point values.

### Interpreting and extending the calculation

The program screens one candidate, so it does not demonstrate choosing among alternatives.
For a production policy, evaluate each eligible candidate on the same decision horizon and counterfactual.
Do not compare candidates evaluated using incompatible prices, customer populations, or pick windows.
If a customer's selected replacement is available, apply the customer's instruction before applying a merchant-defined ranking objective.
If several candidates remain, choose a policy that states how customer relevance and incremental economics interact.

A candidate with a higher success probability is not necessarily more valuable.
Suppose candidate X has a joint acceptance-and-pick probability of 0.50 and produces USD 1.00 on success.
Suppose candidate Y has a joint probability of 0.40 and produces USD 2.00 on success.
Ignoring all other costs, X produces USD 0.50 per offer and Y produces USD 0.80 per offer.
However, that comparison is incomplete if Y causes additional failed-pick handling, returns, or customer dissatisfaction.
The branch ledger must include those terms before a decision is made.

The acceptance probability also depends on displayed terms.
A price increase may reduce customer acceptance, so it is not generally valid to hold acceptance fixed while increasing price.
The acceptance model must be conditioned on the offer the customer actually sees.
Similarly, an availability model should reflect whether the candidate is held while the customer responds.
If the workflow changes the order of holding and asking, retrain or recalibrate the relevant conditional probabilities.

### Sensitivity and value of information

The break-even threshold in the worked example can guide investment in better information.
Suppose the best current estimate of conditional pick probability is 0.30, close to the approximately 0.274 break-even threshold.
Small estimation errors could change the economic decision.
A fresh stock check may be valuable if it meaningfully changes the decision and its expected benefit exceeds its cost.
The value of information is not the value of the check itself; it is the expected improvement in the decision after paying for the check.

For a simplified illustration, assume a free stock signal creates two equally likely groups.
In the first group the conditional pick probability is 0.50; in the second it is 0.10.
The overall probability remains 0.30.
Using the worked example's other assumptions, the no-signal policy offers to everyone because its value is positive:

```text
0.65 × (0.30 × 2.40 − 0.70 × 0.80) − 0.05 = USD 0.054 per line
```

If the signal is genuinely available before deciding, the retailer can offer only to the 0.50 group:

```text
High group: 0.65 × (0.50 × 2.40 − 0.50 × 0.80) − 0.05 = USD 0.470
Low group:  0.65 × (0.10 × 2.40 − 0.90 × 0.80) − 0.05 = −USD 0.362
With signal: 0.50 × USD 0.470 + 0.50 × USD 0.00 = USD 0.235 per line
```

Under these assumptions, the gross expected value of the signal is USD 0.235 − USD 0.054 = USD 0.181 per eligible line.
A check costing USD 0.20 per line would not pay for itself on this narrow economic model.
A check costing USD 0.05 per line would leave an estimated USD 0.131 benefit per line.
This calculation assumes the signal is accurate, available before the offer, and does not change customer acceptance or other costs.
It is not evidence that any actual stock-check technology achieves those properties.

The same logic applies to learning whether a customer permits a particular alternative.
However, repeatedly asking a customer has a time and experience cost, and an unanswered question does not become permission.
A decision owner should compare the benefit of asking with the operational and customer costs, then follow the applicable consent policy regardless of estimated value.

### Reconciliation checklist

At the end of each accounting period, compare the decision forecast with realized outcomes from a mature cohort.
The finance owner should reconcile at least the following:

1. The count of eligible stockout lines against the experiment or policy assignment ledger.
2. The count of rendered offers against delivery logs, excluding offers merely generated.
3. The response count against accepted, declined, and timed-out offer states.
4. The successful-pick count against fulfilled order lines and actual item identifiers.
5. Customer charges against the offer terms and any applicable price cap.
6. Refunds and returns against the original order line and the stated maturity window.
7. Incremental handling and support costs against the no-offer comparison group.
8. Predicted probabilities against observed frequencies by relevant operational segment.

An accounting discrepancy is not automatically a model defect.
It can come from a missing event, a late refund, a different definition of cost, or a legitimate change in the fulfillment process.
Resolve the source and event timing before changing a decision threshold.

## 8. Data, standards, and regulatory context

### The minimum data contract

| Data element | Grain and timing | Why it matters | Responsible owner |
|---|---|---|---|
| Order line | Order, item, quantity, promised window | Defines the requested outcome and analysis unit | Order platform |
| Product identity | Sellable variant and effective dates | Avoids mixing different package sizes or formulations | Catalog owner |
| Product attributes | Attribute, source, value, validity and verification time | Supports eligibility and critical-attribute review | Category and quality |
| Inventory snapshot | Item, location, channel, source event time | Provides a point-in-time stock observation | Inventory operations |
| Allocation event | Item, location, order, quantity, expiry | Prevents reuse of committed stock | Reservation service |
| Customer instruction | Customer or order scope, option, language version, time, withdrawal | Establishes permitted action | Product and privacy |
| Offer event | Order line, candidate, terms, render time, policy version | Separates generation from actual exposure | Recommendation service |
| Response event | Offer, accept/decline/timeout, event time | Measures choice and permission scope | Customer interface |
| Pick event | Order line, actual item, quantity, reason, time | Measures physical execution | Fulfillment operations |
| Financial event | Order line, price, cost, credit, refund, return | Reconciles economics | Finance and payments |
| Experiment assignment | Randomization unit, arm, timestamp, eligibility version | Supports unbiased policy comparison | Experiment platform |

Every event needs a stable identifier and an event time.
Use a separate ingestion timestamp to assess latency.
Retain product-rule and model versions so a customer-care agent can reconstruct the decision actually made.
Prevent later catalog changes from rewriting the attributes that were known at the original decision time.

A location or product key can be mapped incorrectly even when every field is present.
Validate join cardinality, unit conversion, and timezone handling.
For delayed or corrected source events, record the correction rather than deleting evidence of the original observation.
The point-in-time training dataset must not use a later inventory correction as if it were known earlier.

### GS1 identifiers and product information

[GS1's GTIN standard](https://www.gs1.org/standards/id-keys/gtin) defines an identifier for trade items.
A GTIN can help match an item across trading partners, but it does not itself establish whether two products are safe or acceptable substitutes.
The merchant still needs reliable attributes, variant resolution, and customer-specific rules.
[GS1 Global Data Synchronisation Network](https://www.gs1.org/standards/gdsn) describes a framework for sharing product data among trading partners.
Use of a standard does not guarantee that every trading partner's attribute values are correct, current, or complete.
The organization needs provenance and quality checks regardless of the identifier format.

### Food and product information

For U.S. food products, the [U.S. Food and Drug Administration food allergen resources](https://www.fda.gov/food/food-labeling-nutrition/food-allergies) describe major allergens and labeling context.
Actual obligations depend on product type, jurisdiction, and facts; consult qualified counsel and quality specialists.
A product catalog must not convert an absent or outdated label field into a guarantee that an allergen is absent.
The example's lactose-free condition is a customer requirement and is not a general claim that lactose is an allergen.

For medical devices, medicines, industrial safety parts, and other regulated categories, equivalence can require professional authorization and additional technical evidence.
A generic recommendation policy is not a substitute for those controls.
An organization must identify the rules applicable to its own products and workflow before deployment.

### Privacy and marketing choices

In the European Union, the [General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj) establishes obligations concerning processing personal data, including principles in Article 5 and legal bases in Article 6.
If consent is the chosen legal basis, its specific conditions include those in Articles 4(11) and 7.
A customer's commercial permission to substitute is not automatically the same thing as GDPR consent for data processing.
Identify the applicable legal basis, purpose, retention period, rights process, and any relevant national rules with counsel.

In the United States, state privacy laws may apply depending on the business and data; for example, see the [California Privacy Protection Agency](https://cppa.ca.gov/).
Privacy requirements vary by jurisdiction and change over time.
The regulatory references here are current as educational pointers as of November 2026, not legal advice or a claim that every law applies to every merchant.
A dietary preference can reveal sensitive information in context; collect and expose only what is needed for the authorized purpose.
Use role-based access, retention limits, and an auditable withdrawal process.

### Distinguish standards, rules, and local policy

An identifier standard establishes how an identifier is structured and used, not whether a substitution is permissible.
A regulation imposes obligations only where it applies and must be interpreted in context.
A merchant's price cap, stale-data threshold, and no-response default are local policy decisions unless a specific legal or contractual requirement says otherwise.
Document the source and effective date of each rule.
Do not cite a generic standard as proof that the merchant's particular recommendation is compliant.

## 9. Industry practice and case evidence

### Instacart: customer choice and substitution workflow

[Instacart's help on replacing items](https://www.instacart.com/help/section/360007902831/360039572411) describes customer choices around replacements and shopper communication.
The cited material is a company description of its service, not independent evidence of causal impact or an assurance that every marketplace uses the same process.
The practical design lesson is to distinguish an item-level customer instruction from the eventual picker action.
A customer-approved replacement still requires an available, correctly picked item.

**Inference:** a substitution system using such choices needs events for the original request, selected replacement, any subsequent customer changes, and actual fulfillment.
That inference is architectural, not a claim that Instacart exposes or stores precisely those fields internally.
An experiment would be required to determine whether a particular interface improves satisfaction or contribution.

### Walmart: a reported substitution approach

Walmart describes the use of artificial intelligence to suggest online grocery substitutions in its corporate article, ["How AI Is Helping Walmart Make Smarter Substitutions in Online Grocery Orders"](https://corporate.walmart.com/news/2021/06/14/how-ai-is-helping-walmart-make-smarter-substitutions-in-online-grocery-orders).
The article is a company source and should be read as a description of Walmart's reported approach, not an independent evaluation of a policy in another retailer.
It describes the importance of finding alternatives when an ordered grocery product is not available.
Company-reported outcomes, where present in the source, are not automatically transferable to a different assortment, customer population, or period.

**Inference:** a retailer implementing a similar process must join product knowledge, local availability, customer preference, and picker execution.
This unit's probability tree and numerical example are not Walmart's published implementation or results.

### Research on substitution and lost sales

The peer-reviewed paper by Anupindi, Dada, and Gupta, ["Estimation of Consumer Demand with Stock-Out Based Substitution: An Application to Vending Machine Products"](https://doi.org/10.1287/mksc.17.4.406), studies demand with substitution when products stock out.
Its context is not the same as a modern personalized online grocery flow.
Its relevant lesson is that observed sales depend on the set of available products; missing a first choice can redirect demand or produce a lost sale.
A sales log alone does not reveal every customer's unconstrained first choice.

**Inference:** training a recommender only on completed substitutions can overrepresent offers that were both shown and available.
Offer exposure, stock state, and rejected or unserved requests need to be retained to diagnose that selection.
This does not by itself identify a causal treatment effect; an experiment or defensible identification strategy is still needed.

### Transfer to other business settings

In business-to-business maintenance, a replacement bearing may require precise dimensions, load ratings, certifications, and equipment approval.
A catalog similarity model can help retrieve candidates, while an authorized engineer determines whether a candidate is compatible.
A stockout can lead to downtime costs that exceed the part margin, but those costs should not be invented when absent from the counterfactual.
Contract terms may require a particular part, approval workflow, or service level.

In fashion, size and fit uncertainty may dominate physical stock uncertainty.
A substitute with the correct label size may fit differently across brands.
Return and reverse-logistics costs belong in the measured economics, and the customer must understand that a different item is offered.

In grocery, freshness, dietary restrictions, package size, and delivery-window timing can dominate the decision.
A customer may permit one alternative but prohibit automatic replacement of another item in the same basket.
A category-specific policy is therefore safer than a universal semantic-similarity threshold.

Across settings, do not generalize a vendor's aggregate result into a promise for your organization.
Define the actual use case, check operating constraints, and measure incremental outcomes against a relevant baseline.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Stale stock feed | Event age exceeds the operating threshold | A cached count misses recent demand | Failed picks and customer disappointment | Reject stale data, query source, or use safe fallback |
| Double-counted reservations | ATP falls unexpectedly after a service change | On-hand already excludes committed units | Unnecessary no-offer decisions | Document inventory semantics and reconcile counts |
| Uncounted reservations | Several orders claim the same final unit | Ranking reads stock without atomic allocation | Overselling and substitution failures | Use authoritative reservation service and expiry |
| Location mismatch | High failures in one channel or store | Warehouse and store identifiers are joined incorrectly | Unfulfillable offers | Validate location mapping and pick-level outcomes |
| Unit conversion error | Unusual per-unit prices or quantities | Pack size and sellable-unit metadata differ | Overcharge or wrong quantity | Normalize units and test conversions with category owners |
| Unsafe equivalence | A prohibited attribute appears among offers | Similarity overrides a hard restriction | Customer harm and policy breach | Apply independent hard rules before ranking |
| Unknown treated as safe | Critical fields are often blank | Missing attribute defaults to false or empty | Undetected incompatible products | Use explicit unknown state and fail safely |
| Overbroad permission | Complaints say customer approved a different item | A one-item acceptance is reused for another SKU | Unwanted substitution and loss of trust | Bind permission to item, terms, scope, and time |
| Silent timeout interpreted as acceptance | Substitute rate rises while response rate falls | No-response path is coded as approval | Unauthorized changes | Implement the disclosed default and audit response events |
| Price mismatch | Customer receipts exceed displayed prices | The price cap or credit is not applied at checkout | Complaints, refunds, and potential legal exposure | Verify actual charge and credit against offer terms |
| Conditional probability misuse | Forecast is materially above realized fill | Acceptance and stock are multiplied using inconsistent cohorts | Inflated benefit | Specify event sequence and calibrate conditional estimates |
| Training leakage | Offline accuracy is implausibly high | Final pick or corrected stock enters historical features | Online failure | Build point-in-time features from event timestamps |
| Exposure selection | High offline acceptance but poor rollout | Only previously shown candidates have labels | Overconfident model and unequal coverage | Log exposure and randomize within safe boundaries |
| Ignored failed picks | Finance forecast exceeds realized margin | Only successful offers are costed | Negative incremental contribution | Include every outcome branch and reconcile to ledger |
| Expired hold used | Pick failures cluster after long delays | Reservation TTL passes before execution | Overselling and rework | Check live reservation and handle expiry explicitly |
| Duplicate command | Duplicate holds or credits appear | Retries are not idempotent | Inventory loss or incorrect charge | Use stable idempotency keys and reconciliation |
| Picker override bypasses rules | Manual substitutions dominate incident reviews | Interface permits unvalidated alternatives | Safety or consent violation | Apply rules to manual actions and capture reason codes |
| Aggregate metric masks harm | Overall fill rises while a store's complaints rise | Segment-level degradation is hidden | Unequal customer experience | Monitor relevant segments with uncertainty and privacy controls |
| No rollback | Teams cannot disable offers during a feed outage | Model and refund flows are tightly coupled | Extended incident | Provide a tested policy switch and safe refund path |
| Feedback loop | Repeatedly shown candidates dominate training data | Past ranking controls future labels | Reduced choice and biased estimates | Track exposure and use governed randomized exploration |

### Responding to an incident

First stop unsafe or unauthorized substitutions, preserving the ability to refund.
Then identify affected orders and customers using event and policy versions.
Customer care and the appropriate safety owner decide communications and remediation.
Engineering preserves logs needed for investigation under the retention policy.
Do not restart merely because the average metric improves; resolve the mechanism and obtain the accountable owner's approval.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or monitor |
|---|---|---|---|
| Stock records refer to the pick location | Enables a meaningful ATP estimate | Cross-location mapping or transfer delays | Reconcile location keys against physical pick events |
| Commitments are counted once | Avoids overstating or understating ATP | Different systems define on-hand differently | Reconcile reservation ledger with inventory snapshots |
| Historical probabilities apply to the next period | Supports a forecast | Seasonality, new interface, or store process changes | Temporal holdout and rolling calibration by segment |
| Conditional pick probability matches the workflow | Makes the probability tree valid | Reservation timing or customer response changes | Compare predicted and observed outcomes by event sequence |
| Eligibility data are accurate and current | Hard rules depend on attributes | Reformulation or incomplete supplier updates | Source provenance, effective dates, and quality audits |
| The stated customer instruction is current | Defines allowed actions | Withdrawal, changed order, or ambiguous scope | Event ordering, expiry rules, and customer-care reviews |
| Successful-value estimate includes the specified costs | Prevents omitted or duplicated costs | Later credits, returns, and fee changes | Reconcile cohort outcomes to finance ledger |
| The no-offer baseline is realistic | Defines incrementality | Customers would buy another item or cancel the basket | Randomize the policy and measure both arms |
| One line does not affect another | Allows per-line evaluation | Shared inventory, delivery fees, or basket cancellation | Examine basket-level outcomes and contention |
| Offer cost is constant | Simplifies calculation | Variable messaging, human review, or service escalation | Measure costs by channel and exception type |
| One-unit candidate represents demand | Simplifies quantity checks | Multiunit orders or pack-size differences | Analyze quantity segments and define conversion policy |
| Outcomes mature within 30 days | Closes the example's economics window | Late returns or longer-term customer effects | Compare windows and separately track retention |
| Probability estimates are sufficiently precise | Supports a threshold decision | Small sample or drift | Report confidence intervals and sample sizes |
| Customer response is observed reliably | Separates acceptance from timeout | Notification failures or missing event logs | Audit delivery, render, response, and timeout events |

The worked numbers are illustrative point estimates.
They are not evidence that a retailer will realize the forecast.
The simple model does not solve a joint allocation problem when many customers compete for scarce alternatives.
For shared inventory, an allocation policy must account for capacity, priority, fairness, and contractual obligations rather than independently promising the same unit to every customer.

## 12. Exercises

### Exercise 1 — Map the operational decision

**Time:** 15 minutes.
**Deliverable:** one flow diagram and a list of five accountable owners.
A shopper orders a package of gluten-free pasta for delivery tomorrow.
At picking, the original product is not on the shelf and a similar non-gluten-free product is available.
The customer has allowed the merchant to ask about replacements but has not approved a specific product.

Include the stockout trigger, retrieval, hard eligibility, permission, offer, pick, and refund branches.
Name the owners of product equivalence, customer permission, inventory, economics, and deployment.

**Acceptance criteria:** the ordinary pasta is rejected before ranking; the diagram includes an offer-none path and distinguishes asking permission from customer acceptance.

### Exercise 2 — Calculate stock and funnel outcomes

**Time:** 15 minutes.
**Deliverable:** ATP in units and three branch probabilities.
At the relevant store, on-hand is 9 units, other commitments are 3 units, and the buffer is 2 units.
On-hand includes the commitments, and the current order is not yet reserved.
The customer accepts with probability 0.60; the probability of a successful pick conditional on acceptance is 0.75.

**Acceptance criteria:** show the ATP arithmetic, list success, failed-pick, and rejection probabilities, and explain why ATP is not a guarantee.

### Exercise 3 — Apply hard rules and permission

**Time:** 20 minutes.
**Deliverable:** an eligibility table and a customer action for each candidate.
The customer needs one 1-liter lactose-free milk and permits offers up to USD 4.50 per unit.
Candidate A is lactose-free, 1 liter, USD 4.20, and has adequate book ATP.
Candidate B is ordinary milk, 1 liter, USD 3.50, and has the highest similarity score.
Candidate C has unknown lactose-free status, 1 liter, USD 4.00, and has adequate book ATP.
Candidate D is lactose-free, 1 liter, USD 4.80, and has adequate book ATP.
The customer has authorized an offer, not an automatic substitution.

**Acceptance criteria:** use explicit reasons for all four candidates; do not treat missing status as safe or the price cap as optional; identify whether acceptance has occurred.

### Exercise 4 — Compute incremental value

**Time:** 20 minutes.
**Deliverable:** probability tree, expected USD per offer, and a decision under the stated objective.
An offer is accepted with probability 0.60 and, conditional on acceptance, picked successfully with probability 0.75.
The successful branch is worth USD 3.00 incremental to the no-offer baseline.
A failed pick after acceptance costs USD 1.00 incremental; rejection has zero branch-specific value.
Each offer costs USD 0.10 regardless of outcome.
Assume all costs are mutually exclusive and the offer satisfies all hard rules.

**Acceptance criteria:** calculate all branch probabilities, include both failure and universal offer costs, and state the baseline.

### Exercise 5 — Design consent and execution states

**Time:** 30 minutes.
**Deliverable:** state machine, event schema, and three failure paths.
The system must support no substitutions, a customer-selected SKU, a bounded offer, and withdrawal.
A hold expires after 15 minutes and a picker can discover damage after acceptance.

**Acceptance criteria:** bind acceptance to offer terms and version; include timeout, expiry, and failed pick; prevent a new SKU from inheriting old acceptance; name the source of truth and idempotency key.

### Exercise 6 — Plan a randomized evaluation

**Time:** 35 minutes.
**Deliverable:** one-page experiment plan.
A retailer proposes replacing the no-offer policy with the worked policy for eligible stockouts in selected stores.
Customers may have multiple affected lines and inventory is shared within stores.

**Acceptance criteria:** identify a feasible randomization unit and interference risk; define the intention-to-treat estimand, primary outcome, time window, at least three guardrails, and a stop rule; distinguish assignment from exposure.

### Exercise 7 — Critique an observational comparison

**Time:** 20 minutes.
**Deliverable:** a short critique and a better measurement plan.
The retailer reports that 80 of 100 shoppers who accepted an offer received an item, while 40 of 100 shoppers in another store who received no offer bought another item.
It concludes that offers caused a 40 percentage-point increase in fulfillment.

**Acceptance criteria:** identify at least three selection or denominator problems, state why the difference is not a causal lift, and propose a safer comparison.

### Exercise 8 — Roll out a new category

**Time:** 30 minutes.
**Deliverable:** a launch checklist with named roles, staged rollout, monitoring, and rollback.
The new category contains replacement water filters whose dimensions and certification matter.
Some catalog records lack compatibility data and inventory is updated every two hours.

**Acceptance criteria:** require an authorized compatibility source; define handling for unknown attributes and stale inventory; include customer choice, a measurable operational guardrail, a stop condition, and a responsible owner for each decision.

## 13. Self-check answers

### Exercise 1

The ordinary pasta fails the customer's gluten-free requirement and is not eligible regardless of its similarity score.
The system may retrieve it but must reject it before ranking and offering.
If no verified gluten-free candidate remains, the result is no offer and the disclosed refund path.
Permission to ask permits an offer, not an automatic replacement.
The category or quality owner approves equivalence; product and privacy own permission; inventory operations own stock semantics; finance and the commercial owner own economics; the service owner owns deployment and rollback.
A complete diagram also checks stock at pick time and records what was actually fulfilled.

### Exercise 2

```text
ATP = max(0, 9 − 3 − 2) = 4 units
P(accept and pick) = 0.60 × 0.75 = 0.45
P(accept and fail) = 0.60 × 0.25 = 0.15
P(reject) = 1 − 0.60 = 0.40
Total = 0.45 + 0.15 + 0.40 = 1.00
```

The ATP is a book estimate at one moment, not a reserved physical unit.
Competing demand, shrinkage, delayed scans, or an inaccurate count can still prevent the pick.
The multiplication is valid because the pick probability is explicitly conditional on acceptance.

### Exercise 3

| Candidate | Result | Reason |
|---|---|---|
| A | Eligible to offer | Verified requirement, size, price, and book stock satisfy the stated conditions |
| B | Exclude | Ordinary milk violates the lactose-free hard requirement |
| C | Exclude or resolve before offering | Unknown critical attribute is not evidence of compliance |
| D | Exclude under the stated cap | USD 4.80 exceeds the customer's USD 4.50 limit |

Even for A, permission to show an offer is not acceptance of that offer.
The merchant must obtain the required customer response and recheck or reserve stock according to the actual workflow.

### Exercise 4

```text
P(success) = 0.60 × 0.75 = 0.45
P(failed pick) = 0.60 × 0.25 = 0.15
P(reject) = 0.40
Expected value = 0.45 × USD 3.00 + 0.15 × (−USD 1.00) − USD 0.10
               = USD 1.10 per offer
```

The baseline is the no-offer outcome, defined as USD 0.00 incremental.
Under the stated short-term objective and assumptions, the offer has positive expected incremental value.
That arithmetic does not establish that the probabilities are calibrated or that the policy causes the projected lift.

### Exercise 5 rubric

A strong state machine separates instruction recorded, offer generated, offer rendered, accepted, rejected or timed out, held, picked, failed, and refunded.
It defines whether the hold happens before or after acceptance and specifies the resulting branch logic.
It records item, price, quantity, language or policy version, customer scope, event time, and withdrawal.
A hold-expiry event releases inventory and invalidates reliance on the hold; a retry uses an idempotency key.
A damaged item follows a failed-pick branch and does not automatically authorize another SKU.
The authoritative customer-instruction and reservation services are named.

A weak answer has only `consent=true`, assumes a notification was seen, or uses a customer's prior approval for a different item.

### Exercise 6 rubric

A strong plan selects an assignment unit before measuring outcomes and explains trade-offs.
Customer-level assignment can reduce within-customer contamination but does not remove competition for shared store stock.
Store or store-time cluster randomization may better contain inventory interference, but requires enough independent clusters and an analysis that accounts for clustering and time effects.
The choice depends on operational feasibility and the estimand; neither automatically eliminates all interference.

The intention-to-treat estimand compares mean outcomes for all eligible lines or customers assigned to the policy against those assigned to the baseline over a prespecified window.
The denominator and weighting must match the assignment unit and business question.
A suitable primary outcome is incremental contribution per eligible stockout line, with a 30-day maturity period if late returns are captured.
Guardrails can include prohibited substitutions, unwanted-substitution complaints, failed-pick rate, refund rate, and customer satisfaction.
A confirmed prohibited substitution can trigger immediate suspension and investigation.
Offer exposure is a downstream event, not the randomized assignment.

### Exercise 7

The first numerator is conditioned on accepting an offer, while the second describes a different group and action.
The first comparison excludes offer rejections and possibly non-exposed eligible customers.
The two stores can differ in assortment, stock quality, customer mix, and timing.
The outcomes may not even measure the same event: receiving a substitute is not necessarily equivalent to independently buying another item.
Therefore 80% minus 40% is an observed difference between selected groups, not a 40 percentage-point causal effect.
Randomize a policy among comparable eligible units or suitable clusters, retain every assigned unit, and compare consistently defined outcomes with appropriate uncertainty.

### Exercise 8 rubric

A strong checklist names a technical or quality owner to define and approve compatibility attributes and certification evidence.
Unknown compatibility is excluded or escalated for verified review, never guessed from text similarity alone.
The inventory owner sets an age threshold appropriate to the category and decides whether a fresh check or no-offer fallback is required; the exercise does not supply enough evidence to assert a universally safe threshold.
The product owner ensures the customer can approve the particular replacement and sees its terms.
The service owner stages a limited rollout with alerts for pick failure and prohibited replacements, and can disable automated offers while preserving refund processing.
A confirmed incompatible replacement is a stop condition with incident ownership and customer remediation.

## 14. Interview questions

| Question | Strong answer contains | Weak answer reveals |
|---|---|---|
| Why is a similar product not necessarily an acceptable substitute? | Customer purpose, hard requirements, scope of authorization, and verified attributes | Reliance on a single similarity score |
| What is the difference between on-hand and ATP? | Commitments, buffers, location, time, and local accounting semantics | Assumption that all recorded stock can be promised |
| How would you forecast availability at pick time? | Event age, horizon, local history, calibration, and conditional outcome | Treating current stock count as a guarantee |
| When can you multiply acceptance and availability probabilities? | A correctly conditioned probability tree matching the actual sequence | Multiplying unrelated unconditional estimates |
| How would you handle an unknown allergen or compatibility field? | Explicit unknown, trusted verification, exclusion or qualified escalation | Treating blank as safe |
| What does a customer acceptance authorize? | The specific item, terms, scope, and time, subject to the recorded agreement | Unlimited approval for any future replacement |
| What happens when a reservation expires? | Recheck or re-reserve, release inventory, communicate failure, and follow authorized fallback | Picking as if an expired hold were valid |
| How do you calculate incremental contribution? | All outcome branches, conditional probabilities, actual variable costs, and a counterfactual | Counting revenue from successful substitutions only |
| How do you evaluate a new policy? | Randomized assignment, intention-to-treat, guardrails, maturity window, and interference | Comparing accepted offers with unrelated non-offer shoppers |
| What should be logged to investigate a complaint? | Original instruction, offer and terms, response, policy version, stock and actual pick, with access limits | Only the final item and a generic consent flag |
| How can the system fail safely during a feed outage? | Staleness detection, controlled fallback, refund path, owner, and rollback | Continuing to serve cached stock without qualification |
| Why might line-level optimization be insufficient? | Shared inventory, basket effects, fairness, contract commitments, and allocation constraints | Assuming each candidate can be promised independently |

## 15. Key terms

- **Acceptance:** customer agreement to specific offer terms; see [Step 5](#step-5--resolve-permission-and-customer-terms).
- **Allocation:** assignment of stock to an order or competing demand; see [Step 8](#step-8--reserve-offer-and-execute).
- **ATP:** available-to-promise quantity under an explicit bookkeeping policy; see [Step 2](#step-2--obtain-and-interpret-inventory-evidence).
- **Audit trail:** ordered evidence of a decision and its outcome; see [data contract](#the-minimum-data-contract).
- **Baseline:** the alternative outcome against which incrementality is measured; see [Step 1](#step-1--define-the-decision-and-the-baseline).
- **Book inventory:** recorded count, potentially different from physical shelf stock; see [Step 2](#step-2--obtain-and-interpret-inventory-evidence).
- **Calibration:** correspondence between predicted probabilities and observed frequencies; see [Step 6](#step-6--estimate-customer-and-inventory-outcomes).
- **Candidate:** product retrieved before eligibility decisions; see [Step 3](#step-3--retrieve-a-broad-candidate-set).
- **Commitment:** a promise made under defined terms, not a forecast; see [mental model](#five-separations-to-preserve).
- **Compatibility:** satisfaction of product and purpose requirements; see [Step 4](#step-4--enforce-hard-compatibility).
- **Conditional probability:** probability defined given an earlier event; see [probability tree](#state-the-probability-tree).
- **Consent event:** recorded permission or withdrawal with a defined scope; see [Step 5](#step-5--resolve-permission-and-customer-terms).
- **Contribution:** revenue less stated variable costs; see [branch values](#define-the-counterfactual-and-branch-values).
- **Counterfactual:** outcome under the alternative policy; see [Step 7](#step-7--compare-incremental-economics).
- **Eligibility:** whether a candidate passes hard rules; see [Step 4](#step-4--enforce-hard-compatibility).
- **Exposure:** offer actually delivered or rendered; see [Step 9](#step-9--measure-the-entire-funnel).
- **Fill rate:** fulfilled quantity or lines divided by a defined requested denominator; see [Step 9](#step-9--measure-the-entire-funnel).
- **Forecast:** estimate of a future outcome, not a guarantee; see [worked inventory snapshot](#interpret-the-inventory-snapshot).
- **GTIN:** GS1 trade-item identifier; see [standards](#gs1-identifiers-and-product-information).
- **Guardrail:** a monitored constraint on deployment or operation; see [Exercise 6](#exercise-6--plan-a-randomized-evaluation).
- **Hard constraint:** non-negotiable eligibility condition; see [Step 4](#step-4--enforce-hard-compatibility).
- **Idempotency:** safe repetition of a command without duplicating its effect; see [Step 8](#step-8--reserve-offer-and-execute).
- **Incremental value:** value relative to a specified counterfactual; see [worked economics](#define-the-counterfactual-and-branch-values).
- **Intention-to-treat:** analysis by assigned policy, including downstream failures; see [Step 9](#step-9--measure-the-entire-funnel).
- **Pick horizon:** time between decision and physical pick; see [Step 2](#step-2--obtain-and-interpret-inventory-evidence).
- **Reservation:** inventory allocation with an explicit lifecycle; see [Step 8](#step-8--reserve-offer-and-execute).
- **Stockout:** inability to fulfill the original in the relevant context; see [Step 1](#step-1--define-the-decision-and-the-baseline).
- **TTL:** time to live before a hold expires; see [Exercise 5](#exercise-5--design-consent-and-execution-states).
- **Unknown attribute:** missing or unverified product fact, not a negative fact; see [Step 4](#step-4--enforce-hard-compatibility).

## 16. Sources

Sources are provided for verification and further study; the worked example and code are original teaching illustrations.
Company descriptions are not independent impact evaluations.
Regulatory references are educational and require jurisdiction-specific review.

1. GS1. [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin). Identifier standard and use.
2. GS1. [Global Data Synchronisation Network (GDSN)](https://www.gs1.org/standards/gdsn). Product-data exchange framework.
3. U.S. Food and Drug Administration. [Food Allergies](https://www.fda.gov/food/food-labeling-nutrition/food-allergies). Food allergen and labeling information.
4. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Official legal text.
5. California Privacy Protection Agency. [Agency website and privacy resources](https://cppa.ca.gov/). State privacy guidance and regulatory materials.
6. Instacart. [Replacing items](https://www.instacart.com/help/section/360007902831/360039572411). Company description of customer and shopper replacement options.
7. Walmart. [How AI Is Helping Walmart Make Smarter Substitutions in Online Grocery Orders](https://corporate.walmart.com/news/2021/06/14/how-ai-is-helping-walmart-make-smarter-substitutions-in-online-grocery-orders), 2021. Company account of grocery substitution.
8. Anupindi, R., Dada, M., and Gupta, S. [Estimation of Consumer Demand with Stock-Out Based Substitution: An Application to Vending Machine Products](https://doi.org/10.1287/mksc.17.4.406). *Marketing Science*, 17(4), 1998. Peer-reviewed demand and substitution study.
9. Schnabel, T., Swaminathan, A., Singh, A., Chandak, N., and Joachims, T. [Recommendations as Treatments: Debiasing Learning and Evaluation](https://proceedings.mlr.press/v48/schnabel16.html). *Proceedings of the 33rd International Conference on Machine Learning*, 2016. Treatment perspective on recommendation bias.
10. Kohavi, R., Tang, D., and Xu, Y. *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press, 2020. Controlled-experiment design and interpretation.

## 17. Links

- [Module 24 index](_index.md): topic map and prerequisites.
- [Recommendation tasks](01_recommendation-tasks.md): distinguish substitutes from related products and complements.
- [Offline and online evaluation](07_offline-online-evaluation.md): evaluation metrics and experiments.
- [Feedback loops and bias](08_feedback-loops.md): exposure and selection effects in recommendation data.
- [Module 24 survey and lab context](../../../course/24_recommendation_systems_personalization.md): broader module and lab narrative.
