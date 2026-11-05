# Disposition Economics

> Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise  
> Topic 8 of 10  
> Estimated study time: 3–4 hours  
> Prerequisites: inventory visibility and returns-network fundamentals  
> Version 2026.5

## 1. Why this topic exists

A returned item is not automatically sellable inventory.
It may be new, used, incomplete, unsafe, obsolete, contaminated, or simply in the wrong location.
A refund decision and a physical disposition decision are related but distinct.
The former concerns the customer and commercial policy; the latter determines what happens to the unit.

A returns supervisor must decide whether to send a jacket back to stock, repair a zipper, sell it through a secondary channel, donate it, or pay for destruction.
Each choice uses different people, transport, space, sales channels, and time.
A channel with a high eventual sale price can still lose value if work queues or markdowns delay the sale.
A low-value option can be the right one when safety, contract, or law rules out the others.

Finance needs a consistent decision boundary.
The original acquisition cost and the customer refund may be important for the overall order profit, but they are often already incurred or unavoidable at the disposition decision.
Charging either to one physical path and not its alternatives makes the comparison misleading.
Future processing costs, incremental recovery, and constraints belong in the choice.

Merchandising owns pricing and acceptable channels.
Operations owns feasible handling and capacity.
Quality and safety determine whether the item can be released.
Legal, compliance, and environmental teams interpret applicable obligations.
Finance validates the economic model.
A designated disposition owner resolves trade-offs and monitors outcomes.
No algorithm should override a mandatory hold or authorize an unsafe sale.

The decision is not finished when a system prints a label.
A resale item must actually be listed, available, and sold or monitored for aging.
A donation must be accepted and evidenced.
A destruction instruction requires a verified downstream outcome.
The loop from recommendation to realized recovery is essential because quotes, grades, and cycle-time forecasts can be wrong.

## 2. Learning outcomes

After this lesson you can:

1. **Classify** a returned unit and distinguish customer refund, physical ownership, and disposition status; exercise 1 checks this distinction.
2. **Compute** comparable expected incremental cash contributions for resell, refurbish, liquidate, donate, and destroy; exercises 2 and 3 apply the calculation.
3. **Exclude** infeasible alternatives before optimizing value; exercise 4 tests safety and contractual restrictions.
4. **Model** timing, sale probability, and markdown exposure without counting expected sales twice; exercises 3 and 5 test these mechanics.
5. **Allocate** constrained refurbishment capacity using opportunity cost rather than gross resale value; exercise 6 requires an explicit decision.
6. **Design** a traceable decision record and outcomes feedback loop; exercise 7 specifies required controls.
7. **Critique** recovery metrics and public case evidence without confusing revenue, contribution, and environmental impact; exercise 8 addresses both.
8. **Explain** uncertainty, ownership, and escalation in a disposition policy; the design rubrics and interview questions assess this outcome.

## 3. The core mental model

The unit is an asset with several conditional futures, not a certain sale at its original retail price.
Compare feasible futures from the same decision time, using the same cash-flow boundary.

```text
Customer return / unsold item
          |
          v
Identify + secure + establish ownership
          |
          v
Inspect, grade, safety/recall and legal checks
          |
          +---- unresolved restriction ----> HOLD and investigate
          |
          v
Build feasible actions and capacity needs
  |          |           |          |          |
Resell   Refurbish   Liquidate   Donate     Destroy
  |          |           |          |          |
  +----------+-----------+----------+----------+
                     |
                     v
Compare incremental, time-adjusted expected value
and policy priorities; authorize and execute
                     |
                     v
Track actual receipt, sale, acceptance or destruction
and reconcile to forecasts and item-level records
```

A feasibility gate comes before a financial ranking.
A recalled electrical product cannot be made eligible merely because an expected resale margin is attractive.
A no-cost theoretical donation is not a route if no recipient accepts the item.
A quoted liquidation price is not a committed sale until the contract and actual acceptance conditions are understood.
If no action is currently admissible, holding with an owner and review date is an operational state, not a final disposition.

A route is a sequence, not just a label.
For example, refurbish can lead to resale after a successful repair and to liquidation after a failed repair.
The cost of inspection, repair, freight, failed tests, and unsold inventory must appear on the relevant branches.
Expected value weights each branch by its probability and includes costs incurred before the branch.

Use an incremental comparison at a stated decision time.
Let a branch's future cash flows be `CF_t` in currency at `t` days from that point.
Then `NPV = sum(CF_t / (1 + r) ** (t / 365))`, where `r` is the effective annual discount rate.
Expected net present value is the probability-weighted sum of branch NPVs.
For a short cycle the direct effect of discounting may be small, but delay can also reduce selling prices, trigger storage fees, or miss a season; those effects must be modeled separately.
Do not count the same markdown both in the forecast proceeds and as a second cost.

Expected value is not the only objective.
A company can impose a recovery-rate target, donation commitment, landfill restriction, brand constraint, or service-level requirement.
State the rule and owner before comparing feasible paths.
If a social or environmental benefit is reported in a different unit, keep it separate unless a defensible monetization method has been approved.
Do not represent a decision-maker's preference as a legal requirement.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Disposition | Authorized physical or commercial outcome for a specific item or lot | Treating the refund as proof of physical disposition |
| Decision time | Time from which future differences are compared, recorded as timestamp | Mixing pre-return and post-inspection costs |
| Unit | Identified returned item; serial or lot where necessary | Assuming all pieces in an order have the same condition |
| Return authorization | Permission under policy to return an order line | Proof that the goods were received or are safe |
| Custody | Physical control of an item | Proof of legal title |
| Ownership | Right to direct the asset, established by contract and applicable law | Assuming a marketplace seller's goods belong to the platform |
| Chain of custody | Traceable record of transfers and responsible parties | A single scan at final destination |
| Condition grade | Evidence-based classification of condition under a versioned rubric | Treating customer-selected return reason as inspected grade |
| Inspection | Documented examination against defined criteria | Guaranteeing that a hidden defect cannot exist |
| Functional test | Test of specified functions under a documented protocol | Proof of compliance with every safety requirement |
| Hold | Temporary prohibition on movement or release pending review | A final route without owner or expiry |
| Recall | Safety or corrective action identified by an authorized source | A blanket assumption about permitted resale |
| Resell | Sell an eligible item through an approved channel | Assuming the item qualifies as new |
| Open-box | Channel or condition description governed by seller policy and law | Universal legal grade with identical meaning everywhere |
| Refurbish | Inspect, repair, clean, replace, and test as defined by the program | Merely repackaging a product |
| Repair yield | Successful repairs divided by attempted repairs, same cohort | Successful repairs divided only by completed repairs |
| Liquidate | Sell through an approved secondary buyer or channel, generally at a discount | Equating buyer's bid with net cash received |
| Donation | Transfer to an eligible, willing recipient without ordinary sales consideration | Assuming a tax deduction or zero handling cost |
| Destruction | Controlled physical disposal or other prescribed treatment, evidenced | Assuming every discarded item reaches the intended facility |
| Recycling | Processing material for recovery under an appropriate route | Treating recycling as equivalent to reuse or guaranteed recovery |
| Recovery proceeds | Incremental cash expected or realized from the route, currency per unit | Gross retail price or refunded amount |
| Incremental cost | Future cost that changes because of the action, currency per unit | Allocating unchanged sunk overhead as a route-specific cost |
| Sunk cost | Cost already incurred and unaffected by the choice | Treating acquisition cost as avoidable at a post-return decision |
| Avoidable cost | Future cost eliminated if an action is not chosen | All accounting overhead |
| Contribution | Revenue less specified costs, currency per unit | Calling gross proceeds profit |
| Net present value (NPV) | Discounted sum of future net cash flows, currency at decision time | Discounting both nominal and already discounted amounts |
| Sale probability | Probability of a sale in the specified horizon | A forecast price rather than likelihood |
| Conditional sale price | Price given sale, currency per sold unit | Multiplying an already probability-weighted price by probability again |
| Markdown | Price reduction relative to a stated baseline, currency or percent | Adding a markdown cost after using the reduced proceeds |
| Time to disposition | Elapsed time from defined receipt or decision event to verified final event, days | Measuring only the label-print event |
| Opportunity cost | Value of the best feasible alternative displaced by a decision | An arbitrary penalty independent of a scarce resource |
| Capacity | Available units of labor, space, transport, or buyer intake per time bucket | A static number without a period |
| Shadow price | Marginal value of one additional unit of a binding resource, currency per resource unit | A universal price valid outside the modeled range |
| Realized recovery | Actual proceeds less defined actual route costs for a mature cohort | Summing eventual sales only and ignoring unsold items |
| Cohort | Items sharing a defined receipt period and segmentation | Mixing recent unresolved returns with mature outcomes without adjustment |
| Censoring | Outcomes not yet observed by the measurement cutoff | Counting every open item as zero recovery or every open item as a future sale |
| Policy exception | Approved departure from a rule with rationale and authority | An undocumented override |
| Displacement | A recovered sale or donation replacing an activity that would otherwise occur | Assuming every recovered unit creates a new sale or benefit |

## 5. The mechanics, step by step

### 5.1 Establish the decision boundary

**Input:** return identifier, receipt event, order line, title terms, refund status, and decision timestamp.

**Action:** identify the exact object and who may direct it.
Decide whether the calculation starts at return authorization, physical receipt, or after inspection.
This lesson uses the post-inspection decision point in its numerical case.
The choice matters because costs incurred before that point are no longer avoidable.

**Output:** a documented decision unit and time boundary.

**Branch:** if identity, title, or custody cannot be established, hold and investigate rather than presume the goods are available.
If an unrefunded customer can still reject an offered remedy, coordinate with customer service before transferring the item.
The disposition owner is responsible for resolving the hold, while legal interprets ambiguous ownership terms.

### 5.2 Inspect and grade consistently

**Input:** item, product attributes, inspection rubric, recall data, and observed evidence.

Record seal status, visible condition, missing parts, functional test results, packaging, and any hazard indicator appropriate to the product.
A camera photo or test result should support the grade, not substitute for product-specific inspection rules.
Record the rubric version and inspector or automated rule version.

Use customer-reported return reason as a lead, not the definitive condition grade.
A customer's claim of defect may be correct even if a quick test passes.
A no-fault-found result is bounded by the tests performed.
Where an item contains personal information, ensure an approved data-erasure process before any transfer.
For batteries, chemicals, food, cosmetics, and medical products, use product-specific handling controls and escalate uncertainty.

**Output:** a grade and evidence attached to the unique item or traceable lot.

**Branch:** unidentified contamination, a potential recall, or failed safety test leads to controlled hold until the authorized expert decides the permitted routes.
Sample and audit grading across sites to measure disagreement and identify drift.

### 5.3 Construct the feasible action set

**Input:** grade, product restrictions, applicable law, commercial agreements, approved buyer and recipient lists, and available capacity.

Create candidate routes: direct resale, refurbish then resale, liquidation, donation, and destruction.
A route may be unavailable because of a product restriction, recall remedy, data security requirement, brand agreement, buyer acceptance rule, lack of recipient capacity, or missing repair capability.
A safe and lawful recycle route can be considered separately where available.

Record both the rule and the version that excluded a route.
Do not hard-code all products into one universal hierarchy.
For instance, a used cosmetic may require a different assessment from a sealed sweater.
A manufacturer return agreement may require return to vendor instead of any of the five local routes.
If applicable, include return to vendor as another feasible route with its own economics rather than forcing the item into an inappropriate category.

**Output:** feasible route list and explicit exclusions.

**Branch:** if no terminal route is currently authorized, hold, assign ownership, and set the next review date.
A hold has storage and risk costs, but it is not permission to choose an unlawful route.
Quality and compliance own eligibility; commercial teams own channel authorization.

### 5.4 Map the future cash-flow branches

**Input:** route recipes, service-provider rates, labor standards, test yields, sell-through forecasts, and fallback contracts.

For each feasible route, list every incremental future receipt and payment with timing and probability.
Separate an unconditional upfront cost from costs incurred only if a repair succeeds or a sale occurs.
Add fallback routes for failure and unsold inventory if they are feasible and priced.
Use a finite horizon and a terminal value, or explicitly model all remaining outcomes.

Typical direct resale components are cleaning, repackaging, listing, expected net sale proceeds, outbound shipping, marketplace fees, and unsold fallback.
Typical refurbish components include repair labor, parts, retest, failure fallback, and the later sale costs.
Typical liquidation components include buyer payment, sorting, palletization, freight, commissions, rejection probability, and payment delay.
Donation can require inspection, packing, transfer, recipient coordination, and evidence retention.
Destruction can require safe handling, transport, treatment, and certified evidence.

A customer refund is common across routes only if the decision truly cannot change it.
If one route changes a refund, warranty liability, or contractual chargeback, include the difference.
Similarly, a cost already incurred is sunk, but a future storage contract cancellation or variable facility charge can be avoidable.
Include avoided future disposal only once, by comparing complete alternatives rather than adding an arbitrary credit to both.

**Output:** branch cash-flow table in currency per item, dates relative to decision, and source for each assumption.

**Branch:** when an input is unknown, use a justified range and show sensitivity; do not silently insert zero.
Finance approves the economic boundary and treatment of shared costs.
Operations validates labor and physical flows; commercial teams validate channel economics.

### 5.5 Estimate demand and recovery

**Input:** historical item-level outcomes, grade, season, channel, age, price, and observation cutoff.

Estimate probability of sale over a specified horizon and conditional net proceeds given a sale.
If historical revenue is already averaged across all listed items, it is unconditional; do not multiply it by sell-through again.
Calculate fees and shipping at the appropriate stage, recognizing that some are charged only on sale.
A sale may be reversed, returned again, or charged back; include the expected incremental effect when material and observable.

Use similar cohorts, but do not assume last season's price, brand, or demand is stable.
Separate high-quality grade data from unverified classifications.
Account for unsold units, price changes, buyer rejections, and open cases.
Right-censored recent cohorts need sufficient maturity or an explicit time-to-event method.
A high average among sold items can coexist with poor sell-through.

**Output:** conditional branch probabilities, prices, costs, and uncertainty with source dates.

**Branch:** if sample sizes are small or selection is strong, use wider ranges and require human review for high-value decisions.
A randomized or carefully controlled pilot can improve estimates, subject to safety and commercial constraints.

### 5.6 Price time and capacity

**Input:** forecast cycle times, storage costs, seasonal dates, available work hours, and route-specific price curves.

Apply a consistent annual discount rate to dated cash flows.
Model calendar effects separately: a coat may command less after winter, while an electronics model may lose value after a launch.
Apply warehouse rent only if it changes with the decision, such as an avoidable overflow lease or per-pallet storage charge.
A booked building cost that will not change is not a marginal cost merely because accounting allocates it per unit.

Capacity must use a compatible time bucket.
A repair path needing 30 minutes cannot be scheduled against an unqualified statement that the shop has capacity for 100 units.
Account for staffing, skill, parts, space, vendor collection slots, and deadlines.
If capacity is scarce, compare each candidate's incremental advantage over its best non-scarce fallback per unit of the binding resource, not its gross expected proceeds.
With several constraints or indivisible batches, solve a constrained portfolio problem; a single ratio can fail.

**Output:** dated, capacity-aware candidate values and resource requirements.

**Branch:** when a queue makes the original estimate stale, reprice the item or escalate rather than assume the original ranking still holds.
Operations owns capacity and schedule; finance owns the approved valuation convention.

### 5.7 Choose, authorize, and communicate

**Input:** feasible route values, confidence ranges, capacity, policy objectives, and approval thresholds.

Choose the highest expected incremental value if the stated objective is financial and all other rules are satisfied.
If values are close relative to uncertainty, a simpler or faster route may be preferable under a documented tie policy.
If a policy deliberately favors donation or repair over a higher cash alternative, calculate the difference and identify the decision owner.
Do not label a socially motivated choice as financially optimal unless the relevant monetization is explicitly included.

Assign an executor, route, destination, deadline, grade, and audit trail.
Capture model version and input values at decision time rather than overwriting history after outcomes arrive.
An override should state the reason, approver, and applicable authority.
A high-value exception or a potential safety issue should route to the appropriate owner, not disappear into an automatic queue.

**Output:** authorized disposition order and reason code.

**Branch:** unresolved restriction or uncertain identity returns to hold.
A path that needs a buyer or recipient remains conditional until accepted.

### 5.8 Execute and close the loop

**Input:** authorized order, custody events, inventory record, sale or transfer events, invoices, and final evidence.

Scan movements through collection, warehouse, repair, carrier, and destination.
Reconcile item counts and serials where appropriate.
On resale, update the available-to-promise record only after the item passes release criteria and is physically available in the correct node.
On donation, obtain recipient acceptance and evidence of transfer.
On destruction, retain the provider's relevant certificate or other evidence and reconcile quantities; a certificate is not automatically proof of every downstream material outcome.

Close the item only when the defined terminal event occurs.
Record actual price, fees, transport, labor, repair result, cycle time, and failed or reversed transfers.
Compare forecasts with mature cohorts and calibrate both probability and magnitude.
Investigate whether policies systematically route the easiest items to one channel, creating selection bias in comparisons.

**Output:** closed item record, actual economics, exceptions, and learning feedback.

**Branch:** missing confirmation, rejected donation, failed repair, or buyer refusal triggers an authorized fallback or a new decision.
Finance reconciles cash; operations reconciles quantity; quality investigates safety failures.

### 5.9 Direct resale: execute the commercial promise

**Input:** an eligible unit, its grade and accessories, an approved sales channel, and expected demand.
The unit must be physically present at the node that will fulfill the next order.
The release event should identify the precise condition offered.
For example, a sealed-box grade and an open-box grade should not share a listing if the customer is led to expect identical condition.
The pricing owner sets the permissible price and markdown schedule.
The quality owner defines what evidence supports the grade.

The sale forecast must be specific to that price and channel.
A lower price can improve sell-through but reduce proceeds per sale.
A longer listing window may increase the cumulative probability of sale while consuming space and delaying the fallback.
Estimate the combined effect rather than applying an arbitrary percentage to all products.
Use net proceeds after only the costs already included in the forecast definition.
If outbound shipping is paid only on sale, it belongs inside the sale branch.
If listing labor is incurred whether or not the unit sells, it belongs before the sale/no-sale split.

Suppose a unit costs USD 3 to list.
It has a 50% chance of a USD 40 sale and a 50% chance of USD 8 fallback.
If the USD 40 is already net of USD 5 shipping, expected contribution is `-3 + 0.5 × 40 + 0.5 × 8 = USD 21`.
Subtracting USD 5 shipping again would understate this contribution by USD 2.50.
If the USD 40 were gross instead, the sale branch would be USD 35 and the correct contribution would be USD 18.50.
The distinction comes from the data definition, not a different mathematical formula.

Customer expectations also affect downstream economics.
An item described incorrectly can be returned again, generate a chargeback, or damage trust.
Where historical data supports it, include the expected cost of a subsequent return in the sale branch.
Keep the physical inspection standard and customer-facing condition language aligned.
Do not treat a second customer refund as impossible merely because the original return was processed.

Set a listing-age checkpoint.
At that checkpoint, compare the remaining expected proceeds with the best current fallback.
The original listing cost is then sunk.
Do not retain an item only because money was already spent listing it.
Similarly, do not liquidate merely to meet a clock if waiting has higher expected incremental value and no binding policy deadline applies.
The owner should document how age and condition change the recommendation.

### 5.10 Refurbishment: manage yield and quality

**Input:** an eligible repair candidate, fault evidence, approved parts, technician skill, and a verified test procedure.
Refurbishment is a production process with work-in-process inventory, queues, and rework.
Its output is not simply “fixed”; it is an item that passes defined release criteria.
Separate diagnosis, repair, final testing, cosmetic preparation, and listing.
Each step may consume capacity or discover a defect that changes the route.

Define the repair yield denominator before comparing partners.
For example, 80 successful repairs from 100 attempted units gives an 80% first-pass success rate if every attempt has reached a decision.
If 20 units remain unfinished, reporting 80 successes out of 80 completed units as a 100% success rate conceals unfinished work.
Report completion and success separately.
Also define whether repeat repairs count as a second attempt or as rework on the same unit.

A failed repair can still have recoverable value.
The fallback may be parts harvesting, liquidation, recycling, or required treatment, subject to applicable rules.
It can also have a negative value if a dismantled item costs more to handle.
Use the actual conditional fallback for the failure branch rather than assuming zero.
If the original fallback is no longer available after a repair attempt, model the opportunity lost.

Consider a candidate with USD 12 incremental repair cost.
A successful repair has a 75% probability and yields USD 52 net after sale and further handling.
A failed repair yields USD 4 net.
Ignoring timing, expected repair contribution is `-12 + 0.75 × 52 + 0.25 × 4 = USD 28`.
If direct liquidation contributes USD 25, repair adds USD 3 of expected value before considering constrained hours.
If repair requires one scarce technician hour that can add USD 5 elsewhere, allocating that hour to this unit is not the best portfolio choice.

Repaired inventory may require different warranty terms, labels, or consumer disclosures from new inventory.
The permitted wording depends on the product and jurisdiction.
The quality and legal owners approve the release standard and claims.
The economic analyst should not substitute a predicted value for that approval.
After sale, monitor defects and repeat returns by repair partner, part, technician process, and test version.
Include any material incremental warranty expense in subsequent forecasts.

### 5.11 Liquidation: convert the quote into realizable cash

**Input:** buyer contract, eligible assortment, grading rules, expected rejection, freight, fees, and payment terms.
A buyer may quote a percentage of original retail, a fixed amount per unit, or a price per mixed pallet.
These are different commercial structures.
The quoted base must be defined: original list price, current price, cost, or appraised value.
Neither a bid nor a signed schedule guarantees that each submitted item will be accepted.

Read the contract for minimum lot size, permitted categories, prohibited products, audit rights, and geographic restrictions.
Check whether the buyer can regrade goods after receipt.
Record the time allowed to contest a downgrade and who pays return freight.
Check commission, deductions, payment timing, and credit risk.
Where a buyer pays on resale rather than purchase, there is additional uncertainty in both timing and amount.
If goods are consigned, clarify title and the allocation of unsold inventory.

Suppose a liquidator quotes USD 30 per accepted unit.
There is a 90% probability of acceptance and a 10% probability that a rejected unit yields USD 6 net after a fallback.
Packing and outbound freight cost USD 4 per submitted unit regardless of acceptance.
Ignoring timing, expected value is `-4 + 0.90 × 30 + 0.10 × 6 = USD 23.60`.
Using `30 - 4 = USD 26` assumes every unit is accepted.
If the USD 6 fallback excludes return freight, add that freight to the rejection branch.
If it already includes return freight, do not subtract it twice.

For mixed lots, allocate buyer receipts using a consistent method and retain the unallocated total.
An allocation by unit, weight, original retail, or appraised value answers different questions.
No allocation creates item-level prices that the buyer actually agreed to pay.
Use item-level attribution for learning only to the extent that the source supports it.
Reconcile the lot settlement and track buyer regrades by product and source location.

### 5.12 Donation: confirm demand and account for costs

**Input:** an eligible item, recipient acceptance, transfer terms, costs, and any approved reporting requirements.
Donation is a transfer, not an automatic escape from the return process.
Ask whether the recipient can use the category, condition, quantity, and delivery date.
A large unsolicited shipment may burden a nonprofit with sorting and disposal costs.
Commercial, community-relations, or sustainability staff may own the relationship, while operations owns the physical transfer.

Identify who pays to pack, move, unload, and dispose of rejects.
If the recipient requires a grade, provide a defensible grade and truthful product information.
If goods contain identifying customer data, remove it under an approved process before transfer.
Retain acceptance, quantities, dates, and any other records required by the organization and applicable tax rules.
Do not report donation as completed merely because a truck departed.

The financial model starts with incremental costs and any defensible incremental cash effects.
Suppose packing costs USD 3, transport costs USD 5, and no cash benefit is assumed.
The direct expected contribution is USD -8 per accepted item, ignoring timing.
If destruction would instead cost USD 14, donation improves cash contribution by USD 6 relative to that fallback.
It is still not the same thing as generating USD 6 in sales.
Its broader social value may be important, but it needs a separately stated measurement approach.

If a tax expert determines an incremental deduction is available, model the resulting tax cash effect and its timing rather than treating the deduction itself as cash.
For illustration only, a USD 10 additional deduction at a 20% marginal tax rate could produce a USD 2 tax cash effect if all relevant assumptions hold.
This is not advice that any particular donation qualifies or that a 20% rate applies.
Limits, entity type, valuation, and usable taxable income can change the result.
Keep tax evidence and valuation review with qualified owners.

### 5.13 Destruction and material recovery: close the chain

**Input:** product classification, approved handling instructions, treatment provider, cost, and evidence requirements.
Destruction is sometimes a required control rather than an economic failure.
An unsafe, contaminated, counterfeit, or legally restricted item may have no permitted reuse path.
The correct treatment method depends on the product and applicable rules.
The disposition owner must not invent a treatment classification from a generic category label.

Differentiate the instruction to destroy from physical receipt by the provider.
Differentiate provider receipt from evidence that the specified treatment occurred.
Some providers issue certificates covering a batch rather than each item.
Reconcile the certificate's batch, quantity, date, and site with shipment records.
Investigate shortages, substitutions, or unexplained differences.
Require a documented escalation path when evidence is overdue.

Material recovery may produce a payment, a fee, or both.
Do not assume a recycling claim means every component was recycled.
Freight, sorting, contamination, and commodity prices can make the net cash outcome negative.
If a processor's payment depends on recovered material yield, model yield and price uncertainty.
Avoid double counting recovered-material revenue both as a buyer credit and as a separate sales receipt.

Suppose treatment costs USD 9 and certified transport costs USD 4 per item.
The route contribution is USD -13 before any other incremental charges.
If the only permitted alternative is a USD -18 treatment route with comparable compliance, the first route saves USD 5.
If the cheaper route is not authorized for that material, its price is irrelevant.
Environmental or safety requirements can make a financially negative route mandatory.

### 5.14 Compare decisions at multiple levels

The per-item choice, the constrained portfolio, and the accounting report answer different questions.
Do not force one number to answer all three.
At item level, compare future incremental cash flows among feasible routes.
At portfolio level, include shared bottlenecks, batch size, and deadlines.
In financial reporting, apply the organization's accounting policy to revenue, inventory, expense, and impairment.
Accounting recognition can differ from the operational decision model without either necessarily being wrong.

Consider 100 items with identical expected route values.
Resale adds USD 8 per item compared with liquidation and needs 10 minutes of listing labor.
Only 500 listing minutes are available before the selling season ends.
At most 50 items can be listed under these assumptions.
The resulting incremental advantage is `50 × 8 = USD 400`, not `100 × 8 = USD 800`.
The other 50 items use the best feasible fallback.
If listing capacity can be expanded at a cost below the value of additional listings, examine that investment separately.

If items have different incremental advantages, rank by benefit per minute only when listing time is the sole binding resource and the relevant divisible or simple allocation assumptions hold.
When item batches are indivisible, a ratio can leave unusable capacity.
When space, labor skill, deadlines, and buyer quotas bind simultaneously, use a constrained optimization model or explicit scenario comparison.
The model should allow a hold or mandatory route where relevant, not require every item to be sold.
An operations owner must validate that the chosen schedule is executable.

### 5.15 Establish service levels and aging controls

Timeliness is measured against a clearly defined start and terminal event.
Receipt-to-inspection, inspection-to-decision, decision-to-transfer, and transfer-to-confirmation are different intervals.
A single average can hide a growing tail of stuck items.
Monitor the median, a high percentile, open backlog by age, and the share of items over their review date.
Segment by product, route, location, and reason for hold when volumes support it.

Use aging thresholds as review triggers rather than unexamined destruction rules.
A 14-day review threshold is a management policy, not automatically a legal disposal deadline or customer commitment.
At review, refresh the item condition, current demand, buyer offer, and feasible alternatives.
Decide whether to continue, reprice, route to fallback, or escalate.
Capture the owner and next review time for any continued hold.

A processing target should not encourage unsafe shortcuts.
If staff are rewarded only for rapid closure, they may close on shipment rather than on verified outcome.
Pair timeliness measures with grade accuracy, recall compliance, recipient acceptance, and evidence completeness.
Review the number of unconfirmed transfers and reversals.
Use root-cause analysis on aging and exceptions rather than merely moving the oldest cases into a different queue.

### 5.16 Design decision rights and an operating cadence

The item decision needs a single accountable owner even when many teams contribute.
The safety team owns the interpretation of technical release controls.
Legal and compliance interpret obligations and contractual restrictions.
Merchandising approves brands, channel claims, price, and markdowns.
Operations confirms custody, capacity, and execution.
Finance approves valuation conventions and reconciles economic outcomes.
Tax specialists determine whether any tax effect is supportable.

At shift level, the supervisor reviews ungraded receipts, safety holds, overdue movements, and rejected transfers.
At weekly level, operations and merchandising review capacity, age, forecasts, and buyer acceptance.
At monthly level, finance and the disposition owner compare expected and realized contribution for sufficiently mature cohorts.
At periodic policy review, quality and compliance assess recall controls, grades, overrides, and vendor performance.
Escalate material safety concerns immediately rather than waiting for a routine meeting.

Define approval thresholds before employees face an exception.
A routine price override might require a merchandising manager; a safety hold release requires the designated quality authority.
An economic advantage does not grant authority to waive a safety control.
Preserve the original recommendation, override reason, approver, and subsequent outcome.
Audit patterns across sites and partners, not only individual exceptions.

### 5.17 Measure performance without gaming the metric

Define a recovery metric with an explicit numerator and denominator.
One useful operational measure is realized incremental route contribution per received item for a mature cohort.
Its numerator includes proceeds minus defined incremental route costs.
Its denominator includes all eligible items in the cohort, including items that did not sell.
Report the share still open at cutoff alongside the metric.
Otherwise recent cohorts may look artificially weak or artificially strong.

A gross recovery ratio can also be useful if clearly labeled.
For example, gross resale proceeds divided by original retail value describes a price relationship.
It does not subtract labor or freight, establish profit, or measure the alternative route.
Use a consistent retail-price definition and avoid combining original and current prices.
Report the units and period for every ratio.

Track decision quality with forecast calibration.
Group items predicted to sell at approximately 60% probability and compare their mature observed sale rate with 60%.
Also compare predicted conditional net proceeds with actual proceeds among sales.
Calibration of probability does not guarantee calibration of price, repair cost, or time.
Use product and channel segments to locate the source of error, while protecting against unstable small samples.

Compare route performance fairly.
If repair receives the easiest units, its higher observed recovery may reflect selection rather than superior treatment.
Record the inputs used in the decision and the feasible alternatives at the time.
Where ethical, safe, and commercially acceptable, use a randomized comparison among comparable eligible units.
Otherwise consider matched analysis and explain remaining confounding.
Never randomize an item into a prohibited or unsafe route.

### 5.18 Reconcile forecasts, plans, targets, and commitments

A **forecast** estimates what may happen, such as 70% sell-through within 20 days.
A **plan** specifies actions and resources, such as listing 50 eligible units this week.
A **target** expresses a management goal, such as a 10-day median decision cycle.
A **commitment** is a promise to a customer, partner, or regulator when the relevant agreement or rule creates one.
These are not interchangeable.

Do not turn an expected sale date into a promised delivery date.
Do not treat a partner's unaccepted quote as a payment commitment.
Do not characterize a target as a historical result.
Preserve the version and date of each forecast and policy target.
After outcomes mature, explain differences between forecast, actual execution, and external changes.
Reforecast the remaining open inventory instead of retroactively rewriting the original forecast.

### 5.19 Handle uncertainty and exceptional cases

Point estimates hide uncertainty in grade, demand, price, processing time, and buyer performance.
Build low, base, and high scenarios from defensible assumptions.
Change correlated inputs together when appropriate: a demand downturn may lower both sell-through and price.
An independent variation of each input can understate the chance that several unfavorable outcomes occur together.
For material decisions, record the range of route rankings, not just the base-case winner.

If resale is expected to exceed liquidation by USD 1 but the credible error in each estimate is much larger, the ranking is fragile.
The owner may prefer a faster or less operationally complex route under an approved tie policy.
If a policy explicitly values donation, calculate the cash difference relative to the best feasible financial alternative.
This exposes the trade-off without claiming the noncash benefit has a verified dollar value.
Record who accepted that trade-off and under what authority.

Unknown inputs are not zero.
If an item may contain hazardous material and the cost of compliant treatment is unknown, obtain classification and a provider quote.
If a buyer's rejection terms are unclear, review the contract or request terms.
If the recipient has not accepted the goods, the donation route remains conditional.
If a key assumption cannot be resolved promptly, hold with an owner, deadline, and an estimate of the cost of delay.

### 5.20 Implement a small pilot responsibly

Choose a well-defined product category with an established safety and ownership process.
Specify the population, time horizon, comparison routes, and exclusion criteria before observing results.
Assign a responsible analyst, operations lead, quality approver, and commercial owner.
Define what decision the pilot is intended to inform: route value, repair yield, cycle time, or another measurable quantity.
Do not claim that one category validates all future categories.

Collect every eligible item's initial grade and feasible set.
Track both accepted and rejected items, including those that remain unsold.
Capture the forecast made before the outcome and the actual cash flows after the outcome.
Where permissible, allocate comparable units between approved routes using a documented method.
Monitor safety, customer complaints, and exception rates throughout.
Stop or modify the pilot if a safety issue or unauthorized route appears.

At the end, calculate contribution and cycle time by route, with uncertainty and unresolved-item counts.
Explain any imbalance in item quality or eligibility between groups.
Compare the results to the predeclared decision rule rather than changing the criterion after seeing the best result.
If the sample is too small or follow-up is too short, state what is not yet known.
An inconclusive result is better than a precise but unsupported recovery claim.

## 6. Worked example

### 6.1 Situation and scope

A retailer receives a returned premium countertop appliance.
Inspection establishes that it is owned by the retailer, has no identified recall or hazard, and is eligible for all five routes under the retailer's documented policy.
The item has cosmetic damage but passes the initial electrical safety screen.
This is a hypothetical teaching case, not a claim about an actual retailer or product.

At the decision timestamp, the customer has already been refunded and the item inspected.
The original purchase cost, refund, and completed inspection cost are identical across the five choices and cannot be recovered through the choice.
They are omitted from incremental comparison, not from the full order profit-and-loss statement.
Currency is U.S. dollars (USD) per returned item.
Assume a 10% effective annual discount rate and a 365-day year.
All stated price figures are net of sale-specific marketplace commission and outbound delivery unless noted.

This is a **forecast** of outcomes, not a committed buyer payment.
The operational **plan** is to execute the selected route immediately if repair capacity is available.
A 14-day cycle is a planning assumption, not a customer-facing promise or a service commitment.
There is no donation or environmental target in the financial objective for this example.

### 6.2 Direct resale route

The retailer can clean and list the item for USD 8 now.
There is a 70% chance of a sale at USD 64 net proceeds in 20 days.
If it does not sell, there is a 30% chance of liquidation at USD 18 net proceeds in 35 days.
The USD 18 is conditional on the no-sale branch and includes the liquidation handling cost in this route.

The expected undiscounted contribution is:

```text
-8 + 0.70 × 64 + 0.30 × 18
= -8 + 44.80 + 5.40
= USD 42.20 per item
```

The discounted estimate is:

```text
-8 + 0.70 × 64 / 1.10^(20/365)
   + 0.30 × 18 / 1.10^(35/365)
≈ -8 + 44.57 + 5.35
≈ USD 41.92 per item
```

### 6.3 Refurbish route

Repair and retest cost USD 16 now regardless of outcome.
The repair succeeds with 80% probability.
If it succeeds, there is an additional USD 4 listing cost at day 7 and a 75% chance of sale for USD 82 net at day 30.
If the repaired item does not sell, its net liquidation proceeds are USD 22 at day 45.
If repair fails, it can be liquidated for USD 12 net at day 14.
These outcomes are mutually exclusive and exhaustive for this simplified case.

The undiscounted expected contribution is:

```text
-16 + 0.80 × (-4 + 0.75 × 82 + 0.25 × 22)
    + 0.20 × 12
= -16 + 0.80 × 63 + 2.40
= USD 36.80 per item
```

The discounted contribution is:

```text
-16 + 0.80 × (-4 / 1.10^(7/365)
              + 0.75 × 82 / 1.10^(30/365)
              + 0.25 × 22 / 1.10^(45/365))
    + 0.20 × 12 / 1.10^(14/365)
≈ USD 36.36 per item
```

A successful repair has higher conditional sale proceeds than direct resale, but repair cost, failure, sell-through, and time make the expected contribution lower in this case.
The USD 4 listing cost occurs only on the successful repair branch.
Do not charge it again inside the net proceeds.

### 6.4 Other routes

Liquidation costs USD 5 now to sort and ship and has a certain USD 31 payment at day 14 under the case contract.
Donation costs USD 7 now to prepare and USD 3 at day 7 to transfer; an approved recipient has agreed to accept the item.
No tax benefit is assumed, and no cash proceeds are attributed to the donation.
Destruction costs USD 12 now for handling and USD 6 at day 7 for an approved treatment provider.
A destruction certificate is required for closure.

```text
Liquidate = -5 + 31 / 1.10^(14/365) ≈ USD 25.89
Donate    = -7 - 3 / 1.10^(7/365)   ≈ USD -9.99
Destroy   = -12 - 6 / 1.10^(7/365)  ≈ USD -17.99
```

These negative values do not mean donation or destruction is undesirable in all circumstances.
They mean cash outflow exceeds modeled cash inflow at this boundary.
Noncash benefits, required safety outcomes, contractual restrictions, and policy preferences are separate considerations.

### 6.5 Decision and sensitivity

| Feasible route | Expected NPV, USD per item | Decision note |
|---|---:|---|
| Direct resale | 41.92 | Highest modeled value |
| Refurbish | 36.36 | Higher potential price but lower expected contribution |
| Liquidate now | 25.89 | Faster, lower but contractually certain payment in this case |
| Donate | -9.99 | Accepted recipient; no monetized tax or social benefit |
| Destroy | -17.99 | Approved treatment and evidence required |

Choose direct resale under the stated financial objective.
The advantage over refurbish is approximately USD 5.56 per item, before rounding.
The advantage over immediate liquidation is approximately USD 16.03 per item.
These are expected values, not guarantees for an individual item.

Ignoring discounting for a quick threshold calculation, direct resale value as a function of sale probability `p` is:

```text
-8 + p × 64 + (1 - p) × 18 = 10 + 46p
```

It equals the undiscounted refurbish contribution of USD 36.80 when `p = 26.80 / 46 ≈ 0.583`.
Below about 58.3% sell-through, refurbish beats direct resale on this simplified undiscounted comparison, holding all other assumptions constant.
The exact discounted threshold differs because the branches occur at different dates.
This is not a forecast that sell-through will be 58.3%; it is a break-even assumption.

A 30-day listing delay can also affect price, sell-through, storage, and availability.
Do not model only the financial discount and claim to have captured all seasonality.
Re-estimate net proceeds and fallback timing if the season changes.
If a recall is discovered after the calculation, remove prohibited routes and follow the authorized recall remedy; the arithmetic does not grant an exception.

## 7. Code, query, or calculation

The following Python 3 program reproduces the case using only the standard library.
Each branch contains a probability and dated cash flows in USD per item.
Common costs appear in every branch where incurred; this makes the weighting explicit.
The probabilities for each route sum to one.
The program also calculates the undiscounted break-even sell-through threshold.

```python
from math import isclose

RATE = 0.10  # Effective annual rate, not a monthly rate.

# Each tuple is (branch probability, [(day, cash flow USD), ...]).
# All amounts and probabilities refer to one item at the decision time.
routes = {
    "Direct resale": [
        (0.70, [(0, -8), (20, 64)]),
        (0.30, [(0, -8), (35, 18)]),
    ],
    "Refurbish": [
        (0.80 * 0.75, [(0, -16), (7, -4), (30, 82)]),
        (0.80 * 0.25, [(0, -16), (7, -4), (45, 22)]),
        (0.20, [(0, -16), (14, 12)]),
    ],
    "Liquidate": [(1.0, [(0, -5), (14, 31)])],
    "Donate": [(1.0, [(0, -7), (7, -3)])],
    "Destroy": [(1.0, [(0, -12), (7, -6)])],
}


def npv(flows, annual_rate=RATE):
    """Discount a sequence of (days after decision, USD) cash flows."""
    return sum(amount / (1 + annual_rate) ** (day / 365)
               for day, amount in flows)


def expected_npv(branches):
    probability = sum(p for p, _ in branches)
    if not isclose(probability, 1.0, abs_tol=1e-12):
        raise ValueError(f"branch probabilities sum to {probability}")
    if any(p < 0 or p > 1 for p, _ in branches):
        raise ValueError("invalid branch probability")
    return sum(p * npv(flows) for p, flows in branches)


values = {name: expected_npv(branches)
          for name, branches in routes.items()}
for name, value in sorted(values.items(), key=lambda x: -x[1]):
    print(f"{name:14s} USD {value:7.2f}")

# For the undiscounted comparison: 10 + 46*p = 36.8.
threshold = (36.8 - 10) / 46
print(f"Undiscounted direct-sale threshold: {threshold:.1%}")
```

Expected output, allowing a cent for display rounding:

```text
Direct resale  USD   41.92
Refurbish      USD   36.36
Liquidate      USD   25.89
Donate         USD   -9.99
Destroy        USD  -17.99
Undiscounted direct-sale threshold: 58.3%
```

The program compares only the alternatives already judged feasible.
It is not a safety classifier or an optimizer over scarce repair capacity.
It assumes independent future cash flows as specified by each mutually exclusive branch, a fixed discount rate, no taxes, and no unmodeled salvage beyond the stated horizon.
The no-sale liquidation values are net proceeds; adding another liquidation cost would double count it.
In production, route configuration, probabilities, and cash flows should come from versioned governed data rather than hard-coded constants.
Validate date ordering, eligibility, currency, branch completeness, and input provenance before using a ranking operationally.

## 8. Data, standards, and regulatory context

### 8.1 Minimum data contract

An auditable item record needs an item or traceable lot identifier, order-line reference where applicable, product identifier, location, receipt timestamp, owner, custody status, and current disposition state.
For serialized products retain serial identifiers as needed, with access controls appropriate to the product and jurisdiction.
Record grade, inspection evidence, safety hold, applicable rule version, and disposition authorization.
Capture the route, actor, recipient or buyer, planned and actual dates, and terminal evidence.
Separate the timestamp at which a fact became known from the timestamp of the underlying event.
Late-arriving scans otherwise produce misleading cycle-time and model-training data.

Use a cash-flow ledger tied to the item or defensible allocation cohort.
Store gross proceeds, fees, freight, handling, parts, and refunds as separate line types with currencies and dates.
A buyer's remittance may cover a mixed lot; maintain an explicit allocation key and reconciliation total rather than inventing precise item-level proceeds.
Maintain customer information only where necessary and apply approved retention and access rules.

### 8.2 Identifiers and event standards

GS1's Global Trade Item Number (GTIN) identifies a trade item class, not necessarily an individual physical unit.
Serial numbers, lot identifiers, and location identifiers may be required for item-level or batch-level traceability.
GS1 Electronic Product Code Information Services (EPCIS) provides a standard structure for visibility events describing what, when, where, why, and how.
It can carry events needed for custody and process visibility, but using EPCIS does not itself prove ownership, condition, payment, or lawful disposition.
Implementers must agree on event semantics, identifiers, and trading-partner responsibilities.

Maintain separate states for `received`, `held`, `graded`, `authorized`, `in transit`, `accepted`, and `closed` or equivalent local codes.
A label print is not equivalent to a physical scan, and a warehouse shipment is not equivalent to recipient acceptance.
Include corrections as new traceable events, not silent replacement of historical facts.
Reconcile exceptions between the warehouse management system, repair partner, point of sale, and finance ledger.

### 8.3 Product safety and recalls

In the United States, the Consumer Product Safety Commission (CPSC) publishes recall information and explains requirements concerning recalled consumer products.
The applicable obligations depend on the product, recall, entity, and jurisdiction.
Consult the actual recall notice and qualified counsel; a generic lesson cannot determine whether a particular unit may be repaired, transferred, or sold.
A recall gate needs a current data source, identifier matching, review of ambiguous matches, and a process for new recalls discovered after initial grading.
False negatives can put unsafe goods into commerce; false positives can unnecessarily block legitimate stock.

Functional testing and safety certification are not interchangeable.
A passing power-on test does not prove electrical compliance or establish that a recall remedy has been completed.
Product-specific requirements for food, cosmetics, medical devices, batteries, and chemicals may differ substantially.
Handle hazardous material, transport, and waste classification under applicable rules and trained-person procedures.

### 8.4 Waste and environmental claims

The U.S. Environmental Protection Agency (EPA) describes a non-hazardous materials and waste management hierarchy that generally places source reduction and reuse above recycling, followed by energy recovery and treatment/disposal.
This is a planning framework, not a blanket permission to reuse unsafe goods or a calculation of a specific item's lifecycle impact.
Hazardous-waste requirements depend on the material, generator, jurisdiction, and handling activity.
A destruction invoice or certificate alone does not establish recycling yield or the climate impact of downstream processing.

If reporting avoided emissions or landfill diversion, document the counterfactual, system boundary, allocation, transport, processing, and uncertainty.
A donated unit may displace a new purchase, a secondhand purchase, or nothing; do not assume one-for-one displacement without evidence.
Keep metric tons of carbon dioxide equivalent distinct from USD cash contribution.
Any internal carbon price or social value is an explicit decision assumption, not a verified cash receipt.

### 8.5 Donations, data security, and contracts

In the United States, the Internal Revenue Service (IRS) provides guidance on charitable contributions, substantiation, valuation, and business-specific treatment.
Tax treatment is fact-specific; recipient eligibility, asset type, entity, valuation, limits, and documentation can matter.
A transfer is not automatically deductible, and a deduction is not cash equal to the donated item's retail price.
No tax benefit is assumed in the worked example.
Use a tax professional to determine applicable treatment before adding any incremental tax cash flow to a model.

Products storing personal data need an appropriate sanitation or destruction process before transfer.
The National Institute of Standards and Technology (NIST) publishes media-sanitization guidance; the appropriate technique depends on media, threat, and policy.
Commercial contracts can independently restrict secondary channels, export, resale condition claims, or transfer of customer-owned goods.
Commercial policy is not the same thing as a statute, and neither a buyer's acceptance nor an internal model overrides applicable law.

## 9. Industry practice and case evidence

### 9.1 Patagonia Worn Wear

Patagonia's Worn Wear program publicly describes buying used Patagonia clothing through trade-in and offering used products for sale.
Its published program materials describe accepted item categories and condition requirements.
This demonstrates a branded resale channel with intake standards; the materials do not establish that every return is repairable or that resale maximizes profit on every garment.

A practitioner should distinguish a trade-in from a customer return.
The acquisition payment, ownership transfer, grading costs, and customer expectations can differ.
The transferable lesson is the need for an accepted-condition rubric, a channel, a price, and a feedback loop.
It is not evidence for the numerical assumptions in the worked appliance case.

### 9.2 Best Buy recycling

Best Buy publicly describes an electronics recycling program and identifies accepted products and conditions, including certain fees and restrictions.
The exact rules vary by location, product, and program version, so check current local terms.
The program is evidence that retailers can provide a collection route for electronics; collection does not prove that every item is refurbished or that every component is recovered.

The decision analyst should record collection and downstream treatment separately.
A recycling or disposal route may have negative direct cash contribution but still be required by law, contract, safety, or company policy.
The company's public program page is not a complete audited item-level environmental lifecycle assessment.

### 9.3 IKEA buyback and resale

IKEA describes a buyback and resale service in participating markets, with eligibility rules and value determined by item condition and program terms.
This shows a commercial pathway in which a retailer assesses used goods and offers them for another life.
Country-specific eligibility and compensation should not be generalized across every store or product.

**Inference:** a resale program needs space, grading labor, and a route for items not sold within its intended period.
The public description alone does not reveal the retailer's unit-level contribution margin, counterfactual displacement, or actual sell-through for a given item.
A learner cannot infer those results from the existence of the program.

### 9.4 Practical portfolio choices

Different operating models may use a store as the first sorting point, a central return center for testing, or a specialist partner for repairs.
A distributed store can shorten the path to local resale but may lack skilled technicians and consistent safety controls.
Central repair can pool expertise and parts, but added transport and queues may erase a higher selling price.
Compare those alternatives at the same item and time boundary.

A secondary-market contract can offer certainty but may constrain brand, geography, or product condition.
An apparently high bid may exclude freight or permit chargebacks for rejects.
A donation partner's needs can change, making an accepted-goods list and confirmation essential.
Track accepted quantity, rejected quantity, realized proceeds, and the time to terminal outcome rather than treating planned routes as completed recovery.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unsafe resale | Recall match bypassed or unresolved | Financial ranking precedes eligibility | Harm and potential legal exposure | Safety gate and accountable hold owner |
| Phantom sellable stock | Available-to-promise increases before release | Return receipt treated as saleable inventory | Canceled orders and customer disappointment | Release only after grade and physical availability |
| Gross-price optimization | High sales but negative route margin | Fees, freight, and labor omitted | False impression of recovery | Compare incremental net cash flows |
| Sunk-cost contamination | Same acquisition cost applied inconsistently | Historic cost mistaken for avoidable cost | Wrong path ranking | Define decision boundary and reconcile full P&L separately |
| Probability double count | Expected proceeds multiplied by sell-through twice | Conditional and unconditional measures mixed | Resale undervalued | Label denominators and branch probabilities |
| Unsold units omitted | Sold-item average rises while open stock grows | Only successful sales enter analysis | Inflated expected recovery | Include unsold fallback and cohort maturity |
| Slow repair queue | Cycle time rises and seasonal price falls | Capacity assumption ignores backlog | Markdown and storage losses | Model queue, deadlines, and re-route trigger |
| Misgraded items | High subsequent defect or return rate | Inspection rubric is inconsistent | Chargebacks and unsafe or misleading listings | Audit grades, train inspectors, track outcomes |
| Buyer rejection | Shipped lots repeatedly returned | Contract specifications differ from local grading | Extra freight and delayed recovery | Validate buyer acceptance and price rejection branches |
| Donation dumping | Partner declines or accumulates unneeded goods | Transfer treated as benefit without acceptance | Cost shifted to recipient and failed closure | Confirm demand and retain acceptance evidence |
| Unsupported tax credit | Assumed benefit equals retail value | Deduction confused with cash credit | Overstated value and tax risk | Tax review and documented incremental cash flow |
| Lost custody | Missing scans or serial mismatches | Handoffs not reconciled | Shrink, fraud, and unresolved liability | Event chain, quantity reconciliation, exception owner |
| Incorrect destruction closure | Label issued but no destination confirmation | Instruction confused with final treatment | False closure and compliance gap | Require evidence and reconcile quantities |
| Hidden selection bias | Refurbished items appear superior only on easy grades | Routes receive different product mixes | Miscalibrated future decisions | Segment, adjust, and pilot where feasible |
| Misleading environmental claims | Diversion reported as emissions saved | No counterfactual or lifecycle boundary | Unsupported public claims | Document assumptions and obtain expert review |
| Inconsistent dates | Negative elapsed time or sudden process improvement | Event time confused with load time | Bad service and model metrics | Preserve both timestamps and validate ordering |
| Unpriced capacity | Repair hours exceed weekly availability | Unit-level optimum ignores shared resource | Backlog and missed opportunities | Optimize against capacity and fallback values |
| Uncontrolled override | Unusual approvals cluster by site or buyer | Authority or audit trail is weak | Leakage, favoritism, or unsafe transfer | Versioned policy, approval limits, periodic review |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Item is owned and transferable | Enables legal route choice | Marketplace title terms or unresolved customer claim | Verify contract, owner record, and hold exceptions |
| Grades are comparable | Supports cohort-based recovery estimates | Site-specific standards or inspector drift | Blind reinspection and inter-rater agreement |
| Safety checks are current | Defines the feasible set | New recall or missed identifier match | Refresh source, audit matches, recheck before release |
| Branches are exhaustive | Expected value covers all outcomes | Buyer rejection, second return, or abandoned inventory omitted | Reconcile outcome counts to all items in cohort |
| Probabilities are calibrated | Averages reflect likely outcomes | Season, mix, channel, or policy shifts | Compare forecast and observed outcomes by segment and vintage |
| Net sale proceeds are conditional on sale | Prevents incorrect weighting | Data already averages over all listings | Trace numerator and denominator to source ledger |
| Future costs are incremental | Choice uses avoidable cash flows | Fixed overhead will actually change or labor is constrained | Obtain finance and operations review of cost drivers |
| The chosen discount rate is appropriate | Makes dated alternatives comparable | Different risk or currency conventions | Sensitivity analysis and finance-approved rate |
| Unsold fallback is available | Prevents stranded inventory from being valued as certain cash | Buyer exits, contract limits, or hazardous condition | Confirm current terms and model rejection |
| Capacity is sufficient or priced | Unit values can be executed | Multiple items compete for scarce technician hours | Compare workload to capacity by period and optimize allocation |
| Donations are accepted | Makes transfer a real option | Recipient demand or legal eligibility changes | Confirm item-specific acceptance and record evidence |
| No tax benefit is modeled | Avoids unsubstantiated tax assumptions | Verified, route-specific tax consequences are material | Obtain qualified tax advice and model incremental timing |
| Time dependence is represented | Captures delay effects on value | Price cliff or seasonal demand not in forecast | Reforecast by age and simulate delay scenarios |
| Historical outcomes transport to current policy | Allows learning from prior data | New channel or grade policy changes item selection | Use controlled pilots or adjustment with uncertainty |
| Expected value matches the objective | Ranking implements management preference | Risk, safety, social, or contractual objectives dominate | Set hard constraints and document multiobjective trade-offs |
| Unit outcomes are sufficiently independent for aggregation | Makes simple portfolio totals informative | Correlated demand or buyer default | Stress correlated scenarios and monitor concentration |

## 12. Exercises

### Exercise 1 — Separate events and ownership

A marketplace customer receives a refund before the item arrives at a store.
The order belongs to a third-party seller under a consignment agreement.
The store receives a box with an unverified serial number and marks it sellable immediately.

**Deliverable:** list the distinct customer, ownership, custody, inspection, and inventory states and specify the next action.

**Acceptance criteria:** distinguish refund from physical receipt, do not presume retailer title, require serial and eligibility checks, and name a hold owner.

**Time:** 10 minutes.

### Exercise 2 — Compute route contribution

At the decision time, resale costs USD 6 immediately.
There is a 60% chance of USD 50 net sale proceeds and a 40% chance of USD 10 net fallback proceeds.
Immediate liquidation costs USD 4 and pays USD 24.
Ignore discounting and assume both routes are feasible.

**Deliverable:** calculate expected contribution for both routes and choose one.

**Acceptance criteria:** include both sale and fallback branches, subtract each cost once, state units, and identify the USD difference.

**Time:** 10 minutes.

### Exercise 3 — Timing and discounting

A buyer offers USD 100 net at day 365 or USD 92 net immediately.
Assume an effective annual discount rate of 10%, both offers are certain, and no other cost or restriction differs.

**Deliverable:** compute both NPVs and identify the preferred cash option.

**Acceptance criteria:** use the effective annual rate, report USD at the decision date, and explain why the larger nominal payment is not necessarily better.

**Time:** 10 minutes.

### Exercise 4 — Safety and feasibility

A returned electrical item matches a manufacturer recall that requires a specified remedy.
A liquidator offers USD 15, a resale forecast is USD 40, and a repair workshop claims it can test the item without reviewing the notice.

**Deliverable:** describe the immediate disposition state and the owner who must authorize a next step.

**Acceptance criteria:** prohibit financial ranking from bypassing the recall review, refer to the actual notice and applicable rules, and avoid assuming all recalled products have the same remedy.

**Time:** 10 minutes.

### Exercise 5 — Sell-through threshold

A direct resale route costs USD 5 now and pays USD 55 net if sold or USD 15 net if unsold.
A certain liquidation route has a USD 22 contribution.
Ignore discounting and assume the two branches exhaust outcomes.

**Deliverable:** solve for the direct-sale probability at which the routes have equal expected value.

**Acceptance criteria:** show the equation and threshold and state which route is better above the threshold.

**Time:** 15 minutes.

### Exercise 6 — Scarce repair hours

Three candidate item groups each contain one unit.
Repair versus each item's best non-repair fallback has incremental benefit of USD 18, USD 15, and USD 8, respectively.
The repair routes consume 2, 1, and 1 technician hours, respectively.
Exactly 2 technician hours are available and units are indivisible.
All benefits and fallback values are already net and feasible.

**Deliverable:** choose the repair portfolio and quantify its incremental benefit above the fallback portfolio.

**Acceptance criteria:** respect the 2-hour capacity, compare feasible combinations, and distinguish incremental advantage from gross repair revenue.

**Time:** 15 minutes.

### Exercise 7 — Design an audit trail

A retailer uses store intake, a central refurbisher, a secondary buyer, and a donation partner.
Some transfers are rejected and returned to the center.

**Deliverable:** specify at least eight item-level fields, six event types, two reconciliation checks, and named owners for safety and economic decisions.

**Acceptance criteria:** cover identity, condition, timestamps, custody, authorized route, acceptance or rejection, and terminal evidence; preserve decision-time inputs and later actuals separately.

**Time:** 25 minutes.

### Exercise 8 — Critique a recovery claim

A manager says: “Our resale channel recovered 80% of retail because items that sold averaged USD 80 against a USD 100 retail price, so all resale returns are profitable and emissions fell by 80%.”

**Deliverable:** identify at least four unsupported inferences and propose two measurements and one controlled comparison.

**Acceptance criteria:** address unsold units, costs, decision boundary, and environmental counterfactual; distinguish correlation from a causal claim.

**Time:** 20 minutes.

## 13. Self-check answers

### Exercise 1

The refund is a customer financial event, not proof of receipt or ownership transfer.
The store has custody of an unidentified box, but title may remain with the third-party seller under the agreement.
The item is not yet verified, graded, or eligible for sale.
Hold it out of available-to-promise inventory while operations verifies the serial and receipt and the designated ownership or legal owner resolves authority.
After inspection and safety checks, the authorized owner can choose a permitted route.

### Exercise 2

Expected resale contribution is `-6 + 0.60 × 50 + 0.40 × 10 = USD 28` per item.
Liquidation contribution is `-4 + 24 = USD 20` per item.
Resale leads by USD 8 per item under the stated assumptions.
Both figures are expected or certain future contribution from the decision point, not whole-order profit.

### Exercise 3

The day-365 receipt has NPV `100 / 1.10 = USD 90.91`.
The immediate receipt has NPV USD 92.00.
The immediate offer is higher by approximately USD 1.09 at the decision date.
This comparison assumes equivalent certainty, currency, and other costs.

### Exercise 4

Place the item on a controlled safety hold and prevent release, transfer, or sale until the responsible safety/compliance owner reviews the actual recall notice and applicable rules.
The notice may require a particular repair, return, disposal, or other remedy; no universal route can be assumed.
Only after authorized eligibility is established should feasible routes be ranked on economics.
A generic electrical test does not demonstrate completion of the specified remedy.

### Exercise 5

Expected direct contribution is `-5 + 55p + 15(1-p) = 10 + 40p` USD per item.
Set `10 + 40p = 22`, giving `p = 0.30`, or 30%.
Direct resale has greater expected contribution above 30%; liquidation is greater below 30%.
At exactly 30% the two are tied before uncertainty or other criteria.

### Exercise 6

Repairing group 1 alone uses 2 hours and adds USD 18.
Repairing groups 2 and 3 uses 2 hours and adds `15 + 8 = USD 23`.
Groups 1 and 2 or groups 1 and 3 exceed capacity; each single group is worth at most USD 18.
Choose groups 2 and 3 for USD 23 incremental benefit above the fallback portfolio.
The decision depends on incremental advantages, not the gross values of the repair routes.

### Exercise 7 rubric

A strong design includes unique unit or lot identity, ownership, location, grade, safety status, rule version, timestamp, authorized route, destination, expected cash flows, and actual results.
Events include receipt, inspection, hold or release, authorization, shipment, partner receipt, acceptance or rejection, and final sale, donation, or destruction.
It reconciles shipped to received quantities and expected to actual financial proceeds, with exceptions assigned to owners.
Quality/compliance owns release eligibility, finance validates the economic treatment, and operations owns custody and execution.
The record preserves original recommendation, inputs, override, and subsequent events.
A weak design closes an item on shipment without recording partner acceptance or rejection.

### Exercise 8 rubric

A strong critique identifies that USD 80 is conditional on sale, not average revenue per listed return.
It requests the denominator and includes unsold fallback, time, fees, freight, labor, and returns in contribution.
Retail price is not a cash-profit benchmark and original cost has different relevance for route choice and whole-order P&L.
An 80% financial ratio does not establish an 80% emissions reduction.
Measure mature-cohort net contribution per received item and time-to-terminal outcome, and assess environmental outcomes under a documented lifecycle boundary and counterfactual.
A controlled comparison could allocate otherwise eligible comparable items between channels where legally and operationally appropriate, tracking both sale and unsold outcomes.

## 14. Interview questions

1. **Why can the highest resale price be the wrong choice?**
   A strong answer includes probability, route costs, time, and fallback.
   A weak answer compares only gross price with original retail.
2. **Which costs are sunk at a post-inspection decision?**
   A strong answer ties the answer to whether the cost can still change and separates route choice from full-order P&L.
   A weak answer treats every recorded cost as avoidable.
3. **How do you handle a recall match?**
   A strong answer applies a hold and checks the actual remedy with safety/compliance.
   A weak answer allows the model to authorize a profitable sale.
4. **What does a 70% sale probability mean?**
   A strong answer names the cohort, horizon, channel, and denominator.
   A weak answer treats the forecast as a guaranteed price.
5. **How would you model a failed repair?**
   A strong answer includes incurred repair cost, failure probability, and feasible fallback with timing.
   A weak answer drops failed items from the average.
6. **How do you account for a repair queue?**
   A strong answer models scarce hours, delay, and changing proceeds, with a re-route threshold.
   A weak answer adds only a small discount-rate adjustment.
7. **Why is a donation not necessarily free?**
   A strong answer includes recipient acceptance, handling, transport, evidence, and qualified tax treatment.
   A weak answer assumes a deduction equal to retail price.
8. **What proves a disposition is complete?**
   A strong answer names route-specific terminal evidence and reconciliation.
   A weak answer treats a printed label as proof of destruction or sale.
9. **How would you compare recovery across sites?**
   A strong answer uses consistent boundaries, mature cohorts, product mix, and uncertainty.
   A weak answer compares sold-only revenue percentages.
10. **What does an environmental benefit claim require?**
    A strong answer identifies the counterfactual, lifecycle boundary, transport, displacement, and uncertainty.
    A weak answer equates a donation count with avoided emissions.

## 15. Key terms

- **Available-to-promise:** inventory eligible to support a new promise; see [mechanics](#52-inspect-and-grade-consistently).
- **Branch probability:** likelihood of one modeled outcome; see [worked example](#63-refurbish-route).
- **Capacity:** resource available within a time bucket; see [capacity](#56-price-time-and-capacity).
- **Chain of custody:** record of transfers and control; see [data contract](#81-minimum-data-contract).
- **Cohort:** items grouped under a common definition for measurement; see [demand estimation](#55-estimate-demand-and-recovery).
- **Conditional proceeds:** proceeds given a specified outcome; see [resale calculation](#62-direct-resale-route).
- **Contribution:** proceeds less defined costs; see [decision boundary](#51-establish-the-decision-boundary).
- **Counterfactual:** alternative outcome used to assess impact; see [environmental context](#84-waste-and-environmental-claims).
- **Decision boundary:** point from which costs and benefits are compared; see [step 1](#51-establish-the-decision-boundary).
- **Destruction:** controlled treatment with route-specific evidence; see [other routes](#64-other-routes).
- **Disposition:** authorized outcome of a returned item; see [mental model](#3-the-core-mental-model).
- **Donation:** transfer to a willing eligible recipient; see [donation context](#85-donations-data-security-and-contracts).
- **Expected value:** probability-weighted branch value; see [mental model](#3-the-core-mental-model).
- **Fallback:** authorized route used after another route fails; see [cash-flow branches](#54-map-the-future-cash-flow-branches).
- **Grade:** documented item condition classification; see [inspection](#52-inspect-and-grade-consistently).
- **Hold:** controlled temporary restriction pending review; see [safety context](#83-product-safety-and-recalls).
- **Incremental cost:** future cost changed by a choice; see [cash-flow branches](#54-map-the-future-cash-flow-branches).
- **Liquidation:** sale through an approved secondary channel; see [other routes](#64-other-routes).
- **Markdown:** reduction from a defined price baseline; see [mental model](#3-the-core-mental-model).
- **Net present value:** discounted value at a common date; see [code](#7-code-query-or-calculation).
- **Opportunity cost:** value of the displaced feasible alternative; see [capacity](#56-price-time-and-capacity).
- **Recall:** a product corrective action requiring specific review; see [safety context](#83-product-safety-and-recalls).
- **Recovery proceeds:** cash from a route before specified route costs; see [definitions](#4-definitions-and-vocabulary).
- **Refurbishment:** documented repair, preparation, and testing; see [worked example](#63-refurbish-route).
- **Resale:** sale of an eligible item through an approved channel; see [worked example](#62-direct-resale-route).
- **Right censoring:** outcome not yet observed at measurement cutoff; see [demand estimation](#55-estimate-demand-and-recovery).
- **Sunk cost:** cost unaffected by the current choice; see [decision boundary](#51-establish-the-decision-boundary).
- **Terminal event:** verified route completion; see [execution](#58-execute-and-close-the-loop).

## 16. Sources

Accessed as general reference for this educational lesson; program terms, regulations, and web pages can change.
The case numbers above are illustrative assumptions, not sourced company results.

1. [GS1 EPCIS and CBV standards](https://www.gs1.org/standards/epcis) — event-sharing standards and documentation.
2. [GS1 GTIN](https://www.gs1.org/standards/id-keys/gtin) — trade-item identification and its scope.
3. [U.S. Consumer Product Safety Commission recalls](https://www.cpsc.gov/Recalls) — searchable product recall notices and remedies.
4. [CPSC: Resellers and product safety](https://www.cpsc.gov/Business--Manufacturing/Business-Education/Resellers) — U.S. reseller responsibilities and resources.
5. [U.S. EPA: Sustainable materials management hierarchy](https://www.epa.gov/smm/sustainable-materials-management-non-hazardous-materials-and-waste-management-hierarchy) — non-hazardous waste management hierarchy and limits.
6. [U.S. EPA: Hazardous waste](https://www.epa.gov/hw) — hazardous-waste regulatory and educational resources.
7. [IRS Publication 526, Charitable Contributions](https://www.irs.gov/publications/p526) — U.S. charitable contribution rules and substantiation; specific business circumstances require professional review.
8. [NIST SP 800-88, Guidelines for Media Sanitization](https://csrc.nist.gov/pubs/sp/800/88/r1/final) — media sanitization guidance; consult the current revision and organizational policy.
9. [Patagonia Worn Wear](https://wornwear.patagonia.com/) — public trade-in and used-product program description; company source, not independent outcome evidence.
10. [Best Buy electronics recycling](https://www.bestbuy.com/site/services/recycling/pcmcat149900050025.c?id=pcmcat149900050025) — public program conditions; verify current local terms.
11. [IKEA buyback and resell](https://www.ikea.com/us/en/customer-service/services/buy-back-resell/) — U.S. program description; country terms may differ.
12. Fleischmann, M., et al. (1997), “Quantitative models for reverse logistics: A review,” *European Journal of Operational Research*, 103(1), 1–17. [DOI: 10.1016/S0377-2217(97)00230-0](https://doi.org/10.1016/S0377-2217(97)00230-0) — research context for reverse-logistics modeling, not validation of this case's inputs.

## 17. Links

- [Module 23 index](_index.md)
- [Inventory visibility](01_inventory-visibility.md)
- [Available-to-promise](02_available-to-promise.md)
- [Order routing and distributed order management](03_order-routing-dom.md)
- [Ship-from-store economics](04_ship-from-store.md)
- [Promise accuracy](06_promise-accuracy.md)
- [Module 23 survey and lab material](../../../course/23_omnichannel_fulfillment_returns.md)

The returns-network, return-fraud, and channel-P&L topics are listed in the module index; consult them when available.
