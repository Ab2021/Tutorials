# Replenishment, Shelf Life, and Waste

> Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics  
> Topic 04 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Module 22, Topics 01–02; basic probability and inventory accounting  
> Version 2026.5

## 1. Why this topic exists

A dark store may promise a short delivery window while holding only a few days of fresh food.
A replenishment order placed now can arrive after a demand peak, arrive short, or arrive with too little remaining shelf life.
The planner cannot repair all three problems by adding safety stock: additional stock itself can expire.
The store manager also cannot sell a physically present unit if it is unsafe, reserved, expired, recalled, or too old for the customer promise.

This unit follows one store and one chilled product from the order cutoff through receipt, picking, sale, and disposition.
The decision is how many usable units to order, not how many cartons fit on a shelf.
The analytical objective combines the economic consequences of missed demand and unsold stock while respecting safety, storage, and supplier constraints.
There is no general rule that stockouts are always worse than waste or vice versa.

The replenishment planner chooses a proposed order before the supplier cutoff.
The store receiver validates the lot and records its usable life.
The picker applies first-expiry-first-out (FEFO), subject to customer freshness and safety rules.
The food safety owner controls holds and disposal; finance defines cost treatment; the commercial owner approves availability and markdown policy.
These roles must agree on what a sellable unit means before a model can recommend a quantity.

A forecast is a probability distribution for future demand.
A plan is the selected order and operating action.
A target is an intended service or waste level.
A commitment is a promise made to a supplier or customer.
The four are not interchangeable, and none permits an unsafe sale.

The numerical examples are synthetic; prices, probabilities, demand, and costs are assumptions rather than market estimates.
The decision horizon is explicitly stated in each calculation.
The model is educational and does not substitute for a validated food safety plan or local legal advice.

## 2. Learning outcomes

After this unit you can:

1. **Define** the decision clock, sellable inventory, life-at-delivery constraint, and exposure window; Exercise 1 checks the contract.
2. **Allocate** demand across lots using FEFO and calculate stockout and expired units; Exercise 2 checks the ledger.
3. **Calculate** underage and overage costs and identify the corresponding critical fractile; Exercise 3 checks the arithmetic.
4. **Compare** feasible case-pack orders under uncertain demand and life-at-receipt; Exercise 4 checks expected cost.
5. **Design** a point-in-time data model and reconciliation controls; Exercise 5 tests the design.
6. **Diagnose** censored demand and supplier uncertainty without treating sales as demand; Exercise 6 tests the critique.
7. **Evaluate** an intervention with paired scenarios and operational guardrails; Exercise 7 tests the experiment.
8. **Assign** decision rights and escalation for safety and inventory exceptions; Exercise 8 tests governance.

A correct forecast is not a replenishment decision until it is combined with inventory, uncertainty, economics, and constraints.
A correct order calculation is not an executable plan until its cutoff, pack size, capacity, and owner are known.

## 3. The core mental model

```text
forecast origin and supplier cutoff
            |
            v
  demand scenarios + availability history ----> latent demand distribution
            |                                      |
  current lots: quantity, status, expiry            |
            |                                      |
  inbound scenarios: arrival time, fill, life        |
            |                                      |
            +-------------------+------------------+
                                v
              simulate eligible inventory over time
                                |
          receipt -> safety release -> FEFO pick -> delivery
                                |             |
                         markdown / transfer   unmet demand
                                |
                      expiry -> quarantine -> disposition
                                |
                                v
       expected contribution, service, waste and capacity outcomes
                                |
            planner recommendation -> approval -> supplier order
                                |
                 actual events -> reconciliation -> review
```

Each arrow is time-sensitive.
A lot that expires before receipt of the next shipment cannot cover the entire protection period.
A late receipt is not inventory during the hours before arrival, even if it is counted in an end-of-day total.
A unit with enough life at picking may still fail the minimum-life promise at delivery.

The recurring decision balances two marginal risks.
The underage cost is the incremental loss when one more eligible unit would have met demand.
The overage cost is the incremental loss when an extra unit remains unsold at the end of its usable life.
Both must be measured on the same decision horizon and the same incremental basis.
A purchase cost cannot be counted twice as both an overage expense and the entire cost of lost revenue.

FEFO is an allocation rule, not an ordering rule.
It allocates an eligible earliest-expiring lot first; it does not make an expired, unsafe, or recalled lot eligible.
First-in-first-out (FIFO) follows receipt order and can select a newer-expiry lot ahead of an older-expiry lot.
FEFO needs lot-level expiry data and an execution mechanism, not just a policy statement.

Uncertainty occurs jointly.
High demand and late supplier delivery may occur together during bad weather.
Low demand and short-life receipt can combine to increase waste.
Scenario generation must preserve such dependence where evidence supports it.
When evidence is too thin, show sensitivity rather than presenting an invented precise correlation.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Stock-keeping unit (SKU) | One sellable product and pack configuration, measured in units | Treating different pack sizes as interchangeable |
| Store | A physical fulfillment location with its own inventory and clock | Pooling inventory across stores without travel time |
| Lot | Traceable batch sharing assigned identifiers and relevant date attributes | Assuming every receipt has one expiry |
| Expiry instant | Last instant when a lot is eligible under the applicable policy, with timezone | Comparing date strings with no boundary rule |
| Use-by date | A date marking with legal and safety significance depending on jurisdiction and product | Treating every date label as a quality-only suggestion |
| Best-before date | A date marking generally associated with quality, subject to local rules | Assuming it universally permits sale after the date |
| Remaining shelf life | Time from a stated instant to the applicable expiry boundary, in hours or days | Reporting total manufacturer life as life at receipt |
| Minimum life | Minimum remaining life required at receipt, pick, or delivery | Using one threshold for all three checkpoints |
| FEFO | Pick eligible lot with earliest expiry first | Picking an unsafe lot solely because it expires first |
| FIFO | Pick earliest received lot first | Calling FIFO equivalent to FEFO |
| On-hand | Physically recorded inventory, units | Treating held or expired units as sellable |
| Sellable available | Eligible on-hand less reservations and other commitments, units | Subtracting the same reservation twice |
| Inventory position | On-hand plus eligible on-order less obligations under a declared convention | Adding late or short-life inbound as certain coverage |
| Protection period | Time until the next opportunity for replenishment can arrive | Assuming a constant interval under uncertain lead time |
| Review interval | Time between ordering decisions | Confusing it with supplier lead time |
| Lead time | Time from released order to usable receipt | Measuring only truck transit and omitting receiving delays |
| Demand | Customer desired quantity before supply restriction | Equating observed sales with unconstrained demand |
| Lost sale | Desired unit that cannot be filled and is not backordered | Counting a substitute as both a lost sale and full recovery |
| Backorder | Unfilled commitment fulfilled later under a defined policy | Assuming fast-delivery orders can always wait |
| Stockout | Interval or event in which eligible inventory cannot meet demand | Using one stockout event as a count of all missed units |
| Fill rate | Filled requested units divided by requested units | Confusing it with probability of no stockout |
| Cycle service level | Probability of no stockout during a defined replenishment cycle | Interpreting it as unit fill rate |
| Write-off | Financial recognition that inventory value is lost, currency | Counting the purchase cost twice |
| Physical waste | Disposed or destroyed quantity, units or mass | Assuming all accounting write-offs are food waste |
| Salvage | Net recoverable value of an unsold eligible unit, currency per unit | Treating unsafe or prohibited sale as salvage |
| Markdown | Approved price reduction before the sale cutoff | Assuming all marked-down units sell |
| Underage cost | Incremental consequence of ordering one unit too few, currency per unit | Using retail revenue instead of contribution |
| Overage cost | Incremental consequence of ordering one unit too many, currency per unit | Ignoring recovery, disposal, or procurement terms |
| Critical fractile | Underage cost divided by underage plus overage cost | Treating its quantile as a universal service guarantee |
| Case pack | Minimum supplier order increment, units per case | Rounding an unconstrained answer without reevaluating cost |
| Demand censoring | Observation of sales capped by availability | Learning a low demand forecast from stockouts |
| Quarantine | Segregation of stock pending release or disposition | Treating held stock as sellable |
| Shrink | Inventory loss from causes such as damage, spoilage, or discrepancies | Equating every shrink category with expiry |
| Scenario | Joint possible demand, receipt, and life path with assigned weight | Assuming scenarios are independently sampled when they are not |
| As-of snapshot | Information actually available at the decision timestamp | Using a corrected future receipt record in an earlier forecast |
| Contribution | Revenue less relevant variable costs, currency per unit | Mixing gross revenue with incremental margin |

## 5. The mechanics, step by step

### 5.1 Set the decision contract

1. Identify the store, SKU, supplier, and order cutoff.
   State the operational timezone, current timestamp, and next review timestamp.
   Record whether the product can be sold during the full day or only in specified delivery slots.
2. Define the action as an integer quantity in supplier case packs.
   Include minimum order quantity, maximum supply, shelf capacity, and purchase cutoff.
   If the product shares cold space with other SKUs, an isolated calculation is only a preliminary recommendation.
3. Define the service promise at the customer delivery instant.
   Set the minimum remaining shelf life under the actual commercial policy, not a universal assumption.
   Note that supplier receipt specifications, store handling rules, and customer promises may differ.
4. Record the decision owner and authorized overrides.
   The planner selects the proposed quantity; the store and supplier confirm feasibility.
   The food safety owner can block or quarantine a lot even if the order model expects it to be sold.
5. Choose an evaluation window long enough to observe the disposition of marginal inventory.
   If the window stops before expiry, include a defensible terminal value and test its sensitivity.
   Do not compare a short-horizon stockout cost to a lifetime waste cost without reconciling the horizons.

### 5.2 Reconcile physical and eligible stock

1. Start with lot-level physical counts, not only an aggregate SKU balance.
   Each lot needs a quantity, location, status, expiry boundary, and last verification time.
   Separate physical count corrections from sales and disposal events.
2. Remove recalled, quarantined, damaged, and otherwise blocked units from eligibility.
   Exclude reserved quantities from free availability exactly once.
   Keep excluded units in the physical ledger until the relevant physical movement occurs.
3. Check the remaining life at each relevant event.
   A unit may be eligible at receipt but ineligible for a delivery several hours later.
   Compute remaining life from an instant with a timezone, rather than subtracting ambiguous calendar dates.
4. Reconcile opening quantity plus receipts and returns against sales, transfers, disposals, adjustments, and closing quantity.
   Investigate discrepancies by lot and reason code.
   A quantity adjustment is not automatically evidence of spoilage.
5. If expiry is missing, apply the safety and master-data policy rather than inventing a date.
   The model may exclude the lot and flag it for investigation.
   An authorized receiver, not the optimization engine, resolves the physical exception.

### 5.3 Build a demand distribution

1. Define desired demand by store, SKU, and decision-relevant interval.
   Fast-moving products may require hourly or smaller buckets to represent arrival before replenishment.
   Forecasting the daily total alone can hide a morning stockout followed by an afternoon receipt.
2. Use only information available at the ordering cutoff.
   Historical promotions, weather forecasts, calendars, and assortment status need publication or availability timestamps.
   Actual future weather and corrected future inventory are not legitimate historical inputs.
3. Identify periods with limited or uncertain availability.
   Sales in those periods are censored observations, not zero-demand labels.
   Use page views, substitution, comparable stores, or an explicit censored-demand model where justified.
4. Separate demand for the original SKU from demand shifted to alternatives.
   A substitute may protect some basket contribution but may also cannibalize another product.
   State whether demand scenarios already include such behavior to avoid double counting.
5. Produce scenarios or calibrated quantiles for the protection period.
   Preserve intraday demand timing and meaningful dependencies across hours.
   Report uncertainty bands and segment error by product velocity and promotion state.
6. Backtest from historical decision origins with archived data vintages.
   Compare forecast quality and downstream decision loss against simple baselines.
   A model can improve mean error while worsening the shortage tail that drives service.

### 5.4 Represent uncertain supply and usable life

1. Decompose lead time into supplier preparation, transit, receipt processing, and release.
   Use the time the stock becomes pickable, not merely the truck arrival time.
   Treat supplier cutoff and store receiving hours explicitly.
2. Estimate fill-rate uncertainty separately from demand.
   An ordered case may be unavailable, short-shipped, damaged, or rejected on receipt.
   Define scenario quantities as actual usable units rather than nominal ordered units.
3. Model remaining life at receipt, preferably using observed lot distributions.
   Some receipts may have enough life for one selling day but not the promised delivery buffer.
   Capture correlation with arrival time where the data justify it.
4. Simulate joint adverse conditions, such as high demand with late arrivals.
   Independent sampling can materially understate simultaneous shortage risk.
   With sparse observations, present named stress cases rather than fitted false precision.
5. Account for the next feasible order and its possible arrival.
   Periodic review protection is approximately review interval plus lead time only when timing is deterministic.
   Variable timing and perishability require event-based paths or a justified approximation.

### 5.5 Select an economic objective

1. Define the relevant incremental contribution from satisfying one additional unit.
   Start with selling price and subtract purchase and other avoidable variable fulfillment costs.
   Include a measured incremental basket effect only when the unit actually changes the basket outcome.
2. Define the net loss on an extra unit that cannot be sold in the normal channel.
   Include purchase cost, handling and disposal, less realistic net salvage or supplier credit.
   Specify whether tax and common overhead are excluded; never count the same cost twice.
3. Consider substitution explicitly.
   If 40% of missed customers buy an alternative with positive incremental contribution, the effective shortage cost may be lower.
   Use causal evidence or a sensitivity range rather than assuming every customer disappears.
4. Include other decision-relevant consequences only if incremental and measurable.
   Waste handling, a supplier minimum, or an extra delivery charge can change the preferred quantity.
   Customer lifetime value and reputational penalties require a defensible attribution window and should be shown separately if uncertain.
5. Set hard constraints before optimizing.
   Unsafe stock is never a permissible option even if its apparent salvage value is high.
   Service targets and waste caps should have named owners and an explicit infeasibility escalation.

### 5.6 Apply the simple single-period benchmark

For one order arriving before any demand, with all units equally usable until the period ends,
the classical newsvendor benchmark compares the marginal underage cost $C_u$ with the marginal overage cost $C_o$.
The critical fractile is $C_u/(C_u+C_o)$ when both costs are nonnegative and their sum is positive.
Choose the smallest integer total stock $S$ with $P(D\le S)$ at least that fraction.
If a probability mass lies exactly at the threshold, more than one integer can have equal expected cost.

For a feasible pack order, evaluate neighboring feasible quantities directly.
The optimum over unrestricted individual units may not be feasible under case packs.
The benchmark assumes a single sales window, immediate certain receipt, homogeneous inventory, known costs, and no future order.
When those assumptions fail, use a time-phased lot simulation or another explicitly justified model.

### 5.7 Simulate inventory by lot and event

1. Initialize a ledger of free units per lot and any reservations at the decision instant.
   Include explicit expiry boundaries and customer delivery-life requirements.
   Take care that past receipts are not inserted again as future events.
2. For each candidate order and scenario, generate its receipt event if and when it occurs.
   Apply the scenario's accepted quantity and remaining life to create one or more lots.
   A rejected shipment creates no sellable receipt, although it may create other costs.
3. Process events in a documented order at equal timestamps.
   For example, an expiry boundary may precede a demand event at the same instant under an exclusive-end convention.
   A receiving event only adds eligible inventory after quality release.
4. At each demand event, filter lots by release status, quantity, expiry, and delivery-life policy.
   Sort the remaining lots by expiry, with a stable tie-breaker such as lot identifier.
   Allocate demand until it is met or the eligible lot list is exhausted.
5. Record filled quantity, lost quantity, and the exact time and reason for any shortfall.
   If substitutions are in scope, model them as a separate state transition with separate economics.
   Do not silently carry lost quick-commerce demand as a backorder.
6. At each lot cutoff, remove remaining unsold units from the sellable ledger.
   Route them through approved markdown, donation, return, or disposal processes where applicable.
   A legal and safe disposition is an operational decision, not a mathematical assumption.
7. Compute scenario-level incremental cost and operational metrics.
   Weight scenario outcomes by probabilities that sum to one.
   Compare expected values and tails, not just the single most likely outcome.
8. Repeat with the same scenario paths for each candidate order.
   Common scenarios isolate the impact of changing the policy from random simulation noise.
   For stochastic Monte Carlo, report the seed, replications, uncertainty, and data version.

### 5.8 Choose and release a feasible plan

1. Rank candidate orders by expected incremental cost or contribution under the stated baseline.
   Screen out orders that violate case packs, space, supplier limits, or safety constraints.
   If objectives conflict, show a frontier rather than hiding the trade-off in an arbitrary weight.
2. Show the recommended quantity alongside expected shortage, expiry, and cost.
   Include an adverse scenario and the difference from the current policy.
   A precise-looking expected value is not a promise of the same realized outcome.
3. Request an override reason if the operator selects another quantity.
   Record the actor, timestamp, reason, alternative, and evidence available at that time.
   Compare overrides with the original recommendation without assuming an override is automatically bad.
4. Send the order before cutoff and capture supplier acknowledgement.
   Acknowledged quantity, scheduled arrival, and actual accepted quantity are distinct fields.
   Escalate any material variance before the store's response options expire.
5. At receipt, verify identity, quantity, condition, temperature where relevant, and remaining life.
   The authorized receiver accepts, rejects, or quarantines according to approved policy.
   Update lot locations and make only released stock available to pickers.

### 5.9 Monitor and learn

1. Compare expected and actual demand only after correcting for availability where possible.
   Split forecast error from receipt error, inventory-record error, and execution error.
   Review missed demand and wasted units in their original units before converting to money.
2. Track FEFO adherence using the earliest eligible lot at the time of each pick.
   Exclude held and ineligible lots from the denominator and log valid exceptions.
   A low adherence rate signals process or data issues, not necessarily picker negligence.
3. Separate controllable expiry from recall, supplier rejection, handling damage, and count corrections.
   Analyze by store, SKU, lot, supplier, and cohort of remaining life at receipt.
   Avoid aggregating an outlier cold-chain incident into an ordinary demand error.
4. Monitor unit fill, time in stock, waste, write-off, markdown, and incremental contribution together.
   Define denominator and reporting period for every metric.
   Improving fill by ordering much more stock may destroy contribution or exceed space.
5. Review assumptions when assortment, promise, supplier schedule, or commercial price changes.
   Re-estimate costs and scenarios when the decision environment changes materially.
   Preserve prior decisions and inputs to support audit and retrospective comparison.

### 5.10 Specify the stock ledger precisely

An aggregate inventory position is useful for an initial screen, but its components require explicit definitions.
For an illustration, assume 18 physical units at 09:00.
Of these, two are on safety hold and three are already reserved for accepted customer orders.
If reservations refer only to released units and do not overlap the hold, free eligible stock is $18-2-3=13$ units.
If one reserved unit belongs to the held group, that subtraction would count it twice.
Instead, identify the intersection explicitly and compute the union of exclusions.
The resulting free eligible stock would be $18-(2+3-1)=14$ units.

This calculation is a snapshot, not a prediction that all 14 units remain eligible all day.
The lot ledger must remove each quantity at its own relevant cutoff.
It must also add only the quantity from inbound stock that has been received and released.
A convenient inventory position may add confirmed inbound, but the simulation still needs its arrival and expiry times.
An acknowledged shipment scheduled tomorrow cannot cover an order due today.
Similarly, an unacknowledged purchase order should not be represented as a certain receipt.

Reservations require an event policy.
Define when a customer order creates a reservation and when cancellation releases it.
Define whether the reservation is assigned to a particular lot or is only a SKU-level obligation.
If it is assigned to a lot, an expiry or recall may force reallocation or cancellation.
If it is not assigned, the model must avoid promising more eligible units than can be picked by the delivery time.
The inventory owner reconciles the reservation feed with the order-management system.

### 5.11 Calculate remaining life consistently

Suppose a lot has an exclusive expiry boundary at 18:00 on Friday, local store time.
Its remaining life at 10:00 Friday is eight hours.
If the planned delivery is at 16:30 Friday, its remaining life at delivery is 90 minutes.
With a minimum delivery-life rule of two hours, the lot is not eligible for that delivery.
With a one-hour rule, it meets that specific numerical threshold, provided all other safety conditions are satisfied.
The store must use its actual approved interpretation rather than adopting these illustrative thresholds.

When a label contains only a date, the system needs a product- and jurisdiction-specific interpretation of that date.
The data owner should store the original printed date as well as the derived instant and derivation rule.
Do not assume midnight at the start or end of the printed day without an approved rule.
If products cross timezones, convert instants consistently and retain the local operational timezone.
Daylight-saving changes can create ambiguous or nonexistent local times.
Timestamp libraries and explicit offsets help avoid silently moving the cutoff by an hour.

Projected delivery time is itself uncertain.
An operator can use a conservative promised delivery endpoint or another approved risk policy.
Using the median delivery time alone may leave customers near the tail with less remaining life than promised.
The life rule must be applied at the same point in the customer journey as the promise.
If delivery is delayed after picking, escalation depends on the customer policy and safety plan.
The model should record its assumed delivery-time buffer rather than hiding it in a modified expiry.

### 5.12 Separate date rotation from physical access

FEFO requires the earliest eligible lot to be physically accessible.
An older lot behind a newer pallet may be technically known but impractical to pick under a short promise.
The store can use distinct lot locations, shelf labels, and replenishment procedures to reduce that conflict.
Those controls cost labor and space, so they should be included in an operational assessment.
An optimizer that assumes free access may overstate the benefit of FEFO.

For a directed pick, the system can return lot identifier, location, quantity, and expiry.
The picker confirms the actual lot scanned rather than accepting the suggested lot automatically.
If the suggested lot is unavailable, the system should record an exception and offer another eligible lot.
The inventory controller investigates repeated location mismatches.
A valid exception may include a damaged lot or an inaccessible location with an active safety restriction.
The exception reason should not silently convert a prohibited lot into an eligible one.

If the store does not track individual picks by lot, FEFO adherence is harder to measure.
A periodic physical audit can compare shelf placement with the lot sequence.
This gives partial evidence, not a complete transaction-level adherence rate.
State the sample design, sampling frequency, and observed exceptions.
Avoid claiming exact compliance from a shelf audit that did not observe actual picks.
Use the audit results to prioritize data and process improvements.

### 5.13 Handle partially fulfilled demand

Suppose a customer requests three units and only two eligible units are available.
The customer policy may permit a partial fulfillment, require all three, or offer a substitute.
These choices produce different filled-unit and completed-order metrics.
If partial delivery is permitted, two units are filled and one is unmet.
If the entire line is canceled, all three units may be unmet, depending on the measurement contract.
Capture the customer decision rather than assuming the same response for every product.

If a substitute is offered, distinguish offered, accepted, rejected, and unavailable.
Record the original SKU, alternative SKU, quantity, and price adjustment.
The substitution may require another inventory reservation and another pick.
Its contribution includes the alternative product's avoidable costs and any price guarantee.
Only incremental recovered contribution offsets the original shortage.
Do not count the substitute's full revenue as a benefit if the customer would have purchased it anyway.

For example, assume an original shortage would lose USD 2.00 contribution.
Suppose an accepted substitute has USD 1.20 of genuinely incremental contribution.
If the acceptance probability is 25% and no further costs apply, expected recovery is USD 0.30 per shortage.
The illustrative effective underage cost is USD 1.70, not USD 2.00.
If the acceptance probability or incremental contribution is uncertain, show a range.
This arithmetic is conditional on the customer and basket assumptions, not a universal substitution factor.

### 5.14 Derive shortage and overage from a common baseline

An easy way to avoid double counting is to compare the candidate decision with an idealized matched-demand baseline.
Under that baseline, each demanded unit is available and each purchased unit is sold.
One unit too few loses the contribution that the missing sale would have earned.
One unit too many incurs the purchase and disposal costs of the unsold unit, less net salvage.
The mismatch loss is a decision-comparison construct, not a substitute for a financial statement.
Actual profit also contains baseline contribution, fixed costs, and other items outside the incremental comparison.

Suppose price is USD 6.00, purchase cost is USD 3.00, and avoidable selling cost is USD 0.50.
The contribution from one additional sale is USD 2.50.
If disposal costs USD 0.20 and the unit has no salvage, its overage cost is USD 3.20.
The critical fractile is $2.50/(2.50+3.20)$, approximately 43.86%.
The lower quantile reflects the relative cost of an unsold unit under these particular inputs.
Changing customer impact or salvage can move the quantile materially.

If the supplier refunds USD 1.00 for an unsold eligible unit, the net overage cost becomes USD 2.20.
The new critical fractile is $2.50/(2.50+2.20)$, approximately 53.19%.
That calculation assumes the credit is certain, applies to every excess unit, and costs nothing to claim.
If the supplier rejects some claims or recovery involves transport, adjust the expected net credit.
If the credit is contingent on placing a future order, model the commercial condition explicitly.
Do not treat a discretionary request as an unconditional cash recovery.

### 5.15 Interpret service measures

Unit fill rate measures the proportion of requested units fulfilled.
Order fill rate measures the proportion of orders fulfilled completely, using an explicit order definition.
Cycle service level measures the probability that no stockout occurs in a defined replenishment cycle.
Time in stock measures the proportion of a specified interval in which a product is available.
These measures answer different questions and can move in different directions.
A policy can fill most units while a large share of orders experience at least one unavailable item.

Consider 10 orders of one unit each and one order of 10 units.
If the large order loses one unit and the smaller orders are all complete, unit fill is $19/20=95\%$.
Complete-order fill is $10/11$, approximately 90.91%.
If the missing unit instead occurs in one small order, the same unit fill rate applies.
Complete-order fill is still $10/11$ in this particular example, but the affected customer and basket differ.
The metric must match the operational decision and customer promise.

Availability measurement also needs a denominator.
Time in stock measured only while the app displays the product can exclude periods when the product was hidden.
It can therefore improve when the store suppresses the listing rather than improves inventory.
Use scheduled selling time or another explicit denominator and report suppression separately.
Define whether a product with one eligible unit counts as fully available for a multiunit request.
For a demand-weighted measure, explain how desired demand is estimated during unavailable periods.

### 5.16 Work with demand forecast uncertainty

Point forecasts alone do not identify the likelihood of very low or very high demand.
Two products can have the same mean but different dispersion and different optimal orders.
For example, a constant demand of eight units differs from demand of zero or 16 units with equal probability.
Both have a mean of eight units.
With finite shelf life, the high-variance product faces both greater shortage and greater expiry risk.
The choice depends on the full distribution and asymmetric costs, not the mean alone.

Check whether forecast quantiles are calibrated at the relevant decision horizon.
If a claimed 90th percentile is exceeded far more than 10% of comparable observations, investigate.
Observations must account for censored demand before drawing conclusions.
Check calibration separately for promotion days, short-life products, and disrupted supply periods.
Sparse segments may require pooling or wider uncertainty, not spurious precision.
Use a holdout period and record how many decision origins support each estimate.

The forecast should also preserve temporal shape.
A ten-unit daily scenario can consist of nine units before receipt and one after receipt.
Another ten-unit scenario can consist of one before receipt and nine after receipt.
They have identical daily totals but different shortages for a late delivery.
Generate intraday paths or conditional profiles that preserve this distinction.
Use comparable historical days and stress tests if a full stochastic model is not justified.

### 5.17 Model supplier variability and dependence

Supplier performance should be measured against the promised usable receipt, not a vague daily delivery date.
Record ordered, acknowledged, shipped, delivered, accepted, and released quantities separately.
Calculate the elapsed time between each stage.
A full truck arrival can still leave the store short if part of the shipment fails quality checks.
An on-time arrival can still be operationally late if receiving is delayed.
The supplier and store may own different stages of the delay.

Do not multiply average fill rate by ordered quantity and assume the resulting fractional amount arrives with certainty.
If a case is either fully delivered or not delivered, the outcome distribution can be bimodal.
The expected shortage under that distribution can differ from shortage at the average receipt.
This difference is particularly important near the availability threshold.
Represent observed short-ship patterns at a unit, case, or shipment level that matches the process.
Keep partial receipt and rejection as separate causes where the data permit.

Weather can affect both demand and supply.
For an illustrative storm scenario, demand may rise while the truck is delayed.
An independent model that pairs high demand with ordinary receipt too often can understate risk.
Estimate dependence only when sufficient joint data exist.
Otherwise present an explicit adverse scenario and label its probability as assumed or leave it unweighted.
Decision makers can then see whether the selected order depends on an unsupported correlation.

### 5.18 Incorporate storage and handling constraints

Available cold space varies through the day as receipts and picks occur.
A daily average volume can hide a capacity breach immediately after a delivery.
For each event, compute occupied volume from remaining units and their storage requirements.
Include other SKUs sharing the same space when evaluating an executable store plan.
Reserve space for safe handling and required air circulation under the approved facility procedure.
An individual SKU recommendation is provisional until the shared constraint is checked.

Case packs may be indivisible for ordering but divisible after receipt.
Clarify whether open cases can be stored, transferred, or returned.
Minimum order quantities, supplier caps, and delivery fees may be defined at supplier rather than SKU level.
For shared fixed delivery costs, evaluate the joint basket of orders or allocate the cost carefully.
Charging the entire fixed fee to every SKU will overstate the cost of ordering several items.
Ignoring a fee triggered only by a particular order can understate marginal cost.

Shelf placement also affects picking speed and rotation.
More facings may increase capacity but can displace another product.
A high-velocity product may need separate backroom and pick-face stock.
Backroom replenishment should not bury an earlier-expiry lot behind later stock.
Measure labor and travel time as well as product waste when comparing arrangements.
The store manager owns the physical operating plan, subject to safety requirements.

### 5.19 Compare prevention, markdown, and redistribution

Prevention changes the amount or timing of stock entering the system.
Examples include smaller cases, more frequent deliveries, more accurate forecasts, and supplier life agreements.
Each can reduce excess exposure but may raise purchase, transport, receiving, or stockout cost.
Compare the incremental end-to-end economics rather than only the disposal invoice.
Verify that the supplier can reliably perform the proposed change.
An additional delivery may require extra cold-chain capacity and labor.

Markdown changes price and can change both the timing and composition of demand.
Estimate the uplift relative to what would have sold at the normal price.
If 10 units would have sold without a discount and 12 sell with a discount, only two sales are incremental before displacement effects.
The discount also reduces revenue on the 10 units that would otherwise have sold at full price.
An apparently successful clearance can therefore have negative incremental contribution.
Include changes in fulfillment cost, demand for substitutes, and the fate of units still unsold.

Redistribution moves eligible surplus to another channel or recipient.
Its net value depends on acceptance, collection timing, handling, and transportation.
It may reduce physical waste even when the direct financial recovery is zero or negative.
Separate social or environmental objectives from monetary contribution rather than assigning an unsupported cash value.
Document the recipient's requirements and the safety owner who approves the transfer.
Never assume that every expired or held unit is eligible for donation.

### 5.20 Account for transfer decisions

A transfer from Store A to Store B can improve utilization if B has genuine incremental demand.
It also removes stock from A and can create a shortage there.
Model both stores' demand distributions and the transfer departure and arrival times.
Apply remaining-life requirements at the receiving store and at eventual customer delivery.
Include handling, travel, and cold-chain requirements in the decision.
If the transfer arrives after B's demand peak, aggregate daily demand will overstate its value.

Suppose A has three excess eligible units and B has capacity for three units.
The transport cost is USD 4.00 for the trip, not USD 4.00 per unit.
If expected incremental recovered contribution is USD 2.00 per unit and all three are sold, gross contribution is USD 6.00.
The illustrative incremental value before other handling is USD 2.00.
If only one unit is expected to sell, the same transfer loses USD 2.00 before additional costs.
This simple calculation excludes the possible opportunity cost of taking stock from A.

Transfers should be approved before the sending lot becomes ineligible.
The sender records the outbound movement; the receiver records the inbound movement and release status.
In-transit stock is neither free inventory at A nor immediately available at B.
Use a transfer identifier to avoid counting the same units at both stores.
Record any loss or temperature excursion during transit for safety review.
An unresolved transfer discrepancy requires an owner and an investigation deadline.

### 5.21 Backtest the decision rather than only the forecast

Start at a historical order origin and reconstruct information available then.
Generate candidate orders using only that information and historical commercial constraints.
Replay demand and supply events over the same horizon for each candidate.
Compare a new policy against the actual policy and a simple baseline.
Report aggregate cost and the distribution across stores, products, and weeks.
One average can conceal a policy that performs poorly for the most perishable products.

Historical replay has an important limitation: observed demand and operations reflect the historical policy.
If the old policy stocked out, its sales do not reveal all demand the new policy would have served.
If the new policy changes price or availability, customer behavior may change as well.
Estimate censored demand carefully and identify the resulting uncertainty.
Replay is evidence for a counterfactual, not direct observation of the counterfactual.
A controlled field test is stronger when feasible and safe.

For each replay, report incremental contribution, unit fill, expired units, and safety exceptions.
Also report sensitivity to disposal cost, recovery value, and demand uncertainty.
Use the same candidate scenarios when comparing policies to reduce noise.
Report confidence intervals only when their construction and sampling unit are justified.
If products within a store share demand shocks, treating every unit as independent exaggerates precision.
Cluster or otherwise account for dependence in the analysis design.

### 5.22 Design a controlled operational pilot

Choose treatment and comparison units before seeing outcomes.
Stores or store-SKU groups may be more practical randomization units than individual orders.
Balance on historical velocity, supplier schedule, remaining life, and baseline waste where feasible.
Define a run-in period for inventory data quality and team training.
Fix the evaluation horizon long enough to observe the disposition of incremental orders.
Document any exclusions before analyzing the outcome.

The primary outcome might be incremental contribution per eligible SKU-week.
Secondary outcomes can include unit fill, expiry units, kilograms discarded, and picker exceptions.
Safety incidents and breaches of required handling controls are guardrails, not tradeable benefits.
Set the stopping authority and escalation route before launch.
Capture operational workload and overrides so that a theoretically better policy does not conceal extra labor.
If the policy changes assortment or prices, measure those interventions separately where possible.

Interference can occur when customers substitute between treatment and control products.
It can also occur when stores share supplier capacity or transfer stock.
Choose cluster boundaries and analysis assumptions with these paths in mind.
Monitor whether randomization remains intact when managers override recommendations.
Analyze the assigned policy and separately describe actual adherence without treating adherence as randomly assigned.
Publish limitations when the pilot is too small to estimate rare safety or shortage outcomes precisely.

### 5.23 Define operational exception playbooks

For a missing expiry, the receiving system should not invent a placeholder date that looks valid.
The receiver segregates the lot according to policy and asks the supplier or data owner to resolve the record.
The planner recalculates the order or availability using only released stock.
The incident record captures time, lot, quantity, owner, and final disposition.
If the supplier supplies a correction, preserve the original record and evidence.
Only the authorized safety process can release the lot.

For a late shipment, compare the revised release estimate with demand before arrival.
The planner may request an expedited delivery, alternative supplier, or safe transfer if feasible.
The commercial owner may restrict availability or adjust the customer promise.
The store should not count the unreceived shipment as stock merely to maintain a target.
After arrival, record actual release time and life, not only the revised appointment.
Review whether the delay was foreseeable and whether the response was timely.

For a failed FEFO scan, determine whether the issue is data, physical placement, damage, or a legitimate exception.
Do not automatically blame the picker or automatically correct the data to match the pick.
The inventory controller reconciles the actual location and lot quantity.
The safety owner reviews any uncertain product status.
The store manager addresses layout or training issues if the evidence supports them.
Use recurring exception patterns to improve the process.

For an impending expiry, quantify units, time remaining, forecast demand, and available approved actions.
Possible actions include an authorized markdown, transfer, return, donation, or disposal.
Each action has a cutoff, capacity, labor requirement, and owner.
Select an action before its own operational deadline, not after the product becomes ineligible.
Record outcomes separately from the original recommendation.
This distinguishes a good order decision from an effective or ineffective recovery intervention.

### 5.24 Build an auditable decision record

Store the recommendation identifier, model version, input snapshot, scenario version, and cost assumptions.
Record candidate quantities, infeasible candidates, and the reason each was excluded.
Preserve the recommended quantity and the final released quantity as separate fields.
Record the approving actor, time, and reason for any override.
Link the purchase order to supplier acknowledgment and eventual lot receipts.
These links allow a reviewer to trace an outcome back to the information available at the decision.

Keep realized demand, filled demand, substitute activity, receipt, expiry, and disposition outcomes.
Distinguish outcomes observed directly from demand estimated after a stockout.
Do not overwrite a prior forecast with a retrospectively improved estimate.
Record the date when a label became mature enough for evaluation.
Protect customer and employee information according to the organization's privacy policy.
Use access controls appropriate to commercial and operational sensitivity.

The review cadence should match the decision.
An hourly exception may need immediate attention from the store.
A weekly product review can identify recurring supplier or forecast problems.
A monthly finance review can reconcile credits, write-offs, and disposal invoices.
A quarterly policy review can reconsider service and waste trade-offs.
Named owners and deadlines are more useful than an unassigned dashboard.

## 6. Worked example

### 6.1 Decision setting

A planner at Store A orders a chilled yogurt SKU at 08:00 on Monday.
The supplier delivers at 12:00 on Monday; the store closes at 20:00.
For the initial benchmark, receipt is certain and demand before receipt is out of scope.
A later stress test relaxes that assumption.
Each case contains four units; the store may order zero, one, two, or three cases.
The existing sellable stock at receipt is two units, both usable through the end of the sales window.
All examples count individual units and use US dollars (USD).

The retail price is USD 5.00 per unit.
The purchase cost is USD 2.50 per unit.
The avoidable handling cost on a sold unit is USD 0.50 per unit.
An unsold unit has zero salvage value and costs USD 0.25 to dispose of.
The planning horizon ends at 20:00, after which all benchmark stock is assumed unsellable.
These are illustrative incremental values, not a claim about any operator.

The store has no relevant fixed-order charge, and a lost sale is not backordered.
No substitute is modeled; no separate customer lifetime penalty is included.
The existing units' purchase costs are sunk for this immediate order decision, but their opportunity to satisfy demand is included.
The target is a unit fill rate of 90%; it is not a guaranteed commitment.
The forecast is the following probability distribution for desired demand from 12:00 to 20:00:

| Demand, units | Probability | Cumulative probability |
|---:|---:|---:|
| 3 | 0.10 | 0.10 |
| 6 | 0.25 | 0.35 |
| 9 | 0.40 | 0.75 |
| 12 | 0.20 | 0.95 |
| 15 | 0.05 | 1.00 |

Expected demand is $3(0.10)+6(0.25)+9(0.40)+12(0.20)+15(0.05)=8.55$ units.
A forecast of 8.55 units is not an order for 8.55 units: feasible orders are case packs and inventory already exists.

### 6.2 Marginal costs and benchmark

One additional unit sold earns contribution of $5.00-2.50-0.50=2.00$ USD.
Thus $C_u$ is USD 2.00 per unmet unit under the stated no-substitution assumptions.
One additional unsold purchased unit costs $2.50+0.25=2.75$ USD.
Thus $C_o$ is USD 2.75 per expired unit, with zero salvage.

The critical fractile is $2.00/(2.00+2.75)=0.42105$.
The smallest demand value whose cumulative probability reaches that fractile is nine units.
The unrestricted benchmark therefore recommends total stock of nine units.
With two units already present, the unrestricted incremental order is seven units.
It is infeasible because the supplier sells only cases of four units.

### 6.3 Compare feasible quantities

With zero, one, two, or three cases, total stock at receipt is respectively 2, 6, 10, or 14 units.
For each scenario, shortage is $\max(D-S,0)$ and expiry is $\max(S-D,0)$.
Expected mismatch cost is $2.00E[(D-S)^+]+2.75E[(S-D)^+]$.
This cost is an incremental comparison against a perfect-information matching baseline, not the store's accounting profit.
The baseline is constant across candidates; applying the same overage charge to existing units does not change the ranking.

| Cases | Ordered units | Total stock $S$ | Expected shortage, units | Expected expiry, units | Expected mismatch cost, USD |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 2 | 6.55 | 0.00 | 13.1000 |
| 1 | 4 | 6 | 2.85 | 0.30 | 6.5250 |
| 2 | 8 | 10 | 0.65 | 2.10 | 7.0750 |
| 3 | 12 | 14 | 0.05 | 5.50 | 15.2250 |

For one case, expected shortage is $0.40(3)+0.20(6)+0.05(9)=2.85$ units.
Expected expiry is $0.10(3)=0.30$ units.
The expected mismatch cost is $2.00(2.85)+2.75(0.30)=6.525$ USD.
For two cases, expected shortage is $0.20(2)+0.05(5)=0.65$ units.
Expected expiry is $0.10(7)+0.25(4)+0.40(1)=2.10$ units.
The expected mismatch cost is $2.00(0.65)+2.75(2.10)=7.075$ USD.
The lowest expected cost is one case, not a simple rounding up of seven units to two cases.

Expected unit fill rate for one case is $(8.55-2.85)/8.55=66.67\%$.
Expected unit fill rate for two cases is $(8.55-0.65)/8.55\approx92.40\%$.
This is the ratio of expected filled units to expected requested units, not the mean of scenario-specific fill rates.
If the 90% target is a hard constraint, the planner selects two cases among these candidates.
If the target is aspirational, the commercial owner must explicitly decide whether the extra expected mismatch cost of USD 0.55 is worth the improvement.
A service target is not the same thing as customer demand or the contractual delivery promise.

### 6.4 Lot allocation and a timing stress

Suppose the two existing units are lot A, expiring at 16:00, and the eight newly received units are lot B, expiring at 20:00.
Both are released at 12:00, and the customer-life rule permits each pick before its expiry boundary.
There are three units of demand at 14:00 and seven units at 17:00.
FEFO allocates two units from A and one from B at 14:00.
At 17:00 it allocates the remaining seven units of B.
No units expire and no units are missed.
A FIFO instruction based only on receiving order happens to agree in this instance; it need not agree when expiry and receipt order differ.

Now change only the demand timing to one unit at 14:00 and nine units at 17:00.
FEFO sells one unit from A at 14:00.
The other A unit expires at 16:00; eight B units satisfy eight of the nine 17:00 requests.
The result is one expired unit and one missed unit, despite total stock and total demand both equaling ten units.
An aggregate one-period comparison would incorrectly report no expiry and no shortage.
This timing stress is a separate deterministic example, not another probability in the preceding distribution.

Finally, suppose the shipment arrives at 15:00 rather than 12:00 and three units are requested at 14:00.
Only the two existing units are available, so at least one unit is missed even if the receipt later contains enough units.
The size of the final loss also depends on subsequent demand and remaining life.
A supplier-delay scenario must therefore include both timing and lot eligibility, not only a reduced daily fill percentage.

### 6.5 Decision and handoff

The unconstrained economic recommendation is one case under the simplified benchmark.
The two-case plan meets the stated expected-fill target but increases expected expiry by 1.80 units.
The planner presents both outcomes and checks capacity before releasing either order.
The commercial owner resolves whether the target is binding; the food safety owner retains a veto on unsafe stock.
The receiver checks the actual lot life and the store executes FEFO on released eligible lots.
When the timing or life assumptions are materially wrong, the one-period ranking must be replaced by an event-based comparison.

## 7. Code, query, or calculation

The following standard-library Python 3 program reproduces the discrete benchmark and the lot-timing stress.
It requires no files, network, or third-party packages.
Quantities are integer units; prices are USD per unit; probabilities are dimensionless.
It assumes immediate full receipt for the benchmark and an exclusive expiry boundary for the timing example.
It is not a production order engine or food safety system.

```python
from dataclasses import dataclass

DEMAND = [(3, 0.10), (6, 0.25), (9, 0.40), (12, 0.20), (15, 0.05)]
EXISTING = 2
CASE = 4
UNDERAGE = 2.00
OVERAGE = 2.75


def evaluate(stock):
    shortage = sum(p * max(d - stock, 0) for d, p in DEMAND)
    expiry = sum(p * max(stock - d, 0) for d, p in DEMAND)
    cost = UNDERAGE * shortage + OVERAGE * expiry
    expected_demand = sum(d * p for d, p in DEMAND)
    fill = (expected_demand - shortage) / expected_demand
    return shortage, expiry, cost, fill


@dataclass
class Lot:
    name: str
    expiry_hour: int
    units: int


def run_fefo(demands):
    # Hours are local; stock expires before demand at the same hour.
    lots = [Lot('A', 16, 2), Lot('B', 20, 8)]
    lost = 0
    expired = 0
    for hour, requested in demands:
        for lot in lots:
            if lot.expiry_hour <= hour:
                expired += lot.units
                lot.units = 0
        remaining = requested
        for lot in sorted(lots, key=lambda x: (x.expiry_hour, x.name)):
            used = min(lot.units, remaining)
            lot.units -= used
            remaining -= used
        lost += remaining
    # Close at 20:00; remaining stock reaches its expiry boundary.
    for lot in lots:
        if lot.expiry_hour <= 20:
            expired += lot.units
            lot.units = 0
    return lost, expired


for cases in range(4):
    stock = EXISTING + CASE * cases
    shortage, expiry, cost, fill = evaluate(stock)
    print(f'{cases} {stock} {shortage:.2f} {expiry:.2f} '
          f'{cost:.4f} {fill:.2%}')

print('timed:', *run_fefo([(14, 1), (17, 9)]))
```

Expected output, columns cases, total stock, expected shortage, expected expiry, expected mismatch cost, and expected fill:

```text
0 2 6.55 0.00 13.1000 23.39%
1 6 2.85 0.30 6.5250 66.67%
2 10 0.65 2.10 7.0750 92.40%
3 14 0.05 5.50 15.2250 99.42%
timed: 1 1
```

The last line gives lost units followed by expired units.
The routine assumes events are supplied chronologically and that both lots are released at the start.
It omits delayed receipts, reservations, temperature excursions, customer minimum-life checks, substitutions, and multiple days.
A production implementation must add those events and verify lot conservation and time ordering.
It must also represent currency carefully, for example using integer cents or decimal arithmetic.
Use floating point here only to keep the educational calculation short.

## 8. Data, standards, and regulatory context

### 8.1 Minimum viable event model

A decision record needs store, SKU, unit of measure, order cutoff, timezone, and decision version.
A lot record needs supplier, lot identifier, expiry boundary, received quantity, accepted quantity, and release status.
A transaction record needs event time, recording time, event type, lot, quantity, and reason code.
A demand record needs requested quantity, fulfilled quantity, substitutions, availability, and cancellation status.
A supplier record needs order time, acknowledgment, expected arrival, actual arrival, accepted quantity, and rejection reason.
A policy record needs minimum-life requirements at receipt and delivery, effective dates, and owner.
A cost record needs cost version, currency, tax basis, and period.

These are logically distinct even if a physical implementation stores them together.
Event time tells you when the operation happened; recording time tells you when the information became available.
For an as-of backtest, corrections entered after the decision cannot be fed to the earlier model as known facts.
Use immutable event identifiers and idempotent ingestion to avoid duplicating receipts.
Preserve original and corrected values rather than silently overwriting history.

The conservation identity is opening units plus accepted receipts plus eligible returns, less sales, transfers out, disposals, and other legitimate outflows, plus signed adjustments, equals closing units.
Transfers in are receipts only if they are not already counted in accepted receipts.
Returns require safety inspection before they return to the sellable pool.
Reconcile the physical ledger separately from the sellable ledger.
A quarantine changes eligibility but does not necessarily change physical on-hand.

### 8.2 Identifier and traceability standards

GS1 General Specifications describe globally used identification and barcode standards, including Global Trade Item Numbers and application identifiers for attributes such as batch or lot and expiry date.
Their use can improve interoperability between supplier labels, receiving systems, and lot records.
A barcode standard does not guarantee that a label is accurate, that a lot was stored safely, or that a picker actually follows FEFO.
Record the applicable specification version and partner agreements; encoding details and date interpretation must be validated for the implementation.

GS1 Electronic Product Code Information Services (EPCIS) is a standard for sharing visibility-event data across parties.
It can describe what, when, where, why, and how of traceability events, depending on the event and implementation.
A local store does not need to deploy EPCIS to implement a basic lot ledger.
If it does share EPCIS events, validate identifiers, event times, business steps, and corrections against the agreed standard and partner profile.

### 8.3 Date labels and food safety

In the United States, the US Department of Agriculture Food Safety and Inspection Service explains that food product dating generally concerns quality and that federal law does not require dates on food products except infant formula, with important product- and jurisdiction-specific exceptions.
That statement does not grant permission to sell unsafe food or override state rules, supplier agreements, or a company's safety plan.
The US Food and Drug Administration (FDA) Food Code is a model for state, local, tribal, and territorial authorities; adoption and modifications vary by jurisdiction.
Check the locally adopted code, product category, and current legal advice before setting a sale or discard rule.

In the European Union, Regulation (EU) No 1169/2011 distinguishes date of minimum durability from use-by dates for certain highly perishable foods.
Article 24 specifies the use-by concept for foods highly perishable from a microbiological point of view and its safety implications after that date.
Consult the current consolidated regulation, product-specific rules, and competent authority for an operational interpretation.
A best-before date is not a universal authorization to sell an unsafe product, and a use-by label must not be reclassified by an inventory model.

FDA guidance and regulations under the Food Safety Modernization Act address food safety and, for covered entities and foods, traceability requirements.
The applicability and effective compliance deadlines can change; check the current FDA rule and official updates before designing a legal compliance program.
This course does not determine whether a particular store, food, or event is covered.
FEFO supports disciplined handling but is not itself proof of legal compliance or a substitute for temperature control.

### 8.4 Operational measurement

Capture temperature measurements with sensor identity, time, units, calibration status, and excursion handling.
A temperature excursion does not automatically change expiry by a universal number of hours.
A qualified food safety process determines disposition based on the product and validated controls.
Never let a replenishment model infer that an unsafe lot is sellable because forecast demand is high.

For waste reporting, distinguish physical mass, item count, inventory cost, and disposal cost.
If converting units to kilograms, retain pack weight and whether the measure includes packaging.
For supplier credits, record approved versus merely requested amounts.
For donation or redistribution, record accepted quantity and net cost separately from sale revenue.
Avoid reporting a write-off reversal as negative physical waste without explaining the accounting event.

## 9. Industry practice and case evidence

The following examples are documented public evidence, not a claim that the synthetic case represents any company.
Each source has a different evidentiary role: standards explain data exchange, regulators explain jurisdictional rules, and research studies evaluate particular settings.
Results do not automatically transfer between a supermarket, a warehouse, and a quick-commerce dark store.

### 9.1 Retail measurement and waste prevention

Tesco publishes food waste data and describes its measurement and reduction approach in its public sustainability reporting.
This establishes that a large food retailer can publish operational food-waste measures and associated actions.
It does not establish the result of a particular FEFO algorithm or prove the loss function in this unit.
[INFERENCE] A dark-store operator can use the same discipline of defining boundaries and tracking changes over time, while needing a different short-horizon inventory model.
Compare numerator, denominator, product coverage, and reporting years before comparing retailers.

WRAP's food surplus and waste resources discuss the food-use hierarchy and prevention, redistribution, and other treatment pathways.
The hierarchy is a management framing rather than permission to bypass food safety controls.
[INFERENCE] An operator should prioritize preventing avoidable excess before relying on donation or waste processing as a universal solution.
Donation capacity, handling time, eligibility, and local requirements can constrain the achievable recovery value.

### 9.2 Evidence on perishability and inventory decisions

Nahmias's review of perishable inventory theory documents a broad literature on inventory models with finite lifetimes.
Its importance here is conceptual: age matters, and common stationary inventory formulas need assumptions before they can be applied.
A review paper does not supply store-specific demand, cost, supplier reliability, or legal rules.
A planner must validate those inputs locally.

The Food and Agriculture Organization of the United Nations (FAO) reports on food loss and waste across supply chains.
Its global or regional estimates describe broad measurement categories, not a particular SKU's dark-store expiry rate.
Use its definitions to structure measurement, then collect the relevant local events for this decision.
Avoid converting a published system-wide loss percentage directly into a replenishment parameter.

### 9.3 Operating patterns and trade-offs

Some retailers and fulfillment operations use date-aware rotation, markdowns, and redistribution alongside ordering controls.
These are distinct interventions: rotation changes which unit is sold, markdown changes price and potentially demand, and redistribution changes a disposition channel.
The effect of any intervention depends on product, customer behavior, lead time, and execution.
A markdown may reduce expiry while lowering contribution or shifting demand from full-price sales.
A transfer may save one store's stock but increase labor, transport cost, temperature exposure, and waste at another store.

[INFERENCE] A short delivery promise makes a precise inventory-availability signal especially valuable because substitutes and delayed replenishment may not preserve the original customer promise.
That inference follows from the decision structure, not from a causal estimate in the sources cited here.
A local pilot should measure the actual impact rather than adopting a generic improvement percentage.
The pilot should retain safety checks and predefine its comparison window and success metrics.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| FIFO mistaken for FEFO | Later-expiry lot picked while earlier lot is eligible | Receipt order differs from expiry order | Avoidable expiry | Display eligible lot priority and audit picks |
| Expired stock included | Available quantity exceeds physically eligible quantity | Aggregate balance ignores date | False availability and failed picks | Time-bound lot eligibility |
| Held stock sold | Quarantine status missing in picker feed | Safety state not propagated | Unsafe fulfillment | Synchronous block and safety escalation |
| Customer life ignored | Complaints about short remaining life | Pick check omits delivery threshold | Broken freshness commitment | Check projected delivery life |
| Sales treated as demand | Forecast drops after stockouts | Observed sales are censored | Repeated under-ordering | Estimate lost demand and log availability |
| Demand timing omitted | Morning stockouts despite sufficient daily units | Late receipt cannot cover early demand | Missed orders | Simulate intraday arrivals |
| Constant lead time assumed | Late supplier arrivals cluster in disruption | Average hides tail and dependence | Understated shortage risk | Model joint supply-demand scenarios |
| Short-life receipts hidden | Waste rises for a supplier lot cohort | Ordered units treated as equally usable | Excess write-offs | Track life at receipt and rejection rules |
| Disposal cost double-counted | Costs do not reconcile to transactions | Purchase and write-off treated twice | Biased order choice | Establish incremental cost bridge |
| Excessive salvage assumed | Paper recovery exceeds actual credit | Uncertain donation or return counted as cash | Systematic over-ordering | Use realized net recovery scenarios |
| Pack rounding without evaluation | Rounded-up order has higher loss | Asymmetric discrete distribution | Avoidable expected cost | Compare all feasible neighbor packs |
| Receipt timestamp confused | Stock appears available before quality release | Truck arrival used instead of pickable time | False service coverage | Separate arrival, inspection, and release |
| Duplicate receipt | Stock jumps twice for one shipment | Replayed message lacks idempotency | Phantom stock | Unique event identifiers and reconciliation |
| Wrong expiry boundary | Midday expiry treated as end of local day | Missing timezone or date convention | Invalid pick or premature disposal | Store instant, timezone, and boundary rule |
| Scenario weights invalid | Probabilities do not sum to one | Poor scenario generation | Misleading expected loss | Validate and document weights |
| Stale supplier contract | Order ignores changed minimum or credit | Master data not versioned | Rejected orders or miscosted waste | Effective-dated commercial terms |
| Uncontrolled markdown | Waste falls but contribution collapses | Discount shifts full-price demand | False improvement | Measure incremental margin and substitution |
| Blind KPI optimization | Waste drops while stockouts rise | One metric is optimized alone | Poor customer and economic outcome | Monitor service, cost, safety, and waste jointly |
| Overrides unlogged | Large recurring manual changes | Local information cannot be evaluated | Untraceable decisions | Log rationale and outcomes |
| Leakage in backtest | Unrealistically good historical performance | Future corrections used as inputs | Failed deployment | Reconstruct point-in-time snapshots |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Demand distribution is representative | Computes expected shortage | Promotion or competitor change | Rolling-origin calibration by regime |
| Discrete scenarios sum to one | Defines expected values | Missing or overlapping scenarios | Validate probabilities before evaluation |
| Demand is observed or estimated | Calculates lost sales | Stockouts censor sales | Compare availability, substitutions, and page views |
| Benchmark receipt is immediate and complete | Makes one-period formula valid | Delay, rejection, short shipment | Compare actual release times and quantities |
| Benchmark units share one end of life | Reduces state to total stock | Multiple expiry lots | Run event-based lot simulation |
| Demand is lost rather than backordered | Defines shortage cost | Customers accept later delivery | Measure customer response and policy |
| Cost per marginal unit is stable | Supports linear mismatch loss | Tiered pricing or fixed delivery fee | Recompute by feasible quantity |
| Salvage is zero in example | Fixes overage cost | Credible supplier credit or recovery | Audit realized net proceeds and timing |
| Existing stock cost is sunk | Enables incremental order comparison | Return credit varies with order | Include decision-dependent opportunity cost |
| Delivery-life policy permits illustrated picks | Allows FEFO stress arithmetic | Longer minimum-life requirement | Compare lot expiry with promised delivery time |
| Capacity can hold chosen order | Keeps one-SKU problem feasible | Shared cold-room space is full | Check volume at each receipt event |
| No substitution in benchmark | Keeps shortage calculation transparent | Customers switch products | Estimate incremental substitute contribution |
| Disposal cost is constant | Treats each excess unit equally | Batch pickup or weight-based tariffs | Compare invoices and quantities |
| Target is assessed over expected units | Defines 90% fill calculation | Contract uses order-level service | Recalculate using contractual denominator |
| Lot records are accurate | Makes FEFO executable | Missing scans and count errors | Cycle count and scan-to-lot reconciliation |
| No additional future replenishment | Defines terminal exposure | Intraday or next-day orders arrive | Extend horizon with new order events |

The assumptions are not all equally dangerous.
For a slow-moving fresh product, remaining life and demand censoring can dominate the order decision.
For a high-volume product during a storm, delivery delay and correlated demand may be more important.
Rank assumptions by how much the recommendation changes under plausible alternatives.
Do not represent sensitivity ranges as measured confidence intervals unless they are statistically estimated.

## 12. Exercises

Work independently before consulting the answers.
For each calculation, state the unit, horizon, and assumptions.

### Exercise 1 — Decision contract (15 minutes)

A store orders chilled meals at 09:00 for a nominal 13:00 receipt.
The next ordering opportunity is tomorrow at 09:00.
A customer requires two hours of remaining life at delivery.
**Deliverable:** one-page decision contract with time fields, owner, eligibility, and action.
**Acceptance:** distinguish truck arrival from pickable release, forecast from order, and receipt life from delivery life.
Include an escalation if no safe lot meets the promise.

### Exercise 2 — Lot allocation (15 minutes)

At 12:00, lot A has three units expiring at 15:00 and lot B has four units expiring at 20:00.
Two units are requested at 14:00 and four units at 16:00.
Use an exclusive expiry boundary; assume all pre-expiry picks satisfy the customer-life rule.
**Deliverable:** a lot-by-event ledger showing fills, expiry, and missed units.
**Acceptance:** preserve quantity, use FEFO, and explain why aggregate stock is misleading.

### Exercise 3 — Marginal economics (15 minutes)

A product sells for USD 7.00, costs USD 3.20, and has avoidable selling cost USD 0.80 per unit.
An unsold unit has net salvage USD 0.50 and disposal cost USD 0.20.
Assume no substitution, fixed charges, or other impacts.
**Deliverable:** underage cost, overage cost, and critical fractile.
**Acceptance:** use incremental contribution, do not count purchase cost twice, and show the formula.

### Exercise 4 — Feasible packs (20 minutes)

Use the worked example's demand distribution and cost parameters.
The supplier changes case size to five units; existing stock remains two units.
Consider orders of zero, five, and ten units only.
**Deliverable:** expected shortage, expiry, and mismatch cost for each feasible order and a recommendation.
**Acceptance:** compare all three candidates rather than rounding the unrestricted optimum mechanically.
Also identify whether the cost-minimizing choice meets the 90% expected-unit-fill target.

### Exercise 5 — Data model (20 minutes)

Design a minimal event model for supplier receipt, safety hold, release, pick, expiry, and disposal.
**Deliverable:** table or diagram with keys, event time, recording time, quantities, and statuses.
**Acceptance:** demonstrate physical and sellable balances, support as-of reconstruction, and prevent duplicate receipt processing.
Name the owner of a discrepancy investigation.

### Exercise 6 — Censored sales (15 minutes)

A SKU sells four units every day for a week, but it is out of stock after noon on five days.
An analyst proposes forecasting four units per day because sales are stable.
**Deliverable:** critique and a data collection plan.
**Acceptance:** identify the censoring mechanism, at least two useful evidence sources, and an uncertainty limitation.
Do not invent a precise missing-demand number.

### Exercise 7 — Intervention design (25 minutes)

A team proposes a 20% markdown for lots nearing expiry.
**Deliverable:** a pilot design with a comparison, randomization or credible alternative, outcome metrics, and guardrails.
**Acceptance:** measure incremental contribution and waste; address cannibalization, timing, safety, and heterogeneous store demand.
Specify who can stop the pilot and how adverse effects are detected.

### Exercise 8 — Ownership and escalation (20 minutes)

A supplier truck arrives late with a shorter-life lot than acknowledged.
The forecast predicts a high evening peak, but the lot's status is pending safety inspection.
**Deliverable:** a decision-rights matrix and sequence of actions.
**Acceptance:** identify who may release, quarantine, reject, revise the promise, and order a replacement.
The urgency of demand must not override the safety hold.

## 13. Self-check answers

### Answer 1 — Decision contract

The action is an integer supplier order placed at the 09:00 cutoff in the store's stated timezone.
The supplier's 13:00 arrival is an estimate, not a guarantee that the product is released for picking at 13:00.
The exposure window extends until the next feasible replenishment becomes usable, subject to expiry and delivery promises.
The forecast is a demand distribution; the order is the plan; the freshness requirement is part of a customer commitment.
Record the delivery instant and require at least two hours of remaining life then, under the defined boundary convention.
The planner proposes the order, the supplier acknowledges it, and the safety owner controls held stock.
If no lot is safe and eligible, show unavailable or offer an approved alternative; do not bypass the hold.

### Answer 2 — Lot allocation

At 14:00, allocate two units from A; A has one unit and B has four units remaining.
At 15:00, A's remaining unit expires and leaves the sellable ledger.
At 16:00, allocate all four requested units from B.
Totals are six filled units, one expired unit, and zero missed units.
Opening stock of seven units equals six sold plus one expired unit.
A time-blind ledger can miss when the remaining A unit stopped being available; with later demand it could also understate shortage.

### Answer 3 — Marginal economics

Underage cost is $7.00-3.20-0.80=3.00$ USD per unit.
Overage cost is $3.20+0.20-0.50=2.90$ USD per unit.
The critical fractile is $3.00/(3.00+2.90)\approx0.5085$.
Use the smallest demand quantile reaching 50.85% only under the one-period assumptions.
A realized salvage or disposal amount different from the assumptions changes the result.

### Answer 4 — Feasible packs

For total stock two units, expected shortage is 6.55, expiry is zero, and mismatch cost is USD 13.10.
For total stock seven units, expected shortage is $0.40(2)+0.20(5)+0.05(8)=2.20$ units.
Expected expiry is $0.10(4)+0.25(1)=0.65$ units.
The cost is $2(2.20)+2.75(0.65)=6.1875$ USD.
For total stock twelve units, expected shortage is $0.05(3)=0.15$ units.
Expected expiry is $0.10(9)+0.25(6)+0.40(3)=3.60$ units.
The cost is $2(0.15)+2.75(3.60)=10.20$ USD.
Order five units for the minimum expected mismatch cost.
Its expected unit fill is $(8.55-2.20)/8.55\approx74.27\%$, below the 90% target.
Ten ordered units achieve approximately 98.25% expected unit fill and are the least-cost feasible option among those meeting that target.

### Answer 5 — Rubric

A strong design has a stable event identifier, lot and SKU keys, store, signed quantity, status, event time, and recorded time.
It separates physical movement from eligibility changes, and supports corrections without erasing the original event.
It makes receipt idempotent and reconciles openings, movements, and closings.
The store inventory controller investigates discrepancies with receiving and the safety owner as needed.
A weak design stores only today's aggregate available quantity and cannot reconstruct past decisions.

### Answer 6 — Rubric

Four sales are a lower bound on desired demand on censored days, not proof of four units of demand.
Useful evidence includes in-stock intervals, product views, unsuccessful requests, substitutions, comparable stores, and uncensored days.
These sources are imperfect: views are not purchases, comparable stores differ, and substitutions can conceal or create demand.
Estimate a range or a model with documented assumptions; do not manufacture a point estimate from the information given.

### Answer 7 — Rubric

A strong pilot randomizes eligible comparable store-SKU-time units where operationally feasible, with a stable control policy.
It records assignment, eligible stock, prices, customer demand, full-price cannibalization, disposal, and net contribution.
It predefines the evaluation period and accounts for clustered outcomes and interference between substitutes.
Safety and minimum-life rules remain unchanged, and the safety owner can stop unsafe handling immediately.
The commercial owner stops or modifies the pilot if predeclared service or contribution guardrails are breached.
A simple pre/post reduction in waste is weak evidence when demand and assortment also changed.

### Answer 8 — Rubric

The receiver records arrival, quantity, temperature or condition checks, lot identity, and observed remaining life.
The food safety owner or authorized delegate releases, quarantines, or directs rejection under the approved procedure.
The planner revises the replenishment proposal; the supplier confirms any replacement and timing.
The commercial or customer experience owner adjusts displayed availability or promise under the established policy.
The store manager coordinates execution and escalation, but no one should make pending stock sellable merely because demand is high.
Keep timestamps and reasons so the eventual outcome can be reconstructed.

## 14. Interview questions

1. **Why can a high service target increase food waste?**
   Strong: explains demand uncertainty, extra stock, finite life, and the cost frontier.
   Weak: says safety stock is always waste or that service has no cost.
2. **When does FEFO differ from FIFO?**
   Strong: gives lots whose expiry order differs from receipt order and filters for eligibility.
   Weak: assumes the earliest received lot necessarily expires first.
3. **How do you compute remaining life for delivery?**
   Strong: uses expiry instant, projected delivery instant, timezone, and customer threshold.
   Weak: subtracts calendar dates with no boundary or delivery-time convention.
4. **How would you estimate demand during a stockout?**
   Strong: identifies censoring and triangulates imperfect signals with uncertainty.
   Weak: treats observed sales as unconstrained demand.
5. **What does the critical fractile mean?**
   Strong: relates marginal underage and overage costs to a one-period demand quantile and its assumptions.
   Weak: calls it a universal contractual fill rate.
6. **How do case packs change the answer?**
   Strong: evaluates feasible quantities using the discrete distribution and constraints.
   Weak: rounds every continuous recommendation up automatically.
7. **Why simulate time and lots?**
   Strong: explains early demand, late receipts, expiry, and eligibility at each event.
   Weak: relies only on end-of-day total inventory.
8. **What is the difference between expiry and write-off?**
   Strong: separates physical units and disposition from accounting recognition and credits.
   Weak: equates all financial shrink with food physically discarded.
9. **How would you handle a held lot during a demand spike?**
   Strong: preserves the safety block, escalates to the authorized owner, and changes availability if needed.
   Weak: releases the lot based on the predicted profit from sales.
10. **How would you evaluate a markdown?**
    Strong: measures incremental waste and contribution against a credible counterfactual, including cannibalization.
    Weak: counts discounted units sold as pure incremental benefit.
11. **How do you prevent backtest leakage?**
    Strong: reconstructs the as-of information set including corrections and forecast vintages.
    Weak: uses the latest corrected records for every historical order decision.
12. **What should appear in a decision handoff?**
    Strong: provides recommended and feasible quantity, assumptions, service and waste consequences, ownership, and exception path.
    Weak: delivers only a point forecast with no timing or operational context.

## 15. Key terms

The definitions in [Section 4](#4-definitions-and-vocabulary) provide units and common misuses.

- **As-of snapshot:** information known at a specific decision origin.
- **Backorder:** demand intentionally fulfilled after the initial requested time.
- **Best-before date:** quality-oriented date marking subject to local law.
- **Case pack:** supplier ordering increment in units.
- **Contribution:** revenue less relevant avoidable costs.
- **Critical fractile:** marginal underage cost divided by total marginal mismatch cost.
- **Cycle service level:** probability of no stockout over a defined cycle.
- **Demand censoring:** observed sales limited by availability.
- **Expiry boundary:** last permitted instant under an explicit policy.
- **FEFO:** first-expiry-first-out allocation among eligible lots.
- **FIFO:** first-in-first-out allocation by receipt order.
- **Fill rate:** fulfilled requested units divided by requested units.
- **Forecast:** probability statement about an uncertain future quantity.
- **Inventory position:** a declared accounting of available and on-order inventory less obligations.
- **Lead time:** elapsed time from released order to usable receipt.
- **Lot:** traceable grouping with shared product and date attributes.
- **Markdown:** approved price reduction before disposition.
- **Minimum life:** required remaining shelf life at a named checkpoint.
- **Overage cost:** incremental loss from an unnecessary unit.
- **Protection period:** interval until the next replenishment opportunity can supply usable stock.
- **Quarantine:** non-sellable hold pending disposition or release.
- **Remaining shelf life:** time between a stated instant and the expiry boundary.
- **Review interval:** elapsed time between order decisions.
- **Salvage:** net value recovered from an unsold unit through an approved channel.
- **Scenario:** joint possible path of demand, supply, and eligibility with a probability.
- **Sellable available:** eligible on-hand inventory not already committed.
- **Shrink:** inventory loss from identifiable or unexplained causes.
- **Stockout:** inability to supply from eligible inventory at a specified time.
- **Substitution:** fulfillment with an alternative product under an explicit rule.
- **Underage cost:** incremental loss from one unit of unmet demand.
- **Use-by date:** date marking with product- and jurisdiction-specific safety implications.
- **Write-off:** accounting recognition of lost inventory value.

## 16. Sources

Access regulatory pages and current consolidated law before operational use; regulatory discussion in this unit is dated November 2026 and is educational, not legal advice.

1. GS1, [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications): identification, application identifiers, and barcode rules. Standard; implementation and version matter.
2. GS1, [EPCIS standard](https://www.gs1.org/standards/epcis): visibility-event data sharing. Standard; not a substitute for physical controls.
3. US FDA, [FDA Food Code](https://www.fda.gov/food/fda-food-code/food-code-2022): model code and jurisdictional adoption context. Regulatory source; local law controls.
4. US FDA, [Food Traceability Final Rule](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods): scope, requirements, and current implementation information. Check the page for changes and deadlines.
5. USDA Food Safety and Inspection Service, [Food Product Dating](https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/food-product-dating): US date-label explanations and scope.
6. European Parliament and Council, [Regulation (EU) No 1169/2011](https://eur-lex.europa.eu/eli/reg/2011/1169/oj), especially Articles 9 and 24 and Annex X: food information and date marking; consult the current consolidated text.
7. Nahmias, S. (1982), “Perishable Inventory Theory: A Review,” *Operations Research* 30(4), 680–708, [DOI: 10.1287/opre.30.4.680](https://doi.org/10.1287/opre.30.4.680). Peer-reviewed theoretical review; context and assumptions differ by model.
8. FAO, [The State of Food and Agriculture 2019: Moving forward on food loss and waste reduction](https://www.fao.org/3/ca6030en/ca6030en.pdf). Official international report; aggregate estimates are not SKU-level inputs.
9. WRAP, [Food and drink surplus and waste](https://www.wrap.ngo/taking-action/food-drink): practical waste prevention and redistribution resources; not a store-specific causal estimate.
10. Tesco, [Food waste data](https://www.tescoplc.com/sustainability/planet/food-waste/): company-reported measurement and actions; boundaries and reporting year matter.

## 17. Links

- [Module 22 index](_index.md) for topic sequence and scope.
- [Promise tiers and dark-store design](01_promise-tiers-dark-stores.md) for the physical promise and operating footprint.
- [Short-horizon forecasting](02_short-horizon-forecasting.md) for demand distributions and as-of backtesting.
- [Module 22 survey](../../../course/22_quick_commerce_on_demand_delivery.md) for the full operating model and lab context.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md) for broader simulation practice.

