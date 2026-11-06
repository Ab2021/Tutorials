# Cross-sector comparison: transfer the decision logic, not the policy

> Module 29 — Industry Electives: Sector Primers  
> Topic 07 of 08  
> Estimated study time: 4 hours plus the lab  
> Prerequisites: Modules 14, 15, and 28; one sector primer  
> Version 2026.5

## 1. Why this topic exists

A planner moving from consumer packaged goods (CPG) to medicine can recognize a replenishment problem immediately.
That familiarity is useful and dangerous.
A retail item can be available in recorded inventory but absent from a shelf.
A medicine can be physically present but unavailable for use because its quality status is unresolved.
A vehicle component can be on the line but wrong for the next vehicle in sequence.
A wafer can be in a fab but still require many process steps before it becomes usable chips.
A relief commodity can be in a warehouse but inaccessible to a community because a route is closed.
A garment can be sellable but worth less after the season has passed.
Each situation contains inventory, time, uncertainty, and a decision.
The meaning of a usable unit, a successful delivery, and a justified intervention differs.

This topic teaches transfer across six sectors:
- CPG and retail;
- pharmaceuticals and temperature-controlled medicines;
- automotive;
- semiconductors;
- humanitarian logistics; and
- fashion and apparel.

The operational sponsor may be a retail category buyer, a pharmaceutical quality leader, an automotive material planner, a semiconductor capacity manager, a humanitarian response coordinator, or a fashion merchandiser.
These roles buy or authorize different outcomes.
They do not necessarily own the budget, control the physical network, or speak for the ultimate recipient.
A successful brief names each party instead of treating the supply chain as one decision maker.

The transferable asset is a decision discipline: define alternatives, feasible actions, information, consequences, uncertainty, and accountability.
The sector-specific work is to define whose outcome counts, which goods can be used, which constraints are non-negotiable, and which evidence justifies action.
An inventory formula can be reused only when its assumptions and unit definitions remain valid.
A service target copied from another industry is not a strategy.
A cost-only optimization can be unacceptable even if its mathematics is correct.

Consider six expensive mistakes:
- Measuring only shipped cases misses empty retail shelves.
- Counting a quarantined pharmaceutical lot as available can create an unsafe allocation.
- Counting the correct automotive part number without its sequence can still stop assembly.
- Treating a wafer start as a finished die ignores yield and cycle time.
- Allocating relief according to registered demand can miss unregistered, inaccessible households.
- Optimizing full-season garment units without size-level demand can leave the wrong stock after markdown.

The synthetic worked example and lab compare a common allocation calculation across settings.
They are illustrations, not recommendations or descriptions of any named organization's results.
Numerical values are deliberately constructed; they are not estimates of actual sector averages.
Regulatory discussion is educational, dated, jurisdiction-dependent, and not legal advice.

## 2. Learning outcomes

After this topic you can:

1. **Map** the decision owner, buyer or authorizer, user, and affected party for each sector; Exercise 1 assesses the map.
2. **Separate** transferable analytical structure from sector-specific constraints and definitions; Exercise 2 assesses the classification.
3. **Calculate** qualified available inventory, allocation, and unmet requirement under stated priorities; Exercises 3 and 4 assess the arithmetic.
4. **Critique** a service metric whose denominator or time window changes between contexts; Exercise 5 assesses the critique.
5. **Design** an admissible intervention using data lineage, quality release, capacity, and equity controls; Exercise 6 assesses the design.
6. **Adapt** the executable allocation lab without mistaking its results for a validated deployment; Exercise 7 assesses the adaptation.
7. **Recommend** a transfer pilot with accountability, uncertainty checks, and stop conditions; Exercise 8 assesses the decision brief.
8. **Distinguish** documented evidence, inference, and synthetic illustration in a cross-sector claim; Exercise 9 assesses the evidence ledger.

A forecast is an estimate of a future outcome under stated information and assumptions.
A plan is a proposed set of actions over a specified horizon.
A target expresses a desired outcome.
A commitment is an accepted obligation under a defined agreement or authorization.
These four things may differ in every sector.
A forecast does not guarantee access, release, demand, capacity, or funding.
A target does not authorize a shipment.
A proposed plan does not establish a contractual obligation.

## 3. The core mental model

```text
                  TRANSFERABLE DECISION SKELETON
 Need or demand --> state of network --> feasible alternatives --> consequences
       |                  |                      |                    |
       v                  v                      v                    v
 evidence quality   qualified inventory   constraints / authority   outcome and harm
       \____________________ compare under uncertainty _______________/
                                      |
                                      v
                    authorize --> execute --> observe --> revise
                                      ^
                                      |
             SECTOR ADAPTER: units, horizon, quality, rights,
             network, cost, priority, regulation, and measurement
```

The skeleton does not imply that all objectives collapse to one currency value.
Some organizations use a cost objective subject to hard safety and eligibility constraints.
Others use a lexicographic rule: satisfy safety first, prioritize critical needs second, and compare cost only among feasible options.
The order must be authorized by the responsible decision maker and, where necessary, legal or ethical governance.

### The six contexts

| Sector | Immediate decision | Typical economic buyer or authorizer | Ultimate user or affected party | Distinct network feature |
|---|---|---|---|---|
| CPG/retail | Set an order and shelf allocation around a promotion | Retail category buyer and replenishment planner, with supplier counterpart | Shopper | Supplier, distribution center, store backroom, shelf |
| Pharma | Release and allocate temperature-sensitive medicine | Supply planner under quality-unit and clinical governance | Patient and care provider | Lot, serial or pack, cold-chain lanes, licensed trading partners |
| Automotive | Secure and sequence constrained components | Original equipment manufacturer (OEM) material planner and purchasing team | Vehicle customer and production workforce | Multi-tier bill of materials and line sequence |
| Semiconductor | Reserve wafer capacity and allocate qualified output | Fabless buyer, foundry capacity owner, or integrated producer | Device producer and eventual device user | Wafer fabrication, test, packaging, long work-in-process |
| Humanitarian | Route eligible supplies to priority populations | Response lead and authorized logistics or program owner | Affected people, including unregistered and inaccessible groups | Donor funding, multiple organizations, changing access |
| Fashion | Buy and allocate style-color-size inventory | Merchandiser or retail buying team | Shopper and potentially return-processing workers | Seasonal assortment, global sourcing, stores, e-commerce and returns |

These are common patterns, not universal organization charts.
A pharmaceutical supply planner cannot release a suspect lot merely because it is needed.
A humanitarian logistics coordinator may move goods only within the mandate and funding rules of the response.
A semiconductor buyer may receive allocation from a foundry rather than control the factory schedule.
A fashion merchant's purchase commitment may precede reliable sales observations by months.
A CPG supplier may recommend a promotion without controlling the retailer's shelf price.
An automotive supplier may serve several customers whose assembly sequences are not visible to it.

### What transfers

The analyst can reuse a flow balance, a time-phased inventory ledger, a scenario framework, and a decision review cadence.
The analyst can reuse a hierarchy of measures: inputs, process reliability, outputs, and outcomes.
The analyst can reuse an experiment design that compares alternatives and reports uncertainty.
The analyst can reuse a data contract requiring units, timestamps, provenance, and ownership.
The analyst can reuse escalation for exceptions that invalidate a model's assumptions.
The analyst cannot reuse a numerical safety stock, threshold, or priority ranking merely because the columns look similar.

### What changes

The decision boundary changes from maximizing commercial contribution to preserving qualified supply, production continuity, equitable access, or seasonal value.
The item grain changes from consumer unit to lot, serial, sequenced component, wafer or die, relief kit, or style-color-size.
The clock changes from hourly shelf availability to remaining shelf life, line sequence, fab cycle time, response access window, or selling season.
The bottleneck changes from shelf or shelf-replenishment labor to quality release, tooling, cleanroom capacity, access, or product-specific sell-through.
The buyer may not be the person who bears the consequences of the decision.
This distinction is especially important when money, authority, and need sit in different institutions.

## 4. Definitions and vocabulary

| Term | Working definition and unit | Common misuse |
|---|---|---|
| Decision | Selection among feasible actions by an authorized owner at a specified time | Treating a dashboard as a decision |
| Decision owner | Person or role accountable for choosing and documenting the action | Assuming the data scientist owns all consequences |
| Buyer | Party purchasing or authorizing goods or capacity | Assuming the buyer is always the end user |
| Beneficiary | Person intended to benefit from a humanitarian intervention | Assuming a registration record captures every affected person |
| CPG | Frequently purchased consumer packaged goods | Treating shelf availability as identical to warehouse fill |
| OEM | Original equipment manufacturer, such as a vehicle producer | Assuming an OEM directly sees every upstream tier |
| SKU | Stock keeping unit, an operationally distinguished item | Aggregating sizes or packaging levels without conversion |
| Lot | Quantity with a defined production or distribution history | Treating all lots as equally usable |
| Serial | Identifier for an individual item or package when assigned | Treating a batch identifier as an individual serial |
| Style-color-size | Fashion assortment grain with distinct demand and stock | Using a style total to conceal unavailable sizes |
| Wafer start | Wafer entering a defined fabrication process | Treating the start as finished good die |
| Good die | Tested acceptable semiconductor units from wafer processing | Applying yield twice or ignoring test stage |
| Yield | Acceptable output divided by a specified input population, percent | Comparing yields with different denominators |
| Work in process (WIP) | Items started but not yet completed at the defined stage | Adding all WIP to immediately available inventory |
| Lead time | Elapsed time from defined trigger to usable receipt, hours, days, or weeks | Excluding quality release or inaccessible travel time |
| Cycle time | Time taken to traverse a specified process or stage | Confusing average fab cycle time with committed ship date |
| Inventory position | On hand plus eligible on-order supply minus defined obligations, in units | Including quarantined goods or counting receipts twice |
| Qualified available supply | Stock that satisfies the relevant release, eligibility, timing, and access rules | Equating every recorded unit with usable supply |
| Quarantine | Restricted inventory pending a decision or disposition | Interpreting quarantine as confirmed product failure |
| Temperature excursion | Recorded condition outside a specified temperature range for a period | Automatically declaring every excursion product failure without assessment |
| Shelf life | Time for which a product remains within its defined usable period under specified conditions | Ignoring remaining life at the destination |
| FEFO | First-expire, first-out allocation rule | Applying it ahead of a mandatory release or eligibility rule |
| Just in time (JIT) | Supplying material close to the time of use | Treating low inventory as a guarantee against disruption |
| Just in sequence (JIS) | Supplying parts in the order needed for assembly | Treating aggregate component availability as sequence readiness |
| Allocation | Assignment of constrained supply to eligible demands | Assuming a proposed assignment is delivered |
| Priority rule | Authorized ordering of eligible demand when supply is scarce | Substituting the model author's preference for governance |
| Fill rate | Supplied quantity divided by requested quantity under a defined time and scope | Comparing measures with different denominators |
| On-shelf availability | Share of specified item-location-time observations available to a shopper | Replacing observation with recorded backroom stock |
| Need | Required goods or services for an affected population under an assessment method | Treating recorded requests as complete needs |
| Access | Practical ability to reach and serve a population or location | Assuming a route open yesterday remains open today |
| Coverage | People or need served divided by a defined eligible population or requirement | Hiding excluded groups in the denominator |
| Stockout | Period or event in which a defined product is unavailable at a defined place | Treating low sales during a stockout as low underlying demand |
| Lost sales | Demand not fulfilled and not recovered, estimated or observed under a stated method | Counting all unmet requests as permanently lost |
| Markdown | Reduction in selling price during a defined period | Treating markdown revenue as full-price contribution |
| Return | Item sent back after sale or distribution, subject to a disposition process | Assuming every return is immediately resalable |
| Counterfactual | Estimated outcome without the intervention | Calling last period's observed result a proven counterfactual |
| Forecast | Estimate of future demand or other outcome under assumptions | Calling an estimate a binding order |
| Plan | Intended operational actions and timing | Calling an unapproved plan an executed shipment |
| Target | Desired level of performance | Calling the target an observed result |
| Commitment | Accepted obligation under an agreement or authority | Treating a scenario as a contractual commitment |
| Traceability | Ability to follow defined items and events through a supply chain | Assuming an identifier alone proves every event |
| Data lineage | Record of a data element's source, transformation, and use | Treating a copied spreadsheet as primary evidence |
| Constraint | Condition that a feasible decision must satisfy | Converting safety requirements silently into soft cost penalties |
| Scenario | Coherent set of assumed future conditions | Presenting one scenario as a probability distribution |
| Equity | Fair treatment or allocation according to an explicitly justified principle | Assuming equal quantities imply equal access or need satisfaction |

## 5. The mechanics, step by step

The procedure below is a transfer assessment, not a request to standardize each sector's rules.
Complete it before porting a model, target, or operating process.
At each step capture the input, decision, output, and branch.

### Step 1 — Define the action and decision time

**Input:** sponsor request, planning calendar, pending orders, and current disruptions.
Ask what decision will change because of the analysis.
State the available actions and when the decision can still affect the outcome.
Record the owner and the last responsible moment for action.
State the time bucket and the delivery window.

For CPG, the action may be a four-week promotional order by store and item.
The relevant deadline may be a retailer order cut-off before the promotional start.
For pharma, it may be release and allocation of a lot before expiry or a treatment window.
The release decision belongs to the authorized quality function, not the forecasting algorithm.
For automotive, it may be an expedite or resequencing decision before an assembly line runs short.
The decision window can be measured in hours even when component lead time is months.
For semiconductors, it may be a wafer-start reservation or allocation of tested die.
A start decision and a finished-goods allocation decision have different feasible sets and horizons.
For humanitarian response, it may be routing available kits before a road becomes impassable.
Program authority and access conditions must be established before dispatch.
For fashion, it may be the depth of an initial buy or a later store transfer.
The initial buy can be irreversible or costly to revise after the supplier cut-off.

**Output:** one sentence naming action, owner, deadline, and planning horizon.
**Branch:** if no one can act on the result, change the question before modeling.
If an authority outside the project must approve the action, add an approval gate.

### Step 2 — Map parties, incentives, and affected people

**Input:** commercial agreement, response mandate, organization chart, and stakeholder interviews.
Separate buyer, budget holder, approver, operator, and recipient.
Record who supplies evidence and who can contest the recommendation.
Identify who bears the risk of shortage, waste, delay, or exclusion.
Do not assume that whoever pays necessarily experiences the outcome.

In CPG, a supplier can benefit from unit sales while a retailer judges category contribution.
The shopper experiences shelf availability and price, not a supplier's factory utilization.
In pharma, procurement negotiates supply while clinicians and patients experience availability and quality.
Quality professionals control disposition within their authority.
In automotive, the OEM bears assembly consequences while tiered suppliers may own constrained capacity.
Warranty or recall exposure can extend beyond the initial component buyer.
In semiconductors, capacity reservation agreements may allocate rights differently from physical production control.
End-device demand may be far from the foundry's observed order book.
In humanitarian work, affected people may not choose the commodity or control funding.
Include community representatives and feedback channels where safe and feasible.
In fashion, a merchant controls an assortment while the supplier and retailer share risk according to contract.
The shopper can return an item if fit, quality, or expectations are not met.

**Output:** responsibility matrix with a named decision maker and an escalation owner.
**Branch:** if priorities conflict, document the conflict and obtain an authorized rule.
Do not let a default optimizer weight settle a policy dispute invisibly.

### Step 3 — Choose product and location grain

**Input:** item master, packaging conversions, location master, and traceability requirements.
Choose the smallest grain that can change a feasible action or outcome.
Retain mappings to higher-level families for reporting.
Record effective dates because products, locations, and relationships change.
Distinguish physical location from ownership and legal status.

For CPG, separate consumer units from cases and stores from distribution centers.
Case-pack conversion is required before comparing demand with case orders.
For pharma, include product, presentation, lot, expiration, status, and relevant serial information.
A location total is insufficient when only one lot is released.
For automotive, distinguish part revision, plant, line, supplier, and required sequence.
A compatible part number at the wrong revision may not satisfy a vehicle build.
For semiconductors, distinguish process, technology, product, wafer stage, die, and test status.
Different process nodes and qualified factories are not automatically interchangeable.
For humanitarian response, distinguish commodity, kit composition, location, eligibility, and affected group.
A nominal kit count does not prove its contents match local needs.
For fashion, preserve style, color, size, channel, and store or fulfillment location.
One extra size does not replace a missing size unless substitution is evidenced.

**Output:** grain statement and a versioned mapping table.
**Branch:** if a join combines incompatible grains, stop and repair the mapping.
If disaggregation would expose sensitive individuals, aggregate or restrict access under an approved protocol.

### Step 4 — Build the physical and information flow

**Input:** network map, bills of material, transport lanes, inventory snapshots, and event histories.
Draw nodes for conversion, storage, transportation, release, and consumption.
Add information events separately from physical movements.
For each edge specify time, capacity, ownership, and uncertainty.
Mark where an item changes identity, packaging level, or status.

For CPG, include retailer distribution, store receiving, backroom, and shelf replenishment.
For pharma, add temperature monitoring, receipt inspection, quality review, and authorized disposition.
For automotive, connect tiers to part revisions and plant-specific line consumption.
For semiconductors, distinguish fab processing, test, assembly, and final test where applicable.
For humanitarian work, add access checks, customs or border processes where relevant, and last-mile distribution.
For fashion, include inbound consolidation, allocation, store or e-commerce fulfillment, and return disposition.

A shipment event is not proof of receipt.
Receipt is not proof of quality release.
Release is not proof that the end user received or used the item.
A sale is not necessarily proof of retained consumption when returns are material.

**Output:** network diagram with event owners and decision-relevant latency.
**Branch:** if a critical upstream tier or last-mile link is invisible, record an uncertainty rather than inventing a node's capacity.

### Step 5 — Define qualified available supply

**Input:** on-hand balances, lot or item status, expiry, reservations, access, and delivery timing.
Start with a common ledger identity: opening physical stock plus receipts minus physical issues equals closing physical stock, adjusted for documented corrections.
Then apply eligibility rules to the closing physical stock.
Exclude goods unavailable for the intended decision, without deleting them from the physical ledger.
Avoid subtracting the same stock twice when reasons overlap.
Record reservations and other commitments separately.

In a retail example, damaged stock and stock trapped in a closed store are not available to replenish another location without an approved movement.
In pharma, quarantine is a status pending disposition; a quality decision determines whether affected product can be released or must remain unavailable.
In automotive, a part may be physically present but incompatible with the next approved build configuration.
In semiconductors, WIP is not finished die and must pass remaining processing and applicable testing.
In humanitarian work, a commodity is not operationally available for a location when no authorized safe route exists in the decision window.
In fashion, a returned garment needs inspection and disposition before it is counted as resalable stock.

**Output:** reconciled physical stock, exclusions by non-overlapping reason, and usable supply by time.
**Branch:** if quality or access status is unknown, hold the affected stock outside the usable pool pending an authorized decision.
If an exclusion may be reversible, model it as a separate scenario rather than silently assuming release.

### Step 6 — Define demand, need, and the denominator

**Input:** orders, consumption, sales, needs assessments, forecast records, and availability history.
Specify whether the requirement is an order, a statistical forecast, a clinical allocation requirement, or assessed need.
Keep the quantities separate when they answer different questions.
Specify the numerator, denominator, eligible population, place, and observation window for every service measure.
Record missing, censored, and delayed observations.

For CPG, point-of-sale transactions show purchases, not all demand during empty-shelf periods.
A promotion forecast should consider distribution, seasonality, and substitution.
For pharma, historical shipments can be constrained by supply rather than reflecting patient need.
Allocation rules may depend on approved indication, care pathways, and clinical governance.
For automotive, a build schedule and bill of materials translate vehicle plans into component requirements.
Firm schedule releases and indicative forecasts should not be merged without labels.
For semiconductors, buyer orders may contain buffering or cancellations that differ from end-device consumption.
For humanitarian work, registration records can omit people who cannot safely reach a registration site.
Needs assessment uncertainty and accessibility need explicit treatment.
For fashion, sales by size and week are censored when a size is unavailable.
Returns can reduce retained sales and alter the usable inventory date.

**Output:** a requirement table with source, uncertainty, vintage, and scope.
**Branch:** when no credible denominator exists, report counts and assessment limitations rather than inventing coverage.
A zero recorded requirement is not proof of zero need.

### Step 7 — Identify hard constraints before scoring options

**Input:** quality rules, contracts, safety requirements, technical qualifications, route permissions, and capacity limits.
List constraints as observable tests with an owner and evidence source.
Separate legal requirements from internal policy, contracts, standards, and analytical assumptions.
Record which constraints are absolute and who can approve any exception.
Do not assume an operational manager can waive a legal or patient-safety requirement.

CPG examples include case multiples, shelf capacity, and delivery appointment windows.
Pharma examples include release status, product-specific temperature handling, and remaining shelf life.
Automotive examples include approved revisions, production sequence, and supplier capacity.
Semiconductor examples include qualified process flows, wafer capacity, and packaging or test capacity.
Humanitarian examples include safe access, authorized delivery, security, eligibility, and funding conditions.
Fashion examples include production minimums, size-level availability, selling windows, and store capacity.
The applicability of each example depends on the specific product, contract, and jurisdiction.

**Output:** feasible-action checklist and evidence owner.
**Branch:** if no option is feasible, escalate the conflict rather than returning a fictitious allocation.
A model can quantify the trade-off after authorization; it cannot itself grant authority.

### Step 8 — Choose an objective and priorities

**Input:** authorized policy, stakeholder objectives, costs, service consequences, and ethical constraints.
Specify whose outcome is optimized and over which horizon.
Keep a hard constraint separate from an objective penalty.
Where objectives conflict, present a frontier or ordered priorities instead of hiding weights.
Document who approved the rule and how affected parties can challenge it.

CPG may compare incremental category contribution and item availability.
Pharma may first ensure eligible and quality-released stock, then use a clinically governed allocation rule.
Automotive may minimize expected production disruption subject to approved part and safety constraints.
Semiconductor decisions may compare allocation commitments, expected good output, and constrained capacity.
Humanitarian response may prioritize severity, urgency, vulnerability, and equitable access under a response policy.
Fashion may compare expected contribution after markdowns, returns, and inventory transfer costs.
None of these phrases defines a universally correct objective for the entire sector.

**Output:** objective, ordered priorities, and constraints in plain language.
**Branch:** if priorities are disputed, present affected scenarios and obtain a decision before execution.

### Step 9 — Select a transferable method

**Input:** decision, data history, feasible set, objective, and consequences.
Start with a simple transparent baseline.
A flow balance and ranked allocation often reveal data problems before a larger optimization model is justified.
Use forecasting when uncertainty about future demand materially changes the decision.
Use discrete-event simulation when queues, process variability, and sequencing dominate.
Use constrained optimization when the choices, constraints, and objective are sufficiently explicit.
Use causal methods when asking whether a promotion or intervention caused a change.
Use scenario analysis when probabilities are not defensible but plausible alternatives matter.
Human judgment and approval remain necessary when the model cannot represent a critical condition.

A method transfers only if its assumptions transfer.
A retail demand model trained on price promotions may not capture clinical allocation.
A fab capacity model cannot infer a humanitarian access route from warehouse inventory.
A humanitarian priority rule is not automatically a valid commercial customer ranking.
A fashion size curve cannot establish automotive component interchangeability.

**Output:** method selection with a baseline, assumptions, and a rejected alternative.
**Branch:** if data quality is insufficient, use a bounded manual decision and state what evidence would improve it.

### Step 10 — Stress the transfer

**Input:** baseline recommendation and uncertainty ranges or scenarios.
Vary one assumption at a time to expose local sensitivity.
Also combine correlated disruptions into coherent adverse scenarios.
Consider a delayed supplier, a failed quality review, a route closure, and a sudden demand shift separately where relevant.
Do not multiply unrelated worst cases and describe the result as a likely forecast.

For CPG, test promotion uplift and shelf replenishment labor jointly.
For pharma, test delayed quality disposition and shorter remaining shelf life.
For automotive, test an upstream component shortage with a line-sequence change.
For semiconductors, test yield loss and longer cycle time without assuming their correlation is zero.
For humanitarian work, test access loss and unregistered need together where supported by field evidence.
For fashion, test lower size-specific sell-through and a higher return rate.

**Output:** changed allocation, affected stakeholder, and trigger for each scenario.
**Branch:** if a small assumption change reverses the decision, present the instability and seek more evidence or a reversible action.

### Step 11 — Implement with a human authorization gate

**Input:** approved decision, qualified inventory, current commitments, and operational constraints.
Assign action owners and cut-off times.
Check that the data have not changed since the recommendation was made.
Send a proposed action, not an unreviewed command, to the execution system unless automation is specifically authorized.
Record overrides and the reasons for them.

Retail replenishment staff confirm case quantities and appointments.
Pharmaceutical quality staff make the relevant disposition decision.
Automotive planners confirm part revision and assembly sequence.
Semiconductor planners confirm wafer or die status and downstream capacity.
Humanitarian program and logistics leads confirm community needs, access, and authority.
Fashion merchants confirm size, channel, and remaining selling window.

**Output:** dated authorization, execution record, and exceptions queue.
**Branch:** stop dispatch if a mandatory control fails or status becomes stale.
Recalculate if a material quantity, time, route, or eligibility condition changes.

### Step 12 — Measure, learn, and decide again

**Input:** shipments, receipts, availability, service outcome, cost, incidents, and feedback.
Reconcile plan against execution before evaluating performance.
Report both achievement and coverage of the measurement system.
Investigate whether apparent improvement is caused by a changed denominator, shifted demand, or excluded groups.
Separate factual observations from modeled counterfactuals.
Set an owner and date for the next review.

A retail availability improvement requires an item-store-time definition.
A pharmaceutical service report should distinguish released stock from physical receipts and monitor the relevant treatment window.
An automotive report should distinguish part delivery from sequence-ready availability.
A semiconductor report should distinguish starts, WIP, good die, and promised ship dates.
A humanitarian report should distinguish distributions from actual access and use, and protect sensitive feedback.
A fashion report should distinguish gross sales, returns, retained sales, and markdown contribution.

**Output:** decision log, outcome scorecard, exception analysis, and next review.
**Branch:** pause the transferred method if it produces an unsafe, inequitable, technically incompatible, or untraceable recommendation.

## 6. Worked example

### Case and boundaries

A cross-sector team is evaluating whether a simple ranked allocation routine transfers to six settings.
Each setting has one synthetic week, one product-equivalent unit, and one decision point.
No quantity or outcome describes an actual company, patient group, or emergency.
The unit is deliberately abstract so the same arithmetic can be checked; it is not a claim that goods or people are interchangeable across sectors.
The routine does not choose the priorities.
The relevant authorized owner sets them before the calculation.

The team compares two eligible destinations, A and B, in each sector.
The initial need for A is 70 units.
The initial need for B is 50 units.
The total requirement is therefore 120 units.
The available physical stock is 110 units.
Ten of those units fail a sector-specific availability check and are not part of the usable pool.
Usable supply is therefore 110 − 10 = 100 units.
The 10-unit exclusion is non-overlapping and the only exclusion in this simplified example.
No further receipts occur in the week.
Both destinations are reachable and eligible within the week.
All units in the usable pool are compatible with both destinations.

The sponsor authorizes A ahead of B for this illustration.
That ordering is a synthetic policy input, not an inference that A should have priority in any real setting.
The proposed plan allocates to A first, up to its requirement, and then to B.

### Common calculation

1. Qualified supply is 100 units.
2. A receives `min(70, 100) = 70` units.
3. Remaining supply is `100 − 70 = 30` units.
4. B receives `min(50, 30) = 30` units.
5. Total planned allocation is `70 + 30 = 100` units.
6. A's unmet requirement is `70 − 70 = 0` units.
7. B's unmet requirement is `50 − 30 = 20` units.
8. Combined unmet requirement is `0 + 20 = 20` units.
9. Overall planned fill is `100 / 120 = 83.33%`, rounded to two decimals.
10. A's planned fill is `70 / 70 = 100%`.
11. B's planned fill is `30 / 50 = 60%`.

The plan is not yet a shipment, receipt, sale, treatment, production completion, or distribution to a person.
The forecasted requirement is not itself a contractual commitment.
An 83.33% planned fill is not proof of a service outcome.
It can conceal a much worse outcome for B.

### The six adapters

| Sector | Synthetic unit and destinations | Why 10 physical units are excluded | Additional decision gate |
|---|---|---|---|
| CPG | Consumer units for two stores | Damaged units are recorded but unavailable for sale | Verify case-pack and shelf capacity before shipping |
| Pharma | Treatment packs for two care sites | Lot is quarantined pending quality disposition | Confirm release, expiry and authorized clinical priorities |
| Automotive | Compatible components for two assembly lines | Wrong approved revision for this build | Verify line-specific compatibility and sequence |
| Semiconductor | Tested good die for two device programs | Units have not completed the required test | Confirm qualification and downstream packaging capacity |
| Humanitarian | Complete kits for two delivery areas | Stock cannot reach either area in the relevant safe-access window | Confirm community need, route authorization and distribution capability |
| Fashion | Garments of one style-color-size for two stores | Returns have not passed resale inspection | Confirm the size-level assortment and remaining season |

The physical supply in the humanitarian row can still be present in a warehouse.
Its exclusion concerns this particular time-window and destination set, not ownership or a permanent write-off.
The pharma exclusion similarly does not prove that the quarantined product is defective.
Its eventual disposition requires an authorized assessment.
The semiconductor row excludes untested units, not a forecast of the number that will fail test.
The fashion row does not imply every returned garment is unsalable.
The automotive row does not imply all revisions are interchangeable after a change order.
The CPG row assumes the damaged quantity is known and does not overlap another exclusion.

### Alternative allocation

A reviewer asks what happens if the policy is proportional rather than A-first.
A's share of total need is `70 / 120`.
B's share is `50 / 120`.
With 100 indivisible units and largest-remainder rounding, the fractional allocations are 58.333 and 41.667 units.
The integer allocation is 58 units to A and 42 units to B.
A's unmet amount is 12 units; B's is 8 units.
Their planned fills are `58 / 70 = 82.86%` and `42 / 50 = 84.00%`.
Overall planned fill remains `100 / 120 = 83.33%`.
The slight difference in destination fill rates is caused by indivisible-unit rounding.

The two rules use exactly the same stock and achieve the same aggregate fill.
They distribute the shortage differently.
The arithmetic alone cannot tell us whether A-first or proportional allocation is legitimate.
Clinical urgency, contractual commitments, safety, severity, vulnerability, and approved local policy can matter in different contexts.
For some contexts proportional allocation may violate an applicable requirement or an approved priority rule.
For other contexts A-first may create an unacceptable exclusion.
The comparison is a decision aid, not an ethical or legal determination.

### Time and unit sensitivity

Suppose 20 additional qualified units arrive after the one-week decision window.
They do not change the 100-unit supply available this week.
They might satisfy the 20-unit shortfall later if demand remains and the recipient can still use the product.
That is a separate two-period calculation, not a retroactive increase in this week's fill.
A second example: suppose the 100 units are actually 100 cases of 12 consumer units.
Comparing them directly with 120 consumer units would be a unit error.
Convert using the valid packaging mapping before running the allocation.
Conversely, if the destinations need whole cases, integer case constraints must be represented.

### Decision brief for this example

**Forecast:** 120 units of eligible requirement in the week, subject to sector-specific validation.
**Plan:** allocate 70 units to A and 30 units to B under the supplied A-first rule.
**Target:** the hypothetical sponsor desires 100% fill at both destinations.
**Commitment:** none is created by this illustrative calculation; actual obligations must be checked separately.
**Exception:** B has a 20-unit shortfall and only 60% planned fill.
**Escalation:** the accountable owner should review alternatives, potential harm, and the priority rule before authorizing dispatch.
**Evidence needed:** qualified inventory, destination eligibility, current requirement, compatibility, and delivery feasibility.
**Monitoring:** compare actual receipt and outcome against the approved plan, not merely against a warehouse release.

## 7. Code, query, or calculation

The following Python 3 program reproduces the common calculation and the proportional comparator.
It uses only the standard library.
Save it as `allocation_lab.py` and run `python allocation_lab.py`.
The program is a teaching artifact; it does not make a release decision, validate a route, or authorize a priority.
Inputs are nonnegative whole units.
The highest integer `priority` is served first; ties are resolved by the stated input order.
The proportional rule uses largest remainders, with ties resolved by input order.
The program assumes the excluded stock reasons do not overlap.

```python
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Destination:
    name: str
    need: int
    priority: int


def qualified_supply(physical: int, excluded: int) -> int:
    if physical < 0 or excluded < 0 or excluded > physical:
        raise ValueError("physical and excluded must define a valid stock balance")
    return physical - excluded


def priority_allocation(stock: int, destinations: list[Destination]) -> dict[str, int]:
    if stock < 0 or any(d.need < 0 for d in destinations):
        raise ValueError("stock and needs must be nonnegative")
    if len({d.name for d in destinations}) != len(destinations):
        raise ValueError("destination names must be unique")
    result = {d.name: 0 for d in destinations}
    remaining = stock
    for d in sorted(destinations, key=lambda item: -item.priority):
        quantity = min(d.need, remaining)
        result[d.name] = quantity
        remaining -= quantity
    return result


def proportional_allocation(stock: int, destinations: list[Destination]) -> dict[str, int]:
    if stock < 0 or any(d.need < 0 for d in destinations):
        raise ValueError("stock and needs must be nonnegative")
    if len({d.name for d in destinations}) != len(destinations):
        raise ValueError("destination names must be unique")
    total_need = sum(d.need for d in destinations)
    if total_need == 0:
        return {d.name: 0 for d in destinations}
    distributable = min(stock, total_need)
    exact = [Fraction(distributable * d.need, total_need) for d in destinations]
    whole = [x.numerator // x.denominator for x in exact]
    units_left = distributable - sum(whole)
    order = sorted(range(len(destinations)), key=lambda i: -(exact[i] - whole[i]))
    for i in order[:units_left]:
        whole[i] += 1
    return {d.name: whole[i] for i, d in enumerate(destinations)}


def report(label: str, stock: int, destinations: list[Destination], allocation: dict[str, int]) -> None:
    total_need = sum(d.need for d in destinations)
    total_allocated = sum(allocation.values())
    print(label)
    print(f"Qualified supply: {stock} units")
    for d in destinations:
        assigned = allocation[d.name]
        rate = f"{100 * assigned / d.need:.2f}%" if d.need else "not defined"
        print(f"{d.name}: assigned={assigned}, unmet={d.need - assigned}, fill={rate}")
    overall = f"{100 * total_allocated / total_need:.2f}%" if total_need else "not defined"
    print(f"Total: assigned={total_allocated}, unmet={total_need - total_allocated}, fill={overall}")
    print()


def main() -> None:
    stock = qualified_supply(physical=110, excluded=10)
    destinations = [Destination("A", 70, 2), Destination("B", 50, 1)]
    report("Priority", stock, destinations, priority_allocation(stock, destinations))
    report("Proportional", stock, destinations, proportional_allocation(stock, destinations))


if __name__ == "__main__":
    main()
```

Expected output:

```text
Priority
Qualified supply: 100 units
A: assigned=70, unmet=0, fill=100.00%
B: assigned=30, unmet=20, fill=60.00%
Total: assigned=100, unmet=20, fill=83.33%

Proportional
Qualified supply: 100 units
A: assigned=58, unmet=12, fill=82.86%
B: assigned=42, unmet=8, fill=84.00%
Total: assigned=100, unmet=20, fill=83.33%
```

### Lab interpretation

Both routines cap total allocation at usable stock and each destination's stated need.
Neither routine models time, costs, packaging, route capacity, shelf life, quality release, or product compatibility.
The input must already represent stock that passed those relevant checks.
The priority routine assumes a complete policy ordering; its tie behavior is a technical convention, not an approved fairness rule.
The proportional routine assumes equal relative fulfillment is a useful comparator; it is not a general fairness proof.
If need is zero, the report prints `not defined` rather than dividing by zero.
If stock exceeds need, remaining units are not assigned.
Integer rounding can make proportional fill percentages differ slightly.
Changing a priority changes the plan but does not create authority to execute it.

### Adaptation exercise

1. Change A's need to 55 units while leaving B at 50 units.
2. Keep usable supply at 100 units and A's priority above B's.
3. Predict allocations before running the script.
4. Identify the destination and units of shortfall.
5. Describe one sector-specific precondition that must be checked before interpreting the result.
6. Explain why a later shipment is not necessarily equivalent to on-time fulfillment.
7. Document whether a proposed allocation is a forecast, plan, target, or commitment.

The expected priority result for the adaptation is A = 55 units and B = 45 units.
The shortfall is 5 units at B.
This is a mathematical check, not an operational approval.

## 8. Data, standards, and regulatory context

A standard can provide a vocabulary, identifier, format, or process specification.
It does not automatically make data complete, current, interoperable, or legally sufficient.
Law, regulation, contract, and voluntary standard are different sources of obligation.
Applicability depends on the product, party, jurisdiction, effective date, and transaction.
The descriptions below refer to the cited public materials and are framed as of November 2026; verify current text and implementation with qualified specialists before use.

### Cross-sector data contract

| Field | Definition and example grain | Control |
|---|---|---|
| Item identifier | Product or operational item at the decision grain | Version mappings and packaging conversions |
| Location identifier | Facility, store, site, or delivery area | Track changes and authorized location scope |
| Quantity | Numeric amount with unit and packaging | Reject mixed or absent units |
| Event type | Receipt, release, shipment, sale, distribution, or return | Distinguish observation from inferred event |
| Event time | When an event actually occurred, with timezone | Preserve event and ingestion times separately |
| Status | Released, quarantined, damaged, in transit, or other defined state | Maintain authorized transitions and owner |
| Requirement | Forecast, order, clinical request, assessed need, or build demand | Retain vintage, source, and uncertainty |
| Commitment | Accepted quantity and date under defined authority | Separate firm obligation from scenario |
| Eligibility | Destination or use allowed for the item | Record decision and evidence owner |
| Expiration | Relevant product date and rule | Record format, timezone where relevant, and usable-window calculation |
| Route or lane | Origin, destination, mode, and known access status | Time-stamp access confirmation |
| Provenance | Source system, submitter, extraction time, and transformations | Preserve lineage and corrections |
| Priority | Authorized rule and its effective period | Do not infer authority from a numerical score |
| Outcome | Receipt, availability, treatment, distribution, sale, or production | Match metric to stakeholder and window |
| Privacy class | Permitted purpose, access, retention, and sharing conditions | Minimize personal data and restrict access |

### CPG and retail

GS1 identifies the Global Trade Item Number (GTIN) as an identifier for trade items.
Packaging levels may require distinct identifiers and explicit relationships.
GS1's Global Location Number (GLN) identifies parties or locations under its system.
These are voluntary standards unless incorporated into a contract or applicable rule.
A valid identifier does not prove that a shelf had stock at a specific time.
Point-of-sale, inventory, purchase order, and shelf observations have different event meanings.
Store-level loyalty data can identify or profile people and requires applicable privacy review.

### Pharmaceuticals

The U.S. Drug Supply Chain Security Act (DSCSA) establishes requirements for pharmaceutical distribution security for products and parties within its scope; consult current FDA guidance and statutory text for exact obligations and implementation.
Serialization or transaction data alone do not establish product quality or the appropriateness of a clinical allocation.
European Commission guidance on good distribution practice (GDP) of medicinal products for human use addresses distribution quality systems and relevant handling controls.
Its applicability and interaction with national requirements must be checked for the activity and jurisdiction.
Temperature limits are product- and procedure-specific; an excursion needs an authorized, documented assessment.
Do not automatically call every excursion a product failure or assume every excursion is safe.
Patient-related data requires additional protection; avoid unnecessary identifiers in an allocation prototype.

### Automotive

The International Automotive Task Force's IATF 16949 is a quality management system standard used in automotive supply chains, typically alongside ISO 9001 and applicable customer-specific requirements.
Certification and contractual requirements depend on the organization and customer; the standard is not a blanket law governing every transaction.
Part number, engineering revision, production date, and where necessary vehicle association support traceability.
An electronic delivery schedule or advance shipment notice is an event or instruction, not evidence of line-ready compatibility.
Recall and vehicle safety obligations vary by jurisdiction and facts; consult the responsible safety and legal teams.

### Semiconductors

SEMI publishes standards used across semiconductor manufacturing and related industries.
The applicability of any specific SEMI standard depends on the equipment, materials, interface, and agreement.
Manufacturers may have proprietary process qualifications and customer-specific acceptance criteria.
Wafer identifiers, process step events, test bins, die counts, and packaging lots refer to different stages and grains.
Export controls and sanctions can constrain specific transactions and destinations; verify the current rules with trade compliance.
A commercial capacity reservation does not by itself prove a qualified and deliverable finished device.

### Humanitarian logistics

The Sphere Handbook presents a humanitarian charter and minimum standards for humanitarian response.
It is a widely used voluntary reference, not a universal statute or a substitute for context-specific obligations.
The Core Humanitarian Standard describes commitments to quality and accountability to affected people.
Donor terms, national law, humanitarian principles, local access agreements, and organizational mandates may independently constrain action.
The United Nations Office for the Coordination of Humanitarian Affairs (OCHA) coordinates humanitarian information and response activities; coordination arrangements differ by emergency.
Collect only the personal information necessary for a stated purpose and protect sensitive location or vulnerability data.
An assessment's registered population can omit those without identification, transport, access, or confidence in the process.
Represent uncertainty and consult affected communities rather than equating missing records with no need.

### Fashion and apparel

Product and packaging identifiers, care labels, fiber composition, country-of-origin declarations, and supplier records can be important depending on the market and product.
The U.S. Federal Trade Commission's textile and wool guidance explains labeling requirements applicable to covered products.
EU textile labeling rules apply within their defined scope; verify the product and jurisdiction rather than treating an unrelated jurisdiction's label as universally sufficient.
Material, color, size, season, channel, and return disposition are essential analytical distinctions even when not mandated as a shared format.
Supplier labor and environmental due-diligence obligations vary by jurisdiction, organization, and time.
A product identifier does not establish working conditions at every supplier tier.

### Cross-sector controls

1. Record the exact rule and jurisdiction rather than using the word compliant as a data field without evidence.
2. Identify the party responsible for interpreting the rule.
3. Keep the applicable document version and effective period.
4. Capture an audit trail of changes and approvals.
5. Protect commercially sensitive schedules and pricing.
6. Limit access to identifiable or sensitive humanitarian and health data.
7. Record whether a status represents physical possession, legal ownership, release, or availability.
8. Preserve historical values rather than overwriting all prior events.
9. Use explicit conversion tables when case, pack, wafer, die, or kit quantities differ.
10. Test whether downstream users understand what each measure excludes.

## 9. Industry practice and case evidence

The sources in section 16 support the documented statements below.
A documented sector practice does not prove that a particular model would succeed in another setting.
Where a lesson extends beyond a source, it is labeled inference.
No synthetic calculation above is presented as an actual case result.

### CPG: availability is not a single number

**Documented:** GS1 describes GTINs as identifiers for trade items and GLNs as identifiers for parties and locations.
The standards support consistent identification across participating organizations.
**Inference:** A shared identifier can make store-item reconciliation easier, but a separate shelf observation or reliable proxy is still needed to assess on-shelf availability.
A distribution center shipment is an upstream event, not a shopper purchase.
**Transfer lesson:** preserve identifiers and event semantics when taking the same reporting architecture to another sector.
**Do not transfer:** the assumption that downstream units can be substituted or sold without an additional quality gate.

### Pharma: distribution security and quality are different controls

**Documented:** FDA describes the DSCSA as creating an interoperable electronic tracing framework for certain prescription drugs distributed in the United States.
The European Commission has published GDP guidelines for medicinal products for human use.
Those sources address different requirements and contexts; one does not replace the other.
**Inference:** an allocation system should link a traceable pack or lot to its authorized quality status rather than treating a transaction event as proof of release.
**Transfer lesson:** an item can have both physical and regulatory or quality states.
**Do not transfer:** the assumption that the exact pharmaceutical traceability requirement applies to every fashion garment or relief item.

### Automotive: schedules meet engineering configuration

**Documented:** IATF publishes the automotive quality management system standard IATF 16949 and related materials.
**Inference:** a component supply model is incomplete when it ignores approved part revision and the production configuration it must support.
A delivery can satisfy an aggregate count yet fail a sequence-specific requirement.
**Transfer lesson:** qualify stock against its intended use, not merely its part-family label.
**Do not transfer:** the assumption that all sectors have an assembly line or the same customer-specific quality scheme.

### Semiconductors: starts, WIP, and good output differ

**Documented:** the U.S. Government Accountability Office's semiconductor supply chain work describes the complexity and global nature of semiconductor production.
**Inference:** translating capacity into deliverable chips requires explicit process stages, timing, and yield assumptions.
A wafer start is an input to a process, not a completed delivery of acceptable die.
**Transfer lesson:** separate process input, WIP, tested output, and customer-available supply.
**Do not transfer:** an assumed universal yield or cycle time across products and factories.

### Humanitarian response: the recipient is not necessarily the buyer

**Documented:** the Sphere Handbook sets out humanitarian principles and minimum standards; the Core Humanitarian Standard emphasizes accountability to people affected by crisis.
**Inference:** a logistics scorecard should include an accountable assessment of whether the intended people can access appropriate assistance, not just warehouse dispatch volume.
Reported distribution can be necessary evidence without proving that all affected groups were reached.
**Transfer lesson:** ask who is absent from the recorded demand and who can challenge an allocation.
**Do not transfer:** the assumption that commercial sales transactions are an adequate measure of humanitarian need.

### Fashion: product detail and returns shape the result

**Documented:** the U.S. Federal Trade Commission provides guidance on labeling covered textile and wool products.
The guidance concerns specified labeling duties, not a complete inventory or labor-audit system.
**Inference:** style-color-size and return disposition are important operational grains when size-specific availability and resale eligibility affect decisions.
A style-level total can look healthy while a popular size is unavailable.
**Transfer lesson:** use a product grain that preserves actual substitutability.
**Do not transfer:** an assumption that late demand can be fulfilled at the same value after a seasonal selling window.

### How to use evidence

A source can support a rule's existence without proving a particular firm's implementation.
A company case can describe a practice without establishing a causal effect.
A peer-reviewed estimate can be valid for its sample and still fail to generalize.
For any transfer claim, record the source, population, date, outcome, and untested assumptions.
Ask whether the decision and operating conditions resemble the proposed target setting.
Treat the absence of public information as uncertainty, not proof that a practice does not exist.

### Six transfer experiments

The following examples are synthetic design exercises, not documented organizational deployments.
Each compares a transferable analytical idea with the work needed to make it usable.
The examples do not establish causal effects or sector-wide benchmarks.

**1. From retail shelf monitoring to pharmaceutical availability**

A retailer records a store-item shelf observation every four hours.
The proposed transferable idea is to pair inventory balances with observations at the point of use.
At a care site, the equivalent question is whether an appropriately released medicine is available when a patient needs it.
The retail observation cannot simply be renamed a treatment-availability observation.
The item may have presentation, lot, expiry, and storage conditions that affect usability.
A recorded unit at a hospital loading dock may not be available on a treatment ward.
The analyst should distinguish receipt, quality status, storage location, and dispensing eligibility.
The clinical team must define the relevant treatment window and an acceptable substitute.
If a treatment is scheduled but delayed, the outcome should not be reported as an on-time treatment solely because stock arrived later that day.
Protect patient identifiers and use aggregated measures where individual records are not necessary.
**Pilot measure:** percentage of eligible scheduled administrations with an available released pack by the required time, with an explicit denominator.
**Stop condition:** any recommendation bypasses a quality hold or reveals patient data to unauthorized users.
**Transfer conclusion:** the point-of-use perspective transfers; the product status, recipient and outcome definition change.

**2. From automotive sequence checks to semiconductor delivery**

An automotive planner checks whether the correct revision is ready for a planned vehicle sequence.
The transferable idea is compatibility validation before a release or allocation.
For a semiconductor program, the relevant checks can include device type, qualified fabrication process, test status, and packaging form.
There is no reason to assume automotive revision codes and semiconductor qualifications have the same structure.
The planner must also distinguish tested die from packaged and qualified devices.
A quantity that is adequate at die test may be insufficient after subsequent losses; estimate those losses only with suitable data.
Delivery timing must include remaining production and logistics stages.
If a qualification is customer-specific, one customer's accepted output may not satisfy another customer's order.
**Pilot measure:** percentage of promised device quantities supported by current compatible, qualified stock and feasible dates.
**Stop condition:** the allocation assigns an unqualified process or omits a required test gate.
**Transfer conclusion:** compatibility gating transfers; the engineering evidence and conversion stages change.

**3. From semiconductor capacity scenarios to humanitarian routing**

A semiconductor planner compares throughput under normal and constrained capacity.
The transferable idea is to calculate how a bottleneck changes feasible output over time.
A humanitarian team can similarly compare dispatch and delivery capacity under alternate access conditions.
The bottleneck may be a bridge, border procedure, warehouse handling capacity, or last-mile distribution team.
The scenarios should be informed by qualified local access and security sources.
An estimated road capacity is not permission to travel.
An open route is not evidence that the destination population wants or can use a particular commodity.
The team should represent the difference between dispatch, arrival, and safe distribution.
It should also consider communities that remain inaccessible under all modeled scenarios.
Do not disclose sensitive route or population information beyond authorized users.
**Pilot measure:** deliveries completed within the defined access window, alongside unmet need by affected group.
**Stop condition:** a route loses authorization or a community reports an unresolved protection concern.
**Transfer conclusion:** bottleneck scenarios transfer; authority, safety, and recipient accountability change.

**4. From humanitarian prioritization review to retail allocation**

A humanitarian team documents who approved a priority rule and invites affected people to raise concerns.
The transferable idea is accountability for how shortages are distributed.
A retailer can document whether scarce inventory is assigned by store demand, contractual obligations, customer impact, or another approved rule.
Its stakeholders and legal obligations differ from those of a relief operation.
Store-level demand may reflect previous availability and therefore disadvantage a previously understocked store.
The analyst can compare the current allocation with a proportional or minimum-coverage alternative.
The team should not represent proportional allocation as automatically equitable or profitable.
It should disclose how the rule treats small stores and low-volume locations.
Supplier and retailer commercial incentives may differ.
Any store-level customer data should be handled according to applicable privacy rules.
**Pilot measure:** store-level shortage and availability, with an agreed commercial outcome measured separately.
**Stop condition:** the rule violates an accepted commitment or depends on an unapproved use of customer data.
**Transfer conclusion:** transparent priority governance transfers; the decision criteria and affected parties change.

**5. From fashion size curves to automotive configuration planning**

A fashion merchant estimates demand separately by style, color, and size.
The transferable idea is to preserve features that determine whether stock satisfies a specific request.
An automotive planner can similarly retain vehicle configuration and required component revision.
The analogy does not imply a size is interchangeable with an engineering part.
Clothing demand can involve consumer substitution, while vehicle compatibility is an engineering and safety determination.
Aggregate demand can look balanced while one configuration is short.
Before pooling any stock, the planner should obtain the approved compatibility mapping.
Changes to a build schedule can alter requirements after an earlier forecast was issued.
The analyst should retain forecast vintage and distinguish an indicative mix from a firm release.
**Pilot measure:** share of scheduled builds supported by approved parts at the required sequence and time.
**Stop condition:** a proposed substitution lacks engineering approval.
**Transfer conclusion:** disaggregated demand transfers; substitution rules and approval authority change.

**6. From pharmaceutical lot traceability to fashion returns**

A pharmaceutical operation tracks an item or lot through defined events and authorized status changes.
The transferable idea is a time-stamped state machine rather than a single inventory balance.
A fashion return can move from initiated, to in transit, to received, to inspected, and then to resalable or another disposition.
The specific pharmaceutical serialization requirements do not automatically apply to apparel.
An initiated return is not yet physical inventory at a fulfillment center.
A received return is not necessarily resalable before inspection.
An item routed for repair or recycling should not be promised as immediately available for a customer order.
The merchant should distinguish gross sales from returns and retained sales.
Record the inspection policy and timing because the remaining season affects potential resale value.
**Pilot measure:** elapsed time from physical return receipt to documented disposition, segmented by product category.
**Stop condition:** the system offers an uninspected item as new stock where policy forbids it.
**Transfer conclusion:** event history and status gates transfer; legal obligations and disposition rules change.

### Compare before generalizing

For each experiment, record the source context, target context, claim, assumptions, and observable failure condition.
Ask whether a change in product, location, clock, decision right, or outcome invalidates the original method.
Ask whether the target organization can actually observe the inputs at the required time.
If the data cannot support the intended measure, narrow the claim rather than filling gaps with unsupported estimates.
If the method needs a priority ordering, the relevant policy owner must supply it.
If the method needs a quality or engineering release, the responsible function must provide that decision.
A pilot can show whether the proposed workflow works in its chosen setting.
It does not automatically prove benefits at every site or in every disruption.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Common unit is undefined | Cases, packs, and units appear in one quantity column | Incompatible measures are summed | Allocation overstates or understates stock | Version conversion factors and validate grain |
| Physical stock equals usable stock | Quarantine or damage status is missing | Invalid stock enters feasible pool | Unsafe or impossible recommendation | Gate by authorized status and reconcile exclusions |
| Duplicate exclusions | Multiple holds apply to the same lot | Each reason is subtracted independently | Usable stock is understated | Assign a non-overlapping primary exclusion or calculate set union |
| Availability is assumed from shipment | Dashboard stops at dispatch | Receiving or last-mile failure is invisible | False service claim | Track receipt and destination outcome |
| Priority is embedded in code | No approved policy accompanies the ranking | Developer choice becomes allocation policy | Unjustified or unauthorized shortage distribution | Obtain documented ownership and review |
| Average fill hides exclusion | Overall fill improves while a group has low coverage | Unequal shortages are aggregated away | Harm to a destination or community | Report denominator and disaggregated outcomes |
| Sales equal demand | Low sales coincide with stockouts | Censored sales are treated as preference | Future shortages and biased forecasts | Use availability history and sensitivity ranges |
| Quality release inferred | A quarantined lot is treated as released after a timer | Administrative shortcut bypasses review | Unsafe or nonconforming distribution | Require documented authorized disposition |
| Wrong engineering revision | Part family matches but revision is absent | Compatibility is assumed | Line interruption or quality issue | Validate approved configuration and effective dates |
| WIP counted as finished goods | Wafer starts appear in a finished-stock report | Remaining cycle and yield are ignored | Overpromising delivery | Model stage, expected yield, timing and qualification |
| Road status is stale | Route data lack a timestamp | Access changes during the decision window | Failed delivery or security risk | Confirm route with authorized local sources before dispatch |
| Registered need equals all need | Data cover only reachable registration sites | Inaccessible people are omitted | Allocation reinforces exclusion | Triangulate estimates and involve affected communities |
| Fashion size pooling | Style-level stock appears adequate | Sizes are not interchangeable | Stockouts plus markdown surplus | Forecast and allocate at style-color-size grain |
| Return counted as saleable immediately | Returns are added to inventory on initiation | Inspection and transit are skipped | Orders promise unusable units | Model receipt and disposition before release |
| Target presented as commitment | A slide labels desired fill as guaranteed | Aspirational and contractual fields are merged | Misleading service promises | Separate forecast, target, plan, and commitment |
| Policy ported by sector label | Prior-sector safety stock is copied unchanged | Lead-time and demand distributions differ | Excess cost or shortage | Re-estimate parameters and validate assumptions |
| Hard rule is merely penalized | Unsafe option scores well at low cost | Mandatory condition is modeled as soft | Invalid recommendation | Enforce hard feasibility gates before scoring |
| Sensitive data are overshared | Names and vulnerability fields appear in an allocation export | Data minimization is absent | Privacy or security harm | Restrict fields, access, retention, and sharing |
| Effect is assumed causal | A metric rises after an intervention without a comparator | Seasonal or concurrent changes explain the movement | Wrong investment decision | Use a defensible counterfactual and state limitations |
| Model ignores stale data | Stock and access timestamps lag decisions | State changes after calculation | Plan becomes infeasible | Set freshness thresholds and recalculate |

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | How to test or respond |
|---|---|---|---|
| Each unit is interchangeable within the example | Allows one common supply pool | Lot, size, revision, or clinical eligibility differs | Split the pool by compatibility and rerun |
| Requirements are known for the week | Makes arithmetic reproducible | Forecast error, unregistered need, or cancellation | Compare vintages and run range scenarios |
| Excluded quantities do not overlap | Simple subtraction is correct | A unit is both damaged and quarantined | Reconcile item-level statuses and use the union |
| Qualified supply is available at time zero | Supports single-period allocation | Release, transport, or receipt occurs later | Use dated inventory and a time-phased model |
| A-first rule is authorized | Defines a complete ranking | Authority is missing or priorities conflict | Obtain a documented decision and challenge path |
| Both destinations are eligible and reachable | Allows allocation to either | Route closure, restrictions, or incompatible product | Validate eligibility and access at dispatch |
| Units are indivisible | Supports integer allocation | Actual goods can be divided safely and legitimately | Change the unit and validate handling rules |
| No case-pack or minimum shipment applies | Allows any integer quantity | Contract or physical packaging requires multiples | Add pack constraints and recalculate |
| No post-allocation loss occurs | Separates assignment from execution | Spoilage, damage, return, or diversion | Measure receipt and model losses separately |
| No new supply arrives within the window | Holds supply at 100 units | A confirmed usable receipt arrives earlier | Add a separate time-stamped receipt |
| Total need of 120 units is an appropriate denominator | Allows an overall fill rate | Excluded or unobserved groups have real need | State coverage limits and perform an independent assessment |
| A and B have comparable units of need | Supports proportional comparison | One unit yields different coverage or treatment | Use appropriate outcome measures and governance |
| Stated status is current and accurate | Enables qualified supply calculation | Delayed interfaces or incorrect master data | Check timestamps, source records, and audit samples |
| One-period service is the relevant outcome | Keeps the calculation transparent | Seasonality, mortality risk, line sequence, or patient timing matters | Add time, harm, and sequence where decision-relevant |
| Source descriptions remain current | Supports an educational regulatory overview | Rules or implementation dates change | Consult current primary sources and qualified counsel |

These assumptions are not sector-wide facts.
Their violation does not always require a complex model; sometimes it requires a narrower decision and a manual approval.
A robust transfer includes a plan for detecting when its assumptions stop holding.
A numerical example cannot validate an ethical priority, prove a causal impact, or establish regulatory compliance.

## 12. Exercises

### Exercise 1 — Map decision rights

**Time:** 20 minutes.
Choose one actual or plausible decision in each of the six sectors.
Name the buyer or authorizer, decision owner, operator, end user, and affected party.
Identify a conflict between at least two stakeholders in two sectors.
**Deliverable:** six-row responsibility table and two short conflict notes.
**Acceptance criteria:** every row names a specific action and decision deadline; no row assumes the buyer is always the user.

### Exercise 2 — Transfer or adapt

**Time:** 20 minutes.
Classify each item as transferable structure, sector-specific parameter or policy, or context-dependent control: flow balance; patient-quality release; store case pack; uncertainty scenario; priority ranking; lot traceability.
For context-dependent controls explain what changes between two sectors.
**Deliverable:** six-item classification with justification.
**Acceptance criteria:** distinguishes the generic method from the value or authority that determines its use.

### Exercise 3 — Qualified stock

**Time:** 15 minutes.
A synthetic site records 157 packs physically on hand.
Of these, 12 packs are quarantined and a separate 9 packs are damaged.
No packs are in both groups.
There are no other exclusions.
**Deliverable:** qualified packs and a reconciliation equation.
**Acceptance criteria:** physical and qualified balances reconcile; no quarantined pack is declared defective merely because of its status.

### Exercise 4 — Priority allocation

**Time:** 20 minutes.
Use the 136 qualified packs from Exercise 3.
An authorized policy ranks A before B.
A needs 83 packs and B needs 69 packs during the week.
**Deliverable:** allocation, shortage, and fill for each destination and the total.
**Acceptance criteria:** allocations do not exceed need or qualified stock; show arithmetic and label the result as a plan.

### Exercise 5 — Diagnose a denominator

**Time:** 20 minutes.
A response dashboard reports 90% coverage because 900 registered households received assistance out of 1,000 registered households.
A field assessment suggests another 400 households may be present but unregistered; their eligibility and overlap are not yet verified.
**Deliverable:** a two-paragraph critique and a data-collection plan.
**Acceptance criteria:** reports the registered-household rate accurately, does not call 900/1,400 an established true coverage rate, and specifies how to check overlap and eligibility.

### Exercise 6 — Design a sector adapter

**Time:** 30 minutes.
Port the section 7 routine to either pharma or automotive as a design, without asserting that the current code already implements the adapter.
List input fields, hard constraints, decision rights, action gates, and an override process.
**Deliverable:** one-page design and three test scenarios.
**Acceptance criteria:** includes quality release or approved revision as appropriate, checks time and compatibility, and names the accountable human authority.

### Exercise 7 — Change the lab

**Time:** 25 minutes.
Change needs to 55 and 50 units, keeping qualified stock at 100 units and A first.
Predict the result, run the code, and compare it with a proportional policy.
**Deliverable:** output, manual arithmetic, and one sentence on why neither policy is universally correct.
**Acceptance criteria:** A-first is 55 and 45 units; proportional is 52 and 48 units after largest-remainder rounding; priority authority is explicit.

### Exercise 8 — Write a transfer brief

**Time:** 35 minutes.
A fashion merchant proposes to copy a CPG promotion replenishment model to a seasonal apparel assortment.
Draft a decision brief that identifies what can transfer and what needs new data, constraints, and validation.
**Deliverable:** one-page brief with decision, owner, scope, pilot, outcome metrics, stop conditions, and review date.
**Acceptance criteria:** includes style-color-size, returns, markdown, season timing, and a counterfactual or defensible baseline; separates forecast, plan, target, and commitment.

### Exercise 9 — Evidence ledger

**Time:** 20 minutes.
Select three statements from section 9.
For each, identify the cited source, the supported factual claim, the labeled inference, and a fact not established by that source.
**Deliverable:** three-row evidence ledger.
**Acceptance criteria:** no source is credited with a causal result or operational detail it does not establish.

## 13. Self-check answers

### Exercise 1

A strong response distinguishes the purchaser from the ultimate user and from the authority to release or dispatch.
For example, a retail buyer can negotiate an assortment while a shopper experiences shelf availability.
A humanitarian program lead can authorize distribution while affected people experience whether assistance is accessible and appropriate.
A pharmaceutical quality authority is not interchangeable with a commercial demand planner.
The specific title and division of rights must be verified for the organization in the scenario.
Full credit requires six decisions, six deadlines, and two explicit conflicts.

### Exercise 2

Flow balance is transferable structure, provided units and system boundaries are defined.
A scenario is also transferable structure; the scenario contents and probabilities are context-specific.
A store case pack is a sector- and item-specific operational parameter or constraint.
A priority ranking is policy requiring appropriate authority, not a universal algorithmic constant.
Quality release is a context-dependent control whose specific requirements are governed by the product and jurisdiction.
Lot traceability is a context-dependent control: lots occur in multiple sectors, but obligations, grain, and event requirements differ.
Accept another classification if the learner distinguishes the general mechanism from its specific value and authority.

### Exercise 3

`157 − 12 − 9 = 136` qualified packs.
The reconciliation is `157 physical = 136 qualified + 12 quarantined + 9 damaged` packs.
This works because the two excluded sets are explicitly disjoint.
A quarantined pack is not automatically defective; an authorized disposition is still required.
If the sets overlapped, subtract the union rather than both totals independently.

### Exercise 4

A receives `min(83, 136) = 83` packs.
The remaining stock is `136 − 83 = 53` packs.
B receives `min(69, 53) = 53` packs.
A's shortage is `83 − 83 = 0` packs.
B's shortage is `69 − 53 = 16` packs.
A's planned fill is `83 / 83 = 100%`.
B's planned fill is `53 / 69 = 76.81%`, rounded to two decimals.
Total need is `83 + 69 = 152` packs.
Total planned fill is `136 / 152 = 89.47%`, rounded to two decimals.
Total shortage is 16 packs.
The result is a proposed plan, not a receipt or commitment.

### Exercise 5

The documented ratio is `900 / 1,000 = 90%` of registered households, assuming the numerator is a subset of the denominator and the reporting window matches.
The additional 400 households are an uncertain field estimate, not yet a verified disjoint eligible population.
If they were all distinct and eligible, and none received assistance, `900 / 1,400 = 64.29%` would be an illustrative scenario, not an established coverage statistic.
Triangulate the field estimate, check overlap, assess eligibility, and identify whether people were served through other channels.
Engage affected communities and protect the identity and location of vulnerable households.
Report uncertainty and limitations alongside any revised estimate.

### Exercise 6

A strong pharma design has product and lot identifiers, expiry, quality status, available quantity, destination eligibility, and relevant time windows.
It excludes quarantined stock unless an authorized quality disposition changes its status.
A strong automotive design has part and revision, plant and line, sequence requirement, usable quantity, and approved substitution or change evidence.
It excludes incompatible revisions even when the family matches.
Both designs specify fresh data, authorization, exception handling, and a record of overrides.
Three tests should include a valid allocation, a hard-constraint failure, and a stale or changed input.
A model recommendation cannot override a required human approval.

### Exercise 7

A-first gives A `min(55, 100) = 55` units and B `100 − 55 = 45` units.
The shortage is 5 units at B.
For proportional allocation, exact shares are `100 × 55 / 105 = 52.38095…` and `100 × 50 / 105 = 47.61905…` units.
The floors are 52 and 47 units, leaving one unit.
B has the larger fractional remainder, so the final allocation is 52 and 48 units.
A and B have different shortages under the two rules even though the total shortage remains 5 units.
The appropriate rule depends on authorized priorities and the consequences of the shortage.

### Exercise 8

A strong brief identifies the decision and named merchant owner before selecting a model.
It transfers demand forecasting, lead-time accounting, inventory balance, and an evaluation cadence as methods rather than copying numerical settings.
It adapts the product grain to style-color-size and accounts for seasonal value, markdown, returns, and product availability.
It proposes a limited pilot, a defensible baseline, measurable outcomes, a stop condition, and a dated review.
It separates a forecast of sales from a proposed purchase plan, a desired target, and an accepted purchase commitment.
It identifies relevant stakeholders and does not claim causality from an uncontrolled before-and-after comparison.

### Exercise 9

A strong evidence ledger separates the source's actual statement from the learner's interpretation.
For the FDA example, the factual claim concerns the framework for tracing covered prescription drugs.
The inference is that a model should connect tracing information with quality status.
The source does not establish that a specific firm's model achieved a particular service improvement.
Equivalent distinctions are required for the other two rows.

## 14. Interview questions

1. **What transfers when you move a replenishment model across sectors?**
   A strong answer separates a flow balance, uncertainty handling, and decision review from sector-specific units, release, time, and objective.
   A weak answer copies a safety-stock percentage without testing its assumptions.
2. **Who is the buyer in a humanitarian supply chain?**
   A strong answer distinguishes funder, procurement authority, program owner, logistics operator, and affected people.
   A weak answer assumes the donor and recipient have identical objectives.
3. **How do you define available pharmaceutical inventory?**
   A strong answer requires the relevant quality status, lot, expiry, location, and applicable authorization.
   A weak answer treats a warehouse balance or a serialization event as proof of release.
4. **Why might 100% automotive part fill still fail the plant?**
   A strong answer checks revision, line, timing, and sequence, not just aggregate units.
   A weak answer assumes any item in a part family fits any vehicle configuration.
5. **Why can wafer starts not be used directly as deliverable chips?**
   A strong answer names remaining processing, stage timing, yield, testing, and qualification.
   A weak answer treats each start as one finished die.
6. **How would you compare retail shelf availability with humanitarian coverage?**
   A strong answer states the numerator, denominator, time window, observation method, and excluded population for each.
   A weak answer compares two percentages with incompatible units and denominators.
7. **What changes in a fashion transfer from CPG?**
   A strong answer discusses style-color-size, season, lead time, markdown, and returns while recognizing that some features also occur in CPG.
   A weak answer assumes identical products within a style and unlimited time to sell.
8. **What should happen when a quality or access rule makes every option infeasible?**
   A strong answer escalates to an authorized owner and presents the conflict without manufacturing feasibility.
   A weak answer assigns a finite penalty and ships the prohibited stock.
9. **How do you decide between priority and proportional allocation?**
   A strong answer quantifies distributional effects, identifies authorized policy and affected groups, and tests constraints.
   A weak answer declares one arithmetic rule universally fair.
10. **How do you establish that a transferred model improved outcomes?**
    A strong answer proposes a defensible comparator, stable measures, process checks, and uncertainty reporting.
    A weak answer calls any post-launch increase causal.

## 15. Key terms

- [Allocation](#4-definitions-and-vocabulary): assignment of constrained qualified supply to eligible requirements.
- [Assumption ledger](#11-assumptions-and-limits): explicit record of what the analysis relies on and how it can fail.
- [Beneficiary](#4-definitions-and-vocabulary): person intended to benefit from an intervention.
- [Buyer](#4-definitions-and-vocabulary): purchasing or authorizing party, not necessarily the user.
- [Commitment](#2-learning-outcomes): accepted obligation, distinct from an estimate or target.
- [Constraint](#5-the-mechanics-step-by-step): condition defining feasible action.
- [Counterfactual](#4-definitions-and-vocabulary): estimated outcome without the intervention.
- [Coverage](#4-definitions-and-vocabulary): served share of a defined eligible population or requirement.
- [Cycle time](#4-definitions-and-vocabulary): elapsed time across a defined process.
- [Decision owner](#5-the-mechanics-step-by-step): role accountable for the choice and its record.
- [Eligibility](#8-data-standards-and-regulatory-context): whether the item and destination satisfy the defined use rules.
- [Fill rate](#4-definitions-and-vocabulary): supplied amount divided by a specified requested amount.
- [Forecast](#2-learning-outcomes): estimate of a future quantity or outcome.
- [GTIN](#8-data-standards-and-regulatory-context): GS1 identifier for a trade item.
- [JIS](#4-definitions-and-vocabulary): supply in the sequence needed for assembly.
- [Lead time](#4-definitions-and-vocabulary): time from a defined trigger to usable receipt.
- [Lot](#4-definitions-and-vocabulary): quantity sharing a defined production or distribution history.
- [Markdown](#4-definitions-and-vocabulary): reduction of selling price in a specified period.
- [Need](#4-definitions-and-vocabulary): assessed requirement, distinct from recorded transactions.
- [On-shelf availability](#4-definitions-and-vocabulary): defined share of item-location-time observations with product available to shoppers.
- [Plan](#2-learning-outcomes): proposed operational actions and timing.
- [Priority rule](#5-the-mechanics-step-by-step): authorized ordering of eligible requirements.
- [Qualified supply](#6-worked-example): stock usable for the particular decision after relevant gates.
- [Quarantine](#4-definitions-and-vocabulary): restricted status pending authorized disposition.
- [Scenario](#5-the-mechanics-step-by-step): coherent set of assumptions about future conditions.
- [Serial](#4-definitions-and-vocabulary): identifier for an individual item or package where assigned.
- [SKU](#4-definitions-and-vocabulary): operationally distinct item.
- [Target](#2-learning-outcomes): desired performance level, not evidence it was achieved.
- [Traceability](#8-data-standards-and-regulatory-context): ability to follow specified items and events.
- [WIP](#4-definitions-and-vocabulary): work started but not complete at the defined stage.
- [Yield](#4-definitions-and-vocabulary): acceptable output over a specified input population.

## 16. Sources

The sources below are public starting points, not an exhaustive list of requirements.
Standards may be subject to licensing or require the authoritative edition for implementation.
The primary sources support the specific documented statements in sections 8 and 9; numerical cases in this topic are synthetic.

1. GS1, [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin), identifier purpose and trade-item context.
2. GS1, [Global Location Number (GLN)](https://www.gs1.org/standards/id-keys/gln), identifier purpose for parties and locations.
3. U.S. Food and Drug Administration, [Drug Supply Chain Security Act](https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa), statutory framework and implementation resources.
4. European Commission, [Guidelines of 5 November 2013 on Good Distribution Practice of medicinal products for human use (2013/C 343/01)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52013XC1123(01)), distribution quality guidance.
5. International Automotive Task Force, [IATF 16949](https://www.iatfglobaloversight.org/iatf-169492016/), automotive quality management standard information.
6. SEMI, [Standards](https://www.semi.org/en/standards), semiconductor industry standards program and catalog.
7. U.S. Government Accountability Office, [Semiconductors](https://www.gao.gov/semiconductors), overview and related work on the semiconductor supply chain.
8. Sphere Association, [The Sphere Handbook](https://spherestandards.org/handbook/), humanitarian charter and minimum standards.
9. Core Humanitarian Standard Alliance, [The Core Humanitarian Standard](https://www.corehumanitarianstandard.org/), quality and accountability commitments.
10. United Nations OCHA, [About OCHA](https://www.unocha.org/about-ocha), humanitarian coordination and information mandate.
11. U.S. Federal Trade Commission, [Threading Your Way Through the Labeling Requirements Under the Textile and Wool Acts](https://www.ftc.gov/business-guidance/resources/threading-your-way-through-labeling-requirements-under-textile-wool-acts), guidance for covered textile products.
12. European Union, [Regulation (EU) No 1007/2011 on textile fibre names and related labelling and marking](https://eur-lex.europa.eu/eli/reg/2011/1007/oj), original regulation; consult consolidated and current law for application.

## 17. Links

- [Module 29 topic index](_index.md).
- [Module 29 survey and sector lab](../../../course/29_industry_electives.md).
- [CPG and retail](01_cpg-retail.md).
- [Pharmaceutical cold chain](02_pharma-cold-chain.md).
- [Automotive](03_automotive.md).
- [Semiconductors](04_semiconductor.md).
- [Simulation, optimization, and production labs index](../30_simulation_optimization_production_labs/_index.md).
