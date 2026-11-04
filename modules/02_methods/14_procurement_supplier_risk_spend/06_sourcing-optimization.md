# Sourcing Optimization

> Module 14 — Procurement, Supplier Risk, and Spend Analytics
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: topics 01–05; basic algebra
> Version 2026.5

*Award allocation under capacity, minimum volume, and risk constraints.*

## 1. Why this topic exists

A sourcing event ends with a portfolio decision, not a ranking of quotes.
A buyer must cover demand while respecting supplier capacity, technical qualification, contractual minimums, and concentration limits.
Each rule may be reasonable alone and impossible in combination.
The cheapest unit bid can therefore be an expensive recommendation if it cannot be ordered, delivered, or defended.

Consider a manufacturer buying a component for two plants.
One supplier quotes the lowest price to both plants but has a single shared production line.
Awarding both plants their cheapest bid independently counts the same capacity twice.
A second supplier has a minimum economical annual order.
A third requires engineering approval before it can make a regulated product.
The decision is how much each qualified supplier should receive at each plant, subject to all these conditions at once.

The optimization output is a proposed plan, not a purchase order, a forecast, or a supplier commitment.
The forecast estimates future demand; the plan assigns volume against a stated forecast.
A target describes a desired result; a commitment is an obligation accepted under the contract.
A buyer should not convert a modeled volume into a binding commitment without the appropriate approvals.

The business value is not just the lowest objective value.
A model makes trade-offs explicit and permits a controlled comparison of alternatives.
A sourcing committee can ask what a second qualified supplier costs, how much capacity is needed to satisfy a concentration limit, or whether a required local-content rule creates infeasibility.
A defensible answer shows both the numbers and their assumptions.

This lesson uses a small, fictitious annual event, not observed supplier data.
All prices and costs are in 2026 US dollars (USD), volumes are units per year, and capacity is annual unless a different period is named.
No modeled saving is a guarantee of realized savings.

## 2. Learning outcomes

After this lesson you can:

1. Translate an award decision into variables, objective, and constraints, and distinguish plan from commitment.
2. Build a feasible allocation under demand, qualification, shared capacity, minimum-volume, and concentration rules.
3. Compute total modeled cost and compare a constrained award with an unconstrained price baseline.
4. Diagnose infeasibility using arithmetic, constraint relaxation, and business-owner review.
5. Compare disruption scenarios without treating assumed scenario probabilities as measured facts.
6. Execute and interpret an enumeration calculation for a small award problem.
7. Specify governance, data quality controls, and an award approval record.
8. Critique model assumptions and distinguish a solver result from an implementable contract.

Exercises 1–8 and their answers in sections 12–13 test these outcomes.

## 3. The core mental model

An award is a constrained allocation of demand across eligible supply options.
The model searches the set of allocations and ranks only those that satisfy the rules.
It does not decide which business policy is legitimate; owners must approve the rules and the data.

```text
Demand + quote + eligibility + supplier capacity + policy
                         |
                  validate and freeze
                         v
          decision variables and constraints
                         |
        +----------------+------------------+
        |                                   |
     feasible                            infeasible
        |                                   |
  compare total cost                  diagnose conflicting rules
  and risk scenarios                  and propose approved changes
        |                                   |
        +----------------+------------------+
                         v
             human decision and contract
                         |
             monitor actual awards and supply
```

A feasible plan satisfies every hard constraint at the same time.
An optimal plan is a feasible plan with the best objective value among those considered.
The conclusion is conditional on the chosen horizon, data, constraints, and objective.
A solution can be mathematically optimal and operationally unworkable when those inputs are wrong.

Think of capacity as a shared budget across plants.
Think of a minimum award as a fixed activation decision coupled to a volume decision.
Think of risk limits as policy boundaries, not automatic measurements of resilience.
Demand coverage is a balance equation, not a promise that a forecast will materialize.

A model can contain multiple objectives.
For example, it can minimize cost subject to a maximum supplier share, then calculate costs under disruption.
Alternatively, the team can minimize expected scenario cost if scenario probabilities and recourse assumptions are defensible.
These approaches answer different business questions and should not be presented as equivalent.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Sourcing event | Time-bounded process for obtaining bids and choosing awards | Treating a bid as a signed contract |
| Award | Proposed or approved supplier allocation, units/year | Assuming an optimizer output is already binding |
| Demand | Required units in the modeled horizon, units/year | Calling the demand forecast a firm order |
| Supplier | Commercial counterparty; group parent may differ | Counting affiliates as independent capacity |
| Plant | Destination or consuming location | Ignoring different eligibility or freight by location |
| Bid | Supplier offer with scope, validity, currency, and conditions | Comparing incompatible units or service terms |
| Landed cost | Price plus included delivery and other specified acquisition costs, USD/unit | Adding a freight charge already embedded in price |
| Total cost of ownership | Broader cost across acquisition, operation, and disposal | Adding unvalidated risk costs twice |
| Decision variable | Quantity or choice the model is allowed to select | Treating data as a variable without modeling its source |
| Allocation | Quantity assigned to supplier and plant, units/year | Using a percentage without its demand denominator |
| Eligibility | Technical, commercial, and compliance permission for an award | Inferring approval from a bid |
| Capacity | Available production or delivery quantity in a period, units/year | Adding plant-level capacities that share one line |
| Minimum award | Smallest volume if a supplier is selected, units/year | Requiring minimum volume from an unselected supplier |
| Activation | Binary choice indicating a supplier is used | Confusing activation with a signed agreement |
| Concentration | Fraction of total demand assigned to a supplier or group | Treating entity-level diversification as independent risk |
| Hard constraint | Condition that every accepted plan must satisfy | Hiding a negotiable preference as a legal requirement |
| Soft constraint | Preference with an explicit penalty or controlled relaxation | Allowing a legal prohibition to be violated for a price |
| Feasible | Satisfying all specified constraints simultaneously | Assuming a feasible plan is operationally verified |
| Infeasible | No assignment satisfies every modeled constraint | Calling missing data or a time limit proof of infeasibility |
| Optimal | Best feasible objective under the stated mathematical model | Claiming the result is best under unmodeled realities |
| Mixed-integer linear program (MILP) | Linear model with some integer or binary variables | Assuming a local heuristic proves optimality |
| Scenario | Coherent set of assumed demand, capacity, and cost conditions | Treating a stress test as a forecast |
| Recourse | Allowed action after an uncertain event is observed | Assuming emergency capacity exists without qualification |
| Shortage | Demand not covered by feasible deliveries, units/year | Reporting zero shortage because unmet demand was omitted |
| Shadow price | Local objective change for a small right-hand-side change in a continuous model under conditions | Applying a linear-program dual directly to a discrete award |
| Optimality gap | Difference between incumbent and bound relative to a stated convention | Reporting an incumbent as proven optimal after a time limit |
| Baseline | Explicit comparison plan and common scope | Calling an infeasible cheapest-bid calculation achievable savings |
| Realized savings | Audited actual economic benefit against an agreed baseline | Equating modeled bid delta to cash savings |

## 5. The mechanics, step by step

### Step 1: Frame the decision

The input is the sourcing mandate: products, destinations, horizon, and decision owner.
The output is a decision statement that can be checked against the contract.
For example: allocate forecast annual demand for component X at plants East and West among approved suppliers.
Separate the decision horizon from the contract term and the forecast update cycle.
A three-year contract can have annual minimums and quarterly capacity limits.
An annual total alone cannot prove a quarterly plan works.

Record the decision rights.
Procurement owns the sourcing event; engineering owns technical approval; operations validates capacity and delivery; finance validates economics; legal and compliance review applicable obligations.
The sourcing committee accepts or rejects the proposed award.
A modeler encodes agreed policies but does not independently grant exceptions.

### Step 2: Define grain and units

Use one allocation variable for each supplier–plant–item–period combination that can genuinely be awarded.
For this lesson, item and year are fixed, so the grain is supplier × plant.
Demand is in units/year and cost is USD/unit; their product is USD/year.
Do not combine cases, pallets, kilograms, and pieces without an approved conversion.
If plants use different specifications, preserve those differences in eligibility and price.

Set a consistent currency and effective date.
Document whether foreign exchange is fixed, forecast, or passed through.
Do not call a modeled annual USD cost a cash-flow present value unless timing and discounting are included.
Ensure freight, duties, packaging, and quality allowances are defined at the same scope in all bids.

### Step 3: Validate the data

Reconcile demand by plant to the approved planning snapshot.
Flag duplicate bids, expired prices, missing incoterms, incompatible pack sizes, and ambiguous supplier identities.
Confirm capacity with the supplier for the same item, period, and plant mix.
Distinguish gross capacity from the capacity reserved for this buyer.
Document common equipment, subcontractors, and capacity shared by affiliated suppliers.

An empty eligibility field is not evidence of approval.
Resolve it with the relevant owner before optimization or explicitly treat the lane as unavailable.
A missing price is not zero; an unavailable bid must not become a free source.
Keep immutable input versions so a later reviewer can reproduce the decision.

### Step 4: Choose the decision variables

Let x[s,p] be units/year awarded to supplier s for plant p.
Let y[s] be 1 if supplier s is activated and 0 otherwise.
For continuous divisible volume, x is nonnegative and may be fractional.
For indivisible units or lots, specify integer or lot-size restrictions.
The worked example uses integer units and a coarse 100-unit enumeration grid only to make its exhaustive calculation transparent.

For each supplier s and plant p, let c[s,p] be the comparable landed cost in USD/unit.
Let D[p] be demand at plant p, U[s] supplier capacity, and L[s] the minimum award if activated.
Let e[s,p] be 1 for an approved lane and 0 otherwise.
Let alpha[s] be the maximum share of total demand permitted for supplier s.

### Step 5: Set the objective

The simplest objective is to minimize sum over s,p of c[s,p] × x[s,p].
Include a fixed activation cost F[s] × y[s] if onboarding, tooling, or qualification cost is incremental and in scope.
State whether the cost occurs once or every year; do not add a one-time cost to an annual recurring figure without a consistent horizon.
Subtract only discounts that are earned under the awarded quantity and modeled correctly.
A supplier's declining price schedule may require additional binary variables and tier-specific conditions.

Separate the mathematical objective from decision criteria reported outside it.
If quality risk is not credibly monetized, use an approved eligibility rule or report a scenario metric rather than inventing a penalty.
If a risk penalty is used, show its units, source, and sensitivity.
A weighted objective expresses a management trade-off; it does not discover the correct risk appetite.

### Step 6: Cover demand

For each plant p, impose sum over s of x[s,p] = D[p].
This equality assumes the demand quantity is fixed for the plan and all demand must be covered.
If excess can be stored or demand can be deferred, model those decisions and their costs explicitly.
If shortage is allowed, add a nonnegative shortage variable and a demand balance; disclose shortage separately.
Do not allow shortage merely by changing equality to a less-than inequality.

An unserved-demand variable may diagnose infeasibility even if the final production model forbids shortages.
In that diagnostic model, minimize shortages first, then minimize cost with the minimum shortage fixed.
This lexicographic sequence avoids choosing shortage merely because an arbitrary penalty was too low.
A diagnostic solution with shortage is not a feasible full-service award.

### Step 7: Apply capacity and eligibility

For every supplier, impose sum over p of x[s,p] <= U[s] × y[s].
This couples activation to shared annual capacity.
If capacity varies by plant or period, use the relevant plant and period constraints in addition to any shared constraint.
For a prohibited lane, set x[s,p] = 0.
For a qualified lane, do not infer unlimited capacity; retain shared and lane-specific caps.

A tight upper bound improves clarity and computational performance.
It should be the smallest defensible capacity or demand bound, not an arbitrary huge number.
A line can be qualified while its output is still constrained by a shared bottleneck or raw-material supply.
Group-level constraints may be required when legal entities share a facility.

### Step 8: Model minimum award and supplier count

For every supplier impose sum over p of x[s,p] >= L[s] × y[s].
Together with the capacity link, y=0 forces zero volume and y=1 forces at least the minimum.
A minimum is meaningful only if the supplier is actually selected; otherwise a model could wrongly force every bidder into the award.
The legal commitment may apply by item, plant, quarter, or contract year, not merely at supplier total.
Model it at the level at which the obligation is written.

If management requires at least two suppliers, impose sum of y[s] >= 2.
If every selected supplier must have a meaningful share, combine the count with minimum awards.
A one-unit token award can satisfy a count without providing effective continuity.
Conversely, a high minimum can make diversity impossible given demand and capacity.

### Step 9: Encode concentration and common exposure

For fixed total demand T, impose sum over p of x[s,p] <= alpha[s] × T.
Specify the denominator: total annual units, spend, critical-item volume, or plant-specific demand.
A unit-share limit and a spend-share limit are different when prices differ.
For a supplier group, sum all affiliate allocations before applying the group cap.
For geography, sum suppliers exposed to the same country or facility when policy requires it.

A concentration cap is a proxy for exposure, not proof that spare capacity is available after failure.
Two suppliers sharing a tier-two component or a floodplain may fail together.
Map common dependencies and test plausible correlated disruptions.
Do not assume the remaining awarded supplier can instantly absorb the lost volume.

### Step 10: Check basic feasibility before solving

Total eligible capacity must be at least total demand, but that condition is not sufficient.
For each plant, eligible capacity must cover its demand.
For any subset of plants, shared capacity among suppliers eligible for that subset must cover their combined demand.
Minimum awards, supplier counts, share caps, and time buckets can still make a model infeasible after these checks pass.

Calculate the maximum supplier allocation as the minimum of capacity and the applicable share cap.
If these maxima sum below demand, the capacity and concentration combination cannot cover demand.
If two suppliers must be selected but each requires more than half of demand, the count and minimums conflict.
Write the arithmetic in business terms before reaching for a solver diagnostic.

### Step 11: Solve and inspect status

Use a solver appropriate to the variable types and the size of the event.
For an MILP, preserve the solver name, version, termination reason, time limit, incumbent, bound, and gap.
Only a proven optimal termination supports the claim of mathematical optimality under the model.
A feasible incumbent after a time limit may still be useful, but its quality is bounded by the reported gap.
Infeasible, unbounded, numerical error, and no incumbent are different statuses.

Recompute demand balance, every capacity, eligibility, minimum, share, and objective from the exported allocation.
Use tolerances appropriate to the solver and unit granularity.
Convert tiny numerical residuals carefully; rounding individual quantities can break demand balance or a hard cap.
Re-solve with integer or pack constraints when operational quantities must be discrete.

### Step 12: Diagnose infeasibility

First check data and indexing: duplicated demand, wrong units, zero capacity, and accidental eligibility exclusions are common causes.
Then identify which constraints cannot be satisfied together.
Some solvers can produce an irreducible infeasible subsystem (IIS): a subset of constraints that remains infeasible but becomes feasible if any one member is removed.
An IIS need not be unique, smallest, or a business-approved correction.
Bounds can participate and must be included in the diagnosis.

A feasibility relaxation introduces controlled violations, often with different penalties.
It identifies a least-penalty relaxation under those chosen penalties, not necessarily the least disruptive business change.
Report both the violated constraint and its quantity, such as 150 units/year of extra capacity.
Keep safety, legal, and qualification restrictions hard unless their owners authorize a real-world change.
Do not silently relax a policy to obtain a cheaper result.

### Step 13: Compare scenarios

Freeze the base award when assessing its exposure to a disruption.
Define event timing, affected suppliers, surviving capacity, demand, emergency capacity, lead time, cost, and allowed recourse.
Calculate covered and uncovered units by plant.
If scenarios include probabilities, ensure they are coherent and sum to one over the modeled set.
Expected cost is the sum of scenario probability times scenario cost, but the answer is only as credible as those probabilities and costs.

A stress scenario can be informative without assigning it a probability.
Report cost and shortage in that scenario separately rather than inventing a probability.
Compare alternative base awards under the same scenario and recourse assumptions.
Avoid counting the same disruption loss both as an embedded risk premium and as scenario shortage cost.

### Step 14: Decide and implement

Present the award, cost, constraint utilization, scenarios, and the best relevant alternatives together.
Identify which requirements are binding and who owns any exceptions.
Confirm supplier acceptance, contract language, qualification, capacity reservations, ordering rules, and transition dates.
An award that the supplier has not accepted remains a proposal.
An approved contract may specify tolerances and remedies that differ from the model.

Set a monitoring cadence and trigger re-optimization when demand, price, capacity, qualification, or risk changes materially.
Track actual releases and receipts against allocation rather than assuming that the signed plan was executed.
Measure realized savings against the agreed baseline and adjusted actual volume.
Retain the input snapshot, model, result, approvals, and changes in an access-controlled decision record.

### Applied constraint clinic

The following short cases illustrate distinct modeling choices.
Each uses hypothetical quantities, independent of the worked example.
A sourcing owner should confirm each rule before using it in a live event.

#### 1. Capacity reserved for another customer

A supplier reports gross annual capacity of 2,000 units.
It already has 1,300 units committed to other customers.
Its available capacity is therefore 700 units/year, not 2,000.
If the supplier confirms only 500 units reserved for this buyer, use 500.
The 200-unit difference is not a guaranteed option.
Document whether the reservation is enforceable and when it expires.
A high gross capacity figure does not offset a small actual reservation.
Model the reservation at the period in which deliveries are needed.
If commitments change, replace the snapshot and rerun the award.
Ask operations to validate the resource behind the reported number.
A sales quote alone may not represent production's accepted plan.

#### 2. Bottleneck hours across products

One supplier uses a shared machine for items X and Y.
X requires 2 machine-hours/unit; Y requires 3 machine-hours/unit.
The line has 1,200 available machine-hours/year.
The correct constraint is 2x + 3y <= 1,200.
Independent limits x <= 600 and y <= 400 are insufficient.
Together those independent maxima require 2,400 hours.
The coefficient should include expected setup and yield if justified.
Do not mix labor-hours with machine-hours in the same capacity pool.
Obtain the engineering basis for each coefficient and available-hour estimate.
If setups depend on selected products, add explicit setup decisions.
The result allocates scarce machine time, not merely finished units.

#### 3. Plant-specific capacity and shared capacity

Supplier D can deliver at most 300 units/year to East.
It can deliver at most 400 units/year to West.
Its shared production limit is 500 units/year.
Both lane limits and the 500-unit supplier total must hold.
An award of 300 East plus 400 West violates the shared limit by 200.
An award of 200 East plus 300 West satisfies all three limits.
Delivery slots and production availability are different resources.
Confirm whether each maximum is a genuine limit or a commercial offer.
A missing lane cap is not a reason to discard the shared constraint.
Track which constraint is binding in the approved recommendation.
Ask the supplier whether the proposed geographic split is workable.

#### 4. Minimum order by destination

A supplier offers a minimum of 200 units/year per active plant.
Its total award could be 300 units split 150 East and 150 West.
That split satisfies a 200-unit supplier-level minimum but violates both plant minimums.
Use a separate binary variable for each active plant lane.
For each lane, link its volume to the corresponding minimum and maximum.
Also link lane activation to the supplier activation when needed.
The minimum may instead be per shipment or per contract year.
A year-level total cannot enforce a shipment-level requirement.
Read the commercial clause before choosing the index of the binary variable.
Record whether the minimum is an obligation or a quoted-price condition.
If price changes below the threshold, model the alternative price explicitly.

#### 5. Minimum lots and exact coverage

A plant needs exactly 950 units/year.
Both available suppliers ship only in lots of 100 units.
No combination of whole lots equals 950 units.
A continuous allocation is feasible but not implementable as written.
An integer-lot model exposes the 50-unit mismatch.
If inventory can absorb it, allow 1,000 units with 50 units of excess.
Include storage, expiry, and ownership effects for those extra units.
If overbuying is forbidden, ask whether demand or package size can change.
Do not round 950 to 1,000 without an authorized inventory decision.
The ability to serve demand and the ability to order exact demand differ.
A feasibility diagnosis should name the incompatible lot and demand rules.

#### 6. Maximum share by spend rather than volume

A supplier receives 400 units at USD 20/unit.
Other suppliers receive 600 units at USD 10/unit.
Its unit share is 400/1,000, or 40%.
Its spend share is 8,000/14,000, or approximately 57.1%.
A 50% volume cap is met, but a 50% spend cap is not.
When prices differ, the choice of denominator changes the decision.
For fixed per-unit costs, supplier spend can be expressed linearly.
A variable total-spend denominator needs careful algebra and positive spend.
Use the policy's actual definition rather than the analyst's preferred proxy.
Check whether the relevant measure includes freight, tax, or rebates.
Report both views if they reveal different exposure.

#### 7. Supplier group and site caps

Entities D1 and D2 each have an individual 50% cap.
Each receives 400 of 1,000 total units.
Their individual shares are 40%, yet their combined share is 80%.
If both use the same factory, that factory carries 800 units of exposure.
A 60% group cap would limit their combined award to 600 units.
Create group membership for the modeled period, not a permanent assumption.
Corporate ownership can differ from manufacturing-site dependence.
A site cap and a corporate cap may both be needed.
Validate plant relationships with supplier and risk-management evidence.
Do not infer operational independence from different invoice names.
A business owner must approve the group definition and threshold.

#### 8. Diversity with a meaningful minimum

A buyer needs 1,000 units/year and requires at least two suppliers.
An unconstrained count rule can select one supplier for 999 units and another for one.
That outcome meets the count but may not support continuity.
Suppose operations establishes a 200-unit annual minimum for an active source.
The second supplier must then receive at least 200 units.
This rule increases recurring volume at the backup and may raise cost.
It still does not prove the backup can ramp after an outage.
A qualification and ramp test is a separate operational requirement.
Avoid choosing the minimum merely to make the model look diversified.
Compare the cost and service implications of several approved thresholds.
A threshold should reflect real production, tooling, and contract constraints.

#### 9. A conflicting minimum and count

Demand is 1,000 units/year and two suppliers must be selected.
Each selected supplier requires at least 600 units/year.
The count rule requires at least 1,200 units of awards.
The demand equality permits exactly 1,000 units.
The two conditions cannot both hold, regardless of capacity or price.
A diagnostic should report the 200-unit conflict and the relevant rules.
A possible repair is to negotiate a lower minimum with a supplier.
Another is to revise the required supplier count, if authorized.
A third is to permit inventory overbuying if operationally and financially sound.
No repair should be treated as automatic or free.
Re-solve after a real rule change and inspect the resulting award.

#### 10. Demand shortfall diagnostic

Demand is 1,000 units; the maximum qualified supply is 940.
A hard demand equality makes the award infeasible.
A diagnostic shortage variable can measure the missing 60 units.
First minimize shortage under all non-negotiable restrictions.
Then fix shortage at its minimum and minimize delivered purchase cost.
This two-stage objective avoids trading extra shortage for cheap bids.
The answer is a 940-unit service plan plus a 60-unit gap.
It is not a feasible 1,000-unit full-service award.
An owner can investigate alternative capacity, a demand deferral, or redesign.
Different plant restrictions can make the minimum shortage exceed 60.
The diagnostic must still enforce lane qualification and shared capacity.

#### 11. Capacity is sufficient in total but not at a plant

East and West each require 500 units/year.
Supplier A can deliver 800 units, but only to East.
Supplier B can deliver 200 units, but only to West.
Total stated capacity is 1,000 units, equal to total demand.
West nevertheless has a 300-unit capacity gap.
A's unused East capacity cannot serve the West plant.
A feasibility check must preserve eligibility and destination.
A possible repair is 300 qualified West units from another source.
Moving demand from West to East requires an actual transfer capability.
Transport cost and approval must accompany such a change.
A global sum alone is not a certificate of feasibility.

#### 12. Seasonality defeats annual capacity

A supplier can produce 250 units in each quarter.
Its annual capacity of 1,000 units equals annual demand of 1,000.
Demand is 100, 100, 100, and 700 units in the four quarters.
Without inventory or alternative supply, quarter four is short 450 units.
The annual balance hides the timing mismatch.
If inventory is allowed, early production may serve later demand.
Inventory balance, storage, cash, and shelf life must be modeled.
Starting inventory and terminal inventory assumptions also matter.
A supplier must confirm early production is available to this buyer.
A capacity reservation in quarter one does not automatically transfer to quarter four.
Time-phased feasibility is a different question from annual feasibility.

#### 13. A price tier that applies to all units

A supplier charges USD 12/unit below 500 units/year.
At 500 units or more it charges USD 10/unit on every unit.
At 499 units the purchase cost is USD 5,988/year.
At 500 units it is USD 5,000/year under the stated offer.
This discontinuity cannot be represented by a single linear price.
Use tier-selection binaries with the appropriate volume bounds.
Confirm whether the tier applies retroactively to all units or only incremental units.
The alternative incremental schedule has different economics.
Check currency, eligible products, and the measurement period.
An all-units discount can create incentives to award excess volume.
Demand and inventory constraints must prevent artificial overbuying.

#### 14. One-time qualification cost

A new supplier saves USD 2/unit against the incumbent.
The planned annual volume is 1,000 units and qualification costs USD 3,000 once.
Year-one variable savings are USD 2,000.
The simple year-one net effect is a USD 1,000 cost increase.
Over two unchanged years, undiscounted variable savings are USD 4,000.
The two-year net benefit before discounting is USD 1,000.
The model horizon and timing determine which comparison is appropriate.
If the supplier is not selected, the avoidable qualification cost is zero.
Link that fixed cost to the activation decision.
Do not charge a sunk cost again if it has already been incurred.
A value claim should disclose whether discounting and transition risks are included.

#### 15. Qualification is not the same as capacity

Engineering has approved supplier E for both plants.
Operations confirms only 300 units/year of buyer-specific capacity.
Demand is 800 units/year at the two plants combined.
Approval allows E to receive volume but does not authorize 800 units.
The model must retain the 300-unit supplier capacity.
Conversely, a supplier offering 800 units without approval is not eligible.
Commercial interest does not override the engineering restriction.
The two facts require separate fields and separate owners.
An expired approval should trigger a review, not automatic extension.
If qualification depends on a specific site, preserve the site identity.
A facility transfer can change the validity of the approval.

#### 16. Conditional capacity expansion

Supplier F can provide 400 units/year on its existing line.
It offers another 300 if the buyer funds USD 8,000 of tooling.
Introduce an expansion binary and link the extra capacity to it.
Available capacity becomes 400 + 300 times the expansion binary.
Include the tooling cost in the objective on a consistent horizon.
Also model the tooling lead time before expanded capacity is available.
A selected expansion option is not evidence the supplier built the line.
A signed agreement and acceptance test are needed for implementation.
If the expansion is uncertain, test delays and cost overruns separately.
Do not award expanded volume before the production ramp is realistic.
The buyer should understand ownership and residual value of the tooling.

#### 17. Take-or-pay and unused volume

A contract requires payment for at least 300 units/year.
Actual demand may be only 240 units in a low-demand scenario.
If the contract is take-or-pay, payment can exceed physical deliveries.
Define a paid quantity separately from a delivered quantity.
The paid quantity must reflect the contractual minimum and actual terms.
Otherwise the model understates the cost of low demand.
Check whether unused entitlement rolls forward or expires.
A physical minimum delivery is different from a payment minimum.
Both may coexist and have different inventory consequences.
Use contract-owner review to specify the relevant relationship.
A forecast quantity is not a substitute for the legal commitment.

#### 18. Multi-year award and switching

A buyer expects 1,000 units in each of two years.
Moving to a new supplier costs USD 2,500 once.
The new supplier is USD 2/unit cheaper in both years.
Undiscounted purchase savings are USD 4,000 across two years.
Net savings before other effects are USD 1,500.
If the contract lasts only one year, the same move loses USD 500.
A model that ignores the horizon can choose the wrong supplier.
If suppliers are activated or dropped by year, link transition variables across years.
Include ramp constraints if a new supplier cannot start at full volume.
Discount future amounts when the finance policy calls for present value.
Document assumptions about demand and price changes in each year.

#### 19. Conditional price validity

A supplier quotes USD 10/unit for an award signed by June 30.
The next approved quote is USD 11/unit after that date.
At 1,000 units the annual difference is USD 1,000.
The cheaper offer is not available if the decision is made after expiry.
Model only valid offers for the relevant decision date.
An unsigned extension is not evidence that the old quote remains valid.
If the supplier agrees to extend, store the written revision.
Do not blend two offers without modeling their respective periods.
A negotiation scenario can show a conditional value but not claim it as achieved.
Include implementation timing in the decision record.
The contract owner must verify the final commercial terms.

#### 20. Incumbent baseline versus unconstrained lower bound

The incumbent plan costs USD 10,500/year at current forecast volume.
A feasible optimized proposal costs USD 9,800/year on the same scope.
The modeled difference is USD 700/year before implementation effects.
The USD 9,600 price-only calculation is an infeasible lower bound.
It is not the incumbent baseline and cannot support a savings claim.
If the demand mix changes, normalize both plans to a common mix.
If the incumbent price expires, specify the counterfactual price assumption.
Avoid combining a one-time transition cost with a recurring saving without time alignment.
Finance should approve the baseline and savings methodology.
Actual savings require executed volume, actual prices, and appropriate adjustments.
The optimizer does not by itself establish cash or accounting benefit.

#### 21. Robustness and expected value are different

Plan P costs USD 100 in the base case and USD 400 in disruption.
Plan Q costs USD 120 in the base case and USD 250 in disruption.
With disruption probability 10%, P has expected cost USD 130.
Under the same probabilities Q has expected cost USD 133.
P has the lower expectation, while Q has the lower disruption cost.
A worst-case criterion would prefer Q among these two plans.
The expected-value calculation depends on the assumed 10% probability.
Both scenarios must include comparable operating and shortage consequences.
The risk owner should decide whether the downside is tolerable.
Do not label one criterion universally superior to the other.
The numbers are illustrative and are not a recommended risk appetite.

#### 22. Correlated supplier disruption

Two suppliers each have a 10% assumed outage probability.
If outages are independent, their joint outage probability is 1%.
If they share a factory and always fail together, it can be 10%.
The marginal probability alone does not specify the joint distribution.
A dual-source plan can therefore have very different common-loss exposure.
Map the factory, power, logistics, and tier-two dependencies.
Use joint stress scenarios when dependence is plausible but uncertain.
Do not multiply probabilities unless independence is defensible.
Even a low modeled joint probability does not establish available recovery capacity.
The outage duration and demand during the outage also determine impact.
Use measured evidence where available and label assumptions where not.

#### 23. Recourse capacity after failure

Supplier G has a 500-unit annual nameplate capacity.
Its existing award is 400 units and another supplier fails.
Only 100 units of nominal headroom remain if G's original award continues.
If qualification and material allow a 50-unit ramp, effective recourse is 50.
Adding the full 500 units as emergency capacity double-counts existing production.
Time matters: annual headroom may not be available during the outage month.
Reserve and test the alternative capacity before relying on it.
Check whether extra volume has a different price or expedite charge.
A model can impose both remaining-capacity and ramp constraints.
Contractual permission is necessary but not sufficient for physical delivery.
Report the amount and timing of uncovered demand after feasible recourse.

#### 24. Solver status and decision confidence

A solver finds a feasible award costing USD 1,020,000.
Its valid lower bound is USD 1,000,000 when the time limit ends.
The incumbent is at most USD 20,000 above the unknown optimum.
Using the incumbent as denominator, the relative gap is about 1.96%.
A solver may use a different denominator or convention; report its definition.
The incumbent is feasible only within the solver's stated tolerances.
Independently check the exported award and operational rounding.
Do not report USD 1,020,000 as a proven mathematical optimum.
The committee can decide whether the gap is material to the award.
Keep the time limit, bound, gap, and model version in the record.
A smaller numerical gap does not correct missing business constraints.

#### 25. Why a shadow price can mislead

A continuous model says one more capacity unit reduces cost by USD 3 locally.
That statement applies only within the relevant validity range and assumptions.
If a supplier requires a 200-unit minimum, capacity changes can trigger a discrete switch.
The value of one extra unit may then be zero or discontinuous.
An MILP does not generally have a directly comparable global shadow price.
Re-solve the complete model at the capacity levels of interest.
Compare objective changes and the resulting allocation, not only one dual value.
An increment of 10 units and an increment of 500 units can have different values.
Ensure any extra capacity is qualified and contractually available.
The marginal modeled benefit is not necessarily the amount to pay for expansion.
Include implementation cost and risk in the commercial decision.

#### 26. Detecting incompatible data snapshots

Demand is extracted on Monday and totals 1,000 units/year.
An updated forecast on Friday totals 1,100 units/year.
Supplier capacities were confirmed against the Monday mix.
Running Friday demand with Monday reservations may be technically feasible but misleading.
Record effective timestamps and owners for each data set.
Reconfirm suppliers if the new mix changes location or peak requirements.
A consistent snapshot is a defined decision input, not necessarily the latest row in every table.
If an update is intentionally excluded, record the reason and impact.
Model runs should use stable input identifiers and a versioned configuration.
A reviewer should be able to reproduce the exact award calculation.
A changed input requires a new run and a new comparison record.

#### 27. Separate award authority from execution

The model recommends 500 units to a supplier.
The sourcing committee approves the recommendation subject to contract signature.
The supplier accepts only 400 units in the final agreement.
The remaining 100 units are not covered by the model's original assumption.
Update capacity and rerun or otherwise obtain an approved feasible allocation.
A signed contract may allow releases within a range, not require 500 units.
Map the award into the ordering system with the actual legal terms.
Monitor purchase orders and receipts against the approved allocation.
Investigate deviations caused by demand, qualification, or capacity changes.
Record which decision maker accepted any revised concentration exposure.
The process ends with usable supply and controlled execution, not a solver file.

#### 28. Unit yield versus input capacity

A supplier has material for 1,000 input pieces/year.
Its validated finished-good yield is 90% for this specification.
Expected output is 900 acceptable units, not 1,000.
If the commitment requires 950 good units, the material input is insufficient at that yield.
At 90% yield, 950 good units require approximately 1,056 input pieces.
This calculation uses expected yield, not a guarantee of realized yield.
A conservative plan may need a variability allowance approved by operations.
Avoid applying yield twice if the supplier's quoted capacity already means good units.
The quality owner should define the acceptance standard and measurement period.
Record the yield basis alongside the capacity field.

#### 29. Two-stage policy comparison

A buyer wants the lowest cost while preserving at least 95% service in every specified stress scenario.
First define each stress scenario, its demand, and its available recourse.
For each scenario impose covered demand of at least 95% of that scenario's demand.
Then minimize base cost across awards satisfying those scenario service constraints.
A plan that meets average service but fails one required scenario is not feasible.
A scenario with 1,000 units of demand permits at most 50 units short.
A scenario with 800 units of demand permits at most 40 units short.
Do not use the same absolute shortage allowance for both unless policy says so.
The scenario set is finite and cannot guarantee protection against every possible event.
Identify omitted events and seek management approval of the modeled coverage.

#### 30. Feasibility relaxation and authority

A diagnostic model reports two possible repairs for a 100-unit gap.
One is 100 units of additional approved capacity at a cost of USD 500.
The other is a qualification exception with an assumed penalty of USD 100.
The smaller numerical penalty does not make the exception permissible.
If qualification is mandatory, hold it hard and remove that candidate repair.
If more capacity requires tooling, validate timing and obtain the actual quote.
The remaining relaxation identifies a conditional route to feasibility, not an approved award.
Record the policy owner, required evidence, and decision deadline.
If neither repair is available, retain the infeasible status and escalate the uncovered demand.
Never present the diagnostic penalty as an actual supplier price.

## 6. Worked example

### Case and inputs

A fictional manufacturer sources one interchangeable component for two approved plants over one year.
The demand forecast is 600 units/year at East and 400 units/year at West, for 1,000 units/year total.
The forecast is the planning basis, not a purchase commitment.
All cost figures below are comparable landed USD per unit for the modeled 2026 year.
No onboarding cost or quantity discount is included.

| Supplier | East cost USD/unit | West cost USD/unit | Shared capacity units/year | Minimum if selected units/year | Eligibility |
|---|---:|---:|---:|---:|---|
| A | 10 | 11 | 600 | 200 | Both plants |
| B | 12 | 9 | 500 | 200 | Both plants |
| C | 13 | 12 | 400 | 200 | East only |

A policy limits any supplier to 50% of total units, or 500 units/year.
At least two suppliers must be selected.
A supplier's capacity is shared across both plants.
Each selected supplier must receive at least its listed minimum; an unselected supplier receives zero.
C cannot serve West because its qualification is limited to East.

### Establish a price-only benchmark

Ignoring every capacity, minimum, and concentration constraint, East would buy 600 units from A at USD 10/unit.
West would buy 400 units from B at USD 9/unit.
The price-only calculation is 600 × 10 + 400 × 9 = USD 9,600/year.
This is an infeasible benchmark because A would hold 600/1,000 = 60% of demand, above the 50% cap.
It is not an achievable baseline and the USD difference from it is not a realized loss.

### Build and prove the base award

Assign 500 units/year at East to A, 100 at East to C, and 400 at West to B.
This assignment is not feasible because C's 100-unit award is below its 200-unit minimum.
The minimum therefore changes the optimal mix, not just the supplier count.

Assign 400 East units to A, 200 East units to C, and 400 West units to B.
Demand is covered: East 400 + 200 = 600 and West 400 = 400 units/year.
A has 400 units, B has 400, and C has 200; all are within capacity, cap, and minimum.
The cost is 400 × 10 + 200 × 13 + 400 × 9 = USD 10,200/year.

A cheaper feasible award exists: A supplies 500 East units, C supplies 200 East units, B supplies 100 West units, and A supplies no West units would overfill East.
Correcting that attempted allocation requires moving 100 East units from A to B and increasing B's total above its cap if B retains all West demand.
Instead, allocate A 500 East, B 100 East and 300 West, and C 200 East still overfills East by 200 units.
These examples show why each plant balance must be checked before accepting a supplier-total argument.

The actual optimal award is A: 400 East; B: 100 East and 400 West; C: 100 East, but C's minimum again fails.
Therefore C must either be absent or receive at least 200 units.
If C is absent, A and B must each receive exactly 500 units because each has a 500-unit cap and demand is 1,000 units.
Assign A 500 East, B 100 East and 400 West.
All requirements hold: A uses 500 of 600 capacity, B uses 500 of 500 capacity, C is unselected, and both selected suppliers exceed minimums.
Its cost is 500 × 10 + 100 × 12 + 400 × 9 = USD 9,800/year.

To prove optimality, compare with the price-only USD 9,600 benchmark.
A cannot serve more than 500 East units, so at least 100 East units must move away from A.
The cheapest eligible replacement for those East units is B at USD 12, an extra USD 2/unit.
Hence every feasible award costs at least USD 9,600 + 100 × 2 = USD 9,800/year.
The proposed allocation reaches that bound and is therefore optimal under these assumptions.
The USD 200/year difference from the infeasible price-only benchmark is the modeled cost of the policy and other constraints together, not necessarily the isolated cost of any single policy.

### Capacity, minimum, and policy sensitivity

A is at its 500-unit concentration cap but has 100 units of physical capacity unused.
B is at both its 500-unit cap and 500-unit capacity.
Adding B capacity alone cannot change the plan while B's 500-unit concentration cap remains.
Raising A's share cap alone to 60% permits A 600 East and B 400 West, restoring the USD 9,600 price-only result.
This comparison changes one policy and assumes all other inputs remain fixed.

If B's capacity falls to 300 units/year while all other rules remain, A can provide at most 500 and C at most 400.
Total maximum volume is 1,200, so total capacity alone looks sufficient.
West has only A and B eligible; its 400 units can still be covered in aggregate.
A feasible plan exists: A supplies 400 West and 100 East; B supplies 300 East; C supplies 200 East.
Its cost is 400 × 11 + 100 × 10 + 300 × 12 + 200 × 13 = USD 11,600/year.
This is a reoptimized plan, not evidence that a disrupted supplier can be replaced instantly.

### Explicit infeasibility

Suppose B capacity falls to 200 units/year and C capacity falls to 200 units/year, while the 50% share cap remains.
A can supply at most 500, B at most 200, and C at most 200 units/year.
Their maximum total is 900 units/year against 1,000 units/year demand.
At least 100 units/year are uncovered, regardless of the price or plant assignment.
The hard-demand model is infeasible; it has no optimal award.

A capacity-only relaxation of A's cap to 600 units would permit 100 more A units if A's physical capacity is still 600.
For example, A supplies 400 West and 200 East, B supplies 200 East, and C supplies 200 East.
The plan costs 400 × 11 + 200 × 10 + 200 × 12 + 200 × 13 = USD 11,400/year.
That is a conditional alternative requiring a policy exception; it is not a valid result under the original cap.
Alternatively, find 100 qualified units of capacity elsewhere and test all minimum and plant restrictions again.

### Scenario risk with explicit recourse

Keep the USD 9,800 base award fixed until a disruption occurs.
Suppose A becomes unavailable for the modeled year, demand remains 1,000, and B and C can deliver up to their stated annual capacities after the event.
C remains East-only; B can serve both plants; no emergency supplier exists.
For this stress calculation the base-award share and minimum policies are waived only for post-event recourse, and there is no lead-time or ramp restriction.
B and C can supply at most 500 + 400 = 900 units, so at least 100 units are unserved.
A feasible maximum-coverage recourse is B 400 West plus 100 East, C 400 East, and 100 East units short.
Delivered purchase cost is 400 × 9 + 100 × 12 + 400 × 13 = USD 10,000/year, excluding shortage consequences.
That USD 10,000 is not total economic cost because 100 units are not supplied.

If a hypothetical externally approved shortage consequence is USD 50 per unit, the scenario consequence is 100 × 50 = USD 5,000.
Total delivered purchase cost plus this simplified consequence is USD 15,000/year.
This is not a probability-weighted estimate and does not include switching costs, timing, or reputational loss.
If the base scenario probability were 0.9 and this disruption probability 0.1, and the scenarios were exhaustive, the simplified expected cost would be 0.9 × 9,800 + 0.1 × 15,000 = USD 10,320/year.
The probabilities and shortage consequence are illustrative assumptions, not empirical estimates.
The expectation is meaningful only if the base and disrupted cost scopes are comparable and no overlapping scenario is omitted.

## 7. Code, query, or calculation

The following Python 3 program exhaustively enumerates allocations in 100-unit increments.
It uses only the standard library and no commercial solver.
It checks the demand, eligibility, shared capacity, minimum, concentration, and supplier-count rules.
The grid is a restriction, so enumeration alone proves optimality only on that grid.
The lower-bound proof in section 6 establishes the same optimum for all nonnegative continuous quantities.

```python
from itertools import product

suppliers = ("A", "B", "C")
plants = ("East", "West")
demand = {"East": 600, "West": 400}
capacity = {"A": 600, "B": 500, "C": 400}
minimum = {"A": 200, "B": 200, "C": 200}
price = {
    ("A", "East"): 10, ("A", "West"): 11,
    ("B", "East"): 12, ("B", "West"): 9,
    ("C", "East"): 13,
}
step = 100
share_limit = 500
best_cost = None
best_allocations = []

# Enumerate East awards; demand fixes C's residual East quantity.
for a_e, b_e in product(range(0, 601, step), repeat=2):
    c_e = demand["East"] - a_e - b_e
    if c_e < 0 or c_e % step:
        continue
    # C is not qualified at West; demand fixes B's residual.
    for a_w in range(0, demand["West"] + 1, step):
        b_w = demand["West"] - a_w
        allocation = {
            ("A", "East"): a_e, ("A", "West"): a_w,
            ("B", "East"): b_e, ("B", "West"): b_w,
            ("C", "East"): c_e,
        }
        totals = {
            s: sum(allocation.get((s, p), 0) for p in plants)
            for s in suppliers
        }
        if sum(t > 0 for t in totals.values()) < 2:
            continue
        if any(t > capacity[s] or t > share_limit
               or (0 < t < minimum[s])
               for s, t in totals.items()):
            continue
        cost = sum(price[k] * q for k, q in allocation.items())
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_allocations = [allocation]
        elif cost == best_cost:
            best_allocations.append(allocation)

if best_cost is None:
    print("Infeasible on the 100-unit grid")
else:
    print(f"Best cost: USD {best_cost:,}/year")
    print(f"Number of optimal grid allocations: {len(best_allocations)}")
    for allocation in best_allocations:
        for (supplier, plant), qty in allocation.items():
            if qty:
                print(f"{supplier} -> {plant}: {qty} units/year")
```

Expected output:

```text
Best cost: USD 9,800/year
Number of optimal grid allocations: 1
A -> East: 500 units/year
B -> East: 100 units/year
B -> West: 400 units/year
```

The loop never creates a C–West variable, so the unqualified lane cannot be selected.
The residual calculations enforce each plant's demand equality.
The capacity check uses supplier totals, not independent plant caps.
The minimum check applies only when the supplier has a positive total.
The share cap is fixed at 500 units because total modeled demand is fixed at 1,000 units.
If demand changes, recompute the cap from the approved denominator.
A more realistic event should use an MILP solver, validate its status, and independently check the returned allocation.

## 8. Data, standards, and regulatory context

A procurement optimization depends on the provenance and commercial interpretation of its inputs.
The supplier master should distinguish legal entity, ultimate parent, manufacturing site, and subcontractor.
Those identifiers answer different questions: contract enforceability, concentration, capacity, and common-cause exposure.
No single supplier identifier automatically represents all four.

Maintain bid records at item, destination, period, currency, unit, and service scope.
Keep bid validity, award conditions, capacity statements, and approval evidence with each version.
Preserve original supplier submissions alongside normalized calculations.
A commercial price of USD 10 per case is not directly comparable with USD 10 per unit.

Product and party identifiers can support matching, but they do not establish qualification.
GS1 describes identifiers such as the Global Trade Item Number (GTIN) and Global Location Number (GLN).
Their use is a data convention unless a relevant trading arrangement or law requires it.
A correct identifier does not prove a supplier has capacity or meets a technical specification.

Incoterms rules published by the International Chamber of Commerce allocate specified delivery responsibilities between seller and buyer when incorporated in a contract.
They do not themselves replace the parties' full contract or automatically determine every tax, ownership, or insurance question.
Normalize bids only after reviewing the applicable named rule, place, and contract terms.
The ICC source below is an official overview, not legal advice.

In public procurement, award criteria, transparency, competition rules, and permissible changes depend on jurisdiction and the specific procurement.
For example, the US Federal Acquisition Regulation (FAR) Part 15 addresses contracting by negotiation for covered federal acquisitions.
Do not assume a private-sector sourcing model automatically satisfies a public tender's requirements.
Have counsel identify the governing rules and whether criteria must be disclosed before bids are evaluated.
A hard legal or safety exclusion should not become a tunable cost penalty.

Data access must reflect bid confidentiality and competition-law requirements.
Limit supplier-specific bid access, record who can approve exceptions, and avoid sharing one bidder's confidential terms with another.
Check retention and privacy obligations with the relevant owner.
This teaching example is educational, not procurement, legal, or trade-compliance advice.

## 9. Industry practice and case evidence

Combinatorial sourcing research describes bidding and allocation where a supplier's price may depend on the portfolio it wins.
Caplice and Sheffi discuss optimization-based procurement for transportation services, where award interactions and operational constraints matter.
Their work supports the general point that lane-by-lane selection can miss portfolio effects; it is not evidence about the fictional component case in this lesson.
Transportation capacity and manufacturing capacity differ, so translate the model rather than copying its assumptions.

The public procurement literature also distinguishes an optimization formulation from the process that gives its inputs legitimacy.
An award can be mathematically attractive yet unacceptable if qualification, disclosed criteria, or contract conditions were not observed.
The specific public rules depend on the buyer and jurisdiction.
The model record should show how each input and restriction entered the award decision.

A practical sourcing committee usually compares a small number of interpretable alternatives.
Examples include a lowest-modeled-cost award, a stricter concentration cap, and a continuity-focused stress case.
These are decision designs, not assertions about any named company's practice.
Display their costs, qualification status, capacity utilization, and disruption consequences on a common scope.

[INFERENCE] Where switching requires qualification and tooling, an apparently diversified award may not be a credible contingency unless the secondary supplier has tested capacity and a release process.
The inference follows from the distinction between an allocation and operationally available recourse; it is not a measured universal effect.
A business should validate ramp time, material availability, and commercial terms before relying on that contingency.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Double-counted capacity | Plant caps sum above confirmed line capacity | One resource appears multiple times | Supplier cannot deliver total award | Add shared bottleneck constraint |
| Incorrect minimum | Zero award still violates a minimum | Minimum not linked to activation | False infeasibility | Use binary activation and conditional minimum |
| Token second source | Second award is negligible | Count rule lacks minimum | Diversification is nominal | Set operationally justified minimum |
| Hidden unqualified lane | Missing approval interpreted as yes | Null treated as eligible | Unsafe or prohibited award | Default to review and require evidence |
| Mixed units | Bid prices differ by factor near pack size | Cases and units compared directly | False lowest-cost supplier | Normalize and reconcile units |
| Inconsistent freight | One quote includes delivery, another excludes it | Noncomparable landed cost | Misleading savings | Normalize scope and incoterm |
| Annual capacity masks seasonality | Annual award fits, peak quarter does not | Aggregation hides bottleneck | Late deliveries | Add period-level capacity and demand |
| Affiliate concentration | Two names share a plant | Entity IDs obscure common exposure | Common disruption | Add group and facility constraints |
| Missing recourse limit | Disruption scenario instantly moves all volume | Ramp and qualification omitted | Overstated resilience | Model lead time and available backup |
| Silent shortage | Model returns a cheap partial award | Demand balance is inequality | Unserved production | Require equality or explicit shortage |
| False optimality | Time-limited run reported as optimum | Incumbent confused with proof | Unjustified decision confidence | Report status, bound, and gap |
| Arbitrary risk penalty | Small penalty drives large shortage | Units and trade-off not approved | Hidden policy choice | Calibrate or use explicit constraints |
| Supplier rejection | Winning volume exceeds accepted reservation | Bid or capacity not confirmed | Award cannot be contracted | Obtain acceptance and reservation |
| Rounding violation | Exported units no longer sum to demand | Continuous values rounded independently | Infeasible ordering plan | Use lot-size variables or repair and recheck |
| Unstable baseline | Saving changes when scope changes | Comparison mixes periods or volume | Overstated benefit | Freeze and reconcile baseline |
| Wrong infeasibility diagnosis | Relaxation changes legal restriction first | Penalty represents convenience, not authority | Invalid recommendation | Label hard rules and obtain owner approval |

## 11. Assumptions and limits

| Assumption | Why used | What breaks it | How to test |
|---|---|---|---|
| Demand is fixed | Enables annual balance and fixed share cap | Forecast error or mix changes | Run demand scenarios and compare forecast revisions |
| Costs are linear | Makes unit economics transparent | Tiered discounts, setup, freight thresholds | Compare actual contract schedules to linear price |
| Annual capacity is available | Bounds total allocation | Peak demand or maintenance | Validate capacity by month and bottleneck |
| Units are interchangeable | Permits allocation by plant | Different specifications or revisions | Verify approved part and engineering change |
| Minimum applies to annual supplier total | Simplifies conditional commitment | Contract minimum is quarterly or by plant | Read proposed contract and revise grain |
| Supplier share uses units | Defines the policy denominator | Policy is based on spend or critical units | Obtain the actual policy definition |
| C cannot serve West | Reflects qualification | Approval status changes | Obtain current plant-specific qualification record |
| Bid price remains valid | Treats offers as comparable | Expiry, index adjustment, or negotiation | Confirm validity and adjustment clauses |
| Scenario recourse is instantaneous | Simplifies stress calculation | Long lead time or no reserved capacity | Conduct a time-phased ramp exercise |
| Disruption costs are complete | Supports an expected-cost comparison | Lost sales, production interruption, or overlap omitted | Reconcile consequence model with finance |
| Supplier failures are independent only if modeled so | Supports particular probability calculations | Shared site, tier-two, or geography | Map dependencies and test joint scenarios |
| Grid enumeration represents the small example | Keeps code executable without a solver | Optimal allocation falls between grid points | Use continuous proof or a full MILP |

## 12. Exercises

### Exercise 1 — Formulate the decision

A buyer has three suppliers and two plants, with the input table in section 6.
Write the objective and demand, capacity, eligibility, minimum, and share constraints in words or symbols.
State which variables are binary and which have units.
Deliverable: one page of model specification.
Acceptance: all constraints use the correct supplier or plant grain and minimums apply only when selected.
Time: 15 minutes.

### Exercise 2 — Evaluate an allocation

Evaluate A East 400, B West 400, C East 200 units/year.
Calculate total cost, each supplier's utilization, and whether the allocation satisfies every rule in section 6.
Deliverable: a calculation and feasibility verdict.
Acceptance: include the USD/year cost and identify demand coverage, eligibility, minimums, capacities, and concentration.
Time: 10 minutes.

### Exercise 3 — Find the optimum

Starting from the price-only allocation, identify why it violates policy and derive an optimal feasible award.
Deliverable: allocation, objective value, and a lower-bound argument.
Acceptance: show the 100-unit shift and prove no cheaper eligible replacement exists under the stated prices.
Time: 15 minutes.

### Exercise 4 — Diagnose infeasibility

Use the reduced capacities A 600, B 200, and C 200 units/year, retaining the 50% cap.
Determine the smallest capacity shortfall implied by supplier-level limits and propose one conditional repair.
Deliverable: arithmetic and an approval-aware recommendation.
Acceptance: show the 900-unit maximum, 100-unit gap, and a plant-feasible repaired allocation.
Time: 15 minutes.

### Exercise 5 — Price a stress scenario

With A unavailable, B and C retain their stated capacities and qualification; the recourse policy in section 6 applies.
Calculate maximum service, an allocation, and the simplified cost with USD 50 per unserved unit.
Deliverable: units served, units short, delivered cost, and cost including the hypothetical shortage consequence.
Acceptance: do not describe the scenario as a forecast or omit the shortage.
Time: 15 minutes.

### Exercise 6 — Inspect code and proof

Explain why the enumeration program does not by itself establish the optimum outside its grid.
Explain how section 6 closes that gap for this specific data.
Deliverable: a short paragraph.
Acceptance: distinguish a grid optimum from a continuous lower bound and identify the excluded C–West lane.
Time: 10 minutes.

### Exercise 7 — Design an operating model

Design a review process for a sourcing event with quarterly peaks and suppliers that share a tier-two factory.
Deliverable: a one-page data and approval checklist.
Acceptance: include time-phased capacity, shared exposure, supplier validation, feasibility checks, and decision rights.
Time: 20 minutes.

### Exercise 8 — Critique a business claim

A stakeholder says: “The solver saves USD 200 and guarantees resilience because we have two suppliers.”
Use the worked example to respond.
Deliverable: a concise decision memo with at least three corrections and one next action.
Acceptance: distinguish an infeasible benchmark, model output from commitment, and nominal dual sourcing from verified recourse.
Time: 15 minutes.

## 13. Self-check answers

### Answer 1

Minimize the sum of landed USD/unit times allocated units/year.
For each plant, sum supplier allocation to its demand; for each supplier, sum plant awards and bound by capacity times activation.
Set C–West to zero; bound each selected supplier's total below by its minimum and above by 500 units.
Use binary activation variables and nonnegative allocation variables measured in units/year.
Require at least two activations and ensure the activation upper bound prevents activation without volume.

### Answer 2

The award is feasible: East receives 400 + 200 = 600 and West receives 400 units/year.
Supplier totals are A 400, B 400, and C 200; each is at least its minimum and at most its cap and capacity.
C serves only East, as required.
Utilization of physical capacity is A 400/600 = 66.7%, B 400/500 = 80%, and C 200/400 = 50%.
Total cost is USD 4,000 + USD 3,600 + USD 2,600 = USD 10,200/year.

### Answer 3

The price-only plan uses A for 600 of 1,000 units, violating the 50% cap by 100 units.
Move 100 East units to B: A East 500, B East 100, B West 400.
This allocation costs USD 9,800/year and meets all restrictions.
Any feasible award must replace at least 100 East units otherwise served by A; the cheapest eligible replacement costs at least USD 2/unit more.
The lower bound is USD 9,800, and the feasible award attains it.

### Answer 4

The concentration cap limits A to 500; B and C are each limited to 200 by capacity.
Their combined maximum is 900 units/year, leaving at least 100 units/year uncovered.
If an authorized exception raises A's cap to 600 and its physical capacity is confirmed, allocate A 200 East and 400 West, B 200 East, C 200 East.
The repaired allocation covers both plants and costs USD 11,400/year; it does not satisfy the original policy.

### Answer 5

The maximum is 900 units/year: B supplies 400 West and 100 East, C supplies 400 East, and East has a 100-unit shortage.
Delivered purchase cost is USD 3,600 + USD 1,200 + USD 5,200 = USD 10,000/year.
The hypothetical shortage consequence is USD 5,000, for USD 15,000/year under the stated simplified scope.
The answer assumes post-event policy waivers and instantaneous usable capacity; it is not a prediction.

### Answer 6

The program considers only quantities divisible by 100, so a cheaper off-grid plan is not excluded by enumeration alone.
The independent analytical bound shows every feasible continuous plan costs at least USD 9,800, and the grid solution reaches that bound.
C–West is absent from the price and allocation dictionaries because the lane is not qualified.

### Answer 7 rubric

A strong checklist identifies quarterly demand and line-level capacity, common tier-two dependencies, and validated spare capacity.
It identifies engineering, procurement, operations, finance, and legal owners and keeps evidence of their approvals.
It specifies how conflicts and infeasibility are escalated without silently relaxing mandatory rules.
It includes post-award monitoring and triggers for re-planning.

### Answer 8 rubric

A strong memo notes USD 9,600 is an infeasible price-only benchmark, so USD 200 is not savings.
It distinguishes a mathematical recommendation from accepted supplier commitments and actual purchase orders.
It notes the A-outage stress case leaves 100 units short despite having two base-award suppliers.
A useful next action is to validate qualified, time-phased emergency capacity and its contractual availability.

## 14. Interview questions

| Question | Strong answer contains | Weak answer reveals |
|---|---|---|
| Why optimize the whole award? | Shared capacity and cross-lane constraints | Assumes cheapest quote is always feasible |
| How do you encode a minimum? | Binary activation plus lower and upper links | Forces all bidders to receive the minimum |
| How do you diagnose infeasibility? | Checks data, arithmetic, IIS, and authorized relaxations | Deletes constraints until a result appears |
| What does a supplier share cap mean? | Explicit numerator, denominator, period, and group | Treats all share definitions as equivalent |
| Is two-sourcing resilience? | Qualification, common exposure, spare capacity, and ramp | Assumes supplier count ensures recovery |
| What does optimal mean? | Best feasible value under the modeled assumptions and solver status | Claims real-world global superiority |
| How do you compare scenarios? | Common scope, recourse rules, cost and shortage | Mixes disruption and base assumptions |
| Can a time-limited incumbent be used? | Conditional decision with bound, gap, and feasibility checks | Calls any feasible solution proven optimal |
| How are bids normalized? | Currency, unit, freight, scope, validity, and evidence | Compares raw prices without context |
| How do you handle an expensive policy? | Show incremental cost and obtain authorized policy decision | Quietly removes the policy |

## 15. Key terms

- [Activation](#4-definitions-and-vocabulary): binary indicator for selection.
- [Allocation](#4-definitions-and-vocabulary): assigned quantity by supplier and plant.
- [Award](#4-definitions-and-vocabulary): proposed or approved allocation.
- [Baseline](#4-definitions-and-vocabulary): specified comparison plan.
- [Bid](#4-definitions-and-vocabulary): supplier offer with terms.
- [Capacity](#4-definitions-and-vocabulary): available output in a period.
- [Concentration](#4-definitions-and-vocabulary): share of a stated denominator.
- [Decision variable](#4-definitions-and-vocabulary): quantity selected by the model.
- [Demand](#4-definitions-and-vocabulary): quantity to cover in the horizon.
- [Eligibility](#4-definitions-and-vocabulary): permission to use a supplier lane.
- [Feasible](#4-definitions-and-vocabulary): satisfies all hard constraints.
- [Hard constraint](#4-definitions-and-vocabulary): rule that cannot be violated in the model.
- [Infeasible](#4-definitions-and-vocabulary): no allocation satisfies all model rules.
- [Landed cost](#4-definitions-and-vocabulary): comparable specified acquisition cost.
- [Minimum award](#4-definitions-and-vocabulary): minimum quantity conditional on selection.
- [Mixed-integer linear program](#4-definitions-and-vocabulary): linear model with integer decisions.
- [Optimal](#4-definitions-and-vocabulary): best objective among feasible decisions.
- [Optimality gap](#4-definitions-and-vocabulary): difference between incumbent and bound.
- [Recourse](#4-definitions-and-vocabulary): action allowed after a scenario occurs.
- [Scenario](#4-definitions-and-vocabulary): coherent set of assumptions.
- [Shadow price](#4-definitions-and-vocabulary): local continuous-model marginal value.
- [Shortage](#4-definitions-and-vocabulary): demand not covered by deliveries.
- [Soft constraint](#4-definitions-and-vocabulary): preference with permitted violation.
- [Total cost of ownership](#4-definitions-and-vocabulary): broader lifecycle economics.

## 16. Sources

- Caplice, C. and Sheffi, Y. (2003). “Optimization-Based Procurement for Transportation Services.” *Journal of Business Logistics*, 24(2), 109–128. [DOI: 10.1002/j.2158-1592.2003.tb00047.x](https://doi.org/10.1002/j.2158-1592.2003.tb00047.x). Research on transportation procurement; context differs from component manufacturing.
- GS1, [GTIN identification](https://www.gs1.org/standards/id-keys/gtin) and [GLN identification](https://www.gs1.org/standards/id-keys/gln). Official identifier descriptions; identifiers do not establish qualification.
- International Chamber of Commerce, [Incoterms rules](https://iccwbo.org/business-solutions/incoterms-rules/). Official overview; contract interpretation requires the applicable rule and terms.
- Acquisition.gov, [Federal Acquisition Regulation Part 15](https://www.acquisition.gov/far/part-15). Official US federal acquisition text; applicability depends on the procurement.
- Gurobi Optimization, [Infeasibility analysis](https://docs.gurobi.com/projects/optimizer/en/current/features/infeasibility.html). Solver documentation on IIS and feasibility relaxation; tools and implementation vary by solver.
- Python Software Foundation, [`itertools.product`](https://docs.python.org/3/library/itertools.html#itertools.product). Standard-library reference for the executable enumeration.

The case, costs, capacities, and probabilities in this lesson are illustrative assumptions, not claims about a real company.
Consult the current source and qualified professional advice before applying a legal or contractual rule.

## 17. Links

- [Module 14 topic index](_index.md).
- [Source-to-pay](01_source-to-pay.md).
- [Spend cube construction](02_spend-cube.md).
- [Classification taxonomies](03_classification-taxonomies.md).
- [Module 14 survey and lab sections](../../../course/14_procurement_supplier_risk_spend.md).
- [Lane and tender analytics](../12_transportation_routing_last_mile/03_lane-tender-analytics.md).
