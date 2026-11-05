# Order Routing and Distributed Order Management

> Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise  
> Topic 03 of 10  
> Estimated study time: 4 hours  
> Prerequisites: inventory visibility and available-to-promise  
> Version 2026.5

## 1. Why this topic exists

A customer has placed an order.
The retailer must decide which location will fulfill each line, by which service, and when.
That choice consumes inventory and labor that cannot simultaneously serve other demand.
It can also make the customer-facing delivery promise easier or harder to keep.

The nearest store is not necessarily the cheapest source.
It may lack a saleable unit, have no pick capacity, or miss the carrier cutoff.
A distribution center (DC) can be farther away yet offer a more reliable service and lower handling cost.
A store with a unit may be attractive today but expensive to drain before a local promotion.
The cheapest individual shipment may also use the only unit needed by a more urgent order.

Distributed order management (DOM) coordinates these trade-offs across available sources and channels.
It takes a decision snapshot, applies eligibility and business rules, ranks or optimizes feasible choices, and attempts to reserve the selected resources.
It does not manufacture accurate inventory or guarantee carrier performance.
The execution systems must accept the work and report what actually happened.

A routing decision affects several owners:

- The customer-experience owner defines the promise and acceptable exceptions.
- The inventory owner defines saleability, protection, and reservation policy.
- Store and DC operations set capacity and cutoffs they can actually meet.
- Transportation supplies service eligibility, rates, and delivery performance.
- Finance agrees which costs are incremental and how trade-offs are valued.
- The DOM product owner owns rules, configuration, and incident response.
- Analytics measures realized results against the decision snapshot.

Without a common decision contract, teams can each report a local success while the customer waits.
A carrier quote can be cheap because the store pick failed and the parcel was never tendered.
A store can report high fulfillment utilization by accepting more work than it can complete.
A margin report can count an existing fixed facility cost as a new cost for every routed order.

This topic treats routing as a constrained choice, not as a search for a universally best location.
You will see a rule-based baseline, a batch optimization formulation, and an executable small-instance enumerator.
All monetary examples use United States dollars (USD), and the illustrative data are invented for teaching.
They are not a company performance claim.

The practical question is not only “What did the algorithm choose?”
You must also be able to answer:

1. Which alternatives were considered and excluded?
2. What information was known when the decision was made?
3. Which policy and cost version governed the result?
4. Was inventory actually reserved and work accepted?
5. Did the customer receive the entire order by the committed time?
6. Would a different policy have improved the outcome without creating another violation?

Routing is a recurring operational decision.
Capacity changes during the day, orders arrive asynchronously, and inventory records can be wrong.
An answer that is optimal for a stale snapshot is not necessarily an executable assignment.
The implementation must therefore combine decision logic with reservation, failure handling, and measurement.

## 2. Learning outcomes

After studying this topic, you can:

1. **Draw** the flow from promise and inventory snapshot to reservation, execution, and feedback.
   Exercise 1 checks the information flow and ownership.
2. **Classify** a routing condition as a hard eligibility constraint or a priced preference.
   Exercise 2 tests feasibility before cost ranking.
3. **Calculate** incremental order cost and compare it with an allocated accounting cost.
   Exercise 3 tests the relevant-cost calculation.
4. **Calculate** the reliability of a route and an order under stated dependence assumptions.
   Exercise 4 checks both the calculation and its limits.
5. **Formulate** a capacity-constrained batch assignment and explain why greedy routing can lose.
   Exercise 5 requires a feasible assignment and cost comparison.
6. **Diagnose** a reservation race, failed pick, or missed cutoff and select a controlled response.
   Exercise 6 requires state transitions and customer treatment.
7. **Design** an evaluation and rollout that distinguish predicted costs from observed outcomes.
   Exercise 7 assesses experiment design and guardrails.
8. **Critique** an objective that double counts risk or sacrifices a protected commitment.
   Exercise 8 asks for a corrected objective and policy.

You do not need a commercial optimizer to follow the worked arithmetic.
You should be comfortable with basic probability, integer decisions, and spreadsheet formulas.
The code uses the Python standard library only.

## 3. The core mental model

Think of routing as a controlled allocation of scarce, perishable decision options.
The scarce options include not just physical units, but also time, capacity, carrier service, and a customer's tolerance for delay.
A location is a candidate only when it can legally and operationally perform the requested fulfillment.
Among candidates, the policy chooses a trade-off and then attempts to secure it.

```text
Customer order + immutable promise
                |
                v
Timestamped inventory, capacity, service and policy snapshot
                |
                v
Generate source / service / split candidates
                |
                v
HARD FILTER: eligible SKU, stock, cutoff, capacity, commitment
                |
                v
SCORE or OPTIMIZE: incremental cost + explicitly valued risk
                |
                v
Atomic reservation / conditional acceptance
          | success                 | conflict or rejection
          v                         v
Release work to node          refresh + bounded re-decision
          |                         |
          v                         +--> escalate if none feasible
Pick -> pack -> tender -> deliver
          |
          v
Immutable events -> outcome attribution -> policy review
```

The diagram separates **feasibility** from **preference**.
A restricted item cannot become eligible because a model assigns it a low cost.
A feasible source can still be unattractive because using it is likely to harm another order.
A firm delivery commitment may be modeled as a hard condition using an agreed reliability standard.
Alternatively, a business may permit a trade-off at a disclosed service level.
Those are different policies and require different approval.

### Decisions, predictions, and commitments

A forecast estimates an uncertain future, such as a 97% chance of on-time delivery.
A plan is an intended allocation, such as sending a unit from store S.
A target is an internal aspiration, such as 98% on-time delivery across a segment.
A commitment is what the customer was actually promised, such as delivery by Friday at 20:00 local time.
A predicted arrival date is not automatically a commitment.
A target is not evidence that any individual route meets that target.

Once a promise has been presented and accepted, the routing system should preserve it in its decision record.
Later recalculations may change the intended node or forecast, but must not silently rewrite the original promise.
A changed commitment needs the applicable customer communication and approval policy.

### Local and network views

A local rule chooses the lowest-cost feasible source for the current order.
A network view asks what using that source does to other orders and future demand.
For example, consuming the last local unit may cause a later urgent order to ship expensively from a DC.
An explicit opportunity cost approximates that future harm; a batch model can represent some competing orders directly.
Do not count the same future order both explicitly in a batch and again in an unadjusted opportunity-cost estimate.

The best theoretical assignment is not useful if it cannot be committed before the checkout or release deadline.
A fast feasible heuristic with a recorded optimality gap may be better operationally than an exact answer arriving after cutoff.
Conversely, an unexamined greedy rule can repeatedly waste scarce inventory.
The method depends on decision frequency, state quality, and the consequence of a mistake.

### Feedback is part of the system

A decision is not complete when a score is emitted.
The node must acknowledge work, the pick must find the item, and the carrier must receive the parcel.
Each event can expose a mismatch between the snapshot and reality.
Learning requires joining the original candidate set and decision to those later events.
It also requires recognizing that outcomes are observed mainly for selected routes, not all rejected alternatives.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Distributed order management (DOM) | Coordination of order allocation and execution across nodes and channels | A synonym for a carrier label service |
| Order | Customer purchase with an identifier and one or more lines | Treating every line as a separate customer commitment |
| Order line | Requested stock-keeping unit (SKU), quantity, and applicable service terms | Assuming one line always means one physical unit |
| Unit | One individually allocated item, in eaches unless otherwise specified | Mixing eaches and cases |
| Node | Store, DC, supplier, or other source that can perform a specified fulfillment function | Assuming every inventory location can ship |
| Candidate | A possible allocation of lines, nodes, services, and timing | Treating an unvalidated candidate as executable |
| Allocation | Tentative or committed assignment of quantities to sources | Assuming a tentative plan is a reservation |
| Reservation | A recorded claim on stock or capacity under stated concurrency rules | Assuming a read of stock alone reserves it |
| Available-to-promise (ATP) | Quantity available for a defined promise after policy deductions, in units and by time | Equating ATP with physical shelf count |
| On-hand | Recorded quantity physically held, in units | Treating every on-hand unit as saleable |
| Safety stock | A planned inventory buffer, in units, for uncertainty | Adding the same buffer both in ATP and again as a hard deduction |
| Protected stock | Quantity withheld from a channel or use by explicit policy, in units | Assuming protection is always a legal requirement |
| Inventory accuracy | Agreement of inventory record with verified physical stock, measured on a defined basis | Applying a SKU-level accuracy rate blindly to every unit |
| Phantom stock | Recorded stock that cannot be found or used | Calling every short pick fraud |
| Capacity | Maximum executable workload for a node and time bucket, in labor minutes, orders, or another unit | Comparing minutes with order counts without conversion |
| Cutoff | Latest time for an action to make a service or operating window, with timezone | Using the carrier pickup time as the store's pick cutoff |
| Service | A carrier or delivery option with eligibility, rate, transit, and schedule | Assuming nominal transit days guarantee arrival |
| Promise | Customer-facing commitment, including deadline and timezone | Replacing the original promise with the latest estimate |
| Lead time | Elapsed time from defined start event to defined end event, in hours or days | Omitting pick and handoff time |
| Hard constraint | Condition that every accepted solution must satisfy | Making legal eligibility a small cost penalty |
| Soft preference | Condition that may be traded against another objective at an explicit price or priority | Treating an arbitrary penalty as a firm guarantee |
| Feasible | Satisfying all modeled constraints under the input snapshot | Concluding that stale modeled stock is physically present |
| Objective | Quantity minimized or maximized over feasible assignments | Confusing modeled objective with realized profit |
| Incremental cost | Change in relevant future cost caused by the decision, in USD per decision | Adding unchanged rent to every shipment |
| Sunk cost | Past expenditure unchanged by the current choice, in USD | Using it to justify a more expensive route |
| Opportunity cost | Value of the best foregone use of a scarce resource, in USD per unit or minute | Calling all gross revenue a lost margin |
| Split shipment | An order fulfilled in multiple shipments or from multiple nodes | Counting a per-order cost once per line |
| Fixed shipment charge | Cost incurred once when a shipment is opened, in USD per shipment | Charging the amount for each unit regardless of consolidation |
| Variable handling | Cost that changes with units, lines, or minutes, in USD per relevant driver | Assuming an average wage is the incremental wage in all staffing regimes |
| Expected cost | Probability-weighted cost under a specified distribution, in USD | Treating the expectation as a guaranteed realized charge |
| Reliability | Probability or observed rate of an explicitly defined successful event, dimensionless | Using carrier on-time rate as end-to-end order reliability |
| Service-level agreement (SLA) | A stated service obligation between parties, with definition and measurement | Assuming an internal target is a customer SLA |
| Chance constraint | A requirement that a defined event occur with at least a specified probability | Assuming a predicted probability is a guarantee |
| Calibration | Agreement between predicted probabilities and observed frequencies over a defined population | Claiming a single correct prediction proves calibration |
| Correlation | Dependence between outcomes, such as failures caused by a shared weather event | Multiplying marginal probabilities without an independence assumption |
| Idempotency | Repeating a request with the same key has the same intended effect | Issuing a second reservation under a new key after a timeout |
| Version | Identifier of the data or policy state used for a decision | Recording only the currently deployed rule |
| Reroute | A controlled reassignment after a rejection, failure, or material change | Silently changing a customer promise |
| Shadow price | Marginal value of relaxing a constraint in an appropriate optimization model, in USD per unit of resource | Assuming every integer model has a directly valid linear-program shadow price |
| Regret | Difference between a decision's outcome and a defined comparator, in USD or service units | Claiming unobserved counterfactual regret is directly measured |
| Order management system (OMS) | System maintaining order state and orchestrating associated processes | Assuming all DOM decisions originate in the OMS |
| Warehouse management system (WMS) | System managing warehouse execution and inventory operations | Treating its stock record as instantly synchronized everywhere |
| Point of sale (POS) | Store transaction system recording sales and associated events | Treating a delayed POS feed as real-time inventory truth |

These definitions apply to this topic's examples.
Actual enterprises must document their local definitions rather than assume identical vendor terminology.

## 5. The mechanics, step by step

### Step 1. Freeze the decision question

**Input:** order identifier, line quantities, delivery destination or pickup location, customer commitment, and decision deadline.

**Action:** state whether the decision is made at checkout, after payment, before release, or during exception recovery.
A checkout estimate can affect conversion, while post-order routing must preserve an already accepted promise.
Decide whether partial shipments and substitutions are permitted.
Record the promise shown to the customer, not just the latest calculated estimate.

**Output:** an immutable decision request with an as-of time and policy context.

**Branch:** if the commitment is missing or ambiguous, use the approved exception process rather than inventing a deadline.
If the destination is unavailable or invalid, do not log it as an inventory failure.

### Step 2. Build an as-of resource snapshot

**Input:** saleable stock and reservations by SKU and node, operational capacity, node calendars, carrier schedules, and service restrictions.

For each record, retain source system, effective time, ingestion time, and version.
Effective time says when the source claims the fact applied.
Ingestion time says when the router learned it.
These times can differ because of queues, outages, or late events.
A decision record should support reconstruction of what was actually known, not what a later correction says was true.

A basic same-day inventory calculation may be:

```text
available = saleable on-hand - committed reservations - policy protection
```

Use consistent units and avoid subtracting any component already removed in the provided ATP.
If inbound units are included, use their eligible arrival time rather than treating them as immediately pickable.
If the customer requires a specific lot or serial number, availability must be at the appropriate granularity.
An inventory service may expose only aggregate ATP, but that does not make unobserved lot restrictions disappear.

Capacity must use an agreed time bucket and workload unit.
A store with 120 labor minutes of planned capacity and 100 minutes already committed has 20 minutes of nominal residual capacity.
If four new picks each require 8 minutes, all four cannot fit under that limit.
A rate based on historical average processing does not by itself account for breaks, peak congestion, or unavailable staff.
Operations should own both the capacity feed and the escalation when that feed is stale.

**Output:** versioned, time-bucketed resource quantities with freshness status.

**Branch:** when freshness exceeds the approved threshold, either remove the affected node, use an explicitly approved conservative buffer, or escalate.
The choice is a business policy, not a universal number.
Record which degraded mode was used.

### Step 3. Enumerate plausible fulfillment patterns

**Input:** order lines, source capabilities, and eligible shipping or pickup methods.

For a single-unit order, a pattern can be one node and one service.
For a multi-line order, a pattern may use one node, several nodes, or multiple services.
A shipment is not identical to a line: several lines may share a parcel, and one line may require multiple parcels.
A pickup order usually requires the selected pickup location rather than an arbitrary source unless transfer is explicitly offered.

Prune impossible combinations early.
For example, a store that does not ship cannot be assigned a ship-from-store parcel.
An item that cannot be transported by a particular service must not be sent through that service.
A destination outside a service area is not a candidate for that service.
If an item requires a controlled handling condition, the source and every transport leg must support it.

Candidate generation is itself a modeling choice.
A solver cannot select an omitted feasible candidate.
If the business caps the candidate set at the five closest nodes, it should measure the effect of that cap against a broader search sample.
For large multi-line orders, exhaustive partitioning can be expensive; use an auditable heuristic or decomposition and record the restriction.

**Output:** a set of candidate plans with a clear definition of the resulting shipments.

**Branch:** if no candidate exists, distinguish lack of inventory, unsupported service, stale data, and failed capacity.
Do not use a generic “no stock” reason for all cases.

### Step 4. Apply hard constraints

For each candidate, check the conditions that cannot be traded away.
Typical constraints include:

1. Required quantity is covered exactly, unless partial fulfillment is authorized.
2. Allocated units do not exceed available stock for that SKU, node, and time.
3. Store or DC labor consumption fits the applicable capacity bucket.
4. Node and service can handle the product and destination.
5. The pick, pack, and tender schedule can reach the carrier before its actual cutoff.
6. The candidate respects the stated promise policy.
7. Channel, regional, or contractual restrictions are satisfied.
8. A pickup order is assigned to the specified location unless an agreed transfer is included.
9. The customer’s split and substitution preferences are honored.

A deterministic transit estimate before the deadline is not proof of reliable delivery.
A probability threshold can be a hard modeled constraint, but it remains conditional on a calibrated prediction.
A route with a 99% estimated probability can still be late; uncertainty is not eliminated by the threshold.

Hard constraints must have an owner and an override procedure.
Legal or product-safety restrictions are not ordinary operator overrides.
A capacity override might be possible if a named operations manager confirms additional staffing.
Record the authorizing person, duration, rationale, and resulting capacity version.

**Output:** feasible candidates and per-candidate exclusion reasons.

**Branch:** when no candidate survives, do not select the least-infeasible one as if it were valid.
Escalate, offer a revised commitment if permitted, or place the order in an exception state.
The applicable customer and legal policy determines whether cancellation, delay notice, or consent is required.

### Step 5. Estimate incremental execution cost

Consider future cash flows changed by the choice, not all accounting allocations.
For a shipment candidate, possible terms include:

- Carrier charge, including applicable service, zone, weight, and parcel surcharges.
- Packaging used by that shipment.
- Incremental pick and pack labor, using an operationally defensible marginal rate.
- Per-shipment label, handling, or third-party fee.
- Expected short-pick and recovery cost when modeled separately from reliability.
- Incremental cost of a split, including a second parcel and any additional contact or handling.
- Opportunity cost of consuming a scarce unit or capacity minute.
- An explicitly authorized expected delay cost or penalty if delay is not forbidden.

A labor minute is not always a cash expense avoided by not routing a single order.
If the employee is already paid and cannot be sent home, the immediate incremental cash cost may be near zero.
However, scarce labor can still have an opportunity cost if another task is displaced or overtime becomes necessary.
Use the cost perspective and time horizon agreed with finance and operations.

Likewise, warehouse rent generally does not change because one order is routed elsewhere today.
Facility closure or expansion costs can matter for a long-horizon network decision, but should not be charged as if each individual parcel caused them.
For each term, document the cost driver, source, effective date, and whether it is already embedded in another rate.

**Output:** a cost breakdown for each feasible candidate, in USD per order or shipment as specified.

**Branch:** if a cost input is missing, follow a named conservative fallback or exception policy and label the estimate.
A missing rate should not silently become USD 0.

### Step 6. Value inventory and capacity risk

A unit retained at a store may satisfy later local demand.
A simple one-period opportunity-cost estimate is:

```text
opportunity cost per unit = probability of displaced future sale
                            × contribution lost if displaced
```

This approximation assumes the event and loss amount are defined for the same horizon.
If the later customer can be served from another node, use the expected incremental recovery cost or genuinely lost contribution, not the full selling price by default.
If substitution or replenishment is possible, account for it in the displacement probability or loss.

For example, a 0.25 probability of a displaced sale with USD 16 of unrecoverable contribution gives USD 4 expected opportunity cost.
This is a forecast-based valuation, not an observed USD 4 invoice.
If the expected loss already includes the cost of an emergency transfer, do not add that same transfer again.
If the current order would have used a unit needed for an explicitly modeled future order in the same batch, do not add a separate overlapping forecast penalty.

Protection can be modeled in two distinct ways.
A hard reserve prohibits use below a threshold.
An opportunity-cost term permits use but prices the expected consequence.
Using both may be appropriate only when they address different risks or the threshold represents a non-negotiable policy.
Do not claim that an arbitrary penalty enforces a protected minimum.

Capacity scarcity can be valued similarly, but the value depends on the workload and remaining shift.
If another task has a contractual deadline, it might require a hard capacity reservation instead of a price.
In a linear relaxation, a shadow price can indicate the modeled value of one more capacity unit locally.
For integer or nonlinear models, validate such interpretations using explicit re-optimization and finite differences.

**Output:** documented risk and opportunity-cost assumptions, with a non-overlap check.

**Branch:** if estimated risk is unstable, use scenario analysis and human-approved guardrails rather than pretending the estimate is precise.

### Step 7. Model promise reliability

Define success first.
For this topic, an order is on time only if every required unit is delivered no later than the original promised deadline.
Other organizations may use first attempt, pickup readiness, or shipment-level delivery; those are different metrics.

For a single route, end-to-end success can be decomposed into pick, tender, and delivery stages.
The exact chain rule is:

```text
P(on time) = P(pick succeeds)
           × P(tender on time | pick succeeds)
           × P(delivery on time | pick and tender succeed)
```

No independence assumption is required when these conditional probabilities are genuinely estimated as stated.
Multiplying three unconditional rates is valid only under an appropriate independence assumption.
If the available data contain only marginal rates, identify the limitation and use empirical end-to-end outcomes where possible.

For a split order, multiplying shipment success probabilities requires independence of the shipment outcomes.
Shared weather, a common carrier hub, or the same node can create dependence.
Use historical joint outcomes, scenarios, or a conservative bound when dependence matters.
For events A and B, the union bound gives:

```text
P(A and B) >= max(0, P(A) + P(B) - 1)
```

This bound does not require independence but can be conservative.
A 98% and a 97% marginal success rate imply at least 95% joint success by this bound, not necessarily 95.06% or 97%.
The actual joint probability depends on the dependence structure.

Set a threshold only after deciding the denominator and segment.
A threshold on predicted probability is not equivalent to an observed service-level guarantee.
Monitor calibration by predicted-risk bucket, location, carrier, day, and product class where sample size permits.
Account for censored orders still in transit and changes in the mix of selected routes.

**Output:** route or order reliability estimates, uncertainty, and the policy used to accept them.

**Branch:** if the estimate is insufficiently supported, use an approved conservative rule or escalate.
Do not present a model score as a fact about the future.

### Step 8. Choose the policy: rules or optimization

A rule engine can perform the following transparent sequence:

1. Remove ineligible candidates.
2. Remove candidates that fail the promised service policy.
3. Choose the lowest estimated incremental cost.
4. Break exact ties by a stable, documented rule.
5. Attempt reservation.
6. Recompute if reservation fails and the decision is still valid.

This is a sensible baseline for independent, small decisions.
It is easy to inspect and fast to execute.
It can fail when multiple orders compete for scarce resources, when split fixed charges matter, or when later orders have different alternatives.
A static priority list can also hide changing carrier rates and capacity.

A batch optimization model can represent those interactions explicitly.
Let $O$ be orders and $P_o$ the enumerated feasible plans for order $o$.
Let $x_{op}$ be 1 if plan $p$ is chosen for order $o$, and 0 otherwise.
Let $c_{op}$ be its total incremental cost in USD, including only non-overlapping risk terms.
Let $a_{opr}$ be the amount of resource $r$ consumed by that plan and $b_r$ the resource available in matching units.

```text
minimize    sum over o,p of c[o,p] * x[o,p]
subject to  sum over p in P[o] of x[o,p] = 1       for every order o
            sum over o,p of a[o,p,r] * x[o,p] <= b[r]  for every resource r
            x[o,p] in {0, 1}
```

Here a resource can be SKU-node stock in units, labor minutes in a time bucket, or another explicitly measured limit.
Each plan must already obey its item, destination, cutoff, and promise constraints.
If no full assignment exists, the model is infeasible.
That is information for an exception process, not permission to claim all orders were fulfilled.
If deferral or cancellation is explicitly allowed, represent it as a separate outcome with an approved priority and customer treatment.

Plan enumeration can embed shipment fixed charges in $c_{op}$.
An alternative line-level mixed-integer formulation uses binary shipment-open variables and linking constraints.
The latter can avoid enumerating every complete plan but needs careful modeling of parcel consolidation and service combinations.
For either model, objective coefficients must match the chosen planning horizon and avoid double counting.

A weighted objective is not automatically equivalent to a strict priority order.
If preserving the promise must always outrank cost, filter unacceptable plans or use a genuine lexicographic optimization.
Do not rely on a large but arbitrary cost penalty unless it is proven sufficient over all feasible costs and approved as policy.
For permitted trade-offs, document the penalty in USD per event or minute and how it was derived.

**Output:** selected plan, objective breakdown, solver status, candidate restrictions, and decision version.

**Branch:** if optimization times out, use a known feasible incumbent only if all required constraints were independently checked by the serving system.
If no feasible incumbent exists, use the approved fallback or exception queue.
A timeout does not prove infeasibility.

### Step 9. Reserve and release safely

A chosen plan is tentative until the system has secured the resources it needs.
Two routers can read the same last unit at the same time.
Without concurrency control, both can assign it even though each snapshot looked valid.

Use a conditional inventory reservation or atomic decrement against an inventory version or available balance.
If labor capacity is reserved separately, define an ordered protocol and compensating release when a later reservation fails.
A distributed transaction may not be available across inventory, labor, and external systems.
The resulting workflow must explicitly handle partial success and recovery.

Give the allocation attempt an idempotency key tied to the order and decision attempt.
A network timeout is not proof the reservation failed.
Query by the same key or retry idempotently rather than issuing a fresh unrelated reservation.
Specify expiry, renewal, cancellation, and reconciliation rules for reservations.
A lease that expires before a slow node accepts work can cause double assignment unless acceptance and expiry are coordinated.

After successful reservation, send the work to the execution system and require an acknowledgement within a defined window.
Log rejection reasons, not merely a generic failure.
On rejection or short pick, release or reconcile resources before a bounded re-decision.
Do not endlessly bounce an order among nodes.
The original promise remains visible throughout recovery.

**Output:** a committed assignment or a clearly classified exception.

**Branch:** if the system cannot establish whether reservation succeeded, enter an uncertain state and reconcile before creating a duplicate commitment.
If the remaining choices cannot meet the original promise, use the approved delay-notice and customer-remedy process.

### Step 10. Observe, reconcile, and improve

Capture the decision request, snapshot identifiers, candidate list, exclusion reasons, cost components, selected plan, and reservation outcome.
Join these to acceptance, pick, pack, tender, delivery, cancellation, return, and customer-contact events where applicable.
Use immutable event times and explicit definitions of success.
Record corrections as new events instead of overwriting the customer-facing promise.

At a minimum, monitor:

- Order-level on-time rate against the original commitment.
- Line-level and shipment-level on-time rates separately.
- Short-pick rate by node and inventory age.
- Reservation conflict and timeout rates.
- Node rejection rate and reason.
- Cost per delivered order, including recovery and split costs.
- Split rate and parcels per order.
- Estimated versus realized carrier and labor costs.
- Predicted reliability calibration and coverage.
- Exception age, backlog, and customer notification timeliness.

Compare a new policy against a defined baseline on the same population and time horizon.
A lower quoted shipping cost can be offset by higher recovery cost or late delivery.
A higher on-time rate can reflect a later promise rather than better execution.
Track commitment duration and customer-facing changes alongside on-time performance.

**Output:** an auditable decision-to-outcome record and a review cadence with named owners.

**Branch:** if measured performance violates guardrails, rollback to an approved policy and investigate rather than silently retuning thresholds.

## 6. Worked example

### Scenario and commitments

At 10:00 local time on Monday, a retailer has two paid one-unit orders, A and B.
Both customers were promised delivery by 20:00 local time on Tuesday.
The promise is a commitment; predicted on-time probabilities below are forecasts.
The internal objective is to minimize modeled incremental cost while accepting only routes with at least 96% predicted end-to-end on-time probability.
That threshold is an illustrative business policy, not a universal industry standard.

There are two sources:

- Store S has one saleable unit and 12 pick minutes available for the relevant bucket.
- DC D has two saleable units and 30 pick minutes available for the relevant bucket.

Both orders request the same SKU and one unit each.
Each store pick uses 8 minutes; each DC pick uses 10 minutes.
All stated stock and capacity are already net of prior reservations.
There are no other modeled orders in this batch.
All prices are USD per order; no sales tax or revenue changes by route.

The destination and service data yield the following candidate estimates:

| Order | Source | Carrier | Pick and pack | Packaging | Inventory opportunity | Total | Predicted on time |
|---|---|---:|---:|---:|---:|---:|---:|
| A | S | $4.40 | $2.10 | $0.50 | $1.00 | $8.00 | 97% |
| A | D | $6.20 | $1.70 | $0.60 | $0.00 | $8.50 | 99% |
| B | S | $4.00 | $2.10 | $0.50 | $1.00 | $7.60 | 98% |
| B | D | $9.80 | $1.70 | $0.60 | $0.00 | $12.10 | 97% |

All four candidates meet the 96% forecast threshold.
Assume the opportunity cost describes future demand outside these two orders and does not overlap the modeled batch.
Assume the labor values are genuinely incremental or opportunity-valued under the retailer's stated cost perspective.
The values are estimates, not invoices and not observed delivery outcomes.

### Candidate arithmetic

For A from S:

```text
$4.40 + $2.10 + $0.50 + $1.00 = $8.00
```

For A from D:

```text
$6.20 + $1.70 + $0.60 + $0.00 = $8.50
```

For B from S:

```text
$4.00 + $2.10 + $0.50 + $1.00 = $7.60
```

For B from D:

```text
$9.80 + $1.70 + $0.60 + $0.00 = $12.10
```

A's local saving from S is USD 0.50 compared with D.
B's local saving from S is USD 4.50 compared with D.
The store has only one unit and only enough pick minutes for one of these orders.
The store stock and labor constraints independently forbid assigning both orders there.

### Compare complete assignments

| Assignment | Store units / limit | Store minutes / limit | DC units / limit | DC minutes / limit | Cost | Feasible? |
|---|---:|---:|---:|---:|---:|---|
| A=S, B=S | 2 / 1 | 16 / 12 | 0 / 2 | 0 / 30 | $15.60 | No: store stock and labor |
| A=S, B=D | 1 / 1 | 8 / 12 | 1 / 2 | 10 / 30 | $20.10 | Yes |
| A=D, B=S | 1 / 1 | 8 / 12 | 1 / 2 | 10 / 30 | $16.10 | Yes |
| A=D, B=D | 0 / 1 | 0 / 12 | 2 / 2 | 20 / 30 | $20.60 | Yes |

The minimum-cost feasible batch plan is A from D and B from S, for USD 16.10.
If a greedy router processes A first, it chooses S for A because USD 8.00 is less than USD 8.50.
B is then forced to D, giving USD 20.10.
The batch assignment saves USD 4.00 relative to this order-dependent greedy result.
It does so by using the scarce store unit where it avoids the larger alternative cost.

This saving is a model comparison, not proof of a realized operational benefit.
It assumes that both orders are known before the decision deadline, the candidate forecasts are valid, and both reservations succeed.
If order B had not yet arrived, a router would need a forecast-based opportunity cost or an explicit waiting policy rather than knowledge of the future.
Waiting can itself jeopardize cutoff and customer service.

### Reliability interpretation

The chosen route for A has a predicted 99% on-time probability and B has 98%.
Each exceeds the illustrative per-order 96% threshold.
That does not mean the pair is guaranteed to arrive on time.
If their outcomes were independent, the probability both are on time would be:

```text
0.99 × 0.98 = 0.9702 = 97.02%
```

The assumption is illustrative only.
If both parcels share a weather-affected region or a carrier hub, outcomes can be dependent.
With only the marginal probabilities, the union-bound lower limit is:

```text
max(0, 0.99 + 0.98 - 1) = 0.97 = 97%
```

The probability both succeed could be higher than that lower bound.
This pair-level calculation is not the same as the per-order SLA and should not replace it.

### Reservation and operational handoff

The system first attempts to reserve one unit at D for A and one unit at S for B.
It also reserves 10 DC minutes and 8 store minutes if the capacity system supports binding reservations.
The order assignments should be associated with the same batch decision version.
If S rejects B's reservation because a POS sale consumed the unit, the original batch solution is stale.
The router refreshes availability and may select A=D, B=D if that remains feasible.
The modeled cost becomes USD 20.60, and both route probabilities still exceed 96% under the original assumptions.
If the DC's remaining stock or capacity is no longer sufficient, the system must escalate instead of inventing availability.

The customer commitment remains Tuesday at 20:00 local time during rerouting.
The transport and operations teams should confirm the refreshed plan still meets the applicable cutoff.
The customer-experience owner controls any changed promise and communication, not the cost optimizer.
The eventual delivery event is the evidence for observed success; the 99% and 98% values were predictions.

### Sensitivity to the reliability threshold

Suppose the policy changes from 96% to 98%, inclusive, with all other inputs unchanged.
A=S at 97% and B=D at 97% are excluded.
A=D at 99% and B=S at 98% remain feasible, so the same USD 16.10 assignment survives.
At a 99% threshold, B has no eligible candidate in this table.
The correct model result is infeasible for the full batch, not a USD 16.10 solution.
Any revised promise or exception requires a separate business decision.

### Sensitivity to store opportunity cost

Suppose S's opportunity cost increases from USD 1.00 to USD 6.00 per unit for both orders.
Then A=S costs USD 13.00 and B=S costs USD 12.60.
The three feasible complete plans cost USD 25.10, USD 21.10, and USD 20.60 respectively.
The model now chooses A=D and B=D at USD 20.60.
This assumes the revised risk estimate refers to external future demand and is not counted elsewhere.
A hard reserve of the only store unit would instead make both store candidates infeasible, regardless of their cost.

## 7. Code, query, or calculation

The following standalone Python program enumerates the four plans in the worked example.
It filters candidate routes using the illustrative 96% per-order threshold, checks stock and labor, and chooses the lowest-cost feasible batch assignment.
The deterministic tie-break is the tuple of source names in order A, then B.
It deliberately models a small static batch, not reservation concurrency or a production solver.

Run with Python 3.9 or later, for example `python routing_example.py`.
No third-party packages or network access are required.
All costs use integer cents to avoid binary floating-point money arithmetic.
Probabilities are integer basis points: 9,700 means 97.00%.
The code assumes each candidate supplies one unit and a whole order.

```python
from itertools import product

ORDERS = ("A", "B")
SOURCES = ("D", "S")
MIN_ON_TIME_BP = 9_600

# Capacity is the residual amount after existing commitments.
stock_units = {"S": 1, "D": 2}
labor_minutes = {"S": 12, "D": 30}

# Cost columns: carrier, labor, packaging, opportunity, all in cents.
# Reliability is predicted end-to-end on-time probability in basis points.
routes = {
    ("A", "S"): {
        "costs": (440, 210, 50, 100),
        "minutes": 8,
        "on_time_bp": 9_700,
    },
    ("A", "D"): {
        "costs": (620, 170, 60, 0),
        "minutes": 10,
        "on_time_bp": 9_900,
    },
    ("B", "S"): {
        "costs": (400, 210, 50, 100),
        "minutes": 8,
        "on_time_bp": 9_800,
    },
    ("B", "D"): {
        "costs": (980, 170, 60, 0),
        "minutes": 10,
        "on_time_bp": 9_700,
    },
}


def evaluate(sources):
    """Return total cents if this complete assignment is feasible, else None."""
    used_stock = {source: 0 for source in SOURCES}
    used_minutes = {source: 0 for source in SOURCES}
    total_cents = 0
    for order, source in zip(ORDERS, sources):
        route = routes[(order, source)]
        if route["on_time_bp"] < MIN_ON_TIME_BP:
            return None
        used_stock[source] += 1
        used_minutes[source] += route["minutes"]
        total_cents += sum(route["costs"])
    for source in SOURCES:
        if used_stock[source] > stock_units[source]:
            return None
        if used_minutes[source] > labor_minutes[source]:
            return None
    return total_cents


def dollars(cents):
    return f"${cents // 100}.{cents % 100:02d}"


feasible = []
for sources in product(SOURCES, repeat=len(ORDERS)):
    cost = evaluate(sources)
    assignment = ", ".join(
        f"{order}={source}" for order, source in zip(ORDERS, sources)
    )
    if cost is None:
        print(f"{assignment}: infeasible")
    else:
        print(f"{assignment}: {dollars(cost)}")
        feasible.append((cost, sources))

if not feasible:
    print("No feasible assignment; escalate rather than allocate.")
else:
    best_cost, best_sources = min(feasible)
    best_assignment = ", ".join(
        f"{order}={source}" for order, source in zip(ORDERS, best_sources)
    )
    print(f"Best: {best_assignment}; {dollars(best_cost)}")
```

Expected output:

```text
A=D, B=D: $20.60
A=D, B=S: $16.10
A=S, B=D: $20.10
A=S, B=S: infeasible
Best: A=D, B=S; $16.10
```

The program evaluates capacity across both orders rather than checking each order in isolation.
It does not multiply the two reliability predictions or impose a pair-level chance constraint.
It uses the stated per-order threshold and treats the probabilities as input forecasts.
A production implementation would need timestamped inputs, richer candidate generation, secure interfaces, reservation semantics, and operational monitoring.
A larger optimization service should also report solve status, timeout, feasible incumbent, and model version.

## 8. Data, standards, and regulatory context

### Decision data contract

A routing request needs an order identifier, line identifiers, SKU identifiers, requested quantities, destination or pickup location, and immutable promise.
It also needs an order creation time, decision time, deadline timezone, and policy version.
A destination should be represented at the minimum precision required for routing and retained according to the organization's privacy policy.
Do not put a full address in operational logs merely because a route is difficult to debug.

The resource snapshot needs stock by node and SKU, unit of measure, reservations, protected quantity, source timestamp, and freshness state.
The capacity feed needs node, time bucket, residual workload, unit, and owner.
The transportation feed needs service eligibility, origin and destination coverage, cutoff, operating calendar, price, and expected transit or probability estimate.
Rates should identify effective date and any weight, dimension, fuel, remote-area, or residential surcharges included.

Separate an event's occurrence time from processing time.
An item sold at 09:59 but ingested at 10:03 can cause a 10:00 router to see an outdated stock position.
A reconstructed historical snapshot should reproduce the 10:00 information set rather than leak the 10:03 correction into the earlier decision.
Late events must be reconciled without erasing the original record.

### Identifiers and interoperable messages

GS1 identifies trade items and locations using identifiers such as the Global Trade Item Number (GTIN) and Global Location Number (GLN).
Those identifiers help parties refer consistently to products and locations.
They do not tell a retailer how much saleable inventory is actually available or which source should win an allocation.
The relevant GS1 specifications and partner agreements determine valid structure and usage.

Electronic Product Code Information Services (EPCIS) provides a standard way to share visibility event data about what, when, where, why, and how.
A standardized event does not by itself establish real-time accuracy, make a physical reservation, or encode a customer's commercial promise.
A routing implementation must still specify event semantics, latency, duplication handling, and ownership.

Electronic data interchange (EDI) and application programming interfaces (APIs) can carry order, inventory, shipping, and status messages.
The specific message standard, version, and trading-partner implementation guide matter.
A schema-valid message may still be operationally late or semantically wrong.
Reconciliation and acknowledgements are therefore part of the control design.

### Applicable obligations are contextual

As of November 2026, the United States Federal Trade Commission's Mail, Internet, or Telephone Order Merchandise Rule addresses, among other things, shipment timing representations and delay handling for covered orders.
Applicability, exceptions, required notices, consent, and remedies depend on the facts and the current rule.
Consult the official rule and counsel rather than treating the model's predicted arrival as legal compliance.
The decision record should retain what was represented to the customer and the subsequent relevant events.

Privacy and product-transport obligations depend on jurisdiction, product, carrier, and transaction.
A routing service can process personal information such as names and addresses; access controls, retention limits, and lawful handling must be determined for the deployment.
A regulated or hazardous item may require carrier-specific and legal checks beyond a generic product category flag.
This topic does not provide legal advice or assert that one standard applies to every order.

### Data quality controls

| Control | Evidence | Decision if control fails |
|---|---|---|
| Inventory freshness | Source event age and ingestion lag by node | Apply approved degraded mode or exclude node |
| Unit consistency | SKU conversion and order-unit validation | Reject incompatible candidate quantities |
| Reservation consistency | Balance reconciliation and conflict count | Pause affected allocation path and reconcile |
| Capacity currency | Operations acknowledgement and bucket version | Use approved fallback capacity or escalate |
| Cutoff correctness | Calendar, timezone, holiday, and pickup agreement | Reject uncertain service or manually confirm |
| Rate completeness | Contract version and surcharge coverage | Flag cost as incomplete; do not silently use zero |
| Promise integrity | Immutable displayed commitment and timezone | Escalate missing or conflicting commitment |
| Event identity | Unique IDs, idempotent ingestion, duplicate rate | Quarantine or reconcile duplicate events |
| Delivery completeness | Carrier status and unresolved in-transit age | Mark outcome censored rather than on time |
| Model calibration | Observed versus predicted rates by eligible cohort | Restrict use or recalibrate under governance |

Standards help exchange and interpret data; they do not transfer accountability for correct routing.
The enterprise must assign an owner to every control and define an actionable response.

## 9. Industry practice and case evidence

### Published operational context

The U.S. Census Bureau publishes quarterly e-commerce retail sales estimates and definitions.
These figures are evidence that e-commerce is a material retail channel, not evidence that any particular routing policy is optimal.
The series should be cited with its period and revision status when used in a business case.
A national aggregate cannot estimate a retailer's store pick cost or inventory accuracy.

Walmart's annual reports describe its omnichannel business, including stores and fulfillment capabilities used to serve customers across channels.
That public description supports the existence of a multi-node operating model.
It does not disclose this topic's illustrative costs, reliability threshold, solver, or specific reservation logic.
**Inference:** a network combining stores and fulfillment facilities must coordinate inventory and execution decisions; the public report alone does not establish how a particular order is routed.

Target's annual reports describe the role of stores in fulfilling digital sales and the company's same-day services.
This is company-reported evidence about the operating model, not an independent measurement of marginal routing economics.
**Inference:** store-fulfillment capacity and inventory availability matter to a business using stores for digital fulfillment.
The report does not establish that its local decisions use the objective or coefficients in this lesson.

The cited academic literature on omnichannel operations discusses interactions among inventory, fulfillment, and customer service.
Models are useful for clarifying trade-offs and assumptions; published results should not be transferred to a different network without checking demand, cost, and operational conditions.
Vendor documentation can show how a product represents sourcing rules, but product features are not independent proof of business impact.

### What practitioners need to validate locally

A store may have a lower parcel rate than a DC but a higher short-pick rate.
A DC may have more reliable inventory but a later pickup that misses the customer's deadline.
A single order can look cheap while consuming inventory needed for a local sale.
Each of these is a testable hypothesis about a specific node, SKU, destination, and time bucket.
None follows automatically from the fact that a facility is called a store or a DC.

When comparing alternatives, finance should distinguish actual incremental charges from allocations used for financial reporting.
Operations should validate whether pick-time standards reflect current workload and staffing.
Transportation should verify the difference between a quoted rate and the invoice after surcharges.
Customer experience should define late delivery and the remedy, including whether a first attempt counts.
Analytics should check whether outcome data cover the orders that were accepted, canceled, rerouted, or still in transit.

### Implementation maturity

A reasonable initial capability is a transparent rule engine with eligibility, a cost table, an explicit tie-break, reservation, and exception handling.
The next step is not automatically a complex optimizer.
First establish that inventory snapshots, capacity, cutoffs, and outcome joins are trustworthy enough for the decision.
Then measure where a greedy baseline leaves value, such as competition for a scarce store unit.
Only then select batch optimization, online opportunity costs, or another method appropriate to the available decision window.

A controlled pilot can run a proposed policy in shadow mode using the same information available to the live policy.
Shadow mode shows model decisions but does not observe the counterfactual delivery outcome of routes not actually chosen.
A randomized rollout, when ethically and operationally acceptable, can estimate incremental effects more credibly.
The randomization unit must match interference: orders competing for the same capacity can affect each other.
A location-time or market-time cluster may be more appropriate than independently randomizing individual orders, subject to power and implementation review.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Phantom stock | Store short picks rise for particular SKUs | Recorded units are absent or unsaleable | Reroutes and missed promises | Audit inventory, improve event latency, and use approved buffers |
| Double reservation | Conflict or negative available balance | Concurrent readers allocate the same last unit | Oversell and customer delay | Conditional atomic reservation and idempotent retry |
| Missing reservation acknowledgement | Timeout with uncertain state | Request succeeded but response was lost | Duplicate allocation on naive retry | Query or retry with the same idempotency key |
| Capacity overload | Acceptance exceeds completed picks | The feed ignores breaks, workload, or backlog | Late tender and store disruption | Time-bucket capacity with operations-owned limits |
| Cutoff error | Parcels miss handoff despite prompt picks | Wrong timezone, holiday, or pickup schedule | Late delivery | Versioned calendars and cutoff tests |
| Unpriced split | Parcels per order rise while quoted cost falls | Per-line score omits second shipment charge | Higher actual cost and customer confusion | Model shipment-open costs and customer constraints |
| Double-counted opportunity | Predicted cost rises without a matching driver | Future loss appears in both ATP protection and cost or batch demand | Good routes rejected or overpriced | Document term provenance and non-overlap |
| Ignored opportunity | Local stockouts rise after digital peaks | Last unit assigned without considering local demand | Lost contribution or costly recovery | Estimate displacement or enforce approved reserve |
| Miscalibrated reliability | Predicted 99% cohort delivers well below 99% | Distribution shift or biased training data | Promises fail despite passing threshold | Monitor cohorts, uncertainty, drift, and recalibration |
| Correlated delays ignored | Multi-parcel orders underperform independent estimate | Shared weather, hub, or node affects shipments together | Understated order-level late risk | Use joint data, scenarios, or bounds |
| Promise overwritten | Latest ETA replaces displayed commitment | Mutable order field loses history | Misleading on-time reporting | Preserve immutable original promise and changes |
| Soft penalty used as hard rule | Some restricted routes win when cheap | Penalty is finite and outweighed by cost | Policy or legal breach | Encode restriction as eligibility constraint |
| Allocated fixed cost treated as marginal | Routing avoids an idle node on paper | Rent or salaried overhead assigned per order | Incorrect local decisions | Agree decision horizon and relevant-cost ledger |
| Cost omission | Invoice exceeds model for remote areas | Surcharge or dimensional weight missing | Margin erosion | Reconcile modeled quotes to invoices |
| Infeasible plan treated as success | Solver returns no solution but work is released | Timeout or infeasibility misclassified | Unfillable or duplicate orders | Separate statuses and require validated feasible plan |
| Partial reservation leak | Reserved stock accumulates with no active work | Second resource reservation fails without compensation | Artificial shortage | Release or reconcile partial reservations |
| Endless rerouting | Same order cycles repeatedly | Rejected nodes re-enter without new information | Delay and excessive contact | Bound attempts, track exclusions, escalate |
| Biased offline evaluation | Replay shows large gains but live results do not | Unobserved outcomes for rejected routes are assumed known | False confidence | Randomize when feasible; label simulated outcomes |
| Incomplete denominator | On-time rate improves as cancellations rise | Canceled and unresolved orders omitted without disclosure | Service deterioration hidden | Publish outcome funnel and censoring policy |
| Excessive sensitive logging | Full addresses in diagnostic payloads | Debugging captures more data than necessary | Privacy and security exposure | Data minimization, access controls, retention policy |

A red flag is not a diagnosis by itself.
Investigate the underlying events and compare them with the exact policy and input snapshot in force.
A single late delivery also does not prove a calibrated 99% prediction was wrong.
Calibration is assessed over a population with suitable uncertainty intervals.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Snapshot reflects available stock | Allocation cannot exceed real saleable units | POS lag, damage, or phantom inventory | Compare records to cycle counts and short picks by age |
| Capacity is expressed in a consistent unit | Work must be compared with the same resource limit | Minutes modeled against order-count capacity | Validate unit conversions and observed task times |
| Candidate list covers relevant options | Optimization can only select represented choices | Search radius or heuristic excludes viable source | Compare sampled decisions against expanded enumeration |
| Route costs are incremental | Objective should represent the decision's economic effect | Fixed allocations or missing surcharges | Reconcile to invoices and review drivers with finance |
| Risk terms do not overlap | Each expected loss should be counted once | Same demand appears in batch and opportunity penalty | Trace each cost component to a distinct event and horizon |
| Probability estimates are calibrated | A modeled threshold must mean something operationally | Carrier disruption, sparse data, or selection bias | Compare predicted and observed outcomes by cohort |
| Order success is clearly defined | Split and reroute outcomes need a common denominator | First attempt confused with final delivery | Publish timestamp, unit, exclusions, and outcome funnel |
| Promises are immutable in the record | Evaluation needs the original customer commitment | Later ETA overwrites checkout promise | Audit event history and preserve versions |
| Resource reservations are enforceable | A chosen plan needs execution rights | Non-atomic writes or uncoordinated expiration | Exercise concurrency and reconciliation controls in implementation |
| Batch is known before cutoff | Joint optimization needs time to collect and solve orders | Waiting makes early orders miss their handoff | Measure decision latency and test alternative batch windows |
| One-unit candidate representation fits the example | Small enumeration illustrates shared constraints | Multi-line consolidation or partial shipment | Use complete plan candidates or a richer integer model |
| Static costs approximate the planning horizon | Comparing candidates requires a defined cost basis | Congestion, changing carrier prices, or uncertain future demand | Recompute by time bucket and run sensitivity scenarios |
| External risk is separate from batch orders | Example opportunity costs represent other future demand | The same future order is explicitly included | Remove overlapping penalty or revise forecast |
| Independence is not silently assumed | Joint on-time estimates depend on shared disruptions | Common hub, weather, or carrier outages | Estimate joint outcomes or report valid bounds |
| A feasible snapshot can be committed | Planning result must survive concurrent updates | POS sale or node rejection after the read | Reserve conditionally and reroute on conflict |
| Cost minimization reflects approved priorities | Objective must match customer and business policy | A safety or service rule is priced rather than enforced | Use hard or lexicographic constraints with named owners |

The worked example is deterministic apart from the input forecasts.
It does not optimize returns, choose an inventory replenishment plan, or prove what any real retailer does.
Risk values, threshold, and capacity are educational assumptions, not recommendations for a deployment.

## 12. Exercises

### Exercise 1 — Draw the control loop

**Time:** 15 minutes.

**Task:** Draw the flow from customer promise through candidate generation, eligibility, ranking, reservation, execution, and feedback.
Show where an original promise is preserved and where a failed reservation returns.
Assign a primary owner to inventory freshness, capacity, carrier data, and customer communication.

**Deliverable:** a labeled diagram and a four-row ownership table.

**Acceptance criteria:** feasibility precedes ranking; reservation precedes release; a failed reservation causes refresh or escalation; the original promise remains visible.

### Exercise 2 — Separate feasibility from preference

**Time:** 15 minutes.

A route has a USD 5.00 estimated cost, but the carrier does not accept the product's required handling category.
A second route costs USD 8.00 and meets all product, capacity, and promise requirements.
Another route costs USD 6.00 and is eligible but has an approved USD 1.50 inventory opportunity cost not included in that estimate.

**Task:** classify each route and select the least-cost feasible one.

**Deliverable:** a three-row table of eligibility, comparable cost, and reason.

**Acceptance criteria:** the prohibited route is excluded rather than merely penalized; all included cost terms are stated; the selected route is feasible and has the lowest comparable cost.

### Exercise 3 — Compute incremental cost

**Time:** 15 minutes.

For a route, carrier cost is USD 5.75, packaging is USD 0.65, and incremental labor is USD 1.80.
The facility also allocates USD 2.40 of unchanged monthly rent to each order.
The estimated chance of a displaced future sale is 20%, and lost contribution if displaced is USD 10.00.
No other cost overlaps these terms.

**Task:** calculate modeled incremental cost and explain the treatment of rent.

**Deliverable:** an equation in USD per order and a two-sentence explanation.

**Acceptance criteria:** the opportunity term is probability-weighted; unchanged rent is excluded; the result and assumptions are explicit.

### Exercise 4 — Interpret reliability

**Time:** 20 minutes.

A split order has two required shipments, with marginal on-time probabilities of 0.98 and 0.95.

**Task:** calculate the probability both are on time under independence, and a lower bound without an independence assumption.
Explain why neither value guarantees this particular customer's outcome.

**Deliverable:** two calculations and a short explanation of a shared disruption.

**Acceptance criteria:** multiplication is labeled conditional on independence; the bound is valid without it; the success event covers both shipments.

### Exercise 5 — Allocate scarce capacity

**Time:** 25 minutes.

Two orders each need one unit.
Store S has one unit and can pick only one order; DC D has two units and enough capacity for both.
All routes meet the promise policy.
The incremental costs in USD are: A=S 7, A=D 9, B=S 8, B=D 14.

**Task:** enumerate the feasible assignments and compare the optimal plan with a greedy policy that routes A first to its cheapest source.

**Deliverable:** assignment table, minimum cost, and greedy cost difference.

**Acceptance criteria:** store capacity is enforced; all feasible assignments are compared; savings are calculated in USD for this batch.

### Exercise 6 — Recover from an uncertain reservation

**Time:** 20 minutes.

The router submits a reservation for the final unit, but the request times out before a response arrives.
The customer has already been promised delivery by Friday at 18:00 local time.

**Task:** design a safe state sequence through reconciliation, either successful reservation or failure, and customer escalation if no alternative meets the original promise.

**Deliverable:** a state diagram or numbered sequence with an idempotency strategy.

**Acceptance criteria:** timeout is not treated as certain failure; no unrelated duplicate reservation is issued; the original commitment is preserved; unresolved ambiguity is escalated.

### Exercise 7 — Evaluate a new routing policy

**Time:** 30 minutes.

A team proposes replacing a store-first rule with a cost-and-reliability model.
Orders share store inventory and labor capacity.

**Task:** propose a pilot, primary outcome, at least three guardrails, randomization unit, and analysis of cancellations and incomplete deliveries.

**Deliverable:** a one-page experimental design and a rollback trigger.

**Acceptance criteria:** design addresses shared-resource interference and changing promise length; outcomes include recovery cost and order-level service; unobserved counterfactuals are not claimed as facts.

### Exercise 8 — Repair an objective

**Time:** 20 minutes.

A model minimizes shipping cost plus a USD 100 penalty for each legally prohibited shipment.
It also adds the expected cost of displaced demand for a future order that is already explicitly represented in the batch.
A team says the penalty is large enough because most shipping costs are below USD 100.

**Task:** identify both defects and rewrite the policy in words or equations.

**Deliverable:** a corrected objective and feasibility rule, with ownership of any exception.

**Acceptance criteria:** prohibited routes cannot be selected; the duplicate future-demand term is removed or redefined; the explanation does not rely on an arbitrary penalty.

## 13. Self-check answers

### Exercise 1

A strong diagram includes the customer commitment and a timestamped resource snapshot before candidate generation.
It then shows eligibility filtering, selection, conditional reservation, release, and execution events.
A reservation conflict returns to refresh and re-decision, with a bounded escalation path.
The original promise is retained even if a later forecast changes.

Inventory operations or the inventory data owner owns stock freshness.
Store and DC operations own executable capacity.
Transportation owns carrier calendars, service eligibility, and rate inputs.
Customer experience owns customer-facing commitment changes and communications.
The precise organizational titles can differ, but no control should lack an accountable owner.

### Exercise 2

| Route | Eligible? | Comparable cost | Reason |
|---|---|---:|---|
| Restricted carrier | No | Not ranked | Product handling requirement is not met |
| Second route | Yes | USD 8.00 | Meets stated constraints |
| Third route | Yes | USD 7.50 | USD 6.00 plus USD 1.50 opportunity cost |

Choose the third route at USD 7.50, assuming the opportunity term does not duplicate another cost.
The prohibited USD 5.00 option cannot become eligible through a penalty.

### Exercise 3

Opportunity cost is `0.20 × USD 10.00 = USD 2.00` per order.
The modeled incremental total is `USD 5.75 + USD 0.65 + USD 1.80 + USD 2.00 = USD 10.20` per order.
The allocated USD 2.40 of unchanged rent is not caused by this route choice and is excluded from this short-horizon decision.
If a different long-horizon decision would change the facility's rent, that changed cost could be relevant to that different decision.

### Exercise 4

Under independence, `0.98 × 0.95 = 0.931`, or 93.1% for both shipments on time.
Without independence, the union bound gives `max(0, 0.98 + 0.95 - 1) = 0.93`, or 93%.
The bound is not an estimate of the actual joint probability.
A shared weather disruption can make shipment failures dependent.
Even a well-calibrated probability describes a distribution of outcomes, not certainty for one order.

### Exercise 5

| A source | B source | Feasible? | Total |
|---|---|---|---:|
| S | S | No: two orders exceed one store unit and pick slot | USD 15 |
| S | D | Yes | USD 21 |
| D | S | Yes | USD 17 |
| D | D | Yes | USD 23 |

Choose A from D and B from S for USD 17.
Greedy A-first chooses A from S, then B from D, for USD 21.
The batch plan saves USD 4 under the stated costs and constraints.
This result is conditional on knowing both orders in time to allocate them jointly.

### Exercise 6

A safe sequence is `requested -> outcome unknown -> reconcile by same idempotency key`.
If the reservation exists, confirm its validity and proceed to node acceptance.
If it definitively failed, refresh inventory and capacity and select another feasible route, retaining the original Friday commitment.
If no route meets the commitment, escalate under the approved customer-notification and remedy policy.
If the outcome remains unknown, keep the order in an explicit uncertain state until reconciled; do not create a second unrelated reservation.

### Exercise 7

Use a cluster or time-window randomization if individual orders in the same inventory pool would interfere materially.
Specify eligibility, treatment assignment, duration, sample-size plan, and analysis before launch.
A suitable primary outcome could be incremental cost per placed order over a defined follow-up window, alongside a firm order-level on-time guardrail.
Other guardrails include cancellation rate, short-pick rate, split rate, customer-contact rate, and promise duration.
Report placed, canceled, delivered, late, and still-in-transit orders rather than silently excluding inconvenient outcomes.
Define an operational rollback trigger and owner before traffic starts.
A design is incomplete if it treats simulated outcomes for unchosen routes as observed.

### Exercise 8

First, remove legally prohibited routes from the feasible set; a finite USD 100 penalty can be outweighed by a sufficiently large cost difference.
Second, remove the duplicate expected-loss term or redefine it to cover only demand not represented by the batch.
Minimize the remaining non-overlapping incremental costs over the feasible set.
The applicable compliance owner decides policy interpretation; the optimizer cannot authorize a prohibited shipment.
If no feasible plan remains, escalate rather than releasing an illegal one.

## 14. Interview questions

1. **Why is the nearest source not necessarily the right source?**
   A strong answer considers eligibility, total incremental cost, cutoffs, capacity, inventory risk, and the customer commitment.
   A weak answer assumes distance alone determines cost and on-time performance.
2. **What is the difference between on-hand inventory and ATP?**
   A strong answer distinguishes physical record, saleability, reservations, protection, and time.
   A weak answer treats all recorded units as simultaneously available to every order.
3. **When is a hard constraint preferable to a penalty?**
   A strong answer identifies non-negotiable product, legal, or commitment rules and explains infeasibility handling.
   A weak answer assumes a large arbitrary penalty guarantees compliance.
4. **Why can greedy routing be more expensive than a batch plan?**
   A strong answer gives a scarce-resource example and compares complete feasible assignments.
   A weak answer compares only the first order's cost.
5. **How would you estimate the cost of using the last store unit?**
   A strong answer considers displaced demand, contribution, alternatives, replenishment, and overlapping protections.
   A weak answer multiplies the full selling price by an unsupported probability.
6. **How do you measure a split order's on-time reliability?**
   A strong answer defines all-required-unit success and discusses dependence and joint outcomes.
   A weak answer reports the best parcel's carrier rate as the order's probability.
7. **What happens if a reservation call times out?**
   A strong answer reconciles using the same idempotency key and preserves an uncertain state.
   A weak answer immediately creates a new independent reservation.
8. **What belongs in an auditable decision record?**
   A strong answer includes the original promise, as-of inputs, candidates, exclusions, versions, selected plan, and subsequent events.
   A weak answer stores only the winning node and current ETA.
9. **How would you evaluate a proposed policy without observing rejected routes?**
   A strong answer distinguishes replay assumptions from randomized evidence and considers shared capacity interference.
   A weak answer claims simulated counterfactual delivery outcomes were observed.
10. **What should happen when the optimizer finds no feasible assignment?**
    A strong answer distinguishes infeasibility from timeout, invokes approved exceptions, and retains customer commitments.
    A weak answer releases the least-infeasible solution as if it satisfied all rules.

## 15. Key terms

- **Allocation:** assignment of order quantities to sources; see [mechanics](#5-the-mechanics-step-by-step).
- **Available-to-promise:** policy-adjusted quantity available by time; see [vocabulary](#4-definitions-and-vocabulary).
- **Batch optimization:** joint selection of plans for multiple orders; see [Step 8](#step-8-choose-the-policy-rules-or-optimization).
- **Calibration:** agreement between predicted probabilities and observed rates; see [reliability](#step-7-model-promise-reliability).
- **Candidate:** possible complete routing plan; see [Step 3](#step-3-enumerate-plausible-fulfillment-patterns).
- **Capacity:** executable workload limit by node and time; see [Step 2](#step-2-build-an-as-of-resource-snapshot).
- **Chance constraint:** minimum modeled probability for a defined event; see [vocabulary](#4-definitions-and-vocabulary).
- **Commitment:** customer-facing promise, not a forecast; see [mental model](#3-the-core-mental-model).
- **Cutoff:** latest action time for a service window; see [Step 4](#step-4-apply-hard-constraints).
- **Distributed order management:** cross-node coordination of order allocation and execution; see [why this topic exists](#1-why-this-topic-exists).
- **Eligibility:** requirements that a candidate must satisfy; see [Step 4](#step-4-apply-hard-constraints).
- **Feasibility:** satisfaction of all modeled hard constraints; see [worked example](#6-worked-example).
- **Forecast:** uncertain estimate of a future outcome; see [mental model](#3-the-core-mental-model).
- **Hard constraint:** condition that cannot be traded for a cost reduction; see [Step 4](#step-4-apply-hard-constraints).
- **Idempotency:** safe repeated processing of the same request; see [Step 9](#step-9-reserve-and-release-safely).
- **Incremental cost:** future cost changed by the decision; see [Step 5](#step-5-estimate-incremental-execution-cost).
- **Inventory protection:** units withheld by policy; see [Step 6](#step-6-value-inventory-and-capacity-risk).
- **Node:** source or execution location; see [vocabulary](#4-definitions-and-vocabulary).
- **Opportunity cost:** value of a foregone alternative; see [Step 6](#step-6-value-inventory-and-capacity-risk).
- **Order-level reliability:** probability or observed rate of complete order success under a defined event; see [Step 7](#step-7-model-promise-reliability).
- **Phantom stock:** recorded units that are not usable in practice; see [failure modes](#10-failure-modes-and-red-flags).
- **Reservation:** recorded claim on resources; see [Step 9](#step-9-reserve-and-release-safely).
- **Reroute:** controlled reassignment after a material change; see [worked example](#6-worked-example).
- **Shadow price:** marginal value of a constraint under appropriate model assumptions; see [Step 6](#step-6-value-inventory-and-capacity-risk).
- **Split shipment:** fulfillment of an order through multiple shipments; see [reliability](#step-7-model-promise-reliability).
- **Target:** internal desired level, distinct from a commitment; see [mental model](#3-the-core-mental-model).
- **Union bound:** dependence-free lower bound for joint success from marginal probabilities; see [Exercise 4](#exercise-4--interpret-reliability).
- **Versioned snapshot:** the recorded data state available when a decision was made; see [data contract](#decision-data-contract).

## 16. Sources

1. GS1, [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
   Primary standards source for identification keys and related specifications; consult the applicable version.
2. GS1, [EPCIS](https://www.gs1.org/standards/epcis).
   Primary source for the event-visibility standard; it does not prescribe a routing objective.
3. U.S. Federal Trade Commission, [Business Guide to the FTC's Mail, Internet, or Telephone Order Merchandise Rule](https://www.ftc.gov/business-guidance/resources/business-guide-ftcs-mail-internet-or-telephone-order-merchandise-rule).
   Regulatory guidance on covered shipment representations and delays; check current requirements and applicability.
4. U.S. Census Bureau, [Quarterly Retail E-Commerce Sales](https://www.census.gov/retail/ecommerce.html).
   Official aggregate channel data, not a source for individual-node economics.
5. Walmart Inc., [Annual Reports and Proxies](https://stock.walmart.com/financials/annual-reports-and-proxies/default.aspx).
   Company-reported description of its business and omnichannel operations; not independent evidence of this lesson's model.
6. Target Corporation, [Annual Reports](https://investors.target.com/financial-information/annual-reports).
   Company-reported description of the role of stores and digital fulfillment; check the specific report year used.
7. Hübner, A., Holzapfel, A., and Kuhn, H. (2016), “Distribution systems in omni-channel retailing,” *Business Research*, 9, 255–296, [https://doi.org/10.1007/s40685-016-0034-7](https://doi.org/10.1007/s40685-016-0034-7).
   Academic context on distribution design; transfer of any finding requires matching assumptions and operating context.

Sources are educational references, not endorsements or evidence that the hypothetical numerical example occurred.
The regulatory discussion is dated to November 2026 and is not legal advice.
Check current standards versions, company reports, and applicable law before implementation.

## 17. Links

- [Module 23 index](_index.md).
- [Module 23 survey](../../../course/23_omnichannel_fulfillment_returns.md).
- [Inventory visibility](01_inventory-visibility.md).
- [Available-to-promise](02_available-to-promise.md).
- [Ship-from-store economics](04_ship-from-store.md).
- [Promise accuracy](06_promise-accuracy.md).
- [Simulation and optimization production labs survey](../../../course/30_simulation_optimization_production_labs.md).

Sibling topics may be listed as planned in the module index while the expansion is being produced.
