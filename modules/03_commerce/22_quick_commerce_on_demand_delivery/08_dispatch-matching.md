# Dispatch and Matching for On-Demand Delivery

> Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics
> Topic 8 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 5, 6, and 7
> Version 2026.5

## 1. Why this topic exists

At 18:04 a dark store has two packed orders and two available couriers.
One courier is closer to both orders, but is the only courier who can carry the larger basket.
A nearest-first decision can send that courier to the smaller basket and leave the other customer waiting.
The error is not visible if the dispatcher measures only the first pickup.

Dispatch converts uncertain demand, predictions, and courier availability into offers and commitments.
The decision is repeated as new information arrives, not made once at the start of the shift.
An offer can be declined, an order can be delayed at the store, and a courier's location can become stale.
A good decision must remain executable when these ordinary events occur.

A customer needs an accurate promise and a safe delivery.
A courier needs a legible offer, an opportunity to refuse where applicable, and a safe workload.
The store needs a realistic pickup time rather than a crowd of couriers waiting for unready bags.
The operator needs to protect service and understand all incremental costs, including waiting and failed offers.
The dispatch controller needs a safe fallback when the optimizer or location feed is unavailable.

A match that appears cheap can create expensive downstream effects.
Long unpaid waits may cause courier attrition or legal exposure.
Repeated long pickups can make offers unattractive even when delivery distance is short.
Assigning work solely by speed can concentrate the most rewarding opportunities among a subset of workers.
Averages can hide severe lateness or systematically poor opportunities in particular neighborhoods.

The decision is not simply which courier is nearest.
It is which feasible set of offers to make now, which orders to defer, and what to do if an offer fails.
A plan is a proposed assignment; it becomes an accepted assignment only after the relevant acceptance event.
An estimated time of arrival (ETA) is a forecast, not a customer commitment.
A service target is an aggregate operational goal, not a guarantee for each order.

### Who owns the decision

| Decision | Accountable owner | Required input | Output or control |
|---|---|---|---|
| Customer promise | Product and operations lead | Capacity, service history, legal commitments | Published commitment and escalation policy |
| Physical eligibility | Operations and safety lead | Load, equipment, food and local requirements | Versioned hard constraints |
| Offer and pay terms | Marketplace and legal owners | Contract terms, local rules, pay model | Approved, understandable offer |
| Dispatch objective | Dispatch product owner | Service, cost and fairness trade-offs | Versioned objective and thresholds |
| Live exception | On-duty dispatch lead | State, alerts, courier and store reports | Reassign, defer, or escalate |
| Courier decision | Courier | Offer details, practical constraints | Accept, decline, or raise safety concern |
| Measurement | Analytics owner | Logged decisions and outcomes | Monitoring and experiment readout |
| Fairness governance | Marketplace and responsible-AI owners | Exposure and outcome distributions, legal review | Auditable safeguards and remediation |
| Incident response | Operations incident commander | Failure scope and fallback capacity | Pause, rollback, communicate, recover |

A model can recommend an offer; it cannot transfer accountability for the policy to a model vendor.
The correct fallback may be to defer or stop taking orders, not to bypass a safety constraint.

## 2. Learning outcomes

After this unit you can:

1. **Distinguish** an offer, acceptance, assignment, pickup, and delivery event.
   Exercise 1 checks the state transition and recovery path.
2. **Screen** courier-order edges using capacity, timing, equipment, and state constraints.
   Exercise 2 asks for an eligibility record.
3. **Compute** a joint matching cost and explain why a greedy rule can lose.
   Exercises 3 and 4 test the calculation and executable artifact.
4. **Quantify** courier and customer outcomes without treating unequal raw averages as proof of discrimination.
   Exercise 5 asks for a stratified fairness analysis.
5. **Design** a rolling dispatch policy with deferral and offer-failure recovery.
   Exercise 6 requires a state diagram and exception ownership.
6. **Evaluate** a policy experiment using service, economics, worker outcomes, and interference-aware design.
   Exercise 7 calls for a measurement plan and stop rules.
7. **Audit** a proposed production deployment for safety, privacy, and operational accountability.
   Exercise 8 asks for a go/no-go recommendation.

The exercises are intentionally different from a model accuracy quiz.
A mathematically optimal assignment to an inaccurate or stale state is not necessarily a good operational decision.

## 3. The core mental model

```text
Orders and commitments       Couriers and capacity       External state
 readiness, destination      location, load, status      travel, weather
             \                     |                       /
              \                    |                      /
               ----> timestamped decision snapshot <----
                              |
                    eligible courier-order edges
                  /          |             \
              safety      availability     promise risk
                              |
                   jointly compare match sets
                + explicit defer / unserved option
                              |
                reserve -> offer -> accept or decline
                     |                  |
                     |             release and retry
                     v
               pickup -> deliver -> reconcile
                     |
       service, cost, exposure, safety, and incident evidence
                     |
           monitoring -> approved policy change
```

The left side is demand; the middle is supply.
The edge between an order and a courier represents a possible match at a particular time.
Some edges do not exist because a constraint is violated.
The rest have expected service and cost consequences.

A matching is a set of edges, not a ranking of isolated edges.
A courier who is eligible for only one order is a scarce resource for that order.
The marginal cost of assigning that courier elsewhere includes the lost alternative.
When an order can be left unmatched, the comparison must include the consequence of waiting.
Otherwise an optimizer can report zero cost by choosing no assignments.

The snapshot is not the world itself.
It is a collection of facts and forecasts with different timestamps and confidence levels.
A courier location from five seconds ago and a location from five minutes ago should not be treated as equivalent.
A travel estimate can be internally consistent while systematically underestimating rain or building access time.

The decision closes a control loop only if the system observes the offer outcome.
A declined offer frees the reservation and triggers a new decision; it does not mean that the order was delivered.
Repeated retries need bounded behavior so an inaccessible order does not monopolize the dispatcher.
The objective and its constraints must be reviewed using outcomes, not merely the optimizer's predicted score.

### Three decision layers

| Layer | Main question | Example | Typical owner |
|---|---|---|---|
| Policy | What is permitted and valued? | Maximum load, service thresholds, pay terms | Marketplace and safety leads |
| Real-time decision | What should happen on this snapshot? | Offer order B to courier A | Dispatch service or controller |
| Learning | Did the policy work, and for whom? | Compare late-tail and courier wait distributions | Analytics and governance leads |

Do not confuse these layers.
Changing an objective weight is a policy decision even if the system makes millions of real-time choices automatically.
A one-off dispatcher override is an operational action and must be logged as such.
A dashboard is evidence for review, not a substitute for authority to intervene.

### Hard constraints and preferences

A hard constraint removes an edge before ranking.
For example, a courier cannot be assigned a 12 kg basket when verified remaining carrying capacity is 8 kg.
A preference changes a score among allowable choices.
A small difference in pickup time can be traded against another approved objective.
A safety rule must not become a cheap penalty that can be outweighed by predicted profit.

A customer due time may be treated differently depending on the contract and the operating policy.
An operator might minimize predicted lateness while still serving an already late order.
A strict cutoff may instead disallow an offer that has insufficient safety margin.
Either choice should be explicit; a promise is not made feasible merely by declaring it a hard constraint.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Dispatch | Operational process of deciding and issuing work offers | Treating a forecast as a completed assignment |
| Matching | Selection of a compatible set of courier-order edges | Choosing each edge independently |
| Order | Customer request with items, destination and service commitment | Assuming every order is ready for pickup |
| Courier | Person or delivery resource capable of undertaking eligible work | Treating a worker as always available |
| Decision epoch | Time at which the system evaluates a snapshot, in timestamp units | Assuming all inputs were observed at that time |
| Snapshot | Versioned data visible at an epoch, including source times | Reconstructing with future information |
| Edge | Candidate courier-order pairing at a decision epoch | Treating every physically possible edge as safe |
| Candidate generation | Finding a bounded set of plausible edges | Excluding distant or atypical candidates without measurement |
| Eligibility | Satisfaction of all hard requirements | Ranking an unsafe candidate with a large penalty |
| Capacity | Remaining safe load in kg, volume, slots, or other relevant unit | Equating nominal vehicle capacity with remaining capacity |
| Compatibility | Whether goods, equipment, and handling conditions can coexist | Checking only total weight |
| Availability | State in which the courier can receive and perform an offer | Assuming online status guarantees willingness |
| Reservation | Temporary protection against conflicting offers | Confusing a reservation with acceptance |
| Offer | Proposed job and its disclosed terms | Calling an unanswered offer an assignment |
| Acceptance | Explicit courier agreement under the relevant workflow | Inferring acceptance from delivery of a notification |
| Decline | Explicit refusal of an offer | Assuming it implies future unavailability |
| Timeout | No valid response before an offer deadline | Counting it automatically as a deliberate decline |
| Assignment | Binding operational association under the platform's rules | Assuming an assignment guarantees pickup |
| Release | Removal of a reservation or assignment under defined conditions | Reoffering before the prior state is safely resolved |
| Pickup ETA | Forecast arrival at the pickup point, in minutes | Treating a point forecast as guaranteed readiness |
| Delivery ETA | Forecast arrival or handoff time, with a defined endpoint | Mixing doorstep arrival and completed handoff |
| Readiness time | Time the order is expected or observed to be ready | Substituting pack-start time for ready time |
| Wait time | Time between an arrival and the next event, in minutes | Ignoring wait because the courier is not moving |
| Deadhead | Travel without a customer order, often toward pickup | Counting only the loaded leg as service cost |
| Slack | Due time minus predicted completion time, in minutes | Ignoring uncertainty when slack is positive |
| Lateness | Positive part of completion time minus due time, in minutes | Averaging only orders that arrived late |
| Risk buffer | Extra time or cost allowance for uncertain outcomes | Calling an arbitrary fixed buffer a calibrated quantile |
| Objective | Explicit function used to compare feasible assignments | Reporting its score as actual realized profit |
| Opportunity cost | Value of the best displaced alternative | Adding it twice when joint optimization already captures it |
| Deferral | Decision to reconsider an order later | Silently dropping an order from the denominator |
| Unserved penalty | Modeled consequence of not assigning now, in currency or score units | Treating a modeling penalty as an actual payment |
| Utilization | Productive or engaged time divided by a defined available-time denominator | Comparing metrics with different denominators |
| Offer exposure | Number or rate of relevant offers received by a courier | Equating exposure with earnings or fairness |
| Acceptance rate | Accepted offers divided by eligible offers under a stated definition | Comparing couriers without conditioning on offer quality |
| Fairness | A specified distributional objective or constraint | Assuming a single metric proves absence of unfairness |
| Counterfactual | Outcome under an alternative policy or action | Treating unobserved alternative outcomes as measured facts |
| Interference | One unit's treatment changes another unit's outcome | Assuming orders in the same market are independent |
| Override | Authorized departure from an automated recommendation | Making overrides invisible to evaluation |
| Idempotency | Repeating a request without applying its effect more than once | Assuming retries cannot create duplicate offers |
| Stale data | Information too old for its intended decision, relative to policy | Using one universal age threshold without context |
| Regret | Difference between outcomes of a chosen policy and a specified comparator | Claiming observed regret against an unknowable perfect future |

## 5. The mechanics, step by step

### Step 1: Define the decision and its horizon

**Input:** order backlog, available courier pool, decision clock, and policy version.
**Output:** a list of decisions the engine may make at this epoch.

Decide whether the engine assigns single orders, predefined batches, or route insertions.
This unit uses single-order assignment for the numerical example.
Batch construction and route feasibility require the additional checks in Topic 7.
A precomputed batch must not be treated as feasible merely because its component orders were feasible separately.

Define when the engine is invoked.
Triggers can include an order becoming ready, a courier becoming available, an offer response, or a periodic timer.
Coalesce closely spaced triggers when needed, but bound the delay imposed on orders already waiting.
Record the event time, processing time, and snapshot time separately.

State the horizon over which the policy values a match.
A short horizon is easier to compute but can ignore the next delivery or the likelihood of future demand.
A longer horizon needs forecasts and can be fragile when supply changes quickly.
Do not claim that a single-period optimum is globally optimal over an entire shift.

**Branch:** if no safe current action exists, defer, expand the permitted search under policy, or escalate.
Do not invent availability to make an optimization problem feasible.

### Step 2: Capture a consistent state

**Input:** order, store, courier, travel, and policy event streams.
**Output:** versioned candidate snapshot with provenance and freshness checks.

For each order, capture its identifier, ready state, destination, due time, load, and handling needs.
Separate observed facts from predictions.
For example, `ready_at` may be an actual scan or a store readiness forecast; the two must not share an ambiguous field.
Capture cancellation and substitution state before making an offer.

For each courier, capture online status, coordinates, time of last location, equipment, load already carried, and current reservation.
Availability must reflect the state machine, not just a recent app heartbeat.
Record the courier's permitted work region and any required credentials where relevant.
Use only data that is necessary and authorized for the operational purpose.

For each estimate, record the model version, feature timestamp, prediction time, and endpoint definition.
If the location or order state exceeds a validated age threshold, refresh it or exclude the edge.
A threshold is an operational choice calibrated to context, not a universal constant.
A snapshot with contradictory states should be held for reconciliation rather than silently repaired.

**Branch:** if the feed is degraded, invoke the approved stale-data fallback and alert the on-duty lead.
A fallback may reduce radius, seek confirmation, pause automated dispatch, or use an approved manual workflow.

### Step 3: Generate candidate edges

**Input:** consistent orders and available couriers, plus a permitted search rule.
**Output:** a candidate list and rejection reasons.

A spatial search can reduce computation, but straight-line distance is only a screening approximation.
Road access, bridges, building entrances, and pedestrian restrictions affect real travel.
A candidate outside the initial radius may be necessary when local supply is scarce.
Record the search radius and every expansion so that exclusions can be audited.

Candidate generation affects outcomes before the optimizer sees the problem.
A solver cannot select a courier whose edge was never generated.
Measure coverage by geography, order type, and time bucket, subject to privacy protections.
Check whether sparse neighborhoods systematically receive fewer feasible candidates.

Avoid unlimited search when location freshness or routing capacity degrades.
A larger pool is useful only if the estimates and decision latency remain reliable.

**Branch:** no candidate means an explicit no-candidate event, with an owner and review time.
It is not a zero-cost solution and it is not evidence that the customer cancelled.

### Step 4: Apply hard feasibility constraints

**Input:** candidate edges and current safety and operational rules.
**Output:** eligible edges with machine-readable exclusion reasons.

Check that the courier is eligible for work in the relevant area and is not already reserved elsewhere.
Check remaining weight, volume, and delivery slots against the order.
Use conservative units and compatible dimensions: kg cannot replace liters, and neither proves food separation.
Check required equipment, temperature controls, age-restricted handling, and pickup access where applicable.

Check that the route and offer are executable at the current time.
If a courier is already serving a route, an insertion needs explicit capacity and precedence checks.
If an order is not yet ready, determine whether waiting is allowed and how it is compensated.
Do not use an inferred willingness to accept an unsafe load as a substitute for safe capacity.

A hard constraint should have an owner, source, version, and exception policy.
If the rule is a legal requirement, route questions to qualified local counsel.
If it is an internal operating rule, document who can authorize a change.
An emergency override must not silently disable a non-overridable safety or legal control.

**Branch:** if eligibility cannot be established because a required field is missing, treat the edge as unknown and follow a defined safe path.
Do not convert missing capacity to infinite capacity or missing location to the store address.

### Step 5: Predict pickup and completion

**Input:** eligible edges, readiness estimates, travel times, and handoff estimates.
**Output:** predicted pickup, completion, slack, and uncertainty for every edge.

Let `t` be the decision time in minutes from a stated origin.
For courier `i` and order `j`, let `p_ij` be travel time to pickup.
Let `r_j` be the order's predicted ready time and `d_j` its delivery leg plus handoff duration.
Then predicted pickup completion is `max(t + p_ij, r_j)` if loading time is included elsewhere.
Predicted delivery completion is `max(t + p_ij, r_j) + d_j`.

This formula distinguishes courier wait from order wait.
Courier wait is `max(0, r_j - (t + p_ij))`.
Order wait after readiness is `max(0, (t + p_ij) - r_j)`.
If loading or access time is material, add it explicitly rather than hiding it in whichever component is convenient.

Slack is `due_j - predicted_completion_ij`.
Positive slack does not guarantee on-time delivery because the forecast has error.
Predicted lateness is `max(0, -slack)`; it is not the expected value of random lateness unless the model supports that calculation.
A risk-aware policy can use a calibrated completion-time distribution or an explicitly validated buffer.

Do not sum component quantiles and call the result a total-trip quantile without justification.
Readiness delay and travel delay may be correlated during busy periods.
Backtest the whole-trip prediction by service tier, geography, weather, and time bucket with enough data.

**Branch:** if uncertainty is high, choose a conservative approved policy or obtain additional information.
A point prediction should not be used to override a safety margin required by policy.

### Step 6: Build an interpretable incremental cost

**Input:** edge forecasts, pay rules, service valuation, and policy weights.
**Output:** cost per eligible edge in a common unit such as USD per decision.

A practical score may include courier pay, travel and wait costs, expected lateness cost, and operational risk.
State whether travel and wait costs are platform cash payments, economic resource costs, or both.
Counting the same paid minute once as pay and again as an identical resource charge would double count.
A societal or worker-welfare objective can include noncash burdens, but label them separately.

Expected lateness cost should reflect the chosen loss function.
For example, a simple educational proxy is `USD 2 per predicted late minute`.
That rate is not a universal customer valuation, compensation schedule, or legal penalty.
A tail-risk term may penalize the probability or severity of very late deliveries.
Validate it against observed distributions rather than presenting a chosen coefficient as measured truth.

Include the cost of an unassigned order if the solver may defer.
For a short decision horizon, this is a modeled value of waiting, not necessarily the full lost-order value.
If every alternative is unsafe, the safe action can still be deferral even when its modeled penalty is large.

Fairness and welfare terms require explicit interpretation.
A bounded exposure constraint or audited distributional objective may be appropriate.
A fairness bonus based on protected characteristics can be legally and ethically sensitive; obtain local review.
A policy must not infer protected traits from location or device as a shortcut.

**Branch:** if the economic inputs are unreliable, compare a range of plausible values and report sensitivity.
Do not hide a policy choice in a coefficient named `model_score`.

### A distributional cost example

Point-forecast lateness is not the same as expected lateness.
Suppose an eligible match has two equally likely completion outcomes.
In the first outcome, delivery finishes 4 min before the due time.
In the second, delivery finishes 6 min after the due time.
The expected signed delay is `(−4 + 6) / 2 = 1 min`.
The expected positive lateness is `(0 + 6) / 2 = 3 min`.
At an illustrative USD 2.00 per late minute, expected late cost is USD 6.00.
Scoring the expected signed delay as if it were expected lateness would produce only USD 2.00.
The difference comes from applying the positive-part function before averaging outcomes.

In a deployed model, the possible outcomes and their probabilities are estimated rather than known.
Calibrate them against later observations and monitor error by operational context.
If a model supplies only a point prediction, describe the resulting lateness measure as a proxy.
Do not claim the USD 6.00 value for a real market from this fictional two-outcome illustration.

### Fairness as an explicit design choice

Start by specifying whose burden or opportunity is being assessed.
A courier may care about earnings per available hour, wait, deadhead, and predictable access to work.
A customer may care about late deliveries, cancellations, and unequal access across neighborhoods.
The store may care about congestion and the ability to prepare orders without unsafe shortcuts.
These outcomes are related but not interchangeable.

Define a denominator for every measure.
Offers per available hour need a consistent definition of available time.
Earnings per engaged hour exclude idle time, while earnings per available hour include it.
Comparing the two without labels can make the same shift appear very different.
Also report the share of offers involving excessive travel or waiting under a defined threshold.

Compare like contexts where possible: time of day, geography, equipment, work availability, and job characteristics.
These variables may themselves reflect earlier policy effects, so conditioning on them can obscure a mechanism.
Show both overall and appropriately stratified summaries rather than declaring one adjustment definitive.
Very small samples can reveal individuals and produce unstable rates.
Use aggregation, suppression, and approved access controls appropriate to the data and jurisdiction.

For example, a policy might reserve no particular offer for a particular person, but rotate among truly equivalent ties.
That policy can reduce repeated identifier-order effects without changing the eligible set.
It does not by itself equalize earnings, guarantee adequate pay, or resolve structural differences in access.
Assess actual results and include worker feedback and an appeal or correction route.

If a distributional constraint competes with service or cost, document the trade-off before implementation.
State the protected guardrails, the population, the time horizon, and the legal basis for data use.
Avoid optimizing a group statistic that can be improved by withholding offers from everyone.
Monitor absolute outcomes and harms alongside ratios.

### Step 7: Solve the joint allocation

**Input:** feasible edges, edge costs, and a defer option for each order.
**Output:** a set of proposed offers and deferred orders.

For a single-order problem, binary variable `x_ij` equals 1 if courier `i` is proposed for order `j`.
Binary variable `u_j` equals 1 if order `j` is deferred at this epoch.
Minimize `sum(c_ij * x_ij) + sum(P_j * u_j)`.
Require `sum_i x_ij + u_j = 1` for each order.
Require `sum_j x_ij <= 1` for each available courier.
Only create `x_ij` for eligible edges.

The equations mean every order is either assigned once or explicitly deferred.
Each available courier receives at most one proposed single-order assignment at that epoch.
The objective compares the whole set of choices, not each order independently.
Ties need a documented stable rule or randomized policy with a recorded seed.

For larger instances, a min-cost matching or mixed-integer optimization implementation can solve related formulations.
Batching, time windows, and route insertions add constraints and can change the computational class of the problem.
A solver's termination status matters: a timeout with a feasible incumbent is different from proven optimality.
Validate feasibility independently before sending offers.

**Branch:** if the solver fails or exceeds the latency budget, use an approved fallback on the same hard constraints.
Log the fallback, the reason, and the decisions it produced.

### Contention, leases, and late responses

Two independent dispatch workers can read the same courier as available.
Without a shared atomic reservation, both can issue apparently valid offers.
The system must decide which process owns the reservation before either offer is published.
An atomic compare-and-set, transaction, or equivalent mechanism can enforce the invariant.
The exact implementation depends on the storage and messaging architecture.

A reservation lease bounds how long a failed process can block the courier.
The lease expiry and offer expiry must be coordinated; otherwise an apparently valid offer may outlive its reservation.
On acceptance, the system must verify both the offer version and the current reservation in one protected transition.
If the lease has expired or the order has already been assigned, reject or reconcile the late response.
Communicate the outcome clearly to the courier rather than implying the accepted task still exists.

Use durable event identifiers and idempotency keys for retries.
An at-least-once message channel can deliver an event more than once.
The consumer should not pay, assign, or release twice merely because a message was replayed.
Preserve the original event and a correction trail when reconciling inconsistent systems.

Track contention rate, expired reservations, duplicate event attempts, and failed acceptance transitions.
A sudden rise can indicate a latency or state-store problem rather than poor courier behavior.
Escalate to the incident owner when the threshold is crossed.
Manual dispatch must use the same shared state or it can recreate the conflict outside the solver.

### Step 8: Reserve and issue offers safely

**Input:** proposed match set, current state, offer terms, and expiry policy.
**Output:** reserved offers or a recorded failure to issue.

Before offering, check that the order and courier have not changed since the snapshot.
Use an atomic reservation or equivalent concurrency control to prevent conflicting live offers.
The reservation should have an expiry and an owner.
A retry must use an idempotency key so network duplication does not create duplicate work.

Disclose the information required by applicable rules and the platform's terms.
Depending on the jurisdiction and product, this may include pickup, destination information, estimated pay, and expiry.
An estimated route or pay amount should be identified as such where it can change.
The workflow must support an accessible way to raise a safety concern or correct bad data.

Record offer creation, delivery to the device, response, and expiry as distinct events.
A notification delivery failure is not a courier rejection.
Do not infer that an unanswered offer was seen.

**Branch:** if the state changes before the offer is issued, release the reservation and replan.
If a courier declines or the offer expires, release the reservation and reconsider the order.

### Step 9: Execute and recover

**Input:** accepted assignment and subsequent operational events.
**Output:** pickup, delivery, reassignment, cancellation, or incident record.

On acceptance, confirm the current order state and record the event timestamp.
Provide clear pickup instructions and a channel for store and courier exceptions.
Track arrival, order readiness, pickup, and handoff independently.
Recalculate a customer ETA when material state changes, without presenting an estimate as a new guarantee.

If the store is not ready, apply the approved wait, compensation, and escalation rules.
If the courier cannot safely continue, provide a no-retaliation safety escalation path.
If the order is cancelled or cannot be fulfilled, reconcile the order and courier states before issuing another offer.
A handoff failure needs a documented chain of custody and customer contact procedure.

A recovery policy must define who can reassign and under what conditions.
Repeated reassignment creates both customer delay and courier burden.
When retry limits or promise thresholds are crossed, involve a human operator and communicate with the customer.

**Branch:** in an incident, pause unsafe automation and follow the operational runbook.
The goal is safe continuity and accurate state, not merely keeping the assignment rate high.

### Step 10: Reconcile outcomes and improve the policy

**Input:** decision snapshot, offer history, execution events, payments, complaints, and incidents.
**Output:** a linked decision record and actionable monitoring.

Join events by durable identifiers and preserve sequence and source timestamps.
A single order can have multiple offers and couriers; avoid joining it into multiple delivered orders.
Distinguish forecast error, decision quality, and execution failure.
A late delivery can arise from an unrealistic estimate, an avoidable assignment, store delay, or an unexpected event.

Report the denominator for every rate and state how missing events are treated.
Review service percentiles and late-tail rates, not just mean delivery time.
Report platform cost and courier earnings or waiting separately rather than compressing them into one score.
Investigate outcomes across relevant neighborhoods, shift times, and courier groups where collection and analysis are lawful.

An operations owner approves any production change to constraints or weights.
Analytics provides evidence and uncertainty; legal and worker representatives contribute where appropriate.
Document approval, version, effective time, rollback trigger, and communications.
Review incidents and appeals for harms not captured in the original objective.

## 6. Worked example

### Scenario and scope

At 18:00, a store has two ready orders and two available couriers.
All data below are fictional and for a single dispatch decision.
The currency is US dollars (USD); all travel and due-time figures are minutes after 18:00.
This example assumes each courier can take at most one order and that each order requires one courier.
No batching, future demand, or offer rejection is included in the base calculation.

| Order | Ready time | Due time | Delivery plus handoff | Load |
|---|---:|---:|---:|---:|
| A | 0 min | 14 min | 8 min | 3 kg |
| B | 0 min | 16 min | 10 min | 9 kg |

| Courier | Remaining safe capacity | Pickup travel to store | Status |
|---|---:|---:|---|
| C1 | 10 kg | 4 min | Available |
| C2 | 6 kg | 6 min | Available |

The order load is checked against remaining capacity, not the vehicle's advertised maximum.
The other eligibility conditions are assumed verified.
C2 cannot carry B, so edge C2-B is infeasible.
The customer commitment is the stated due time.
The estimated completion times are predictions, not commitments.

Assume courier pay per assignment is USD 4.00 for C1 and USD 3.50 for C2.
This simplified pay is an illustrative platform cost, not a proposed pay policy.
Add a service-risk proxy of USD 2.00 per predicted minute late.
No separate travel cost is added because the example assumes that courier pay already captures the modeled travel cost.
The deferral penalty is USD 15.00 per order for this one decision epoch.
This penalty represents a modeling assumption, not an actual fee or a measured loss.

### Calculate feasible edge costs

All orders are already ready, so pickup occurs on courier arrival.
For C1-A, completion is `max(4, 0) + 8 = 12 min`.
Its due time is 14 min, so predicted lateness is `max(0, 12 - 14) = 0 min`.
The score is USD 4.00 pay plus USD 0.00 late proxy, or USD 4.00.

For C2-A, completion is `max(6, 0) + 8 = 14 min`.
Its predicted lateness is `max(0, 14 - 14) = 0 min`.
The score is USD 3.50 pay plus USD 0.00 late proxy, or USD 3.50.

For C1-B, completion is `max(4, 0) + 10 = 14 min`.
Its predicted lateness is `max(0, 14 - 16) = 0 min`.
The score is USD 4.00 pay plus USD 0.00 late proxy, or USD 4.00.

For C2-B, 9 kg exceeds 6 kg remaining capacity.
The edge is excluded before scoring; it is not assigned a finite cost.

| Edge | Completion | Predicted lateness | Edge cost | Eligible? |
|---|---:|---:|---:|---|
| C1-A | 12 min | 0 min | USD 4.00 | Yes |
| C2-A | 14 min | 0 min | USD 3.50 | Yes |
| C1-B | 14 min | 0 min | USD 4.00 | Yes |
| C2-B | Not computed | Not computed | Not applicable | No: capacity |

### Compare complete decisions

Consider the nearest-first rule applied to order A first.
C1 is 4 min from the store and C2 is 6 min away, so the rule selects C1-A.
The only remaining courier cannot carry B.
With B explicitly deferred, the modeled total is USD 4.00 + USD 15.00 = USD 19.00.

The joint decision offers A to C2 and B to C1.
The modeled total is USD 3.50 + USD 4.00 = USD 7.50.
Both predicted completion times meet their respective commitments in the point forecast.
The difference is USD 19.00 - USD 7.50 = USD 11.50 in modeled objective value for this epoch.
It is not a measured cash saving: the USD 15.00 deferral penalty is an assumed proxy.

The full feasible decision set makes the comparison explicit:

| C1 action | C2 action | Deferred orders | Modeled total |
|---|---|---|---:|
| B | A | None | USD 7.50 |
| A | None | B | USD 19.00 |
| B | None | A | USD 19.00 |
| None | A | B | USD 18.50 |
| None | None | A and B | USD 30.00 |

No decision assigns C2 to B because that edge fails the capacity check.
No decision assigns both orders to the same courier under the single-order model.
The minimum modeled total is USD 7.50.

### What changes when the state changes

Suppose C1 rejects the B offer after one minute and C2 has accepted A.
The original plan is no longer executable.
Release C1's reservation, retain the accepted A assignment, and re-evaluate B with the current available pool.
C2's prior eligibility for A does not make C2 eligible for B, and an accepted assignment is not free capacity.
If no eligible courier remains, defer B, expand the search under policy, and notify the responsible operator.
The earlier USD 7.50 value describes the initial proposed decision, not the realized outcome after rejection.

Suppose instead B's true packed weight is 11 kg rather than 9 kg.
Neither listed courier is eligible because 11 kg exceeds both remaining capacities.
The safe decision is to defer B and locate appropriate capacity or apply a separately approved fulfillment alternative.
Do not relax load limits because the customer is near the due time.

Suppose the deferral penalty is only USD 2.00 per order while edge costs stay the same.
The formal objective chooses to defer both orders for USD 4.00.
That result follows from the inputs but is operationally unacceptable if the policy requires serving feasible committed orders.
The owner must set an appropriate deferral model or impose explicit service rules.
A solver cannot determine business priorities from arithmetic alone.

### What the case does and does not show

The case illustrates the opportunity cost of consuming a scarce feasible edge.
It does not prove that joint matching will always beat a sensible greedy heuristic in a real marketplace.
It excludes uncertainty, pay variability, correlated delays, cancellation, and future arrivals.
It also does not quantify fairness: two couriers and one decision cannot establish an equitable shift-level distribution.
A deployment needs longitudinal, privacy-aware evidence and a valid comparison policy.

## 7. Code, query, or calculation

The following standalone Python 3 program enumerates every feasible assignment for the worked example.
It uses only the standard library and is intended for small teaching instances.
It is not a production dispatch system or a scalable solver.

The code uses integer cents to avoid floating-point currency artifacts.
Times are whole minutes from 18:00 and loads are kg.
An order must be assigned to exactly one eligible courier or be explicitly deferred.
Each courier can receive at most one order.
The code computes the point-forecast lateness proxy described above.
It does not simulate acceptance, route uncertainty, concurrency, or future decisions.

```python
from itertools import product

# All example values are fictional. Time is minutes after 18:00.
orders = {
    "A": {"ready": 0, "due": 14, "delivery": 8, "kg": 3},
    "B": {"ready": 0, "due": 16, "delivery": 10, "kg": 9},
}
couriers = {
    "C1": {"capacity_kg": 10, "pickup": 4, "pay_cents": 400},
    "C2": {"capacity_kg": 6, "pickup": 6, "pay_cents": 350},
}
late_cents_per_minute = 200
defer_cents = 1500
order_ids = sorted(orders)
courier_ids = sorted(couriers)


def edge_cost(courier_id, order_id):
    """Return cost in cents, or None for an infeasible edge."""
    courier = couriers[courier_id]
    order = orders[order_id]
    if order["kg"] > courier["capacity_kg"]:
        return None
    completion = max(courier["pickup"], order["ready"]) + order["delivery"]
    lateness = max(0, completion - order["due"])
    return courier["pay_cents"] + late_cents_per_minute * lateness


def enumerate_plans():
    # One choice per order: a courier identifier or None for deferral.
    for choices in product([None] + courier_ids, repeat=len(order_ids)):
        selected = [c for c in choices if c is not None]
        if len(selected) != len(set(selected)):
            continue  # A courier cannot take two orders in this model.
        total = 0
        valid = True
        for order_id, courier_id in zip(order_ids, choices):
            if courier_id is None:
                total += defer_cents
                continue
            cost = edge_cost(courier_id, order_id)
            if cost is None:
                valid = False
                break
            total += cost
        if valid:
            yield total, choices


# Sort by cost and an explicit stable tie key for reproducibility.
plans = sorted(
    enumerate_plans(),
    key=lambda plan: (plan[0], tuple(c or "~" for c in plan[1])),
)
for total, choices in plans:
    actions = ", ".join(
        f"{order_id}={courier_id or 'DEFER'}"
        for order_id, courier_id in zip(order_ids, choices)
    )
    print(f"{actions}: USD {total / 100:.2f}")
```

Expected output, with equal-cost plans ordered by the documented tie key:

```text
A=C2, B=C1: USD 7.50
A=C2, B=DEFER: USD 18.50
A=C1, B=DEFER: USD 19.00
A=DEFER, B=C1: USD 19.00
A=DEFER, B=DEFER: USD 30.00
```

The enumeration cost grows exponentially with the number of orders.
For production-sized bipartite single-order matching, use an appropriate tested optimization implementation and enforce the same operational controls.
For a larger route-insertion problem, model sequencing and shared vehicle capacity explicitly.

### Modify and interpret

Change B's load to 11 kg.
The program should exclude both B edges; the best plan assigns A to C2 and defers B for USD 18.50.
Change the deferral cost to 200 cents while leaving the original loads.
The best plan defers both orders for USD 4.00, illustrating the importance of a well-specified objective.

The artifact's tie rule is deterministic, not a fairness policy.
Repeated deterministic tie-breaking can concentrate opportunity on particular identifiers.
A production design should study tie frequency, distributional impact, and an approved alternative while preserving reproducibility.

## 8. Data, standards, and regulatory context

### The minimum decision record

A useful record supports both live execution and later reconstruction.
It should include the decision identifier, UTC timestamp, local market and policy version.
For every order, keep its state, readiness provenance, commitment, load, and handling requirements.
For every considered courier, keep the minimally necessary availability, capacity, location age, and eligible region.
For each candidate edge, record feasibility, reason codes, predicted durations, uncertainty, and objective components.

Record which candidates were generated and why others were excluded where feasible.
An accepted offer alone cannot explain whether other couriers had a real opportunity.
Record offer sent, device delivery if available, response, expiry, reservation release, and any retry.
Record actual pickup, handoff, payment, adjustments, cancellation, and customer or courier complaints.
Associate manual overrides with an authorized actor and a reason.

| Data element | Quality question | Failure if absent | Operational response |
|---|---|---|---|
| Event timestamp | Is it source time, receipt time, or both? | Wrong sequence or leakage | Retain both with clock diagnostics |
| Location and age | Is it recent enough for this use? | False pickup estimate | Refresh or use approved fallback |
| Readiness status | Forecast or observed scan? | Unpaid waiting or late pickup | Store confirmation and alert |
| Capacity | Remaining and unit-consistent? | Unsafe load | Exclude unknown or excessive load |
| Reservation | Atomic and unexpired? | Double booking | Reconcile before offer |
| Offer outcome | Sent, seen, accepted, or timed out? | Misread worker behavior | Preserve distinct events |
| Handoff event | Does it mean arrival or completed delivery? | Incorrect service rate | Define canonical endpoint |
| Policy version | Which rules and weights applied? | Irreproducible decision | Persist version with snapshot |
| Pay adjustment | Is actual settlement linked? | Misstated economics | Reconcile payment ledger |
| Manual intervention | Who acted and why? | Hidden policy behavior | Require structured reason |

A replay should use only data available at the original decision time.
Actual arrival times can evaluate a forecast after the fact but cannot be used as if known to the original policy.
Event-time corrections should be preserved with their arrival time so the historical view remains reproducible.

### Location, privacy, and retention

Courier location is sensitive operational data even when collected for a legitimate service.
Specify the purpose, collection period, access roles, retention, and deletion process with the privacy owner.
Collect no more precision or history than is needed for the approved purpose and applicable law.
Separate operational access from analytics access; aggregate or de-identify where appropriate.
De-identification is not automatically anonymity, especially for repeated location traces.

The EU General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, sets obligations for covered personal-data processing, including principles such as data minimization and storage limitation in Article 5.
It also addresses certain solely automated decisions with legal or similarly significant effects in Article 22, subject to its scope and exceptions.
Whether a particular dispatch or account decision falls within that provision is fact-specific and requires legal review.
A human review label alone does not establish compliance if the review is not meaningful where required.

The EU Platform Work Directive, Directive (EU) 2024/2831, addresses platform work and includes provisions concerning algorithmic management.
Its application depends on national transposition, the provision, and the relevant effective dates.
Do not assume the directive itself supplies a uniform operating rule for every market.
Consult current local implementation and qualified counsel before deployment.

In the United States, worker classification, wage, expense, privacy, and automated-decision obligations vary by jurisdiction and circumstances.
Neither an offer-acceptance button nor a specific routing algorithm determines classification by itself.
Keep legal status, pay calculations, and applicable rules under local professional review.
Educational examples in this unit are not legal advice.

### Standards and operational conventions

GS1 identification standards can help identify products and locations across organizations where participants adopt them.
They do not by themselves guarantee that a courier can carry a basket or that a location scan is accurate.
A location identifier is not a substitute for a dispatchable entrance or current pickup instructions.

The NIST AI Risk Management Framework 1.0 is voluntary guidance for managing AI risks.
Its Govern, Map, Measure, and Manage functions provide a useful way to organize ownership and monitoring.
It does not certify a dispatch policy as fair or legally compliant.

The NIST Privacy Framework provides voluntary privacy-risk management guidance.
It can support purpose limitation, role design, and review of location-data use.
Its use does not replace obligations under applicable privacy law.

Use an unambiguous time representation such as ISO 8601 for exchange, with timezone and daylight-saving behavior defined.
For operational analysis, retaining UTC and the relevant local timezone avoids mixing repeated or missing local clock hours.
A timestamp format alone does not synchronize clocks or resolve delayed events.

### Data-quality gates

1. Check identifiers and mandatory fields before candidate generation.
2. Check that each order has at most one active accepted assignment unless a documented workflow permits otherwise.
3. Check that the courier's reservation and availability states do not conflict.
4. Check unit conversions for kg, distance, currency, and minutes.
5. Check the age and source of every prediction and location.
6. Reject impossible or contradictory timestamps for investigation, preserving raw events.
7. Compare expected and observed offer-response event counts.
8. Reconcile delivered orders with payment and customer records.
9. Track missingness by market and time, not only globally.
10. Limit access and retention according to approved policy and law.

Each gate needs an owner, threshold, alert channel, and safe response.
A quality dashboard that nobody is authorized to act upon is not an operational control.

## 9. Industry practice and case evidence

### Evidence boundaries

Public company material often describes broad platform behavior, not the exact production objective or its measured causal impact.
Use it to establish that a company reports a practice, not to infer hidden coefficients or a verified fairness outcome.
The academic studies below address related settings; their results do not automatically transfer to every market.

### DoorDash: reported offer information

DoorDash's Dasher help documentation describes information shown for delivery opportunities and its platform's offer workflow.
The precise information and rules can differ by market and change over time.
The documented point relevant here is that offer design is part of dispatch, not an afterthought after optimization.
A courier must be able to interpret a job offer using the information actually disclosed in the applicable market.

**Inference:** if displayed information omits an important source of waiting or travel burden, acceptance rates may be difficult to interpret.
That inference is not a finding about a particular DoorDash offer or its compliance.
For an audit, capture the offer as actually presented and distinguish estimated from settled pay.

### Uber: reported upfront information

Uber's driver-facing material describes upfront trip information in applicable locations and products.
Availability and displayed fields depend on local market and product terms.
The practice illustrates why assignment quality includes the usability and transparency of the offer to the worker.
It does not disclose the full matching algorithm or establish a universal earnings result.

**Inference:** a change to offer disclosure could change acceptance behavior even if the matching model is unchanged.
Evaluate the presentation and the matching policy together when assessing operational effects.
Do not attribute any observed change solely to a model without an appropriate comparison.

### Matching theory and the value of waiting

Özkan and Ward study dynamic matching in ride-hailing with abandonment and waiting in a stochastic setting.
Their analysis shows why a decision to match immediately can have an opportunity cost when future arrivals matter under the modeled assumptions.
Ride-hailing and food delivery differ in preparation, capacity, and customer commitments.
Their theoretical result is a framework for reasoning about dynamic matching, not a ready-made quick-commerce policy.

A short dispatch delay can allow a better joint match, but it also consumes customer slack and may increase courier idle time.
Measure both effects in the target market.
The best delay depends on demand intensity, supply availability, promise distribution, and uncertainty.

### Labor allocation and unequal opportunity

Research on algorithmic management documents that platform rules, monitoring, and information asymmetries can shape worker experience.
Rosenblat and Stark describe how information asymmetries affect drivers in ride-hailing.
That evidence supports examining the worker-facing process as well as a mathematical matching objective.
It does not prove a particular disparity or legal violation in the fictional example.

Audit offer exposure, expected and realized pay, travel to pickup, waiting, and cancellation separately.
A high acceptance rate can coexist with poor offers if workers have limited alternatives.
A low acceptance rate can result from infeasible or unattractive offers rather than poor worker performance.
A policy that penalizes declines can change the meaning of acceptance-rate comparisons.

### Capacity and dispatch in practice

A store selling groceries, prepared food, and bulky products has heterogeneous loads.
A bicycle, car, insulated bag, and cargo bike do not provide interchangeable capacity or handling conditions.
Equipment checks should reflect the real load and route, not just a generic vehicle category.

**Inference:** a marketplace with few high-capacity couriers can benefit from explicitly valuing that scarcity.
The magnitude depends on the future order mix and on whether scarce couriers actually accept the work.
The example in Section 6 demonstrates a single-snapshot mechanism, not an empirical estimate of savings.

### Translate evidence into a decision

| Evidence | Directly supports | Does not establish | Practical next step |
|---|---|---|---|
| Public offer documentation | A described worker-facing workflow | Hidden optimization weights | Capture local offer presentation |
| Dynamic matching research | Opportunity cost under modeled assumptions | A universal dispatch delay | Test delay with local guardrails |
| Algorithmic management research | Need to examine information and power | A violation by a specific platform | Include worker feedback and appeals |
| Internal operational logs | Observed outcomes under deployed policy | Untreated counterfactual outcomes | Design a comparison or experiment |
| Courier interviews | Reported experience and failure mechanisms | Representative prevalence by themselves | Combine with appropriate quantitative evidence |

### Experiment design

A courier can serve multiple orders, and one order's assignment changes the supply available for another.
This creates marketplace interference: order-level randomization can contaminate both variants.
Consider geographic or time-based switchback experiments when operationally suitable.
Randomize comparable market-time blocks and account for carryover from ongoing trips.
A washout interval may reduce but not automatically remove carryover.

Before launch, define the estimand, treatment unit, eligible population, observation window, and analysis method.
Use comparable weekdays and demand conditions where possible.
Record concurrent incentives, weather disruptions, holidays, and policy changes.
Choose a sample and duration appropriate to the expected variability and decision risk.
Do not repeatedly inspect results and stop opportunistically without a pre-agreed sequential procedure.

Primary service metrics may include on-time completion and a late-tail measure with all eligible orders in the denominator.
Economic metrics can include actual platform cost per completed and per requested order.
Worker metrics can include paid and unpaid wait, deadhead, offer exposure, and earnings per defined available hour.
Safety incidents, complaints, cancellations, and unserved orders are guardrails, not footnotes.

Predetermine stop conditions, named decision makers, and how the fallback is activated.
Report effect estimates with uncertainty and missing-data treatment.
A non-significant difference is not proof of equivalence or safety.
Analyze heterogeneous outcomes cautiously, with privacy protections and sufficient sample size.
Where experiments are infeasible, describe selection bias and use a credible quasi-experimental comparison if available.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Greedy resource capture | Heavy orders wait despite nearby couriers | Flexible work consumes scarce capacity | Deferral and lateness | Joint matching with capacity edges |
| Stale courier location | Pickup errors rise with location age | Old coordinates treated as current | Missed promise and extra travel | Freshness gate and refresh |
| Unready order | Couriers repeatedly queue at one store | Forecast readiness treated as observed | Worker wait and congestion | Separate readiness states and monitor |
| Double booking | Overlapping active offers to one courier | Non-atomic reservation or retry | Cancellations and failed service | Atomic reservation and idempotency |
| Missing load | Many orders show default zero kg | Missing field becomes safe value | Unsafe assignment | Unknown-state exclusion and correction |
| Low deferral penalty | Solver leaves feasible orders unserved | Objective rewards inaction | Customer harm | Explicit service rules and sensitivity |
| Infinite penalty | Unsafe edges appear attractive | Hard rules encoded as soft costs | Safety or legal breach | Remove ineligible edges before solving |
| Travel cost double count | Modeled savings exceed ledger evidence | Pay and identical travel cost both charged | Misleading decision | Separate cash and resource accounting |
| Selective candidate search | Sparse neighborhoods see few candidates | Radius or index misses plausible edges | Unequal service | Audit candidate coverage and expansion |
| Silent timeout | Unanswered offers counted as refusals | Delivery and response events conflated | Unfair performance inference | Log delivery, timeout, and decline separately |
| Reassignment loop | Many offers per completed order | Repeated failure without escalation | Delay and worker burden | Retry limits and human review |
| Biased ETA | Underprediction concentrated in a market | Access and traffic effects omitted | Persistent late-tail harm | Stratified calibration and buffers |
| Mean-only reporting | Average improves while complaints rise | Tail and subgroup losses hidden | Undetected service or welfare harm | Report distributions and guardrails |
| Pay mismatch | Estimated and settled pay diverge | Adjustments or wait excluded | Worker distrust and cost error | Reconcile offer to payment ledger |
| Fairness proxy misuse | A single score is called proof of fairness | Different burdens and contexts compressed | Hidden distributional harm | Multiple metrics and qualitative review |
| Deterministic tie bias | Same identifiers win repeated ties | Stable ordering concentrates opportunity | Unequal exposure | Audit and approve tie policy |
| Model leakage | Offline accuracy is implausibly high | Future events included in features | Deployment degradation | Point-in-time feature reconstruction |
| Experiment interference | Both variants move together unexpectedly | Shared couriers cross treatment units | Biased effect estimate | Market-time design and carryover analysis |
| Manual override opacity | Overrides cluster in particular shifts | Reasons are absent or unreviewed | Unreproducible outcomes | Structured override and audit |
| Solver timeout | Fallback rate spikes at peak | Candidate graph exceeds latency budget | Stale decisions and lost offers | Bound computation and use safe fallback |
| Privacy overcollection | Location history retained indefinitely | No purpose or retention limit | Privacy and security exposure | Minimize, restrict, and delete data |
| Unsafe incident response | Staff bypass load limits under pressure | Throughput target overrides safety | Injury or damaged goods | Non-overridable controls and escalation |

A red flag is an invitation to investigate, not proof of a single root cause.
For each alert, define a baseline, action threshold, owner, and response time before production.
Monitor both the alert rate and whether the response was completed.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test or respond |
|---|---|---|---|
| Snapshot represents current state | Feasibility uses observed state | Delayed location or reservation | Measure age and state conflicts |
| Loads are accurate | Capacity screening uses kg | Substitutions or packaging omitted | Compare packed weight with recorded weight |
| Single-order assignment | Simple one-to-one matching | Multi-order routes or existing cargo | Use route-insertion model and actual remaining capacity |
| Completion model is calibrated | Scores rely on predicted arrival | Rain, building access, or peak congestion | Backtest residuals and late tails by context |
| Readiness is known | Pickup wait is predictable | Store scanning delays or missing readiness | Compare scan, observed wait, and store reports |
| Pay model reflects policy | Edge cost represents expected payment | Incentives, minimums, or adjustments | Reconcile offer and settled ledger |
| Deferral penalty is meaningful | Enables comparison to waiting | Penalty does not reflect service commitments | Sensitivity analysis and owner approval |
| Offer is accepted | Proposed plan can execute | Declines, timeout, device failure | Track response and replan from fresh state |
| All relevant candidates are generated | Optimization sees alternatives | Radius, permissions, or index misses | Audit candidate recall and geographic coverage |
| One epoch is sufficient | Static objective is interpretable | Future demand and courier repositioning matter | Compare to rolling or dynamic policies |
| Fairness metrics are interpretable | Distributions can be compared | Selection, different availability, small groups | Condition carefully and combine with worker input |
| Experiment units are independent enough | Standard inference is meaningful | Shared couriers and neighboring markets | Use interference-aware design and uncertainty |
| Rules apply uniformly | Eligibility can be coded | Local legal and contractual differences | Version by jurisdiction with legal review |
| Courier can safely accept | Capacity is more than numerical | Fatigue, weather, accessibility, personal constraints | Offer safe refusal and incident channels |
| Recorded outcomes are complete | Metrics reflect actual work | Missing handoffs, pay corrections, cancellations | Reconcile sources and report missingness |
| Costs are additive | Edge scores can be summed | Shared incentives or network effects | Model shared terms or quantify approximation |

Assumptions are not hidden defaults.
A production launch should designate an owner for each material assumption and a review cadence.
If an assumption fails, either change the model, restrict its operating domain, or use the fallback.

## 12. Exercises

### Exercise 1 — State and control

**Time:** 15 minutes.
**Task:** An offer is sent at 12:00, delivered at 12:01, and times out at 12:02.
At 12:03 the same order is offered to another courier, who accepts.
Draw the state sequence for the order and both couriers.
Explain whether the first courier declined and when the reservation should be released.

**Deliverable:** a state diagram and a short event table.
**Acceptance criteria:** distinguish sent, delivered, timeout, release, and acceptance; do not infer a decline; prevent overlapping active reservations.

### Exercise 2 — Eligibility screen

**Time:** 15 minutes.
**Task:** An order weighs 8 kg and needs an insulated bag.
Courier P has 10 kg remaining capacity and no verified bag.
Courier Q has 7 kg remaining capacity and a verified bag.
Courier R has 9 kg remaining capacity and a verified bag, but their availability is stale.

**Deliverable:** an eligibility table and next action for each courier.
**Acceptance criteria:** exclude P and Q under the stated hard rules; hold R pending availability refresh; name the no-safe-candidate path.

### Exercise 3 — Joint matching arithmetic

**Time:** 20 minutes.
**Task:** Reproduce the worked example's five feasible total costs.
Then set the deferral penalty to USD 2.00 per order without changing edge costs.
Find the minimum and explain the policy issue.

**Deliverable:** a table of all feasible plans and a one-paragraph interpretation.
**Acceptance criteria:** USD 7.50 is the original minimum; USD 4.00 for two deferrals is the revised minimum; identify the penalty as a modeled value.

### Exercise 4 — Modify the executable artifact

**Time:** 20 minutes.
**Task:** Set B's weight to 11 kg in the Python example.
Calculate the new best plan and explain why a lower cost cannot justify assigning an overloaded courier.

**Deliverable:** the changed input and expected best output.
**Acceptance criteria:** A goes to C2, B is deferred, total USD 18.50; infeasible edges remain absent rather than receiving a soft penalty.

### Exercise 5 — Fairness and denominator critique

**Time:** 25 minutes.
**Task:** Group X receives 120 offers over 60 available hours; group Y receives 90 offers over 60 available hours.
Group X accepts 60 offers and group Y accepts 54.
Assume these group labels can be used lawfully and safely, and the numbers are aggregated.

**Deliverable:** offer rates, acceptance rates, and a critique of any claim that one group is treated fairly or unfairly.
**Acceptance criteria:** compute 2.0 versus 1.5 offers per available hour and 50% versus 60% acceptance; discuss offer quality, context, selection, uncertainty, and privacy.

### Exercise 6 — Design the live recovery loop

**Time:** 25 minutes.
**Task:** Design a process for an accepted order whose store is not ready when the courier arrives.
Include worker communication, compensation policy reference, customer ETA update, and escalation.

**Deliverable:** a state diagram and owner/action table.
**Acceptance criteria:** distinguish forecast from commitment; preserve safety; specify wait threshold, reassignment conditions, and durable events without inventing legal pay rules.

### Exercise 7 — Evaluate a new policy

**Time:** 30 minutes.
**Task:** A team proposes replacing nearest-first with joint matching across a city.
Shared couriers work throughout the day and adjacent zones interact.
Design an evaluation that accounts for marketplace interference and worker effects.

**Deliverable:** a one-page experiment brief.
**Acceptance criteria:** name unit, comparator, assignment, primary metrics, worker and safety guardrails, carryover treatment, uncertainty, stop rule, and accountable owner.

### Exercise 8 — Deployment review

**Time:** 25 minutes.
**Task:** A prototype matches on stale coordinates, uses an unversioned cost score, and has no override log.
It reports a 10% reduction in modeled cost from historical replay.
Provide a go/no-go recommendation and remediation plan.

**Deliverable:** a decision memo with prioritized controls.
**Acceptance criteria:** distinguish modeled from realized cost; identify leakage and counterfactual limits; require freshness, versioning, event logging, fallback, and staged evidence.

## 13. Self-check answers

### Exercise 1

The first offer moves from created to delivered to timed out.
A timeout is not evidence of an explicit decline or even that the courier read the notification.
The reservation is released when the timeout is safely committed under the system's concurrency rules.
The order can then be offered to the second courier, whose acceptance creates an accepted assignment.
A late first response must be rejected or reconciled according to the defined expiry and concurrency policy.

A strong diagram identifies the order's offer and assignment states separately from each courier's availability.
It records the timestamps and idempotent transition identifiers.
It does not permit both couriers to have live accepted assignments for the same single-order job.

### Exercise 2

| Courier | Decision | Reason | Next action |
|---|---|---|---|
| P | Not eligible | Required insulated bag is not verified | Verify equipment before reconsidering |
| Q | Not eligible | 8 kg exceeds 7 kg capacity | Find suitable capacity |
| R | Unknown; do not issue yet | Availability is stale | Refresh state and re-screen |

If R cannot be verified promptly, there is no verified safe candidate in the listed pool.
Defer and expand the search or escalate under policy.
A forecast of imminent lateness does not convert an unverified bag or insufficient capacity into eligibility.

### Exercise 3

| A action | B action | Original penalty USD 15 each | Revised penalty USD 2 each |
|---|---|---:|---:|
| C2 | C1 | USD 7.50 | USD 7.50 |
| C2 | Defer | USD 18.50 | USD 5.50 |
| C1 | Defer | USD 19.00 | USD 6.00 |
| Defer | C1 | USD 19.00 | USD 6.00 |
| Defer | Defer | USD 30.00 | USD 4.00 |

The original minimum is USD 7.50, with C2-A and C1-B.
The revised minimum is USD 4.00, with both orders deferred.
The result is correct for the revised mathematical objective but may violate the business's service obligations.
Set a defensible deferral value or add explicit approved service constraints.

### Exercise 4

C1's 10 kg capacity and C2's 6 kg capacity are both below B's revised 11 kg load.
The best valid plan assigns A to C2 for USD 3.50 and defers B for USD 15.00.
The total is USD 18.50.
A numeric cost cannot make an unsafe edge eligible.

### Exercise 5

Group X receives `120 / 60 = 2.0` offers per available hour.
Group Y receives `90 / 60 = 1.5` offers per available hour.
Group X accepts `60 / 120 = 50%` of offers; group Y accepts `54 / 90 = 60%`.
These descriptive values do not establish why the rates differ or whether either group's experience is fair.

Compare the timing, locations, travel burden, estimated and realized pay, and feasibility of offers.
Examine availability composition, selection into groups, sample size, and uncertainty.
Protect privacy and seek appropriate legal review of sensitive attributes.
Include courier feedback; avoid turning an acceptance rate into a worker quality score without evidence.

### Exercise 6 rubric

A strong design records arrival, observed readiness, communications, wait, and subsequent action separately.
It identifies who sets the threshold and who can reassign.
It references approved compensation and safety policies without inventing a jurisdiction's rules.
It updates the ETA as a forecast and escalates when a customer commitment is at risk.
It releases and reassigns only through an auditable state transition.

A weak design either leaves the courier waiting indefinitely or automatically cancels without considering pay, safety, and customer communication.

### Exercise 7 rubric

A strong brief proposes comparable market-time blocks or another defensible interference-aware design.
It specifies randomization, duration, carryover handling, and the unit used for uncertainty estimates.
It includes on-time and late-tail service, unserved orders, actual cost, courier earnings and waiting, and safety.
It assigns a decision owner and defines rollback thresholds before treatment begins.
It states how concurrent incentives and missing data will be handled.

A weak brief randomizes individual orders as if they did not compete for shared couriers and reports only modeled savings.

### Exercise 8 rubric

The appropriate immediate recommendation is no-go for unattended deployment on the stated evidence.
Historical replay provides modeled comparisons using assumptions; it does not observe what would have happened under the alternative policy.
Point-in-time reconstruction is needed to avoid future-data leakage.

First establish freshness gates, hard safety constraints, policy versioning, reservation controls, override logging, and a fallback.
Then validate predictions and operational data, conduct shadow evaluation, and stage an interference-aware live test with guardrails.
The actual launch decision belongs to the accountable operations and governance owners.
A precise percentage in a replay is not proof of a realized cost reduction.

## 14. Interview questions

1. **Why can nearest-first be worse than joint matching?**
   A strong answer explains scarcity of feasible edges and compares complete assignments.
   A weak answer assumes the closest courier is always best for every order.
2. **How do you represent a courier who cannot carry an order?**
   A strong answer excludes the edge using remaining capacity and handling requirements.
   A weak answer assigns a large but finite penalty that can be outweighed.
3. **How do you handle an order with no eligible courier?**
   A strong answer explicitly defers, expands search within policy, and escalates with customer communication.
   A weak answer silently drops it from metrics.
4. **What is the difference between an offer and an accepted assignment?**
   A strong answer describes reservation, notification, acceptance, timeout, and concurrency.
   A weak answer treats a sent notification as acceptance.
5. **How would you score lateness?**
   A strong answer defines endpoint, forecast uncertainty, loss function, and units.
   A weak answer treats predicted late minutes as guaranteed actual loss.
6. **How can a cost model double count courier time?**
   A strong answer separates cash pay from any additional economic or welfare valuation.
   A weak answer adds identical time costs without explaining the accounting perspective.
7. **What would you measure for courier fairness?**
   A strong answer covers offer quality, exposure, earnings, deadhead, wait, denominators, privacy, and context.
   A weak answer treats a single acceptance-rate gap as conclusive proof.
8. **Why might order-level randomization be misleading?**
   A strong answer explains shared courier supply, interference, and possible market-time designs.
   A weak answer assumes each order's outcome is independent of every other assignment.
9. **What happens when the solver times out?**
   A strong answer uses a feasible, monitored fallback that preserves hard constraints.
   A weak answer bypasses validation to maintain throughput.
10. **How would you audit a historical replay?**
    A strong answer checks point-in-time inputs, decision coverage, behavioral assumptions, and counterfactual limitations.
    A weak answer equates an offline objective score with observed business profit.
11. **What is a defensible operational override?**
    A strong answer names authorized roles, reason codes, non-overridable constraints, and review.
    A weak answer allows unlogged exceptions to any rule.
12. **How do you protect worker location data?**
    A strong answer describes necessity, access controls, retention, purpose, and applicable legal review.
    A weak answer retains all movement indefinitely because it might later be useful.

## 15. Key terms

- [Assignment](#4-definitions-and-vocabulary): operational association after the relevant acceptance process.
- [Availability](#4-definitions-and-vocabulary): current ability to receive and perform work.
- [Candidate edge](#step-3-generate-candidate-edges): possible courier-order pairing considered by the policy.
- [Capacity](#step-4-apply-hard-feasibility-constraints): remaining safe load and handling capability.
- [Commitment](#1-why-this-topic-exists): customer-facing promise, distinct from an ETA.
- [Counterfactual](#experiment-design): outcome under an alternative that was not directly observed.
- [Deadhead](#4-definitions-and-vocabulary): travel without the customer order, including pickup approach.
- [Decision epoch](#step-1-define-the-decision-and-its-horizon): time at which a matching decision is made.
- [Deferral](#step-7-solve-the-joint-allocation): explicit decision to reconsider an order later.
- [Eligibility](#step-4-apply-hard-feasibility-constraints): satisfaction of every applicable hard constraint.
- [ETA](#step-5-predict-pickup-and-completion): forecast arrival or completion at a defined endpoint.
- [Exposure](#exercise-5--fairness-and-denominator-critique): relevant offers made to a courier or group.
- [Fairness](#labor-allocation-and-unequal-opportunity): specified distributional concern evaluated in context.
- [Fallback](#step-7-solve-the-joint-allocation): approved safe behavior when the primary solver fails.
- [Hard constraint](#three-decision-layers): non-tradable eligibility rule.
- [Idempotency](#step-8-reserve-and-issue-offers-safely): protection against duplicate effects from retries.
- [Interference](#experiment-design): treatment of one unit affecting another unit's outcome.
- [Lateness](#step-5-predict-pickup-and-completion): positive difference between completion and due time.
- [Matching](#3-the-core-mental-model): joint selection of compatible courier-order edges.
- [Objective](#step-6-build-an-interpretable-incremental-cost): function comparing feasible assignment sets.
- [Offer](#step-8-reserve-and-issue-offers-safely): proposed work with disclosed terms and expiry.
- [Opportunity cost](#compare-complete-decisions): value of an alternative displaced by a decision.
- [Override](#step-10-reconcile-outcomes-and-improve-the-policy): authorized, logged departure from a recommendation.
- [Readiness](#step-2-capture-a-consistent-state): observed or forecast state of an order at pickup.
- [Reservation](#step-8-reserve-and-issue-offers-safely): temporary protection against conflicting offers.
- [Slack](#step-5-predict-pickup-and-completion): due time minus predicted completion.
- [Snapshot](#step-2-capture-a-consistent-state): timestamped inputs visible to the decision.
- [Timeout](#exercise-1--state-and-control): expiry without a valid response, not necessarily a refusal.
- [Unserved penalty](#step-6-build-an-interpretable-incremental-cost): modeled consequence of deferring an order.
- [Utilization](#4-definitions-and-vocabulary): activity divided by a precisely defined availability denominator.

## 16. Sources

Sources support the specific claims stated above; they do not validate the fictional numerical example.
Access and local applicability should be checked when a policy is deployed.

1. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Articles 5 and 22. Binding applicability depends on the facts and jurisdiction.
2. European Union, [Directive (EU) 2024/2831 on improving working conditions in platform work](https://eur-lex.europa.eu/eli/dir/2024/2831/oj). Review national transposition and dates rather than assuming identical local obligations.
3. National Institute of Standards and Technology, [AI Risk Management Framework: AI RMF 1.0](https://doi.org/10.6028/NIST.AI.100-1), 2023. Voluntary risk-management guidance.
4. National Institute of Standards and Technology, [Privacy Framework](https://www.nist.gov/privacy-framework). Voluntary privacy-risk guidance.
5. GS1, [Global Location Number](https://www.gs1.org/standards/id-keys/gln). Identification standard, not a dispatch safety certification.
6. ISO, [ISO 8601 date and time format](https://www.iso.org/iso-8601-date-and-time-format.html). Exchange-format overview; timestamp semantics still need an operational definition.
7. Erhun Özkan and Amy R. Ward, “Dynamic Matching for Real-Time Ride Sharing,” *Stochastic Systems*, 2020. [Author manuscript record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2844451). Theoretical ride-hailing setting, not a measured quick-commerce deployment.
8. Alex Rosenblat and Luke Stark, “Algorithmic Labor and Information Asymmetries: A Case Study of Uber's Drivers,” *International Journal of Communication*, 2016. [Published article](https://ijoc.org/index.php/ijoc/article/view/4892). Qualitative platform-work evidence; context matters.
9. DoorDash, [Dasher Central](https://dasher.doordash.com/). Company information about the Dasher experience; consult current market-specific offer terms.
10. Uber, [Upfront fares](https://www.uber.com/us/en/drive/driver-app/upfront-fares/). Company description of driver-facing information in applicable markets; availability may vary.

## 17. Links

- [Module 22 index](./_index.md).
- [Module 22 survey and lab sections](../../../course/22_quick_commerce_on_demand_delivery.md).
- [Courier supply and demand](./05_courier-supply-demand.md).
- [ETA prediction](./06_eta-prediction.md).
- [Batching and routing](./07_batching-routing.md).
- The module survey links its relevant practical exercises and datasets; use those materials to extend the example without assuming that their data describe this fictional case.
