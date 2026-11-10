---
case_id: "CASE-2026-OR001"
case_title: "Last-mile routing optimisation"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Logistics and last-mile parcel delivery"
primary_pattern: "constrained vehicle routing with time windows"
secondary_patterns: ["constructive savings heuristic", "independent feasibility validation", "cost break-even analysis"]
evidence_level: "illustrative"
source_basis: "constructed teaching instance; every demand, distance, duration and cost figure is ILLUSTRATIVE"
as_of_date: "2026-11-10"
owner_role: "Operations research analyst supporting depot dispatch"
source_index: "INDEX.md"
review_trigger: "Revisit when fleet size, shift length, break rules or the customer promise window policy change"
---

# Last-mile routing optimisation

> **Evidence disclaimer.** This case is a teaching construction. No carrier, depot, driver, route or
> measured operational result is asserted. The stop set, the travel matrix, the fleet and the cost
> coefficients are ILLUSTRATIVE values chosen so that the arithmetic is checkable by hand. Every
> number printed below can be re-derived from the inputs stated in section 5; none of them is a
> reported outcome of a real deployment. Statements about what *would* be measured are marked
> `not-measured`, and design elements are marked as proposed rather than built.

## 1. The One-Line Framing

Given a dated set of delivery stops with parcel counts and service windows, choose an assignment of
stops to vans and a visiting order per van that is feasible against vehicle capacity, driver shift
length and customer windows, and that reduces total driven distance relative to the dispatch team's
manual geographic batching — with feasibility checked before any distance number is believed.

**Takeaway:** a distance saving is worthless if the route that produced it cannot be driven inside
the shift; the acceptance gate is feasibility, and the objective is read only after that gate passes.

The framing matters because it fixes what "better" means. A router that returns a schedule 12%
shorter but with two vans over capacity has not improved the operation; it has moved the failure
from the planning conversation to the loading bay, where it is more expensive to discover. The case
therefore treats the routing problem as a two-stage decision: first decide which plans are legal,
then rank the legal plans.

## 2. Context & Background

Last-mile delivery is the segment of a parcel network between the final depot and the recipient's
door. Its cost structure is dominated by labour and vehicle time rather than by line-haul distance,
which is why depot planners care about the number of routes, the hours each route consumes, and the
distance each route covers. In the illustrative operation modelled here, a single depot serves a
dense urban area; the depot's day begins with a pre-load window and ends when every van is back
inside the shift limit.

The planning process being replaced is deliberately described as competent rather than careless.
Dispatchers partition the service area into postal sectors, assign sectors to vans, and order stops
inside a sector by street sequence from memory and from the driver's habitual behaviour. This
produces plans that are usually feasible and that drivers trust, because the sequence is stable
from day to day. The weakness is not competence but scale: with roughly six hundred stops per day
and a handful of tight customer windows, no human can evaluate the interaction between capacity
packing and window sequencing across the whole day.

The operation has four pressures that collide. First, parcel volume per stop is uneven: business
addresses take multi-parcel drops, residential addresses take one or two, so "stops per van" is a
poor proxy for "work per van". Second, a subset of customers buys a morning window, and those
windows are contracts rather than preferences. Third, the driver shift is a hard limit with a
mandatory break, and overtime is permitted only through an approval path that dispatch does not
control in real time. Fourth, the depot has a fixed number of vans, so the plan must either fit the
fleet or escalate a shortfall before the loading bay opens.

The business context also includes an audit obligation. Any plan that dispatch overrides must be
explainable later: who changed it, at what time, for which stop, and under which reason code. This
is not bureaucracy for its own sake. Without an override record, a systematic defect in the router
(consistently under-estimating unloading time at a particular building, say) looks identical to
dispatcher preference, and the defect survives every review.

Two background facts shape the modelling choices. Travel time in the service area is short per leg —
the median leg is a few minutes — so the objective is dominated by the number of legs and by
waiting, not by long hauls. And the matrix of travel times is measured on a different day from the
day being planned, so it carries error that is correlated within a peak period. Both facts push the
design away from exact distance minimisation and toward robustness: a plan should be good and its
slack should be visible.

The organisational question underneath the technical one is about trust. Dispatch will not adopt a
sequence it cannot inspect, and drivers will not follow a sequence that ignores a building's access
rules. The case therefore spends as much attention on the validator, the override record and the
fallback plan as on the optimiser itself, because those are the parts that make the optimiser
usable.

## 3. The Business Question

The primary decision question is: for a given dispatch date and a fixed fleet, which assignment of
stops to vans and ordering within each van minimises total driven distance subject to capacity,
shift, break and window feasibility, and how much of the resulting distance is achievable by
re-sequencing alone before the fleet must grow?

Five subsidiary questions make that operational.

1. Which constraint actually binds on a typical day — capacity, shift time, or window timing? The
   answer determines whether the useful lever is fleet size, route design, or promise management.
2. What is the smallest fleet that can serve the day legally, and how much slack does that fleet
   leave per route? A fleet that is legal but with no slack will fail on the first traffic incident.
3. How much distance does re-sequencing save without changing the number of vans, and how much
   additional saving requires a fourteenth van at a fixed daily cost?
4. Which stops are at risk of lateness under the plan, and what does dispatch do with that list
   before 07:00 rather than after 10:00?
5. What evidence would show, after the fact, that a plan was legal when it was issued and that any
   deviation was recorded rather than absorbed silently?

The success criteria are stated in decision terms. A plan is acceptable when an independent checker
finds zero constraint violations, when the fleet count equals the computed lower bound or the gap is
explained, when the lateness-risk list is produced before dispatch, and when every override carries
a reason code from a closed taxonomy. Distance reduction is a secondary criterion, reported against
the manual baseline for the same stop set and the same date.

Explicit non-goals keep the scope honest. The case does not model dynamic re-routing after dispatch,
does not optimise the depot's inbound line-haul, and does not attempt to learn customer behaviour
from delivery history. Those are separate decisions with separate data requirements; folding them
in would make the feasibility argument unverifiable.

## 4. Assumptions & Constraints

The assumptions are listed with their units and the direction of the error they introduce if wrong.
A planner who cannot state the direction of the error cannot set slack intelligently.

| # | Assumption | Value used (ILLUSTRATIVE) | Units | If wrong, error direction |
|---|---|---|---|---|
| A1 | Stops to serve on the date | 620 | stops/day | understated volume hides a capacity breach |
| A2 | Parcels per stop, distribution | 140 stops at 4, 200 at 2, 280 at 1 | parcels/stop | mean-only planning understates packing difficulty |
| A3 | Van capacity | 100 | parcels/van | overstatement produces an undeliverable load |
| A4 | Vans available | 12 owned, 1 rental approved | vans | rental lead time is the real constraint |
| A5 | Shift length including break | 480 | minutes/driver | overstatement converts lateness into overtime |
| A6 | Mandatory break | 30, taken after 4 h driving | minutes | omitted break makes every long route infeasible |
| A7 | Pre-load and vehicle check | 22 | minutes/van | underestimation delays every departure |
| A8 | Service time per stop | 2.4 mean, 6.0 at multi-parcel business sites | minutes/stop | mean-only service time biases toward dense routes |
| A9 | Mean travel time per leg | 2.9 | minutes/leg | stale peak-hour matrix understates afternoon legs |
| A10 | Mean leg distance | 1.16 | km/leg | distance and time diverge when the matrix is congested |
| A11 | Customer windows | 22 stops in 09:00–11:00 cluster | clock time | closed/half-open misreading flips boundary stops |
| A12 | Cost of a van-day | 118 | currency unit/van-day | fixed cost dominates the van-count decision |
| A13 | Distance cost | 0.46 | currency unit/km | small relative to labour; do not optimise it alone |
| A14 | Driver cost | 27 | currency unit/hour | paid hours are the largest single cost block |
| A15 | Overtime rate | 40.5 (1.5× base) | currency unit/hour | overtime is only cheap below a break-even |

Hard constraints are those that make a plan illegal if violated, and they are checked before the
objective is computed.

```text
CONSTRAINT  C1  capacity        sum of parcels assigned to van v <= 100        for every v
CONSTRAINT  C2  shift           preload + travel + service + waits + break <= 480 min
CONSTRAINT  C3  break placement the 30-minute break starts no later than 240 min into duty
CONSTRAINT  C4  window          service start s_j satisfies e_j <= s_j <= l_j  for windowed stop j
CONSTRAINT  C5  assignment      every stop in the input is assigned to exactly one van
CONSTRAINT  C6  single depot    every route starts and ends at the depot
```

Soft constraints are those the plan may trade against distance but must report: exceeding the
promise buffer by less than 15 minutes, visiting a stop outside its preferred (not contractual)
time band, and using the rental van. The reason these are separated matters: if a soft constraint
is silently enforced as hard, the solver returns "infeasible" on days where a legal plan exists with
a small service concession, and dispatch loses confidence in the tool.

Three constraints are outside the model and must be handled by process rather than by the solver.
Driver familiarity with a building's loading dock cannot be expressed as a linear constraint and is
captured as an override reason. Vehicle height and weight restrictions are captured as a stop
attribute that removes infeasible arcs from the matrix before the solve. Union agreement on shift
length is not negotiable per day, so the shift limit is a parameter, not a decision variable.

The constraint hierarchy is the part practitioners get wrong most often. The case orders it as:
legality (C1–C6) beats service quality, service quality beats distance, and distance beats route
stability. Reversing the first two produces plans that are cheap and late; reversing the last two
produces plans that change every day and that drivers stop following.

## 5. Data, Tools & Method

### 5a Data Specification

The grain of each input table is stated because joins at the wrong grain are the most common source
of capacity breaches that survive review.

| Table | Grain (one row per) | Key | Units / domain | Used for |
|---|---|---|---|---|
| `stops` | delivery_date × stop_id | (delivery_date, stop_id) | count, clock time | population of the routing problem |
| `stop_demand` | delivery_date × stop_id × parcel_class | composite | parcels, kg, m³ | capacity constraint C1 |
| `windows` | delivery_date × stop_id | composite | clock time, half-open | constraint C4 |
| `matrix` | matrix_version × origin × destination | (version, i, j) | minutes, km | travel time and distance |
| `vans` | fleet_date × van_id | composite | parcels, m³, height m | capacity and access restrictions |
| `shifts` | roster_date × driver_id | composite | minutes, clock time | constraints C2 and C3 |
| `promises` | order_id | order_id | clock time | comparator for lateness, not a constraint |

The decision cut-off is 06:30 on the delivery date. Everything used by the model must be knowable
at that instant: the stop list frozen at 06:00, the parcel counts confirmed by the 05:40 scan, the
roster published the previous Thursday, and the travel matrix refreshed from the 05:00 probe run.
Anything that becomes known later belongs to the evaluation, not to the plan.

What is deliberately not available at decision time, and must never be used as a feature:

- realised arrival times for today's stops (they are the outcome being predicted);
- today's traffic incidents and road closures discovered after 06:30;
- delivery outcomes such as refusal, absence or damage, which determine whether a second attempt
  is needed tomorrow;
- next-day order volume, which would let the planner pre-position work across days;
- driver-reported access notes typed after the route was issued.

The stop population is defined by three filters, each of which changes the denominator of every
downstream metric: stops with a confirmed parcel on the manifest, stops inside the depot's service
polygon, and stops not already flagged as held at customer request. A plan built on an unfiltered
list will look 3–5% better on distance per stop simply because it serves stops that should not have
been in the population at all.

Units are declared per field because two of them are the source of real errors. Capacity is
declared in parcels, but vans are also limited by volume in cubic metres and by weight; a plan that
respects only the parcel count can still be unloadable when a business stop takes forty bulky
cartons. Service time is declared in minutes per stop as a mean, but the mean hides a bimodal
distribution: residential drops take about 1.6 minutes, business drops with signature and trolley
handling take about 6.0 minutes.

### 5b Tools & Stack

The stack is described as a set of responsibilities rather than as products, because the case is
about method selection and every component has a cheap and an expensive implementation.

| Component | Responsibility | Chosen implementation (proposed) | Why not the alternative |
|---|---|---|---|
| Input assembler | freeze stops, demand, windows at cut-off | scheduled job writing a versioned snapshot | direct database reads make the plan unreproducible |
| Matrix service | travel time and distance per arc | cached matrix keyed by version + profile | live per-query calls are slow and inconsistent |
| Constructor | build an initial feasible solution | Clarke–Wright savings with capacity and window checks | random insertion gives worse starts and no audit trail |
| Improver | reduce distance inside feasibility | local search: 2-opt, relocate, or-opt with penalty for lateness | exact MIP does not close 620 stops with windows in the time box |
| Validator | independently re-check C1–C6 | separate implementation of the constraint arithmetic | using the solver's own checker reuses the same bug |
| Sequencer | order the route under windows | forward slack recursion plus push-forward repair | pure distance ordering ignores waiting |
| Dispatcher console | accept, edit, override with reason | ordered list plus reason taxonomy | free-text overrides cannot be aggregated |
| Audit log | immutable record of plan and overrides | append-only event stream | mutable tables destroy the evidence trail |

The solver choice is a deliberate rejection of optimality. A mixed-integer formulation of this
instance, with 620 stops, time windows and a heterogeneous fleet, is solvable in principle and not
solvable inside a fifteen-minute planning window with a certificate that dispatch can read. The
practical requirement is a feasible plan of known quality before 07:00, plus an honest statement of
the gap. A savings constructor followed by local search delivers that; the certificate is the
validator, not a dual bound.

The fallback is part of the stack, not an afterthought. If the solver fails, times out, or returns a
plan the validator rejects, dispatch uses the approved manual base plan for that date: the
sector-based assignment produced by the planning team, signed off, with its known imperfections.
Having a named fallback prevents the worst failure mode in this domain, which is dispatching nothing
because the optimiser is unavailable while the vans are already loaded.

### 5c Method Summary

The method runs in seven stages, each with an explicit output artefact so that a failed stage is
visible rather than absorbed.

1. **Freeze.** Read the 06:00 snapshot, apply the population filters, and write a plan request with
   a content hash covering stops, demand, windows, fleet and matrix version.
2. **Bound.** Compute the capacity lower bound on vehicles and the time lower bound, and compare
   both with the fleet size. This stage decides whether the day is legal at all before any
   optimisation happens.
3. **Construct.** Build routes with Clarke–Wright savings, accepting a merge only when the merged
   route passes capacity, window and shift checks at every intermediate stop.
4. **Improve.** Apply local search moves with a lexicographic objective: infeasibility first,
   then total distance, then route instability. Moves that fix a violation are always accepted;
   moves that reduce distance are accepted only if feasibility is preserved.
5. **Validate.** Re-check every route with an independent implementation, reporting violations by
   code rather than by message.
6. **Rank risk.** Compute slack per stop and per route, and emit the lateness-risk list ordered by
   the smallest slack, so dispatch can intervene on the few stops that matter.
7. **Publish.** Write the plan, the validator report and the risk list into the audit log with the
   request hash; only then release to the driver application.

The improver's acceptance rule is the piece that carries most of the engineering judgement. A
pure distance objective will happily move a stop from a morning window into an afternoon slot if
the total kilometres fall, because the window constraint only binds when the stop is windowed. The
case therefore scores plans as a tuple, not a scalar, and refuses to compare plans on distance
alone. This is why the objective is described as lexicographic rather than weighted: a weight is a
licence to trade, and the operation does not want that trade available to the solver.

### 5d Provenance, Quality, Exclusions, Leakage

Quality is quantified with four checks, each expressed as a ratio with a stated denominator, so a
degraded input produces a number rather than an opinion.

```text
FORMULA      geocode_match = matched_stops / total_stops
WHERE        matched_stops = stops whose geocode falls inside the service polygon and within 40 m of the address point
CONSTRAINT   geocode_match >= 0.995 to publish; below 0.98 the plan is withheld
NUMERICAL EXAMPLE  621 matched of 620 stops is arithmetically impossible, so the honest instance is 617/620 = 0.9952
INTERPRETATION  three unresolved geocodes sit inside the tolerance, not outside the publication gate
```

The near-miss in that example is the point: a match rate above 1.0 is a join defect, not excellence.
Duplicate `stop_id` rows inflated the numerator, and the correct response is to de-duplicate at the
grain stated in 5a before computing any ratio.

```text
FORMULA      matrix_completeness = arcs_present / (n_stops + 1)^2
WHERE        arcs_present counts distinct (origin, destination) pairs with a finite duration in the frozen version
CONSTRAINT   completeness must be 1.000 for the frozen version; missing arcs are a hard stop
NUMERICAL EXAMPLE  (620 + 1)^2 = 386,641 ordered pairs including the depot
INTERPRETATION  38 missing arcs is a 0.0098% hole that still makes one route undriveable
```

Demand reconciliation is the third check and the one that most often explains a capacity breach.
The manifest gives parcels per stop; the customer record gives cases; the loading sheet gives
weight. The plan uses parcels, so the reconciliation converts cases to parcels with a declared
factor per parcel class and flags any stop where the conversion is ambiguous instead of guessing.

Exclusions are recorded rather than silent. Stops held at customer request, stops with an
incomplete address, and stops whose window is entirely outside the shift are excluded from the
optimisation and reported as an explicit count on the plan. The count matters because the distance
per stop metric uses the planned population as its denominator; changing that population changes
the metric without changing the operation.

Leakage appears in three distinct places and each needs its own guard.

- **Feature leakage:** using today's realised travel times, or yesterday's actual route durations on
  the same streets, as inputs to today's plan. Guard: the plan request accepts only matrix versions
  dated at or before the cut-off.
- **Evaluation leakage:** tuning savings parameters and local-search weights on the same days used
  to report performance. Guard: a chronological split, tune on weeks 1–3, report on week 4.
- **Population leakage:** including stops that were excluded from the plan when computing plan
  quality, which flatters the result because the hardest stops were excluded by construction.
  Guard: the metric denominator is the published plan's stop list, and the exclusion count is
  reported beside it.

### 5e Reproducibility & Method-Selection Ledger

Reproducibility here means a reviewer can regenerate the same plan from the same inputs and can
see why each method choice was made.

| Decision | Alternatives considered | Reason chosen | Reversal cost |
|---|---|---|---|
| Savings constructor over random insertion | random insertion, sweep, nearest neighbour | deterministic, auditable merge order, naturally respects capacity | low: constructor is one module |
| Local search over exact MIP | MIP with time limit, column generation | planning window is 15 min and the instance has 620 stops | high: certificate and runtime both change |
| Lexicographic objective | weighted sum of distance and lateness | removes the solver's licence to trade service for km | medium: weights would need calibration |
| Independent validator | reuse solver feasibility check | a shared implementation shares its bugs | low: validator is arithmetic, not optimisation |
| Frozen matrix per version | live matrix per query | reproducible plans and comparable days | medium: refresh cadence becomes an ops task |
| Approved manual base as fallback | no dispatch on solver failure | the depot cannot stop operating when software fails | low: base plan already exists |

The ledger is the artefact that answers the question "why is the plan like this" six months later.
Each row names what was rejected, which is the part that is usually lost: a decision record that
lists only the winner reads as inevitable, and the next analyst cannot tell whether the rejection
was reasoned or accidental.

## 6. Analysis Walkthrough

The walkthrough works from bounds to construction to sequencing to cost, so that each later
derivation can lean on an earlier one. All inputs are the ILLUSTRATIVE values declared in section 4.

### 6.1 Vehicle lower bound from demand and capacity

```text
FORMULA      V_cap = ceil(D / Q)
WHERE        D = total parcels frozen at cut-off (parcels/day); Q = parcels per van (parcels/van)
CONSTRAINT   V >= V_cap is necessary, not sufficient: item sizes can force extra bins
NUMERICAL EXAMPLE  D = 1240, Q = 100 -> 1240/100 = 12.4 -> ceil = 13 vans
             load factor = 1240 / (13 * 100) = 1240/1300 = 0.9538
INTERPRETATION  12 owned vans cannot serve the day; the 13th van is a capacity requirement, not a comfort margin
```

The load factor of 0.9538 is the number that should worry a planner, not the van count. At 95%
utilisation there is almost no room to absorb a parcel discovered at the loading bay, so any late
addition converts directly into an overloaded van or an extra trip. A fleet of 13 vans with a load
factor above 0.95 is legal on paper and fragile in practice.

The check on D itself: 140 stops × 4 + 200 stops × 2 + 280 stops × 1 = 560 + 400 + 280 = 1,240
parcels across 140 + 200 + 280 = 620 stops, giving a mean of 1,240/620 = 2.0 parcels per stop. That
mean is the trap: planning on 2.0 parcels per stop suggests 620 × 2 / 100 = 12.4 vans, and the
rounding to 12 looks defensible until a van receives seven four-parcel business stops.

### 6.2 Packing bound: why capacity can force more bins than the volume ratio

```text
FORMULA      V_pack = sum over classes of ceil(n_class * d_class / Q) is an upper bound on need;
             the lower bound is max(V_cap, size_class_bound)
WHERE        size_class_bound = ceil(number of stops that cannot share a bin with another of their class)
CONSTRAINT   a van holding 4-parcel stops fits floor(100/4) = 25 such stops, no more
NUMERICAL EXAMPLE  140 four-parcel stops -> ceil(140/25) = ceil(5.6) = 6 vans fully loaded with 4s
             remaining demand 200*2 + 280*1 = 680 parcels -> ceil(680/100) = 7 vans
             total 6 + 7 = 13 vans, exactly equal to V_cap
INTERPRETATION  the instance is tightly packable at 13 vans; one more four-parcel stop would break it
```

This is the derivation that separates a capacity number from a capacity *proof*. The ratio gives 13;
the packing argument shows that 13 can actually be built given this particular mix of item sizes.
If the mix had been 143 four-parcel stops, the ratio would still print 13 while the packing bound
would print 6 + ceil((200×2 + 280×1 + 12)/100) = 6 + 7 = 13 only if the extra demand fits, and a
reviewer computing both numbers would see the difference immediately. When the two bounds disagree,
the binding one is the larger, and the plan must use it.

Boundary case where the formula stops being useful: with item sizes that do not divide the capacity
evenly — say 37-parcel stops into 100-parcel vans — floor(100/37) = 2 is an integer-packing answer
but the true optimum for that class is ceil(n/2) only when no better combination exists. Bin
packing is NP-hard; the closed-form class bound is an estimate that can over- or under-state the
minimum. Use it to detect that capacity binds, never to certify that a fleet is sufficient.

### 6.3 Shift-time feasibility for a route

```text
FORMULA      T_route = t_pre + sum_{legs} t_ij + sum_{stops} s_j + t_break + W
WHERE        t_pre = preload minutes/van, s_j = service minutes at stop j, W = total waiting minutes
CONSTRAINT   T_route <= 480 min and the break must start by minute 240 of duty
NUMERICAL EXAMPLE  stops/route = 620/13 = 47.69; legs = 48.69; legs*2.9 = 141.2 min
             service = 47.69 * 2.4 = 114.5 min; preload 22; break 30; waiting from 6.4 -> W = 0
             T_route = 22 + 141.2 + 114.5 + 30 + 0 = 307.7 min
             utilisation = 307.7 / 480 = 0.641
INTERPRETATION  the average route uses 64% of the shift, so time and capacity bind in different ways
```

The 64% figure is why the day is capacity-limited rather than time-limited, and that conclusion is
the actionable output of this derivation. With 36% of the shift unallocated, growing the fleet does
not buy time relief, so a dispatcher who responds to lateness by adding vans is treating the wrong
constraint. The correct lever differs by constraint: capacity calls for fleet or for promise
management, time calls for route splitting, and windows call for sequencing.

The maximum stops per route under time only follows from the same formula solved for n, with one
substitution: each additional stop adds one leg and one service, so the marginal cost is
2.9 + 2.4 = 5.3 minutes per stop.

```text
FORMULA      n_max = floor((480 - t_pre - t_break - t_return) / (t_leg + s_mean))
WHERE        t_return = travel minutes from the last stop back to the depot (1.2 in this instance)
NUMERICAL EXAMPLE  (480 - 22 - 30 - 1.2) / 5.3 = 426.8 / 5.3 = 80.5 -> n_max = 80 stops/route
             vans by time only = ceil(620/80) = 8
INTERPRETATION  time alone would allow 8 vans; capacity requires 13; capacity is the binding constraint
```

Boundary: this formula assumes the marginal stop costs one leg and one service. It breaks when the
marginal stop is a multi-parcel business drop (service 6.0 rather than 2.4, marginal 8.9) or when
the marginal stop sits across a river with a long return leg. A planner who applies the mean
marginal cost to a heterogeneous stop mix understates route duration by roughly the variance in
service time, which is exactly where the late routes appear.

### 6.4 Clarke–Wright savings on a five-node example

```text
FORMULA      s(i,j) = d(0,i) + d(0,j) - d(i,j)
WHERE        d in km; 0 = depot; s > 0 means merging the two radial trips shortens the tour
CONSTRAINT   a merge is admissible only if both nodes are route endpoints, capacity holds, and the
             merged sequence passes every window check
NUMERICAL EXAMPLE  d(0,1)=1.2 d(0,2)=2.0 d(0,3)=2.4 d(0,4)=3.1 d(1,2)=1.5 d(1,3)=2.2
             d(1,4)=3.3 d(2,3)=1.1 d(2,4)=2.0 d(3,4)=0.9
             s(3,4) = 2.4+3.1-0.9 = 4.6   <- largest, merge first
             s(2,3) = 2.0+2.4-1.1 = 3.3   s(2,4) = 2.0+3.1-2.0 = 3.1
             s(1,2) = 1.2+2.0-1.5 = 1.7   s(1,3) = 1.2+2.4-2.2 = 1.4   s(1,4) = 1.2+3.1-3.3 = 1.0
INTERPRETATION  merge order by savings gives route 0-3-4-0 then 0-2-3-4-0 then 0-1-2-3-4-0
```

The arithmetic that makes this auditable is the comparison of the merged tour against the tours it
replaces. Start from radial service, where each of the four stops is its own round trip: the radial
distance is 2 × (1.2 + 2.0 + 2.4 + 3.1) = 17.4 km. After the first merge the tour 0-3-4-0 costs
2.4 + 0.9 + 3.1 = 6.4 km, replacing the radial 2.4 + 2.4 + 3.1 + 3.1 = 11.0 km, saving 4.6 km,
which is exactly s(3,4). After the second merge 0-2-3-4-0 costs 2.0 + 1.1 + 0.9 + 3.1 = 7.1 km,
replacing 6.4 + 2 × 2.0 = 10.4, saving 3.3 = s(2,3). After the third merge 0-1-2-3-4-0 costs
1.2 + 1.5 + 1.1 + 0.9 + 3.1 = 7.8 km, replacing 7.1 + 2 × 1.2 = 9.5, saving 1.7 = s(1,2). Total
saving 4.6 + 3.3 + 1.7 = 9.6 km against the radial 17.4, and 17.4 − 9.6 = 7.8 checks.

Feasibility of the merged route is checked with the load, not with the savings. Suppose the four
stops carry 12, 8, 4 and 16 parcels: the merged route carries 40 ≤ 100, so capacity holds and the
saving is real. If the same stops carried 30, 28, 26 and 22 parcels, the total 106 > 100 would make
the merge inadmissible even though s(3,4) = 4.6 is the largest saving in the matrix.

**Boundary / counterexample.** Savings values are pairwise and static; admissibility is sequential
and dynamic. Take a route where node 2 has already become interior after the merge 0-2-3-4-0. The
pair (2,4) still shows s = 3.1, but merging at node 2 would break the existing tour and create a
subtour rather than joining two routes end-to-end. Applying that saving is the classic
Clarke–Wright implementation error: the arithmetic is right and the plan is invalid. The
discriminating check is not the savings matrix but the endpoint flag on each node, and a validator
that re-walks the produced sequence is the only reliable defence.

A second boundary is the windows. Savings assumes distance-only costs, so a merge that saves 4.6 km
and forces 20 minutes of waiting on a windowed stop is a net loss evaluated on the true objective,
and savings cannot see that. In this instance waiting is quantified separately in 6.5 precisely
because the constructor is blind to it.

### 6.5 Window sequencing and waiting: the forward recursion

```text
FORMULA      a_1 = max(e_1, t_dep + t_(0,1));  a_j = max(e_j, a_(j-1) + s_(j-1) + t_(j-1,j))
             W_j = max(0, e_j - (a_(j-1) + s_(j-1) + t_(j-1,j)))
WHERE        a_j = service start at stop j (clock minutes); e_j = earliest start; W_j = waiting
CONSTRAINT   a_j <= l_j for every windowed j, else the route is late by a_j - l_j
NUMERICAL EXAMPLE  van departs 07:00; t(0,A)=18; e_A=09:00 -> arrive 07:18, wait 102 min, start 09:00
             service 2.4 -> 09:02.4; t(A,B)=7 -> arrive 09:09.4; e_B=09:30 -> wait 20.6 min
             start B 09:30, finish 09:32.4; t(B,C)=9 -> arrive 09:41.4; e_C=10:00 -> wait 18.6
             total waiting W = 102 + 20.6 + 18.6 = 141.2 min
INTERPRETATION  the wait is 141.2/480 = 29.4% of the shift, and it is invisible in a distance objective
```

That 29.4% is the single most useful diagnostic in the case. A route can be short in kilometres and
still consume most of its shift in the depot's own waiting, and the only way to see it is to carry
the clock through the sequence rather than to sum leg times. Three remedies exist and they are not
interchangeable: sequence the windowed stops last (reduces wait, increases distance), insert
non-windowed stops into the gap (reduces wait at zero distance cost when the stops are geographically
between), or negotiate the window. The second remedy is usually the cheapest and is the one a
savings constructor will never find, because it has no concept of inserting work into idle time.

```text
FORMULA      lateness_risk = (l_j - a_j) / s_mean   [in stop-equivalents of slack]
WHERE        slack in minutes divided by mean service time gives how many stops of disruption fit
CONSTRAINT   flag any windowed stop with lateness_risk < 1.0 for dispatch review before 07:00
NUMERICAL EXAMPLE  stop C: l_C = 11:30 = 690; a_C = 10:00 = 600; slack = 90 min -> 90/2.4 = 37.5
             a stop with a_C = 11:25 -> slack 5 min -> 5/2.4 = 2.08 stop-equivalents
             a stop with a_C = 11:32 -> slack -2 -> late by 2 min, flag and repair before publish
INTERPRETATION  expressing slack in stop-equivalents lets dispatch compare risk across routes with different densities
```

Boundary and counterexample for the recursion. The formula assumes the van may wait and that
waiting is free of cost other than time; it also assumes windows are closed intervals. Real customer
promises are often half-open — service must *start* before `l`, not at `l` — and the difference
decides the fate of every stop whose computed arrival equals the deadline exactly. In the example
above, a stop arriving at 11:30 with l = 11:30 is feasible under a closed reading and late under a
half-open reading. A second boundary: the recursion is forward-only, so it cannot tell whether a
different sequence would have produced less waiting. Evaluating that requires enumerating orders
inside the window cluster, which is what the local search does with a lexicographic score rather
than a formula.

### 6.6 The manual baseline and what re-sequencing alone buys

```text
FORMULA      improvement = (d_manual - d_plan) / d_manual
WHERE        d in km per day for the same stop set, same date, same population
CONSTRAINT   the comparison is only valid if the manual plan and the plan share population and date
NUMERICAL EXAMPLE  d_manual = 842 km, d_plan = 734.2 km -> (842 - 734.2)/842 = 107.8/842 = 0.1280
             distance per stop: 842/620 = 1.358 km vs 734.2/620 = 1.184 km
INTERPRETATION  a 12.8% distance reduction at the same fleet of 13 vans, or 0.174 km per stop
```

The comparison is deliberately same-day and same-population, because the alternative — comparing
this week's plan against last month's manual average — confounds route design with volume, weather
and driver mix. A 12.8% number produced by that confounded comparison would be indistinguishable
from a change in stop density, which is why the case insists on the matched design even in an
illustrative setting.

The decomposition of the saving matters more than its size. In this instance most of the gain comes
from two mechanisms that are separable: merging radial sector trips into longer chains (the savings
mechanism, worth roughly 9–11% on the toy example) and removing waiting by re-sequencing windowed
stops (worth up to the 141 minutes computed in 6.5 when it applies to more than one cluster). The
first mechanism is a distance effect and the second is a time effect; a single percentage hides
which one is doing the work, and a planner who does not know cannot predict whether the gain will
survive a change in the window policy.

### 6.7 The van-count versus overtime break-even

```text
FORMULA      C(V, h) = V*c_fixed + c_km*d(V) + V*H*c_driver + c_ot*h
WHERE        c_fixed = 118 per van-day, c_km = 0.46 per km, c_driver = 27 per hour, H = 8 paid hours
             c_ot = 40.5 per hour, h = overtime hours per day, d(V) = total distance at fleet size V
CONSTRAINT   compare plans only at equal feasibility; an illegal plan has no cost comparison
NUMERICAL EXAMPLE  C13 = 13*118 + 0.46*734.2 + 13*8*27 + 0 = 1534 + 337.7 + 2808 = 4679.7
             C12 = 12*118 + 0.46*842 + 12*8*27 + 40.5*6.4 = 1416 + 387.3 + 2592 + 259.2 = 4654.5
             C13 - C12 = 4679.7 - 4654.5 = 25.2 in favour of the 12-van plan
INTERPRETATION  the cheaper plan is the illegal one; 12 vans breach capacity by 1240 - 1200 = 40 parcels
```

This is the derivation that stops an analyst from shipping a cost-optimal plan that cannot be
loaded. The 12-van plan is cheaper by 25.2 currency units per day and infeasible on C1: total
capacity 1,200 parcels against demand 1,240. Feasibility is a filter applied before the cost
function, so the correct recommendation is the 13-van plan at 4,679.7, and the 25.2 is the price of
legality — a small and defensible price.

The break-even then tells dispatch when the answer would flip. Setting C13 − C12 = 0 and solving
for overtime hours:

```text
FORMULA      h* = (c_fixed + c_driver*H + c_km*(d13 - d12)) / c_ot
WHERE        h* is the overtime level above which the extra van is cheaper than squeezing the fleet
NUMERICAL EXAMPLE  (118 + 27*8 + 0.46*(734.2 - 842)) / 40.5 = (118 + 216 - 49.6)/40.5
             = 284.4 / 40.5 = 7.02 hours per day
INTERPRETATION  below 7.02 overtime hours the 12-van plan is cheaper in money but still infeasible; above it, 13 vans wins on both counts
```

Boundary: the break-even is a single-day, single-parameter calculation and it breaks when overtime is
capped by agreement (then h cannot exceed the cap and the formula's region is unreachable), when the
rental van is unavailable (then 13 is not an option and the comparison is between 12 vans and a
service failure), and when the extra van's distance is not the plan's distance (a rental van parked
at a satellite yard changes d13). Each of these is a real constraint that the algebra cannot see.

### 6.8 Route stability as a cost that the distance objective ignores

The last derivation is qualitative and is included because it is the mechanism that destroys
adoption. Define churn as the share of stops whose assigned van changes between consecutive days at
constant volume.

```text
FORMULA      churn = |stops changing van day to day| / |stops| 
WHERE        population = stops present on both days; a stop appearing on one day only is excluded
CONSTRAINT   keep churn below 0.25 unless a volume change above 15% forces reassignment
NUMERICAL EXAMPLE  620 stops, 138 reassigned between two consecutive plans -> 138/620 = 0.2226
             at 0.35 churn the same distance plan is operationally worse because drivers re-learn 217 addresses
INTERPRETATION  churn trades against distance; a 3% distance gain bought with 35% churn is a net loss
```

Churn is not a soft aesthetic. A driver who re-learns 217 addresses loses the tacit knowledge that
keeps service times at 2.4 minutes rather than the 6.0 minutes observed at unfamiliar business
sites, so the service-time assumption in §4 becomes invalid in exactly the routes that changed
most. The practical implementation is a penalty in the local search's third objective level: accept
a reassignment only if it improves distance by more than a threshold, or if the current assignment
is infeasible. Stating a threshold is a management decision, not a modelling one, and it belongs in
the ledger in 5e so that it is visible in review.

## 7. The Finding

The finding of the illustrative analysis is a hierarchy, not a single number.

**First, capacity binds and time does not.** The capacity bound requires 13 vans (1,240 parcels at
100 per van), the packing argument confirms 13 are constructible with this item mix, and the time
bound would allow 8 vans at 80 stops per route. Because time is not binding, the average route uses
307.7 of 480 minutes, or 64.1%. The direct consequence is that lateness in this operation is a
sequencing and window problem, not a staffing problem, and adding vans does not fix it.

**Second, the achievable distance saving from re-sequencing is about one eighth of the baseline.**
The plan covers 734.2 km against a 842 km manual baseline, a reduction of 12.8%, or 0.174 km per
stop. Because the fleet is unchanged at 13 vans, this saving requires no capital decision and no
roster change; it is available inside the existing operation, which is the reason it is worth
pursuing first.

**Third, waiting is the hidden cost centre and it is large enough to swamp distance gains.** The
window cluster in 6.5 consumes 141.2 minutes of a 480-minute shift — 29.4% — on a route whose
distance looks short. Any objective that scores kilometres alone will systematically route vans into
early arrival and idle time, then report an improvement while the shift is consumed by waiting.

**Fourth, the cheapest plan is illegal, and the price of legality is small.** The 12-van plan costs
25.2 less per day and overloads the fleet by 40 parcels. The 13-van plan is the recommendation, and
the break-even overtime of 7.02 hours per day is the threshold at which the fleet decision would
genuinely flip for a legal alternative.

**Fifth, the fleet size has an interior optimum in the region that matters.** Between 12 and 13
vans the cost difference is 0.5% of daily cost; between 13 and 14 the additional van adds 118 plus
216 in driver cost, roughly 7% of the day, and buys nothing that the constraint set requires. A
planner who optimises van count aggressively is chasing a rounding difference while the window
cluster wastes a third of a shift.

The counter-finding deserves equal prominence, because it is what an experienced reviewer would
raise. The 12.8% is a planning-time distance reduction against a manual baseline on a constructed
instance; it is not a measured operational saving, and it says nothing about driven distance at the
end of the day, which depends on traffic, on parking search and on driver adherence. The case
therefore reports the number as `ILLUSTRATIVE` and proposes the measurement that would replace it:
plan distance versus telematics distance per route, on the same day, with the ratio tracked as a
drift indicator rather than as a success claim.

## 8. System/Solution Design

The design has six components with explicit contracts, because the failure modes in §12 are almost
all contract failures rather than optimiser failures.

**Component 1 — Plan request builder.** Produces an immutable snapshot: stops, demand by parcel
class, windows, fleet roster, matrix version, and a SHA-256 hash over the canonical serialisation.
Contract: the solver accepts only a request object, never a live database connection. That single
rule removes the largest class of irreproducible plans, because the inputs cannot change underneath
a running solve.

**Component 2 — Feasibility pre-screen.** Computes the capacity bound, the packing bound, the time
bound and the fleet comparison before any construction. Output is a status of `legal`, `tight` or
`short`, with the binding constraint named. Contract: a `short` status blocks publication and
raises a fleet escalation; it never silently produces a plan that violates capacity.

**Component 3 — Constructor and improver.** Deterministic savings followed by local search with a
lexicographic acceptance rule. Contract: the improver returns the move log alongside the plan, so a
reviewer can see which moves were accepted and why. A plan without a move log is not publishable,
because the only way to distinguish an improvement from an artefact is to read the moves.

**Component 4 — Independent validator.** A second implementation of C1–C6 that walks each produced
route stop by stop. Contract: violation codes, not prose. The codes are `V_CAP`, `V_SHIFT`,
`V_BREAK`, `V_WINDOW`, `V_ASSIGN`, `V_DEPOT`. Any non-empty violation list blocks the plan,
regardless of its distance.

**Component 5 — Dispatcher console with a closed override taxonomy.** Overrides are recorded as
`(plan_id, stop_id, from_van, to_van, code, free_text, user, timestamp)`. The code set is closed:
`ACCESS` (building or gate restriction), `SAFETY` (road or parking hazard), `DRIVER_KNOWLEDGE`
(customer relationship), `CAPACITY_UNLOADABLE` (volume or weight despite parcel count), `WINDOW_
NEGOTIATED` (customer agreement), `OTHER_REVIEW`. Free text is allowed and is not the primary field;
a taxonomy with a catch-all that swallows 40% of overrides is a taxonomy that needs revision, and
the share of `OTHER_REVIEW` is itself a monitored quantity.

**Component 6 — Publish, audit and fallback.** Publishes plan, validator report, risk list and
request hash into an append-only log, then releases to the driver application. The fallback path is
explicit: if any gate fails, the approved manual base plan for the date is published instead, with
the failure code attached. The depot therefore always has a plan, and the failure is visible in the
audit trail rather than absorbed by a dispatcher under time pressure.

The data contract for the route object is small on purpose:

```text
route      = { van_id, driver_id, stops: [stop_id...], distance_km, duration_min,
               wait_min, load_parcels, load_m3, shift_start, shift_end, plan_version }
plan       = { request_hash, matrix_version, routes: [route...], excluded: [stop_id...],
               violations: [code...], risk: [{stop_id, slack_min}...] }
```

Two properties of this contract do real work. `request_hash` makes the plan reproducible from the
audit log alone. `excluded` makes the population explicit, which prevents the metric-denominator
slippage described in 5d. Neither property is technically difficult and both are routinely missing,
which is why the design foregrounds them.

Monitoring is defined as four signals with thresholds, not as a dashboard. Validator violations per
plan (target zero, alert on any), share of `OTHER_REVIEW` overrides (alert above 15%), plan-distance
to telematics-distance ratio per route (alert outside 0.95–1.10), and realised shift overrun minutes
per route (alert above 20). Each signal has an owner and a response, and the response for the
distance-ratio alert is to inspect the matrix version, not to retrain anything.

## 9. Evaluation & Validation

Evaluation is specified before it is run, because a metric defined after seeing the result is a
metric chosen to flatter it. Every row below states numerator, denominator, population, window,
comparator, guardrails, uncertainty treatment and execution status. Statuses are honest: this case
is a design and a constructed analysis, so field metrics are `not-executed`.

| Metric | Numerator / denominator | Population, window | Comparator | Guardrails | Uncertainty | Status |
|---|---|---|---|---|---|---|
| Validator violation rate | plans with ≥1 violation / plans published | all plans, 4-week window | zero target | any violation blocks publish | exact count, no estimate | not-executed (design) |
| Plan distance per stop | Σ route km / stops published | planned stops only, daily | manual baseline same day | denominator excludes excluded stops | day-to-day SD reported | illustrative 1.184 km |
| Distance reduction vs manual | (manual − plan) / manual | matched population, matched date | sector-based manual plan | churn ≤ 0.25 and feasibility clean | bootstrap over 20 days | illustrative 12.8% |
| Realised vs plan distance | telematics km / plan km | per route, daily | 1.00 | exclude routes with >2 overrides | distribution, not mean | not-measured |
| Window adherence | stops served in window / windowed stops | windowed stops, daily | manual baseline | excludes customer-cancelled | Wilson interval on proportion | not-executed |
| Shift overrun | routes ending after shift end / routes | all routes, daily | zero target | break placement checked separately | count based | not-executed |
| Override share by code | overrides of code / overrides total | all overrides, weekly | taxonomy health | alert if OTHER_REVIEW > 0.15 | exact ratio | not-executed |
| Waiting share of shift | Σ wait min / Σ route min | all routes, daily | ≤ 0.10 design target | measured from the recursion, not from GPS | route-level variance | illustrative 0.294 in cluster route |

The comparator deserves a note, because it is the part most evaluations get wrong. Comparing the
plan against the manual baseline requires the same stops on the same date. Comparing against last
month's average conflates route design with volume mix, and comparing against the previous plan
conflates improvement with random variation in the stop set. The matched-pair design costs extra
effort in the dispatch team, who must produce a manual plan for the same day for the comparison to
exist, and that effort is the price of a defensible number.

Guardrails are constraints on the evaluation itself, not on the plan. Three are enforced here: the
population must include the excluded-stop count so it cannot drift; the window must be fixed before
the first plan is produced; and no metric may be reported without its uncertainty statement. The
third guardrail is the one that stops the most common abuse, which is quoting a 12.8% improvement
from twenty days without saying that the day-to-day standard deviation is larger than the effect.

The planned validation ladder has four rungs, and only the first two are executed in this case.

1. **Arithmetic validation (executed).** Re-derive bounds, savings and cost break-even by hand
   against the worked values in §6; every printed intermediate value reconciles to a stated input.
2. **Constraint validation on constructed cases (executed).** Small instances with known optima —
   the five-node savings example, a window cluster with a known best sequence, and a deliberately
   infeasible instance where the validator must return `V_CAP`.
3. **Shadow planning (not executed).** Run the router in parallel with manual dispatch for four
   weeks without using its output, and compare distance, window adherence and churn on matched days.
4. **Controlled pilot (not executed).** Publish the router's plan on a subset of routes with the
   manual base retained as fallback, and measure realised-versus-plan distance ratios.

The gap between rung 2 and rung 3 is where most routing projects fail silently: the solver is
correct, the validator is correct, and the plan is still worse in the field because the matrix is
stale in a way that constructed instances cannot express. This is why rung 3 is defined as shadow
rather than pilot — it produces the evidence without exposing customers to the learning curve.

## 10. Action Taken

The action set is a proposal with a named decision owner for each item, because a recommendation
without an owner does not survive the next planning meeting.

| # | Action | Owner | Trigger | Reversible |
|---|---|---|---|---|
| 1 | Adopt the savings-plus-local-search constructor | analytics lead | validator passes constructed cases | yes, module swap |
| 2 | Require independent validation before publish | dispatch manager | first published plan | no, becomes policy |
| 3 | Freeze matrix and inputs per plan hash | data engineering | first published plan | yes |
| 4 | Publish the lateness-risk list before 07:00 | dispatch supervisor | daily | yes |
| 5 | Approve the 13th van (rental) for days with demand ≥ 1,200 parcels | fleet manager | capacity pre-screen says `short` | yes, daily decision |
| 6 | Keep the approved manual base plan as fallback | dispatch manager | any gate failure | no |
| 7 | Instrument churn and review it weekly | analytics lead | first four weeks | yes |

The rejected actions are as important. An exact MIP formulation was rejected because it cannot
close the instance inside the planning window; a dynamic re-optimisation during the day was rejected
because the depot has no reliable mechanism to re-sequence a loaded van without breaking the shift
guarantee; and switching the objective to a weighted sum of distance and lateness was rejected
because a weight is an invitation to trade service for kilometres at 03:00 when nobody is watching.

Two actions were deliberately deferred rather than rejected. Territory redesign — changing which
postcode sectors belong to which van permanently — is deferred because it is a labour-relations
decision with a longer cycle than the routing problem, and because the case's churn analysis
suggests it should be evaluated on its own evidence. Customer promise-window redesign is deferred
to the commercial owner, since the 141-minute wait in 6.5 is an argument about when customers may
receive parcels, not about how vans should drive.

## 11. Outcome & Results

The results of this case are results of an arithmetic analysis on a constructed instance. They are
stated with that label because the alternative — presenting computed values as operational
outcomes — would be a fabrication, and because the values are more useful as a worked example than
as a claim.

| Quantity | Value | Units | Basis | Status |
|---|---|---|---|---|
| Stops served | 620 | stops/day | ILLUSTRATIVE input | constructed |
| Parcels | 1,240 | parcels/day | 140×4 + 200×2 + 280×1 | constructed, arithmetic checked |
| Capacity lower bound | 13 | vans | ceil(1240/100) | derived |
| Packing bound | 13 | vans | 6 vans of 4-parcel stops + 7 vans of the rest | derived |
| Time-only lower bound | 8 | vans | ceil(620/80) | derived |
| Load factor at 13 vans | 0.9538 | ratio | 1240/1300 | derived |
| Mean route duration | 307.7 | minutes | 22 + 141.2 + 114.5 + 30 | derived |
| Shift utilisation | 0.641 | ratio | 307.7/480 | derived |
| Plan distance | 734.2 | km/day | 13 routes × 56.48 km | derived |
| Manual baseline distance | 842 | km/day | ILLUSTRATIVE baseline | constructed |
| Distance reduction | 0.1280 | ratio | (842 − 734.2)/842 | derived |
| Waiting in window cluster | 141.2 | minutes | forward recursion | derived |
| Waiting share of shift in cluster route | 0.294 | ratio | 141.2/480 | derived |
| Daily cost at 13 vans | 4,679.7 | currency units | 1534 + 337.7 + 2808 | derived |
| Daily cost at 12 vans with 6.4 h overtime | 4,654.5 | currency units | 1416 + 387.3 + 2592 + 259.2 | derived, illegal plan |
| Overtime break-even | 7.02 | hours/day | 284.4/40.5 | derived |

What would count as success in a real deployment is stated as thresholds so that the analysis is
falsifiable rather than merely favourable. A pilot is successful when validator violations are zero
for four consecutive weeks, when realised-versus-plan distance stays inside 0.95–1.10 per route,
when window adherence at least matches the manual baseline on matched days, when churn stays below
0.25, and when overrides coded `OTHER_REVIEW` stay below 15% of all overrides. Missing any one of
those thresholds is a partial result, and the case treats a partial result as a finding rather than
as a failure of the programme.

The honest negative result is that the largest single cost in the instance is driver time, not
distance: 2,808 of the 4,679.7 daily units — 60.0% — is paid shift time at 27 per hour, against
337.7 for distance. An operation that reduces kilometres by 12.8% without reducing route hours has
reduced 7.2% of its cost base, not 12.8% of it. That gap between the headline metric and the cost
metric is exactly the sort of thing a case study should surface, because it changes which lever a
manager pulls next.

## 12. Failure Modes & What Went Wrong

Six mechanisms are listed, each with the symptom that reveals it and the check that discriminates it
from its competitors. The checks are chosen so that they can be run on the day, not in a post-mortem.

**M1 — Stale or profile-mismatched travel matrix.** The plan is feasible in the model and late in
the field, with lateness concentrated in the afternoon. Competing causes: service time understated,
break omitted, driver adherence poor. Discriminating check: compare the frozen matrix's median leg
duration against probe-vehicle timings on the same arcs, split by hour; a ratio above 1.10 in the
16:00–18:00 band identifies the matrix rather than the plan. Correction: version the matrix by
profile and re-solve; do not patch the route.

**M2 — Demand unit mismatch between manifest and plan.** A van is loaded beyond capacity even
though the plan reports a legal load. Competing causes: parcel count correct but volume or weight
binding, loading error at the bay, wrong van assigned. Discriminating check: reconcile the plan's
per-van parcel total against the manifest scan count and against a case-to-parcel conversion factor;
if parcel sums agree but cubic metres exceed the van's volume, the binding constraint is volume and
the model is incomplete, not wrong. Correction: carry all three capacity dimensions, and treat the
plan as infeasible when any of them binds.

**M3 — Window boundary semantics.** Stops arriving exactly at the deadline are shown as feasible
and reported as late by the customer. Competing causes: travel time noise, service overrun at the
preceding stop, promise different from the operational window. Discriminating check: count stops
where computed arrival lies within one minute of the deadline; if that count is non-zero and those
stops dominate the complaints, the defect is interval semantics rather than driving. Correction:
declare the interval convention once — the case uses half-open `[e, l)` with service *start* inside
the interval — and enforce it in the validator, not only in the solver.

**M4 — Silent stop loss through optional-node penalties.** The plan looks excellent because several
stops were dropped as unprofitable under a per-stop penalty, and the dropped stops are invisible in
the distance metric. Competing causes: excluded stops by filter, duplicate stop ids merged at join,
stops assigned to a van outside the fleet. Discriminating check: set difference between the request's
stop ids and the union of assigned stop ids; the difference must equal the declared `excluded` list
exactly. Correction: make assignment of every stop a hard constraint (C5), and report the excluded
list as a first-class output.

**M5 — Plan/execution version drift.** Drivers execute yesterday's sequence because the handheld
application cached an older plan, or because the override was applied on the console but not
republished. Competing causes: driver deviation, address change by the customer, matrix refresh
mid-day. Discriminating check: compare `plan_version` hashes held by the console, the audit log and
the driver application at 07:15; a mismatch is version drift, not behaviour. Correction: publish is
a single atomic step that writes the hash to all three surfaces, and the driver application refuses
a plan whose hash does not match the log.

**M6 — Objective gaming at the window boundary.** Distance falls and lateness rises, with the plan
nominally feasible because every stop is inside its window by a margin of seconds. Competing causes:
genuine travel improvement, fewer stops served, longer shifts. Discriminating check: plot the
distribution of slack at windowed stops, not just the count of violations; a distribution shifted
toward zero with an unchanged violation count is objective gaming. Correction: impose a minimum
slack guardrail per route (for example 10 minutes at the last windowed stop) and treat the guardrail
as part of feasibility, so the search cannot spend it.

Three of these six — M1, M3, M5 — are contract and semantics failures that no amount of solver
tuning can fix, and they are the ones that recur across organisations. The pattern worth remembering
is that a routing defect almost never announces itself as "the optimiser was wrong". It announces
itself as drivers ignoring the plan (M5), as customers complaining about a specific street (M3), or
as one shift of the day being consistently late (M1). The diagnostic skill is translating the
operational complaint into the specific check.

The near-miss to record is the geocode ratio in 5d. Three unresolved addresses in 620 stops looked
like a 99.5% match rate, which is comfortably above the gate, until the duplicate rows were removed
and the true numerator fell. The lesson is not that the gate is wrong; it is that a ratio computed
on a join with duplicate keys can exceed its own maximum, and any metric that can exceed 1.0 should
be treated as a data defect until proven otherwise.

## 13. Transferable Lesson

The transferable content of this case is a decision order that survives a change of domain.

**Feasibility is a filter, not a term in the objective.** The cheapest plan in 6.7 is illegal, and
any weighted objective that would have chosen it has a weight that is too small to represent the
cost of a failed dispatch. The general form of this lesson: when a constraint is contractually or
physically absolute, it must be enforced by rejection rather than by penalty.

**An independent checker is cheap and it is the only real certificate.** The validator is a few
hundred lines of arithmetic that re-walks routes. It catches the endpoint error in Clarke–Wright,
the unit mismatch, and the silent stop loss, and it does so without needing to understand the
optimiser. Sharing the solver's own feasibility routine is the mistake that makes all three
invisible.

**Frozen inputs turn a plan into evidence.** A content hash over stops, demand, windows, fleet and
matrix is the difference between "the plan was bad" and "the plan was correct for the inputs it was
given, and those inputs were stale". The second statement is actionable; the first is an argument.

**The obvious lever is usually not the binding one.** Capacity bound 13, time bound 8, and the
average route uses 64% of its shift. In that configuration, lateness is a window problem and fleet
growth is a rounding error. Reaching that conclusion requires two bounds computed before any
optimisation, which is twenty minutes of work that most projects skip.

**Every improvement needs a denominator that cannot drift.** Distance per stop is only comparable
when the stop population is fixed and the exclusions are reported. The population-leakage guard in
5d is the general technique: publish the denominator beside the metric.

The lesson that is easiest to state and hardest to practise is the last one. A 12.8% planning-time
distance reduction is a hypothesis about operational saving, not a measurement of it. The case
reports it as illustrative and proposes the ratio that would test it. Analysts who skip that step
end up defending a number they cannot reproduce when the matrix is refreshed.

## 14. Connections & Cross-References

Within this unit, the routing problem is a constrained network optimisation and connects directly to
the formulation sheet, which derives the primal and dual views used to think about capacity shadow
prices: [Optimisation formulation](../Cheat_Sheets/01_optimisation_formulation.md). The solver
selection argument in 5b is developed further in
[Solver selection](../Cheat_Sheets/02_solver_selection.md), which classifies problems by objective,
integrality and certificate requirements and explains why this instance is heuristic territory.

The vehicle-and-arc structure is the same object studied in
[Network and scheduling models](../Cheat_Sheets/04_network_and_scheduling_models.md), where flow
conservation and cut arguments give the lower bounds that 6.1 and 6.2 compute by hand. The
neighbourhood and acceptance logic of the improver is treated systematically in
[Metaheuristics](../Cheat_Sheets/06_metaheuristics.md), including the endpoint-admissibility
pitfall that M4-adjacent implementations hit when they accept a saving at an interior node.

The scheduling sibling cases share the feasibility-first discipline:
[Workforce scheduling under uncertainty](02_workforce_scheduling_under_uncertainty.md) uses the same
pattern with staffing instead of vans, and
[Scheduling objective and constraints](03_scheduling_objective_and_constraints.md) makes the
greedy-versus-weighted distinction that appears here as savings-versus-window sequencing. The unit
inventory is [the module index](../INDEX.md), and the collection view is
[the case index](./INDEX.md).

Outside this unit, the evaluation design in §9 follows the matched-pair and guardrail conventions
used in the simulation experiment sheet,
[Simulation experiment plan](../Cheat_Sheets/03_simulation_experiment_plan.md), which insists on
pre-declared windows and independent replications for the same reason: a metric defined after the
result is a metric chosen to flatter it.

### Interview pack

**Q1. A 620-stop day carries 1,240 parcels and each van holds 100. How many vans, and what does the answer miss?**
- *Answer:* 13 vans by the capacity ratio (1,240/100 = 12.4, rounded up). What it misses is packing: the
  bound is necessary, not sufficient, and the item mix must be checked to confirm 13 is constructible.
- *Reasoning:* the ratio ignores that 140 stops carry 4 parcels each, so 25 such stops fill a van
  exactly; the class argument (ceil(140/25) = 6 vans, then ceil(680/100) = 7) reconstructs 13.
- *Follow-up:* what changes if the mix shifts to 143 four-parcel stops while total demand is flat?
- *Red flag:* quoting 12.4 and rounding down "because the load factor looks comfortable".
- *Scoring:* 3 points for the number, 5 for naming the packing gap, 2 for the load factor 0.954.

**Q2. Capacity requires 13 vans and the time bound allows 8. Which constraint should a late-running operation attack, and why?**
- *Answer:* capacity, because it is the binding constraint; time has slack (64% utilisation), so more
  vans do not buy route time and lateness is a window-sequencing problem.
- *Reasoning:* the two bounds answer different questions — demand per van and stops per shift — and
  the larger one governs. Adding vans when time is slack raises cost without relieving the real limit.
- *Follow-up:* what evidence would show that time, not capacity, is binding on a different day?
- *Red flag:* "add vans until lateness stops", which conflates a service symptom with a capacity cause.
- *Scoring:* 4 for naming the binding constraint, 4 for the utilisation evidence, 2 for the lever.

**Q3. Why can a Clarke–Wright saving of 3.1 km be inadmissible even though the arithmetic is right?**
- *Answer:* because admissibility is sequential: one of the nodes may already be interior to a route,
  so merging there creates a subtour instead of joining two route ends.
- *Reasoning:* savings are pairwise and static; validity depends on the current route structure. The
  endpoint flag, not the savings value, decides.
- *Follow-up:* which artefact in the pipeline catches this if the implementation is wrong?
- *Red flag:* implementing merges by sorting the savings matrix and applying greedily without state.
- *Scoring:* 5 for the interior-node mechanism, 3 for the endpoint check, 2 for naming the validator.

**Q4. A window cluster consumes 141 minutes of waiting on a 480-minute shift. Name two fixes and say which is cheaper.**
- *Answer:* sequence windowed stops later, or insert non-windowed stops into the idle gap. Insertion
  is usually cheaper because it consumes the waiting with work that would otherwise need its own trip.
- *Reasoning:* waiting is a resource; the insertion fix converts idle minutes into delivered parcels
  with no additional distance when the stops are geographically between the windowed ones.
- *Follow-up:* what does insertion cost when the spare stops are on the far side of the depot?
- *Red flag:* proposing "drive slower" or "leave later" as the fix, which moves the wait, not the work.
- *Scoring:* 3 for each fix, 2 for identifying insertion as cheaper, 2 for the distance condition.

**Q5. Recomputed: at 13 vans the day costs 4,679.7; at 12 vans with 6.4 hours overtime it costs 4,654.5. What do you recommend?**
- *Answer:* 13 vans. The 12-van plan is 25.2 cheaper and breaches capacity by 40 parcels, so it is not
  a candidate; feasibility filters before cost ranks.
- *Reasoning:* 12 × 100 = 1,200 < 1,240, so the cheaper plan cannot be loaded. The 25.2 is the price
  of legality, about 0.5% of daily cost.
- *Follow-up:* at what overtime level would 12 vans become both cheaper and legal, if volume allowed it?
- *Red flag:* presenting 4,654.5 as the recommended cost without the capacity note.
- *Scoring:* 4 for choosing 13, 3 for the 40-parcel breach, 3 for the break-even reference 7.02 h.

**Q6. Derive the overtime break-even and state one condition under which it is meaningless.**
- *Answer:* h* = (c_fixed + c_driver·H + c_km·(d13 − d12)) / c_ot = (118 + 216 − 49.6)/40.5 = 7.02 h.
  It is meaningless when overtime is capped by agreement, because the region above 7.02 h is unreachable.
- *Reasoning:* the extra van trades a fixed cost plus paid hours against overtime hours; solving for
  equality gives the threshold, and any cap truncates the comparison.
- *Follow-up:* how does an unavailable rental van change the decision space?
- *Red flag:* reporting a break-even without the feasibility condition attached.
- *Scoring:* 5 for the formula and numbers, 3 for the cap caveat, 2 for the rental caveat.

**Q7. A plan is validated as feasible, yet two vans are overloaded at the bay. Walk through your diagnosis.**
- *Answer:* reconcile units first: compare plan parcels per van against the manifest scan, then check
  cubic metres and weight. If parcels agree but volume exceeds, the model is incomplete, not wrong.
- *Reasoning:* capacity has three dimensions and most models carry one; the symptom (loading bay
  overflow with a clean validator) points at the missing dimension rather than at the solver.
- *Follow-up:* what field in the plan object would have made this visible before dispatch?
- *Red flag:* "the validator said it was fine", which treats one dimension as the definition.
- *Scoring:* 4 for the unit reconciliation, 4 for the volume/weight insight, 2 for the contract fix.

**Q8. Distinguish a stale-matrix failure from a service-time failure using one measurement.**
- *Answer:* compare frozen matrix median leg time against probe timings on the same arcs, split by hour;
  a ratio above 1.10 concentrated in the afternoon identifies the matrix, while a flat ratio with late
  routes points at service time.
- *Reasoning:* both produce lateness, but only the matrix error is time-of-day dependent, so the
  hour-split comparison discriminates them.
- *Follow-up:* what if the ratio is 1.10 in every hour band?
- *Red flag:* averaging all legs into one ratio, which erases the time-of-day signal.
- *Scoring:* 5 for the hour-split check, 3 for the competing-cause discrimination, 2 for the threshold.

**Q9. Why must the validator be a separate implementation from the solver's feasibility check?**
- *Answer:* because a shared implementation shares its bugs, so a solver that omits the break or
  misreads intervals will pass its own check and publish an illegal plan.
- *Reasoning:* independence is the only mechanism that converts "the solver says it is fine" into
  evidence; it also lets the validator be simple enough to audit line by line.
- *Follow-up:* what class of defect does an independent validator still fail to catch?
- *Red flag:* citing test coverage of the solver as a substitute for an independent check.
- *Scoring:* 5 for shared-bug reasoning, 3 for auditability, 2 for the residual gap (model error).

**Q10. Give three checks a validator should perform beyond capacity and windows.**
- *Answer:* every input stop assigned exactly once; every route starts and ends at the depot; the break
  starts within the first 240 minutes of duty; and the union of assigned ids equals input minus excluded.
- *Reasoning:* these are the constraints whose violation is invisible in distance and in lateness
  counts, so they need explicit arithmetic rather than inference from outcomes.
- *Follow-up:* which of these is most often omitted in practice?
- *Red flag:* listing only capacity, which is the one everyone remembers.
- *Scoring:* 3 per substantive check, 1 for naming the invisible-in-distance property.

**Q11. What is the leakage risk when reporting a 12.8% distance improvement, and how do you remove it?**
- *Answer:* tuning on the same days used for reporting, and drifting the population denominator. Remove
  it with a chronological split and a published denominator that includes the excluded-stop count.
- *Reasoning:* both inflate the effect without changing the operation; the split fixes parameter
  selection and the denominator fixes population drift.
- *Follow-up:* why is comparing against last month's manual average a third form of leakage?
- *Red flag:* reporting the improvement without stating the window and population.
- *Scoring:* 4 for tuning leakage, 4 for population leakage, 2 for the split design.

**Q12. Design the override record. What must it contain and what must it not rely on?**
- *Answer:* plan id, stop id, from/to van, closed-taxonomy code, free text, user, timestamp. It must not
  rely on free text as the primary field, because free text cannot be aggregated or alerted on.
- *Reasoning:* overrides are the only channel through which systematic model defects surface, so they
  must be countable by cause; a catch-all code that swallows most overrides defeats the purpose.
- *Follow-up:* what would a 30% `OTHER_REVIEW` share tell you?
- *Red flag:* "dispatchers can type what they want", which makes the log unusable for review.
- *Scoring:* 4 for the fields, 4 for the taxonomy argument, 2 for the monitoring threshold.

**Q13. Churn is 0.22 while distance falls 12.8%. When is that trade unacceptable?**
- *Answer:* when the churn invalidates the service-time assumption, because drivers lose address
  knowledge and service times rise in the most-changed routes, erasing the distance gain.
- *Reasoning:* churn is a hidden cost that shows up as a service-time change, not as a distance
  change; keeping churn below 0.25 with a distance-improvement threshold is the usual control.
- *Follow-up:* how would you measure whether churn actually raised service time?
- *Red flag:* treating stability as a comfort preference rather than a cost driver.
- *Scoring:* 4 for the service-time mechanism, 3 for the threshold, 3 for the measurement.

**Q14. A stop arrives at exactly its deadline. Is the route feasible?**
- *Answer:* only if the interval convention says so. Under half-open `[e, l)` with service start inside
  the interval, arrival at `l` is late; under a closed reading it is feasible.
- *Reasoning:* the decision is semantic, so it must be declared once and enforced in the validator,
  not left to the solver's default.
- *Follow-up:* how would you detect that a production system mixes the two conventions?
- *Red flag:* "it does not matter, it is one minute", which is how boundary defects survive.
- *Scoring:* 4 for the convention, 4 for citing the validator, 2 for the detection idea.

**Q15. Why is distance per stop a poor primary KPI, and what would you pair it with?**
- *Answer:* because it rewards serving fewer or easier stops and ignores waiting; pair it with waiting
  share of shift and with realised-versus-plan distance on matched routes.
- *Reasoning:* the denominator can drift and the numerator ignores time, so a plan can improve the
  ratio while consuming more shift minutes — the 29.4% waiting case is exactly that.
- *Follow-up:* what does a stable distance per stop with rising waiting share suggest?
- *Red flag:* optimising a single ratio without a guardrail on population.
- *Scoring:* 4 for the denominator problem, 4 for the paired metrics, 2 for the diagnosis.

## 15. Quick Reference Card

| Item | Value or rule | Note |
|---|---|---|
| Instance | 620 stops, 1,240 parcels, 13 vans at 100 parcels | ILLUSTRATIVE |
| Capacity bound | ceil(D/Q) = 13 | necessary, not sufficient |
| Packing bound | 6 vans of 4-parcel stops + 7 vans of rest | confirms 13 constructible |
| Time bound | ceil(620/80) = 8 | time is not binding |
| Load factor | 1,240/1,300 = 0.954 | above 0.95 is fragile |
| Route duration | 22 + 141.2 + 114.5 + 30 = 307.7 min | 64.1% of a 480-min shift |
| Savings formula | s(i,j) = d(0,i) + d(0,j) − d(i,j) | only at route endpoints |
| Waiting recursion | a_j = max(e_j, a_(j−1) + s + t) | waiting = Σ max(0, e_j − arrival) |
| Distance result | 842 → 734.2 km, −12.8% | planning-time, matched population |
| Cost rule | 13 vans unless overtime > 7.02 h/day | illegal plan is not a comparator |
| Publish gate | validator clean, hash recorded, risk list issued | no gate, no plan |
| Override taxonomy | ACCESS, SAFETY, DRIVER_KNOWLEDGE, CAPACITY_UNLOADABLE, WINDOW_NEGOTIATED | alert if OTHER_REVIEW > 15% |
| Failure to check first | matrix vintage by hour band | explains most afternoon lateness |
