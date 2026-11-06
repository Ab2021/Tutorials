# Network mixed-integer linear programming lab

> Module 30 — Simulation, Optimization, and Real-Time Production Labs  
> Topic 04 of 10  
> Estimated study time: 3–4 hours  
> Prerequisites: Python 3.10 or later, terminal basics, linear equations, and Module 30 topics 01–03  
> Version 2026.5

## 1. Why this topic exists

A distribution executive must decide which candidate facilities to operate before negotiating leases.
A commercial team wants nearby capacity, while finance wants a defensible cost comparison.
A spreadsheet that assigns every customer to its nearest building ignores fixed operating costs and capacity.
A network design can appear inexpensive only because it silently serves more demand than its facilities can handle.
Another design can look infeasible because a demand forecast was accidentally expressed in cases while capacity was expressed in pallets.

This lab creates a small capacitated facility-location model.
It chooses which candidate facilities are open and how much demand each open facility serves.
The mixed-integer linear program (MILP) minimizes the sum of fixed facility costs and variable delivery costs.
It then checks the returned solution independently of the solver's reported objective.
You run a baseline, an outage, a demand increase, and a deliberately infeasible case.
The output supports a discussion about cost, capacity, and resilience; it is not an authorization to open or close a real facility.

The fixture is synthetic and covers one planning period.
Its dollar amounts are illustrative, not quotes or measured savings.
Demand is a planning input, not a guaranteed order quantity.
An operating plan is a chosen allocation, a service target is a desired outcome, and a customer commitment is an authorized obligation.
None of the fixture values represents a commitment or a real company's results.

A network optimization model is useful when decisions interact.
An open building has a fixed cost whether it serves one unit or many.
Allocation costs depend on both origin and destination.
Capacity couples customers who compete for the same site.
The cheapest assignment for one customer may prevent another customer from being served cheaply.
The MILP compares the combined cost rather than optimizing each lane in isolation.

A solver is an arithmetic engine, not a business owner.
It can prove optimality only for the mathematical model and the supplied input, subject to its tolerances.
It cannot establish that the costs are complete, the capacity is attainable, a lease is available, or customers accept the delivery promise.
The business owner must review those claims separately.

## 2. Learning outcomes

By the end of the lab you can:

1. Translate a facility decision into binary opening and continuous shipment variables.
2. Write an objective and demand, capacity, and closure constraints with explicit units.
3. Construct and solve a small capacitated facility-location MILP with PuLP and CBC.
4. Independently validate a solution's demand balance, capacity, binary values, closure, and recomputed cost.
5. Explain why an infeasible solve has no usable shipment plan and diagnose insufficient aggregate capacity.
6. Compare baseline, outage, and demand-stress outputs without confusing a scenario with a forecast.
7. Recommend a business next step while naming omitted costs, operational risks, and approvals.
8. Design a stronger test for a network restriction or a data-quality defect.

Exercises 1–8 assess these outcomes in order.
The embedded checks test the synthetic example; they are not evidence that a real-world network is viable.

## 3. The core mental model

```text
      decision question and planning horizon
                      |
         candidate sites + customer zones
                      |
       costs + capacities + period demand
                      |
       validate units, keys, and completeness
                      |
          MILP: open sites + allocate flow
                      |
         CBC status and candidate solution
                      |
     independent residual and objective checks
                      |
    baseline / outage / high-demand comparisons
                      |
       business review, approvals, and action
```

The sites and customer zones form a directed bipartite network.
Each shipment lane connects one site to one customer zone.
The fixture permits all six lanes; an actual business may prohibit some.
A binary variable activates a facility and its finite capacity.
A continuous flow variable allocates nonnegative units along a lane.
Total inbound flow must exactly equal the demand for each customer zone.
Total outbound flow cannot exceed the open site's capacity.
When the site is closed, its available capacity is zero, forcing all its outbound flows to zero.

Opening and allocation choices are optimized jointly.
The fixed cost creates the reason not to open every available facility.
The variable cost creates a reason to position facilities near customers.
Capacity limits can make an otherwise cheap location insufficient.
The solver searches discrete opening combinations while solving associated linear allocation problems.
The model permits a customer's demand to be split between sites; it does not enforce single sourcing.

The model is deterministic: each run uses one fixed demand and cost table.
A stress test is a separate what-if run, not a probability distribution or resilience guarantee.
An outage scenario removes a facility before solving; it permits the remaining network to redesign.
That differs from a disruption with a previously committed network and no time to open a replacement.
A decision maker must distinguish those two questions before interpreting the cost difference.

The validation layer reads the returned numbers and recomputes the constraints and costs from the fixture.
It does not independently prove global optimality; it tests whether the reported candidate satisfies the declared model.
The solver's optimal status and a valid incumbent answer different questions.
Neither establishes that the underlying business data is accurate.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Facility | Candidate origin that can be opened or closed | Assuming an available candidate is already operating |
| Customer zone | Aggregated destination receiving flow | Treating a zone centroid as every customer's address |
| Planning period | Common time bucket for demand, capacity, and cost | Combining monthly demand with annual capacity |
| Demand | Required shipment quantity, units per period | Treating a scenario as a customer commitment |
| Capacity | Maximum throughput, units per period | Substituting storage positions for throughput |
| Fixed cost | Cost incurred if a site opens, USD per period | Adding a one-time capital cost without annualization |
| Variable cost | Shipping cost per unit on a site–zone lane, USD per unit | Confusing a per-trip rate with a per-unit rate |
| Lane | Permitted origin–destination pair | Assuming every geographic pair is operationally feasible |
| Binary variable | Decision restricted to zero or one | Treating a fractional relaxation as an opening plan |
| Flow variable | Nonnegative shipment quantity, units per period | Assuming continuous flow means indivisible orders are feasible |
| Objective | Mathematical quantity minimized or maximized | Interpreting omitted costs as zero in the real business |
| Constraint | Mathematical restriction defining admissible choices | Treating a policy target as binding without approval |
| Feasible solution | Values satisfying all model restrictions within tolerance | Equating feasibility with optimality |
| Optimal solution | Feasible solution with no better model objective, according to a completed solve | Calling a time-limited incumbent proven optimal |
| Infeasible model | No assignment satisfies all constraints | Reporting missing variable values as zeros |
| Residual | Difference between a constraint's left and right sides | Ignoring the sign and the numerical tolerance |
| Slack | Unused capacity or margin, units per period | Treating unused capacity as available without checking operations |
| Scenario | A specified set of changed assumptions | Interpreting its result as a forecast probability |
| Stress test | Deliberately adverse scenario used to probe sensitivity | Calling one stress a complete resilience analysis |
| Baseline | Reference case against which variants are compared | Treating a synthetic baseline as a real current state |
| Solver status | Solver's reported termination classification | Assuming a solver always returns an optimum |
| Objective recomputation | Independent calculation of cost from returned decisions | Trusting a printed objective without checking flows |
| Tolerance | Permitted numerical discrepancy when validating floating-point values | Using a tolerance so large that a real shortage passes |
| Single sourcing | Requiring a customer to receive all demand from one site | Assuming continuous shipment variables enforce it |
| Recourse | A decision made after a scenario is known | Claiming a reoptimized scenario is immediately executable |
| Time limit | Allowed solver runtime before termination | Treating a stopped incumbent as a proof of optimality |

## 5. The mechanics, step by step

### Step 1 — Frame the business choice

**Input:** a decision owner, alternatives, service obligations, planning horizon, and decision deadline.

**Action:** ask whether the decision concerns strategic facility selection, tactical allocation, or a temporary disruption.
Write down which facilities may be opened, which are already committed, and which choices require approval.
A strategic study might span years; this exercise represents one synthetic operating period.
Do not combine a one-time investment with a period operating cost without a documented conversion.

**Output:** a decision statement: choose sites and flows to serve two zones at minimum modeled period cost.

**Branch:** if the question is about dispatching individual trucks, this aggregate model is not enough.
Vehicle routing needs vehicle capacities, stops, travel times, and other restrictions absent here.
If the question is about an immediate outage, fix the set of sites already open before changing availability.

### Step 2 — Define the boundary and units

**Input:** the decision statement and financial calendar.

**Action:** use one synthetic period throughout.
Measure demand, flow, and capacity in units per period.
Measure fixed site costs in USD per period and lane rates in USD per unit.
A quantity times a lane rate yields USD per period.
Add fixed and variable costs only after this dimensional check.

**Output:** a units ledger and a list of excluded processes.
The fixture excludes inventory, taxes, production constraints, transfer lead times, and service penalties.
It also assumes facility capacity can be fully shared across zones.

**Branch:** if capacity is measured in pallets and orders in cases, obtain a reliable conversion.
If product types consume capacity differently, use a common capacity-equivalent unit or introduce separate product flows.
If no trustworthy conversion exists, stop rather than manufacture comparable numbers.

### Step 3 — Build a site table

**Input:** candidate-site identifiers, capacities, and period fixed costs.

**Action:** use the following synthetic fixture.

| Site | Capacity, units/period | Fixed cost, USD/period |
|---|---:|---:|
| A | 100 | 100 |
| B | 80 | 90 |
| C | 80 | 110 |

Each identifier is unique and represents one alternative facility.
Capacity is a throughput ceiling, not physical storage.
The cost is incurred once if the site is open, regardless of utilization.
Zero or negative capacities and costs require a business explanation in a real dataset.
The code validates the given fixture and rejects negative or non-finite values.

**Output:** three candidate sites and 260 units/period of total nominal capacity.
That total alone does not imply every customer can be reached.
Here every lane is allowed, so aggregate capacity is sufficient to establish that this particular fixture can serve any nonnegative demand totaling at most 260 units/period.
With restricted lanes or other constraints, the aggregate test is necessary but not sufficient.

### Step 4 — Build demand and lane tables

**Input:** demand by zone and transportation rates by site–zone pair.

| Zone | Baseline demand, units/period |
|---|---:|
| East | 70 |
| West | 60 |

| Site | East, USD/unit | West, USD/unit |
|---|---:|---:|
| A | 2 | 8 |
| B | 5 | 3 |
| C | 4 | 4 |

**Action:** verify all six site–zone combinations have a rate.
The quantities are not rounded from actual shipment records.
The total baseline demand is 130 units/period.
A is the cheapest East origin and B is the cheapest West origin, before considering fixed costs and capacity.
Their combined capacity is 180 units/period.

**Output:** complete demand and cost mappings in a single currency and period.

**Branch:** a missing lane must not silently become a zero-cost lane.
Either obtain an approved cost or explicitly exclude that lane and formulate the model over the allowed-lane set.
This lab requires a complete matrix and fails input validation when a pair is absent.

### Step 5 — Declare sets, parameters, and variables

Let `I = {A, B, C}` be candidate sites and `J = {East, West}` be zones.
For site `i`, let `K_i` be throughput capacity and `F_i` fixed cost.
For zone `j`, let `D_j` be demand; let `C_ij` be the unit cost on lane `i,j`.
For each site define `y_i` as one if open and zero if closed.
For each lane define `x_ij >= 0` as units shipped in the period.

**Output:** three binary opening variables and six nonnegative continuous flow variables.
The output is an aggregate quantity plan, not a list of individual trips or orders.

### Step 6 — Express the objective

Minimize:

```text
sum over i of F_i * y_i + sum over i,j of C_ij * x_ij
```

The first term is USD per period; the second is also USD per period.
When a site is open, its fixed cost enters once.
A positive flow adds the corresponding per-unit lane cost.
The objective includes no unserved-demand penalty because all demand must be served.
It contains no resilience reward or carbon cost.

**Output:** a scalar modeled cost for each feasible plan.

**Branch:** if a cost has both a fixed charge and a volume-dependent component, model both explicitly.
If all-in delivery cost is unavailable, label the ranking provisional instead of claiming total business cost.

### Step 7 — Enforce customer service

For every zone `j`, impose:

```text
sum over i of x_ij = D_j
```

This equality requires exactly the stated demand to be shipped.
It rules out both shortage and oversupply in the model.
It does not specify shipment timing within the period.
It also permits splitting a zone across multiple facilities.

**Output:** one demand-balance constraint per zone.

**Branch:** if shortages are a real option, introduce a nonnegative shortage variable with a defensible penalty or explicit service requirement.
Do not interpret an infeasible hard-service model as an implicit shortage plan.
If customers require exactly one origin, add assignment binaries and linking constraints; aggregate continuous flows do not enforce that policy.

### Step 8 — Link capacity to opening

For each site `i`, impose:

```text
sum over j of x_ij <= K_i * y_i
```

When `y_i = 0`, nonnegative flows and zero right-hand side force all site shipments to zero.
When `y_i = 1`, total outbound flow cannot exceed `K_i`.
Using the actual capacity as the coefficient is tighter and more interpretable than an arbitrary huge constant.
If a site is unavailable, additionally force `y_i = 0`.

**Output:** three capacity constraints and, in the outage scenario, an explicit closure constraint.

**Branch:** if there are multiple limiting resources, add a capacity constraint for each resource.
A single throughput capacity cannot represent separate labor, dock, refrigeration, and storage bottlenecks unless they have been defensibly converted into one unit.

### Step 9 — Check input and solver prerequisites

**Input:** fixture, installed PuLP package, and an executable CBC solver.

**Action:** validate unique identifiers, nonnegative finite values, required lane costs, and known outage IDs.
Fail before constructing a model if these checks do not hold.
Use a virtual environment and a package index approved for the sandbox.
A PuLP installation normally bundles a CBC executable on supported platforms, but availability is platform dependent.
Check the solver's availability and install an approved CBC executable or use a supported environment if it is absent.
No credentials, customer data, network service, or cloud account are needed for the run after installation.
Installation itself may need network access and authorization.

**Output:** an auditable input snapshot and a usable local solver.

**Branch:** if the solver is unavailable, report the dependency problem rather than treating it as model infeasibility.
A license or package policy is an organizational matter; this example does not grant deployment permission.

### Step 10 — Solve and classify termination

**Action:** build a new model for each scenario, then call CBC.
The code asks for no time limit and requires the status `Optimal` before reading a plan.
It reports `Infeasible` without trying to validate nonexistent decision values.
It raises on other statuses so that an undefined or unbounded solve is not mislabeled as a usable result.
In larger applications, distinguish feasible incumbents, limits, best bounds, and gaps using the solver's own APIs and logs.

**Output:** a solver status and, only for an optimal solve, values for the openings and flows.

**Branch:** if the status is infeasible, inspect demand, capacity, prohibited lanes, and policy constraints.
For this fixture an aggregate capacity shortfall proves infeasibility in the last stress scenario.
When aggregate capacity is sufficient, examine reachability and other restrictions before concluding the data are sound.

### Step 11 — Validate the returned candidate

**Action:** check that every opening is numerically close to zero or one.
Check every flow is finite and nonnegative within a small tolerance.
Recompute every zone's inbound total and compare it with demand.
Recompute each site's outbound total and compare it with `capacity * opening`.
Check required closures explicitly.
Recompute fixed cost, variable cost, and their sum from the input data and decisions.
Compare the sum with the solver objective.

**Output:** validated flow, slack, and cost components.

**Branch:** if any check fails, stop and investigate the model, extraction, input mapping, or tolerances.
Do not publish the solver objective as a successful result while a residual fails.
Validation of a feasible candidate does not prove optimality independently of CBC.

### Step 12 — Run controlled scenarios

Change one assumption at a time where possible.
The baseline has the original demand and all facilities available.
The outage prohibits A but permits the model to choose B and C.
The demand stress raises East demand to 110 units/period and retains West demand at 60 units/period.
The infeasible stress sets East to 200 and West to 100 units/period, exceeding all sites' combined 260-unit capacity.
Costs and capacities otherwise stay fixed.

**Output:** scenario status, openings, flows, cost, and capacity slack where a solution exists.

**Branch:** a closure result is a redesigned network, not a guarantee of immediate failover.
A demand scenario is one chosen future, not a prediction interval.
Compare only scenarios with aligned units and cost definitions.

### Step 13 — Prepare a decision record

Record the exact input fixture, package version, solver status, and scenario changes.
Show the baseline and alternatives with fixed and variable components rather than just one cost total.
Describe assumptions behind capacity, availability, and delivery rates.
Separate decisions that the model makes from approvals the model cannot make.
Obtain owner review before using any result beyond the sandbox.

The final artifact should answer: what was optimized, what was held constant, what changed, what was feasible, and what evidence remains missing?
An apparent cost advantage is not a signed lease, a service commitment, or proof of resilience.

### Step 14 — Separate network design from emergency response

The outage exercise can answer more than one question, depending on what is held fixed.
Write the decision timing beside each scenario before interpreting a result.
Consider these distinct questions:

1. Which facilities would be cheapest if A were permanently unavailable?
2. Can the currently open facilities handle demand immediately after A fails?
3. How much additional capacity would be required to restore full service?
4. What might recovery cost if C could be activated after a documented lead time?

The supplied outage run answers the first question.
It prohibits A and allows B and C to be selected in a fresh optimization.
It does not impose a launch lead time on C.
It does not model transfer of inventory from A.
It assumes the original lane rates remain valid after an outage.
Those assumptions are reasonable for a small mathematical demonstration but need review in a real recovery plan.

The second question fixes the baseline footprint before the outage.
A is then unavailable and C remains closed.
The model can use only B, with 80 units/period of capacity.
Baseline demand is 130 units/period.
The resulting deficit is at least 50 units/period if full service must continue.
The deficit is a capacity calculation, not a prediction of lost sales.
It says nothing about which customer receives the available 80 units.
The hard-demand model returns infeasible rather than selecting customers to short.

For the third question, 50 units/period is an aggregate lower bound on added capacity.
That lower bound assumes the added capacity can reach whichever zones need service.
It also assumes no other constraint limits the shipment.
A temporary facility with only 30 units/period available would not close the 50-unit gap.
A temporary facility with 50 units/period might close it, but only if its lanes and operating resources are feasible.
Its fixed and variable emergency costs would be additional input assumptions.
Do not substitute a normal-period lane rate for an emergency rate without approval.

For the fourth question, agree on a recovery horizon before combining costs.
A one-day interruption and a full-period redesign cannot be compared by adding their costs directly.
Separate transition cost, disrupted-period cost, and steady-state cost.
If the decision spans several periods, use consistent period definitions and discounting approved by finance.
Document whether the candidate site's capacity is usable on day one or ramps up gradually.
Document what demand is deferred, lost, substituted, or served by an external partner.
This lab deliberately does not choose among those policy alternatives.

### Step 15 — Inspect the extracted result as a table

A solver result is easier to audit when transformed into two small views.
The first view is one row per site.
For each row, display opening, available capacity, outbound flow, slack, and fixed cost.
The second view is one row per zone.
For each row, display demand, inbound flow, and balance residual.
The report in section 7 prints positive flows and open-site slack.
The validator still checks closed sites and zero-flow lanes even when the report suppresses them.

For the baseline, an independently prepared site view is:

| Site | Open | Available capacity, units/period | Outbound, units/period | Slack, units/period | Fixed, USD/period |
|---|---:|---:|---:|---:|---:|
| A | 1 | 100 | 70 | 30 | 100 |
| B | 1 | 80 | 60 | 20 | 90 |
| C | 0 | 0 | 0 | 0 | 0 |

Available capacity is `capacity * opening`.
It is not the same as the site's nameplate capacity when the site is closed.
The closed site's slack is zero under that definition.
Showing C's 80-unit nameplate capacity as available slack would misrepresent the solution.
The sum of open-site slack is 50 units/period in this baseline.
That number is a capacity margin in the model, not a guarantee of physical surge capability.

The matching zone view is:

| Zone | Demand, units/period | Inbound, units/period | Inbound minus demand, units/period |
|---|---:|---:|---:|
| East | 70 | 70 | 0 |
| West | 60 | 60 | 0 |

If a zone residual is nonzero beyond the accepted tolerance, the plan fails validation.
If a site has negative slack beyond tolerance, the plan exceeds modeled capacity.
If the fixed-cost column includes a closed site, the reporting logic has diverged from the decision variables.
If a flow is positive on a closed site, the model or extraction is incorrect.
If the printed total differs from recomputed cost, do not repair the display by rounding away the discrepancy.
Locate the cause in the input mapping, solution extraction, or objective construction.

### Step 16 — Interpret capacity changes without inventing shadow prices

The high-East scenario saturates A.
It is tempting to claim that one additional unit of A capacity always saves a fixed amount.
That is only true over a range in which the optimal opening set and marginal allocation remain unchanged.
For a small local thought experiment, increase A capacity from 100 to 101 units/period.
Keep East demand at 110 and West demand at 60 units/period.
With A and B open, send 101 East units from A, 9 East units from B, and 60 West units from B.
The lane-cost change replaces one B-to-East unit costing 5 USD with one A-to-East unit costing 2 USD.
The resulting candidate costs 617 USD/period, a 3 USD/period reduction relative to 620.

The candidate is feasible: A uses 101 of 101 units/period and B uses 69 of 80 units/period.
For the A/B opening set, A is the cheapest East source and B is the cheapest West source.
Thus that allocation is optimal within the set.
A/C costs at least 210 + 101 × 2 + 9 × 4 + 60 × 4 = 688 USD/period.
B/C remains capacity-infeasible at 160 units/period.
Opening all three costs at least 300 + 101 × 2 + 9 × 4 + 60 × 3 = 718 USD/period.
Therefore the 617 USD/period candidate is also the best opening set for this modified fixture.

This calculation does not price a real capacity investment.
An additional unit of throughput may require labor, equipment, space, safety controls, or capital.
The 3 USD/period operating-cost reduction omits those costs.
It also applies only to this specific demand, rate, and capacity change.
After ten extra units of A capacity, the modeled need to ship East from B disappears.
Beyond that point, further A capacity cannot reduce this particular plan's lane cost.
Discrete site-opening changes can make marginal values discontinuous.
Do not apply a linear-program dual value from a fixed opening pattern as a universal MILP investment value.

### Step 17 — Design negative tests before expanding the model

A useful negative test starts with a known valid plan and makes one targeted change.
Specify the expected violation before running the validator.
Keep other values unchanged so that the failed condition is interpretable.
The lab includes a changed-flow test and a missing-cost test.
The following table gives further cases that a learner can implement.

| Test | Controlled alteration | Expected outcome |
|---|---|---|
| Nonbinary opening | Replace A's opening with 0.5 | Reject the opening before accepting the plan |
| Negative flow | Replace a zero flow with −1 unit/period | Reject the negative shipment |
| Excess capacity | Increase A-to-East flow beyond 100 units/period | Reject A's capacity violation |
| Required closure | Require A closed while retaining the baseline opening | Reject the closure violation |
| Wrong objective | Add 1 USD/period to the reported objective | Reject the cost mismatch |
| Unknown closure | Request an outage at site D | Reject the scenario input |
| Nonfinite rate | Set a lane rate to positive infinity | Reject the fixture before solving |
| Missing lane | Delete A-to-East from the rate matrix | Reject the incomplete matrix |
| Negative demand | Set East demand to −1 unit/period | Reject the fixture before solving |
| Insufficient total capacity | Require 300 units/period with only 260 available | Return infeasible without a plan |

A test that violates more than one condition can still fail correctly, but its reported reason may depend on check order.
For example, increasing A's baseline shipment by one unit violates demand balance and also changes the recomputed cost.
The implementation checks demand before objective consistency, so the demand check fails first.
To isolate an objective mismatch, leave the flows untouched and change only the reported objective.
To isolate a capacity violation, consider a flow redistribution that preserves demand while overloading one site.
The correct test oracle is the intended rule, not whatever message happened to appear once.

There are limits to the provided tests.
They do not enumerate all feasible solutions for arbitrary fixtures.
They do not independently implement a second optimization algorithm.
They do not check solver behavior on all operating systems.
They do not establish numeric stability for extremely large coefficients.
They do not check real-world demand, contracts, or transportation rates.
Add independent tests when a new rule is introduced; do not weaken an existing check merely to make a new scenario pass.

### Step 18 — Hand off a scenario comparison

Present each scenario as a separate row with a status and a clearly stated change.
For the synthetic cases, a concise decision view is:

| Scenario | Change from baseline | Status | Open sites | Total, USD/period | Key caveat |
|---|---|---|---|---:|---|
| Baseline | None | Optimal | A, B | 510 | Synthetic deterministic input |
| A unavailable | Prohibit A; allow redesign | Optimal | B, C | 660 | C can be newly opened in the model |
| High East | East 110, West 60 units/period | Optimal | A, B | 620 | A has zero modeled slack |
| Shortfall | East 200, West 100 units/period | Infeasible | Not applicable | Not applicable | Demand exceeds all-site capacity by 40 units/period |

Keep `not applicable` distinct from zero cost.
An infeasible case has no valid objective value under the full-service model.
The A-unavailable cost difference is 150 USD/period for the specified redesign.
It is not an estimate of the full financial impact of a real outage.
The high-East cost difference is 110 USD/period, but demand also increases by 40 units/period.
Do not rank the scenarios as if they represented identical service volumes.

The business choice is not limited to accepting the minimum-cost baseline.
One option is to investigate the A/B design as a low-cost reference.
Another is to price C's standby or activation capability if resilience is important.
A third is to acquire or contract additional capacity for high-demand periods.
Each option requires additional cost, lead-time, and service evidence.
The existing model does not price standby arrangements or compare contract terms.
The immediate next action is to validate the underlying operational assumptions with the accountable owners.
Only then should a decision maker authorize a more detailed design or procurement process.

## 6. Worked example

### Baseline: open A and B

East requires 70 units/period and West requires 60 units/period.
Open A and B, ship all East demand from A, and ship all West demand from B.
A uses 70 of 100 units/period and has 30 units/period slack.
B uses 60 of 80 units/period and has 20 units/period slack.
C is closed, ships zero, and incurs no fixed cost.

| Component | Arithmetic | USD/period |
|---|---|---:|
| A fixed | 100 × 1 | 100 |
| B fixed | 90 × 1 | 90 |
| C fixed | 110 × 0 | 0 |
| A to East | 70 × 2 | 140 |
| B to West | 60 × 3 | 180 |
| Other lanes | 0 units × lane rates | 0 |
| Total | 100 + 90 + 140 + 180 | 510 |

The plan serves 70 East units and 60 West units with 130 total units.
Every customer balance is exact and neither capacity is exceeded.
CBC proves optimality for this fixture; the following enumeration provides an independent explanation.
No single site can serve all 130 units because the largest capacity is 100 units/period.
For A and B, the lowest possible lane cost is 70 × 2 + 60 × 3 = 320 USD/period and both cheapest assignments fit.
Adding their fixed cost gives 510 USD/period.

For A and C, assign East to A and West to C.
That produces 140 + 240 = 380 USD/period variable and 210 USD/period fixed, or 590 USD/period total.
Both flows fit their capacities.
For B and C, assign East to C and West to B.
That produces 280 + 180 = 460 USD/period variable and 200 USD/period fixed, or 660 USD/period total.
Both flows fit their capacities.
With all three open, A and B already provide the cheapest per-zone lane costs, so variable cost cannot fall below 320 USD/period.
Fixed cost becomes 300 USD/period, giving 620 USD/period.
Thus the feasible opening sets have totals 510, 590, 660, and 620 USD/period; A and B are cheapest.
This enumeration works because the fixture is tiny and each cheapest assignment respects capacity.
It is not a scalable method for a large network.

### A unavailable: redesigned network

If A is prohibited, neither B nor C can serve 130 units/period alone.
Open B and C; send 70 East units from C and 60 West units from B.
C has 10 units/period slack and B has 20 units/period slack.
Fixed cost is 90 + 110 = 200 USD/period.
Variable cost is 70 × 4 + 60 × 3 = 460 USD/period.
The total is 660 USD/period, 150 USD/period above the baseline.
This scenario allows C to be opened after A becomes unavailable.
If C is not already operating or cannot be activated immediately, this result is not an outage recovery commitment.

### Higher East demand

Set East demand to 110 units/period; retain 60 West units/period.
A and B jointly have 180 units/period capacity, enough for 170 units/period demand.
Use A for 100 East units, B for the remaining 10 East units, and B for 60 West units.
A has zero slack and B has 10 units/period slack.
Fixed cost is 190 USD/period.
Variable cost is 100 × 2 + 10 × 5 + 60 × 3 = 430 USD/period.
Total cost is 620 USD/period, 110 USD/period above the baseline.
This increase reflects both more volume and a different allocation; it is not a like-for-like unit-cost comparison.

For A and C, total capacity is also 180 units/period.
The cheapest allocation sends 100 East units from A, 10 East and 60 West from C.
Its total is 210 + 200 + 40 + 240 = 690 USD/period.
B and C provide only 160 units/period, so they cannot serve 170 units/period.
Opening all sites costs at least 300 + 100 × 2 + 10 × 4 + 60 × 3 = 720 USD/period.
Therefore 620 USD/period is the best of the four possible multi-site sets.

### Capacity-shortfall stress

The final stress requests 200 East and 100 West units/period, or 300 total.
All facilities together provide only 100 + 80 + 80 = 260 units/period.
The unavoidable aggregate shortfall is at least 40 units/period.
Because the model requires exact service and has no shortage variable, no solution exists.
The result must be `Infeasible`, not a fictitious 260-unit plan or an objective of zero.

## 7. Code, query, or calculation

### Sandbox setup

Use a new empty working directory and Python 3.10 or later.
The following example uses a local virtual environment; PowerShell activation is optional because the interpreter can be invoked directly.
On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install PuLP==2.9.0
```

On macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install PuLP==2.9.0
```

Pinning PuLP makes the package version explicit but is not a complete lock of platform binaries or transitive artifacts.
Record the operating system, Python version, PuLP version, and CBC version for a reproducible experiment.
Use only trusted, approved packages and local synthetic input.
PuLP's bundled CBC availability depends on the distribution and platform.
The script checks availability and fails with an actionable message if CBC cannot be started.

Create `network_lab.py` in the working directory with the following complete contents.
The fixture, implementation, output, and checks are all in this one file; no external CSV is required.

```python
import math
import pulp

SITES = {
    "A": {"capacity": 100.0, "fixed": 100.0},
    "B": {"capacity": 80.0, "fixed": 90.0},
    "C": {"capacity": 80.0, "fixed": 110.0},
}
DEMAND = {"East": 70.0, "West": 60.0}
COST = {
    ("A", "East"): 2.0, ("A", "West"): 8.0,
    ("B", "East"): 5.0, ("B", "West"): 3.0,
    ("C", "East"): 4.0, ("C", "West"): 4.0,
}
TOL = 1e-6


def nonnegative_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def validate_inputs(sites, demand, cost, closed):
    if not sites or not demand:
        raise ValueError("sites and demand must be nonempty")
    if not set(closed).issubset(sites):
        raise ValueError("closed site is not in the site table")
    if set(cost) != {(i, j) for i in sites for j in demand}:
        raise ValueError("lane-cost matrix must contain exactly every site-zone pair")
    for values in sites.values():
        if set(values) != {"capacity", "fixed"}:
            raise ValueError("site must have capacity and fixed cost")
        if not all(nonnegative_finite(v) for v in values.values()):
            raise ValueError("site values must be finite and nonnegative")
    if not all(nonnegative_finite(v) for v in demand.values()):
        raise ValueError("demand must be finite and nonnegative")
    if not all(nonnegative_finite(v) for v in cost.values()):
        raise ValueError("lane costs must be finite and nonnegative")


def check_solution(sites, demand, cost, closed, opening, flow, objective):
    """Recompute feasibility and costs without consulting model constraints."""
    if not math.isfinite(objective):
        raise ValueError("objective is not finite")
    for i in sites:
        y = opening[i]
        if not math.isfinite(y) or min(abs(y), abs(y - 1)) > TOL:
            raise ValueError(f"nonbinary opening at {i}: {y}")
        if i in closed and abs(y) > TOL:
            raise ValueError(f"required closure violated at {i}")
        used = 0.0
        for j in demand:
            x = flow[i, j]
            if not math.isfinite(x) or x < -TOL:
                raise ValueError(f"invalid flow on {i}->{j}: {x}")
            used += x
        if used > sites[i]["capacity"] * y + TOL:
            raise ValueError(f"capacity violated at {i}")
    for j in demand:
        served = sum(flow[i, j] for i in sites)
        if abs(served - demand[j]) > TOL:
            raise ValueError(f"demand balance violated at {j}: {served}")
    fixed = sum(sites[i]["fixed"] * opening[i] for i in sites)
    variable = sum(cost[i, j] * flow[i, j] for i in sites for j in demand)
    if not math.isclose(fixed + variable, objective, rel_tol=1e-9, abs_tol=TOL):
        raise ValueError("recomputed objective disagrees with solver")
    return fixed, variable


def solve(sites, demand, cost, closed=()):
    closed = frozenset(closed)
    validate_inputs(sites, demand, cost, closed)
    solver = pulp.PULP_CBC_CMD(msg=False)
    if not solver.available():
        raise RuntimeError("CBC unavailable; install an approved supported CBC/PuLP environment")
    model = pulp.LpProblem("facility_location", pulp.LpMinimize)
    # Integer indices avoid solver name normalization collisions in arbitrary IDs.
    ids = list(sites)
    zones = list(demand)
    y = {i: pulp.LpVariable(f"open_{n}", cat="Binary")
         for n, i in enumerate(ids)}
    x = {(i, j): pulp.LpVariable(f"flow_{n}_{m}", lowBound=0)
         for n, i in enumerate(ids) for m, j in enumerate(zones)}
    model += (pulp.lpSum(sites[i]["fixed"] * y[i] for i in ids)
              + pulp.lpSum(cost[i, j] * x[i, j] for i in ids for j in zones))
    for j in zones:
        model += pulp.lpSum(x[i, j] for i in ids) == demand[j]
    for i in ids:
        model += pulp.lpSum(x[i, j] for j in zones) <= sites[i]["capacity"] * y[i]
        if i in closed:
            model += y[i] == 0
    model.solve(solver)
    status = pulp.LpStatus[model.status]
    if status == "Infeasible":
        return {"status": status}
    if status != "Optimal":
        raise RuntimeError(f"No proven optimum; solver status: {status}")
    opening = {i: float(pulp.value(y[i])) for i in ids}
    flow = {(i, j): float(pulp.value(x[i, j])) for i in ids for j in zones}
    objective = float(pulp.value(model.objective))
    fixed, variable = check_solution(sites, demand, cost, closed, opening, flow, objective)
    slack = {i: sites[i]["capacity"] * opening[i] - sum(flow[i, j] for j in zones)
             for i in ids}
    return {"status": status, "opening": opening, "flow": flow,
            "fixed": fixed, "variable": variable, "objective": objective, "slack": slack}


def report(name, result):
    print(f"{name}: {result['status']}")
    if result["status"] != "Optimal":
        return
    opened = ",".join(i for i, y in result["opening"].items() if y > 0.5)
    print(f"  open={opened}; fixed=${result['fixed']:.2f}; "
          f"variable=${result['variable']:.2f}; total=${result['objective']:.2f}/period")
    for (i, j), amount in result["flow"].items():
        if amount > TOL:
            print(f"  {i}->{j}: {amount:.0f} units/period")
    for i, y in result["opening"].items():
        if y > 0.5:
            print(f"  {i} slack: {result['slack'][i]:.0f} units/period")


def expect_failure(function, message):
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def main():
    scenarios = [
        ("baseline", DEMAND, ()),
        ("A unavailable", DEMAND, ("A",)),
        ("East demand 110", {"East": 110.0, "West": 60.0}, ()),
        ("capacity shortfall", {"East": 200.0, "West": 100.0}, ()),
    ]
    results = {}
    for name, demand, closed in scenarios:
        results[name] = solve(SITES, demand, COST, closed)
        report(name, results[name])
    base = results["baseline"]
    assert base["status"] == "Optimal"
    assert abs(base["objective"] - 510.0) <= TOL
    assert {i for i, y in base["opening"].items() if y > 0.5} == {"A", "B"}
    assert abs(results["A unavailable"]["objective"] - 660.0) <= TOL
    assert abs(results["East demand 110"]["objective"] - 620.0) <= TOL
    assert results["capacity shortfall"]["status"] == "Infeasible"
    bad_flow = dict(base["flow"])
    bad_flow["A", "East"] += 1.0
    expect_failure(lambda: check_solution(SITES, DEMAND, COST, (), base["opening"],
                                          bad_flow, base["objective"]),
                   "validator accepted a demand imbalance")
    missing_cost = dict(COST)
    del missing_cost["A", "East"]
    expect_failure(lambda: validate_inputs(SITES, DEMAND, missing_cost, ()),
                   "validator accepted a missing lane cost")
    print("All lab checks passed.")


if __name__ == "__main__":
    main()
```

Run from the directory containing the file.
Do not invoke Python with `-O`, which disables the script's `assert` statements.

```powershell
.\.venv\Scripts\python network_lab.py
```

Or on macOS/Linux:

```sh
.venv/bin/python network_lab.py
```

Expected output, with the same site and zone ordering as the fixture:

```text
baseline: Optimal
  open=A,B; fixed=$190.00; variable=$320.00; total=$510.00/period
  A->East: 70 units/period
  B->West: 60 units/period
  A slack: 30 units/period
  B slack: 20 units/period
A unavailable: Optimal
  open=B,C; fixed=$200.00; variable=$460.00; total=$660.00/period
  B->West: 60 units/period
  C->East: 70 units/period
  B slack: 20 units/period
  C slack: 10 units/period
East demand 110: Optimal
  open=A,B; fixed=$190.00; variable=$430.00; total=$620.00/period
  A->East: 100 units/period
  B->East: 10 units/period
  B->West: 60 units/period
  A slack: 0 units/period
  B slack: 10 units/period
capacity shortfall: Infeasible
All lab checks passed.
```

The expected values follow from the arithmetic in section 6.
The checks cover the baseline optimum and selected stress totals, the infeasible status, a tampered flow, and a missing lane.
The explicit validator also checks all returned flows, demand balances, capacity limits, closed sites, binary openings, and the objective for every optimal scenario.
A passing check shows that this example meets its stated contract; it does not validate a production dataset or a solver implementation.
The code has no time limit and does not demonstrate how to accept an unproven incumbent.
Use small fixtures and an approved environment; do not run a large model on shared infrastructure without resource controls.

## 8. Data, standards, and regulatory context

### Data contract

A production input snapshot needs stable site and zone keys, a period identifier, a currency, and unit definitions.
It should distinguish proposed facilities from active sites and committed sites.
Supply a site capacity with its effective dates, measurement method, and operational owner.
The lane cost should identify whether it includes line haul, fuel, handling, accessorial charges, and returns.
The demand table needs a forecast vintage and the time bucket to which the quantity applies.
A real forecast can be revised; save the exact version used in the model run.

Do not join tables on free-text city names when stable keys are available.
An inner join that drops missing lanes can make a model appear easier while silently omitting customers or facilities.
Count rows before and after joins, compare expected key combinations, and investigate unmatched records.
The lab's Python dictionaries cannot contain duplicate keys after construction; an upstream CSV loader must detect duplicates before reducing rows into a dictionary.
Check for duplicate IDs, nulls, unexpected units, stale effective dates, negative quantities, and missing rates before modeling.
Keep the source record and transformation logic available for review.

### Location and product identifiers

GS1 Global Location Numbers (GLNs) can identify legal entities, functions, and physical or digital locations when used under the applicable GS1 rules.
A GLN is an identifier, not evidence that a facility is available, licensed, or able to handle a product.
GS1 Global Trade Item Numbers (GTINs) can identify trade items, but a GTIN alone does not specify throughput equivalence or transport capacity.
Map site and product identifiers to the local planning granularity with documented effective dates.
Do not publish confidential mappings in a public lab repository.

### Accounting and legal boundary

The model uses a synthetic currency and does not determine the accounting treatment of capital expenditure, leases, depreciation, or taxes.
Finance must approve how real one-time and recurring costs are converted into comparable period values.
Land use, environmental permits, labor obligations, transport rules, and customer contracts can restrict real choices.
Applicable requirements depend on jurisdiction, commodity, and operating arrangement; this lab does not identify all applicable law.
Consult qualified legal, tax, safety, and operations reviewers before implementing a facility decision.
A modeling convention is not itself a binding legal requirement.

### Security and retention

Real network locations, customer volumes, and negotiated rates can be commercially sensitive.
Apply the organization's access controls and retention rules to input snapshots and scenario output.
Remove credentials from scripts and logs.
Avoid sending private lane data to an external optimization service without approved contractual and security controls.
This lab runs locally after dependencies are installed and uses no external solver service.
Keep synthetic fixtures separate from operational inputs so demonstrations cannot accidentally leak client data.

## 9. Industry practice and case evidence

### Evidence and inference

The mathematical facility-location problem is a standard operations-research model, not a claim that any particular enterprise uses this exact fixture.
Cornuéjols, Nemhauser, and Wolsey survey uncapacitated facility-location models in the cited chapter.
The lab extends the basic opening-and-allocation idea with throughput capacities.
The cited literature supports the formulation's place in operations research; it does not establish the fixture's business parameters.

UPS publicly describes ORION as a route optimization system for delivery operations.
That documented application concerns route planning, not proof that UPS uses this lab's strategic facility-location formulation.
**Inference:** facility location and delivery routing should be checked together because changing depot locations changes the route-planning problem.
The lab uses static per-unit lane costs and does not reproduce UPS's system or report its results.

A strategic network team typically needs finance, real estate, transport, warehouse operations, customer service, and data owners to review alternatives.
This is a recommended governance practice rather than a claim about a specific company's internal process.
Finance can review cost completeness and the time horizon.
Operations can challenge effective capacity and launch timing.
Commercial owners can review customer service constraints and approved trade-offs.
Legal and safety reviewers can identify facility and commodity restrictions.

A mathematical optimum can be fragile when inputs are uncertain.
A team may prefer a slightly more expensive plan if a stress test exposes lower dependency on one site.
That preference is not present in this single-objective formulation unless it is translated into constraints, scenarios, or another explicit objective.
The decision record should show the trade-off instead of claiming that the optimizer automatically values resilience.

### Scenario design in practice

An outage is not always equivalent to closing a site before the planning period begins.
A long-term redesign may include new facilities and renegotiated lanes.
An immediate outage may be limited to facilities already open, pre-positioned inventory, available labor, and contracted carriers.
The code implements the first kind of counterfactual by allowing other sites to open.
A fixed-network outage model would instead constrain the baseline's opening decisions and adjust only allocations.
Under the baseline A/B footprint, an A outage leaves only B's 80 units/period capacity against 130 units/period demand, so hard full service would be infeasible.
Whether a real firm could source emergency capacity is an additional operational question, not an output of this model.

Likewise, higher demand is not the same as a higher service target.
The stress increases East volume while retaining the hard 100% fulfillment constraint.
An actual business may want multiple demand scenarios with probabilities, multiple periods, and a separate policy on shortages.
Those additions require explicit definitions and validated data; they are not implied by this example.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Mixed time buckets | Demand and capacity have different calendars | Incomparable quantities are combined | False feasibility or shortage | Normalize to one period and document conversion |
| Missing lane interpreted as free | Unexpected zero-cost arcs | Null becomes zero | Misleading site ranking | Require complete rates or explicit prohibited lanes |
| Duplicate source keys | Row count exceeds unique key count | A join multiplies demand or rates | Inflated cost and capacity usage | Reject duplicates before creating mappings |
| Unrealistic capacity | Modeled load exceeds historical achievable throughput | Nameplate capacity ignores downtime | Plan cannot be executed | Review effective capacity by shift and resource |
| Missing fixed costs | Every site opens | Objective omits operating cost | Unaffordable recommendation | Reconcile fixed cost categories with finance |
| Incorrect capacity linkage | Closed facility has positive flow | Capacity not multiplied by binary opening | Physically impossible plan | Independently validate `flow <= capacity * opening` |
| Incorrect demand balance | Zone receives too little or too much | Wrong index, inequality, or units | Service commitment not met | Check each zone residual from extracted values |
| Status ignored | Missing values or stale objective after solve | Infeasible or interrupted solve treated as optimum | Invalid recommendation | Branch on termination status before extracting plan |
| Loose tolerance | Material shortage passes validation | Tolerance exceeds meaningful business quantity | Bad plan appears feasible | Choose tolerance relative to units and solver precision |
| False outage readiness | Model opens replacement instantly | Reoptimization assumes unavailable lead time | Disruption response fails | Fix existing network for immediate recovery scenarios |
| Aggregation bias | Good zone average hides distant customers | Centroid or zone average masks variation | Missed service windows | Test finer geography and routing feasibility |
| Split shipment hidden | One zone receives multiple origins | Continuous flows allow splits | Policy or order constraints violated | Add assignment restrictions if single sourcing is required |
| Cost horizon mismatch | One-time build cost added to one month of freight | Costs cover different time horizons | Wrong investment comparison | Convert with approved finance method and horizon |
| Confusing proof and reality | Report says globally best network | Optimality is conditional on input and formulation | Overconfident decision | State omitted constraints and validate with owners |
| Unmodeled emissions or risk | Lowest-cost design has unacceptable exposure | Objective excludes important outcome | Rejected or harmful recommendation | Add approved constraints or compare explicit alternatives |
| Stale network data | New site or lane absent | Snapshot predates operational changes | Missing options or invalid choices | Record effective date and owner sign-off |

## 11. Assumptions and limits

| Assumption | Why used | What breaks it | How to test |
|---|---|---|---|
| One period | Keeps units and math transparent | Seasonal peaks or phased openings | Solve several periods with dated capacity and demand |
| Deterministic demand | Enables a single comparable case | Demand uncertainty or correlated shocks | Run plausible scenarios and compare decisions |
| Complete lane availability | Fixture contains six rates | Geography, regulation, or contracts prohibit a lane | Obtain an approved lane eligibility table |
| Continuous divisible flow | Linear allocation is simple | Indivisible orders or minimum truckloads | Compare with order-level integer or routing model |
| Fixed cost independent of volume | Represents site activation | Step costs, overtime, or minimum leases | Reconcile cost curve with finance and operations |
| Constant lane unit rate | Keeps objective linear | Distance tiers, vehicle loads, fuel changes | Compare piecewise tariffs or route-based costs |
| Capacity applies to all zones equally | One resource constraint per site | Product-specific handling or dock restrictions | Measure workload and add resource-specific constraints |
| Exact demand fulfillment | Models mandatory service | Shortage, substitution, or backorder is permitted | Specify authorized service and shortage policy |
| Instant scenario redesign | Reveals a long-run alternative | Leases and staffing require lead time | Fix previously open facilities and add transition timing |
| No inventory or lead-time effects | Focuses on network flow | Stock placement or delivery promises bind | Integrate inventory and lead-time feasibility studies |
| Input costs are comparable USD/period | Supports an additive objective | Foreign currency, taxes, or horizon mismatch | Review rates, dates, and financial conversion |
| CBC finds a proven optimum | Makes objective comparisons interpretable | Large model stops early or is numerically difficult | Record status, bounds, gap, settings, and logs |

The sample has no stochastic process, travel-time matrix, product mix, emissions model, or capital investment schedule.
Its solution is an educational plan for a synthetic period, not a real operational recommendation.
A zero modeled slack does not mean every physical hour is scheduled or the site can run at full utilization indefinitely.

## 12. Exercises

### Exercise 1 — Translate the decision

**Time:** 10 minutes.
Write the meanings and units of `y_A` and `x_A,East`.
Explain the business implication of `y_A = 0` under the capacity-linking constraint.

**Deliverable:** two variable definitions and one sentence about closed-site flow.
**Acceptance:** the answer identifies the binary opening, units/period flow, and zero outbound flow from a closed site.

### Exercise 2 — Compute a candidate

**Time:** 15 minutes.
Use the baseline data and open only A and C.
Allocate East to A and West to C, then calculate fixed cost, variable cost, total cost, and site slack.

**Deliverable:** a calculation table and a feasibility statement.
**Acceptance:** every demand and capacity balance is checked and all cost amounts have USD/period units.

### Exercise 3 — Change a fixed cost

**Time:** 20 minutes.
Change C's fixed cost from 110 to 20 USD/period while leaving baseline demand, capacities, and rates unchanged.
Re-solve and explain which sites open and why the answer differs from the original baseline.

**Deliverable:** status, open sites, flow allocation, cost, and a short explanation.
**Acceptance:** the answer checks all feasible opening sets and does not count C's fixed cost when C is closed.

### Exercise 4 — Test the validator

**Time:** 15 minutes.
Construct a copy of the baseline opening values with B set to 0 while leaving its West flow unchanged.
Pass it to `check_solution` with the original flows and objective.

**Deliverable:** the expected error category and a one-sentence explanation.
**Acceptance:** the explanation identifies a capacity-link violation rather than calling it a demand imbalance.

### Exercise 5 — Diagnose infeasibility

**Time:** 15 minutes.
Explain why 200 East and 100 West units/period cannot be served by all three facilities.
Then give one example showing why enough total capacity would not guarantee feasibility if lanes were restricted.

**Deliverable:** an arithmetic proof and a restricted-lane counterexample.
**Acceptance:** the proof calculates the aggregate shortfall and the counterexample identifies an unreachable demand zone.

### Exercise 6 — Distinguish outage questions

**Time:** 20 minutes.
Compare the script's A-unavailable reoptimized scenario with an immediate outage in which only baseline sites A and B may be used and A fails.
Assume no emergency capacity, no shortage, and no new openings.

**Deliverable:** both statuses, a modeled cost if available, and a description of the differing assumptions.
**Acceptance:** the response does not present C's hypothetical opening as immediate failover.

### Exercise 7 — Write a decision memo

**Time:** 25 minutes.
Write a 150–250 word memo to a synthetic network owner comparing baseline, demand stress, and A outage.
Include modeled cost, capacity slack, missing data, and the next business approval.

**Deliverable:** a memo separating observed model output, assumptions, and decision request.
**Acceptance:** it does not claim real savings, forecast certainty, an approved lease, or demonstrated outage readiness.

### Exercise 8 — Design a policy extension

**Time:** 25 minutes.
Design an extension that forces every zone to be supplied by exactly one open site.
Name the new binary variables, assignment constraints, and links to the flow variables.
Discuss how this changes feasibility and what additional test you would write.

**Deliverable:** a formulation and one independently expected test case.
**Acceptance:** the model assigns each zone to exactly one open facility, limits flow to that assignment, and handles zones larger than any single site's capacity.

## 13. Self-check answers

### Answer 1

`y_A` is a dimensionless binary opening decision: 1 means A is open and 0 means closed.
`x_A,East` is the quantity shipped from A to East in units per period.
When `y_A = 0`, A's capacity right-hand side is zero.
Because all flows are nonnegative, every outbound A flow must be zero.

### Answer 2

A and C incur 100 + 110 = 210 USD/period fixed cost.
A ships 70 East units at 2 USD/unit, costing 140 USD/period.
C ships 60 West units at 4 USD/unit, costing 240 USD/period.
Variable cost is 380 USD/period and total cost is 590 USD/period.
A has 100 − 70 = 30 units/period slack; C has 80 − 60 = 20 units/period slack.
Both zones receive exactly their demand, so this candidate is feasible.
It is 80 USD/period more expensive than the 510 USD/period baseline optimum.

### Answer 3

A and C open, sending 70 East units from A and 60 West units from C.
Fixed cost is 100 + 20 = 120 USD/period; variable cost is 140 + 240 = 380 USD/period.
Total cost is 500 USD/period.
A and B still cost 510 USD/period.
B and C cost 90 + 20 + 280 + 180 = 570 USD/period.
All three cost 100 + 90 + 20 + 320 = 530 USD/period.
No single site can meet total demand, so 500 USD/period is optimal among all feasible opening sets.

### Answer 4

The validator raises a `ValueError` for capacity at B.
Its 60-unit West flow exceeds 80 × 0 = 0 units/period available capacity.
Demand itself still balances, so the violation is specifically the opening–capacity link.
The validator checks capacity before comparing the changed objective.

### Answer 5

Total demand is 200 + 100 = 300 units/period.
Total capacity is 100 + 80 + 80 = 260 units/period.
At least 40 units/period cannot be served, so the full-service model is infeasible.
For a restricted-lane counterexample, suppose East is reachable only from A, East demand is 110 units/period, and West demand is 60 units/period.
Total demand is 170 units/period, below total capacity of 260, but East exceeds A's 100-unit capacity.
Aggregate capacity alone cannot repair the unreachable 10 East units/period.

### Answer 6

The script allows B and C to open and returns an optimal 660 USD/period plan.
An immediate scenario fixed to the baseline footprint has only B available after A fails.
B supplies at most 80 units/period against 130 units/period demand, so full service is infeasible.
The 660 USD/period redesign does not demonstrate that C can be staffed or launched immediately.

### Answer 7 rubric

A strong memo gives 510, 620, and 660 USD/period for the respective scenarios and identifies the synthetic inputs.
It identifies A's zero slack in the high-East case and the ability to open C in the outage redesign.
It notes that incremental cost is not a real measured saving and that demand stress is not a probability forecast.
It requests validation of site capacity, rates, lead times, and contractual service requirements before approval.
A weak memo states that the model proves operational resilience or directs a real closure.

### Answer 8 rubric

Let `z_ij` be one when site `i` is assigned zone `j`.
Require `sum_i z_ij = 1` for each zone and `z_ij <= y_i` for each lane.
Link continuous flow with `x_ij <= D_j * z_ij` and retain demand equality and site capacity constraints.
A zone whose demand exceeds every individual site's capacity becomes infeasible under single sourcing even when aggregate network capacity is sufficient.
A good test uses such a zone and checks the infeasible status, then separately checks that a smaller feasible zone has exactly one assigned site.
If zero-demand zones need no assignment, specify that business rule rather than silently accepting arbitrary assignments.

## 14. Interview questions

1. **How does the model stop a closed facility shipping?**
   Strong: explains nonnegative flows and `sum_j x_ij <= K_i y_i`.
   Weak: says fixed cost alone prevents shipping from closed sites.
2. **What is the difference between a feasible and an optimal plan?**
   Strong: distinguishes constraint satisfaction from proof that no better modeled objective exists.
   Weak: assumes any feasible plan is optimal.
3. **What does a solver's infeasible status mean?**
   Strong: says no assignment satisfies the model and recommends checking data and constraints.
   Weak: prints zero shipment and calls it an optimal plan.
4. **Why recompute the objective?**
   Strong: identifies extraction, indexing, and reporting mistakes while acknowledging this is not an independent optimality proof.
   Weak: claims recomputation validates demand forecasts.
5. **What makes a network outage stress misleading?**
   Strong: distinguishes instant recourse from long-term redesign and checks which sites were already open.
   Weak: assumes any selected replacement is immediately operational.
6. **Why can aggregate capacity be sufficient but the model infeasible?**
   Strong: names unreachable lanes, site-specific capacities, or other binding restrictions.
   Weak: assumes capacity totals establish every constraint.
7. **How would you handle indivisible customer assignments?**
   Strong: proposes binary assignment and links it to opening and flows.
   Weak: assumes continuous flow automatically yields one source.
8. **How do you interpret a 110 USD/period stress increase?**
   Strong: notes that both volume and allocation change and avoids calling it a like-for-like saving.
   Weak: attributes the entire change to facility inefficiency.
9. **What evidence is needed before changing a real site network?**
   Strong: names capacity, rates, contracts, lead times, service, finance, legal review, and owner approval.
   Weak: treats a synthetic optimum as a business authorization.
10. **What would you record for a reproducible run?**
    Strong: includes input snapshot, code, versions, solver configuration and status, units, and scenario changes.
    Weak: records only the displayed objective.

## 15. Key terms

- **Allocation:** planned flow from sites to zones; see section 3.
- **Baseline:** reference scenario; see section 5, step 12.
- **Binary opening:** zero-or-one site decision; see section 5, step 5.
- **Capacity:** maximum throughput per period; see section 5, step 3.
- **Capacity linkage:** restriction connecting flow to opening; see section 5, step 8.
- **CBC:** the local solver used in section 7.
- **Constraint:** a mathematical rule for feasible choices; see section 4.
- **Cost horizon:** period covered by a cost; see section 8.
- **Customer zone:** aggregated demand destination; see section 4.
- **Demand balance:** exact fulfillment equation; see section 5, step 7.
- **Facility location:** joint choice of facilities and allocation; see section 3.
- **Feasible:** satisfies model restrictions; see section 4.
- **Fixed cost:** site cost incurred when open; see section 5, step 3.
- **Flow:** quantity shipped on a lane; see section 5, step 5.
- **Infeasible:** no solution satisfies all modeled requirements; see section 6.
- **Lane:** site–zone pair; see section 5, step 4.
- **MILP:** mixed-integer linear program; see section 1.
- **Objective:** modeled cost minimized by the solver; see section 5, step 6.
- **Optimal:** best feasible objective established for the model; see section 4.
- **Outage scenario:** restriction making a site unavailable; see section 6.
- **Planning period:** time bucket shared by demand and capacity; see section 5, step 2.
- **PuLP:** Python modeling library used in section 7.
- **Recourse:** decisions revised after a scenario is known; see section 9.
- **Residual:** constraint difference checked by the validator; see section 5, step 11.
- **Scenario:** a specific set of assumptions; see section 5, step 12.
- **Single sourcing:** one site per zone; see exercise 8.
- **Slack:** unused available capacity; see section 6.
- **Stress test:** adverse what-if scenario; see section 5, step 12.
- **Tolerance:** allowable numerical discrepancy; see section 7.
- **Variable cost:** cost proportional to flow; see section 5, step 6.

## 16. Sources

1. PuLP documentation, [PuLP: a linear programming modeler for Python](https://coin-or.github.io/pulp/). Official modeling API and solver integration documentation.
2. COIN-OR, [CBC User Guide](https://www.coin-or.org/Cbc/cbcuserguide.html). Official documentation for the open-source mixed-integer solver.
3. Cornuéjols, G., Nemhauser, G. L., and Wolsey, L. A. (1990), “The Uncapacitated Facility Location Problem,” in *Discrete Location Theory*, Wiley. Background on facility-location formulation; this lab adds capacity constraints.
4. GS1, [Global Location Number (GLN)](https://www.gs1.org/standards/id-keys/gln). Official identification standard overview; identification does not establish operational feasibility.
5. GS1, [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin). Official trade-item identification overview.
6. UPS, [ORION: On-Road Integrated Optimization and Navigation](https://about.ups.com/us/en/our-company/innovation-driven/orion.html). Company description of a routing application; not a source for this fixture or its quantitative outputs.

Sources describe methods, identifiers, or a named routing application.
None supplies the synthetic costs, demand, or capacity in this lab.
Regulatory applicability and commercial facts require current jurisdiction-specific review.

## 17. Links

- [Module 30 topic index](_index.md).
- [Module 30 survey and lab context](../../../course/30_simulation_optimization_production_labs.md).
- [Environment and repository lab](01_environment-repo.md).
- [Demand forecast pipeline lab](02_demand-forecast-pipeline.md).
- [Inventory policy simulation lab](03_inventory-policy-simulation.md).
- [Network decision ladder](../../02_methods/11_network_design_optimization/01_network-decision-ladder.md).
- [Course expansion specification](../../../course/MODULE_EXPANSION_SPEC.md).
