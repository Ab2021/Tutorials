# Operational Stress Testing of Supply Networks

> Module 14A — Supply Chain Risk, Resilience, and Disruption Analytics
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01–04, particularly exposure mapping and time-to-recover/time-to-survive
> Version 2026.5

## 1. Why this topic exists

A procurement director may believe that two suppliers provide redundancy.
Both can depend on the same upstream plant, port, power grid, or software service.
A simultaneous interruption can therefore defeat a dual-source plan.
Stress testing turns that possibility into an operational sequence and a decision.

The supply-chain manager needs to know what ships, not just which nodes are red.
The finance leader needs the incremental cash consequence, not just exposed revenue.
The commercial team needs to identify which customer promises cannot be kept.
The continuity owner needs to identify actions that can actually be executed in time.
A stress test brings these views onto one consistent clock.

A single-supplier outage can be manageable when independent alternates are ready.
The same outage paired with a shared port restriction may exhaust the available buffer.
A response that is feasible on paper may require qualification or transport lead time.
An inexpensive option may also fail when it shares the initiating hazard.
The test should reveal those dependencies before an emergency.

The decision is often whether to reserve capacity, hold stock, qualify a route, or accept exposure.
The exercise is not an attempt to predict a particular disaster.
It asks how a network performs under specified adverse conditions.
A severity curve shows how the answer changes as the disruption becomes harder.
Reverse stress testing starts at an unacceptable outcome and asks what combination could cause it.

A forecast estimates an uncertain future; it is not an instruction.
A plan is an intended operational configuration, subject to approval and change.
A target is a desired outcome, such as 95% on-time service.
A commitment is a specific promise or contractual obligation; its terms must be checked.
The same percentage can be a target in one context and a commitment in another.

This topic uses a fictional, reproducible example rather than attributing its numbers to a company.
Its daily horizon, fixed customer loss and constant unit economics deliberately simplify reality.
They allow the reader to see the mechanics without treating the result as a general loss model.

## 2. Learning outcomes

After completing this topic you can:

1. **Map** an event to nodes, lanes, capacities, and timed recovery assumptions; see Exercise 1.
2. **Calculate** daily shipments, lost demand, and ending inventory; see Exercise 2.
3. **Distinguish** simultaneous exposure driven by a common cause from independence; see Exercise 3.
4. **Translate** physical shortfall into incremental contribution loss and response cost; see Exercise 4.
5. **Solve** a reverse-stress threshold and state its domain; see Exercise 5.
6. **Compare** mitigation choices with residual risk, implementation timing, and triggers; see Exercise 6.
7. **Critique** a stochastic tail estimate and its dependence assumptions; see Exercise 7.
8. **Design** a decision-ready stress-test memo with ownership and evidence; see Exercise 8.

Success requires explaining the result to an operator and a financial decision-maker.
It also requires separating numerical facts in the model from unverified field assumptions.

## 3. The core mental model

```text
Decision and risk appetite
          |
Baseline network + commitments + available inventory
          |
Hazard -> common causes -> timed node / lane / demand shocks
          |
Operating rules -> constrained flow -> daily shortages and recovery
          |
Financial bridge + nonfinancial service / safety thresholds
          |
Mitigation comparison -> decision -> trigger / owner / retest
          ^                                              |
          +---------- observations and new evidence -----+
```

A shock is an input, not a result.
It changes availability, throughput, transit time, demand, or another modeled constraint.
The network and operating rules determine how that change propagates.
Only then should physical consequences be translated into money.

The same hazard may affect several nodes at once.
Treat a weather footprint, a shared utility, or a shared sub-tier as a causal parent.
Applying independent failure rates to children of that parent can erase the co-movement.
Conversely, putting every shock together without a plausible mechanism is not evidence of dependence.

A test needs two comparisons.
First compare the shocked network with the unshocked baseline using identical demand and accounting.
Then compare feasible response options under the same shock and time horizon.
Do not compare one option in a mild scenario with another in a severe scenario.

A threshold is meaningful only with a unit and measurement period.
For example, 95% of units served within six days differs from 95% of customer orders on time.
A liquidity threshold is a cash condition; a margin threshold is an accrual measure.
A safety threshold cannot be offset with an attractive financial return.

The model should not quietly create supply, borrow inventory from the future, or ship unqualified products.
An actionable result includes the inventory trajectory and the sequence of decisions.
Preserve the model inputs and the response logic so that another analyst can reproduce the conclusion.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Stress test | A conditional evaluation of a defined adverse path | A forecast or a probability estimate | Scenario and horizon |
| Baseline | The comparable no-shock operational and financial path | A perfect forecast or an aspirational target | Units/day and currency |
| Hazard | An event or condition capable of creating a shock | A guaranteed realized loss | Event and geography |
| Exposure | Presence of a vulnerable activity within the hazard footprint | The dollar loss itself | Site, lane, units/day |
| Shock | A prescribed change in a model input | A generic risk label | Capacity %, days, or units/day |
| Severity | The magnitude of a shock relative to a stated reference | Duration alone | % reduction or units/day |
| Duration | Time for which a specified restriction applies | Time until every customer has recovered | Days |
| Recovery path | The timed return of usable capacity or service | A single optimistic repair date | Units/day by day |
| Ramp | A staged capacity restoration after repair | Immediate restoration to full output | Units/day by day |
| Node | A mapped site or activity with operational constraints | A supplier corporate name without a facility | Site identifier |
| Lane | A feasible transport connection between locations | A straight line without transit or capacity | Origin, destination, days |
| Bill of materials | A dated specification of component quantities per product | A supplier list without product relationships | Component units per product |
| Shared dependency | An input or service used by multiple nominal alternatives | Proof that all failures occur together | Dependency identifier |
| Common cause | One mechanism capable of affecting multiple elements | Any pair of events occurring together | Event and affected elements |
| Correlation | A statistical measure of association in specified variables | A causal mechanism or a joint probability by itself | Dimensionless, from -1 to 1 |
| Conditional probability | Chance of an event given another event or state | Unconditional event frequency | Probability, 0 to 1 |
| Joint probability | Chance that specified events occur together | Sum or product of marginal probabilities without conditions | Probability, 0 to 1 |
| Concentration | A high share of an activity in a common exposure | A measure of hazard probability | % of volume or capacity |
| Lead time | Elapsed time from an initiating action to usable arrival | Transit time alone | Days |
| Time-to-recover | Elapsed time to restored usable capability under a stated definition | Physical repair time without qualification or ramp | Days |
| Time-to-survive | Elapsed time before a defined service or inventory threshold is crossed | A universal fixed property of a supplier | Days at a demand and supply rate |
| On-hand inventory | Usable stock at a defined location and moment | All stock including blocked, expired, or in-transit units | Units |
| Allocation rule | A rule for assigning scarce material to competing uses | An assumption that all customers receive their forecast | Priority and eligibility |
| Lost sale | Unfilled demand assumed not to return within the model | A backlog automatically filled later | Units in horizon |
| Backlog | Unfilled demand that remains eligible to be served later | A lost sale or inventory on hand | Units and age |
| Service level | A defined measure of service against demand or promise | Any percentage labeled fill rate | Numerator, denominator, horizon |
| Contribution margin | Revenue less defined avoidable variable costs | Revenue or accounting profit | USD per unit |
| Incremental loss | Difference in economics between stressed and baseline cases | Total revenue or all costs of running the business | USD over a horizon |
| Cash flow | Timing of receipts and disbursements | Accrual profit or contribution | USD per period |
| Penalty | An incremental charge triggered by a specified obligation | An automatic cost on every late unit | USD and contractual trigger |
| Mitigation | An action that changes exposure, consequences, or response | A risk-register entry without executable capacity | Action, cost, readiness |
| Residual risk | Exposure that remains after an implemented action | Zero because a control is purchased | Scenario outcome and uncertainty |
| Reverse stress test | Search from an unacceptable outcome back to a causing path | A prediction of the most likely disaster | Threshold, path, and horizon |
| Risk appetite | A decision-maker-approved limit or boundary for exposure | A model-generated numerical optimum | Metric, threshold, owner |
| Tail percentile | A quantile of an explicitly modeled loss distribution | The maximum possible loss | Probability level and USD |
| Seed | An initial state for reproducible pseudorandom draws | Evidence that simulation assumptions are right | Integer |
| TTR | Time-to-recover, as defined by usable output | Time to restart without its ramp | Days |
| TTS | Time-to-survive until the defined service boundary | Fixed inventory divided by gross demand in every case | Days |

### Reading the vocabulary in an operating model

A facility record should name a physical location rather than only its parent company.
A legal supplier and a facility are different entities and may have different risk footprints.
Exposure can be high while the probability of the event remains unknown.
A defined shock is a conditional input, and its resulting loss is not an annual expected loss.

A lane needs an origin, a destination, a mode, a capacity, and a time to transit.
The usable capacity is the binding capacity after all relevant constraints, not an advertised maximum.
A shared dependency can affect two suppliers without making their failures perfectly correlated.
The dependence assumption must state what happens to each supplier under the common event.

TTR and TTS both depend on the boundary selected for measurement.
Define TTR as restored usable output, not merely a repaired machine.
Define TTS against the specific service obligation or inventory condition under review.
With intermittent supply, inventory divided by average net demand can hide a day-level breach.

A backlog remains outstanding and may consume future capacity.
A lost sale leaves the modeled demand stream and should not also be counted as a backlog.
Contribution loss is incremental only if the baseline sale would have happened.
It is not the same as lost revenue, and fixed costs do not automatically disappear with production.

A quantile describes a model distribution at a stated confidence level and horizon.
Without justified event probabilities and dependence, a conditional scenario is more honest than a percentile.
A reverse stress test can find a threshold without asserting that it is probable.
An owner and trigger make an option operational, but do not establish that it will work.

## 5. The mechanics, step by step

The sequence below is a decision workflow, not a claim that every company needs a large simulation.
For each step, preserve the named output before progressing.
Choose the simplest model that preserves the constraints that change the decision.

### Step 1: Frame the decision

Name the pending decision and the person authorized to approve it.
Set the latest decision date and distinguish a pre-event investment from an emergency action.
Specify alternatives that the organization can actually buy, contract, or execute.
Record the baseline plan and the horizon over which alternatives will be compared.
Output: a decision statement, named owner, and investment or response deadline.
Branch: if no decision can change, treat the exercise as discovery, not an investment recommendation.

### Step 2: Choose success and failure boundaries

Define service, safety, cash, contractual, and operational boundaries with owners.
Identify whether each boundary is a target, a policy limit, or a contractual obligation.
Specify a unit, location, denominator, timing, and measurement method for every boundary.
For example, use units shipped divided by units demanded over six days, not an undefined service %.
Output: a threshold register and priority order for non-tradeable safety constraints.
Branch: if leaders disagree about the threshold, show the outcome curve rather than claiming a pass.

### Step 3: Set the operational baseline

Freeze the demand plan, opening stocks, confirmed inbound supply, and site capacities.
Include product mix, calendar, production shifts, customer commitments, and current supplier allocations.
Separate forecasts from actual orders, and mark the effective date of each record.
Check that the unshocked plan obeys inventory balance and lane capacity.
Output: a dated no-shock flow path against which incremental losses can be measured.
Branch: if the baseline cannot satisfy commitments, show the pre-existing gap separately.

### Step 4: Map the physical network

Identify sites, ports, production processes, warehouses, customers, and inbound and outbound lanes.
Map each product to its dated bill of materials and its qualified sources.
Capture minimum operating levels, yield, bottlenecks, and locations of usable stock.
Trace material and service dependencies beyond tier one when evidence permits.
Output: a versioned network graph with source and confidence for each important edge.
Branch: if a critical edge is unknown, bound its effect instead of presenting it as independent.

### Step 5: Resolve time and units

Choose a time bucket fine enough to reveal the decision and replenishment delays.
Use consistent units for each item and explicit conversion factors where products differ.
Fix the opening inventory time, arrival timing, and when customer demand is served.
Record time zones, shift calendars, nonworking days, and the treatment of partial days.
Output: a model clock and an unambiguous daily or weekly balance convention.
Branch: if a same-day action is impossible, shift the arrival to its physically feasible bucket.

### Step 6: Select adverse paths

Build events that could change the decision, not merely the events with the best historical data.
Include at least one single-node case and one causally linked multi-node case.
For each path specify severity, onset, duration, recovery, and geography.
Document why the combination is physically or operationally plausible without claiming it is likely.
Output: a scenario register with a narrative and parameter table for each case.
Branch: remove combinations that require contradictory conditions, or show them as deliberately hypothetical.

### Step 7: Specify the common cause

Identify a shared power source, sub-tier, transport bottleneck, workforce, or technology service.
Connect the cause to each affected node or lane using explicit mechanisms.
State conditional impacts and timing, including any lag between upstream and downstream effects.
Check whether an alternate is geographically independent but still shares tooling or raw material.
Output: a dependency map and per-node shock table under the common event.
Branch: if the dependency is uncertain, compare linked and independent cases as sensitivities.

### Step 8: Set capacity and timing shocks

Translate the narrative into quantitative daily usable capacity or transit restrictions.
Distinguish capacity loss from longer lead time, yield loss, and actual cancellation of supply.
Apply a shock only to affected activities and respect opening inventory already in the network.
Specify whether repair starts immediately and whether personnel and spares are available.
Output: a time-indexed shock vector for all constrained network elements.
Branch: if a shock only delays supply, model arrivals shifted forward rather than permanently removed.

### Step 9: Represent detection and response delay

State the time the event occurs, when reliable information arrives, and when action is authorized.
Add procurement, production, qualification, booking, transit, and receiving delays where applicable.
Make emergency actions unavailable until their earliest plausible completion time.
Account for escalation capacity and the possibility that a trigger generates a false alarm.
Output: an event-to-action timeline with an accountable role at each handoff.
Branch: if the alternative arrives after stockout, distinguish a recovery benefit from prevention.

### Step 10: Constrain alternates

Verify technical qualification, contractual entitlement, available quantity, and allocation priority.
Subtract capacity already committed to other products, customers, or response plans.
Include minimum order, yield, tooling, transport, and any regulatory release requirements.
Check for common exposure between the original node and the proposed backup.
Output: a dated feasible supply schedule and evidence for the capacity claim.
Branch: if qualification is incomplete, test the delay or exclude the alternate.

### Step 11: Set allocation and customer rules

Specify customer priority, product mix, substitutions, and minimum shipment quantities.
Distinguish safety-critical obligations from discretionary demand and define authorization to deviate.
Choose whether unfilled demand is lost, backlogged, substituted, or partially recovered.
Apply the same rules across compared cases unless a response explicitly changes them.
Output: a reproducible decision rule for scarce stock and constrained capacity.
Branch: if contracts prohibit an allocation, reject the infeasible operating plan.

### Step 12: Run the time-stepped balance

At each bucket add eligible arrivals to opening stock and remove actual shipments.
Limit shipments to eligible demand and physical availability after any production constraints.
Carry closing inventory into the next bucket and keep inventory nonnegative.
If backlog exists, separately carry unfilled eligible demand forward and track its age.
Output: a ledger of available stock, shipments, shortages, and closing balance by bucket.
Branch: if stock goes negative, fix the sequencing or double counting before interpreting results.

### Step 13: Trace propagation through the network

Pass constrained production and shipments through bill-of-materials and lane relationships.
Account for work in process, conversion yields, transit lags, and shared bottlenecks as needed.
Avoid simultaneously spending the same alternate capacity on two affected products.
Record which node or lane first prevents the next unit from being delivered.
Output: a bottleneck explanation and the path from cause to customer impact.
Branch: if flows are infeasible, revise allocation or capacity, not just the reported loss.

### Step 14: Measure the service consequence

Calculate the stated metric for each day, customer class, product, and full horizon.
Separate demand-weighted fill rate from order-level on-time-in-full performance.
Report first breach, duration below threshold, and the shape of the recovery.
Show unserved units as well as percentages so that denominator changes are visible.
Output: service trajectories and an explicit pass or breach against each agreed boundary.
Branch: if an aggregate passes but a protected customer fails, report both results.

### Step 15: Build an incremental financial bridge

Apply a documented unit contribution to lost sales that would have occurred in the baseline.
Add actual incremental response, penalty, disposal, and other costs only when their triggers occur.
Subtract avoided variable costs only if they are not already deducted in the contribution margin.
Separate operating contribution from cash, working capital, insurance, and balance-sheet effects.
Output: a dated bridge from physical outcomes to scenario cost with signs and units.
Branch: if penalties are uncertain or disputed, show a range and legal review rather than a certainty.

### Step 16: Compare response options

Calculate the same service and financial measures for each implementable option under each path.
Include reservation fees, setup, incremental freight, carrying cost, and readiness dates where relevant.
Measure residual loss and any new dependency introduced by an alternative.
Compare scenario-by-scenario results instead of adding adverse cases as though simultaneous.
Output: a matrix showing baseline, gross benefit, option cost, net effect, and constraints.
Branch: an option that fails a mandatory safety rule is not rescued by a favorable financial score.

### Step 17: Conduct reverse stress testing

Start from a stated unacceptable service, financial, or liquidity outcome and horizon.
Search combinations of shock duration, capacity, response delay, and demand that reach the boundary.
Impose physical and contractual constraints so that the search cannot invent a response or event.
Present more than one materially distinct boundary-crossing path when it changes the decision.
Output: a set of failure paths, assumptions, and distances from current operating conditions.
Branch: if an algebraic solution assumes a constant rate, verify the boundary on the daily ledger.

### Step 18: Test sensitivity and model risk

Vary uncertain inputs individually and in coherent combinations around the central path.
Test recovery duration, inventory usability, demand, lead time, and alternate capacity.
Identify which assumptions change the preferred decision or cross an approved threshold.
Do not label a range a confidence interval unless it was calculated as one.
Output: a ranked decision-sensitivity table and an explicit statement of unknowns.
Branch: if the decision reverses under plausible inputs, seek evidence or choose a reversible action.

### Step 19: Add probabilities only when defensible

Distinguish observed frequencies, expert judgments, and conditional assumptions.
Use joint event structure or a common-cause model rather than multiplying marginal probabilities blindly.
Check that event probabilities, including the no-event state, sum to one over the modeled horizon.
Sample full paths and preserve timing and dependence if using Monte Carlo simulation.
Output: a distribution conditional on stated parameters, with sample size and uncertainty.
Branch: if data are inadequate, present scenario costs and threshold distances without a fabricated percentile.

### Step 20: Review implementation and execution

Check that contracts, budget, tooling, training, and data access exist before calling an option ready.
Identify lead time from approval to operational readiness and the latest viable activation time.
Assign trigger thresholds, decision rights, alternates, and a communications path.
Conduct a tabletop using the same assumptions that generated the numerical benefit.
Output: an action card and a dated implementation plan with evidence requirements.
Branch: if the intervention cannot be activated in time, reclassify it as a future capability.

### Step 21: Obtain challenge and approval

Ask operations, finance, sales, procurement, legal, and continuity owners to challenge assumptions.
Reconcile the physical ledger to source data and the financial bridge to unit economics.
Make unresolved data gaps and estimates visible rather than averaging away disagreement.
Record the selected option, dissent, residual exposure, and next review date.
Output: a decision log and a versioned model package.
Branch: if material evidence is absent, authorize a bounded evidence-gathering step rather than a false pass.

### Step 22: Monitor and retest

Assign owners to update supplier locations, allocations, recovery estimates, and contract limits.
Define events that invalidate the test, including acquisition, route change, new product, or hazard change.
Compare realized recovery and costs with prior assumptions after any incident or exercise.
Update the model and reapprove triggers when material assumptions change.
Output: a scheduled review and lessons that are traceable to changed inputs.
Branch: if monitoring reveals a broken dependency, trigger immediate reassessment before the regular review.

### Core balance equations

For a single product in day $t$, let $I_{t-1}$ be opening usable inventory in units.
Let $A_t$ be arrivals usable before demand in units and $D_t$ be demand in units.
If demand is lost rather than backlogged, shipments are $S_t=\min(D_t, I_{t-1}+A_t)$.
Lost demand is $L_t=D_t-S_t$ and closing inventory is $I_t=I_{t-1}+A_t-S_t$.
These identities require nonnegative inputs and no hidden consumption or production yield loss.
A multi-product model must also honor shared production and transport capacity.

If demand is backlogged, define the backlog at the start of day $t$ as $B_{t-1}$.
Then the eligible quantity may be $D_t+B_{t-1}$, subject to the specified priority and cancellation rules.
The closing backlog is the eligible quantity less shipments and cancellations.
A backlog is not an asset and does not appear in physical inventory.
A late shipment can generate a different financial effect from a sale that disappears permanently.

For incremental loss, compare stressed and baseline outcomes over the same horizon.
One simple bridge is lost units times contribution per unit, plus triggered penalties and incremental response costs.
Deduct documented recoveries only once, and identify their timing and conditions.
A negative incremental loss indicates a modeled benefit, not proof that the option is free.

When a shock merely postpones an arrival, conserve the quantity unless cancellation is explicit.
An arrival deferred beyond the horizon may affect working capital or service after the horizon.
An end-of-horizon stock balance is not itself a loss; reconcile its valuation separately.
The case below deliberately fixes a six-day horizon to make this boundary visible.

## 6. Worked example

### Decision and boundaries

A fictional medical-device assembler sells a nonregulated accessory through several distributors.
The product and figures are illustrative, not observations about a real manufacturer.
The sourcing director must decide whether to reserve an independent emergency supplier for six days.
The emergency contract must be arranged before a forecasted seasonal hazard window.

The finance owner sets a conditional stress-loss review threshold of USD 4,500 over six days.
This threshold is a management risk appetite for this exercise, not a statutory requirement.
The commercial target is at least 90% unit fill over the same six days.
The risk committee treats a threshold hit or exceedance as requiring escalation.
Neither threshold establishes a contractual obligation.

The demand forecast is 100 units/day for six days.
The operating plan assumes production and delivery of up to that amount each day.
The service target is 90% fill over the full horizon.
A separate customer commitment is not modeled; any real commitment needs contract-level review.
The forecast, plan, target, and commitment therefore have different meanings.

### Network and baseline

There is one finished product and one unit of component required per finished unit.
One component is assumed to convert immediately into one saleable product without yield loss.
The starting usable component buffer is 40 units at the assembly site before day 1 arrivals.
The site can assemble and ship at least 100 units/day, with sufficient labor and outbound transport.
No other component or lane constrains output in the example.
Demand is served after same-day component arrivals, and shipments leave immediately.
Unfilled daily demand is permanently lost, not backlogged.

Supplier A normally delivers 80 component units/day.
Supplier B normally delivers 40 component units/day.
Both suppliers use the same upstream power grid despite being different legal entities.
The common-cause scenario reduces usable arrivals from each supplier on days 1–3.
No probability is attached to this conditional event.

| Day | A baseline, units | B baseline, units | Total baseline arrival, units | Demand, units |
|---:|---:|---:|---:|---:|
| 1 | 80 | 40 | 120 | 100 |
| 2 | 80 | 40 | 120 | 100 |
| 3 | 80 | 40 | 120 | 100 |
| 4 | 80 | 40 | 120 | 100 |
| 5 | 80 | 40 | 120 | 100 |
| 6 | 80 | 40 | 120 | 100 |

In the baseline, daily supply exceeds daily demand by 20 units.
All 600 units of forecast demand are shipped within their day.
Ending inventory is 40 + 6 × (120 − 100) = 160 units.
The example assumes inbound supply is already paid for consistently across scenarios.
Any incremental value of unused stock is excluded from the contribution bridge and needs separate review.

### Common-cause path

Suppose a grid restriction cuts A to 20 units/day and B to 20 units/day for three days.
A recovers from 25% to 100% of baseline on day 4.
B recovers from 50% to 100% of baseline on day 4.
The full restoration on day 4 is an assumption, not a demonstrated repair estimate.
The shared grid is the causal link; it does not imply that their ordinary failures are always simultaneous.

| Day | Opening stock, units | A, units | B, units | Available, units | Demand, units | Shipped, units | Lost, units | Closing stock, units |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 20 | 20 | 80 | 100 | 80 | 20 | 0 |
| 2 | 0 | 20 | 20 | 40 | 100 | 40 | 60 | 0 |
| 3 | 0 | 20 | 20 | 40 | 100 | 40 | 60 | 0 |
| 4 | 0 | 80 | 40 | 120 | 100 | 100 | 0 | 20 |
| 5 | 20 | 80 | 40 | 140 | 100 | 100 | 0 | 40 |
| 6 | 40 | 80 | 40 | 160 | 100 | 100 | 0 | 60 |

Day 1 has 40 opening units plus 40 arrivals, so 80 units ship and 20 units are lost.
Days 2 and 3 each ship 40 units and lose 60 units.
Days 4–6 each meet that day's demand and rebuild stock because demand lost earlier does not return.
The cumulative loss is 20 + 60 + 60 = 140 units.
Total shipments are 460 units and unit fill is 460 / 600 = 76.67%.
The six-day service target of 90% is missed.
Ending stock is 60 units, 100 units below the baseline ending stock of 160 units.

If demand instead remained on backlog, the day 4–6 production plan would need to absorb it.
That alternative would change both service and the financial bridge and is not the present case.
Likewise, applying daily averages without inventory sequencing would conceal the day 1 shortage.
The inventory count cannot go below zero or borrow arrivals from day 4.

### Financial consequence

Assume USD 40/unit of lost contribution after avoidable variable costs.
Assume a separately verified illustrative contractual penalty of USD 5 per lost unit in this scenario.
For teaching, the penalty is charged on every lost unit; real contract terms may differ.
These are distinct costs, and neither includes the other.
Ignore tax, inflation, insurance recoveries, stock valuation, cash timing, and reputational effects here.

Lost contribution is 140 units × USD 40/unit = USD 5,600.
Penalty is 140 units × USD 5/unit = USD 700.
Incremental scenario loss is USD 5,600 + USD 700 = USD 6,300.
This exceeds the USD 4,500 escalation threshold by USD 1,800.
It is a conditional six-day amount, not an expected annual loss.
No likelihood or return on an annual investment can be inferred from it alone.

### Independent capacity option

A qualified supplier C is assumed to be independent of the grid and upstream inputs in this scenario.
Evidence for that independence and for product qualification must be obtained before approval.
C can deliver an extra 30 units/day on days 1–3, starting on day 1.
The capacity reservation costs USD 600 for this six-day window whether exercised or not.
The incremental premium is USD 12 per C unit actually delivered.
C supplies no units on days 4–6 in this scenario.
No other action uses C's reserved capacity.

| Day | Opening stock, units | A + B, units | C, units | Shipped, units | Lost, units | Closing stock, units |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 40 | 30 | 100 | 0 | 10 |
| 2 | 10 | 40 | 30 | 80 | 20 | 0 |
| 3 | 0 | 40 | 30 | 70 | 30 | 0 |
| 4 | 0 | 120 | 0 | 100 | 0 | 20 |
| 5 | 20 | 120 | 0 | 100 | 0 | 40 |
| 6 | 40 | 120 | 0 | 100 | 0 | 60 |

On day 1, available supply is 40 + 40 + 30 = 110 units; 10 units remain.
On day 2, 10 + 40 + 30 = 80 units are available; 20 units are lost.
On day 3, 70 units are available; 30 units are lost.
C delivers 90 units in total, and losses fall from 140 units to 50 units.
The 550 shipped units yield 91.67% fill, above the 90% horizon target.
An aggregate target can still hide day-level customer failures.

Remaining lost contribution is 50 × USD 40 = USD 2,000.
The remaining penalty is 50 × USD 5 = USD 250.
The premium is 90 × USD 12 = USD 1,080, plus the USD 600 reservation.
Total conditional cost with C is USD 2,000 + USD 250 + USD 1,080 + USD 600 = USD 3,930.
Conditional savings against no option are USD 6,300 − USD 3,930 = USD 2,370.
The option is USD 570 below the USD 4,500 threshold in this scenario.
The reservation must also be paid in any no-shock window for which it is contracted.

### Break-even and decision

The avoided physical loss is 90 units, worth 90 × USD 45/unit = USD 4,050.
The option's incremental cost is USD 600 + USD 1,080 = USD 1,680.
Net scenario benefit is USD 4,050 − USD 1,680 = USD 2,370.
This is a conditional comparison, not an annual expected-value recommendation.
If this were the only possible shock and reservation bought protection for one independent window,
the expected-cost break-even probability would be USD 600 / USD 2,970 = 20.20%.
Here USD 2,970 is the USD 4,050 avoided loss less the USD 1,080 premium paid only on exercise.
That threshold follows only if the no-shock option cost is USD 600 and there are no other outcomes or costs.
It is a sensitivity calculation, not an estimated event probability or an investment approval.

The conditional decision is to investigate and provisionally favor C for this specific stress path.
Approval requires proof of 30 units/day of incremental capacity, qualification, and independent inputs.
It also requires budget, activation authority, and a trigger early enough for day 1 availability.
If those conditions cannot be verified, rerun the model with the actual first delivery date.
A second adverse path should test the resilience of C itself and a late-notice grid restriction.

### Reverse-stress calculation

Hold the same 100 units/day demand, 40-unit opening stock, and three-day shock duration.
Let $x$ be total A plus B arrivals per day during days 1–3, with $0 \le x \le 100$ units/day.
Assume days 4–6 have enough supply for their own demand and lost demand does not return.
C is absent, and there are no extra costs beyond USD 45 per lost unit.
The cumulative lost units over the three shock days are $\max(0, 300-(40+3x))$.
For this domain, cumulative stock can be used without violating the daily delivery sequence.

The USD 4,500 loss boundary corresponds to 4,500 / 45 = 100 lost units.
Set 300 − (40 + 3x) = 100, giving $x=160/3=53.333\ldots$ units/day.
At exactly that constant arrival rate, the loss is USD 4,500 and the boundary is hit.
Below that rate, the loss exceeds the boundary; above it, the loss is below the boundary.
The modeled common-cause case delivers 40 units/day and therefore lies beyond the boundary.
Discrete whole units require a feasible integer arrival schedule and may shift the exact daily threshold.

At the threshold, day 1 can ship 93.333 units and lose 6.667 units in a continuous approximation.
Days 2 and 3 each ship 53.333 units and lose 46.667 units.
The sum is 100 lost units, consistent with the aggregate equation.
If the 40-unit buffer is not usable on day 1, the threshold is materially different.
If supply is uneven across days, its total alone is insufficient because late stock cannot repair earlier lost sales.
The boundary is one failure path, not the most likely cause or a complete map of all failure paths.

## 7. Code, query, or calculation

The following standard-library Python 3 program reproduces the two daily ledgers.
It models one product, lost sales, no production bottleneck, and arrival before demand.
It accepts the fixed six-day paths in Section 6; it is not a network optimization solver.
All values are in units, days, or USD as shown.
The reservation fee applies to the option even if the supplier were not called.
The premium applies to delivered C units, not to the maximum reserved quantity.

```python
DEMAND = 100
OPENING = 40
BASE = [120, 120, 120, 120, 120, 120]
SHOCK = [40, 40, 40, 120, 120, 120]
C_SUPPLY = [30, 30, 30, 0, 0, 0]
LOSS_PER_UNIT = 45


def run(arrivals, extra, reservation=0, premium=0):
    inventory = OPENING
    rows = []
    for day, (normal, backup) in enumerate(zip(arrivals, extra), 1):
        opening = inventory
        available = opening + normal + backup
        shipped = min(DEMAND, available)
        lost = DEMAND - shipped
        inventory = available - shipped
        rows.append((day, opening, normal, backup, shipped, lost, inventory))
    lost_total = sum(row[5] for row in rows)
    cost = lost_total * LOSS_PER_UNIT + reservation + sum(extra) * premium
    return rows, lost_total, inventory, cost


zero = [0] * len(SHOCK)
for label, arrivals, extra, fee, premium in [
    ('baseline', BASE, zero, 0, 0),
    ('stress', SHOCK, zero, 0, 0),
    ('stress+C', SHOCK, C_SUPPLY, 600, 12),
]:
    rows, lost, closing, cost = run(arrivals, extra, fee, premium)
    print(label, 'lost', lost, 'closing', closing, 'cost USD', cost)
    for row in rows:
        print('  day/open/AB/C/ship/lost/close:', *row)

boundary = (300 - OPENING - 4500 / LOSS_PER_UNIT) / 3
print('reverse threshold units/day:', round(boundary, 3))
```

Expected summary lines are:

```text
baseline lost 0 closing 160 cost USD 0
stress lost 140 closing 60 cost USD 6300
stress+C lost 50 closing 60 cost USD 3930
reverse threshold units/day: 53.333
```

Each summary is followed by six daily rows, except the reverse threshold line.
The first stress row ends with `80 20 0` for shipped, lost, and closing units.
The first stress-plus-C row ends with `100 0 10` for those same quantities.
The baseline's zero cost is incremental loss relative to itself, not total operating cost.
The script deliberately fixes equal-length input lists; `zip` would truncate unequal inputs.
A production tool should reject mismatched horizons before execution rather than silently omit days.

### Calculation audit

Check the opening stock before applying the first arrival.
Check that A and B arrivals are combined once and C is not included in them.
Check that each day's closing stock equals its opening stock plus arrivals less shipment.
Check that the next day's opening stock equals the prior day's closing stock.
Check that shipment never exceeds demand or available physical supply.
Check that lost units equal daily demand minus daily shipment.
Check that no lost unit reappears as demand later in the six-day case.
Check that every C unit charged the premium appears in the availability ledger.
Check that the reservation is charged once per window, not once per day.
Check that lost contribution and the penalty are separate cost categories.
Check that baseline losses are zero before subtracting the stress result.
Check that all three runs use the same demand and closing date.

### Adapting the artifact

For a demand surge, replace the constant demand with a dated demand vector.
Do not use the unchanged reverse threshold formula with a changed demand pattern.
For backorders, carry outstanding demand forward with its age and cancellation rules.
For alternate lead time, put zero in C's arrival schedule until the first feasible receipt.
For a partial qualification, constrain C by product and date, not a single global flag.
For a lane disruption, move the affected quantities to their feasible arrival days.
For cancellations, explicitly remove rather than defer the affected quantities.
For multiple products, enforce shared component, assembly, and freight constraints.
For contractual penalties, calculate eligibility and caps from the actual agreement.
For a financial report, add an explicit baseline and timed cash-flow bridge.
For stochastic paths, sample and store one coherent scenario per run.
For a reverse search, enforce integer units if physical quantities cannot be fractional.
For independent review, retain the input version, program version, and decision date.

## 8. Data, standards, and regulatory context

The model is only as credible as its site, timing, capacity, and contractual evidence.
Record a source, timestamp, owner, and uncertainty for each material input.
The following data checks do not require a particular software vendor.

### Site master

Capture: Physical address, geocode, owner, operating calendar.
Typical source: Procurement and supplier confirmation.
Common defect: A corporate headquarters used as a factory location.
Control: Verify coordinates and the operating entity before overlaying hazards.

### Network edges

Capture: Origin, destination, mode, transit time, and usable capacity.
Typical source: Transportation management and carrier records.
Common defect: A route displayed without a feasible receiving location.
Control: Check actual paths and record route validity dates.

### Bills of materials

Capture: Effective component-to-product ratios and approved substitutions.
Typical source: Engineering and product lifecycle system.
Common defect: Old revision applied to current production.
Control: Join on product, plant, and effective date.

### Inventory

Capture: Location, product, quantity, status, expiry, and ownership.
Typical source: Warehouse and enterprise resource planning records.
Common defect: Blocked, expired, or in-transit stock treated as available now.
Control: Reconcile a timestamped stock snapshot to physical usability.

### Supplier capacity

Capture: Daily usable committed supply, allocation, and recovery path.
Typical source: Supplier contracts and operational confirmation.
Common defect: Nameplate output treated as dedicated capacity.
Control: Ask for reserved units, competing obligations, and evidence date.

### Demand and orders

Capture: Daily product/customer demand and cancellation or backlog rules.
Typical source: Sales planning and order management.
Common defect: Forecast units labeled as binding orders.
Control: Separate forecasts, firm orders, and target service.

### Hazard footprint

Capture: Event extent, onset, severity, and duration.
Typical source: Official hazard services and scenario designers.
Common defect: Regional hazard applied uniformly to every site.
Control: Record location accuracy and use site-level impact assumptions.

### Common dependencies

Capture: Sub-tier components, energy, logistics, and information services.
Typical source: Supplier declarations and engineering investigation.
Common defect: Unreported dependency treated as absent.
Control: Label missing coverage and run dependence sensitivities.

### Recovery evidence

Capture: Repair, ramp, staffing, and qualification durations.
Typical source: Exercises, incidents, supplier engineering.
Common defect: Vendor promise presented as a validated recovery time.
Control: Compare estimates with a dated test or incident and its relevance.

### Commercial terms

Capture: Service clauses, thresholds, penalties, and caps.
Typical source: Executed contracts and legal owner.
Common defect: Every lost unit automatically penalized.
Control: Record the actual trigger, exceptions, and dispute process.

### Unit economics

Capture: Price, avoidable costs, contribution, and currency.
Typical source: Finance and product accounting.
Common defect: Revenue multiplied by lost units as profit loss.
Control: Reconcile with baseline and specify the cost definition.

### Response cost

Capture: Reservation, premium, transport, labor, and setup costs.
Typical source: Procurement quotations and finance.
Common defect: Backup assumed free or double counted.
Control: Separate fixed readiness costs from exercise-dependent spending.

### Cash and insurance

Capture: Payment dates, liquidity, coverage, deductibles, and sublimits.
Typical source: Treasury and insurance documents.
Common defect: Accounting loss assumed to equal insured cash recovery.
Control: Use cash timing and policy conditions with responsible review.

### Incident history

Capture: Dates, duration, impact, data quality, and operating context.
Typical source: Internal event register and external reports.
Common defect: Few historical incidents used as a precise tail distribution.
Control: Track exposure changes, missing cases, and sampling uncertainty.

### Model metadata

Capture: Input versions, owner, date, code revision, and assumptions.
Typical source: Model governance record.
Common defect: A figure copied without its scenario or horizon.
Control: Keep inputs and outputs together for reproducibility.

### Standards and legal interpretation

ISO 22301:2019 is an international standard for business continuity management systems.
It specifies requirements for an organization's management system; the standard is not itself a law.
A customer contract or applicable legal regime may separately require particular arrangements.
Certification is not established by producing a stress-test spreadsheet or reading this topic.
The organization must check applicable standard text, scope, audit criteria, and contractual obligations.

ISO 31000:2018 provides risk management guidelines.
It is a general framework, not a prescribed probability distribution or loss formula.
It does not validate an untested supplier recovery claim or an event correlation.
Use it to organize context, assessment, treatment, and monitoring while documenting local choices.

GS1 identification standards can help align product and location identifiers across organizations.
An identifier does not establish that a facility is physically independent or that an item is available.
Maintain the crosswalk between corporate entity, physical site, product, and time-valid relationship.
A duplicated or stale location code can make a shared dependency appear to be two alternatives.
Use the applicable GS1 documentation and trading-partner agreement for identifier details.

Privacy, competition, sanctions, safety, and trade rules may affect information sharing or alternate sourcing.
Whether a particular rule applies depends on jurisdiction, entity, product, date, and activity.
Legal and trade specialists should check the actual law and planned counterparty and route.
Do not assume an emergency makes an otherwise restricted supplier or transport lane permissible.
Do not treat this teaching example as legal, insurance, or regulatory advice.

Document access controls and the need-to-know basis for detailed supplier and customer information.
A shared hazard map may reveal confidential supplier sites or critical inventory locations.
A supplier may reasonably supply ranges or controlled-access evidence instead of raw proprietary data.
Maintain lineage from the aggregated result to the underlying authorized source.

## 9. Industry practice and case evidence

### Toyota and the 2011 Great East Japan Earthquake

Toyota's company history describes the March 11, 2011 earthquake and tsunami.
It reports damage to production sites, dealers, and completed vehicles at port facilities.
The account states that Toyota established companywide and functional countermeasures headquarters.
It places protection of human life and relief before resuming production.
It also describes centralized information collection and the decision to suspend Japanese plant operations.
These are company-reported facts; they do not reveal the case study's complete loss distribution.

[INFERENCE] A useful test based on these facts would represent several affected activities at once.
[INFERENCE] It would include information delay and the time required to coordinate procurement and production.
[INFERENCE] Safety and employee welfare would appear as constraints ahead of financial recovery.
[INFERENCE] Ports, suppliers, workforce, and facilities should not be treated as isolated by default.
These modeling implications are not claims that Toyota used the numerical model in Section 6.
A single company's account also does not prove that any specific backup contract would have succeeded.

### Dali and the Francis Scott Key Bridge, Baltimore

The U.S. National Transportation Safety Board (NTSB) documents the March 26, 2024 contact
between containership Dali and the Francis Scott Key Bridge and the subsequent bridge collapse.
The investigation page and final report are sources for the event and its investigation.
The event illustrates the need to distinguish a waterway, landside transport, and people-safety effects.
Do not assume that every shipment used the same route, faced the same delay, or had the same alternative.

[INFERENCE] A port stress test can represent channel availability separately from bridge-road availability.
[INFERENCE] It should check whether proposed diversions have capacity and onward transport.
[INFERENCE] A gradual reopening can be represented by a capacity ramp rather than a single binary switch.
[INFERENCE] A diversion's commercial effect depends on booking lead time and product-specific demand.
The investigation itself does not validate any numerical cost, recovery curve, or probability in this topic.

### Translating public cases into model inputs

A public case is evidence that a mechanism can matter, not an estimate of today's probability.
A source may describe an event after the fact without identifying all affected customers or substitutes.
One organization's reported recovery time need not apply to a different plant, product, or contract.
A documented closure does not tell you which shipments were canceled versus delayed.
A company statement can be informative while still reflecting the company's perspective.
An official investigation may address safety causes rather than private commercial losses.

Use case material to build a causal story and a list of testable data requests.
For example, ask operations which nodes share a port and when an alternate could be booked.
Ask procurement whether diverted volume has reserved space rather than only theoretical capacity.
Ask finance how detention, premium transport, and lost contribution would be recognized.
Ask customer service whether orders would be canceled, accepted late, or switched to another product.
Then test the resulting quantitative inputs against records relevant to the current network.

### Comparison of practice choices

A supplier-by-supplier outage test is useful for tracing a single dependency.
It may fail to reveal a common utility or upstream material shared by the nominal backups.
A correlated event test makes that interaction explicit but depends on a credible causal pathway.
A historical replay can anchor timing, yet the present network may be materially different.
A hypothetical extreme can expose a threshold even when no matching historical case exists.
A tabletop can reveal authorization delays that a capacity-only model would miss.
None of these approaches alone establishes the probability of a future disruption.
The strongest decision package combines physical balances, evidence checks, and executable decisions.

## 10. Failure modes and red flags

The table links an observable warning to mechanism, consequence, and an actionable control.
A control is useful only when an owner can provide evidence that it works.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Nominally diverse suppliers share a tier-2 input | Both suppliers name the same upstream fabricator | A common cause removes both outputs | Backup capacity disappears with the primary | Trace sub-tier and test shared-event paths. |
| Overstated usable inventory | Stock report includes blocked or expired lots | Nonshippable material enters the buffer | Survival time and fill rate are overstated | Use status-coded inventory and physical checks. |
| Late action assumed instantaneous | Backup starts at event onset despite booking or qualification delay | The model moves future supply into the first day | Preventable shortfall appears eliminated | Model detection, approval, lead time, and first receipt. |
| Wrong event duration | A one-day outage is used despite a longer repair range | Recovery starts before capacity is usable | Loss and liquidity need are understated | Test a dated repair and ramp range. |
| Independent draws hide a common cause | Joint failures occur at the product of marginal probabilities by default | A shared hazard has no parent in the simulation | Tail exposure can be understated | Model a common event and conditional impacts. |
| Perfect dependence asserted without evidence | Every supplier always fails whenever one fails | Exposure is exaggerated without a mechanism | Money is directed away from better controls | Separate plausible common causes and unrelated failures. |
| Inventory borrowed from future arrivals | Negative daily stock appears in the output | Aggregate totals ignore arrival sequencing | Early customer losses disappear | Enforce daily nonnegative balances. |
| Backlog treated as a lost sale and later fulfilled | The same unit appears in both loss and later revenue | Demand state is not explicitly carried forward | Double-counted service and distorted margin | Use exclusive lost, backlog, or fulfilled states. |
| Nominal alternate has no contractual allocation | Many customers claim the same backup production | Nameplate capacity exceeds reserved capacity | Multiple response plans compete for one unit | Document entitlement and shared allocation. |
| Response doubles capacity across products | The same truck or machine is allocated twice | Scenarios are solved without shared constraints | Infeasible service projections reach leadership | Constrain capacity jointly across products. |
| Revenue labeled contribution loss | Every lost sale is charged at full selling price | Avoidable variable costs are ignored | Financial impact is overstated | Reconcile to documented incremental unit economics. |
| Penalty charged without a trigger | All delays attract a full penalty regardless of contract | Caps, grace periods, and causation are omitted | A false mitigation business case results | Check executed terms and show uncertain ranges. |
| Insurance netted prematurely | Gross loss is reduced by expected full reimbursement | Waiting periods and exclusions are ignored | Liquidity and residual exposure are understated | Show gross loss and conditional recovery separately. |
| Average service hides priority failure | Overall fill passes while a critical account is unserved | Aggregate denominators conceal distribution | Commitments or critical users are harmed | Report service by customer class and day. |
| Model horizon ends before the consequences | The stress finishes while backlog remains | Future recovery and cancellation are excluded | A long tail of loss is omitted | Extend the horizon or report a terminal liability. |
| Reverse stress boundary is infeasible | The algorithm crosses the limit only by impossible capacity | Constraints or timing were dropped | Effort is spent on an irrelevant failure path | Constrain search and review the operational narrative. |
| An option is counted before readiness | The plan assumes an unsigned contract is already effective | Qualification, funding, and authority are missing | Leadership believes risk has been reduced prematurely | Separate proposed, funded, qualified, and tested states. |
| False precision in rare-event output | A percentile is presented to the dollar from sparse data | Uncertain dependence and severity are hidden | Risk appetite decisions rely on a fragile number | Show assumptions, uncertainty, and scenario sensitivities. |

### Review: Nominally diverse suppliers share a tier-2 input

Look for: Both suppliers name the same upstream fabricator.
Why it matters: A common cause removes both outputs.
Decision exposure: Backup capacity disappears with the primary.
Practical response: Trace sub-tier and test shared-event paths.

### Review: Overstated usable inventory

Look for: Stock report includes blocked or expired lots.
Why it matters: Nonshippable material enters the buffer.
Decision exposure: Survival time and fill rate are overstated.
Practical response: Use status-coded inventory and physical checks.

### Review: Late action assumed instantaneous

Look for: Backup starts at event onset despite booking or qualification delay.
Why it matters: The model moves future supply into the first day.
Decision exposure: Preventable shortfall appears eliminated.
Practical response: Model detection, approval, lead time, and first receipt.

### Review: Wrong event duration

Look for: A one-day outage is used despite a longer repair range.
Why it matters: Recovery starts before capacity is usable.
Decision exposure: Loss and liquidity need are understated.
Practical response: Test a dated repair and ramp range.

### Review: Independent draws hide a common cause

Look for: Joint failures occur at the product of marginal probabilities by default.
Why it matters: A shared hazard has no parent in the simulation.
Decision exposure: Tail exposure can be understated.
Practical response: Model a common event and conditional impacts.

### Review: Perfect dependence asserted without evidence

Look for: Every supplier always fails whenever one fails.
Why it matters: Exposure is exaggerated without a mechanism.
Decision exposure: Money is directed away from better controls.
Practical response: Separate plausible common causes and unrelated failures.

### Review: Inventory borrowed from future arrivals

Look for: Negative daily stock appears in the output.
Why it matters: Aggregate totals ignore arrival sequencing.
Decision exposure: Early customer losses disappear.
Practical response: Enforce daily nonnegative balances.

### Review: Backlog treated as a lost sale and later fulfilled

Look for: The same unit appears in both loss and later revenue.
Why it matters: Demand state is not explicitly carried forward.
Decision exposure: Double-counted service and distorted margin.
Practical response: Use exclusive lost, backlog, or fulfilled states.

### Review: Nominal alternate has no contractual allocation

Look for: Many customers claim the same backup production.
Why it matters: Nameplate capacity exceeds reserved capacity.
Decision exposure: Multiple response plans compete for one unit.
Practical response: Document entitlement and shared allocation.

### Review: Response doubles capacity across products

Look for: The same truck or machine is allocated twice.
Why it matters: Scenarios are solved without shared constraints.
Decision exposure: Infeasible service projections reach leadership.
Practical response: Constrain capacity jointly across products.

### Review: Revenue labeled contribution loss

Look for: Every lost sale is charged at full selling price.
Why it matters: Avoidable variable costs are ignored.
Decision exposure: Financial impact is overstated.
Practical response: Reconcile to documented incremental unit economics.

### Review: Penalty charged without a trigger

Look for: All delays attract a full penalty regardless of contract.
Why it matters: Caps, grace periods, and causation are omitted.
Decision exposure: A false mitigation business case results.
Practical response: Check executed terms and show uncertain ranges.

### Review: Insurance netted prematurely

Look for: Gross loss is reduced by expected full reimbursement.
Why it matters: Waiting periods and exclusions are ignored.
Decision exposure: Liquidity and residual exposure are understated.
Practical response: Show gross loss and conditional recovery separately.

### Review: Average service hides priority failure

Look for: Overall fill passes while a critical account is unserved.
Why it matters: Aggregate denominators conceal distribution.
Decision exposure: Commitments or critical users are harmed.
Practical response: Report service by customer class and day.

### Review: Model horizon ends before the consequences

Look for: The stress finishes while backlog remains.
Why it matters: Future recovery and cancellation are excluded.
Decision exposure: A long tail of loss is omitted.
Practical response: Extend the horizon or report a terminal liability.

### Review: Reverse stress boundary is infeasible

Look for: The algorithm crosses the limit only by impossible capacity.
Why it matters: Constraints or timing were dropped.
Decision exposure: Effort is spent on an irrelevant failure path.
Practical response: Constrain search and review the operational narrative.

### Review: An option is counted before readiness

Look for: The plan assumes an unsigned contract is already effective.
Why it matters: Qualification, funding, and authority are missing.
Decision exposure: Leadership believes risk has been reduced prematurely.
Practical response: Separate proposed, funded, qualified, and tested states.

### Review: False precision in rare-event output

Look for: A percentile is presented to the dollar from sparse data.
Why it matters: Uncertain dependence and severity are hidden.
Decision exposure: Risk appetite decisions rely on a fragile number.
Practical response: Show assumptions, uncertainty, and scenario sensitivities.

## 11. Assumptions and limits

The ledger below is specific to the fictional six-day case.
Its inputs are teaching assumptions, not market benchmarks or observed supplier performance.

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Opening stock is 40 usable units | Initial physical resource for the daily ledger | Quality hold, expiry, wrong location, or inaccurate count | Reconcile stock status and conduct a location-level count. |
| Demand is 100 units/day for six days | Keeps option comparisons on a common baseline | Surge, cancellations, mix change, or wrong forecast | Run demand paths and compare with orders and forecast error. |
| One component yields one saleable product | Connects inbound units directly to shipments | Scrap, assembly delay, or additional component constraints | Check yield, manufacturing routing, and bottleneck capacity. |
| Arrivals are usable before same-day demand | Sets the order of operations in each time bucket | Arrival after cutoff, inspection delay, or a missed shift | Check receiving timestamps and rerun with next-day availability. |
| Lost sales do not return | Avoids carrying backorders into later days | Customers wait, substitute, cancel later, or reorder | Use customer evidence and compare lost-sale versus backlog cases. |
| A and B recover fully on day 4 | Defines a finite three-day shock | Repair fails, labor is unavailable, or output ramps slowly | Validate recovery evidence and test staged ramp paths. |
| C supplies 30 units/day on each shock day | Represents a reserved and usable alternate | Supplier has competing demand or insufficient tooling | Obtain allocation, test qualification, and verify upstream inputs. |
| C is independent of the specified grid event | Makes C a hedge against the modeled common cause | Shared sub-tier, utility, transport route, or cyber service | Map sub-tier and lanes; run a correlated-C sensitivity. |
| USD 40/unit is lost contribution | Values a sale that disappears rather than revenue | Variable costs or achievable sales differ from the estimate | Reconcile with product finance and test margin ranges. |
| USD 5/unit penalty applies to all lost units | Illustrates a separate contractual charge | Grace periods, exclusions, caps, or no enforceable clause | Have legal and sales review the actual contract terms. |
| Reservation costs USD 600/window | Accounts for capacity available even without activation | Longer contract, setup, minimum take, or cancellation fees | Obtain a quote and match the payment period to the horizon. |
| The USD 12/unit premium applies to all C units | Represents incremental exercise cost | Fuel, duties, tiered pricing, or another included cost | Reconcile quote to delivered quantity and baseline unit cost. |
| No additional demand can be served after day 6 | Closes the modeled financial horizon | Backlog, stock value, future order cancellation, or recovery lasts longer | Extend the horizon and account for terminal stock and obligations. |
| No other costs or benefits are material | Keeps the illustrative bridge auditable | Cash timing, labor, freight, tax, safety, insurance, or reputation matter | Add material categories with sources and avoid overlap. |

### Assumption: Opening stock is 40 usable units

Role in the calculation: Initial physical resource for the daily ledger.
Failure condition: Quality hold, expiry, wrong location, or inaccurate count.
Evidence or sensitivity: Reconcile stock status and conduct a location-level count.

### Assumption: Demand is 100 units/day for six days

Role in the calculation: Keeps option comparisons on a common baseline.
Failure condition: Surge, cancellations, mix change, or wrong forecast.
Evidence or sensitivity: Run demand paths and compare with orders and forecast error.

### Assumption: One component yields one saleable product

Role in the calculation: Connects inbound units directly to shipments.
Failure condition: Scrap, assembly delay, or additional component constraints.
Evidence or sensitivity: Check yield, manufacturing routing, and bottleneck capacity.

### Assumption: Arrivals are usable before same-day demand

Role in the calculation: Sets the order of operations in each time bucket.
Failure condition: Arrival after cutoff, inspection delay, or a missed shift.
Evidence or sensitivity: Check receiving timestamps and rerun with next-day availability.

### Assumption: Lost sales do not return

Role in the calculation: Avoids carrying backorders into later days.
Failure condition: Customers wait, substitute, cancel later, or reorder.
Evidence or sensitivity: Use customer evidence and compare lost-sale versus backlog cases.

### Assumption: A and B recover fully on day 4

Role in the calculation: Defines a finite three-day shock.
Failure condition: Repair fails, labor is unavailable, or output ramps slowly.
Evidence or sensitivity: Validate recovery evidence and test staged ramp paths.

### Assumption: C supplies 30 units/day on each shock day

Role in the calculation: Represents a reserved and usable alternate.
Failure condition: Supplier has competing demand or insufficient tooling.
Evidence or sensitivity: Obtain allocation, test qualification, and verify upstream inputs.

### Assumption: C is independent of the specified grid event

Role in the calculation: Makes C a hedge against the modeled common cause.
Failure condition: Shared sub-tier, utility, transport route, or cyber service.
Evidence or sensitivity: Map sub-tier and lanes; run a correlated-C sensitivity.

### Assumption: USD 40/unit is lost contribution

Role in the calculation: Values a sale that disappears rather than revenue.
Failure condition: Variable costs or achievable sales differ from the estimate.
Evidence or sensitivity: Reconcile with product finance and test margin ranges.

### Assumption: USD 5/unit penalty applies to all lost units

Role in the calculation: Illustrates a separate contractual charge.
Failure condition: Grace periods, exclusions, caps, or no enforceable clause.
Evidence or sensitivity: Have legal and sales review the actual contract terms.

### Assumption: Reservation costs USD 600/window

Role in the calculation: Accounts for capacity available even without activation.
Failure condition: Longer contract, setup, minimum take, or cancellation fees.
Evidence or sensitivity: Obtain a quote and match the payment period to the horizon.

### Assumption: The USD 12/unit premium applies to all C units

Role in the calculation: Represents incremental exercise cost.
Failure condition: Fuel, duties, tiered pricing, or another included cost.
Evidence or sensitivity: Reconcile quote to delivered quantity and baseline unit cost.

### Assumption: No additional demand can be served after day 6

Role in the calculation: Closes the modeled financial horizon.
Failure condition: Backlog, stock value, future order cancellation, or recovery lasts longer.
Evidence or sensitivity: Extend the horizon and account for terminal stock and obligations.

### Assumption: No other costs or benefits are material

Role in the calculation: Keeps the illustrative bridge auditable.
Failure condition: Cash timing, labor, freight, tax, safety, insurance, or reputation matter.
Evidence or sensitivity: Add material categories with sources and avoid overlap.

### Limits of the reverse result

The threshold of 53.333 units/day is conditional on constant daily arrivals and continuous units.
It is not a universal supplier capacity threshold or a probability of failure.
If early arrivals differ from later arrivals, use the time-stepped ledger.
If customer demand is backlogged, count when and whether the customer ultimately receives it.
If the penalty has a cap, solve the revised loss function rather than multiplying by USD 45/unit.
If the threshold is a cash shortfall, model receipts and payments on their actual dates.
The threshold can be reached by long duration, lower capacity, greater demand, or delayed action.
A complete reverse exercise seeks decision-relevant combinations, not just one algebraic solution.

## 12. Exercises

Work independently first, then compare with Section 13.
Unless a problem changes an assumption, use the six-day example and USD-denominated costs.

### Exercise 1. Map a common-cause event

Type: Recall and map.
Time: 12 minutes.
Deliverable: Draw a graph for A, B, their shared grid, the assembler, customer, and C.
Mark affected links on days 1–3 and restored links on day 4.
Identify which assumed C dependency requires evidence.
Acceptance: the grid is a common parent, the product flow direction is correct, and missing evidence is explicit.

### Exercise 2. Calculate the stress ledger

Type: Application.
Time: 20 minutes.
Deliverable: Recreate days 1–6 of the unmitigated case using 40 opening units.
Compute each day’s shipment, lost units, and closing inventory.
Calculate total fill and compare it with the 90% target.
Acceptance: the balances reconcile and no inventory or lost demand is carried incorrectly.

### Exercise 3. Evaluate joint probability

Type: Application and critique.
Time: 15 minutes.
Deliverable: Assume a common grid event has probability 0.10 over one specified window.
Conditional on the event, A and B are both restricted with probability 0.80; without it, the joint restriction is 0.01.
Compute the joint restriction probability and explain why multiplying two unknown marginals is unjustified.
Acceptance: apply the law of total probability, report the window, and identify the dependence assumption.

### Exercise 4. Compare a delayed backup

Type: Calculation.
Time: 20 minutes.
Deliverable: Use the original stress path but let C deliver 30 units only on days 2 and 3.
Keep the USD 600 reservation, USD 12/unit premium, and USD 45 per lost unit.
Calculate the daily service, six-day loss, total cost, and conditional savings.
Acceptance: no C arrives on day 1, and the reservation and 60-unit premium are included once.

### Exercise 5. Find a reverse-stress boundary

Type: Calculation.
Time: 15 minutes.
Deliverable: Use the original no-C three-day path, but set the loss threshold to USD 3,600.
Solve for the constant A+B daily arrival rate at the threshold with continuous units.
State the direction of a breach and the domain on which the equation is valid.
Acceptance: derive the threshold from lost units and check timing, units, and inequality.

### Exercise 6. Design a mitigation decision

Type: Design.
Time: 25 minutes.
Deliverable: Compare C with an additional 60-unit opening buffer costing USD 900 for the window.
Assume the buffer is usable on day 1, has no other cost, and is not replenished during the shock.
Compute service and conditional costs for both; recommend an option with at least two readiness checks.
Acceptance: compare like-for-like horizons, include residual risk, and do not claim a probability-based return.

### Exercise 7. Challenge a tail estimate

Type: Critique.
Time: 15 minutes.
Deliverable: A report says the 95th percentile loss is USD 1 million based on independent supplier failures.
Identify four inputs or assumptions needed to interpret or challenge this statement.
Explain one way common-cause dependence can change the result and one limitation of historical events.
Acceptance: name the horizon, loss definition, dependence, and distribution evidence without assuming direction is guaranteed.

### Exercise 8. Write a decision and trigger memo

Type: Design.
Time: 25 minutes.
Deliverable: Write a one-page memo recommending or rejecting C for the fictional case.
Include the financial bridge, service consequence, threshold, causal path, and unverified assumptions.
Name a trigger, a decision owner, a deadline, and a condition that would change the decision.
Acceptance: distinguish conditional savings from expected savings and specify an executable next step.

## 13. Self-check answers

### Exercise 1: dependency map

The product paths are A → assembler → customer and B → assembler → customer.
The grid is a common causal parent of A and B, not a shipment flowing into the customer.
The grid event restricts the two arrivals to 20 units/day each on days 1–3.
The two streams return to 80 and 40 units/day respectively on day 4 by assumption.
The C → assembler path adds 30 units/day on days 1–3 in the mitigation case.
Its grid, input, transportation, and qualification independence require evidence.
A correct drawing distinguishes causal dependencies from physical component flows.

### Exercise 2: unmitigated ledger

Day 1: 40 opening + 40 arriving = 80 available, 80 shipped, 20 lost, 0 closing units.
Day 2: 0 + 40 = 40 available, 40 shipped, 60 lost, 0 closing units.
Day 3: 0 + 40 = 40 available, 40 shipped, 60 lost, 0 closing units.
Day 4: 0 + 120 = 120 available, 100 shipped, 0 lost, 20 closing units.
Day 5: 20 + 120 = 140 available, 100 shipped, 0 lost, 40 closing units.
Day 6: 40 + 120 = 160 available, 100 shipped, 0 lost, 60 closing units.
The total is 460 shipped units out of 600 demanded, or 76.67% fill.
The target is missed by 90% − 76.67% = 13.33 percentage points after rounding.
The 140 lost units do not return as backlog in this model.

### Exercise 3: joint probability

Let G be the grid event and J be the joint restriction over the specified window.
The law of total probability gives P(J) = P(J|G)P(G) + P(J|not G)P(not G).
The calculation is 0.80 × 0.10 + 0.01 × 0.90 = 0.089.
The resulting joint restriction probability is 8.9% for that window, conditional on the supplied assumptions.
Neither the individual marginal restrictions nor their independence were specified.
Multiplying invented marginal probabilities would not preserve the specified common cause.
A change in conditional restriction probabilities changes the answer even if P(G) stays fixed.

### Exercise 4: backup available only from day 2

Day 1 has 80 units available, so 80 ship and 20 are lost.
Days 2 and 3 each have 40 A+B units plus 30 C units, so 70 ship and 30 are lost per day.
The remaining days each ship 100 units and rebuild stock in the same way as the original case.
Total lost demand is 20 + 30 + 30 = 80 units; shipments are 520 units.
Fill is 520 / 600 = 86.67%, so the 90% target is missed.
C delivers 60 units at a USD 12/unit premium, costing USD 720.
Lost-unit cost is 80 × USD 45 = USD 3,600.
Total is USD 3,600 + USD 720 + USD 600 = USD 4,920.
Conditional savings are USD 6,300 − USD 4,920 = USD 1,380.
The delayed option is less effective than day 1 availability and exceeds the USD 4,500 boundary.

### Exercise 5: alternative reverse boundary

USD 3,600 / USD 45 per lost unit = 80 lost units.
Solve 300 − 40 − 3x = 80, giving x = 60 units/day.
At 60 units/day, day 1 ships 100 units, and days 2 and 3 ship 60 units each.
That produces 0 + 40 + 40 = 80 lost units and USD 3,600 of loss.
A rate below 60 units/day exceeds the boundary; exactly 60 hits it.
The calculation assumes constant arrivals in the 0–100 units/day domain and the original timing rules.
If arrivals or customer behavior change, run the daily ledger instead of applying this shortcut.

### Exercise 6: alternative buffer and rubric

With the extra 60 units, opening stock is 100 units rather than 40 units.
Day 1 starts with 100 and receives 40; it ships 100 and closes with 40 units.
Day 2 has 80 units, ships 80, loses 20, and closes with 0 units.
Day 3 has 40 units, ships 40, and loses 60 units.
The last three days ship 100 units/day and close with 60 units in total on day 6.
The buffer loses 80 units and serves 520 / 600 = 86.67% of demand.
Its conditional cost is 80 × USD 45 + USD 900 = USD 4,500.
C loses 50 units and costs USD 3,930, so it is USD 570 cheaper in this scenario.
C also meets the aggregate fill target while the buffer does not.
The buffer hits the loss threshold; C is below it under these specific assumptions.

A strong recommendation checks the availability and independence of C.
It also considers whether day 1 delivery is possible and the contract can be funded.
A buffer may still be preferable under an adverse case that also disables C.
An option is not approved merely because its cost is lower in one selected case.
The decision should identify the next evidence request and residual scenario exposure.

### Exercise 7: tail estimate critique rubric

A strong critique asks for the loss definition and whether it is gross or net of insurance.
It asks whether 95% refers to one window, one year, or another period.
It identifies the supplier-event marginals, conditional distributions, and common dependencies.
It asks how repair durations and correlated recovery were modeled.
It requests the simulation sample size, parameter evidence, and uncertainty in the percentile.
It explains that common causes can increase joint stress, but the percentile direction depends on the full model.
It notes that sparse historical observations may not cover present network structure or rare severe events.
A weak critique assumes the 95th percentile is the worst possible loss.

### Exercise 8: memo rubric

A strong memo identifies the six-day common-grid scenario and its unmitigated USD 6,300 cost.
It contrasts C's USD 3,930 conditional cost, 91.67% fill, and USD 2,370 scenario saving.
It states that the benefit is not an expected annual saving or a guaranteed loss avoidance.
It names a sourcing owner and a trigger with an actionable lead time before the first required receipt.
It records checks for reserved volume, qualification, independence, funding, and customer terms.
It specifies a decision deadline and who can authorize activation.
It explains how later first delivery, a shared input, or a different penalty could change the answer.
It includes remaining 50 lost units and asks owners to review day-level customer consequences.
A weak memo reports a percentage without the denominator, horizon, source, or operational response.

## 14. Interview questions

Use the contrasts to test reasoning, not memorization of a particular software package.

### Question 1: What makes a stress test decision-relevant?

A strong answer contains: A fixed decision, owner, horizon, threshold, baseline, and feasible actions.
A weak answer reveals: A generic disaster narrative with no consequence or decision.

### Question 2: How would you test two suppliers for a shared exposure?

A strong answer contains: Map their sites, upstream inputs, utilities, lanes, and conditional impacts under the common cause.
A weak answer reveals: Assume different company names imply independent failures.

### Question 3: How do severity and duration interact?

A strong answer contains: Translate the timed capacity path into inventory and service; long mild and short severe shocks can differ.
A weak answer reveals: Multiply a severity percentage by an annual spend number without time sequencing.

### Question 4: When is an alternate supplier not usable?

A strong answer contains: When qualification, available allocation, tooling, transport, legal clearance, or lead time prevents delivery.
A weak answer reveals: Treat an advertised supplier capacity as reserved and immediately available.

### Question 5: What is reverse stress testing?

A strong answer contains: Start with a defined breach and find feasible paths to it; test several drivers and report assumptions.
A weak answer reveals: Predict a single most likely disaster or choose an unconstrained extreme.

### Question 6: Why is lost revenue not the same as loss?

A strong answer contains: Bridge lost baseline units to contribution, penalties, costs, cash, and recoveries without overlap.
A weak answer reveals: Count gross sales revenue and all variable costs as simultaneous losses.

### Question 7: How do you treat backlog differently from a lost sale?

A strong answer contains: Track outstanding orders, cancellations, timing, priority, and later capacity; value only actual economic differences.
A weak answer reveals: Count the same unit as permanently lost and later shipped.

### Question 8: What does a 95th percentile loss mean?

A strong answer contains: A specified quantile of a modeled horizon and loss distribution, conditional on data and dependence.
A weak answer reveals: The worst possible loss, independent of assumptions and sample uncertainty.

### Question 9: How would you compare buffer stock with reserved capacity?

A strong answer contains: Use identical scenarios, timing, readiness, incremental costs, service metrics, and residual exposures.
A weak answer reveals: Compare the fixed cost of one option with the gross benefit of another.

### Question 10: What does a credible executive recommendation contain?

A strong answer contains: State the decision, threshold, conditional evidence, option costs, owner, trigger, readiness, and residual risk.
A weak answer reveals: Provide a colorful risk score with no timing, source, or execution owner.

## 15. Key terms

The links return to the vocabulary or to the relevant calculation and process.

- [Stress test](#4-definitions-and-vocabulary): A conditional evaluation of a defined adverse path.
- [Baseline](#4-definitions-and-vocabulary): The comparable no-shock operational and financial path.
- [Hazard](#4-definitions-and-vocabulary): An event or condition capable of creating a shock.
- [Exposure](#4-definitions-and-vocabulary): Presence of a vulnerable activity within the hazard footprint.
- [Shock](#4-definitions-and-vocabulary): A prescribed change in a model input.
- [Severity](#4-definitions-and-vocabulary): The magnitude of a shock relative to a stated reference.
- [Duration](#4-definitions-and-vocabulary): Time for which a specified restriction applies.
- [Recovery path](#4-definitions-and-vocabulary): The timed return of usable capacity or service.
- [Ramp](#4-definitions-and-vocabulary): A staged capacity restoration after repair.
- [Node](#4-definitions-and-vocabulary): A mapped site or activity with operational constraints.
- [Lane](#4-definitions-and-vocabulary): A feasible transport connection between locations.
- [Bill of materials](#4-definitions-and-vocabulary): A dated specification of component quantities per product.
- [Shared dependency](#4-definitions-and-vocabulary): An input or service used by multiple nominal alternatives.
- [Common cause](#4-definitions-and-vocabulary): One mechanism capable of affecting multiple elements.
- [Correlation](#4-definitions-and-vocabulary): A statistical measure of association in specified variables.
- [Conditional probability](#4-definitions-and-vocabulary): Chance of an event given another event or state.
- [Joint probability](#4-definitions-and-vocabulary): Chance that specified events occur together.
- [Concentration](#4-definitions-and-vocabulary): A high share of an activity in a common exposure.
- [Lead time](#4-definitions-and-vocabulary): Elapsed time from an initiating action to usable arrival.
- [Time-to-recover](#4-definitions-and-vocabulary): Elapsed time to restored usable capability under a stated definition.
- [Time-to-survive](#4-definitions-and-vocabulary): Elapsed time before a defined service or inventory threshold is crossed.
- [On-hand inventory](#4-definitions-and-vocabulary): Usable stock at a defined location and moment.
- [Allocation rule](#4-definitions-and-vocabulary): A rule for assigning scarce material to competing uses.
- [Lost sale](#4-definitions-and-vocabulary): Unfilled demand assumed not to return within the model.
- [Backlog](#4-definitions-and-vocabulary): Unfilled demand that remains eligible to be served later.
- [Service level](#4-definitions-and-vocabulary): A defined measure of service against demand or promise.
- [Contribution margin](#4-definitions-and-vocabulary): Revenue less defined avoidable variable costs.
- [Incremental loss](#4-definitions-and-vocabulary): Difference in economics between stressed and baseline cases.
- [Cash flow](#4-definitions-and-vocabulary): Timing of receipts and disbursements.
- [Penalty](#4-definitions-and-vocabulary): An incremental charge triggered by a specified obligation.
- [Mitigation](#4-definitions-and-vocabulary): An action that changes exposure, consequences, or response.
- [Residual risk](#4-definitions-and-vocabulary): Exposure that remains after an implemented action.
- [Reverse stress test](#4-definitions-and-vocabulary): Search from an unacceptable outcome back to a causing path.
- [Risk appetite](#4-definitions-and-vocabulary): A decision-maker-approved limit or boundary for exposure.
- [Tail percentile](#4-definitions-and-vocabulary): A quantile of an explicitly modeled loss distribution.
- [Seed](#4-definitions-and-vocabulary): An initial state for reproducible pseudorandom draws.

## 16. Sources

The standards and public cases support the named claims and methods; they do not supply the fictional case inputs.
Use original documents and current terms when making an actual commercial or compliance decision.

1. International Organization for Standardization (ISO), **ISO 22301:2019**,
   *Security and resilience — Business continuity management systems — Requirements*.
   [ISO standard record](https://www.iso.org/standard/75106.html).
   Scope: business continuity management-system requirements; not a stress-test certification by itself.
2. ISO, **ISO 31000:2018**, *Risk management — Guidelines*.
   [ISO standard record](https://www.iso.org/standard/65694.html).
   Scope: general risk-management guidance, not a prescribed financial model.
3. GS1, **Global Location Number (GLN)**.
   [GS1 identification-key reference](https://www.gs1.org/standards/id-keys/gln).
   Scope: identifying locations and parties; an identifier alone does not prove site independence.
4. Toyota Motor Corporation, **75 Years of Toyota: The Great East Japan Earthquake**.
   [Company history](https://www.toyota-global.com/company/history_of_toyota/75years/text/leaping_forward_as_a_global_corporation/chapter5/section5/item1.html).
   Scope: the company's account of safety, operational disruption, and response.
5. U.S. National Transportation Safety Board, **Contact of Containership Dali with Francis Scott Key Bridge and Subsequent Bridge Collapse**.
   [Investigation DCA24MM031](https://www.ntsb.gov/investigations/Pages/DCA24MM031.aspx).
   Scope: the 2024 Baltimore event and investigation, not private firm-specific loss estimates.
6. Tang, Christopher S. (2006), **Perspectives in supply chain risk management**,
   *International Journal of Production Economics*, 103(2), 451–488.
   [DOI: 10.1016/j.ijpe.2005.12.006](https://doi.org/10.1016/j.ijpe.2005.12.006).
   Scope: supply-chain risk management and mitigation context.
7. Sheffi, Yossi and Rice, James B. Jr. (2005), **A Supply Chain View of the Resilient Enterprise**,
   *MIT Sloan Management Review*, 47(1).
   [MIT Sloan Management Review article](https://sloanreview.mit.edu/article/a-supply-chain-view-of-the-resilient-enterprise/).
   Scope: resilience and supply-chain managerial response.

Sources 1–3 are standards or identification references, sources 4–5 are case records,
and sources 6–7 are methodological or management literature.
The standards may be voluntary unless adopted through an applicable law or contract.
The company's historical account and the investigation have different purposes and evidentiary limits.
No source assigns probabilities, validates the cost figures, or approves the option in the fictional case.

## 17. Links

- [Module 14A index](_index.md): topic sequence and assessment context.
- [Disruption timeline](01_disruption-timeline.md): detection, response, and recovery stages.
- [Mapping and exposure](03_mapping-exposure.md): physical nodes, lanes, and dependencies.
- [Time-to-recover and time-to-survive](04_ttr-tts.md): the recovery and survival clocks.
- [Scenario analysis](06_scenario-analysis.md): constructing and comparing coherent scenarios.
- [Buffering versus flexibility](07_buffering-flexibility.md): mitigation design and trade-offs.
- [Dual sourcing and nearshoring](08_dual-sourcing-nearshoring.md): qualification and sourcing choices.
- [Module 14A survey](../../../course/14A_resilience_disruption_analytics.md): module overview and sources.
- [Lab 14A.2 — TTR/TTS stress test](../../../course/14A_resilience_disruption_analytics.md#lab-14a2--ttrtts-stress-test): extend this case to three paths.
- [Lab 14A.4 — Scenario analysis and board memo](../../../course/14A_resilience_disruption_analytics.md#lab-14a4--scenario-analysis-and-board-memo): communicate decisions and insurance assumptions.

