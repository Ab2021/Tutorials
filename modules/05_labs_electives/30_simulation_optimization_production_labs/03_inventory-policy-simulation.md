# Inventory Policy Simulation

> Module 30 — Simulation, Optimization, and Real-Time Production Labs  
> Topic 03 of 10  
> Estimated study time: 3–4 hours  
> Prerequisites: basic Python, arithmetic, inventory position, and the module environment and forecasting labs  
> Version 2026.5

## 1. Why this topic exists

A planner must choose when and how much to reorder before the next demand realization is known.
A spreadsheet with one average-demand scenario hides the sequence of peaks, the time an order spends in transit, and the costs of running out.
A simulation exposes those consequences without changing a live replenishment parameter.
The output is decision support, not a production replenishment instruction.

The operational decision in this lab is a fixed-quantity reorder rule for one item at one location.
At the end of each day, the planner compares inventory position with a reorder point.
When the position is at or below the point, the planner places one fixed-size order.
The exercise compares a small, explicitly listed set of such rules.

The buyer cares about the number of orders and supplier workload.
The warehouse manager cares about physical inventory and storage capacity.
The service owner cares about units that cannot be filled immediately.
The finance owner cares about a consistent cost horizon and cost definitions.
These interests can conflict even when everyone agrees on the demand data.

Consider a high reorder point.
It can reduce lost sales by placing orders earlier.
It can also leave more stock sitting in the warehouse.
Consider a large order quantity.
It can spread fixed ordering costs over more units.
It can also increase exposure to overstock and expiry, which this model does not represent.
A credible recommendation states which of these consequences is priced and which is not.

The lab uses synthetic demand generated on the learner's computer.
It does not contact a supplier, connect to an enterprise resource planning system, or update stock records.
Its costs are illustrative US dollars, not a statement of any company's costs.
Its demand distribution is an experimental input, not a calibrated forecast.
A favorable result is not evidence of production service or savings.

Simulation is especially useful when sequencing matters.
Two demand paths can have the same total units but different runs of high-demand days.
The runs can cause different stockouts while replenishment is in transit.
Comparing policies on the same paths makes the differences easier to attribute to policy rather than random demand.

The lab deliberately uses a limited search grid.
The best rule in that grid is not necessarily globally optimal.
A policy outside the grid may have lower cost or better service.
A simulation can rank the policies it evaluates under its assumptions; it cannot establish that those assumptions describe the real system.

## 2. Learning outcomes

By the end of the lab you can:

1. Trace daily receipts, demand, lost units, inventory position, and an order decision.
   Exercise 1 checks the event order and its arithmetic.
2. Distinguish fill rate from the fraction of days with no stockout.
   Exercise 2 asks you to compute both measures.
3. Run a seeded synthetic-demand policy sweep and reproduce its inputs.
   Exercise 3 requires a clean rerun and recorded provenance.
4. Reconcile shortage, holding, and order costs with the daily trace.
   Exercise 4 asks for an auditable cost calculation.
5. Compare candidate policies on common demand paths while preserving replication-level variation.
   Exercise 5 asks for a paired comparison rather than two unrelated averages.
6. Critique a recommendation when service, capacity, lead time, or cost assumptions change.
   Exercise 6 asks you to explain the relevant tradeoff.
7. Design a validation and release plan that separates a sandbox result from a live control change.
   Exercise 7 asks for acceptance gates and rollback conditions.
8. Diagnose a misleading simulation caused by censoring or a timing error.
   Exercise 8 asks for evidence and a corrective test.

Success means more than identifying the row with the lowest printed cost.
You should be able to explain the row's inputs, timing, denominator, and limitations.
The stakeholder must be able to reproduce the calculation and understand why another policy may be preferable.

## 3. The core mental model

A policy is a feedback rule operating on a changing stock position.
Demand removes stock, replenishment adds stock after a delay, and the rule creates future replenishment.
The simulator records that loop one day at a time.

```text
 synthetic scenario seed
          |
          v
 daily demand path -------------------------------+
          |                                       |
          v                                       v
 opening stock + receipts --> serve --> closing stock --> costs and service
          ^                         |       |
          |                         |       v
 scheduled arrivals <--- order <--- position <= reorder point?
          ^
          |
     lead-time delay
```

The policy does not see future demand.
The experiment harness creates a full path in advance only so that every policy experiences the same realization.
The simulation reads one daily demand value at a time.
This distinction prevents accidental future-information leakage.

The stock balance is physical.
Opening on-hand plus receipts equals units served plus closing on-hand.
Unserved demand is recorded as lost sales, not carried as backlog.
The position balance includes outstanding replenishment orders as well as on-hand stock.
It does not include lost demand because a lost sale is not owed later.

Lead time joins today's ordering decision to a future morning receipt.
In this lab an order placed after demand on day `t` arrives before demand on day `t + L`.
For `L = 2` days, an order after day 1 arrives on the morning of day 3.
It does not protect against demand on day 2.
A one-day discrepancy in this convention can change the ranking of rules.

Each candidate faces a collection of independent replications.
Each replication is one possible sequence of daily demand.
Candidates within the same replication share that sequence.
The average across replications summarizes performance under the generator, not certainty about reality.
The difference between two candidates can be summarized by subtracting their results within each replication.

You can redraw the loop from memory using five objects:

- A stream of demand, measured in units per day.
- On-hand stock, measured in units.
- Open orders, each with a quantity and arrival day.
- A decision rule, with a reorder point and fixed quantity.
- A scorecard, with explicit cost and service definitions.

Do not confuse an input forecast with an operational plan.
The synthetic generator defines possible demand; it is not a forecast published to users.
A policy is a proposed decision rule; it is not an approved order plan.
A service threshold can be a decision target; it is not a customer commitment.
A commitment exists only when an accountable business owner accepts one outside this exercise.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Item | One product or stock-keeping unit (SKU), counted in units | Treating an aggregate family as one interchangeable item |
| Location | One stocking point | Mixing store and warehouse balances |
| Day | One discrete review and demand bucket | Assuming within-day events are modeled |
| Demand | Units requested during a day | Substituting sales during a stockout without adjustment |
| Synthetic demand | Artificial observations drawn from a specified generator | Calling generated values measured demand |
| Demand path | Ordered daily demand values for one replication | Keeping only a total and losing sequence |
| Seed | Integer initializing a pseudorandom generator | Assuming a seed alone describes the entire experiment |
| Replication | One independently seeded path and its policy outcomes | Treating days within one path as independent replications |
| On-hand | Physically available stock in units | Adding stock that is still in transit |
| Opening on-hand | Stock before that morning's receipts | Calling it the quantity after receipts |
| Receipt | Arrival of an earlier order before daily demand | Receiving an order on its placement day |
| Lead time | Integer days from end-of-day order to morning receipt | Mixing placement-to-dispatch and placement-to-availability |
| Pipeline | Orders placed but not yet received | Counting an already received order twice |
| Inventory position | Closing on-hand plus open order units in this lost-sales model | Treating on-hand alone as position |
| Reorder point `R` | Position threshold in units, inclusive | Assuming a reorder point is a target inventory level |
| Order quantity `Q` | Fixed units requested when the rule triggers | Assuming the rule orders up to `R` |
| Continuous review | Review at every relevant inventory event | Calling a daily end-of-day review continuous |
| Periodic review | Review at defined intervals | Ignoring the extra exposure between reviews |
| Fixed-quantity rule | Place a constant `Q` when the trigger holds | Placing enough orders to restore position without specifying it |
| Served units | Demand immediately satisfied from available stock | Including unavailable demand in fulfilled units |
| Lost units | Demand not immediately satisfied and not backordered | Treating lost sales as a deferred obligation |
| Stockout day | A day with at least one lost unit | Assuming it means closing inventory is zero |
| Fill rate | Served units divided by demand units | Calling the proportion of stockout-free days fill rate |
| Stockout-free day rate | Days without lost units divided by simulated days | Treating all days as equally weighted units of demand |
| Holding cost | Cost per closing unit per day in this model | Applying a per-year rate as if it were daily |
| Shortage penalty | Cost assigned per lost unit | Claiming an unobserved customer loss is a measured cost |
| Order cost | Fixed cost for each order placed | Charging on receipt while counting placements |
| Total modeled cost | Sum of modeled holding, shortage, and order costs | Presenting it as full profit or total landed cost |
| Horizon | Number of simulated days | Ignoring orders arriving after its end |
| Initial condition | Starting on-hand and pipeline | Assuming an arbitrary start is steady state |
| Warm-up | Period removed before measurement to reduce initialization effects | Deleting initial losses without a documented reason |
| Common random numbers | Same stochastic input paths for competing policies | Giving each candidate unrelated random streams |
| Paired difference | Candidate A outcome minus B outcome on the same path | Subtracting incompatible replications |
| Grid | Explicit finite set of candidate parameter pairs | Claiming the best grid row is a global optimum |
| Service constraint | Minimum required service measure over a stated population | Using a mean threshold as a per-path guarantee |
| Sensitivity analysis | Repeating a comparison after changing assumptions | Calling sampling noise a scenario change |
| Validation | Checking whether model behavior is fit for the intended use | Equating syntactically correct code with realistic behavior |
| Verification | Checking that the implementation follows its specification | Treating a passing unit test as empirical validation |
| Censored demand | Demand partly hidden because unavailable stock limits recorded sales | Assuming zero sales means zero customer demand |

## 5. The mechanics, step by step

### Step 1 — Frame a decision

**Input:** a named item, location, review frequency, and stakeholder decision.

**Action:** write the proposed policy change and the decision date.
Identify the service measure and the costs to be compared.
Separate a diagnostic question from a control decision.
For example, “Which of these rules is worth a pilot?” is narrower than “What should the system order?”

**Output:** one-page decision statement.
The lab's statement is: compare daily fixed-quantity policies for one synthetic SKU over 180 days.
The proposed choice is conditional on a synthetic unit fill-rate target of 95%.
The target is an exercise constraint, not an external service agreement.

**Branch:** if a customer order can wait, model backlog instead of lost sales.
That requires different balances and a different interpretation of service.
Do not switch between these mechanisms only by renaming the shortage metric.
If the actual problem includes multiple sites sharing stock, this one-location lab is insufficient.

### Step 2 — State the system boundary

**Input:** the process map and available observations.

**Action:** mark the inventory, the demand source, the replenishment source, and the clock.
The lab starts with 30 units on hand and no open orders.
It assumes a deterministic three-day replenishment lead time.
The process is reviewed once per day after demand.
There is one SKU, one location, and an unlimited supplier.
There are no returns, transfers, substitutions, expiry, capacity limits, or partial receipts.

**Output:** boundary and initial-state ledger.
A planner should be able to point to each quantity included in position.
A pipeline record contains an arrival day and a positive integer quantity.
The day's receipts are removed from pipeline before position is calculated.

**Branch:** if the item has a known open purchase order at the start, initialize its quantity and arrival date.
Do not silently treat that pipeline as zero in a production study.
If a physical count and a system count differ, investigate the discrepancy before using either as truth.

### Step 3 — Define event order

**Input:** the review cadence and the operational promise for a receipt.

**Action:** specify the daily timeline explicitly.

1. Begin with yesterday's closing on-hand and today's outstanding orders.
2. Receive all orders whose arrival day is today.
3. Observe that day's requested demand.
4. Serve the smaller of demand and available on-hand.
5. Record the difference as lost units.
6. Record closing on-hand after demand.
7. Sum closing on-hand and all remaining open order quantities.
8. If position is at most `R`, place exactly one order of `Q` units.
9. Schedule the order for morning of `today + L`.
10. Charge holding on closing on-hand, shortage on lost units, and order cost on placement.

**Output:** a timing contract that both code and manual traces follow.
Demand is not reserved ahead of the receipt.
An order cannot satisfy demand already lost on its placement day.
Multiple outstanding orders can coexist because the rule is reviewed daily.

**Branch:** if an actual system can issue several orders in one review, write that as a different rule.
A `while position <= R` implementation is not identical to the single-order rule here.
If receipts arrive after customer demand, reverse those two events deliberately and rerun the study.
Do not compare results from different timelines without labeling the change.

### Step 4 — Specify the policy grid

**Input:** operationally plausible thresholds, order sizes, supplier constraints, and storage constraints.

**Action:** list the candidate values before looking at simulation outcomes.
The executable lab uses `R` in {10, 20, 30} units and `Q` in {20, 40, 60} units.
The Cartesian product has nine candidate rules.
Each value is an integer; `Q` must be positive and `R` nonnegative.
The same initial stock and lead time apply to every candidate.

**Output:** nine labeled parameter pairs.
The grid is an experimental design, not a mathematical optimizer.
A larger grid costs more computation and can make in-sample selection more optimistic.
A small grid can miss a better rule between or beyond the chosen values.

**Branch:** if order sizes must be multiples of a case pack, eliminate infeasible `Q` values before scoring.
If each supplier order must exceed a minimum, encode that condition explicitly.
If maximum warehouse occupancy matters, report the simulated peak and exclude infeasible rules.
The provided script does not enforce capacity, so no row is certified capacity-feasible.

### Step 5 — Construct synthetic scenarios

**Input:** demand distribution, horizon, replication count, and seed schedule.

**Action:** create demand before evaluating candidates.
The script draws one integer uniformly from 2 through 10 units inclusive for each day.
Each replication uses a generator initialized with `BASE_SEED + replication_index`.
Its length is 180 days; there are 100 replications.
Every candidate uses each pre-generated path unchanged.

**Output:** 100 ordered synthetic demand paths.
Their expected daily demand under this generator is 6 units per day.
The generator has no seasonality, autocorrelation, trend, promotion, or rare large demand shock.
It is appropriate for learning the feedback loop, not for representing an arbitrary SKU.

**Branch:** if you replace the generator, declare the new distribution and its parameters.
Preserve the same path per candidate within a replication.
If lead time is random too, generate the applicable exogenous randomness independently of policy evaluation.
A simplistic shared sequence of random draws can become misaligned when policies place different numbers of orders.
Design shared supplier scenarios by calendar day or another explicit common key instead.

### Step 6 — Choose measurement units

**Input:** a cost model and agreed service definitions.

**Action:** express all costs on the same horizon and in US dollars.
The lab charges $0.05 per unit of closing stock per day.
It charges $8 per lost unit and $3 per order placed.
The shortage value is a hypothetical penalty, not necessarily accounting expense.
There is no purchase cost, salvage value, disposal cost, tax, or financing adjustment beyond the holding charge.

**Output:** a documented objective and units.
For each path, modeled cost equals holding plus shortage plus order cost.
Unit fill rate equals total served divided by total demand on that path.
The stockout-free day rate is the number of days with zero lost units divided by 180 days.
The two rates answer different questions.

**Branch:** if the shortage penalty already includes lost margin, do not add the same margin loss twice.
If purchase cost matters, account for opening stock, ending stock, receipts, and pipeline consistently.
A naive purchase charge at placement can punish orders still in transit at the horizon.
If total demand is zero, fill rate is undefined rather than automatically 100%.
The supplied positive-demand generator avoids that case, and the code uses `None` for it.

### Step 7 — Simulate each day

**Input:** one path, one policy, initial state, lead time, and costs.

**Action:** iterate through the timing contract.
Compute receipts from all due pipeline records.
Compute available stock from opening on-hand plus receipts.
Compute served units as `min(available, demand)`.
Compute lost units as demand minus served units.
Compute closing stock as available minus served units.
Compute position using closing stock and the remaining pipeline.
Place one order if the position is at most the reorder point.

**Output:** daily trace and path summary.
Every trace row contains day, demand, receipt, served, lost, closing stock, pre-order position, and placed quantity.
The label “pre-order” matters: position in the trace excludes an order placed later that day.
The next day's pipeline does include that order until its receipt.

**Branch:** if stock is negative after service, the accounting is wrong for this lost-sales model.
If a zero-unit order appears, validate `Q`.
If an order arrives earlier than `day + L`, inspect date arithmetic.
If a receipt appears twice, inspect pipeline removal.

### Step 8 — Reconcile invariants

**Input:** trace rows and scenario totals.

**Action:** check nonnegative stock and loss.
Check that demand equals served plus lost each day.
Check that available stock equals served plus closing stock each day.
Check that the ending stock equals initial stock plus receipts minus served units over the path.
Check that position is closing stock plus open orders before the new decision.
Check that each placed order arrives on the configured day if that day is within the horizon.

**Output:** a testable balance for every simulated day.
The example script asserts several of these properties and supplies a deterministic trace test.
The test is deliberately short enough to verify by hand.
The test is not proof that the stochastic demand generator is realistic.

**Branch:** a passing balance with implausibly high service can still indicate a modeling error.
Review receipt timing, artificial demand, and initial stock.
A model can conserve units and still represent the wrong business process.

### Step 8A — Challenge the trace with boundary cases

**Input:** a known initial state, a short demand path, and one candidate rule.

**Action:** design cases with outcomes that follow directly from the timing contract.
Run these cases separately from the random sweep so a lucky average cannot hide an error.
Record the exact expected trace before looking at the implementation output.

Start with demand equal to available stock.
All demand is served and closing on-hand is zero.
Zero closing stock is not itself a stockout day because no unit was lost.
The review may nevertheless trigger an order if position is at or below the threshold.

Next use demand greater than available stock by one unit.
Exactly one unit is lost if there is no receipt that morning.
An order placed at the end of that day must not reduce that loss.
The order changes future pipeline, not the historical demand outcome.

Next place a receipt on the day of demand.
In the specified event order, the receipt is available before demand.
If the receipt is sufficient, that day can have no loss even when opening on-hand was zero.
Changing this assumption is a model change, not merely a reporting change.

Next put the position exactly at the reorder point.
The inclusive comparison requires an order.
Position one unit above the point must not trigger an order.
These two adjacent cases detect an incorrect strict inequality.

Next end the path immediately after placing an order.
The order cost is included even if the receipt is outside the horizon.
The pipeline should retain its future arrival date and quantity.
This case makes the model's terminal convention visible.

**Output:** a small suite of hand-derived timing and boundary checks.
Each check should identify inputs, the expected row, and the reason for the expectation.
A passing suite is implementation evidence, not evidence that the generator fits actual demand.

### Step 8B — Investigate comparison uncertainty

**Input:** replication-level results for two candidate policies on the same paths.

**Action:** form one paired difference per path before summarizing.
List the minimum, maximum, and median difference as well as the mean.
Count positive, negative, and zero differences to show how often each candidate wins.
Examine the paths with the largest losses for both policies.
Inspect whether an unusual run of demand or a late replenishment explains the result.

For an uncertainty interval, use the standard deviation of the paired differences.
For independent replications, divide that standard deviation by the square root of the replication count to obtain the estimated standard error of their mean.
A conventional approximate interval can be constructed around the mean if its assumptions are appropriate.
For a small sample or a skewed distribution, consider a replication-level bootstrap instead.
Resample whole paired differences, not individual days from within a path.
Individual daily observations within a replication share inventory state and are not independent outcomes.

**Output:** a comparison showing both central tendency and variability.
The supplied script prints only the mean paired difference; it retains individual results in `results` for further analysis.
Any added interval is an extension to the lab, not an output already asserted by the script.

**Branch:** if a difference is small relative to its uncertainty, do not present a definitive rank.
Increase independent replications or select on a separate evaluation set if that supports the decision.
More replications reduce Monte Carlo error under the generator but do not repair a wrong demand model.
Sensitivity to assumptions and sampling variability are separate forms of uncertainty.

### Step 9 — Aggregate without hiding denominators

**Input:** one summary per policy and replication.

**Action:** average the path-level total cost, lost units, order count, and fill rate across paths.
The printed fill rate is the arithmetic mean of replication-level ratios.
It is not a pooled ratio of served units to demand units across all paths.
Both are legitimate summaries when explicitly labeled, but they differ when total demand varies.
The 95% selection threshold in the lab applies to the mean of path-level rates.
It does not require every path to reach 95%.

**Output:** one comparison row per candidate.
For each row, retain the replication-level results for diagnostic comparison.
The script computes and prints the best feasible row, if any, and a cost-only row.
The two may coincide; that is a possible experimental outcome, not an error.

**Branch:** if no candidate meets the target, report that no tested policy is feasible under the chosen criterion.
Do not choose the least bad candidate and label it compliant.
Either revise the decision, expand the grid, or gather better inputs with stakeholder agreement.
A mean-service constraint should not be used when a contractual minimum is required for each customer or period.

### Step 10 — Compare paired paths

**Input:** two policies evaluated on the same path indices.

**Action:** calculate `cost_A - cost_B` for each replication.
A negative difference favors A on modeled cost for that path.
The mean of differences equals the difference between the two mean costs when all paths are paired.
Inspect the distribution, not just its mean.
Count paths where A has higher loss even if its mean cost is lower.

**Output:** a paired cost comparison and service comparison.
Common demand paths often reduce comparison noise when outcomes are positively correlated.
They do not guarantee variance reduction for every pair or every metric.
They do not compensate for an unrealistic generator.

**Branch:** if scenarios have different weights, use an agreed weighted comparison.
If a candidate's results are missing for a path, resolve the missing result instead of silently changing the denominator.
If choosing among many candidates, evaluate the selected rule on new independent paths to reduce selection optimism.

### Step 11 — Examine finite-horizon effects

**Input:** starting inventory, pipeline, simulation horizon, and final state.

**Action:** inspect the first and last days.
Starting with 30 units and no open orders can temporarily favor some rules.
Orders placed near day 180 can remain in the pipeline after the score stops.
The code charges their order cost but does not value their residual units.
Closing inventory receives holding charges while present but has no terminal value.

**Output:** a statement of horizon limitations.
Compare a longer horizon or a documented warm-up before drawing steady-state conclusions.
If you add terminal inventory value, define the financial objective and reconcile all purchase costs as well.
A finite-horizon cost without a consistent terminal treatment can favor delaying purchases.

**Branch:** if the operational decision is a finite seasonal campaign, the end condition may be central rather than a nuisance.
Model expiry, salvage, leftover commitments, and demand after the campaign if they matter.
Do not assume a long-run stationary policy is appropriate for a discontinuing product.

### Step 12 — Decide and document

**Input:** candidate table, service requirement, sensitivity checks, and operational constraints.

**Action:** eliminate infeasible candidates before comparing costs.
Describe what changed relative to the current rule.
Document expected benefit as a modeled outcome under named scenarios, not as realized savings.
Specify who owns the source data, the proposed parameters, the approval, and the rollback.

**Output:** a decision record with recommendation, alternatives, evidence, and caveats.
Record the script, Python version, seeds, scenario configuration, units, and date of run.
Include the full table or an accessible artifact, not only the winning row.
State whether the recommendation is to pilot, gather data, change the grid, or make no change.

**Branch:** if inputs cannot be validated, stop before operational deployment.
If another policy better meets a non-modeled constraint, present that fact rather than hiding it behind cost rank.
If results are close relative to their scenario variability, treat the ranking as uncertain.

## 6. Worked example

This hand calculation uses a short deterministic demand fixture.
It is independent of the 180-day sweep below and can be checked without Python.
All quantities are units; every period is one day.

### Setup

| Input | Value | Interpretation |
|---|---:|---|
| Opening stock | 3 units | Before day 1 receipt |
| Initial pipeline | 0 units | No outstanding replenishment |
| Reorder point | 2 units | Trigger when position is at most 2 |
| Fixed quantity | 4 units per order | At most one order per day |
| Lead time | 2 days | Order on day `t` arrives at start of day `t + 2` |
| Demand | 2, 3, 1, 2 units | Four-day deterministic path |
| Holding cost | $0.50 per closing unit-day | Charged after each day's demand |
| Shortage penalty | $10 per lost unit | Lost sales, no backlog |
| Order cost | $3 per placed order | Charged on placement |

### Day 1

Opening stock is 3 units and no order arrives.
Demand is 2 units.
The location serves 2 units and loses 0 units.
Closing stock is `3 - 2 = 1` unit.
There is no open order before the decision, so pre-order position is 1 unit.
Since `1 <= 2`, place an order for 4 units.
Its receipt is scheduled for the morning of day 3.
The day's holding charge is `1 × $0.50 = $0.50`.
The day's order charge is $3 and its shortage charge is $0.

### Day 2

Opening stock is 1 unit.
The order from day 1 has not arrived because its lead time is two days.
Demand is 3 units.
Serve 1 unit and lose `3 - 1 = 2` units.
Closing stock is 0 units.
The outstanding 4-unit order makes pre-order position `0 + 4 = 4` units.
Because `4 > 2`, do not place another order.
The holding charge is $0 and the shortage charge is `2 × $10 = $20`.

### Day 3

The 4-unit order arrives before demand.
Available stock is `0 + 4 = 4` units.
Demand is 1 unit, so serve 1 and lose 0.
Closing stock is 3 units.
No order remains in the pipeline before the decision.
Position is 3 units; because `3 > 2`, no order is placed.
The holding charge is `3 × $0.50 = $1.50`.

### Day 4

Opening stock is 3 units and no order arrives.
Demand is 2 units, all served.
Closing stock is 1 unit.
Position is 1 unit before any new order.
Because `1 <= 2`, place another 4-unit order for arrival on day 6.
The horizon ends on day 4; the day 6 receipt is outside the horizon.
The holding charge is $0.50 and the order charge is $3.

### Trace and reconciliation

| Day | Opening | Receipt | Demand | Served | Lost | Close | Pre-order position | Order | Arrival |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 0 | 2 | 2 | 0 | 1 | 1 | 4 | 3 |
| 2 | 1 | 0 | 3 | 1 | 2 | 0 | 4 | 0 | — |
| 3 | 0 | 4 | 1 | 1 | 0 | 3 | 3 | 0 | — |
| 4 | 3 | 0 | 2 | 2 | 0 | 1 | 1 | 4 | 6 |

Total demand is `2 + 3 + 1 + 2 = 8` units.
Total served is `2 + 1 + 1 + 2 = 6` units.
Total lost is 2 units, so `6 + 2 = 8` units.
Total receipts within the horizon are 4 units.
Closing stock reconciles as `3 + 4 - 6 = 1` unit.
The second order remains open at the end of the measurement window.

Unit fill rate is `6 / 8 = 0.75`, or 75%.
Three of four days have no lost units, so stockout-free day rate is 75%.
The equality of these two rates in this fixture is accidental.
For example, losing one unit on a day with 100 units demanded would affect unit fill rate differently from the day count.

Closing-stock sum is `1 + 0 + 3 + 1 = 5` unit-days.
Holding cost is `5 × $0.50 = $2.50`.
Shortage cost is `2 × $10 = $20.00`.
There are two orders, so order cost is `2 × $3 = $6.00`.
Total modeled cost is `$2.50 + $20.00 + $6.00 = $28.50`.
The $28.50 includes the day 4 order fee even though its goods arrive after the horizon.
It excludes the purchase price and the economic value of the outstanding units.
That convention is intentional for the calculation, but it is not a complete profit model.

### Interpreting the example

The day 2 shortage is not prevented by the day 1 order because receipt occurs on day 3.
The 4 units in transit prevent another order on day 2 because inventory position includes the pipeline.
Using on-hand instead of position would produce a different order decision.
Counting the pipeline as physically available would incorrectly eliminate the day 2 shortage.
Both errors change business conclusions even though both use the same demand fixture.

A different reorder point might have triggered earlier only if there had been an earlier review opportunity.
The model begins immediately before day 1, so it cannot place an order before the start.
This highlights why starting conditions matter when comparing short simulations.
No general service claim should be inferred from a four-day path.

## 7. Code, query, or calculation

### Environment and sandbox

Use Python 3.10 or later with the standard library only.
No package installation, credentials, external data, network access, or paid service is needed.
Use a local scratch directory rather than a production repository or connected system.
The script writes no files and never places a real order.
All costs, demand values, and service targets are synthetic.

1. Create an empty local directory.
2. Save the code below as `inventory_lab.py` in that directory.
3. Run `python inventory_lab.py` from the directory; on some Windows installations use `py inventory_lab.py`.
4. Read the `fixture test passed` line and the nine candidate rows.
5. Record the Python version with `python --version`, the script content, and the seed configuration.
6. Repeat the run in the same environment and compare output; exact float formatting is fixed in the report.

The script uses deterministic pseudorandom input and a fixed evaluation order.
Python's random implementation and version are part of the reproducibility record.
The fixed fixture asserts the worked example before the sweep starts.
The assertions are executable checks, not a substitute for independent business validation.

```python
from dataclasses import dataclass
from random import Random
from statistics import mean


@dataclass(frozen=True)
class Policy:
    reorder_point: int
    quantity: int


@dataclass(frozen=True)
class Costs:
    holding_per_unit_day: float
    shortage_per_unit: float
    per_order: float


def simulate(demand, policy, lead_days, initial_stock, costs):
    """Receipt, demand, then one end-of-day review; unfilled demand is lost."""
    if lead_days < 1 or initial_stock < 0:
        raise ValueError("lead_days must be >= 1 and initial_stock >= 0")
    if policy.reorder_point < 0 or policy.quantity <= 0:
        raise ValueError("invalid policy")
    if any(d < 0 or not isinstance(d, int) for d in demand):
        raise ValueError("demand must contain nonnegative integers")

    stock = initial_stock
    pipeline = []  # (arrival day, quantity)
    trace = []
    total_demand = total_served = total_lost = 0
    closing_sum = orders = receipt_sum = 0
    for day, requested in enumerate(demand, start=1):
        opening = stock
        received = sum(q for arrival, q in pipeline if arrival == day)
        pipeline = [(arrival, q) for arrival, q in pipeline if arrival != day]
        available = opening + received
        served = min(available, requested)
        lost = requested - served
        stock = available - served
        position = stock + sum(q for _, q in pipeline)
        ordered = 0
        arrival_day = None
        if position <= policy.reorder_point:
            ordered = policy.quantity
            arrival_day = day + lead_days
            pipeline.append((arrival_day, ordered))
            orders += 1
        assert requested == served + lost
        assert available == served + stock
        assert stock >= 0 and lost >= 0
        total_demand += requested
        total_served += served
        total_lost += lost
        closing_sum += stock
        receipt_sum += received
        trace.append({"day": day, "opening": opening, "receipt": received,
                      "demand": requested, "served": served, "lost": lost,
                      "close": stock, "position": position, "order": ordered,
                      "arrival": arrival_day})

    assert initial_stock + receipt_sum - total_served == stock
    holding = closing_sum * costs.holding_per_unit_day
    shortage = total_lost * costs.shortage_per_unit
    ordering = orders * costs.per_order
    return {"trace": trace, "demand": total_demand, "served": total_served,
            "lost": total_lost, "orders": orders, "closing_sum": closing_sum,
            "fill": total_served / total_demand if total_demand else None,
            "stockout_free_days": sum(row["lost"] == 0 for row in trace),
            "holding": holding, "shortage": shortage, "ordering": ordering,
            "cost": holding + shortage + ordering, "pipeline": pipeline}


def fixture_test():
    result = simulate([2, 3, 1, 2], Policy(2, 4), 2, 3,
                      Costs(0.50, 10.0, 3.0))
    expected = [
        (3, 0, 2, 2, 0, 1, 1, 4, 3),
        (1, 0, 3, 1, 2, 0, 4, 0, None),
        (0, 4, 1, 1, 0, 3, 3, 0, None),
        (3, 0, 2, 2, 0, 1, 1, 4, 6),
    ]
    fields = ("opening", "receipt", "demand", "served", "lost", "close",
              "position", "order", "arrival")
    assert [tuple(row[k] for k in fields) for row in result["trace"]] == expected
    assert (result["demand"], result["served"], result["lost"]) == (8, 6, 2)
    assert result["closing_sum"] == 5 and result["orders"] == 2
    assert result["fill"] == 0.75 and result["stockout_free_days"] == 3
    assert result["cost"] == 28.50 and result["pipeline"] == [(6, 4)]


def main():
    fixture_test()
    print("fixture test passed")
    base_seed, replications, days = 2026, 100, 180
    paths = []
    for replication in range(replications):
        rng = Random(base_seed + replication)
        paths.append([rng.randint(2, 10) for _ in range(days)])
    policies = [Policy(r, q) for r in (10, 20, 30) for q in (20, 40, 60)]
    costs = Costs(0.05, 8.0, 3.0)
    results = {}
    print("R  Q   mean_cost_USD  mean_lost_units  mean_orders  mean_fill")
    for policy in policies:
        runs = [simulate(path, policy, 3, 30, costs) for path in paths]
        results[policy] = runs
        print(f"{policy.reorder_point:2d} {policy.quantity:2d} "
              f"{mean(x['cost'] for x in runs):14.2f} "
              f"{mean(x['lost'] for x in runs):16.2f} "
              f"{mean(x['orders'] for x in runs):12.2f} "
              f"{mean(x['fill'] for x in runs):10.4f}")
    average_cost = lambda p: mean(x["cost"] for x in results[p])
    feasible = [p for p in policies
                if mean(x["fill"] for x in results[p]) >= 0.95]
    cheapest = min(policies, key=average_cost)
    print(f"cost-only: {cheapest}")
    if feasible:
        best = min(feasible, key=average_cost)
        print(f"mean-fill >= 0.95: {best}")
        differences = [a["cost"] - b["cost"]
                       for a, b in zip(results[best], results[cheapest])]
        print(f"paired feasible-minus-cost-only mean USD: {mean(differences):.2f}")
    else:
        print("no tested policy meets mean-fill >= 0.95")


if __name__ == "__main__":
    main()
```

### Expected outcomes and checks

The first output line must be `fixture test passed`.
An assertion failure stops the script rather than printing a misleading sweep.
The table must contain exactly nine policy rows, each with a nonnegative mean cost, mean lost units, and mean order count.
Every printed mean fill rate must be between 0 and 1 inclusive.
The script prints the cheapest tested rule and either a feasible tested rule or an explicit no-feasible-policy message.
If the two rules are the same, the paired difference is $0.00 by construction.
If they differ, a nonnegative mean cost difference follows because the cost-only rule minimizes mean cost over the same grid.
No particular policy or output number is claimed here without an actual run.

For a separate timing check, change the four-day fixture by hand, not the supplied expected result, and recalculate its arrivals before modifying assertions.
Try a zero-demand path and confirm that fill rate is `None` rather than dividing by zero.
A path with zero demand may still generate orders if the initial position meets the trigger; that is the stated rule.
For performance interpretation, restore the supplied configuration before comparing candidate rows.

### Suggested experiment after the baseline

Copy the baseline output into a local experiment record.
Change only the shortage penalty, preserving paths, policies, and all other settings.
Compare the resulting cost ranking while recognizing that service outcomes for a fixed policy do not change when only the cost coefficient changes.
Then restore the coefficient and increase lead time by one day.
The latter changes the physical dynamics, so it can change both service and cost.
Neither scenario establishes a production business case.

## 8. Data, standards, and regulatory context

### Data contract

The synthetic lab does not require customer or supplier data.
A real deployment needs at least item and location identifiers, dates, inventory snapshots, transactions, and purchase order events.
Each quantity needs a unit of measure and a conversion to the simulation's base unit.
Each timestamp needs a time zone and a defined business-day boundary.
A receipt date should mean available for allocation, not merely a truck's arrival at a gate.

A useful demand fact is the customer's requested quantity at the time of request.
Shipped units alone may be lower because inventory was unavailable.
A canceled order, a substitution, and a deferred order need distinct reason codes where possible.
Without that distinction, observed sales during a stockout censor demand.
Calibrating the generator to those sales can understate demand precisely when service was poor.

An inventory snapshot should distinguish available, reserved, damaged, and in-transit units.
The model uses sellable available stock, not all physically present stock.
If reservations cannot be separated, the simulator may overstate the inventory available to new demand.
Reconcile opening and closing snapshots with receipts, shipments, adjustments, and transfers.
Investigate negative balances and unexplained adjustments before treating a path as ground truth.

For replenishment, preserve order placement, supplier confirmation, dispatch, physical receipt, and availability dates separately.
The relevant lead time depends on the event definition in the simulation.
Partial receipts require multiple arrival events or a policy for aggregating them.
Late or canceled orders cannot be represented by the lab's fixed lead time without loss of realism.

### Identifiers and exchange formats

GS1 publishes identification standards including the Global Trade Item Number (GTIN) for identifying trade items.
A GTIN identifies a trade item; it does not by itself encode the inventory state, the demand distribution, or the correct policy.
A real data contract should also identify location, package level, unit conversions, and effective dates.
Do not assume a case-level identifier and an each-level identifier carry interchangeable quantities.

GS1 electronic data interchange (EDI) standards support structured business-message exchange.
A purchase order or despatch advice message can supply events for a pipeline ledger when its semantics are mapped carefully.
A syntactically valid message is not evidence that the receipt became available at the modeled time.
Local implementation agreements and partner-specific fields still require documentation.
The synthetic lab does not require a GTIN or an EDI connection.

### Standards versus obligations

A standard may be adopted voluntarily, required by a trading partner, or incorporated into a binding rule in a particular context.
The lab does not assert that any specific inventory policy is legally required.
Applicable consumer protection, product safety, privacy, financial reporting, and sector-specific rules depend on jurisdiction and use.
Consult the appropriate official authority and qualified owner before relying on a compliance interpretation.
The model itself makes no regulatory certification claim.

For sensitive customer-order data, apply data minimization and access controls.
Pseudonymization does not automatically eliminate privacy obligations.
Retention, access, and processing purposes must follow the organization's policy and applicable law.
The synthetic demand paths here contain no real customer identifiers.

### Versioning and lineage

A reproducible run needs the script version and the full input configuration.
Record demand-generation method, seed schedule, horizon, initial state, and candidate grid.
Also record cost coefficients, units, receipt timing, service denominator, and selection criterion.
If a dataset replaces the generator, record its snapshot, extraction time, filters, and transformations.
Record Python version and dependency versions if external packages are introduced.

A seed does not reconstruct a data extract that has changed.
It also does not describe a generator whose algorithm or call order changed.
Keep both the seed and the code or immutable version identifier.
If another person cannot determine which inputs produced the table, the table is not a reliable decision artifact.

## 9. Industry practice and case evidence

### What the literature establishes

Inventory theory has a long history of comparing ordering rules under uncertain demand.
The literature distinguishes continuous review, periodic review, backorders, and lost sales.
These are different models because the timing of information and the obligations after a shortage differ.
A rule that performs well in one setting does not automatically transfer to another.
See Silver, Pyke, and Thomas in the sources for a standard inventory-management treatment.

Zipkin's work on periodic-review systems with lost sales studies the structure and comparison of policies in that setting.
That published research supports treating lost sales as a distinct modeling problem.
It does not validate this lab's particular grid, costs, lead time, or synthetic demand.
The present simulation is an educational implementation, not a reproduction of a published case.

### Public company evidence and inference

Walmart's annual reporting discusses the role of inventory management and supply chain operations in its business.
The cited report is a public company description, not an independently verified estimate of the effect of this lab's policy.
It does not disclose a comparable experiment for the parameter pairs here.
**Inference:** a large retailer's inventory decision needs operational constraints and financial measures beyond a one-item synthetic simulation.
Do not infer a measured service improvement, cost saving, or deployed reorder threshold from this example.

### Practice: compare rules against a credible baseline

A practical study begins with the current executable policy, not just an idealized policy from a textbook.
The baseline may contain manual overrides, supplier calendars, and minimum quantities.
Record whether the simulator reproduces those choices before attributing a difference to a new rule.
If historical orders cannot be reconstructed, state that the baseline is approximate.

Historical replay can expose the consequences of a new policy on an observed sequence.
It does not automatically reveal demand that was never observed during historical shortages.
Nor does it guarantee that a supplier would have delivered different hypothetical orders on the historical schedule.
Supplier capacity and customer behavior can respond to the policy being evaluated.
Those counterfactual assumptions should be visible in the decision record.

### Practice: select for service as well as cost

A cost-only optimum reflects the chosen shortage penalty.
If the penalty omits strategic customers, contractual obligations, or substitution effects, the selected rule may be inappropriate.
A service constraint makes the decision criterion explicit but does not solve measurement problems.
The lab's mean fill-rate threshold is not a per-customer, per-day, or worst-case guarantee.
A production owner may require segment-specific thresholds or tail-risk limits.

### Practice: move through controlled stages

A useful progression is unit tests, historical reconciliation, scenario testing, shadow operation, limited pilot, and monitored release.
Each stage has a different claim.
Unit tests show that specified calculations behave as expected on examples.
Historical reconciliation checks whether inputs and recorded outcomes can be explained.
Shadow operation compares suggested decisions without sending them to a supplier.
A pilot can observe real operational behavior but must have approval, monitoring, and a rollback owner.

A policy may fail because its upstream data is late or because a supplier cannot honor order frequency.
Such a failure is not fixed by reporting a better simulated mean cost.
The operational decision must include who investigates exceptions and who can stop automation.
No stage of this synthetic lab itself authorizes a live parameter change.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Sales mistaken for demand | Low recorded demand coincides with empty shelves | Unavailable units suppress observed sales | Reorder points are understated | Investigate lost requests and censoring |
| Pipeline omitted | Many duplicate orders in trace | On-hand is used instead of position | Overordering and excess stock | Reconcile open orders before review |
| Pipeline treated as on-hand | No simulated loss during a long transit | Future receipts satisfy current demand | Service is overstated | Separate physical and in-transit balances |
| Wrong event timing | Receipt appears on order day | Lead-time convention is inconsistent | Invalid ranking | Hand-check the day-by-day fixture |
| Arrival counted twice | Stock balance rises without a new receipt | Due order remains in pipeline | Phantom inventory | Remove arrivals once and assert balance |
| Backlog confused with loss | Unfilled units vanish despite promised delivery | Wrong shortage mechanism | Obligations and service misreported | Implement and test a backlog ledger |
| Average hides tail | Good mean with several severe paths | High variance or rare shocks | Unacceptable service for some periods | Inspect path distributions and tail measures |
| Different paths per policy | Large rank changes across reruns | Unpaired scenario noise | Comparison is unstable | Use common paths and paired differences |
| Seed recorded without code | Recreated paths differ | Generator or version changed | Results cannot be reproduced | Save code, runtime, inputs, and seeds |
| Horizon too short | End-of-horizon orders dominate cost | Terminal pipeline has no value | Policy ranking is biased | Extend horizon or model terminal economics |
| Initial stock dominates | Early performance differs sharply | Artificial opening condition | Misleading steady-state inference | Test initial states and warm-up |
| Cost units mixed | Holding cost dwarfs purchase value | Annual rate charged daily | Wrong objective | Convert and label all units |
| Stockout penalty double-counted | Profit and penalty both include lost margin | Same loss charged twice | Excessive buffer recommendation | Map every cost to one economic effect |
| Grid misses feasible region | Best row sits at grid boundary | Search range too narrow | Better candidates untested | Expand grid and validate on new paths |
| Selection on noise | Winner changes on fresh paths | Many candidates tuned to one sample | Optimistic result | Reserve independent evaluation paths |
| Mean target misread | Some paths fail despite feasible label | Average is mistaken for guarantee | Service promise breached | Define population and required percentile |
| Capacity ignored | Peak stock exceeds storage space | Unlimited warehouse assumption | Rule cannot be executed | Model peak and apply capacity constraint |
| Supplier constraint ignored | Order count exceeds vendor capacity | Unlimited immediate acceptance assumed | Orders delayed or rejected | Encode calendar and supplier limits |
| Random stream misalignment | Policies experience different supply shocks | Random draws consumed per order | Confounded comparisons | Key exogenous scenarios explicitly |
| Double-counted receipts | Ledger and snapshot both add same transaction | Snapshot time is unclear | Phantom stock | Define snapshot cutoffs and reconcile |
| Fractional or inconsistent units | Orders violate case packs | Unit conversions are missing | Infeasible recommendations | Normalize units and enforce pack sizes |
| Changing multiple assumptions | Ranking moves but cause is unknown | Scenarios are confounded | Weak explanation | Change one assumption at a time initially |
| Unreviewed automation | Suggested order flows straight to supplier | Sandbox and live controls are coupled | Real operational harm | Separate credentials and require approval |

## 11. Assumptions and limits

| Assumption | Why used | What breaks it | How to test or respond |
|---|---|---|---|
| One SKU and location | Keeps state and attribution transparent | Shared inventory, substitution, transfers | Reconcile by item and location; extend state when needed |
| Daily periodic review | Matches the lab clock | Intraday triggers or weekly ordering | Compare event timestamps to review cadence |
| Demand occurs after receipts | Defines an unambiguous daily order | Receiving completes after customer demand | Check warehouse availability timestamps |
| Demand is independent uniform 2–10 | Simple synthetic generator | Seasonality, bursts, zero demand, trend | Compare empirical distribution and serial dependence |
| Demand is exogenous | Makes policy comparison tractable | Customers substitute or abandon in response to availability | Analyze customer behavior and scenario bounds |
| Lost sales | Makes unserved demand leave the system | Customers wait for backorders | Inspect order statuses and commitments |
| Fixed three-day lead time | Isolates inventory feedback | Supplier delays, calendars, partial receipts | Measure placement-to-availability distributions |
| Unlimited supplier | Avoids supplier constraints | Allocation, minimums, capacity, rejection | Compare orders with supplier agreements |
| Unlimited warehouse | Avoids an additional feasibility model | Finite space, handling capacity, temperature limits | Compare simulated peak to capacity |
| No expiry | Appropriate only for nonperishable illustration | Shelf life and obsolescence | Include age cohorts and disposal |
| One fixed order per review | Defines the rule precisely | System can place multiple lots or order up to target | Compare decision logic with actual configuration |
| Constant cost coefficients | Keeps objective interpretable | Volume tiers, lost customers, seasonal rates | Obtain cost-owner ranges and stress them |
| All demand has equal value | Allows one penalty per unit | Different margins or customer priorities | Segment service and cost where justified |
| Initial stock 30, no pipeline | Provides a common start | Existing orders or different current stock | Reconstruct live state; vary starts |
| 180-day horizon | Makes runs short | Long cycles, seasonal and terminal effects | Extend horizon and inspect ending pipeline |
| No terminal valuation | Keeps arithmetic simple | Residual stock or purchase obligations material | Use a coherent full-horizon economic model |
| Mean fill threshold | Simple exercise selection | Individual customer guarantees or tail limits | Specify a percentile or segment constraint |
| Independent replications | Enables scenario-level summaries | Shared latent shocks across paths | Define scenarios and their dependence explicitly |

These assumptions are not evidence about an actual business.
A scenario is useful when its simplifications are appropriate for the decision or when their effects are bounded.
If a material assumption cannot be checked, label the result as conditional and identify the missing evidence.

## 12. Exercises

### Exercise 1 — Trace a policy

**Time:** 15 minutes.

Start with 5 units and no pipeline.
Use `R = 2`, `Q = 4`, and a two-day lead time.
Daily demand is 3, 2, 4, and 1 units.
Use the lab's receipt-before-demand and one-order-per-review conventions.

**Deliverable:** a four-row daily trace showing receipts, served units, lost units, closing stock, pre-order position, and order arrivals.

**Acceptance criteria:** each day's stock and demand balances reconcile; all arrival dates follow the specified lead time.
See answer 1.

### Exercise 2 — Distinguish service measures

**Time:** 10 minutes.

Across four days, demand is 10, 2, 8, and 5 units.
Lost units are 1, 0, 3, and 0 respectively.

**Deliverable:** calculate unit fill rate and stockout-free day rate.

**Acceptance criteria:** state both denominators and explain why the rates differ.
See answer 2.

### Exercise 3 — Reproduce the sweep

**Time:** 20 minutes.

Run section 7 in a local sandbox, record the runtime version and output, then rerun unchanged.
Do not submit real demand or connect a production system.

**Deliverable:** a run record with seed schedule, grid, cost units, output, and comparison of the two runs.

**Acceptance criteria:** the fixture passes, nine candidates appear, and the same environment reproduces the same output.
Explain why the seed alone would be insufficient if the code or runtime changed.
See answer 3.

### Exercise 4 — Reconcile a cost

**Time:** 10 minutes.

A path has 70 unit-days of closing inventory, 4 lost units, and 3 placed orders.
Use $0.05 per unit-day, $8 per lost unit, and $3 per order.

**Deliverable:** the three components and their sum in US dollars.

**Acceptance criteria:** show multiplication and identify a cost omitted by the model.
See answer 4.

### Exercise 5 — Use paired comparisons

**Time:** 15 minutes.

On the same three paths, policy A costs $100, $130, and $90.
Policy B costs $110, $100, and $100.

**Deliverable:** compute each `A - B` difference, the mean difference, and the number of paths favoring A.

**Acceptance criteria:** preserve path pairing and distinguish the mean result from a guarantee.
See answer 5.

### Exercise 6 — Critique the choice

**Time:** 20 minutes.

Suppose policy X has lower modeled cost but 94% mean fill.
Policy Y has higher modeled cost but 96% mean fill.
The exercise target is at least 95% mean fill.
A stakeholder also requires every individual path to achieve at least 90% fill, but that statistic was not recorded.

**Deliverable:** identify the candidate eligible under the first condition and describe what is needed for the second.

**Acceptance criteria:** do not claim that either rule meets a per-path target from a mean alone.
Name at least two missing business constraints.
See answer 6.

### Exercise 7 — Design a release gate

**Time:** 25 minutes.

A team proposes to replace synthetic paths with a year of order and receipt data and then update live replenishment parameters.

**Deliverable:** a staged validation and release plan with a data owner, business approver, monitoring, and rollback.

**Acceptance criteria:** include demand censoring, event-timing reconciliation, out-of-sample evaluation, shadow mode, and a named rollback trigger.
Do not represent synthetic output as realized savings.
See the rubric in answer 7.

### Exercise 8 — Diagnose an error

**Time:** 15 minutes.

An analyst reports nearly perfect fill using a model in which the stock variable includes both on-hand and all open orders.
Actual customer orders cannot be filled from goods in transit.

**Deliverable:** explain the mechanism, give a minimal counterexample, and propose a test.

**Acceptance criteria:** distinguish physical stock from inventory position and specify when the order actually arrives.
See answer 8.

## 13. Self-check answers

### Answer 1

| Day | Receipt | Demand | Served | Lost | Close | Pre-order position | Order and arrival |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 3 | 3 | 0 | 2 | 2 | 4 units on day 3 |
| 2 | 0 | 2 | 2 | 0 | 0 | 4 | None |
| 3 | 4 | 4 | 4 | 0 | 0 | 0 | 4 units on day 5 |
| 4 | 0 | 1 | 0 | 1 | 0 | 4 | None |

On day 2 the first order is still open, so position is 4 units, not zero.
On day 3 the order is received before demand and is removed from pipeline.
On day 4 the second order has not yet arrived.
Total demand is 10 units, served is 9 units, and lost is 1 unit.
The physical balance is `5 + 4 - 9 = 0` units at the end of day 4.

### Answer 2

Total demand is `10 + 2 + 8 + 5 = 25` units.
Total lost is `1 + 0 + 3 + 0 = 4` units.
Served units are `25 - 4 = 21`.
Unit fill rate is `21 / 25 = 84%`.
Two of four days have no loss, so the stockout-free day rate is `2 / 4 = 50%`.
One metric weights days by demanded units; the other gives each day equal weight.

### Answer 3

The fixture line and nine rows are structural expectations, not invented observations from your machine.
Unchanged runs in the same environment should produce the same output.
A complete record includes the script version, Python version, base seed 2026, 100 replications, and 180 days.
It also includes the generator, initial state, lead time, grid, and costs.
Changing code, draw order, or runtime can change paths even if the integer seed stays the same.

### Answer 4

Holding is `70 × $0.05 = $3.50`.
Shortage is `4 × $8 = $32.00`.
Ordering is `3 × $3 = $9.00`.
The modeled total is `$3.50 + $32.00 + $9.00 = $44.50`.
Purchase cost is one omitted cost; expiry and capacity costs are other examples.

### Answer 5

The paired differences are `-$10`, `$30`, and `-$10`.
Their mean is `($-10 + $30 - $10) / 3 = $3.33` approximately.
A has lower cost on two of three paths, but its mean cost is higher by about $3.33.
The third path's $30 difference outweighs the two $10 advantages.
Three paths are not enough to establish a production ranking.

### Answer 6

Y meets the stated 95% mean-fill condition and X does not.
The path-level fill values are needed to determine whether Y meets the additional 90% minimum on every path.
The same is true for X if the first condition is reconsidered.
Missing constraints can include storage capacity, supplier minimums, customer-specific service, lead-time variability, and cost validity.
If Y fails the second condition, the correct conclusion may be that neither tested candidate is eligible.

### Answer 7 — Rubric

A strong plan names owners for demand, inventory, supplier events, cost inputs, and the decision itself.
It checks sales censoring and separates requested demand from shipments.
It reconciles order placement and availability timestamps to the modeled event order.
It preserves an independent time period or scenario set for evaluating the selected candidate.
It compares recommendations in shadow mode before any live action.
It defines service and operational monitoring, an accountable approver, and a specific stop or rollback threshold.
It records what evidence would block release and does not promise savings from synthetic outcomes.
A weak plan jumps from a passing Python assertion to automatic live orders.

### Answer 8

Open orders contribute to position but are not physical stock available to serve demand.
For a counterexample, start with zero units on hand and an open order of 4 units arriving tomorrow.
Today's demand of 1 unit is entirely lost under the stated lost-sales model.
A test must assert one lost unit today, zero on-hand afterward, and a four-unit receipt tomorrow before demand.
The test should separately assert that the open order contributes to position before receipt.

## 14. Interview questions

1. **How do you distinguish on-hand stock from inventory position?**
   A strong answer includes open orders in position but not in immediately available stock.
   A weak answer says they are the same number.
2. **When does an order placed at the end of day 1 arrive with a two-day lead time?**
   A strong answer states the clock convention and answers morning of day 3 here.
   A weak answer gives a date without specifying receipt timing.
3. **Why compare candidates on the same demand paths?**
   A strong answer describes paired differences and the conditions under which noise may fall.
   A weak answer claims shared randomness proves the model is realistic.
4. **How is fill rate different from stockout-free day rate?**
   A strong answer gives the unit and day denominators.
   A weak answer treats a day with one lost unit as equivalent to a unit-level loss fraction.
5. **What changes when shortages are backordered?**
   A strong answer introduces an outstanding obligation and revises position and service accounting.
   A weak answer merely renames lost units.
6. **Why might observed sales understate demand?**
   A strong answer explains censoring during unavailable inventory and potential remedies.
   A weak answer assumes no sale always means no request.
7. **What is the role of a seed?**
   A strong answer records code, version, generator, configuration, and data along with the seed.
   A weak answer promises cross-version reproducibility from one integer alone.
8. **What does the best row in a finite grid establish?**
   A strong answer limits the result to tested candidates, scenarios, and the chosen score.
   A weak answer calls it a global optimum or a realized saving.
9. **How can horizon boundaries bias a comparison?**
   A strong answer discusses initial stock, pipeline, terminal value, and measurement horizon.
   A weak answer ignores orders arriving after the final day.
10. **How would you take this lab toward deployment?**
    A strong answer covers data reconciliation, independent validation, shadowing, approval, monitoring, and rollback.
    A weak answer connects the script directly to the purchasing system.

## 15. Key terms

- [Backlog](#4-definitions-and-vocabulary): an unfilled obligation carried forward; absent from this lab.
- [Common random numbers](#5-the-mechanics-step-by-step): shared scenario paths for candidate comparisons.
- [Cost-only selection](#7-code-query-or-calculation): lowest modeled cost among tested candidates without the service filter.
- [Demand censoring](#8-data-standards-and-regulatory-context): missing requests when availability constrains observed sales.
- [Demand path](#4-definitions-and-vocabulary): chronological sequence of requests in one replication.
- [Event order](#5-the-mechanics-step-by-step): receipt, demand, and review ordering within a day.
- [Fill rate](#4-definitions-and-vocabulary): immediately served units divided by requested units.
- [Fixed-quantity rule](#4-definitions-and-vocabulary): one specified quantity when the position threshold triggers.
- [Grid](#5-the-mechanics-step-by-step): finite set of parameter pairs under comparison.
- [Holding cost](#6-worked-example): charge on closing unit-days in this example.
- [Horizon](#11-assumptions-and-limits): finite period over which results are measured.
- [Initial state](#5-the-mechanics-step-by-step): opening stock and outstanding orders before the first event.
- [Inventory position](#3-the-core-mental-model): closing on-hand plus outstanding units in this model.
- [Lead time](#4-definitions-and-vocabulary): placement-to-availability delay in days.
- [Lost sales](#6-worked-example): requested units that leave unfilled and are not backordered.
- [Mean fill](#7-code-query-or-calculation): arithmetic average of replication-level fill rates.
- [On-hand](#4-definitions-and-vocabulary): physically available stock, excluding goods in transit.
- [Order cost](#6-worked-example): fixed charge for placing an order.
- [Paired difference](#5-the-mechanics-step-by-step): difference in outcomes for two rules on the same path.
- [Pipeline](#3-the-core-mental-model): outstanding orders awaiting receipt.
- [Policy](#5-the-mechanics-step-by-step): decision rule with a reorder point and fixed quantity.
- [Receipt](#6-worked-example): inventory becoming available on a scheduled day.
- [Reorder point](#4-definitions-and-vocabulary): inclusive position threshold triggering an order.
- [Replication](#4-definitions-and-vocabulary): one generated demand path and its outcomes.
- [Seed](#8-data-standards-and-regulatory-context): pseudorandom generator initialization value.
- [Service constraint](#5-the-mechanics-step-by-step): condition a candidate must meet before selection.
- [Shortage penalty](#4-definitions-and-vocabulary): modeled cost per lost unit.
- [Stockout-free day rate](#12-exercises): fraction of days without lost units.
- [Terminal effect](#11-assumptions-and-limits): influence of residual inventory and pipeline at the horizon.
- [Validation](#9-industry-practice-and-case-evidence): evidence that the model is fit for its intended decision.
- [Verification](#7-code-query-or-calculation): checks that the code implements the specified behavior.

## 16. Sources

1. Edward A. Silver, David F. Pyke, and Douglas J. Thomas, *Inventory and Production Management in Supply Chains*, fourth edition, CRC Press, 2017. [Publisher page](https://www.routledge.com/Inventory-and-Production-Management-in-Supply-Chains/Silver-Pyke-Thomas/p/book/9781482210303). Inventory terminology and policy context; not evidence for this synthetic scenario.
2. Paul Zipkin, “On the Structure of Lost-Sales Inventory Models,” *Operations Research* 56(4), 2008, pp. 937–944. [DOI: 10.1287/opre.1070.0471](https://doi.org/10.1287/opre.1070.0471). Research context for lost-sales inventory models; the lab is not a reproduction of this paper.
3. Python Software Foundation, [`random` — Generate pseudo-random numbers](https://docs.python.org/3/library/random.html). Generator behavior and reproducibility notes; consult the documentation matching the runtime used.
4. Python Software Foundation, [`statistics` — Mathematical statistics functions](https://docs.python.org/3/library/statistics.html). Definition and use of the arithmetic mean in the executable report.
5. GS1, [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin). Trade-item identification context, not a prescription for inventory parameters.
6. GS1, [Electronic Data Interchange](https://www.gs1.org/standards/edi). Business-message standards context; local trading-partner agreements determine implementation details.
7. Walmart Inc., [Annual reports and proxies](https://stock.walmart.com/financials/annual-reports-and-proxies/default.aspx). Public company reporting context; no result in this lab is attributed to Walmart.

## 17. Links

- [Module 30 index](./_index.md).
- [Module 30 survey and lab overview](../../../course/30_simulation_optimization_production_labs.md).
- [Environment and repository layout](./01_environment-repo.md).
- [Demand forecast pipeline](./02_demand-forecast-pipeline.md).
- [Network mixed-integer linear programming lab](./04_network-milp.md).
- [Inventory and service foundations](../../01_foundations/01_supply_chain_foundations/05_inventory-and-service.md).

Sibling topic availability follows the module index; a planned topic may not yet be published.
