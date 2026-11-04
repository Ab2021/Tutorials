# Lot Sizing: Four Methods Compared

> Module 15 — Manufacturing and Production Planning  
> Topic 07 of 11  
> Estimated study time: 4 hours  
> Prerequisites: [master data](02_manufacturing-master-data.md), [MRP explosion](04_mrp-explosion.md), [capacity requirements](05_capacity-requirements.md)  
> Version 2026.5

## 1. Why this topic exists

A planner often knows that a component is required in several future weeks.
The next question is how many units to make on each production order.
Making exactly what is needed minimizes planned carryover but may consume many setups.
Combining requirements reduces the number of setups but makes some units early.
Those early units consume cash and storage, and they can crowd the bottleneck.

A lot-sizing rule converts dated requirements into proposed order quantities.
It does not decide the customer promise, validate every component, or sequence the plant.
A quantity that is attractive in a spreadsheet can overload the machine in the week when it is proposed.
A rule that reduces changeovers can also force production before material or quality approval is available.
Conversely, a rule that appears to minimize inventory can produce more setups than the crew can execute.

The production planner owns a recommendation, not an automatic right to release an order.
Operations supplies setup times, run rates, and calendar constraints.
Finance helps identify incremental setup and holding costs.
Quality and engineering identify shelf-life, traceability, and process limits.
Customer service explains which due dates are commitments and which are changeable forecasts.
The master scheduler decides whether a revised production plan is authorized.

Consider a fictional plant that machines a pump housing on a shared resource.
Orders for the housing compete with other products for that machine.
The plant has six weekly net requirements and a choice of batching rules.
The comparison in this lesson uses the same demand, costs, and horizon for all methods.
It then tests the proposed quantities against a simplified weekly capacity limit.
This separates the economic lot-sizing question from the physical feasibility question.

A forecast is an estimate of future demand.
A plan is an intended production and inventory trajectory.
A target is a desired performance level, such as inventory days.
A commitment is an authorized obligation, such as an accepted customer due date.
The numeric case treats its six requirements as fixed inputs for calculation, not as proof of six customer commitments.
A planner must preserve the source and status of each requirement in the real system.

The cost comparison uses United States dollars (USD), good units, and weekly periods.
The fictional values are instructional assumptions, not a benchmark for machining plants.
The calculations do not claim that any one method is universally best.
They show how to choose a rule, expose its assumptions, and escalate its consequences.

## 2. Learning outcomes

After this lesson, you can:

1. **Explain** the setup-versus-inventory trade-off and distinguish a lot-sizing plan from a finite schedule.
   Exercise 1 checks the information flow.
2. **Calculate** lot-for-lot quantities and their setup and holding costs.
   Exercise 2 checks a complete period ledger.
3. **Compute** an economic order quantity (EOQ) from consistent annual inputs and apply it to dated demand.
   Exercise 3 checks rounding, receipts, and residual inventory.
4. **Construct** a period order quantity (POQ) plan with an explicitly stated period-rounding convention.
   Exercise 4 checks coverage and cost.
5. **Solve** a small Wagner-Whitin dynamic lot-sizing problem and interpret its optimality conditions.
   Exercise 5 checks the recurrence and backtracking.
6. **Compare** lot plans against a capacity calendar and recommend an authorized response.
   Exercise 6 checks weekly resource hours and feasibility.
7. **Critique** cost, demand, and horizon assumptions and describe data controls.
   Exercise 7 checks the assumption ledger.
8. **Design** a repeatable review and decision process with measurable outcomes.
   Exercise 8 uses a decision-rights rubric.

Each calculation must identify its unit and period.
A correct economic formula is not evidence that a proposed order is executable.
A feasible order is not necessarily the least costly option.

## 3. The core mental model

The causal chain is:

```text
Demand forecast + accepted orders + inventory and scheduled receipts
                              |
                              v
              Net requirements by item and week
                              |
             +----------------+----------------+
             |                |                |
       Setup economics   Holding economics   Constraints
             |                |                |
             +----------------+----------------+
                              |
                              v
                Lot-sizing rule or optimization
                              |
                              v
            Planned receipt quantities and due weeks
                              |
               Offset lead time and explode BOM
                              |
                              v
          Check material, capacity, shelf life and policy
                              |
                  +-----------+-----------+
                  |                       |
               Feasible               Conflict
                  |                       |
           Approve and release    Replan or escalate
                  |                       |
                  +-----------+-----------+
                              |
                              v
            Execute, measure and update assumptions
```

A material requirements planning (MRP) run starts with gross demand and scheduled supply.
Netting accounts for usable inventory, existing receipts, and the policy for safety stock.
The lot-sizing rule acts on the resulting time-phased requirement.
Where gross demand is used directly, the apparent optimal lot may duplicate existing supply.
In a multilevel bill of materials (BOM), a parent lot can create lumpy component demand.
The component planner should therefore use the relevant netted time series, not a smooth annual average alone.

A setup is the activity required to prepare a resource for a lot.
One setup may consume time, labor, material, and an approval or cleaning cost.
Combining orders can reduce the number of setups.
However, an earlier receipt creates inventory for future periods.
Holding cost accumulates according to the number of units and the time carried.
The best trade-off depends on both the magnitude and the timing of requirements.

Lot-for-lot orders each positive net requirement separately.
EOQ uses a constant economic quantity derived from a steady-demand approximation.
POQ converts that quantity into an interval and orders the requirements within each interval.
Wagner-Whitin optimizes the grouping of known requirements over a finite horizon under a restrictive model.
Each method produces a different set of order dates and quantities.
It is wrong to compare only the total number of units, because timing changes both cost and capacity load.

A weekly lot-sizing model is not a minute-by-minute schedule.
It may assume that production is available early enough to cover all demand in a bucket.
A capacity check can reject a bucket with more required hours than available hours.
Passing that aggregate check does not prove that every due-time, tool, operator, and sequence constraint works.
A finite scheduler must check those details before the plant treats the plan as executable.

A setup reduction changes the economics of all four rules.
If a changeover takes less time and money, more frequent production can become attractive.
That is a reason to improve a process, not permission to assume a saving before it is validated.
Similarly, expensive bottleneck time may justify a different decision from a simple accounting setup cost.
The decision maker must identify which hours and costs are incremental and which are already committed.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Demand | Units required in a stated period | Treating forecast as a customer commitment |
| Gross requirement | Total dated need before netting, units | Adding demand twice after a BOM explosion |
| Net requirement | Uncovered dated need after the stated netting rules, units | Ignoring available inventory or a scheduled receipt |
| Lot | Quantity assigned to one order or production run, units | Assuming every transaction is a separate physical setup |
| Planned receipt | Quantity planned to become usable in a period, units | Treating a proposal as an already completed order |
| Planned release | Order start date after lead-time offset, dated units | Releasing on the receipt date despite nonzero lead time |
| Setup cost | Incremental fixed cost per production lot, USD/setup | Charging all fixed overhead as avoidable cash expense |
| Setup time | Resource time needed per lot, hours/setup | Assuming money and hours are the same constraint |
| Holding rate | Cost to carry one unit for one period, USD/unit/week here | Combining an annual percentage with weekly demand without conversion |
| Carrying charge | Annual proportion of inventory value, 1/year | Adding overlapping insurance and overhead twice |
| Cycle stock | Inventory caused by replenishing in batches, units | Calling it protection against all forecast error |
| Safety stock | Policy buffer for uncertainty, units | Treating a fixed buffer as ordinary scheduled demand without explanation |
| Ending inventory | Available units after a period's transactions, units | Using beginning inventory when the model charges ending inventory |
| Shortage | Unmet requirement at its due time, units | Concealing a late shipment in negative projected stock |
| Backorder | Unfilled demand carried to a later period, units | Assuming a customer accepts lateness without permission |
| Lost sale | Demand not recovered after a stockout, units | Valuing it as a simple backorder without evidence |
| EOQ | Constant quantity minimizing a stylized annual setup-plus-holding expression, units/order | Claiming its formula optimizes arbitrary time-varying demand |
| POQ | Fixed number of demand periods grouped per order, weeks/order in this lesson | Treating it as a fixed number of units |
| Wagner-Whitin | Dynamic programming solution for a deterministic, uncapacitated finite-horizon lot-sizing model | Calling its cost globally optimal after adding unmodeled capacity |
| Horizon | Periods included in the calculation, weeks | Treating the last modeled week as the end of the business |
| Terminal inventory | Stock remaining at the end of the horizon, units | Ignoring its value or future disposal and demand |
| Demand bucket | Time interval over which requirements are aggregated, one week here | Assuming all intraweek demand occurs after production |
| Capacity | Available resource time over a stated interval, machine-hours/week here | Using theoretical hours after also deducting the same loss twice |
| Run rate | Good output per resource hour, units/hour here | Applying an average rate to an unapproved product or routing |
| Setup family | Products sharing a changeover configuration | Assuming all changes within or between families take equal time |
| Bottleneck | Resource limiting feasible system throughput under the current mix | Calling every highly utilized machine a system constraint |
| Shadow price | Marginal objective value of one more unit of a binding resource in a specified model | Treating it as a universal labor or machine hourly rate |
| Fixed order quantity | A constant order multiple selected by a policy, units | Assuming a final order can be truncated without declaring a different rule |
| Coverage | Future periods or demand supplied by a receipt, weeks or units | Counting zero-demand periods inconsistently |
| Time fence | Boundary governing how freely near-term orders can be changed | Assuming a calculated proposal can override released orders |
| Lot multiple | Required increment for an order, units | Rounding each demand before checking its commercial or technical basis |
| Minimum order quantity | Smallest permissible lot, units/order | Treating a supplier's minimum as the same thing as an EOQ |
| Opportunity cost | Value of the best forgone use of a constrained resource or funds | Adding an unmeasured shadow cost to actual overtime and counting it twice |
| Schedule nervousness | Frequent changes to dates or quantities as the plan is regenerated | Measuring only stock and ignoring release churn |
| Incremental cost | Cost that changes between the alternatives compared, USD | Comparing a full absorption cost against a cash-only cost |
| Service constraint | A rule requiring demand by a specified date or fill rate | Assuming cost minimization alone establishes acceptable service |

## 5. The mechanics, step by step

### 5.1 Define the decision and the boundary

1. Identify the item, plant, unit of measure, and decision owner.
   A lot policy for finished housings is not automatically appropriate for a purchased seal.
   Specify whether the decision is a rule parameter or a one-time order proposal.
2. Set the planning horizon and the time bucket.
   A week may be sufficient for an initial tactical comparison.
   A daily or shift-level check may be necessary before release.
3. Record the planning snapshot and the latest permitted change time.
   Near-term released lots may be fixed even when the model would prefer a different pattern.
   Treat fixed orders as constraints, not as suggestions to cancel without authorization.
4. Define the service rule.
   The comparison below requires each week's demand to be met in that week.
   It forbids backorders and lost sales in the model.
5. Define what counts as a lot and a setup.
   If two orders are combined into one physical run, charging two independent setups can distort the comparison.
   If one order needs multiple independent setups, one setup per order can understate load.
6. Specify the objective and its boundary.
   The worked example minimizes modeled setup plus inventory holding costs.
   It does not include revenue, transport, scrap, or the future value of the final stock.
7. Specify the baseline and approvals.
   A different policy is a proposal until planning, operations, and relevant control owners approve it.
   Document which costs are actually avoidable when evaluating a change.

**Input:** item, horizon, service rule, decision rights, and fixed orders.
**Output:** an agreed scenario boundary and acceptance criteria.
**Branch:** if customers permit lateness, use a model that explicitly prices or constrains lateness; do not silently change this model.

### 5.2 Build a clean net-requirement series

1. Take the approved demand snapshot, preserving order and forecast identifiers.
2. Net usable opening inventory according to the documented allocation rule.
3. Include eligible scheduled receipts once, using their available dates and quantities.
4. Respect reservations and quality holds rather than counting physically present but unusable stock.
5. Apply the stated safety-stock policy consistently.
   A minimum projected balance is different from a demand shipment.
6. Check that a requirement at a parent level has not also been counted independently at a component level.
7. Reconcile the total requirement and each period's input with the MRP ledger.
8. Flag negative quantities, missing periods, and obsolete item versions for investigation.
9. Preserve the units of measure and conversions used in the ledger.
10. Save the input snapshot so another analyst can reproduce the run.

The six-period case starts from already-netted requirements and zero opening stock.
It contains no scheduled receipts and no safety stock.
Those simplifications are not default settings for a real planning system.
If a released order is already in process, its remaining output and remaining load belong in the baseline.

**Input:** dated gross demand, stock, reservations, scheduled supply, and policy.
**Output:** a validated series of net requirements, in units per week.
**Branch:** if the demand ledger changes materially before approval, regenerate all alternatives from the same new snapshot.

### 5.3 Estimate setup and holding economics

1. Identify setup work that changes when a lot is added or removed.
   Examples may include additional consumables, paid overtime, external cleaning, and incremental inspection.
2. Separate a physical setup duration in machine-hours from its modeled monetary cost.
   The plant can face a capacity shortage even if no overtime is actually paid.
3. Decide how to value the opportunity cost of a constrained resource.
   Document the alternative work displaced and avoid counting the same lost contribution twice.
4. Estimate holding cost from inventory value and a carrying charge, or use directly observed incremental cost.
   Capital, storage, handling, insurance, obsolescence, and expiry may be relevant.
5. Remove overlapping components.
   A finance-provided carrying rate may already include storage and insurance.
6. Convert all rates to the chosen bucket.
   For a weekly model with 52 weeks/year, annual holding USD/unit/year divided by 52 gives USD/unit/week under a simple linear convention.
7. State whether costs are nominal or discounted and whether inflation is relevant.
   The numerical example uses undiscounted constant values over six weeks.
8. Use sensitivity analysis for estimates that are uncertain or disputed.
   A single accounting number should not be presented as precise economic truth.

**Input:** finance assumptions, setup observations, resource alternatives, and inventory value.
**Output:** a consistent setup cost and holding rate with a documented owner.
**Branch:** if rates vary by period, a dynamic model can use dated rates; the constant-rate EOQ formula no longer represents that detail.

### 5.4 Calculate lot-for-lot

For lot-for-lot (LFL), make a receipt equal to each positive net requirement.
Under zero opening stock, no existing receipts, and no policy buffer:

```text
receipt[t] = demand[t]
ending_inventory[t] = 0 units
setup_cost = number_of_positive_receipts * setup_cost_per_lot
holding_cost = 0 USD
```

The zero-inventory result depends on these assumptions.
A minimum lot, transit delay, safety stock, or receipt timing can create inventory even with a nominal LFL setting.
A week with zero demand produces no lot in this simple formulation.
Production still needs to begin before the requirement if manufacturing lead time is positive.

LFL is useful as a transparent baseline.
It reveals the cost and resource impact of frequent replenishment.
It can be attractive when setups are cheap, shelf life is short, or demand is highly uncertain.
It can be impractical where the sequence of setups occupies the resource or a minimum batch is mandatory.
A lower inventory balance is not sufficient evidence that it improves the plant's total outcome.

**Input:** net requirements and per-lot setup and holding assumptions.
**Output:** one lot for each positive requirement and its dated resource load.
**Branch:** if the setup burden exceeds capacity, compare aggregation or a capacity action rather than quietly dropping demand.

### 5.5 Calculate and apply EOQ

The classical EOQ expression is:

```text
Q* = sqrt(2 * D * K / H)
```

Here `D` is demand in units/year, `K` is USD/setup, and `H` is USD/unit/year.
The result is in units/order.
The underlying approximation uses stable demand, constant setup cost, immediate replenishment, and average cycle stock of `Q/2`.
Its annual variable cost expression is:

```text
annual_setup_cost = (D / Q) * K
annual_holding_cost = (Q / 2) * H
```

The square-root formula balances these two terms under the stylized assumptions.
It does not directly optimize the dated six-week sequence in the example.
If demand is uneven, its fixed quantity may create surplus or multiple closely spaced orders.
If a production lot accumulates gradually while demand is consumed, an economic production quantity model may be more appropriate than instantaneous receipt EOQ.
If a supplier offers a price break, minimum, or fixed multiple, explicitly evaluate those terms rather than claiming the unconstrained EOQ is feasible.

For the finite-horizon comparison, this lesson applies a declared fixed-order-quantity policy.
When projected stock cannot cover a week's requirement, receive as many full rounded EOQ lots as needed in that week.
A final partial lot is not allowed under this convention.
Any remaining units are visible as terminal stock.
A different business convention, such as trimming the final order, is a different policy and should be evaluated separately.

**Input:** annual demand, annual holding rate, setup cost, and a fixed-quantity application rule.
**Output:** rounded quantity and a period-by-period stock ledger.
**Branch:** if a rounded quantity is not technically or commercially valid, use valid multiples and rerun the comparison.

### 5.6 Calculate and apply POQ

POQ uses EOQ as a guide to the average number of periods covered by one order.
Given an average of `d_bar` units/week:

```text
raw_coverage_weeks = EOQ_units / d_bar
coverage_weeks = max(1, floor(raw_coverage_weeks + 0.5))
```

This lesson rounds positive half values upward.
Some systems use floor, ceiling, a configured integer, or a different rounding convention.
Record the actual convention rather than treating one convention as universal.
The numerator and denominator must describe the same item and compatible units.

Start at the first uncovered positive requirement.
Order the sum of its demand and the demand in the next `coverage_weeks - 1` calendar buckets, capped at the horizon.
Start the next coverage interval at the next uncovered positive requirement.
Zero-demand buckets within a coverage interval still count as calendar buckets.
A zero-demand bucket before the next interval does not itself trigger a setup.
In the worked example all six requirements are positive.

POQ has a fixed interval but a variable quantity.
It responds to time-varying requirements more directly than a constant EOQ lot.
It does not search all possible interval boundaries.
A fixed two-week coverage can place an unnecessarily large lot in a capacity-constrained week.
It can also combine demands that would be cheaper to separate when holding costs or setup costs change.

**Input:** net requirements, EOQ, average weekly demand, and the rounding and coverage rule.
**Output:** demand-specific quantities over fixed-length coverage windows.
**Branch:** if capacity or shelf life disallows a window, test a revised plan under explicit constraints; do not call the unconstrained window feasible.

### 5.7 Solve the Wagner-Whitin model

Define periods `1` through `T` with known nonnegative demand `d[t]`.
Let `K[j]` be the cost of a setup in period `j`.
Let `h[k]` be the holding cost per unit carried from the end of period `k` to period `k+1`.
The example uses constant values `K` and `h`.
No backorders, capacity limits, minimum lots, or initial inventory are included.

For a lot produced in period `j` that serves requirements through period `t`, define:

```text
C(j,t) = K[j] + sum(d[i] * sum(h[k] for k=j,...,i-1)
                    for i=j,...,t)
```

The empty sum for `i = j` is zero.
With constant weekly holding rate, this simplifies to:

```text
C(j,t) = K + h * sum((i-j) * d[i] for i=j,...,t)
```

For positive requirements, let `F[t]` be the minimum cost to satisfy demand through period `t`.
Initialize `F[0] = 0 USD` and calculate:

```text
F[t] = min(F[j-1] + C(j,t) for j=1,...,t)
```

Record the minimizing start `j` as well as the cost.
Backtrack from `T`: the final lot begins at the recorded `j` and covers through `T`.
Then repeat at `j-1` until reaching period zero.
If candidates tie, either lot plan is cost-optimal under the model; specify the tie rule.
The code below retains the earliest candidate in its search order when costs tie.

The no-speculation structure permits this interval decomposition under the classical assumptions.
A lot covers consecutive demand periods; it does not skip an intervening positive requirement to leave that requirement for a later lot.
Nonnegative holding costs make carrying unwanted surplus beyond the horizon unnecessary in this model.
If an explicit terminal target is required, add that requirement to the model rather than ignoring it.
For all-zero demand, no setup is needed; implementations must not charge a setup for an empty lot.
The provided case and executable example have positive demand in every week.

The dynamic program is exact for the defined uncapacitated deterministic objective and horizon.
Its result is not necessarily best when demand is uncertain, setup times vary by sequence, or production hours are limited.
A capacity-constrained lot-sizing model must represent the resource by period and may have a different optimum.
Alternatively, use the uncapacitated result as a lower bound and screen practical alternatives.

**Input:** dated requirements and period-consistent setup and holding costs.
**Output:** an optimal set of lots and cost for the declared model.
**Branch:** if the result violates a binding constraint, reject it as a release plan and solve or select a feasible alternative.

### 5.8 Translate receipts into a feasible proposal

1. Offset each planned receipt by its manufacturing lead time.
   The result is a planned release, not necessarily the exact start of every operation.
2. Check that all required components are available by their operation dates.
   A larger lot can bring a component shortage forward.
3. Convert each lot into resource load using its routing.
   In the example, load equals setup hours plus quantity divided by good units per machine-hour.
4. Compare load with available hours for the same resource and period.
   Include other products and already released work in a real shared-resource analysis.
5. Check any production, labor, tooling, quality, storage, and shelf-life constraints.
6. Use a finite scheduler for sequencing and intraweek timing where required.
7. Document violations and possible remedies with cost, timing, and approver.
8. Recalculate inventory and economics after any moved or split lot.
9. Preserve original demand due dates unless an authorized party changes the commitment.
10. Release only after required approvals and system checks.

One large lot can reduce setups yet consume more hours in its start week.
A smaller lot can fit a weekly calendar while requiring more setup events over the horizon.
Moving work earlier can solve one overload but raises stock and can create an earlier material conflict.
Subcontracting can add capacity but may introduce qualification, transport, and purchase commitments.
Overtime is an option only if people, equipment, and authorization are actually available.

**Input:** candidate lots, routing, calendars, materials, constraints, and decision rights.
**Output:** an executable proposal or an explicitly unresolved exception.
**Branch:** if no candidate is feasible, redesign the plan or escalate the service and capacity trade-off.

### 5.9 Monitor and revise the policy

1. Record the chosen policy, parameter values, effective date, and exception authority.
2. Measure actual setup count and setup hours separately from accounting setup cost.
3. Compare actual and planned inventory by period, including obsolete and quarantined units.
4. Track customer service by due date, not just total production volume.
5. Compare planned versus realized lead time and release stability.
6. Identify whether a change came from demand, master data, capacity, or manual override.
7. Re-estimate costs and setup times after a verified process change.
8. Retain a stable baseline long enough to distinguish noise from a systematic problem.
9. Recompute the decision when an input crosses an agreed threshold.
10. Keep rejected alternatives and rationale in the decision record.

A rule can be appropriate for one product family and poor for another.
An infrequent high-value part with expiry risk has a different holding profile from a common low-value part.
A setup reduction can make a previously economical long run obsolete.
The policy is a controlled operating parameter, not a permanent attribute of an item.

## 6. Worked example

### 6.1 Scenario and assumptions

A fictional pump-housing plant has already converted demand and available stock into six weekly net requirements.
The series represents a planning snapshot; it does not by itself specify which requirements are firm commitments.
All quantities are good finished housings.
The same six weeks and the same demand apply to every candidate.

| Week | Net requirement, units |
|---:|---:|
| 1 | 60 |
| 2 | 90 |
| 3 | 40 |
| 4 | 110 |
| 5 | 50 |
| 6 | 70 |
| **Total** | **420** |

Use these assumptions:

| Input | Assumption | Interpretation |
|---|---:|---|
| Opening usable stock | 0 units | Requirements are already netted |
| Scheduled receipts | 0 units | No existing order is duplicated |
| Safety stock | 0 units | No minimum buffer in this illustrative comparison |
| Setup cost | 120 USD/lot | Constant, incremental modeled cost |
| Holding cost | 0.80 USD/unit/week | Charged on ending inventory for each week |
| Run rate | 10 good units/machine-hour | Constant across quantities |
| Setup time | 2 machine-hours/lot | Independent of sequence in this simplified example |
| Available machine time | 20 machine-hours/week | Reserved for this item; no other product load in this illustration |
| Receipt timing | Available to meet that week's requirement | Bucket convention, not a detailed shop sequence |
| Lead time | Zero weeks in the lot ledger | Release and receipt week coincide for calculation only |
| Shortages | Not allowed | All six weekly requirements must be met |
| Lot minimum or multiple | None except the EOQ policy itself | Fractional production is not needed in these data |
| Discounting | None | Six-week modeled costs are undiscounted |
| Terminal value | Not credited | EOQ surplus is reported separately; future cost/value is outside the horizon |

The 20-hour weekly allocation is a simplifying scenario, not the machine's entire plant calendar.
The assumption of availability within each week needs a finer schedule in practice.
If manufacturing lead time is nonzero, offset releases and check earlier capacity and components.
The rates are not a claim about a real pump manufacturer.

### 6.2 Lot-for-lot ledger

| Week | Receipt, units | Requirement, units | Ending stock, units | Setup cost, USD | Holding cost, USD | Load, machine-hours |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 60 | 0 | 120 | 0 | 2 + 60/10 = 8 |
| 2 | 90 | 90 | 0 | 120 | 0 | 2 + 90/10 = 11 |
| 3 | 40 | 40 | 0 | 120 | 0 | 2 + 40/10 = 6 |
| 4 | 110 | 110 | 0 | 120 | 0 | 2 + 110/10 = 13 |
| 5 | 50 | 50 | 0 | 120 | 0 | 2 + 50/10 = 7 |
| 6 | 70 | 70 | 0 | 120 | 0 | 2 + 70/10 = 9 |
| **Total** | **420** | **420** | **0 terminal** | **720** | **0** | **54** |

Six setups consume 12 machine-hours.
Running 420 units consumes 42 machine-hours.
The maximum weekly load is 13 machine-hours, below the illustrative 20-hour limit.
The modeled setup-plus-holding cost is 720 USD.
This conclusion does not include any intraweek sequencing or material checks.

### 6.3 EOQ calculation and ledger

Average demand in the sample is 420 / 6 = 70 units/week.
Annualizing that mean at 52 weeks/year gives 3,640 units/year.
Annual holding cost is 0.80 × 52 = 41.60 USD/unit/year.
The sample mean is an illustrative estimate, not evidence of stable annual demand.

```text
EOQ = sqrt(2 * 3,640 units/year * 120 USD/setup
           / 41.60 USD/unit/year)
    = 144.9138 units/order
rounded fixed lot = 145 units/order
```

Apply a full 145-unit lot when projected stock would otherwise be insufficient.
The final lot is not shortened, and all end-of-week stock is charged holding cost.

| Week | Receipt, units | Requirement, units | Ending stock, units | Setup cost, USD | Holding cost, USD | Load, machine-hours |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 145 | 60 | 85 | 120 | 68 | 16.5 |
| 2 | 145 | 90 | 140 | 120 | 112 | 16.5 |
| 3 | 0 | 40 | 100 | 0 | 80 | 0 |
| 4 | 145 | 110 | 135 | 120 | 108 | 16.5 |
| 5 | 0 | 50 | 85 | 0 | 68 | 0 |
| 6 | 0 | 70 | 15 | 0 | 12 | 0 |
| **Total** | **435** | **420** | **15 terminal** | **360** | **448** | **49.5** |

The modeled cost is 360 + 448 = 808 USD.
The 15 terminal units represent additional production, not free or discarded inventory.
The six-week cost does not credit their future use or charge any holding after week 6.
That boundary makes the cost comparison incomplete for a continuing business.
A decision maker should extend the horizon or specify a defensible terminal value before comparing final economics.
EOQ has the largest modeled cost in this snapshot; that does not disprove EOQ under its steady-demand assumptions.
Its 16.5-hour peak fits the simplified weekly allocation, subject to material and sequence checks.

### 6.4 POQ calculation and ledger

Divide the unrounded EOQ by average weekly demand:

```text
144.9138 units / (70 units/week) = 2.0702 weeks
rounded coverage = 2 weeks
```

The fixed two-week windows are weeks 1–2, 3–4, and 5–6.
Each receipt equals the requirements in its window.

| Week | Receipt, units | Requirement, units | Ending stock, units | Setup cost, USD | Holding cost, USD | Load, machine-hours |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 150 | 60 | 90 | 120 | 72 | 17 |
| 2 | 0 | 90 | 0 | 0 | 0 | 0 |
| 3 | 150 | 40 | 110 | 120 | 88 | 17 |
| 4 | 0 | 110 | 0 | 0 | 0 | 0 |
| 5 | 120 | 50 | 70 | 120 | 56 | 14 |
| 6 | 0 | 70 | 0 | 0 | 0 | 0 |
| **Total** | **420** | **420** | **0 terminal** | **360** | **216** | **48** |

The setup cost is 3 × 120 = 360 USD.
The holding cost is 0.80 × (90 + 110 + 70) = 216 USD.
The modeled cost is 576 USD.
The largest weekly load is 17 machine-hours, below the illustrative 20-hour allocation.
The plan still needs a detailed release and material check.

### 6.5 Wagner-Whitin calculation and ledger

Using `K = 120 USD/lot` and `h = 0.80 USD/unit/week`, calculate the optimal prefix cost.
`Start` is the first week covered by the final lot of each prefix.

| Prefix ending week | Best final-lot start | Minimum prefix cost, USD | One optimal set of lot coverage intervals |
|---:|---:|---:|---|
| 1 | 1 | 120 | 1–1 |
| 2 | 1 | 192 | 1–2 |
| 3 | 1 | 256 | 1–3 |
| 4 | 4 | 376 | 1–3; 4–4 |
| 5 | 4 | 416 | 1–3; 4–5 |
| 6 | 4 | 528 | 1–3; 4–6 |

For example, the first lot covering weeks 1–3 has a holding charge of:

```text
90 units * 1 week * 0.80 USD/unit/week = 72 USD
40 units * 2 weeks * 0.80 USD/unit/week = 64 USD
setup = 120 USD
prefix cost through week 3 = 120 + 72 + 64 = 256 USD
```

A second lot at week 4 covering weeks 4–6 costs:

```text
50 units * 1 week * 0.80 USD/unit/week = 40 USD
70 units * 2 weeks * 0.80 USD/unit/week = 112 USD
setup = 120 USD
second lot cost = 120 + 40 + 112 = 272 USD
full-horizon cost = 256 + 272 = 528 USD
```

The ledger is:

| Week | Receipt, units | Requirement, units | Ending stock, units | Setup cost, USD | Holding cost, USD | Load, machine-hours |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 190 | 60 | 130 | 120 | 104 | 21 |
| 2 | 0 | 90 | 40 | 0 | 32 | 0 |
| 3 | 0 | 40 | 0 | 0 | 0 | 0 |
| 4 | 230 | 110 | 120 | 120 | 96 | 25 |
| 5 | 0 | 50 | 70 | 0 | 56 | 0 |
| 6 | 0 | 70 | 0 | 0 | 0 | 0 |
| **Total** | **420** | **420** | **0 terminal** | **240** | **288** | **46** |

The 190-unit lot needs 21 machine-hours in week 1.
The 230-unit lot needs 25 machine-hours in week 4.
Both exceed the stated 20-machine-hour weekly allocation.
The result is the uncapacitated cost optimum, not a feasible weekly release plan.
It supplies a 528 USD lower bound for this same setup-plus-holding objective when the capacity restriction is added, provided all other model assumptions remain unchanged.

### 6.6 Decision comparison

| Method | Setups | Setup cost, USD | Holding cost, USD | Total modeled cost, USD | Peak load, machine-hours/week | Terminal stock, units | Weekly capacity screen |
|---|---:|---:|---:|---:|---:|---:|---|
| LFL | 6 | 720 | 0 | 720 | 13 | 0 | Pass |
| EOQ fixed 145 | 3 | 360 | 448 | 808 | 16.5 | 15 | Pass |
| POQ two weeks | 3 | 360 | 216 | 576 | 17 | 0 | Pass |
| Wagner-Whitin | 2 | 240 | 288 | 528 | 25 | 0 | Fail in weeks 1 and 4 |

Among these four candidates, POQ has the lowest modeled cost that passes the weekly screen.
Its 576 USD is 48 USD above the uncapacitated lower bound of 528 USD.
The gap does not prove POQ is optimal with capacity constraints.
Another feasible lot pattern could have a lower cost, and a full constrained optimization would be needed to prove the best one.

A defensible recommendation is to take the POQ pattern to detailed scheduling and component review.
Do not release it solely because its weekly load fits.
If the week 1 or week 3 machine time is not actually available, the planner must revise the option.
If management can obtain extra capacity, compare the incremental capacity cost and approval requirements with the 48 USD modeled gap.
It takes at least 1 extra machine-hour in week 1 and 5 extra machine-hours in week 4 to fit the uncapacitated lot sizes in those weekly buckets.
That comparison still excludes sequencing, material, terminal-value differences, and any costs outside the model.

### 6.7 Sensitivity: cheaper setups

Suppose a verified process improvement reduces the setup cost to 40 USD/lot while all other inputs stay unchanged.
This is a scenario, not an observed improvement at the fictional plant.
Hold the existing LFL and two-week POQ patterns fixed for an apples-to-apples comparison:

```text
LFL fixed-pattern cost = 6 * 40 + 0 = 240 USD
POQ fixed-pattern cost = 3 * 40 + 216 = 336 USD
```

LFL now costs 96 USD less than that unchanged POQ pattern.
This comparison does not re-optimize the POQ interval or the Wagner-Whitin plan.
It demonstrates why a setup improvement should trigger a policy review rather than merely making the old large lots cheaper.
If the setup improvement also changes setup time, update the capacity ledger separately.
The monetary reduction and the time reduction are related only to the extent confirmed by the process owner.

### 6.8 Interpretation of the horizon

EOQ ends with 15 extra units; the other displayed plans end with none.
The comparison includes the holding charge for those 15 units through the end of week 6.
It excludes their purchase or conversion cost because the modeled objective includes only setup and holding.
If those units satisfy known week 7 demand, they may have future value.
If they expire or the design changes, their value may be lower or negative after disposal costs.
A common extension is to add realistic later requirements and recompute every method.
Another is to include a carefully justified terminal value and consistent production costs.
Do not subtract the nominal unit price from the EOQ cost without also modeling unit production cost consistently for all candidates.

## 7. Code, query, or calculation

The following Python 3 program is self-contained and uses only the standard library.
It reproduces the four ledgers, costs, peak weekly loads, and Wagner-Whitin prefix costs above.
It assumes positive demand in every week, zero opening stock, no shortages, constant costs, and no lead-time offset.
EOQ is a fixed whole-unit quantity rounded to the nearest unit; POQ uses the explicitly described nearest-period convention.
The dynamic program is uncapacitated; capacity is evaluated only after the lots are generated.
The program does not search for the optimal capacity-constrained plan.
The setup and holding costs are floats, so formatted output is shown to two decimal places.

```python
from math import floor, sqrt

DEMAND = [60, 90, 40, 110, 50, 70]  # good units/week
SETUP_COST = 120.0                    # USD/lot
HOLD_COST = 0.80                      # USD/unit/week
SETUP_HOURS = 2.0                     # machine-hours/lot
RUN_RATE = 10.0                       # good units/machine-hour
CAPACITY = 20.0                       # machine-hours/week
WEEKS_PER_YEAR = 52


def lot_for_lot(demand):
    return [([d] if d > 0 else []) for d in demand]


def fixed_eoq(demand, quantity):
    # One list entry per week; each inner entry is a separate lot.
    lots = [[] for _ in demand]
    stock = 0
    for t, need in enumerate(demand):
        while stock < need:
            lots[t].append(quantity)
            stock += quantity
        stock -= need
    return lots


def period_order_quantity(demand, coverage):
    lots = [[] for _ in demand]
    t = 0
    while t < len(demand):
        if demand[t] == 0:
            t += 1
            continue
        end = min(len(demand), t + coverage)
        lots[t].append(sum(demand[t:end]))
        t = end
    return lots


def wagner_whitin(demand, setup, holding):
    # Case-specific precondition: all weekly demands are positive.
    n = len(demand)
    best = [0.0] + [float('inf')] * n
    start = [0] * (n + 1)
    for end in range(1, n + 1):
        for first in range(1, end + 1):
            carry = sum(
                (i - first) * demand[i - 1] * holding
                for i in range(first, end + 1)
            )
            candidate = best[first - 1] + setup + carry
            if candidate < best[end]:
                best[end] = candidate
                start[end] = first
    lots = [[] for _ in demand]
    end = n
    while end > 0:
        first = start[end]
        lots[first - 1].append(sum(demand[first - 1:end]))
        end = first - 1
    return lots, best


def ledger(demand, lots):
    stock = 0
    setup_total = 0.0
    holding_total = 0.0
    loads = []
    rows = []
    for t, (need, week_lots) in enumerate(zip(demand, lots), 1):
        receipt = sum(week_lots)
        stock += receipt - need
        if stock < 0:
            raise ValueError(f'shortage in week {t}')
        setup = len(week_lots) * SETUP_COST
        holding = stock * HOLD_COST
        load = len(week_lots) * SETUP_HOURS + receipt / RUN_RATE
        setup_total += setup
        holding_total += holding
        loads.append(load)
        rows.append((t, receipt, need, stock, setup, holding, load))
    return rows, setup_total, holding_total, max(loads, default=0.0)


mean_weekly = sum(DEMAND) / len(DEMAND)
annual_demand = mean_weekly * WEEKS_PER_YEAR
annual_holding = HOLD_COST * WEEKS_PER_YEAR
raw_eoq = sqrt(2 * annual_demand * SETUP_COST / annual_holding)
fixed_quantity = max(1, floor(raw_eoq + 0.5))
coverage = max(1, floor(raw_eoq / mean_weekly + 0.5))
ww_lots, prefix = wagner_whitin(DEMAND, SETUP_COST, HOLD_COST)

plans = [
    ('LFL', lot_for_lot(DEMAND)),
    ('EOQ', fixed_eoq(DEMAND, fixed_quantity)),
    ('POQ', period_order_quantity(DEMAND, coverage)),
    ('Wagner-Whitin', ww_lots),
]
print(f'EOQ={raw_eoq:.4f}; fixed lot={fixed_quantity}; POQ weeks={coverage}')
print('WW prefix:', ', '.join(f'{x:.2f}' for x in prefix[1:]))
for name, lots in plans:
    rows, setup, holding, peak = ledger(DEMAND, lots)
    print(f'\n{name}:')
    print('week receipt demand stock setup holding load')
    for week, receipt, need, stock, s, h, load in rows:
        print(f'{week:4} {receipt:7} {need:6} {stock:5} '
              f'{s:5.2f} {h:7.2f} {load:4.1f}')
    print(f'setup={setup:.2f}; holding={holding:.2f}; '
          f'total={setup + holding:.2f}; peak={peak:.1f}; '
          f'capacity_pass={peak <= CAPACITY}; terminal={rows[-1][3]}')
```

Expected summary lines, in USD except where labeled:

```text
EOQ=144.9138; fixed lot=145; POQ weeks=2
WW prefix: 120.00, 192.00, 256.00, 376.00, 416.00, 528.00
LFL: setup=720.00; holding=0.00; total=720.00; peak=13.0; capacity_pass=True; terminal=0
EOQ: setup=360.00; holding=448.00; total=808.00; peak=16.5; capacity_pass=True; terminal=15
POQ: setup=360.00; holding=216.00; total=576.00; peak=17.0; capacity_pass=True; terminal=0
Wagner-Whitin: setup=240.00; holding=288.00; total=528.00; peak=25.0; capacity_pass=False; terminal=0
```

The code prints a heading and individual ledger rows between the summary lines.
`peak` is in machine-hours/week; `terminal` is in units.
`capacity_pass` is an aggregate screen, not a finite schedule.
The recurrence implementation is deliberately transparent rather than optimized for a large production instance.
Its inner summation makes it slower than an implementation that precomputes interval costs.
For real use, validate inputs, use an auditable money representation, reconcile order status, and implement the organization's actual constraints.

## 8. Data, standards, and regulatory context

### 8.1 Minimum data contract

A reproducible lot-sizing run needs an item identifier and effective plant location.
It needs a dated demand series and the source status of each requirement.
It needs usable stock, reservations, quality holds, and existing planned or firm receipts.
It needs the relevant unit of measure and conversion rules.
It needs a production calendar and a planning-bucket convention.
It needs setup and holding parameters with their sources, units, and effective dates.
It needs routing data for each constrained resource and the valid product revision.
It needs lead times and the dates when components and capacity are available.
It needs policy constraints such as minimum lots, multiples, safety stock, and time fences.
It needs explicit authority to change a firm order or customer date.

A useful record for an individual demand row contains:

| Field | Purpose | Example or control |
|---|---|---|
| Item and revision | Identify what is needed | Approved housing revision |
| Site | Distinguish local stock and routing | Machining plant |
| Requirement date | Put demand in the correct bucket | Customer due date converted to plant need date |
| Quantity and unit | Support arithmetic | Good units, not kilograms or raw pieces |
| Source | Distinguish forecast, order, or dependent demand | Order identifier or MRP parent reference |
| Snapshot time | Reproduce the calculation | Common extraction timestamp |
| Status | Prevent unauthorized changes | Forecast, planned, firm, or released |
| Availability eligibility | Exclude held stock | Quality release and reservation rules |
| Netting allocation | Explain remaining requirement | Reference to matched stock and receipts |
| Owner | Resolve exceptions | Named planning role |

A useful record for a setup parameter contains:

| Field | Purpose | Risk if missing |
|---|---|---|
| Resource and product family | Select the correct process | Applying another machine's setup |
| From/to family | Capture sequence dependence if relevant | Treating a difficult clean as a constant setup |
| Observed setup time | Load the calendar | Understating occupied hours |
| Incremental cost basis | Compare financial alternatives | Treating absorbed overhead as cash saved |
| Effective date | Match the current process | Using an obsolete time after a validated improvement |
| Sample and variability | Understand uncertainty | Planning on a single unusually fast setup |
| Approval | Establish ownership | Uncontrolled parameter changes |

### 8.2 Traceability and manufacturing standards

GS1 identifies the Global Trade Item Number (GTIN) as a way to identify trade items.
GS1 Application Identifiers include identifiers for batch or lot and serial information in relevant barcodes.
The precise identifier used depends on the organization's trading and traceability requirements.
A planning lot is not automatically identical to a regulatory batch, supplier lot, or serialized production unit.
Maintain explicit relationships between a planned order, physical production batch, input lots, and shipped units where traceability is required.
See the [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications) for the current technical definitions.
The standard does not prescribe an EOQ, POQ, or capacity policy.

IEC 62264 addresses integration between enterprise systems and manufacturing operations management.
The standard provides models and terminology for information exchange; a licensed current edition is the authoritative source.
An enterprise planning order and a shop-floor production record can have different status and granularity.
An interface should preserve identifiers, quantities, units, timestamps, and revisions across those boundaries.
IEC 62264 does not prove that a chosen lot will fit the machine calendar.
See the [IEC 62264-1 publication page](https://webstore.iec.ch/en/publication/6675) and the relevant edition for scope.

### 8.3 Product and sector constraints

Requirements depend on jurisdiction, product, process, and authorization.
For example, United States finished pharmaceuticals are subject to applicable current good manufacturing practice regulations in 21 CFR Parts 210 and 211.
The electronic Code of Federal Regulations is the current source for the text and applicability.
Section 211.188 addresses batch production and control records for drug products within its scope.
It does not prescribe a lot-sizing optimization formula.
A pharmaceutical batch-size change may need quality, validation, and filing assessment under the applicable facts.
Do not assume a planner can resize a regulated batch merely because a mathematical rule recommends it.

Food traceability obligations likewise depend on product, covered activities, effective dates, and jurisdiction.
The United States Food and Drug Administration's Food Traceability Rule information describes the relevant scope and current implementation status.
Check the official current material before making a compliance decision.
This lesson does not assert that every pump-housing manufacturer is subject to those food or drug provisions.
Engineering drawings, customer contracts, and quality procedures can impose additional restrictions even outside a regulated sector.

### 8.4 Controls and reproducibility

The analyst should retain the input extract, code version, parameter values, and scenario identifier.
The report should identify the planning time zone and calendar.
It should show quantity and date changes from the previous approved plan.
It should distinguish a model-generated suggestion from a released shop order.
It should log who approved an override and why.
It should reconcile actual production and inventory against the model after execution.
Access to cost and customer order data should follow the organization's information-security policy.
The educational example contains no real supplier, customer, or personal data.

## 9. Industry practice and case evidence

### 9.1 What the public evidence supports

The original Wagner and Whitin paper, “Dynamic Version of the Economic Lot Size Model,” presents a deterministic dynamic approach to economic lot sizing.
Its contribution is a mathematical model and solution method, not evidence that an unconstrained optimum will fit any particular plant.
The paper is a primary academic source for the classical model.
Its publication predates modern enterprise planning software and does not certify the implementation of a vendor's product.

Harris's 1913 paper “How Many Parts to Make at Once” is a historical source for the economic lot-size idea.
It discusses the opposing effects of setup-related expense and inventory carrying expense.
The simplified EOQ expression is useful as a benchmark, but its assumptions must be checked before applying it to uneven weekly demand.
A historic derivation is not a case study proving savings at the fictional plant.

Toyota's official production-system material describes reducing waste and producing what is needed when it is needed.
Its description of just-in-time and flow is relevant to the operational motivation for smaller batches.
It does not establish a specific setup cost, EOQ, POQ interval, or result for the example in this lesson.
**[INFERENCE]** Reducing setup friction can make smaller, more frequent production lots economically practical; whether that happens at a particular plant requires local data.
Do not claim that Toyota applies the exact four-method comparison in this lesson unless separately documented.

### 9.2 Translating evidence into a plant decision

A plant should not choose LFL simply because a lean source favors flow.
It should measure actual changeover time and identify when frequent setups reduce throughput.
It should not choose EOQ merely because an enterprise system offers a fixed-lot parameter.
It should test the resulting order sequence against current dated demand.
It should not choose POQ only because the average coverage seems intuitive.
It should check the largest individual lot and its production week.
It should not release a Wagner-Whitin result solely because the mathematical objective is minimal.
It should check the model's missing resource, lead-time, material, and service constraints.

In a mixed-product plant, planners may use family batching to reduce sequence-dependent cleaning.
That practice requires a family and transition-time matrix, not the constant two-hour setup in the case.
A campaign can reduce cleaning events but can increase waiting time and stock.
The decision should compare the added holding and customer risk with the saved changeovers.
A separate constrained scheduling model may be required to evaluate the actual sequence.
The present four methods do not capture all of those interactions.

### 9.3 Evidence hierarchy for a local proposal

Start with dated production orders, timestamps, scrap, and approved operating instructions.
Reconcile those records with the physical setup definition.
Compare the proposed setup saving with observed performance across representative runs.
Use finance-approved cost assumptions and make the avoidable portion explicit.
Review stock age, write-offs, and space constraints by item family.
Document service exposure by actual due date and customer agreement.
Run the same demand snapshot through the candidate policies.
Confirm material and resource feasibility before requesting authorization.
After adoption, compare measured outcomes with the baseline using a consistent period and mix.

A public mathematical source supports the structure of a model.
A company description supports only what that company actually says about its operation.
Neither substitutes for a plant-specific causal evaluation.
A before-and-after inventory decrease can result from a demand decline rather than a policy change.
A setup reduction can shift work to another department rather than eliminate the work.
An apparent service improvement can reflect changed customer promises rather than improved execution.
The local decision record should identify such alternative explanations.

### 9.4 Reading the fictional result responsibly

The numeric comparison is an illustration, not a public case study.
The 48 USD gap is between a screened POQ candidate and an uncapacitated lower bound under the specified objective.
It is not a guaranteed saving from buying six additional machine-hours.
The extra hours could cost more than 48 USD and may not be available in the necessary weeks.
The larger lots could require an earlier component receipt or exceed a physical storage limit.
A robust proposal should quantify those consequences rather than extrapolate from the model alone.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Gross demand treated as net | Planned receipts exceed expected consumption | Existing stock or receipts omitted | Duplicate production and excess stock | Reconcile to MRP stock and receipt ledger |
| Forecast treated as commitment | Customer dates change without approval | Demand statuses collapsed | Unnecessary overtime or broken promises | Retain source and authority for each requirement |
| Annual and weekly units mixed | EOQ is implausibly high or low | Inconsistent demand and holding periods | Wrong lot quantity | Check dimensions before calculation |
| Setup expense overstated | Savings exceed avoidable spend | Fixed overhead treated as incremental | False business case for large lots | Reconcile to avoidable activities and bottleneck opportunity |
| Holding rate understated | Old stock and write-offs rise | Capital, expiry, or storage omitted | Large batches look too cheap | Build rate with finance and test sensitivity |
| Overlapping carrying components | Cost rate exceeds constituent evidence | Storage included twice | Excessively small lots | Document each cost component and overlap |
| Capacity ignored | Receipt week exceeds available hours | Large batch concentrated in one period | Late orders and queue growth | Screen load, then finite-schedule |
| Sequence ignored | Actual changeovers exceed standard | Family transition times differ | Schedule cannot execute | Use transition-specific setup data |
| Lead time ignored | Release date follows material need | Receipt treated as instantaneous | Late components or finished goods | Offset release and operation dates |
| Terminal stock ignored | Last-period surplus appears free | Horizon omits future value or disposal | Biased EOQ comparison | Extend horizon or model terminal value consistently |
| Demand changes after snapshot | Many manual order changes | Known-demand assumption breaks | Obsolete work and nervousness | Refresh by policy and respect time fences |
| Stock quality ignored | System balance exceeds usable balance | Quarantined stock counted as available | Unexpected shortage | Include quality status and reservations |
| Multiple orders counted as one setup | Accounting setups differ from actual transitions | Lot/order mapping is wrong | Misstated cost and capacity | Reconcile physical runs to order records |
| One order counted as one setup when split | Actual setups exceed plan | Shift, equipment, or cleaning requires restarts | Capacity and cost overruns | Model actual process interruptions |
| Lot minimum omitted | Suggested lot violates equipment minimum | Model assumes arbitrary quantity | Unreleasable or nonconforming order | Enforce technical minimum and multiples |
| Shelf life omitted | Inventory ages before use | Long coverage window precedes demand | Expiry and disposal | Apply shelf-life and consumption-date rules |
| Weekly pass mistaken for schedule | Production completes after intraweek due time | Buckets hide sequence and timing | Missed deliveries despite capacity pass | Validate daily or shift sequence |
| Double-counted resource losses | Available hours lower than credible calendar | OEE or downtime deducted twice | False overload and unnecessary intervention | Trace every capacity deduction once |
| Unapproved order changes | Released orders repeatedly rescheduled | Optimizer ignores time fence | Scrap, expedite cost, and nervousness | Fix or explicitly penalize near-term changes |
| Invalid annualization | Six-week mean used as a long-run forecast | Seasonal or exceptional mix ignored | EOQ parameter misrepresents demand | Use relevant forecast and seasonality scenario |
| Unmodeled yield loss | Good output below planned receipts | Run rate applies to input rather than good output | Insufficient production and load | Use compatible good-output yield and rate |
| Residual cost misinterpreted | Fixed-lot EOQ looks unfairly expensive or cheap | Terminal stock has unmodeled future value | Wrong policy recommendation | Compare over a longer horizon with consistent value |
| Uncapacitated optimum sold as feasible | Claimed savings vanish on scheduling | Capacity constraints absent from recurrence | Failed release or overload | Label bound and solve constrained alternative |
| Manual override not recorded | Planner cannot explain lot history | No audit trail for exception | Repeat problems and weak accountability | Log reason, owner, timing, and approval |

A red flag is a reason to investigate, not proof of a defect.
An unusually large lot may be justified by a genuine cleaning or supplier constraint.
The analyst's task is to explain its causal basis and its consequences.

## 11. Assumptions and limits

| Assumption | Why needed in this lesson | What breaks it | How to test or respond |
|---|---|---|---|
| Demand known over six weeks | Enables deterministic comparison | Forecast revisions and cancellations | Compare snapshots and run demand scenarios |
| Requirements already netted | Avoids modeling the full MRP ledger | Reservations or receipts change | Reconcile each bucket to the source system |
| Zero opening stock | Simplifies ledgers | Usable stock exists at snapshot | Include and allocate stock before lot sizing |
| No safety-stock requirement | Makes zero ending inventory permissible | Service policy requires a buffer | Add a dated minimum-stock constraint |
| All requirements met on time | Makes no-shortage costs comparable | Backorders are permitted or demand is lost | Model explicit lateness or service constraints |
| Constant setup cost | Allows one cost per lot | Product sequence, shift, or cleaning changes cost | Measure transitions and use dated or sequence-specific costs |
| Constant holding rate | Allows simple unit-week multiplication | Value, expiry, or warehouse cost changes | Use period- and item-specific rates |
| Costs are incremental | Makes modeled savings decision-relevant | Fixed salaried time cannot be redeployed | Validate with finance and operational alternatives |
| Instant weekly availability | Lets a receipt cover the same week's demand | Production finishes after a customer due time | Schedule at the necessary time resolution |
| Zero lead-time offset | Keeps worked receipt and release weeks aligned | Production or quality release takes time | Offset releases and check earlier resources |
| Constant run rate | Converts quantity to hours | Yield, learning, or quantity changes rate | Compare routing standards with actual good output |
| One independent setup per lot | Connects order count to setup count | Orders are combined or runs interrupted | Map physical setup events to lot plans |
| 20 hours reserved each week | Supports a simple capacity screen | Other products, maintenance, or labor compete | Replace with approved net available calendar |
| No minimum or maximum lot | Makes any positive lot admissible | Equipment, supplier, or regulation limits quantity | Add explicit lot bounds and multiples |
| Unlimited uncapacitated WW supply | Permits classical recurrence | Machine or component supply is limited | Solve a capacitated model or reject violating lots |
| No disposal or terminal credit | Keeps objective simple | Surplus has meaningful future value or loss | Extend horizon or model terminal economics |
| Inventory held at period end | Defines the cost ledger | Consumption or production occurs midweek | Use finer buckets or an appropriate time-weighted model |
| No discounting | Simplifies six-week sums | Long horizon or material financing effects | Apply a consistent discounted cash-flow convention |
| No setup carryover across horizon | Counts each new lot as a setup | Machine is already configured at the start | Adjust initial state and setup transitions |
| Costs do not depend on quantity | Supports linear run and holding assumptions | Yield, purchase price, or labor changes at thresholds | Add breakpoints and solve an appropriate model |
| No interactions among items | Allows a single-item comparison | Shared materials or constrained equipment | Build a multi-item capacity and material model |
| No uncertainty buffer | Separates batching economics from service protection | Demand or supply uncertainty is material | Set safety policy independently and test scenarios |

The table is a ledger of model conditions, not a claim that the case satisfies real plant controls.
An assumption may be acceptable for a screening calculation and unacceptable for an order release.
If a model cannot represent a material constraint, expose the limitation before communicating its cost result.

## 12. Exercises

### Exercise 1 — Explain the decision flow

**Type:** recall and interpretation.  
**Time:** 10 minutes.  
**Deliverable:** a six-step flow from demand snapshot to authorized release.

Include netting, lot sizing, capacity, and a distinction between a plan and a commitment.
Name one reason the process may return to an earlier step.

**Acceptance criteria:**

- The flow uses dated net requirements, not gross demand alone.
- It separates a lot proposal from a feasible and authorized order.
- It identifies a feedback path after a constraint violation.

### Exercise 2 — Build an LFL ledger

**Type:** calculation.  
**Time:** 15 minutes.  
**Deliverable:** a four-week receipt and cost table.

Requirements are 30, 50, 20, and 40 units in weeks 1–4.
Opening stock and safety stock are zero.
Each lot costs 75 USD in setup; holding is 1 USD/unit/week on ending stock.
Demand must be met on time and there are no lot constraints.

**Acceptance criteria:**

- Report every week's receipt and ending inventory, in units.
- Report the number of setups and the setup-plus-holding total, in USD.
- Explain why the modeled holding cost is zero.

### Exercise 3 — Derive and apply EOQ

**Type:** calculation and critique.  
**Time:** 20 minutes.  
**Deliverable:** an EOQ calculation and two-week stock ledger.

Annual demand is 2,600 units/year, setup cost is 50 USD/lot, and annual holding cost is 5 USD/unit/year.
Round the EOQ to the nearest whole unit with halves upward.
Beginning inventory is zero; week 1 demand is 100 units and week 2 demand is 150 units.
Receive full fixed lots when needed and do not trim the last lot.

**Acceptance criteria:**

- Show the formula and correct annual units.
- State the rounded lot size and both week's receipts and ending stocks.
- Identify the horizon issue created by the ending stock.

### Exercise 4 — Construct POQ

**Type:** application.  
**Time:** 15 minutes.  
**Deliverable:** a five-week receipt and holding ledger.

Weekly demand is 20, 30, 10, 40, and 25 units.
Use a two-week coverage period, 50 USD/setup, and 1 USD/unit/week holding on ending stock.
Begin with zero inventory and cap the final coverage window at week 5.

**Acceptance criteria:**

- Show each window, receipt quantity, and week's ending inventory.
- Calculate total setup and holding cost, in USD.
- Explain why the last window contains only one week.

### Exercise 5 — Solve a small dynamic program

**Type:** calculation.  
**Time:** 25 minutes.  
**Deliverable:** prefix costs and final lot coverage for three weeks.

Demand is 10, 20, and 10 units.
Setup cost is 30 USD/lot and holding cost is 1 USD/unit/week.
No opening stock, shortages, or capacity limits apply.

**Acceptance criteria:**

- Show `F[1]`, `F[2]`, and `F[3]` and the relevant candidate costs.
- Backtrack an optimal lot plan and quantify its setup and holding costs.
- State what optimality does and does not prove.

### Exercise 6 — Capacity decision

**Type:** application and decision.  
**Time:** 15 minutes.  
**Deliverable:** an exception note for the main six-week example.

Use the Wagner-Whitin lots in section 6 and the 20 machine-hour weekly limit.
Assume 2 machine-hours/setup and 10 good units/machine-hour.

**Acceptance criteria:**

- Calculate the week 1 and week 4 loads and gaps, in machine-hours.
- Identify a candidate that passes the stated weekly screen.
- Explain why its screen pass is not a complete production schedule.

### Exercise 7 — Challenge the economics

**Type:** critique.  
**Time:** 20 minutes.  
**Deliverable:** an assumption ledger with four prioritized risks.

The finance team says setup cost is uncertain and the fixed EOQ policy ends with 15 units in stock.
A scheduler reports that components have a one-week lead time.
A sales manager says half of the six-week requirements are forecasts.

**Acceptance criteria:**

- Address setup cost, terminal value, lead time, and demand status separately.
- Give one evidence source and one decision consequence for each risk.
- Distinguish an assumption change from an actual authorized commitment change.

### Exercise 8 — Design governance and a monitoring test

**Type:** design.  
**Time:** 25 minutes.  
**Deliverable:** a one-page decision and measurement protocol.

The plant wants to replace a fixed lot size with a POQ rule for one product family.
It also plans a changeover improvement, but the saving has not yet been verified.

**Acceptance criteria:**

- Identify decision owners for parameters, capacity, quality, and release.
- State how you will retain the baseline and input snapshots.
- Specify setup, stock, service, and plan-stability measures with units.
- Describe how the POQ policy will be re-evaluated after verified setup changes.
- Set an escalation path for overloads or unapproved changes.

## 13. Self-check answers

### Answer 1 — Flow

One valid sequence is:

1. Collect a dated demand and supply snapshot with status.
2. Net usable stock, reservations, and scheduled receipts.
3. Generate candidate lots using documented rules and costs.
4. Offset lead time and check material and resource requirements.
5. Revise or escalate infeasible candidates and obtain approvals.
6. Release the authorized plan and monitor execution against it.

A capacity overload can send the process back to step 3.
A change in customer commitment needs its own authorization; the optimizer does not grant it.

### Answer 2 — LFL

| Week | Requirement, units | Receipt, units | Ending stock, units | Setup, USD | Holding, USD |
|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 30 | 0 | 75 | 0 |
| 2 | 50 | 50 | 0 | 75 | 0 |
| 3 | 20 | 20 | 0 | 75 | 0 |
| 4 | 40 | 40 | 0 | 75 | 0 |
| Total | 140 | 140 | 0 terminal | 300 | 0 |

There are four setups and the total modeled cost is 300 USD.
The ledger makes precisely the requirement in each week and has no buffer or other stock.
Real LFL production may still have work-in-process or stock within a bucket.

### Answer 3 — EOQ

```text
EOQ = sqrt(2 * 2,600 * 50 / 5)
    = sqrt(52,000)
    = 228.035 units/order
rounded lot = 228 units/order
```

| Week | Receipt, units | Demand, units | Ending stock, units |
|---:|---:|---:|---:|
| 1 | 228 | 100 | 128 |
| 2 | 228 | 150 | 206 |

Week 2 needs a second full lot because 128 units cannot cover 150 units.
The 206 terminal units have future value or future holding and disposal consequences outside the stated two-week horizon.
A final partial order would be a different policy from the one specified.

### Answer 4 — POQ

| Window | Receipt week | Receipt, units | End stock by week, units |
|---|---:|---:|---|
| 1–2 | 1 | 50 | Week 1: 30; week 2: 0 |
| 3–4 | 3 | 50 | Week 3: 40; week 4: 0 |
| 5 | 5 | 25 | Week 5: 0 |

Setup cost is 3 × 50 = 150 USD.
Holding cost is (30 + 40) × 1 = 70 USD.
Total modeled cost is 220 USD.
The last window is truncated because no week 6 demand is included in the specified horizon.
If the business continues after week 5, test a longer horizon instead of assuming no future need.

### Answer 5 — Dynamic program

For week 1, one lot costs 30 USD, so `F[1] = 30 USD`.
For week 2:

```text
lot in week 1 covers weeks 1–2: 30 + 20 * 1 = 50 USD
separate week 2 lot: F[1] + 30 = 60 USD
F[2] = 50 USD
```

For week 3:

```text
one lot in week 1 covers weeks 1–3: 30 + 20 * 1 + 10 * 2 = 70 USD
week 1 prefix plus week 2 lot covering weeks 2–3: 30 + 30 + 10 = 70 USD
week 2 prefix plus separate week 3 lot: 50 + 30 = 80 USD
F[3] = 70 USD
```

There is a tie.
One optimum produces 40 units in week 1: setup 30 USD and holding 40 USD.
Another produces 10 units in week 1 and 30 units in week 2: setups 60 USD and holding 10 USD.
Both total 70 USD under the stated objective.
Neither result proves feasibility when machine hours, lead times, or materials are constrained.

### Answer 6 — Capacity

Week 1 load is 2 + 190/10 = 21 machine-hours, exceeding capacity by 1 machine-hour.
Week 4 load is 2 + 230/10 = 25 machine-hours, exceeding capacity by 5 machine-hours.
The two-week POQ candidate peaks at 17 machine-hours/week and passes this simplified screen.
The screen omits intraweek sequencing, shared resources, material availability, and any actual approval.
The POQ candidate is not proven to be the lowest-cost feasible plan.

### Answer 7 — Critique rubric

A strong response includes at least the following evidence and consequence pairs:

| Risk | Evidence to obtain | Possible decision consequence |
|---|---|---|
| Setup cost uncertainty | Observed changeover activities and finance cost decomposition | Re-rank lot patterns under plausible cost ranges |
| Terminal stock | Week 7 onward demand, stock age, expiry, and unit economics | Extend horizon or assign consistent terminal value |
| One-week component lead time | Approved BOM, supplier receipt dates, and availability | Offset releases and recheck material feasibility |
| Forecast status | Demand records and customer order status | Scenario-test revisions without changing firm commitments |

The answer should rank risks by materiality and explain the choice.
A setup-cost scenario does not establish that a setup improvement occurred.
A forecast revision does not authorize changing a confirmed customer date.

### Answer 8 — Design rubric

A strong protocol has:

- A named planner responsible for the scenario and exception record.
- Operations ownership of setup and resource calendars.
- Finance ownership of the economic assumptions and their incremental basis.
- Quality and engineering review of batch, shelf-life, and process restrictions.
- An authorized role for release and customer commitment changes.
- A retained baseline, common input snapshot, and effective parameter version.
- Setup events and machine-hours per period, with family and resource.
- Inventory units and unit-weeks, with aged stock and write-offs where relevant.
- On-time fulfillment against unchanged customer due dates.
- Count and magnitude of changes to firm planned orders.
- A method to compare like product mix and period lengths.
- A trigger to recalculate after a verified changeover cost or time change.
- A documented path for unresolved overloads and exceptions.

A weak protocol reports only average inventory and assumes every change is caused by the policy.
It also treats an unverified future improvement as an approved current parameter.

## 14. Interview questions

### 1. Why is a smaller lot not automatically better?

A strong answer compares lower cycle stock with more setups and resource consumption.
It distinguishes actual setup cost from bottleneck time and mentions service and shelf-life context.
A weak answer assumes inventory reduction has no operational cost.

### 2. What are the main assumptions behind EOQ?

A strong answer identifies stable demand, fixed setup cost, constant holding rate, and instantaneous replenishment in the basic model.
It explains that units and time bases must match.
A weak answer calls EOQ the exact optimum for every time-phased plant schedule.

### 3. How does POQ differ from a fixed EOQ policy?

A strong answer says POQ fixes an interval and sums requirements in that window, producing variable lot quantities.
It states the period rounding and treatment of the final window.
A weak answer describes both as the same fixed number of units.

### 4. What does Wagner-Whitin optimize?

A strong answer explains the finite-horizon deterministic setup-plus-holding objective and the dynamic-programming recurrence.
It states that the standard formulation is uncapacitated and forbids shortages under the lesson's assumptions.
A weak answer describes the result as a universally executable schedule.

### 5. How do you use an uncapacitated optimum when a machine is overloaded?

A strong answer treats the result as a model-specific lower bound, quantifies dated gaps, and compares feasible alternatives.
It preserves service commitments and seeks authorization for capacity actions.
A weak answer either releases the overloaded lot or claims the lower bound is the constrained optimum.

### 6. How do you compare policies with different terminal stock?

A strong answer identifies the horizon boundary, consistent production and inventory costs, and possible future value or disposal.
It proposes a longer common horizon or a justified terminal valuation.
A weak answer ignores extra stock or credits its full price while omitting its production cost.

### 7. What changes when setup time falls?

A strong answer re-estimates both setup time and incremental cost, then recalculates lots and capacity.
It recognizes that smaller lots may become attractive without claiming this is guaranteed.
A weak answer preserves the old lot unconditionally or assumes all time saved becomes cash savings.

### 8. Why can a weekly capacity pass still fail?

A strong answer names intraweek due times, sequence, operator availability, shared tooling, and component timing.
It recommends a finite schedule or other appropriate detailed check.
A weak answer treats total weekly hours as proof of a feasible sequence.

### 9. How would you check that demand was netted correctly?

A strong answer reconciles usable stock, reservations, receipts, safety policy, statuses, and item revisions to the MRP ledger.
It avoids double-counting parent and dependent demand.
A weak answer sums sales forecasts and calls the total net requirements.

### 10. How do you decide whether a lot policy improved performance?

A strong answer defines a baseline and measures setup hours, inventory, service, and stability over comparable product mixes.
It checks alternative explanations, exceptions, and quality outcomes.
A weak answer attributes any inventory decline to the new rule without a baseline.

## 15. Key terms

- [Backorder](#4-definitions-and-vocabulary) — demand fulfilled after its due date under an agreed or modeled rule.
- [Capacity](#4-definitions-and-vocabulary) — resource time available in a specified period.
- [Carrying charge](#5-the-mechanics-step-by-step) — periodic inventory cost rate used to derive holding cost.
- [Commitment](#1-why-this-topic-exists) — an authorized obligation, not merely a forecast.
- [Coverage](#5-the-mechanics-step-by-step) — periods or demand supplied by a lot.
- [Cycle stock](#4-definitions-and-vocabulary) — inventory caused by batched replenishment.
- [Demand bucket](#4-definitions-and-vocabulary) — a time interval used to aggregate requirements.
- [Economic order quantity](#55-calculate-and-apply-eoq) — steady-demand fixed lot derived from setup and holding costs.
- [Ending inventory](#6-worked-example) — units remaining after a period's receipts and demand.
- [Finite schedule](06_finite-scheduling.md) — time- and resource-feasible sequence of operations.
- [Forecast](#1-why-this-topic-exists) — estimate of demand rather than a confirmed obligation.
- [Gross requirement](#4-definitions-and-vocabulary) — total dated requirement before netting.
- [Holding cost](#53-estimate-setup-and-holding-economics) — cost per unit of inventory carried per period.
- [Incremental cost](#4-definitions-and-vocabulary) — cost that changes between alternatives.
- [Lead-time offset](#58-translate-receipts-into-a-feasible-proposal) — shift from receipt date to required release date.
- [Lot-for-lot](#54-calculate-lot-for-lot) — policy matching each positive net requirement with a lot.
- [Lot multiple](#4-definitions-and-vocabulary) — required increment of an order quantity.
- [Material requirements planning](04_mrp-explosion.md) — time-phased netting and dependent-demand planning.
- [Net requirement](#52-build-a-clean-net-requirement-series) — uncovered dated demand after specified netting.
- [Opportunity cost](#53-estimate-setup-and-holding-economics) — value of an alternative use forgone.
- [Period order quantity](#56-calculate-and-apply-poq) — fixed coverage interval with demand-dependent quantities.
- [Planned receipt](#4-definitions-and-vocabulary) — proposed quantity available in a specified period.
- [Planned release](#4-definitions-and-vocabulary) — proposed order start after lead-time offset.
- [Run rate](#6-worked-example) — good units produced per resource hour.
- [Safety stock](#11-assumptions-and-limits) — policy buffer against uncertainty.
- [Setup cost](#53-estimate-setup-and-holding-economics) — incremental modeled cost of preparing one lot.
- [Setup time](#6-worked-example) — resource hours consumed by preparation for a lot.
- [Shadow price](#4-definitions-and-vocabulary) — model-specific marginal value of additional constrained resource.
- [Terminal inventory](#68-interpretation-of-the-horizon) — stock remaining at the model horizon.
- [Time fence](#4-definitions-and-vocabulary) — boundary restricting near-term plan changes.
- [Wagner-Whitin](#57-solve-the-wagner-whitin-model) — dynamic lot-sizing method for a specified deterministic model.

## 16. Sources

1. Wagner, Harvey M., and Thomson M. Whitin. “Dynamic Version of the Economic Lot Size Model.”
   *Management Science* 5, no. 1 (1958): 89–96.
   [https://doi.org/10.1287/mnsc.5.1.89](https://doi.org/10.1287/mnsc.5.1.89).
   Primary source for the dynamic economic lot-size model.
2. Harris, Ford W. “How Many Parts to Make at Once.”
   *Factory, The Magazine of Management* 10, no. 2 (1913): 135–136, 152.
   Reprinted in *Operations Research* 38, no. 6 (1990): 947–950.
   [https://doi.org/10.1287/opre.38.6.947](https://doi.org/10.1287/opre.38.6.947).
   Historical source for the economic lot-size trade-off.
3. Toyota Motor Corporation. [Toyota Production System](https://global.toyota/en/company/vision-and-philosophy/production-system/).
   Company description of production-system concepts; it is not evidence for the fictional cost figures.
4. GS1. [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
   Standards source for identification and barcode application identifiers; consult the applicable edition.
5. International Electrotechnical Commission. [IEC 62264-1, enterprise-control system integration](https://webstore.iec.ch/en/publication/6675).
   Scope and information-model context; check the applicable current edition.
6. United States National Archives. [21 CFR Part 211](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-C/part-211), including [§ 211.188](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-C/part-211/subpart-J/section-211.188).
   Current regulatory text for covered drug-product manufacturing; applicability is product- and jurisdiction-specific.
7. United States Food and Drug Administration. [Food Traceability Rule](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods).
   Official scope and implementation information; check current dates and applicability.

The original articles and standards support the methods and definitions cited, not the fictional numerical results.
The case calculations follow the assumptions stated in sections 6 and 7.
Regulatory links are educational, not legal advice; consult current authoritative text and qualified specialists.

## 17. Links

- [Module 15 index](_index.md) — topic map and sequence.
- [Planning cascade](01_planning-cascade.md) — decision levels and planning governance.
- [Manufacturing master data](02_manufacturing-master-data.md) — items, routings, and calendars.
- [Master production scheduling](03_master-production-scheduling.md) — upstream production commitments.
- [MRP explosion](04_mrp-explosion.md) — gross-to-net logic and lead-time offsetting.
- [Capacity requirements planning](05_capacity-requirements.md) — resource load and calendar reconciliation.
- [Finite scheduling](06_finite-scheduling.md) — detailed sequence and resource feasibility.
- [Module 15 survey](../../../course/15_manufacturing_production_planning.md) — full manufacturing planning overview.
- [Production planning labs survey](../../../course/30_simulation_optimization_production_labs.md) — optimization and simulation practice context.
