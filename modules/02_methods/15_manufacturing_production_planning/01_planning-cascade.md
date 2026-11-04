# The Planning Cascade and S&OP

> Module 15 — Manufacturing and Production Planning  
> Topic 01 of 11  
> Estimated study time: 4 hours  
> Prerequisites: Modules 01 and 10; basic arithmetic and inventory balances  
> Version 2026.5

## 1. Why this topic exists

A customer wants a delivery date.
Sales sees an attractive order.
The plant sees a full work center.
Purchasing sees a component with a long replenishment lead time.
Finance sees a revenue target that assumes all three problems can be solved.
Unless somebody connects these views, each team can be locally correct while the company makes an impossible promise.

The planning cascade is the sequence of decisions that turns a commercial view into executable work and returns execution evidence to decision makers.
It is not just a software workflow.
A plan has an owner, a time horizon, a unit of measure, a decision, a version, and an effective time.
A production order without available material is not made feasible by appearing in an enterprise resource planning (ERP) system.
Likewise, a valid sequence on one machine does not automatically support the revenue plan.

Sales and operations planning (S&OP) reconciles demand and supply at an aggregate level.
Integrated business planning (IBP) extends or integrates that reconciliation with financial and strategic decisions; implementations differ.
Sales and operations execution (S&OE) manages near-term exceptions against the approved plan.
A master production schedule (MPS) translates the chosen aggregate direction into time-phased end-item quantities.
Material requirements planning (MRP) calculates dependent material requirements and planned orders.
A finite schedule assigns operations to actual resources and times within modeled constraints.
Execution dispatches and confirms work, captures deviations, and supplies feedback.

The decisions cannot all be made in the same meeting.
A monthly executive forum can authorize a capacity investment but cannot sensibly choose the next job at a machine every hour.
A shift supervisor can respond to a minor stoppage but should not silently sacrifice a strategic customer's delivery commitment.
The handoff between these decision rights is where the cascade succeeds or fails.

Consider the cost of ignoring the handoff.
If an aggregate plan authorizes a volume that cannot be disaggregated into feasible item mix, an MPS will inherit a hidden capacity gap.
If an MRP run assumes a supplier can deliver in two weeks when the confirmed lead time is four, a schedule may reserve a machine for material that will not arrive.
If actual yield falls and the record is not updated, the next planning cycle repeats the same shortfall.

A useful cascade makes these contradictions visible early enough to choose among imperfect alternatives.
The correct answer might be overtime, a different product mix, a revised customer promise, or a capacity investment.
The analyst's job is to show the implications, identify who can decide, and preserve the decision record.
The software's job is to calculate and communicate consistently; it does not own the business trade-off.

This topic teaches the interfaces and decisions, not every algorithm inside each layer.
Later Module 15 topics deepen master data, master scheduling, MRP, capacity, and detailed sequencing.
The numbers below are illustrative, not a claim about a particular plant.

## 2. Learning outcomes

After completing this topic, you can perform the following observable tasks.

1. **Map** S&OP, IBP, S&OE, MPS, MRP, scheduling, and execution to decisions, owners, units, and time horizons.
   Exercise 1 checks the mapping.
   The point is not to memorize a universal calendar; it is to explain why a particular decision needs its horizon.
2. **Distinguish** a forecast, a target, an approved plan, and a customer commitment in a concrete situation.
   Exercise 2 tests the distinction and its consequences for demand consumption.
3. **Reconcile** a family plan to item-week production and inventory using a stated balance equation.
   Exercise 3 asks you to calculate the consequences of a mix or volume change.
4. **Calculate** a simple dependent component requirement, net supply, and time-phased release.
   Exercise 4 checks both quantities and dates.
5. **Diagnose** an aggregate-feasible plan that violates a work-center or component constraint.
   Exercise 5 asks you to trace the conflict to the correct decision forum.
6. **Choose and justify** a near-term response using service, capacity, cash, risk, and decision authority.
   Exercise 6 requires a documented recommendation and escalation trigger.
7. **Design** a closed-loop operating cadence with versioned inputs, an exception path, and an execution feedback measure.
   Exercise 7 uses a rubric rather than one prescribed design.
8. **Critique** planning stability and performance measures for timing, denominator, and unintended incentives.
   Exercise 8 asks you to construct a comparable metric.

These outcomes are sequenced deliberately.
You first need to recognize the object being planned.
You can then calculate its consequences and decide which human has authority to change it.
A correct calculation without an owner is not an actionable plan.

## 3. The core mental model

### 3.1 A cascade with two-way information flow

```text
  Strategy / financial guardrails / demand and supply scenarios
                |                   ^
                v                   | strategic gap and escalation
      Monthly S&OP / IBP: family, site, month
                | approved volume, mix limits, capacity and inventory policy
                v
      MPS: end item, plant, week; rough-cut capacity check
                | planned finished-goods receipts and releases
                v
      MRP: component, plant, day/week; lead-time and stock netting
                | planned orders, exceptions, purchase proposals
                v
      Finite schedule: operation, resource, shift or hour
                | authorized dispatch list, materials and labor checks
                v
      Execution: release, make, inspect, confirm, ship
                | actual quantity, time, scrap, status and disruptions
                +----> S&OE: near-term exception response ----+
                           |                                     |
                           +--> MPS / MRP / schedule revisions --+
                           +--> escalate beyond local authority --> S&OP / IBP
```

The downward arrows represent increasingly specific decisions, not increasing certainty.
The upward arrows represent actual results, feasibility evidence, and requests for decisions.
Information should travel both ways even when the schedule is stable.
A planning layer may discover a constraint that its parent did not model.
The parent must then accept the revised consequence or authorize a different option.

### 3.2 Aggregation and disaggregation

S&OP might approve 400 finished units for one family over four weeks.
That number says nothing by itself about whether the units are the high-speed or low-speed variant.
A feasible mix must fit the bottleneck hours, material constraints, available skills, and demand dates.
Disaggregation maps family quantities to item-location-week quantities.
Reaggregation checks that the item plan still fits the approved family and financial view.
Neither direction is automatic when conversion factors vary by item.

Suppose variants A and B belong to the same family.
Variant A requires 0.4 constrained machine-hours per unit, while B requires 0.8 hours per unit.
An aggregate family plan of 400 units can require from 160 to 320 run-hours before setups.
The family unit is useful for commercial volume but insufficient for capacity feasibility.
An equivalent bottleneck-hour measure or an explicit mix assumption exposes the difference.

### 3.3 Four different quantities

A **forecast** is an estimate of demand, with a horizon and uncertainty.
A **target** is a desired outcome, such as a revenue or service objective.
A **plan** is an authorized set of intended actions under stated assumptions.
A **commitment** is an agreed promise, such as a customer delivery date or a supplier-confirmed receipt.
The same number can appear in more than one category only if the context and authority justify it.
A target is not evidence of demand, and a forecast is not a customer order.

The MPS is a production plan, not automatically an unconditional promise to each customer.
MRP planned orders are proposals, not proof that a supplier has accepted them.
The schedule is an intended sequence, not proof that an operation occurred.
An execution confirmation is an observation, subject to validation, not an instruction for future work.

### 3.4 Time and commitment

A common design separates a frozen window, a slushy or controlled-change window, and a liquid planning window.
Inside the frozen window, changes require explicitly assigned authority and a documented reason.
In the controlled-change window, specified swaps or quantity changes may be permitted within guardrails.
Farther out, the plan can be revised more freely as demand and supply evidence changes.
The exact boundaries depend on cumulative lead times, customer promises, changeover cost, and process risk.
A frozen window is a governance rule, not a claim that physical events cannot change.

### 3.5 The loop at a glance

Read the cascade from left to right as a question-and-answer chain.

- What demand and financial outcomes are plausible?
- What family-level supply, inventory, and capacity response is authorized?
- Which finished items should be completed in which weeks?
- Which components and intermediate items must be available, and when?
- Which feasible operations should run on which resources and shifts?
- What actually happened, and what decision is now needed?

If the answer at one layer invalidates an upstream assumption, send an exception upward rather than silently changing the assumption.
If a decision is made upstream, translate it into specific dated instructions downstream.
The cascade works when both directions are timely and traceable.

## 4. Definitions and vocabulary

The table uses illustrative grains and units; an organization may choose a different grain if it documents the conversion and ownership.

| Term | Definition | Common misuse | Typical unit or grain |
|---|---|---|---|
| Planning cascade | Linked decisions from aggregate reconciliation to execution and feedback | Treating it as a one-way export | Decision level × period |
| S&OP | Cross-functional process reconciling demand, supply, inventory, and business choices | A sales presentation with no decisions | Family × site × month, units |
| IBP | Integrated planning approach connecting operating scenarios with financial and strategic outcomes | Assuming every process using the name has the same scope | Family or business × month, units and currency |
| S&OE | Near-term process for managing exceptions and executing within agreed guardrails | Relabeling every daily expediting call as a strategic review | Item × site × day/week |
| Demand forecast | Estimate of future independent demand | Treating it as an order or guaranteed minimum | Item × location × period, units |
| Target | Desired outcome set by management | Adding it to the forecast as if it were another order | Currency or units per period |
| Demand plan | Versioned operational view of expected demand with assumptions | Hiding a commercial override without attribution | Item/family × period, units |
| Customer order | A request or accepted commercial transaction with quantity and due date | Assuming all orders are immediately confirmed or firm | Order line × requested date, units |
| Commitment | Authorized promise to a party, subject to its terms | Equating a proposed MRP receipt with supplier acceptance | Order line × promised date, units |
| Plan | Authorized intended actions, constraints, and assumptions | Describing an unapproved scenario as the baseline | Item or family × period, units |
| Scenario | Internally consistent alternative assumptions and consequences | Mixing quantities from alternatives into a single plan | Version × entity × period |
| Baseline | Selected reference version for comparison | Overwriting history after a change | Version ID × effective timestamp |
| Horizon | Future span covered by a decision or model | Assuming it equals the update frequency | Days, weeks, or months |
| Cadence | Frequency of review or publication | Assuming daily review means a one-day horizon | Meetings or runs per week/month |
| Time bucket | Interval in which quantities are aggregated | Treating weekly data as precise hourly feasibility | Calendar day, week, or month |
| Planning fence | Boundary controlling permitted changes | Assuming no emergency change can ever occur | Timestamp or planning week |
| Frozen window | Near-term interval where changes need restricted authority | Confusing a governance freeze with physical certainty | Days or weeks |
| Controlled-change window | Interval with limited changes under defined rules | Allowing any change without recording its impact | Weeks |
| Planning family | Aggregation of items for a chosen decision | Assuming items in one family consume identical capacity | Family × site × month |
| Mix | Distribution of volume across variants or channels | Comparing family totals while ignoring variant constraints | Share or units by item |
| Disaggregation | Mapping an aggregate quantity to detailed item-period quantities | Using stale proportions without checking bottleneck load | Item × period, units |
| Reconciliation | Checking detailed and aggregate views against each other | Forcing totals to match while ignoring timing or units | Family × period, units and currency |
| MPS | Authorized time-phased end-item production or receipt plan | Treating forecast demand as the production quantity | End item × plant × week, units |
| Projected available balance | Expected inventory after dated receipts and demand | Counting blocked stock or backorders as usable supply | Item × end of period, units |
| Safety stock | Policy buffer intended to protect against uncertainty | Describing it as customer demand or physically reserved stock by default | Item × location, units |
| Bill of materials (BOM) | Effective-dated parent-component relationship and usage | Using a current BOM to reconstruct an old plan without versioning | Parent × component × version, component units per parent |
| Routing | Ordered operations and resource standards for a production item | Assuming the routing represents all real-world constraints | Item × operation × version, hours per unit/setup |
| Work center | Resource or group with a capacity calendar | Treating all pooled machines or skills as interchangeable | Center × period, available hours |
| Rough-cut capacity planning (RCCP) | Coarse comparison of MPS load with key-resource capacity | Calling a family-volume balance a finite schedule | Key resource × week, standard hours |
| Capacity requirements planning (CRP) | Time-phased comparison of order-derived load and resource availability | Assuming a capacity report has fixed an overload | Work center × period, hours |
| MRP | Time-phased netting, lot sizing, lead-time offsetting, and BOM explosion | Treating its proposed dates as supplier or machine commitments | Item × location × period, units |
| Gross requirement | Total requirement before netting available supply | Adding a parent receipt requirement twice | Component × need date, units |
| Net requirement | Additional supply needed after available stock, receipts, and policy | Subtracting the same receipt in multiple periods | Item × need date, units |
| Planned order receipt | Proposed quantity to arrive or finish in a given period | Calling it a released purchase order | Item × receipt date, units |
| Planned order release | Proposed start or order date after lead-time offset | Assuming an unapproved release has actually occurred | Item × release date, units |
| Scheduled receipt | Existing order expected to arrive or complete on a stated date | Assuming an unconfirmed or overdue receipt is certain | Order × date, units |
| Lead time | Assumed elapsed or working time from release to receipt | Applying a calendar lead time to working-day dates without conversion | Days or weeks, calendar specified |
| Lot size | Quantity dictated by a production or purchasing rule | Rounding demand but ignoring residual stock | Units per order |
| Available-to-promise (ATP) | Policy-specific uncommitted supply available to support a promise | Equating ATP with physical inventory without allocations | Item × promise date, units |
| Finite schedule | Time-phased operation assignments respecting modeled resource constraints | Assuming feasibility for constraints omitted from the model | Operation × resource × time |
| Dispatch list | Prioritized work released to an operator or resource | Treating an unapproved sequence as a dispatch instruction | Work order × operation × shift |
| Execution confirmation | Report of actual work, time, quantity, or status | Counting an incomplete or uninspected quantity as saleable stock | Event × timestamp, units and hours |
| Manufacturing execution system (MES) | System supporting production execution and shop-floor information | Assuming all MES events are validated quality releases | Order/operation/event |
| Exception | Condition outside a defined tolerance requiring attention or a decision | Flooding planners with every ordinary change | Item/resource × date × severity |
| Escalation | Routing a decision to an owner with sufficient authority | Forwarding an issue without options or deadline | Issue × decision owner × due time |
| Plan nervousness | Unnecessary frequent changes to planned dates or quantities | Measuring all changes as bad, including justified disruption response | Changed item-periods or absolute units |
| Pegging | Trace from a supply or component requirement to its demand drivers | Assuming a single fixed link where stock is pooled and reallocated | Demand/supply relationship |
| Yield | Good output divided by input or completed quantity under a defined convention | Applying scrap allowance twice | Good units / input units, percent |
| Master data | Controlled reference data used to calculate or execute a plan | Treating a corrected current value as proof of historical accuracy | Item, BOM, routing, calendar, version |
| Transaction data | Time-stamped orders, inventory movements, and confirmations | Treating a stale extract as a live view | Event or order line × timestamp |
| Decision right | Explicit authority to approve a specified change within limits | Assuming that meeting attendance implies approval authority | Role × decision × threshold |

## 5. The mechanics, step by step

### 5.1 Establish the planning contract

1. Name the business decision and its accountable owner.
   A monthly family plan and a shift dispatch are separate decisions even if the same person attends both meetings.
   Input: the operating model and management delegations.
   Output: a decision register with authority, timing, and escalation route.
2. Choose the planning horizon based on the decisions it must cover.
   A capacity investment may need more than a year of visibility.
   A component purchase plan must reach beyond its replenishment lead time.
   A detailed schedule may be credible for only a few days when resources or arrival times are volatile.
   Horizon and review cadence are separate: a weekly meeting can examine a 26-week horizon.
3. Select the grain for each decision.
   Aggregate months and product families reduce noise for leadership.
   End-item weeks support inventory and production timing.
   Component dates support procurement, and shifts or hours support dispatch.
   State calendar boundaries and time zone where dates cross plants or suppliers.
4. Define the units and conversions.
   Specify whether a volume is input units, good finished units, cases, kilograms, or machine-hours.
   Define exchange rates and the effective period for financial views.
   Separate variable cost, contribution, and revenue; do not sum unlike measures.
5. Define the cutoffs and version identifiers.
   Input snapshots should identify when demand, stock, receipts, and capacity were extracted.
   If their cutoffs differ materially, record the mismatch before calculating a reconciled plan.
   Output: a timestamped input manifest and a uniquely identified scenario.
6. Define control boundaries.
   Establish which periods are frozen, who may approve an exception, and which changes require customer communication.
   A boundary without an escalation path encourages unauthorized workarounds.
   Review boundaries when actual supplier and manufacturing lead times change.

### 5.2 Construct a demand view

1. Bring a statistical or judgmental forecast together with known orders and commercial assumptions.
   Preserve each input separately, including the owner of any manual override.
   Input grain must support the later item-location plan or have an explicit allocation method.
2. Decide how orders consume forecast.
   A simple policy may use the larger of forecast and orders in each bucket.
   A more complex policy may consume forecasts across nearby dates, channels, or demand classes.
   The rule must be explicit because adding forecast and orders can double count the same future sale.
3. Separate unconstrained demand from a capacity-constrained promise.
   A lost opportunity is invisible if the demand forecast is reduced merely to make the supply plan balance.
   Record requested demand, constrained plan, and accepted commitment as separate objects.
4. Reconcile the demand view to commercial assumptions.
   Price, promotions, product launches, and end-of-life changes may affect mix as well as total volume.
   Sales should explain overrides and their expiry, not merely enter a higher target.
5. Quantify uncertainty sufficiently for the decision.
   A low, base, and high case can be more useful than a falsely precise single number.
   Show how each case changes capacity, inventory, and commitments.
   A scenario is not a probability distribution unless probabilities are actually estimated.
6. Output a versioned demand plan and a list of assumptions.
   Flag a demand change inside the frozen window for S&OE, rather than automatically changing the schedule.

### 5.3 Reconcile family supply in S&OP or IBP

1. Aggregate demand using a stable family hierarchy.
   Check that every item has exactly the intended family and site mapping for the effective date.
   Unmapped or multiply mapped items must be resolved before comparing family totals.
2. Build an inventory bridge.
   Beginning inventory + planned production receipts − planned demand = ending inventory, before adjustments.
   Account separately for returns, transfers, scrap, and backorders when they exist.
   A family balance only works if the units are genuinely comparable.
3. Construct a rough supply view.
   Identify bottleneck resources, planned downtime, labor availability, and supplier constraints.
   Compare required hours with available hours using the proposed mix and setup assumptions.
   A monthly total can conceal a weekly overload, so expose tighter time buckets where appropriate.
4. Develop options when demand and supply disagree.
   Examples include overtime, rescheduling, subcontracting, mix changes, alternative components, and revised promises.
   For each option, state feasibility, service effect, incremental cost, inventory effect, and risk.
   Do not count a supplier expedite as feasible until the supplier can actually support it.
5. Connect the physical view to the financial view.
   Translate units into revenue, cost, margin, and cash using documented assumptions and timing.
   A physical production receipt is not necessarily revenue: shipment and recognition rules matter.
   Keep a financial target visible even when the operational plan cannot reach it.
6. Hold the executive decision forum.
   Present the gap, alternatives, recommendation, decision deadline, and owner.
   The authorized executive selects or rejects an option within the organization's governance.
   Record unresolved decisions and deadlines instead of describing them as consensus.
7. Publish the approved family plan and guardrails.
   Include quantities, periods, mix assumptions, capacity authorization, inventory policy, and financial implications.
   Record the version, effective time, approver, and open risks.
   Output becomes an instruction and boundary for detailed planning, not an hourly schedule.

### 5.4 Disaggregate into the MPS

1. Allocate approved family production to end items, plants, and weeks.
   Use item-level orders, forecast consumption, opening stock, service policies, and feasible product mix.
   An allocation rule is a starting point, not authority to violate an item-specific commitment.
2. Calculate projected finished-goods balances.
   For each item and week: ending balance = beginning balance + scheduled and planned receipts − demand.
   Compare the balance with the intended safety stock and any explicit reservations.
   Treat backorders consistently rather than hiding negative balances.
3. Check timing and feasibility at key resources.
   Convert proposed quantities to bottleneck hours, including relevant setups and planned downtime.
   If the monthly family total fits but a week does not, evaluate a feasible shift of work or escalate.
   Confirm that earlier production has material, storage space, and an acceptable quality or shelf-life consequence.
4. Reconcile upward.
   Sum item-week quantities back to the approved family-period plan.
   Explain rounding, calendar boundaries, yield conversion, and changes in mix.
   If the reconciled volume or guardrails cannot be honored, obtain an authorized decision rather than forcing a match.
5. Publish the MPS with status by item and week.
   Mark released, firm-planned, and tentative quantities according to the local system's definitions.
   A finished-goods receipt date is not necessarily the start date; use manufacturing lead time to offset it.
6. Monitor MPS changes.
   Compare like-for-like snapshots at a fixed lead time before execution.
   Split approved emergency changes from unexplained churn.
   Use the metric to investigate causes, not to discourage necessary safety or customer decisions.

### 5.5 Explode and net material requirements

1. Start from valid parent planned order releases, not a second copy of parent receipts or demand.
   A child is generally needed when its parent operation consumes it; this simplified lesson assumes parent release.
   In real routings, operation-specific issue dates may be later.
2. Select the correct effective BOM and production version.
   Multiply the parent release quantity by the component usage quantity, accounting for documented yield and scrap conventions.
   Consolidate all parent requirements for a shared component before netting it.
3. Establish available opening stock and dated receipts.
   Exclude quality-blocked, allocated, expired, or otherwise unavailable stock under the stated policy.
   Separate a firm confirmed supplier receipt from an unconfirmed planned receipt.
4. Net requirements in time order.
   Beginning balance + scheduled receipts − gross requirements gives the pre-replenishment balance.
   A shortage relative to the policy floor creates a net requirement.
   Apply the lot-size rule and carry the resulting balance to the next bucket.
5. Offset each proposed receipt by the replenishment lead time using the correct calendar.
   A receipt needed in week 3 with a one-week lead time calls for release in week 2 under a weekly bucket convention.
   An order already released should not be treated as a newly executable proposal.
6. Review exceptions.
   A release before the current planning date, an overdue receipt, or an unconfirmed supplier date signals a decision.
   Check whether to expedite, substitute, replan production, or change the customer promise.
   The chosen option must be confirmed by the responsible owner.
7. Publish actionable order proposals and traceability.
   Include item, quantity, release and receipt dates, demand drivers, lead-time assumption, and status.
   Pegging should indicate affected demand, while acknowledging that pooled stock can be reallocated under policy.

### 5.6 Check capacity and make a finite schedule

1. Translate orders into work-center load using routing versions and valid standard times.
   Count setup once per lot or changeover, not once per unit unless the process actually requires that.
   Add run time, processing loss allowances, maintenance, and other modeled capacity use without double counting.
2. Compare load with a dated capacity calendar.
   Available hours should reflect machines, staffing, breaks, holidays, and scheduled maintenance.
   Clearly distinguish theoretical hours from effective hours and identify where any efficiency factor is applied.
3. Classify an overload.
   Determine whether it is caused by an inaccurate standard, a demand spike, a mix shift, a missing shift, or a real bottleneck.
   Verify available materials and labor before selecting a capacity response.
4. Generate alternatives.
   Pulling work earlier consumes earlier capacity and may create inventory.
   Overtime consumes authorized labor and may incur cost or fatigue risk.
   A subcontractor can add supply only with qualified capacity, lead time, quality controls, and contractual approval.
   Moving a job later requires an explicit service consequence.
5. Sequence released operations within the finite horizon.
   Respect precedence, machine eligibility, material readiness, setup, labor, and maintenance constraints that are actually modeled.
   A schedule is finite only with respect to those constraints; unmodeled constraints can still make it infeasible.
6. Agree on dispatch authority.
   The scheduler may propose a sequence, but the plant must know when it becomes the controlled dispatch list.
   A supervisor may change a sequence within a specified tolerance; other changes require escalation.
7. Publish a schedule with revision number, resource, start and finish times, material status, and known risks.
   Do not describe a weekly capacity balance as a proven hour-by-hour schedule.
   Confirm that all affected teams have received the same revision.

### 5.7 Execute, sense, and respond through S&OE

1. Release work only when authorization, material, routing, resource, and quality prerequisites are satisfied.
   Record any approved exception instead of changing stock or dates solely to pass a system check.
2. Dispatch the next operation and record actual start and completion times.
   Capture good output, scrap, downtime reason, labor or machine time, and material consumption at an appropriate grain.
   Separate a production completion from quality release if inspection is required.
3. Compare actual results with the frozen plan and current schedule.
   Identify missing material, downtime, late orders, poor yield, and demand changes.
   Use exception thresholds to avoid drowning the team in changes that do not require a decision.
4. Resolve local exceptions within delegated authority.
   A scheduler might swap two eligible jobs without changing a customer promise or exceeding approved overtime.
   A supervisor might stop unsafe work regardless of plan stability.
   Record the action, owner, reason, and expected downstream impact.
5. Escalate decisions that cross guardrails.
   Examples include a missed customer commitment, a frozen-window quantity change, unbudgeted overtime, or a quality compromise.
   The escalation must include options, consequences, decision owner, and required decision time.
6. Communicate accepted changes to affected parties.
   Update the MPS, purchase commitments, dispatch list, and customer promise as appropriate.
   A message to one planner does not update every dependent system automatically.
7. Close the loop.
   Feed validated actuals and the reason for variance into the next planning cycle.
   If lead time, yield, or capacity repeatedly differs from the assumption, change the controlled master data through its owner.
   Do not overwrite historical snapshots needed to learn what the plan previously assumed.

### 5.8 Decision rights and illustrative horizons

The following arrangement is a design example, not a universal standard.
Actual authority depends on legal entities, safety rules, contracts, and the organization's delegation of authority.

| Layer or forum | Typical horizon and cadence | Main decision | Accountable role | Escalate when |
|---|---|---|---|---|
| Strategy / capital | 1–5 years; quarterly or annual | Network, technology, major capacity | Executive or board under delegation | Investment exceeds delegated limit |
| S&OP | 3–18 months; monthly | Family demand-supply balance and capacity choices | Business executive | Cross-business trade-off or policy breach |
| IBP | Often 18–36 months; monthly/quarterly | Integrated operating and financial scenarios | Executive with finance | Strategic gap or investment decision |
| S&OE | 0–13 weeks; daily/weekly | Resolve exceptions inside agreed guardrails | Operations or supply lead | Commitment, budget, safety, or policy boundary crossed |
| MPS | 1–26 weeks; weekly or more often | Item-week planned production | Master scheduler | Family plan or frozen window cannot be honored |
| MRP | Through component cumulative lead time; daily/weekly | Planned material order quantities and dates | Materials planner / buyer | Supplier or stock cannot meet need |
| Finite scheduling | Hours–2 weeks; shift/daily | Resource and operation sequence | Production scheduler | Missing constraint, material, or capacity |
| Execution | Minutes–days; continuous | Dispatch, stop, record, recover within limits | Supervisor / operator by role | Safety, quality, or unauthorized commitment |

Horizon ranges overlap deliberately.
A 20-week supplier lead time can require an MRP horizon longer than the MPS's firm window.
A monthly S&OP meeting can examine a weekly exception if the business impact exceeds local authority.
A sophisticated implementation may combine forums or systems while preserving the distinct decisions.

### 5.9 Define a good decision record

A useful record can be concise, but it should contain enough information to reconstruct the decision.

- The decision identifier and date.
- The accountable approver and consulted roles.
- The planning version and source-data cutoff.
- The problem, affected items, locations, periods, and customers where applicable.
- The baseline assumption and the evidence that invalidated it.
- Feasible alternatives and the cost, service, capacity, inventory, and risk consequences of each.
- The chosen option, rejected options, and reason.
- Required downstream updates and named owners.
- A decision deadline and the next review date.
- Any unresolved risk, including whether a supplier or customer must confirm.

A record does not replace action.
The owner must verify that the decision was translated into the relevant operational systems and communications.
A later change should create a new version rather than erasing the original decision.

## 6. Worked example

### 6.1 Situation and planning boundary

Northstar Pumps is a fictional manufacturer of one pump family at one plant.
The example uses one finished item, P-24, to make the interfaces visible without implying that one-item planning is typical.
All quantities are good finished units unless otherwise stated.
Weeks are consecutive seven-day planning buckets; week 0 is the current decision week.
The example assumes no backorders, returns, transfers, or production scrap.
Costs are in U.S. dollars (USD).

The planning team must convert a four-week family plan into an executable supply plan.
It must decide how to avoid a week 3 bottleneck overload without changing the approved four-week total.
The customer commitments are assumed accepted and feasible only if the dated supply and execution steps below are carried out.
They are not inferred from the forecast.

### 6.2 Demand, targets, and commitments

| Quantity | Week 1 | Week 2 | Week 3 | Week 4 | Four-week total |
|---|---:|---:|---:|---:|---:|
| Forecast demand, units | 90 | 100 | 110 | 100 | 400 |
| Accepted customer orders, units | 80 | 95 | 70 | 30 | 275 |
| Planning demand, units | 90 | 100 | 110 | 100 | 400 |

The local forecast-consumption rule is the larger of forecast and accepted orders in each week.
Here, accepted orders are already included in the forecast; they are not added again.
The 400-unit forecast is uncertain demand, not a customer commitment.
The 275-unit accepted-order total is a commitment under the example's assumptions.
The commercial target is 420 units sold over the four weeks, including 20 units of potential business not in the base forecast.
The target is not inserted into MRP as if those 20 units were already orders.
If sales later secures additional orders, the team re-evaluates demand consumption and the capacity decision.

### 6.3 S&OP family balance

Starting finished-goods inventory is 40 units at the beginning of week 1.
The policy floor is 20 units at each week end.
S&OP approves 380 units of planned production receipts over the four weeks.
The approved family ending balance is therefore 40 + 380 − 400 = 20 units.
The 380-unit plan is not the 400-unit forecast: the difference is the planned drawdown of 20 units of inventory.

A first MPS proposal is shown below.
It is a scenario until the capacity check and authorization are complete.

| Week | Opening stock, units | Demand, units | Proposed receipt, units | Ending stock, units |
|---|---:|---:|---:|---:|
| 1 | 40 | 90 | 70 | 20 |
| 2 | 20 | 100 | 100 | 20 |
| 3 | 20 | 110 | 110 | 20 |
| 4 | 20 | 100 | 100 | 20 |
| Total | — | 400 | 380 | 20 final |

For week 1, 40 + 70 − 90 = 20 units.
For week 3, 20 + 110 − 110 = 20 units.
All weeks meet the 20-unit policy floor, but the table does not yet establish capacity feasibility.
The 380-unit total matches the family authorization, but a total alone does not authorize a weekly overload.

### 6.4 Rough-cut capacity check

The bottleneck is a test cell.
Each finished unit requires 0.4 cell-hours of run time.
One production lot per week requires a 2-hour setup.
The available cell capacity is 42, 48, 42, and 42 hours in weeks 1–4, respectively.
The extra six hours in week 2 are an already staffed and authorized shift extension, not a new overtime decision.
The receipt week is assumed to coincide with the bottleneck processing week in this simplified example.

| Week | Initial receipt, units | Run hours | Setup hours | Load, hours | Available, hours | Gap, hours |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 70 | 28 | 2 | 30 | 42 | +12 |
| 2 | 100 | 40 | 2 | 42 | 48 | +6 |
| 3 | 110 | 44 | 2 | 46 | 42 | −4 |
| 4 | 100 | 40 | 2 | 42 | 42 | 0 |

Week 3 exceeds available capacity by 4 hours.
The total four-week load is 160 hours, while total availability is 174 hours.
Thus the aggregate capacity total hides a dated overload.
Summing capacity without checking the week would produce the wrong feasibility conclusion.

### 6.5 Compare options

Assume production can be shifted between weeks without changing the number of weekly setups.
All options below still require verification of material availability and actual staffing.

| Option | Physical response | Incremental cost | Service and risk |
|---|---|---:|---|
| A: pull forward | Move 10 units from week 3 to week 2 | USD 0.77 holding cost under the stated convention | Keeps demand dates if component and week 2 capacity are available |
| B: overtime | Add 4 cell-hours in week 3 | USD 120 premium at USD 30 per hour | Keeps dates if qualified staffing is approved and available |
| C: delay | Make 10 units after week 3 | No expedite assumed; shortage cost not quantified | Violates the policy floor and may affect commitments; customer impact requires order allocation |

For option A, 10 units require 10 × 0.4 = 4 additional run-hours in week 2.
Week 2 load rises from 42 to 46 hours, below 48 hours available.
Week 3 load falls from 46 to 42 hours, equal to available capacity.
The 10 units are held for one additional week at a unit inventory value of USD 40 and an annual carrying rate of 10%.
Using 52 weeks per year, the incremental holding cost is 10 × USD 40 × 0.10 / 52 = USD 0.769..., rounded to USD 0.77.
This calculation is a decision comparison, not a claim that holding cost is the only relevant cost.
It excludes physical handling, quality, obsolescence, working-capital constraints, and any change in setup.

For option B, 4 hours × USD 30 per hour = USD 120 incremental premium.
Base production cost common to both options is omitted.
For option C, ending week 3 stock would be 10 units if production were reduced by 10 units and demand remained unchanged.
That is below the 20-unit policy floor; it is not automatically a missed order because the policy stock is not necessarily allocated to an order.
Whether an accepted order is late depends on dates, allocations, and any order-level service rule.

The master scheduler recommends option A.
The S&OE operations lead approves the shift because the example delegation permits a 10-unit move between nonfrozen weeks within the 380-unit family total and available capacity.
The lead cannot unilaterally change accepted customer promises or approve unbudgeted overtime.
Purchasing confirms the required week 1 component receipt before the schedule is frozen.
If that confirmation fails, option A is not yet feasible and the issue is escalated before the decision deadline.

### 6.6 Revised MPS and reconciliation

| Week | Opening stock, units | Demand, units | Approved receipt, units | Ending stock, units | Cell load, hours |
|---|---:|---:|---:|---:|---:|
| 1 | 40 | 90 | 70 | 20 | 30 |
| 2 | 20 | 100 | 110 | 30 | 46 |
| 3 | 30 | 110 | 100 | 20 | 42 |
| 4 | 20 | 100 | 100 | 20 | 42 |
| Total | — | 400 | 380 | 20 final | 160 |

Week 2 closes at 20 + 110 − 100 = 30 units.
Week 3 closes at 30 + 100 − 110 = 20 units.
The receipt total remains 70 + 110 + 100 + 100 = 380 units.
The family plan and final inventory remain unchanged, but the timing and temporary inventory change.
This is a time-phased reconciliation, not a new demand forecast.

### 6.7 Offset parent orders and explode the component

Assume one week of manufacturing lead time from a P-24 release to its receipt.
The approved P-24 receipts are 70, 110, 100, and 100 units in weeks 1–4.
Their corresponding parent releases are 70 units in week 0, 110 in week 1, 100 in week 2, and 100 in week 3.
The week 0 production order has already been released; it is not a proposal to release an order in the past.
Assume each parent unit requires two identical valve kits at parent release, with no scrap.
The gross valve-kit requirements are 140, 220, 200, and 200 kits in weeks 0–3.

Available valve-kit stock at the start of week 0 is 160 kits.
The component safety stock is zero for this teaching example.
Purchasing uses lot-for-lot quantities, with a one-week lead time and no other receipts.

| Week | Opening available, kits | Gross requirement, kits | Planned receipt, kits | Ending available, kits | New release for next week, kits |
|---|---:|---:|---:|---:|---:|
| 0 | 160 | 140 | 0 | 20 | 200 |
| 1 | 20 | 220 | 200 | 0 | 200 |
| 2 | 0 | 200 | 200 | 0 | 200 |
| 3 | 0 | 200 | 200 | 0 | 0 |

Week 0 ends with 160 − 140 = 20 kits.
Week 1 needs 220 kits and has 20 carried in, so its net requirement is 200 kits.
The 200-kit week 1 receipt requires a release in week 0.
Week 2 and week 3 each need 200 kits, with releases in weeks 1 and 2, respectively.
The release for the week 1 receipt is a proposal until the buyer places it and the supplier confirms it.
The example assumes that confirmation occurs in time; a missed confirmation triggers S&OE, not a fictional on-time receipt.

### 6.8 Schedule and execution handoff

For week 2, the approved 110-unit lot requires 44 run-hours and a 2-hour setup at the test cell.
The plant has 48 staffed cell-hours in that week, leaving 2 hours of modeled slack.
Suppose that time is arranged as five 9.6-hour shifts and that setup can be performed in the first shift.
A simple tentative allocation is 2 setup hours plus 7.6 run-hours in shift 1, then 9.6 run-hours in each of shifts 2–4 and 7.6 run-hours in shift 5.
The run-time total is 7.6 + 9.6 + 9.6 + 9.6 + 7.6 = 44 hours.
This is an illustrative resource allocation, not proof of feasibility for unmodeled labor breaks, maintenance, upstream flow, or unit completion times.
The scheduler must verify those constraints and the component readiness before issuing a dispatch list.

At execution, suppose the first shift loses 1 hour to an unplanned stop.
With the 2 hours of modeled weekly slack, the team may recover without changing the receipt or customer promise if the loss does not create an unmodeled precedence conflict.
The supervisor records the stop and reason; the scheduler assesses the remaining sequence.
If the loss grows beyond slack, the S&OE lead compares recovery options and escalates any commitment or budget breach.
Validated actuals then inform the next MPS and capacity review.

## 7. Code, query, or calculation

The following standalone Python 3 program reproduces the weekly balances, bottleneck loads, component netting, and release offset from the example.
It uses only the standard library.
Save the block as `cascade_example.py` and run `python cascade_example.py`.
No files, packages, or network connection are required.

The script deliberately models one finished item and one component.
It assumes a single component consumption point at parent release, whole-week lead times, lot-for-lot ordering, no component safety stock, and no scrap.
The week 0 parent production order is already released.
The week 0 component purchase release remains a proposal until a buyer and supplier confirm it.
The program checks arithmetic feasibility against weekly capacity; it does not claim a finite schedule or supplier confirmation.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Week:
    number: int
    forecast: int
    orders: int
    receipt: int
    capacity_hours: float


weeks = [
    Week(1, 90, 80, 70, 42),
    Week(2, 100, 95, 110, 48),
    Week(3, 110, 70, 100, 42),
    Week(4, 100, 30, 100, 42),
]

opening_fg = 40
fg_floor = 20
hours_per_unit = 0.4
setup_hours = 2.0
component_per_parent = 2
component_opening = 160
component_lead_weeks = 1

print("week demand receipt ending_fg load_h capacity_h")
fg_balance = opening_fg
for row in weeks:
    # Orders consume forecast in the same bucket; do not add both.
    demand = max(row.forecast, row.orders)
    fg_balance += row.receipt - demand
    load = row.receipt * hours_per_unit + setup_hours
    print(
        f"{row.number:>4} {demand:>6} {row.receipt:>7} "
        f"{fg_balance:>9} {load:>6.1f} {row.capacity_hours:>10.1f}"
    )
    assert fg_balance >= fg_floor, "Finished-goods policy floor violated"
    assert load <= row.capacity_hours, "Weekly bottleneck overload"

# Parent release is one week before the finished receipt.
parent_releases = {row.number - 1: row.receipt for row in weeks}
gross = {
    week: quantity * component_per_parent
    for week, quantity in parent_releases.items()
}

balance = component_opening
planned_receipts = {}
planned_releases = {}
print("week gross_component receipt ending_component")
for week in sorted(gross):
    requirement = gross[week]
    # Zero safety stock and lot-for-lot: order exactly the shortage.
    receipt = max(0, requirement - balance)
    if receipt:
        release_week = week - component_lead_weeks
        if release_week < 0:
            raise ValueError("Past-due component release requires a decision")
        planned_receipts[week] = receipt
        planned_releases[release_week] = receipt
    balance += receipt - requirement
    print(f"{week:>4} {requirement:>15} {receipt:>7} {balance:>16}")

print("component receipts:", sorted(planned_receipts.items()))
print("component releases:", sorted(planned_releases.items()))
print("four-week FG receipts:", sum(row.receipt for row in weeks))
```

Expected output:

```text
week demand receipt ending_fg load_h capacity_h
   1     90      70        20   30.0       42.0
   2    100     110        30   46.0       48.0
   3    110     100        20   42.0       42.0
   4    100     100        20   42.0       42.0
week gross_component receipt ending_component
   0             140       0               20
   1             220     200                0
   2             200     200                0
   3             200     200                0
component receipts: [(1, 200), (2, 200), (3, 200)]
component releases: [(0, 200), (1, 200), (2, 200)]
four-week FG receipts: 380
```

The printed component release pairs are `(week, kits)`.
The receipt pairs have the same convention.
The code raises an error for a component release before week 0 rather than silently treating it as feasible.
In a production system, a past-due release needs an exception workflow and a dated decision.

To adapt the script responsibly, add explicit calendars, order status, effective BOM versions, shared-component aggregation, allocations, and a supplier confirmation state before using the results operationally.
If lot sizing is introduced, carry surplus into subsequent periods instead of ordering the net shortage every week.
If multiple requirements can produce releases in the same period, sum those quantities rather than overwriting the dictionary entry.
A real capacity check also needs setup policy, alternate routings, staffing, maintenance, and operation-level timing.
Those additions change the model's assumptions; they are not implied by the simple calculation.

## 8. Data, standards, and regulatory context

### 8.1 The minimum data contract

Planning requires a consistent picture of both reference data and transactions.
Each data set needs an owner, timestamp, unit, grain, status, and quality rule.
The following contract is illustrative; the physical tables vary by system.

| Data set | Key fields | Why it matters | Control before use |
|---|---|---|---|
| Item master | Item, plant, unit, status, effective dates | Defines what can be planned and how quantities convert | Check active status and valid unit conversion |
| Product-family hierarchy | Item, family, site, valid dates | Connects item MPS to S&OP | Detect unmapped and overlapping membership |
| Demand forecast | Item, location, period, version, quantity | Describes expected independent demand | Retain statistical baseline and override |
| Customer orders | Order line, requested/promised dates, status, quantity | Captures accepted commitments and demand | Separate cancelled, requested, and confirmed lines |
| Inventory | Item, lot, location, status, quantity, timestamp | Defines available supply | Exclude blocked, expired, and allocated stock as policy requires |
| Open supply | Order, item, quantity, due date, confirmation status | Prevents ordering a second time and exposes late supply | Distinguish supplier-confirmed from planned dates |
| BOM | Parent, component, usage, unit, version, validity | Converts parent releases to child requirements | Check effectivity, cycles, substitutions, and unit conversions |
| Routing | Item, operation, resource, setup, run standard, version | Converts quantities into resource load | Check version, standard basis, and alternate eligibility |
| Calendar | Site, resource, shift, downtime, time zone | Defines dated capacity and working time | Reconcile maintenance and staffing calendars |
| Lead-time policy | Item, supplier or process, start/end convention, calendar | Offsets proposed receipts | Compare assumptions with observed and confirmed lead times |
| Yield and scrap | Process, effective period, numerator and denominator | Converts good output to required input | Ensure scrap is not counted twice |
| Plan snapshots | Version, scenario, approver, cutoff, timestamp | Makes decisions reproducible | Protect previous versions from overwrite |
| Execution events | Order, operation, start/finish, good/scrap, reason | Shows what happened and why | Reconcile duplicates, late events, and quality status |
| Financial assumptions | Product, currency, unit value, rate, effective period | Translates physical options to money | Document recognition timing and common-cost treatment |

Stock at 08:00 and orders at 18:00 may describe different worlds.
A version manifest should reveal that mismatch.
If the mismatch is material, refresh the inputs or quantify the exposure before publishing.
A plan generated from internally inconsistent snapshots can look precise while being unreproducible.

### 8.2 Item and location identification

GS1 publishes standards for identifying trade items and locations and for sharing supply-chain data.
Global Trade Item Numbers (GTINs) and Global Location Numbers (GLNs) are examples of identifiers where organizations adopt the relevant standards.
Identifiers improve matching only if master data mappings and ownership are maintained.
They do not determine a company's planning policy, capacity, inventory availability, or forecast consumption rule.
An internal ERP item ID can coexist with a GTIN; the mapping must be versioned and scoped correctly.
Consult the current GS1 specifications and trading-partner agreements before implementing identifiers or messages.

### 8.3 Enterprise and control-system integration

ISA-95 describes models and terminology for integration between enterprise and manufacturing operations.
It helps teams define boundaries and exchanged information between business planning and manufacturing operations.
It does not prescribe a specific S&OP cadence, a particular MRP algorithm, or a guarantee of schedule feasibility.
An MES event such as a production completion must be interpreted alongside quality disposition and inventory transaction status.
A corrected or late-arriving event should retain its event time and processing time when the architecture supports both.

### 8.4 Quality and regulated production

ISO 9001 sets requirements for a quality management system for organizations seeking to conform to that standard.
It is not a universal law requiring every manufacturer to be certified.
In regulated industries, applicable laws, licenses, customer contracts, and quality procedures may impose additional requirements.
For example, U.S. medical-device manufacturers should consult the current FDA Quality Management System Regulation and the requirements applicable to their products and activities.
A planning model may show capacity for a substitute or alternate route, but authorized quality and regulatory controls determine whether that alternative may be used.
This lesson is educational, not legal or regulatory advice.

### 8.5 Time, units, and accounting

Calendar weeks should have explicit start and end dates, including holiday and time-zone conventions.
A one-week lead time can mean seven calendar days, five working days, or one bucket offset; those are not interchangeable.
Item units, purchase units, and production units require controlled conversions.
Currency translation, inventory valuation, and revenue recognition require approved financial assumptions.
A plan that produces an item does not by itself establish revenue recognition or a customer shipment.
Record the assumptions with the scenario rather than presenting a financial conversion as a universal accounting rule.

### 8.6 Data controls that make the loop auditable

1. Validate that every planned end item has an effective family, BOM, routing, and plant calendar.
2. Reconcile opening inventory to the system of record at the stated cutoff.
3. Identify overdue and unconfirmed receipts separately from confirmed future receipts.
4. Retain both original and revised promise dates, with change reason and approver.
5. Check that consumption of forecast by orders follows a documented rule.
6. Preserve the plan version used to make each material and capacity decision.
7. Reconcile planned releases to actual purchase and production orders by status.
8. Capture quality holds and blocked stock before treating quantities as available.
9. Compare reported execution quantities with inventory movements and investigate differences.
10. Assign correction ownership and a deadline for each material data defect.

A data validation rule is not a substitute for a business decision.
For example, an overdue component receipt requires confirmation and a response, not just a red cell in a dashboard.

## 9. Industry practice and case evidence

### 9.1 What public cases can and cannot show

Companies rarely publish their complete S&OP decision logs, exact planning fences, or live MRP parameters.
A public annual report can establish a disclosed exposure or action without proving the internal process that produced it.
The examples below therefore separate public evidence from teaching inference.
They illustrate why planning interfaces matter; they do not validate Northstar Pumps' fictitious numbers.

### 9.2 Toyota and the 2011 earthquake

Toyota publicly reported the impact of the March 2011 Great East Japan Earthquake on its operations and parts supply.
The disruption illustrates how dependencies in a multi-tier supply chain can affect production even when an assembly plant itself is available.
Toyota's newsroom and annual reporting provide primary material on the event and subsequent business context; see Sources.

**Documented fact:** the earthquake and associated disruption affected production and the supply of parts.
**Inference for this lesson:** an aggregate vehicle production plan could not be treated as executable without updated component availability and plant-level decisions.
**Inference for this lesson:** an escalation process linking supply constraints, production revisions, and customer communication would be valuable.
The public fact alone does not establish the precise meeting cadence or software logic Toyota used at each plant.

A useful discussion question is where the first trustworthy evidence of the disruption would enter the cascade.
A tier-two supplier interruption may reach the company through a tier-one supplier, a missed shipment, or a demand-supply review.
Its arrival time and confidence affect whether the buyer can expedite, the scheduler can resequence, or an executive must allocate scarce output.
The lesson is to distinguish a plausible response from a documented company action.

### 9.3 Apple and disclosed supply constraints

Apple's fiscal 2024 Form 10-K discusses reliance on outsourcing partners and supply-chain risks, including concentration and potential disruptions.
The filing also describes inventory and supply management as significant operational considerations.
The filing is a company disclosure, not an independent audit of a particular S&OP workflow.

**Documented fact:** Apple identifies supply-chain dependencies and disruption risks in its filing.
**Inference for this lesson:** a business with concentrated sourcing needs a process to connect supplier constraints with product availability and commercial decisions.
**Not established by the filing alone:** the firm's internal frozen window, exact allocation rule, or a quantified benefit from a specific planning cadence.
A public risk statement is evidence of exposure, not evidence that every mitigation worked.

### 9.4 Common operating patterns

In a make-to-stock plant, the MPS often plans finished-item replenishment against a forecast and inventory policy.
In a make-to-order plant, customer orders and confirmed dates may dominate the near-term MPS.
An assemble-to-order business may plan common components ahead of final configuration.
Each pattern changes what can be promised and where uncertainty is held.
None removes the need to check materials, capacity, and authority.

A company with long imported-component lead times may approve a rolling material outlook many months ahead while keeping a much shorter finite schedule.
A process plant with long campaigns may choose larger lots to avoid costly cleaning or changeovers.
A contract manufacturer may need customer approval before changing a production site or material source.
These are operating patterns, not claims that every firm in a sector follows the same rule.

### 9.5 Questions to ask before transferring a case

- Does the public source describe actual behavior, a management intention, or a vendor's marketing claim?
- Is the reported result a measured causal effect or a before-and-after association?
- Are the production process, lead times, demand volatility, and supplier dependencies comparable?
- Was the action technically feasible because of capacity, qualification, and available materials?
- Were service, cost, inventory, and quality measured over the same period?
- Are the customer and financial outcomes independently observable?
- What was the counterfactual alternative, and is it documented?
- Which part of the conclusion is inference rather than an explicit fact?

A credible case is most useful when it sharpens a decision question.
It should not be turned into a fabricated return on investment or a universal process prescription.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Forecast treated as order | Forecast plus all orders exceeds both independently | The same demand is counted twice | Excess material and inventory | Define and test forecast-consumption logic |
| Target treated as forecast | Overrides always equal sales targets | Desired outcome replaces evidence | Capacity and procurement commitments exceed likely demand | Preserve baseline, override rationale, and target separately |
| Aggregate mix hidden | Family units fit but bottleneck hours vary sharply | Item mix changes resource consumption | Family plan cannot be disaggregated | Track mix and key-resource hours explicitly |
| Monthly bucket hides weekly peak | Monthly capacity is below load only in individual weeks | Timing mismatch is averaged away | Late production despite apparent monthly slack | Compare at the tightest material or capacity decision bucket |
| Opening stock overstated | Frequent inventory corrections or blocked lots | Unavailable stock is counted as usable | Missing components at release | Reconcile stock by status and cutoff |
| Unconfirmed receipt treated as firm | Supplier date differs from planning date | A proposal is mistaken for a commitment | Schedule waits for absent material | Capture confirmation status and escalate at deadline |
| Wrong BOM version | Component usage differs from shop-floor issues | Current or wrong version is exploded | Shortages, excess, or invalid material | Control effective-dated production versions |
| Wrong time unit | Releases consistently late around holidays | Calendar and working-day offsets differ | Replenishment arrives after need | State and validate lead-time calendar |
| Setup omitted | Utilization appears low but changeovers consume shifts | Run-hour-only model understates load | Finite sequence overruns capacity | Include setup by lot or sequence as appropriate |
| Setup counted per unit | Calculated load is implausibly high for large lots | One-time setup is multiplied by output quantity | Unnecessary overtime or missed opportunity | Separate setup hours per lot from run hours per unit |
| Quality release ignored | Completed quantities accumulate in inspection | Output is assumed saleable at completion | Customer promise uses unavailable stock | Model inspection and release status |
| Frozen window routinely overridden | High share of orders change at short notice | Decision rights or upstream plan are ineffective | Expediting, scrap, unstable schedule | Classify reasons and enforce exception authority |
| Freeze treated as absolute | Teams hide safety or quality changes to protect a metric | Stability incentive overrides real constraints | Unsafe or unreported work | Allow authorized exceptions and distinguish their causes |
| Excess alerts | Hundreds of unchanged exceptions recur daily | No thresholds, ownership, or deduplication | Critical shortages are missed | Prioritize by impact, due time, and decision owner |
| Inconsistent snapshots | Reports disagree about the same order | Different cutoffs or version overwrites | Meetings debate numbers instead of options | Publish a shared input manifest and immutable baseline |
| Schedule bypass | Operators use a private spreadsheet sequence | Published schedule is stale or impractical | Lost traceability and conflicting priorities | Establish controlled dispatch and rapid feedback |
| Local optimization | Utilization rises while late orders and WIP rise | Non-bottleneck output exceeds downstream capacity | More inventory without more throughput | Measure flow and bottleneck performance, not utilization alone |
| Silent promise change | Order dates move without customer acknowledgment | Internal replan is mistaken for accepted commitment | Service breach and trust loss | Require authorized customer communication and confirmation |
| Capacity response not funded | Plan assumes overtime but staffing approval is absent | An option is published as if authorized | Missed schedule and cost surprise | Record approver, hours, cost, and labor availability |
| No closed loop | Same shortage appears every cycle | Variance is reported but master data or policy is unchanged | Repeated expediting | Assign root-cause action and effective date |
| Financial double count | Savings appear in both price and margin assumptions | Multiple views use inconsistent cost bases | Misleading option comparison | Reconcile bridge and identify incremental versus common costs |
| Model called finite without scope | Schedule fits machines but not qualified labor | A constraint is omitted from the model | Work is impossible despite a solver result | Publish modeled and unmodeled constraints |

Investigate red flags with evidence rather than assuming that a specific software module caused them.
A late order can result from demand, data, physical failure, or a deliberate trade-off.
The response should identify the mechanism and the authorized decision.

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | How to test or respond |
|---|---|---|---|
| One item represents the family | Makes aggregate-to-item reconciliation visible | Multiple variants consume different resources | Reconcile each item and use mix-weighted capacity |
| Demand uses same-week maximum of forecast and orders | Prevents simple double counting | Orders consume forecast across periods or channels | Compare to documented consumption and test boundary cases |
| Demand is met within each weekly bucket | Allows a weekly inventory balance | Orders are due before receipts within the week | Use daily or event-time balances and allocations |
| Opening stock is usable | Allows netting against demand | Quality hold, reservation, expiry, or location restriction | Reconcile by status, ownership, and expiry |
| No scrap or yield loss | Keeps component and output arithmetic direct | Defects or process loss reduce good output | Estimate effective yield and apply it once at the correct operation |
| One parent release precedes receipt by one week | Gives a simple date offset | Queue time, split lots, and operation dates vary | Compare actual release-to-receipt lead times and calendars |
| Components are consumed at parent release | Gives a clear dependent-demand date | Components are issued at later operations | Use operation-specific consumption dates |
| Component lot-for-lot and zero safety stock | Makes gross-to-net calculation transparent | Minimum order quantity, pack size, or policy buffer | Apply the actual lot rule and carry excess stock forward |
| Supplier can confirm the week 1 receipt | Makes the pull-forward plan potentially executable | Capacity, transit, or cutoff prevents confirmation | Get a dated supplier commitment; evaluate alternatives if absent |
| One weekly lot needs one 2-hour setup | Supports a comparable capacity check | Multiple lots, changeovers, or shared setup resources | Build an operation and sequence-specific schedule |
| Test-cell hours are the binding modeled capacity | Focuses the example on one resource | Labor, upstream machines, storage, or transport bind first | Check other resources and material readiness |
| Week 2 has 48 authorized hours | Allows a 10-unit pull-forward | Staff, maintenance, or shift authorization changes | Confirm the resource calendar and named staffing |
| Unit inventory value is USD 40 | Converts one-week inventory into a cost estimate | Valuation differs by lot or accounting basis | Recalculate using approved incremental carrying assumptions |
| Ten-percent annual carrying rate applies linearly | Provides a transparent option comparison | Nonlinear storage, expiry, financing, or risk costs matter | Add decision-relevant costs and show sensitivity |
| One-week extra holding is acceptable | Makes the earlier build a candidate | Shelf life, engineering change, or customer specification prevents it | Verify quality, shelf-life, storage, and revision constraints |
| Local authority permits the 10-unit shift | Explains why S&OE can approve the response | Frozen window or contract requires higher approval | Check actual delegation and escalate if required |
| Confirmed customer orders are included in the forecast | Supports the max rule in the example | Forecast excludes a customer/channel or orders are incremental | Segment demand and apply appropriate consumption rules |
| Execution events are accurate and timely | Makes feedback meaningful | Missing, duplicate, or late confirmations | Reconcile event records to physical and inventory movements |

These assumptions are not recommendations for a universal operating model.
They are conditions under which the example's arithmetic and decision are interpretable.
When an assumption fails, the correct response is to change the model or seek a decision, not conceal the variance.

## 12. Exercises

Work the calculations before consulting Section 13.
Use units and dates in every answer.
The exercises refer to the fictional case unless stated otherwise.

### Exercise 1 — Map the decisions

**Time:** 15 minutes.

A company must decide whether to invest in an extra line next year, place a purchase order for a valve kit, sequence tomorrow's test jobs, and respond to an unexpected late supplier delivery this week.
Map each decision to an appropriate forum or role and a reasonable horizon.
Name one escalation condition for the late delivery.

**Deliverable:** a four-row decision table with decision, forum/role, horizon, and escalation.

**Acceptance criteria:** distinguish aggregate capacity authorization from order placement and detailed sequencing; assign the late delivery to a near-term exception process; make authority conditional on local delegation.

### Exercise 2 — Separate four business quantities

**Time:** 15 minutes.

A weekly forecast is 140 units, accepted orders are 125 units already included in that forecast, management's target is 160 units, and the approved production receipt is 135 units.
The organization uses same-week maximum forecast consumption.

**Deliverable:** identify the forecast, target, commitment, plan, and planning demand; explain why they are not summed.

**Acceptance criteria:** show 140 units of planning demand, 125 units of commitments, 160 units of target, and 135 units of planned production; identify that forecast and target do not guarantee sales.

### Exercise 3 — Reconcile the MPS

**Time:** 20 minutes.

Opening finished-goods stock is 30 units, demand is 95 units, the safety floor is 20 units, and the planned receipt is 80 units.
Calculate the ending balance and the minimum additional receipt needed to reach the floor, assuming lot-for-lot and receipt before demand.
State whether adding that receipt is authorized by the arithmetic alone.

**Deliverable:** a balance equation, shortage, and decision note.

**Acceptance criteria:** compute 15 units ending stock and a 5-unit shortfall to the floor; distinguish a proposed 5-unit increase from an approved and feasible production change.

### Exercise 4 — Explode and offset

**Time:** 20 minutes.

A parent order releases 75 units in week 5.
Each parent consumes three component kits at release.
At the start of week 5, 45 kits are available and no other receipt is scheduled.
The component policy has zero safety stock, lot-for-lot supply, and a two-week lead time.
Assume week 3 is not in the past.

**Deliverable:** gross requirement, net requirement, planned receipt week and quantity, and planned release week and quantity.

**Acceptance criteria:** compute 225 kits gross and 180 kits net; place the 180-kit receipt in week 5 and proposed release in week 3; identify that the release is not supplier confirmation.

### Exercise 5 — Detect a hidden capacity gap

**Time:** 20 minutes.

Two weeks each have 40 available machine-hours.
A planned lot needs 92 units in week 1 and 58 units in week 2.
Each unit requires 0.5 machine-hours and each week's lot needs a 2-hour setup.
Demand and material availability allow up to 20 units to move from week 1 to week 2.

**Deliverable:** original weekly loads, a feasible shift quantity if one exists, revised loads, and one remaining feasibility check.

**Acceptance criteria:** compute 48 hours and 31 hours initially; move at least 16 and at most 18 units to obtain loads no greater than 40 hours, assuming unchanged setup count; mention materials, labor, or other constraints.

### Exercise 6 — Recommend an exception response

**Time:** 25 minutes.

In the Northstar case, the supplier refuses the week 1 component receipt and can deliver only in week 2.
No approved alternate component exists.
The next customer order is due in week 2.
You do not know the exact intraweek order dates or whether a qualified supplier expedite is available.

**Deliverable:** a one-page S&OE decision brief with at least two options, decision owner, missing evidence, deadline, and communication actions.

**Acceptance criteria:** do not claim the earlier production is feasible; compare at least an expedite if confirmed and a production/promise replan; identify quality approval for any proposed alternate and escalate any commitment change.

### Exercise 7 — Design a planning cadence

**Time:** 30 minutes.

Design a process for a manufacturer with a 20-week critical-component lead time, a two-week frozen production window, and a monthly executive review.
Demand is reviewed weekly and shop-floor operations run daily.

**Deliverable:** a diagram or table covering S&OP or IBP, S&OE, MPS, MRP, finite scheduling, and execution; include horizon, cadence, inputs, output, decision owner, and feedback.

**Acceptance criteria:** the material horizon covers the critical lead time; the frozen window has a named exception authority; the design reconciles item and family views; execution actuals return to planning; decisions and versions are recorded.

### Exercise 8 — Measure plan stability without hiding risk

**Time:** 20 minutes.

A four-week frozen-window baseline has 100, 120, 80, and 100 planned units.
The comparable final plan has 100, 110, 90, and 100 units.
The second week's decrease and third week's increase were an approved response to a safety stop.

**Deliverable:** calculate the sum of absolute changes divided by baseline units, identify the two changed buckets, and propose a companion measure or reason-code split.

**Acceptance criteria:** report 20 changed units / 400 baseline units = 5%; do not net the changes to zero; retain the approved safety reason and avoid penalizing necessary safety action.

## 13. Self-check answers

### Exercise 1 — Decision map

| Decision | Example owner or forum | Illustrative horizon | Important boundary |
|---|---|---|---|
| Extra line | Executive capacity or capital forum informed by S&OP/IBP | Next year and beyond | Capital approval follows actual delegation |
| Valve-kit order | Buyer/materials planner using MRP output | Through supplier lead time | Supplier acceptance is separate from an internal proposal |
| Tomorrow's sequence | Production scheduler and supervisor | Next day or shift | Safety, material, labor, and resource constraints apply |
| Late delivery | S&OE lead with purchasing and production | Current days or weeks | Escalate an at-risk customer promise or unauthorized expenditure |

Different organizations may combine roles.
The test is whether a decision has an accountable owner with sufficient authority and a timely escalation route.

### Exercise 2 — Demand and commitment

The forecast is 140 units.
The accepted-order commitment is 125 units.
The management target is 160 units.
The approved production receipt is 135 units.
Under the stated rule, planning demand is max(140, 125) = 140 units.
Adding 140 forecast units and 125 order units would count the orders twice because they are already included in the forecast.
The 160-unit target represents desired performance, not an additional 160-unit demand event.

### Exercise 3 — Finished-goods balance

Ending stock before a change is 30 + 80 − 95 = 15 units.
The policy floor is 20 units, so the shortfall to the floor is 20 − 15 = 5 units.
A 5-unit additional receipt would restore the floor under lot-for-lot assumptions.
It remains a proposal until capacity, material, timing, and decision authority are checked.
A positive 15-unit balance is not the same as meeting the 20-unit policy.

### Exercise 4 — Component calculation

Gross requirement is 75 parent units × 3 kits per parent = 225 kits in week 5.
Net requirement is 225 − 45 = 180 kits.
With lot-for-lot supply and zero safety stock, plan a 180-kit receipt in week 5.
A two-week offset gives a proposed 180-kit release in week 3.
The buyer must place the order and obtain confirmation; a calculated release is not a commitment.
If week 3 were already past, the calculation would identify an exception requiring a decision.

### Exercise 5 — Capacity reconciliation

Week 1 load is 92 × 0.5 + 2 = 48 hours.
Week 2 load is 58 × 0.5 + 2 = 31 hours.
To remove 8 hours from week 1, move at least 8 / 0.5 = 16 units.
Week 2 has 9 hours of slack, so it can receive at most 9 / 0.5 = 18 units.
Any integer shift from 16 to 18 units meets the stated capacity limits.
For a 16-unit shift, revised loads are 40 hours in week 1 and 39 hours in week 2.
The demand and material assumption permits the shift, but qualified labor, other machines, and detailed timing still need checking.

### Exercise 6 — Decision-brief rubric

A strong brief says the original pull-forward option is unsupported because the required component is not confirmed on time.
It asks the buyer for a documented expedite quote and earliest feasible delivery, including the decision deadline.
It also shows a replan with affected order dates and obtains customer-facing approval before changing a promise.
It does not assume a substitute is allowed; quality and other applicable approval must precede its use.
It identifies who may approve cost, who communicates with the customer, and when the decision must be made.
It distinguishes confirmed information from unresolved possibilities.

A weak brief silently moves the supplier date in the system, treats an unqualified alternate as available, or changes the customer promise without authority.

### Exercise 7 — Cadence-design rubric

A strong design includes a monthly executive family and financial reconciliation and a weekly or more frequent near-term exception forum.
It publishes an item-week MPS, component plan covering at least the 20-week lead time, and a shorter finite schedule appropriate to resource certainty.
It assigns authority for changes inside the two-week frozen window.
It defines a versioned baseline, cutoff, reconciliation of family and item totals, and a path for validated execution data to change future assumptions.
It names who owns customer commitments and supplier confirmations.
A shorter MRP horizon would miss the release decision even if the daily schedule looked feasible.

### Exercise 8 — Stability calculation

The absolute changes are 0, 10, 10, and 0 units.
Their sum is 20 units; the baseline total is 400 units.
The stated metric is 20 / 400 = 5%.
Weeks 2 and 3 changed, even though the net total change is zero.
Report the safety-stop reason separately and track recovery or service impact alongside stability.
The metric is a change magnitude relative to baseline, not a percentage of unique items or a service-level measure.

## 14. Interview questions

### 1. How does S&OP differ from an MPS?

A strong answer identifies aggregate, cross-functional demand-supply decisions in S&OP and dated end-item production decisions in the MPS.
It explains that the two must reconcile and that actual horizons depend on the business.
A weak answer treats the S&OP total and the item schedule as interchangeable.

### 2. Is IBP simply another name for S&OP?

A strong answer notes that terminology varies and asks what financial, strategic, and decision integration actually occurs.
It evaluates the process by its decisions, ownership, and evidence rather than its label.
A weak answer assumes a universal definition or claims that a renamed meeting is automatically integrated.

### 3. What is S&OE allowed to decide?

A strong answer describes near-term exception decisions within documented cost, service, safety, and commitment guardrails.
It gives an example of an issue that must be escalated.
A weak answer gives S&OE unlimited authority or treats it as a duplicate monthly executive review.

### 4. Why can a monthly production plan be feasible in total and infeasible in practice?

A strong answer points to weekly or daily capacity peaks, product mix, material dates, setup, and other local constraints.
It offers a time-phased load check rather than merely adding monthly hours.
A weak answer assumes aggregate slack can always be moved across weeks without cost or material consequences.

### 5. When do customer orders consume a forecast?

A strong answer states that the policy determines the relationship, including item, channel, time window, and treatment of incremental orders.
It avoids double counting and retains separate forecast and order records.
A weak answer always adds the two or always assumes a single universal consumption formula.

### 6. What does a past-due MRP release mean?

A strong answer says the required release date precedes the planning date under current assumptions.
It proposes checking data, supplier confirmation, expedite, replanning, and escalation.
A weak answer silently changes the release date or claims that MRP has already purchased the material.

### 7. What does a frozen window protect, and what can override it?

A strong answer describes protection against avoidable short-term churn and names an authorized exception process.
It recognizes safety, quality, and critical customer consequences.
A weak answer treats the freeze as either optional for everyone or impossible to override under any circumstance.

### 8. What makes a finite schedule credible?

A strong answer identifies resource calendars, operation precedence, material readiness, setups, and the constraints actually modeled.
It distinguishes model feasibility from physical certainty and requires execution feedback.
A weak answer assumes a solver output automatically includes every real constraint.

### 9. How do you measure MPS nervousness?

A strong answer compares versioned snapshots at a consistent lead time and grain, reports a clear denominator, and separates justified changes from unexplained churn.
It considers the incentive not to hide safety or service exceptions.
A weak answer compares incomparable snapshots or reports only net volume changes.

### 10. What information should accompany an escalation?

A strong answer names the decision owner, deadline, baseline, alternatives, impacts, assumptions, and affected commitments.
It records the chosen response and downstream communication owners.
A weak answer forwards a red dashboard alert without a decision request or options.

## 15. Key terms

The links point to the definitions in this topic so that terms remain usable when read out of order.

- [Aggregate plan](#4-definitions-and-vocabulary) — family-level intended actions over a time horizon.
- [Available-to-promise](#4-definitions-and-vocabulary) — policy-specific uncommitted supply available for promises.
- [Baseline](#4-definitions-and-vocabulary) — selected version used as the comparison reference.
- [Bill of materials](#4-definitions-and-vocabulary) — effective-dated parent-component usage structure.
- [Capacity requirements planning](#4-definitions-and-vocabulary) — comparison of dated order load to resource availability.
- [Commitment](#4-definitions-and-vocabulary) — authorized promise with quantity and timing.
- [Controlled-change window](#4-definitions-and-vocabulary) — period where changes require compliance with specified rules.
- [Decision right](#4-definitions-and-vocabulary) — authority to approve a specified class of decision.
- [Demand forecast](#4-definitions-and-vocabulary) — estimate of future demand.
- [Demand plan](#4-definitions-and-vocabulary) — versioned operational demand view and assumptions.
- [Disaggregation](#4-definitions-and-vocabulary) — allocation from an aggregate plan to detailed entities and periods.
- [Dispatch list](#4-definitions-and-vocabulary) — controlled list of operations to execute.
- [Escalation](#4-definitions-and-vocabulary) — routing a decision to an owner with sufficient authority.
- [Execution confirmation](#4-definitions-and-vocabulary) — recorded actual production event or quantity.
- [Finite schedule](#4-definitions-and-vocabulary) — time-resource assignments that respect modeled constraints.
- [Frozen window](#4-definitions-and-vocabulary) — near-term period with restricted change authority.
- [Gross requirement](#4-definitions-and-vocabulary) — requirement before supply netting.
- [IBP](#4-definitions-and-vocabulary) — integrated business planning linking operating and financial choices.
- [Lead time](#4-definitions-and-vocabulary) — elapsed or working time between a release and receipt.
- [Master production schedule](#4-definitions-and-vocabulary) — authorized time-phased end-item production plan.
- [Material requirements planning](#4-definitions-and-vocabulary) — time-phased component netting and order proposal logic.
- [Net requirement](#4-definitions-and-vocabulary) — incremental supply needed after netting and policy.
- [Pegging](#4-definitions-and-vocabulary) — trace from supply or a requirement to demand drivers.
- [Plan nervousness](#4-definitions-and-vocabulary) — frequent changes to dates or quantities.
- [Planned order release](#4-definitions-and-vocabulary) — proposed start or purchase date.
- [Projected available balance](#4-definitions-and-vocabulary) — expected remaining usable stock at a stated time.
- [Reconciliation](#4-definitions-and-vocabulary) — checking and explaining differences across planning views.
- [Rough-cut capacity planning](#4-definitions-and-vocabulary) — approximate MPS load check on key resources.
- [Safety stock](#4-definitions-and-vocabulary) — policy buffer against uncertainty.
- [Sales and operations execution](#4-definitions-and-vocabulary) — near-term management of execution exceptions.
- [Sales and operations planning](#4-definitions-and-vocabulary) — cross-functional aggregate demand and supply reconciliation.
- [Scenario](#4-definitions-and-vocabulary) — consistent alternative set of assumptions and outcomes.
- [Scheduled receipt](#4-definitions-and-vocabulary) — existing order expected to arrive or complete on a date.
- [Time bucket](#4-definitions-and-vocabulary) — interval in which quantities are grouped.
- [Work center](#4-definitions-and-vocabulary) — capacity-owning resource or resource group.

## 16. Sources

The sources below support terminology, standards context, and the public case descriptions.
The fictional case, exercises, and calculations are original teaching examples.
URLs are provided so that readers can consult the current source and its scope.

1. [ASCM Dictionary](https://www.ascm.org/learning-development/ascm-dictionary/), Association for Supply Chain Management.
   Reference for production and inventory management terminology; specific organizations may configure processes differently.
2. [ISA-95 Standard: Enterprise-Control System Integration](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard), International Society of Automation.
   Describes the standard family and enterprise/manufacturing integration scope.
3. [GS1 Global Standards](https://www.gs1.org/standards), GS1.
   Entry point for identification, data capture, and sharing standards.
4. [ISO 9001: Quality management systems](https://www.iso.org/standard/62085.html), International Organization for Standardization.
   Standard overview; consult the applicable current edition and organizational requirements.
5. [Quality Management System Regulation (QMSR)](https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr), U.S. Food and Drug Administration.
   Regulatory context for applicable U.S. medical-device activities; not a general manufacturing mandate.
6. [Toyota Annual Reports](https://global.toyota/en/ir/library/annual/), Toyota Motor Corporation.
   Primary company reporting around the 2011 earthquake and its business effects; distinguish disclosed facts from process inference.
7. [Toyota Global Newsroom](https://global.toyota/en/newsroom/), Toyota Motor Corporation.
   Primary company announcements and event context; search the archive for 2011 earthquake and production updates.
8. [Apple Inc. Form 10-K, fiscal year ended September 28, 2024](https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm), U.S. Securities and Exchange Commission filing.
   Primary disclosure of business operations and supply-chain risk; it does not disclose a complete internal planning workflow.
9. Vollmann, Thomas E., William L. Berry, D. Clay Whybark, and F. Robert Jacobs, *Manufacturing Planning and Control for Supply Chain Management*, 6th ed., McGraw-Hill, 2011.
   Textbook treatment of planning hierarchy, master scheduling, material requirements, and capacity.
10. Wallace, Thomas F., and Robert A. Stahl, *Sales & Operations Planning: The How-To Handbook*, 3rd ed., T. F. Wallace & Company, 2008.
    Practitioner treatment of S&OP process and management decisions; apply its proposed cadence to the context rather than as a regulatory rule.

## 17. Links

- [Module 15 topic index](_index.md) — sequence and planned topic scope.
- [Module 15 survey](../../../course/15_manufacturing_production_planning.md) — broader production-planning overview and lab context.
- [Manufacturing master data](02_manufacturing-master-data.md) — item, BOM, routing, and calendar controls.
- [Master production scheduling](03_master-production-scheduling.md) — detailed end-item scheduling and reconciliation.
- [Planning processes](../../01_foundations/01_supply_chain_foundations/04_planning-processes.md) — related foundational topic on planning decisions.
- [Manufacturing basics](../../01_foundations/01_supply_chain_foundations/07_manufacturing-basics.md) — related foundational topic on manufacturing flows.
- [Simulation and optimization production labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md) — lab-module map.

