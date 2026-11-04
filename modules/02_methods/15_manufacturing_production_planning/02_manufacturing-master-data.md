# The Manufacturing Master Data Model

> Module 15 — Manufacturing and Production Planning  
> Topic 02 of 11  
> Estimated study time: 4 hours  
> Prerequisites: Module 01, Module 10; [Module 15 index](_index.md)  
> Version 2026.5

## 1. Why this topic exists

A production plan converts a promise about finished goods into a sequence of material and capacity decisions.
The conversion depends on records that look administrative but encode physical facts and management choices.
An item record identifies what is being made or bought and how it is measured.
A bill of materials (BOM) says what components the parent consumes.
A routing says which operations transform those components and how much resource time they need.
A work center describes the resource whose capacity those operations consume.
A calendar says when that resource is available.
A production version, or equivalent approved selection rule, connects a valid BOM and routing to a plant, date, and production context.

If these records disagree, the plan can be mathematically consistent and physically impossible.
For example, a valid BOM may use a substitute motor while the chosen routing requires a fixture that only fits the previous motor.
The material planner can order all the components and still stop the line.
Conversely, a routing may show enough hours while the calendar omits planned maintenance.
The resulting available-to-promise date can be attractive but unsupported.
The important question is not whether the fields are populated; it is whether the connected records describe an approved way to make the item when and where the order will run.

Different people bear different costs of a defect.
A buyer expedites a component that the BOM failed to demand.
A supervisor works overtime because a setup time was stored in minutes but interpreted as hours per hundred units.
A quality engineer investigates a lot made to the wrong engineering revision.
A controller sees a variance because a standard time no longer reflects the process.
A customer receives a delivery promise based on a machine that was scheduled for overhaul.
None of these people can repair the whole problem by editing a spreadsheet after planning has run.

Master data are not identical to a plan or an actual transaction.
They are governed inputs to many plans and transactions, often shared by multiple applications.
An inventory balance is a time-specific state, not the item's identity.
An actual operation confirmation is evidence about execution, not automatically the new standard time.
A demand forecast estimates future demand; a production plan chooses intended output; a target expresses a desired performance level; a commitment is an accepted obligation.
Only an authorized decision can turn a proposed plan into a commitment.

This lesson gives you a way to inspect the relationships, timing, units, and authority behind a manufacturing dataset.
The example is a fictional pump assembly; the numbers illustrate a decision and are not evidence about a real plant.
The calculation is deliberately small enough to check by hand.
A production-grade material requirements planning (MRP) system needs additional rules for netting, alternatives, inventory, and order execution.

## 2. Learning outcomes

After this lesson you can:

1. **Map** item, BOM, routing, work-center, calendar, and version records at their correct keys and grains.
   Exercise 1 asks you to identify the relationships and the owner of each record.
2. **Select** an approved, date-effective BOM-routing combination for a plant and order context.
   Exercise 2 tests effectivity boundaries and ambiguous alternatives.
3. **Calculate** component requirements and operation load with explicit units, batch setup, and yield assumptions.
   Exercise 3 reproduces the worked arithmetic.
4. **Reconcile** a dated resource calendar with load without double counting downtime or performance loss.
   Exercise 4 asks you to calculate and interpret a capacity gap.
5. **Diagnose** referential, temporal, unit, and semantic defects and explain their business consequences.
   Exercise 5 asks you to prioritize incidents and contain their effects.
6. **Design** a controlled change from engineering request through approval, publication, and post-release evidence.
   Exercise 6 requires a control and decision record.
7. **Critique** a historical analysis that applies today's master data to old production orders.
   Exercise 7 asks you to distinguish transaction-time and valid-time evidence.
8. **Specify** a practical quality dashboard with denominators, owners, and thresholds.
   Exercise 8 asks you to design measures and escalation rules.

A successful learner can explain not only which number is wrong but also which decision should be paused and who can authorize the correction.

## 3. The core mental model

Treat master data as a versioned graph with a date and a plant attached to the question.
The graph is not merely a collection of separate lookup tables.
An order can use only a connected path whose nodes and edges are valid and compatible for its context.
A path can be incomplete even when every individual table has rows.

```text
 customer demand / forecast          engineering change approval
             |                                    |
             v                                    v
      proposed production order ------> select approved production version
      item + plant + dates + quantity                |
                                          +----------+----------+
                                          |                     |
                                          v                     v
                                BOM revision + lines      routing revision
                                          |                     |
                                          v                     v
                                  component items       operations + times
                                                                |
                                                                v
                                                      work-center resource
                                                                |
                                                                v
                                                     dated shift calendar
                                          |                     |
                                          v                     v
                                  gross component need    capacity load vs supply
                                          \                     /
                                           \                   /
                                            v                 v
                                              planner decision
                                                     |
                                                     v
                                       release / hold / revise promise
                                                     |
                                                     v
                                   execution evidence and controlled feedback
```

The arrow from an order to a production version represents a decision rule, not a universal database field name.
One enterprise may store an explicit production version; another may use approved recipe, routing, and allocation records.
Either implementation must identify the permitted combination and explain how a planner chooses among alternatives.
The BOM and routing must be selected as a compatible pair, not independently by taking the largest revision number in each table.
A revision label is an identifier; it does not by itself tell you whether the revision applies to the order.

The physical chain and the information chain answer different questions.
The physical chain consumes components, uses machine and labor time, and produces acceptable units.
The information chain describes those transformations and makes them available to planning and execution applications.
Both chains must refer to the same plant, product configuration, process, and effective period.
A record can be syntactically complete and still describe the wrong physical chain.
A valid item code does not establish that the item fits the assembly.
A valid work-center code does not establish that its calendar includes the intended shift.

Use three time concepts explicitly.
**Valid time** is when a record is supposed to apply to production, such as 1 July through 30 September.
**Transaction time** is when the system recorded the fact or a correction.
**Decision time** is when a planner made a plan with the information then available.
A backdated correction can change today's reconstruction of valid time without changing what the planner could have known at the earlier decision time.
Order-specific snapshots or retained change history are needed for a defensible comparison.

Choose a consistent interval convention.
This lesson uses half-open validity intervals `[start, end)`: the start date is included and the end date is excluded.
A version ending on 1 October does not apply on 1 October; an adjacent version beginning that day can apply without overlap.
Real enterprise resource planning (ERP) products may use inclusive end dates or timestamps instead.
Translate their semantics rather than silently imposing this lesson's convention on the source system.

A plant is an essential scope, not a display attribute.
One item can have different approved components, lead times, work centers, and calendars at two plants.
The selection may also depend on order quantity, configuration, customer, lot, or regulatory status.
The correct selection date can be an order release, operation start, production start, or other date according to the local process and system.
State it explicitly and carry it through the analysis.
If more than one valid option remains, a documented priority or allocation decision is required.
Multiple valid alternatives are not necessarily a defect; an unexplained or prohibited overlap is.

Finally, distinguish measurement from control.
A data-quality rule can identify suspicious records, but it cannot determine engineering suitability from a join alone.
The record owner must investigate, approve a correction, assess affected orders, and communicate the operational decision.
A low exception count is useful only if the denominator, scope, and test logic are visible.

## 4. Definitions and vocabulary

The table defines terms as used in this lesson.
Actual field names and configuration vary across systems.

| Term | Meaning and grain | Common misuse | Unit or code |
|---|---|---|---|
| Item | Identified material, intermediate, or finished good; item identity is distinct from its plant settings. | Treating a description as a unique key. | Item ID; base unit. |
| Plant item | Planning and execution settings for an item at one plant. | Assuming all plants share the same lead time. | Item ID × plant ID. |
| Base unit | Unit used for core inventory or planning calculations under the local configuration. | Multiplying kilograms by pieces without conversion. | Each, kg, m, or another named unit. |
| Unit conversion | A stated relationship between units for a specific item and context. | Assuming every package contains the same count. | For example, 12 each/case. |
| Make/buy | Sourcing or replenishment classification for a specified context. | Assuming it cannot differ by plant. | Controlled code. |
| Lead time | Modeled elapsed or working time from a defined start to a defined end. | Treating it as pure machine run time. | Days, hours, and calendar basis. |
| Lot-size rule | Rule that translates a net need into an order quantity. | Calling a minimum quantity a fixed multiple. | Each, kg, or other order unit. |
| Safety stock | Inventory policy buffer for uncertainty or service. | Treating it as physical on-hand inventory. | Item base unit. |
| BOM header | Identifies a product structure, alternative, status, and context. | Treating the newest header as automatically approved. | Parent × plant × alternative × revision. |
| BOM line | One component relationship within a BOM, including occurrence and effectivity. | Assuming one component can occur only once in a BOM. | Header ID × line ID. |
| Parent | Item being assembled or produced. | Assuming every parent is a finished good. | Item ID and unit. |
| Component | Item consumed in an operation or assembly. | Confusing a substitute with an additive component. | Item ID and unit. |
| Quantity per | Component quantity required for a specified BOM base quantity. | Omitting the parent base quantity. | Component unit per parent base unit. |
| Fixed quantity | Consumption per order or batch, independent of output quantity within defined limits. | Scaling it as if it were per unit. | Component unit/order or batch. |
| Variable quantity | Consumption that scales with the stated parent quantity. | Treating a per-unit amount as per order. | Component unit/parent unit. |
| Scrap allowance | Explicit allowance for expected process or component losses under a named formula. | Applying both a component uplift and a yield loss to the same loss. | Fraction or percent, with formula. |
| Yield | Good output divided by input to a specified process step. | Treating yield as interchangeable with an additive scrap percentage. | Dimensionless fraction. |
| Phantom | Logical assembly whose treatment in planning and execution is configured by the system. | Assuming every phantom is physically stocked. | Item and planning code. |
| Co-product | Additional intended output of a process. | Treating it as a consumed BOM component. | Output quantity and unit. |
| By-product | Secondary output recognized under the local process and accounting rules. | Assuming every output has the same planning treatment. | Output quantity and unit. |
| Routing | Ordered or networked operations for a manufacturing method. | Assuming operation numbers alone guarantee precedence. | Item × plant × routing alternative × revision. |
| Operation | A process step with resource and time requirements. | Treating setup and run as the same kind of time. | Routing ID × operation ID. |
| Setup time | Time incurred per setup event under stated conditions. | Multiplying one setup by every unit. | Hours/setup. |
| Run time | Processing time related to a quantity and resource basis. | Ignoring whether the rate is per unit or per batch. | Machine-hours/unit or labor-hours/unit. |
| Queue and move | Waiting and transport time between operations. | Adding elapsed time directly to machine load. | Hours elapsed or working time. |
| Work center | Defined resource or resource group with an accountable capacity basis. | Treating a group with three machines as one machine without adjustment. | Work-center ID; machine-hours or labor-hours. |
| Calendar | Dated availability of a resource, including shifts and exceptions. | Assuming a national holiday applies to every shift. | Date/time interval and time zone. |
| Capacity | Available resource time after explicitly stated exclusions. | Comparing machine-hours with labor-hours. | Machine-hours/period or labor-hours/period. |
| Production version | Approved association or selection rule for BOM and routing alternatives. | Treating it as a global item property without plant and dates. | Item × plant × version × validity context. |
| Effectivity | Conditions under which a record applies to an order or process. | Using a creation timestamp as its valid-from date. | Date/time interval and other predicates. |
| Alternative | A permitted choice among approved methods or components. | Summing all alternatives as simultaneous consumption. | Alternative ID and priority or allocation. |
| Status | Workflow or release state of a record. | Equating a populated record with an approved record. | Draft, approved, blocked, or local codes. |
| Data owner | Accountable person or role for a business definition and decision. | Assigning accountability to a software system. | Named role and escalation path. |
| Steward | Role that maintains data quality and coordinates corrections. | Making the steward the sole approver of engineering changes. | Named role and responsibilities. |
| Transaction time | When the system recorded a row or change. | Using it alone to infer when the fact applied. | Timestamp with time zone. |
| Valid time | When the represented fact applies in the business process. | Treating a backdated correction as knowledge available earlier. | Effective interval. |
| Order snapshot | Retained record of the master data or revision selected for a specific order. | Reconstructing a released order from current records only. | Order ID and recorded version. |
| Defect | Failure to meet a defined data rule or approved business meaning. | Treating every unusual value as conclusively wrong. | Rule ID, affected grain, severity. |
| Traceability | Ability to relate a production event to relevant material, process, and change records. | Assuming an item ID alone establishes lot genealogy. | Order, lot, serial, and event IDs as applicable. |

## 5. The mechanics, step by step

### Step 1 — Define the decision and its scope

**Input:** the business question, planning horizon, products, plants, orders, and intended users.
**Output:** a documented scope with the required grain and decision date.

1. Identify the decision the analysis will support.
   A capacity check, material buy, historical investigation, and customer promise have different minimum evidence requirements.
2. Name the accountable business decision-maker.
   A planner may recommend a schedule, but a commercial owner may have to authorize a changed customer commitment.
3. Specify whether the analysis describes a proposed plan or a released order.
   Do not imply that a simulation is an instruction to the plant.
4. State the plant, item family, and time bucket.
   For example, use pump assemblies at Plant P1 with daily resource capacity for the week beginning 7 September 2026.
5. State the selection date for revisions.
   In the example, select the production version using the planned production start date.
6. Record the order quantity and its unit.
   Distinguish requested good output from planned starts if a yield adjustment is used.
7. Define the capacity basis.
   Machine-hours and labor-hours are different resources even when one operator tends one machine.
8. Determine whether historical analysis must reproduce what users knew at the time.
   If yes, collect transaction-time history or retained order snapshots as well as valid-time records.
9. Specify what a missing or ambiguous record will do to the decision.
   A released production order should not silently inherit the newest available revision.
10. Agree on escalation authority before the first exception occurs.
    A quality or engineering block must not be overridden by an analyst's default choice.

**Branch:** If the request spans plants, retain plant in every join unless the business key is provably global.
If it spans a revision change, calculate each relevant order or time period separately.

### Step 2 — Inventory source systems and establish identifiers

**Input:** system catalog, extracts, data dictionaries, and named application owners.
**Output:** a source-to-target map and authoritative keys for each object.

1. List the systems that hold engineering definitions, planning settings, shop-floor execution, and calendars.
   The system of record may differ by attribute, not only by table.
2. For each attribute, record its source, owner, update frequency, and meaning.
   Engineering might own component compatibility while planning owns a replenishment parameter.
3. Capture primary keys and cross-system identifiers.
   An engineering drawing number, an ERP item number, and a supplier part number may identify related but nonidentical objects.
4. Preserve leading zeros and meaningful characters in identifiers.
   Coercing `00125` to integer `125` can merge distinct keys or break a join.
5. Include the plant, alternative, revision, and line occurrence wherever they define uniqueness.
   A parent-component pair alone may not identify a BOM line.
6. Store extraction timestamps and the source system's time zone.
   A daily extract is not proof that a change was known at an earlier hour.
7. Record how deletes, blocks, and corrections are represented.
   A current-state export can hide a removed BOM line that was once valid.
8. Compare record counts before and after transformations.
   A count check cannot establish semantic correctness, but it can reveal unintended duplication or loss.
9. Check the join cardinality expected for each relationship.
   A many-to-many join can multiply component requirements without producing a null value.
10. Retain raw values and mapping versions so that a result can be reconstructed.
    Do not overwrite source IDs with normalized labels and discard the originals.

**Branch:** If two systems disagree, use the agreed attribute-level authority and open an exception.
Do not choose the larger number, the most recent timestamp, or the most convenient value as an undocumented resolution.

### Step 3 — Validate item records

**Input:** item identities, plant extensions, units, lifecycle states, and planning parameters.
**Output:** usable item records and a queue of missing or conflicting attributes.

1. Verify the item exists and is eligible for the intended plant and transaction.
   A globally active item may still be blocked for production at a specific site.
2. Establish whether the item is a finished good, intermediate, raw material, or service for this use.
   These categories may differ by plant or process.
3. Check make/buy and sourcing settings against the actual replenishment method.
   A purchased subassembly should not trigger an internal routing merely because another plant makes it.
4. Record the inventory base unit and the order unit.
   For example, a seal may be stocked in eaches and purchased in cases of 12 each.
5. Validate the conversion factor, the direction of conversion, and its scope.
   Twelve each per case is not twelve cases per each.
6. Review whether conversion is fixed or depends on density, moisture, grade, or pack configuration.
   Variable conversion needs an approved method, not a universal constant invented for convenience.
7. Inspect lead time with its start event, end event, and calendar basis.
   A five-working-day lead time differs from five elapsed calendar days.
8. Review lot minimums, multiples, and maximums separately.
   A minimum of 100 each and a multiple of 25 each allow 125 each; a fixed lot of 100 each does not.
9. Check whether safety stock is a policy or a physical balance.
   It should not be added to inventory on hand.
10. Compare lifecycle status with open orders and expected consumption.
    Obsolete does not necessarily mean physically absent, and blocked does not mean approved substitute.
11. Test whether the item's description matches its controlled identity and engineering specification.
    A description alone is not an approval document.
12. Record exceptions at item × plant grain and route them to the responsible owner.
    Avoid creating duplicate tickets for the same underlying defect in each consuming order.

**Branch:** If a required conversion is missing, hold the affected calculation or obtain an approved conversion.
Never infer a factor from similar-looking item descriptions.

### Step 4 — Select the approved production method

**Input:** item, plant, order context, selection date, and released production versions or equivalent rules.
**Output:** a specific BOM-routing pair or an explicit exception.

1. Filter to the correct item and plant.
   Other plant methods are evidence about alternatives, not automatically transferable methods.
2. Filter to approved status and the stated valid-time interval.
   Under `[start, end)`, test `start <= selection_date < end`.
3. Apply any quantity, configuration, customer, lot, or regulatory predicates supported by the business process.
   Date validity alone is not always enough.
4. Identify the BOM and routing linked by each remaining method.
   Validate that both exist and have compatible statuses and effectivity.
5. Apply an approved priority or order-specific assignment when multiple options are allowed.
   A primary and a qualified backup can legitimately overlap.
6. If no rule resolves multiple candidates, mark the order ambiguous.
   An arbitrary `first row` or `max(version)` is not a planning policy.
7. If no candidate remains, distinguish a genuine missing method from an incorrect date or scope.
   Ask the owner rather than expanding the interval in an analytics layer.
8. Persist the selected method and the reason for selection with the order or analysis snapshot.
   Include the source extraction time and rule version where feasible.
9. When a process changes during a long order, apply the organization's documented cutover policy.
   Some processes freeze the released revision; others require an approved amendment or lot split.
10. Review orders crossing the cutover date before approving the change.
    An open order can otherwise consume old material against new instructions.

**Branch:** Multiple candidates are acceptable only when the documented selection logic makes the choice unambiguous for the order.
A uniquely matching method is still unsuitable if its physical compatibility is unverified.

### Step 5 — Check the BOM as a dated, directed graph

**Input:** selected BOM header, component lines, item master, units, and effectivity rules.
**Output:** approved component requirements and structural exceptions.

1. Check that the selected header belongs to the intended parent, plant, and alternative.
   An approved BOM for a similar model is not a substitute.
2. Retain a line identifier, not just parent and component.
   The same screw may appear in two operations with different consumption points.
3. Validate each line's component reference and effective interval.
   A header can be valid while one component line is expired or blocked.
4. Read the BOM base quantity and its unit.
   A quantity of 4 seals per 2 pumps is 2 seals per pump, not 4 seals per pump.
5. Separate variable, fixed-per-order, and other configured quantities.
   Determine whether a fixed quantity applies once per order, per batch, or per operation.
6. Identify alternatives and their selection rules.
   Two approved alternatives usually represent a choice, not two additive requirements.
7. Identify substitutes and whether approval is automatic, conditional, or order-specific.
   Inventory availability does not establish technical interchangeability.
8. Check whether phantom assemblies are exploded according to the local planning configuration.
   Physical stock, costing, and execution treatment may differ by system.
9. Traverse all relevant levels and detect cycles.
   A parent that eventually consumes itself is a structural exception unless a supported process representation explicitly handles it.
10. Look for duplicate line identifiers and implausible repeated quantities.
    Repeated component IDs alone are not necessarily duplicates.
11. Confirm component and parent units and approved conversions before multiplication.
    Convert to a common basis once and retain the conversion used.
12. State how loss is modeled.
    A requirement uplift of `1/(1-s)` and an additive allowance of `1+s` are different formulas.
13. Check whether operation yield already covers the same loss.
    Applying both allowances to one physical loss inflates requirements.
14. If the process produces co-products or by-products, use the system's approved output model.
    A simple consumed-component tree does not represent all process manufacturing cases.
15. Inspect whether the component requirement is associated with the appropriate operation or issue point.
    This matters for timing and for handling changes during production.
16. Capture evidence for engineering review, including drawing, specification, and change notice as appropriate.
    A statistical anomaly is a prompt to inspect, not proof that a design is wrong.

**Branch:** If a component is missing, unapproved, or has an unknown unit, do not replace it with zero.
Report the affected order and the uncomputable requirement.

### Step 6 — Check the routing and its resource basis

**Input:** selected routing, operation sequence, work-center references, time standards, and operating assumptions.
**Output:** standard load by operation and resource type.

1. Verify that the routing revision is approved for the selected BOM and production context.
   A routing that is valid for another configuration may not fit this one.
2. Confirm operation identifiers and precedence relationships.
   Numerical operation labels are not a complete precedence model when parallel branches exist.
3. Check the resource assigned to every capacity-consuming operation.
   Unassigned operations must not disappear from capacity reports.
4. Separate machine, labor, tool, and other constrained resources where relevant.
   A single machine-hour field cannot represent a shortage of certified operators.
5. Read setup time as time per setup event.
   For one order with one setup, load is one setup, not one setup per unit.
6. Determine the number of batches or changeovers required by the lot and process.
   If one order requires two batches, one setup per order can understate load.
7. Read run time and its denominator.
   A rate in machine-hours per 100 pieces must be divided by 100 before multiplying by pieces.
8. Identify whether operation quantity is starts, good output, or another basis.
   Yield loss changes the number of units processed upstream.
9. Keep queue, wait, and transport time separate from resource load unless the resource is actually occupied.
   They can affect lead time without consuming machine capacity.
10. Confirm whether setup and run can overlap or require different resources.
    Do not sum them into one capacity pool without checking the process.
11. Review parallel machines, alternate work centers, and qualification rules.
    The existence of a second machine does not imply it can perform every operation.
12. Compare standards to observed execution using a consistent scope and enough representative runs.
    Investigate changeovers, product mix, and measurement error before proposing a new standard.
13. Avoid treating every fast run as the sustainable standard.
    Approved standards should reflect a documented method and relevant safety and quality constraints.
14. Record the effective revision and source of each time standard.
    A routing without a dated basis makes variance analysis difficult to reproduce.

**Branch:** If a routing's unit or quantity basis is unknown, report load as indeterminate.
A zero substituted for a missing run time produces a false capacity surplus.

### Step 7 — Construct dated work-center capacity

**Input:** resource definition, shift calendar, exceptions, planned downtime, and any approved capacity factors.
**Output:** capacity by resource and time bucket on the same basis as load.

1. Define whether the work center is one machine, a group of machines, a labor pool, or another resource.
   A group requires a count of eligible resources for each interval.
2. Identify shift start and end timestamps, time zone, and breaks.
   Crossing midnight or a daylight-saving transition can make a nominal shift length misleading.
3. Expand the calendar into intervals for the planning horizon.
   Keep resource-specific calendars distinct when machines have different maintenance windows.
4. Subtract unpaid breaks or other exclusions if they are not already excluded from shift hours.
   A documented net shift duration avoids subtracting the same break twice.
5. Subtract approved planned maintenance from the affected resource intervals.
   An event outside a scheduled shift does not reduce that shift's capacity.
6. Prevent overlapping downtime records from being subtracted twice.
   Union overlapping intervals or apply an equivalent controlled rule.
7. Determine whether any capacity factor reflects availability, efficiency, quality, or another loss.
   Name the factor and its denominator before using it.
8. Do not apply an efficiency factor to both routing times and capacity for the same loss.
   Nominal load and effective capacity must be defined consistently.
9. Separate planned absence from unplanned reliability risk.
   A forecasted breakdown allowance is an assumption, not a scheduled maintenance event.
10. Aggregate machine-hours and labor-hours separately by the chosen time bucket.
    Include the resource count and period in the result.
11. Check calendar coverage for the entire horizon, including weekends and holidays.
    A missing row may mean zero availability or missing data; the source contract must say which.
12. Resolve local exceptions with the supervisor or calendar owner.
    A global holiday table is not a substitute for the actual site shift pattern.
13. Retain the calendar revision or snapshot used by a released plan.
    A later maintenance change should not rewrite the evidence of the earlier decision.
14. Compare load with capacity in the same resource unit and bucket.
    A weekly surplus does not prove every day or operation sequence is feasible.

**Branch:** If a machine and a worker are jointly required, check both capacities and their timing.
Passing the machine-hour check does not establish operator availability.

### Step 8 — Reconcile the connected plan

**Input:** selected version, exploded requirements, routing load, dated capacity, and order context.
**Output:** a decision-ready exception list and proposed response.

1. Compute component gross requirements from the appropriate parent quantity and quantity-per basis.
   Keep material already available and scheduled receipts separate from gross demand.
2. If a yield adjustment is justified, calculate starts from the required good output and round according to the physical unit rule.
   Explain whether rounding occurs at each operation or only once.
3. Calculate each setup event and run load using its actual basis.
   Show resource type and time unit beside every subtotal.
4. Compare the load to the relevant calendar bucket.
   A positive gap is an overload, not an automatic instruction to work overtime.
5. Check whether the method selected in planning is the method intended for execution.
   A material explosion for one revision and a routing load for another is not a coherent plan.
6. Trace each material and capacity figure back to item, version, line, operation, and calendar records.
   Include row IDs so the owner can find and correct the source.
7. Distinguish missing data from zero consumption, zero time, and zero capacity.
   These states have different operational consequences.
8. Rank exceptions by exposure: safety and compliance first, then confirmed commitments, shortage risk, capacity, and cost.
   A high-volume benign warning is not automatically the highest-risk issue.
9. Develop feasible responses with owners and constraints.
   Responses can include correcting data, rescheduling, qualifying an alternate, adding an authorized shift, or changing a commitment.
10. Document the chosen action, approver, timestamp, and affected orders.
    A planner's proposed action is not itself an authorization.
11. Recalculate after an approved change using a new snapshot.
    Preserve the original result so that the decision history is auditable.
12. Monitor execution evidence and feed material variances and observed time differences to the correct owners.
    A single unusual order should not automatically rewrite a standard.

### Step 9 — Govern changes and exceptions

**Input:** change request, supporting evidence, affected records, and decision authority matrix.
**Output:** approved, rejected, or deferred change with traceable downstream communication.

1. The requester describes the physical or policy change and the business reason.
   A request to make a dashboard green is not evidence that the underlying process changed.
2. The steward identifies the affected item-plant combinations, BOMs, routings, calendars, and open orders.
   Include alternate methods and downstream plants when relevant.
3. Engineering evaluates design and process compatibility.
   Quality evaluates inspection, qualification, traceability, and disposition implications where required.
4. Planning evaluates demand, inventory, purchase orders, capacity, and the cutover horizon.
   Procurement and suppliers participate when component availability or pack sizes change.
5. The accountable owner approves the record and effective date under local policy.
   Technical access to edit a field is not the same as authority to approve it.
6. Decide how to handle open orders, existing stock, work in process, and the first production lot.
   State whether the prior revision can be consumed, reworked, quarantined, or must be scrapped.
7. Validate the change in a controlled environment or by an equivalent independent review.
   Test the selection rule, unit conversion, line count, and calendar impact as applicable.
8. Publish the approved record and record who did so and when.
   Avoid shared accounts if they prevent attribution.
9. Confirm distribution to planning and execution systems.
   A change in engineering is not effective in production merely because a document was saved.
10. Review the first affected plan or order and capture the result.
    Compare the selected revision and expected quantities with the approved change notice.
11. Close the exception only after the owner accepts the correction and downstream impact is addressed.
    A corrected master row does not automatically repair already released orders.
12. Retain a reversible change history and use a compensating approved correction if rollback is needed.
    Do not erase evidence that an incorrect revision was ever published.

### Step 10 — Measure quality and make the control useful

**Input:** scoped records, exception rules, business impact, and owner assignments.
**Output:** measures that guide remediation rather than merely report a data score.

1. Define the denominator before counting exceptions.
   For example, count released production methods evaluated for active order contexts, not all items ever created.
2. Define whether the numerator counts defective records, affected orders, or distinct incidents.
   These are different measures and must not be added together.
3. Report coverage as well as failure rate.
   A rule that evaluates only half the active items can look artificially healthy.
4. Split missing, invalid, ambiguous, and stale values.
   Each needs a different response and often a different owner.
5. Segment by plant, product family, risk, and aging.
   Averages can hide a concentrated safety-critical failure.
6. Pair exception counts with exposure such as orders affected or capacity-hours at risk.
   Label estimated exposure as an estimate rather than realized loss.
7. Set thresholds based on risk and operating policy.
   A zero-tolerance regulatory block and an investigation threshold for a time-standard anomaly need not be the same.
8. Track time from detection to containment and from containment to permanent correction.
   Closing a ticket without changing the source or dispositioning affected orders is not a permanent correction.
9. Review recurring causes with engineering, planning, operations, and quality.
   A monthly score without a decision owner changes little.
10. Audit a sample of passing records against physical evidence.
    Automated completeness checks cannot prove that a BOM matches the approved design.

## 6. Worked example

### Business decision and assumptions

A fictional plant, P1, plans one order for a compact pump assembly, item P-410.
The customer requires 97 good pumps for shipment in the week beginning 7 September 2026.
The requested delivery is a customer requirement; the planner has not yet made a new delivery commitment.
A demand forecast is not needed to calculate this single order and is not asserted here.
The production target is 97 acceptable pumps; the candidate production plan is 100 starts on 7 September.
The plant must decide whether to release the order, reschedule other work, or seek authorized capacity.

The example uses these explicit rules:

- Version selection uses planned production start date, plant, approval status, and documented priority.
- Validity intervals are `[start, end)` and dates are local plant dates.
- One final assembly operation processes all starts.
- Expected yield is 97% good output from assembly starts.
- Starts must be whole pumps; the planned starts are rounded up.
- The BOM quantities below apply per assembly start.
- All seals are consumed at the start of assembly, including on units that fail.
- There is no additional component scrap allowance.
- One production order requires one setup, and its quantity fits one batch.
- The run time is machine-hours per assembly start.
- Machine-hours are the only capacity constraint considered in this numerical example.
- The order is the only load in the comparison bucket; other loads require a separate check in practice.
- No overtime or subcontracting is authorized at the point of calculation.
- Inventory is assumed available for the netting illustration; reservations and timing still need verification.

### Step A — Select the production version

| Version | Plant | Valid interval | Status | Priority | BOM | Routing |
|---|---|---|---|---:|---|---|
| V1 | P1 | [2026-01-01, 2026-09-01) | Approved | 1 | B1 | R1 |
| V2 | P1 | [2026-09-01, 2027-01-01) | Approved | 1 | B2 | R2 |
| V3 | P1 | [2026-09-01, 2027-01-01) | Approved backup | 2 | B3 | R3 |
| V4 | P2 | [2026-09-01, 2027-01-01) | Approved | 1 | B4 | R4 |

The planned production start date is 7 September 2026.
V1 has expired under the stated half-open rule.
V4 belongs to a different plant.
Both V2 and V3 are technically valid in P1 on the date.
The documented rule selects the lowest priority number; V2 is the primary method and V3 is a permitted backup.
The result is V2, including B2 and R2 together.
If that priority rule did not exist, the correct result would be an ambiguity requiring a decision, not a silent choice of V3 because its number is larger.

Suppose a historical order started on 31 August 2026.
Under the same date rule it would select V1, not V2.
A released order that started later but was frozen to an earlier revision under a documented exception would require its order-specific record and authorization.
The current table alone cannot establish what was selected when that order was released.

### Step B — Calculate required assembly starts

Required good output is 97 pumps.
Expected good fraction is 0.97 good pump per pump started.
The calculated starts are `ceil(97 / 0.97) = 100` pumps.
At the expected yield, `100 × 0.97 = 97` good pumps.
This is a planning expectation, not a guarantee that exactly 97 units will pass inspection.
If yield is lower, the order may need rework, additional starts, or an approved change to the delivery plan.

The yield is applied once to translate required good output into assembly starts.
The component calculation uses those 100 starts directly.
Applying another 3% component allowance for the same failed assemblies would double count that particular loss.
Independent seal handling scrap could justify another allowance only if it is separately defined and evidenced.

### Step C — Explode the selected BOM

B2 defines the following variable quantities per one pump start:

| Component | Description | Quantity per pump start | Unit |
|---|---|---:|---|
| H-20 | Pump housing | 1 | each |
| M-31 | Motor | 1 | each |
| S-04 | Seal | 2 | each |
| F-08 | Fastener | 4 | each |

For 100 pump starts, the gross requirements are:

| Component | Calculation | Gross requirement |
|---|---|---:|
| H-20 | 100 pumps × 1 housing/pump | 100 each |
| M-31 | 100 pumps × 1 motor/pump | 100 each |
| S-04 | 100 pumps × 2 seals/pump | 200 each |
| F-08 | 100 pumps × 4 fasteners/pump | 400 each |

The motor and housing quantities do not represent purchase orders.
They are gross component requirements before netting existing available stock, scheduled receipts, reservations, lot rules, and timing.
The same distinction applies to the seals and fasteners.

For the seal illustration only, assume 38 each are available after reservations and there are no scheduled receipts before use.
Assume zero safety stock and a lot-for-lot purchase rule before pack conversion.
The net need is `max(0, 200 - 38) = 162` seals.
The supplier sells cases of 12 seals and only whole cases can be ordered.
The minimum purchase is `ceil(162 / 12) = 14` cases, or 168 seals.
If all other assumptions hold, the projected remaining quantity attributable to this calculation is `38 + 168 - 200 = 6` seals.
That six-seal remainder is not a service buffer unless the inventory policy explicitly treats it as one.
The purchase must arrive before the assembly issue date; a quantity calculation alone does not establish this.

### Step D — Calculate operation load

R2 has a single assembly operation at work center WC-A.
The approved setup standard is 1.25 machine-hours per setup.
The approved run standard is 0.075 machine-hours per pump start.
The 100-start order requires one setup and one batch under the stated assumption.

| Load element | Calculation | Load |
|---|---|---:|
| Setup | 1 setup × 1.25 machine-hours/setup | 1.25 machine-hours |
| Run | 100 starts × 0.075 machine-hours/start | 7.50 machine-hours |
| Total | 1.25 + 7.50 | 8.75 machine-hours |

The run rate is equivalent to 4.5 machine-minutes per start because `0.075 × 60 = 4.5`.
Using 0.075 hours per 100 starts rather than per start would understate run load by a factor of 100.
Multiplying 1.25 hours by 100 starts would overstate setup load by a factor of 100.
Neither error would necessarily create a missing value in an automated report.

### Step E — Calculate available capacity

WC-A is one eligible machine.
The planned Monday shift has a net scheduled duration of 8.00 hours after breaks.
A planned maintenance event occupies 1.00 hour inside that shift.
The calendar has no overlapping exception in this simplified example.
No additional availability or efficiency factor is applied.

`Available capacity = 1 machine × (8.00 - 1.00) hours = 7.00 machine-hours.`

`Capacity gap = load - available capacity = 8.75 - 7.00 = 1.75 machine-hours.`

The required load is 125% of the available machine capacity: `8.75 / 7.00 = 1.25`.
This is an overload in the Monday bucket, not proof of a weekly overload.
Moving work to another day may resolve it if materials, labor, the customer date, and the later calendar permit.
A different sequencing decision can also matter when work overlaps with maintenance.

### Step F — Compare responses and record the decision

The operations supervisor confirms that an additional 2.00 machine-hours after the shift are technically possible and would not overlap maintenance.
The hypothetical incremental cost is USD 90 per machine-hour, or USD 180 for the two-hour extension.
That rate is a scenario assumption, not a quoted market price.
The extension would provide 9.00 machine-hours in total and leave `9.00 - 8.75 = 0.25` machine-hours of nominal headroom.
The example does not calculate operator availability, energy cost, fatigue exposure, or downstream shipping feasibility.

| Option | Machine-hour effect | Decision implication |
|---|---|---|
| Leave plan unchanged | 1.75 machine-hours overload | Do not release as feasible without a resolution. |
| Add 2.00 authorized machine-hours | 0.25 machine-hours nominal headroom | Conditional option; confirm labor, cost authority, and downstream timing. |
| Move part or all of order | Uses a later or earlier bucket | Check setup count, material timing, delivery, and customer approval if commitment changes. |
| Use V3 backup | Unknown without R3 and calendar data | Do not assume it has spare capacity or equivalent qualification. |
| Reduce the 97-good-unit target | Reduces required starts | Requires an explicit business and customer decision; not a data correction. |

The planner recommends the two-hour extension subject to operations and labor approval.
The production manager can authorize overtime only within the local delegation of authority.
The commercial owner must approve any change to a customer commitment.
The record should retain V2, B2, R2, the calendar snapshot, the 100-start plan, the calculated gap, the recommendation, and the final approver's decision.
Until the required approvals and material timing are confirmed, the plan remains a proposal.

### Step G — Investigate a defect without concealing exposure

Suppose the calendar extract mistakenly omits the maintenance event.
The model would show 8.00 machine-hours available rather than 7.00 machine-hours.
It would report an overload of `8.75 - 8.00 = 0.75` machine-hours, understating the gap by 1.00 machine-hour.
The error does not disappear merely because both results show overload.
The operational recommendation and the cost estimate may change with the larger gap.

The calendar owner confirms the maintenance work order and publishes the missing exception.
The steward identifies every order loaded on WC-A during that interval, not just P-410.
Planning recalculates the affected orders and communicates the revised exposure.
The original extract and the correction timestamp are retained so an earlier decision can be interpreted fairly.
The permanent control reconciles approved maintenance events with the planning calendar and assigns unresolved differences to an owner.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It implements the fictional order above, including half-open version effectivity, priority selection, yield rounding, component gross requirements, pack conversion, and a simple capacity comparison.
It does not perform full MRP netting, time-phased scheduling, or a regulatory release check.
Data values are illustrative and are not read from an ERP system.
Dates are local dates; the example does not model shifts crossing time zones or midnight.
The production version priority is an assumed business rule, not a universal ERP rule.

```python
from datetime import date

order = {
    "item": "P-410",
    "plant": "P1",
    "start": date(2026, 9, 7),
    "good_required": 97,
}

versions = [
    {
        "id": "V1", "item": "P-410", "plant": "P1",
        "start": date(2026, 1, 1), "end": date(2026, 9, 1),
        "approved": True, "priority": 1, "bom": "B1", "routing": "R1",
    },
    {
        "id": "V2", "item": "P-410", "plant": "P1",
        "start": date(2026, 9, 1), "end": date(2027, 1, 1),
        "approved": True, "priority": 1, "bom": "B2", "routing": "R2",
    },
    {
        "id": "V3", "item": "P-410", "plant": "P1",
        "start": date(2026, 9, 1), "end": date(2027, 1, 1),
        "approved": True, "priority": 2, "bom": "B3", "routing": "R3",
    },
    {
        "id": "V4", "item": "P-410", "plant": "P2",
        "start": date(2026, 9, 1), "end": date(2027, 1, 1),
        "approved": True, "priority": 1, "bom": "B4", "routing": "R4",
    },
]

bom_lines = {
    "B2": {"H-20": 1, "M-31": 1, "S-04": 2, "F-08": 4},
}

routings = {
    "R2": {"setup_hours": 1.25, "run_hours_per_start": 0.075},
}

candidates = [
    v for v in versions
    if v["item"] == order["item"]
    and v["plant"] == order["plant"]
    and v["approved"]
    and v["start"] <= order["start"] < v["end"]
]
if not candidates:
    raise ValueError("No approved production version for the order")

best_priority = min(v["priority"] for v in candidates)
best = [v for v in candidates if v["priority"] == best_priority]
if len(best) != 1:
    raise ValueError("Ambiguous production versions at the best priority")
version = best[0]

# Integer percent avoids binary floating-point rounding at the yield boundary.
yield_percent = 97
starts = (order["good_required"] * 100 + yield_percent - 1) // yield_percent
components = {item: starts * qty for item, qty in bom_lines[version["bom"]].items()}

seal_available = 38
seal_net = max(0, components["S-04"] - seal_available)
seals_per_case = 12
cases = (seal_net + seals_per_case - 1) // seals_per_case
seal_remaining = seal_available + cases * seals_per_case - components["S-04"]

routing = routings[version["routing"]]
load_hours = routing["setup_hours"] + starts * routing["run_hours_per_start"]
shift_hours = 8.0
maintenance_hours = 1.0
available_hours = shift_hours - maintenance_hours
gap_hours = load_hours - available_hours

print(f"Selected version: {version['id']} ({version['bom']}, {version['routing']})")
print(f"Starts: {starts} pumps")
for item, qty in components.items():
    print(f"Gross {item}: {qty} each")
print(f"Seal net need: {seal_net} each; purchase: {cases} cases")
print(f"Seal remainder: {seal_remaining} each")
print(f"Load: {load_hours:.2f} machine-hours")
print(f"Capacity: {available_hours:.2f} machine-hours")
print(f"Overload: {max(0, gap_hours):.2f} machine-hours")
```

Expected output:

```text
Selected version: V2 (B2, R2)
Starts: 100 pumps
Gross H-20: 100 each
Gross M-31: 100 each
Gross S-04: 200 each
Gross F-08: 400 each
Seal net need: 162 each; purchase: 14 cases
Seal remainder: 6 each
Load: 8.75 machine-hours
Capacity: 7.00 machine-hours
Overload: 1.75 machine-hours
```

Integer ceiling division keeps the illustrated whole-unit arithmetic exact.
In a production implementation, represent decimal quantities and financial amounts using an appropriate exact decimal type.
The sample intentionally assumes valid positive yield and fixed pack size; a production interface must establish and validate those contracts.
The dictionaries contain only the selected BOM and routing because the other versions are not executed in this case.
For a live implementation, require every selected reference to resolve and report missing data as an exception.
Do not replace a missing routing with zero hours.

The example's capacity calculation subtracts one maintenance interval entirely inside one shift.
A real calendar implementation needs interval intersection, resource counts, overlap handling, local time zones, and effective calendar snapshots.
It should preserve both overload and surplus in the underlying calculation even if a display shows only the positive overload.
The output is evidence for a human decision, not an instruction to release an order or authorize overtime.

## 8. Data, standards, and regulatory context

### Data contracts and provenance

A useful master-data contract defines meaning and behavior in addition to field type.
For a BOM quantity, record the parent base quantity, component unit, variable or fixed basis, effectivity, and treatment of loss.
For a routing time, record whether it is machine or labor time, the operation quantity denominator, and whether it is setup or run.
For a calendar, record the local time zone, resource count, net or gross shift basis, and exception precedence.
For a production method, record status, compatible BOM and routing, selection predicates, and priority semantics.
These definitions should travel with extracts and analytical models, not live only in the memory of a system administrator.

A minimal lineage record includes source system, source key, extraction time, transformation version, and the analysis or order that used the record.
For a regulated or safety-critical process, the organization's approved quality procedures determine the evidence and retention needed.
This lesson does not prescribe a universal retention period.
Access controls should separate request, approval, publication, and emergency override where the risk and staffing model justify separation.
Small sites may use compensating independent review rather than assume that every role is a different employee.

Data sharing introduces confidentiality risks.
A BOM can reveal product design and supplier relationships.
A routing can expose production capability, and a calendar can reveal planned downtime.
Use least-privilege access, approved sharing agreements, and the organization's information-security controls.
A public identifier standard does not make a company's product structure public.

### ISA-95 and integration boundaries

The International Society of Automation (ISA) describes the ISA-95 series as addressing integration between enterprise and control systems.
Its models and terminology can help a team define interfaces among business planning, manufacturing operations, and control activities.
The standard does not certify that a particular BOM or routing is physically correct.
Nor does it require every organization to use the same ERP table names or production-version implementation.
An organization should consult the applicable licensed edition and implementation guidance when claiming conformance.

For this lesson, the relevant practical distinction is between a planning definition and an execution event.
A routing standard may flow toward manufacturing operations; execution confirmations and quality results can flow back as evidence.
An interface must identify the material, order, operation, resource, timestamps, status, and version sufficiently for the intended use.
The exact mapping is implementation-specific and must be tested against local business rules.

### GS1 identifiers and traceability

GS1 publishes standards for identification and for exchanging supply-chain event information.
A Global Trade Item Number (GTIN) identifies a trade item at an appropriate packaging level; it is not automatically an internal plant item number or a production-order ID.
GS1's Electronic Product Code Information Services (EPCIS) standard supports sharing visibility event data, including what, when, where, why, and how dimensions.
An EPCIS event is not a substitute for a validated engineering BOM or a complete internal production genealogy.
A mapping between external identifiers, internal item IDs, lots, and serials needs an owner and clear scope.
GTIN allocation and data-carrier rules depend on the applicable GS1 specifications and trading-partner requirements.

### Regulated manufacturing

The U.S. Food and Drug Administration (FDA) publishes current good manufacturing practice regulations for finished pharmaceuticals in 21 CFR Parts 210 and 211.
For firms and products within scope, relevant provisions include requirements for production and process controls and for written production and control procedures; the actual applicability depends on the activity and facts.
The FDA's Quality Management System Regulation for medical devices, effective 2 February 2026, incorporates ISO 13485:2016 by reference with FDA-specific provisions in 21 CFR Part 820.
Determine the applicable regulatory text, jurisdiction, product, and transition or inspection context with qualified quality and legal personnel.
This educational lesson is not legal advice and does not claim that its simplified example satisfies any regulated release requirement.

A regulated master-data change may require formal assessment, review, validation, training, and retained records under applicable rules and the firm's quality system.
Which controls are mandatory cannot be inferred solely from the existence of a BOM or routing.
Do not infer that a standard is law merely because a regulator references it in a different jurisdiction or context.
Likewise, a contractual or internal policy can impose controls even when a particular statute does not prescribe the same field or workflow.

### Practical integration checks

1. Agree whether dates are local dates or timestamps and how end boundaries are interpreted.
2. Map the same plant and resource identifiers across planning and execution systems.
3. Preserve revision and approval status across interface messages.
4. Reconcile message counts, failed deliveries, and delayed updates.
5. Decide how a source correction affects already released orders and snapshots.
6. Retain the unit conversion and base quantity used in each calculation.
7. Distinguish a failed interface from a legitimate zero-quantity record.
8. Verify that the target system does not silently default missing data to current values.
9. Review access to sensitive product structures and change records.
10. Periodically test recovery and correction procedures, not only the normal publish path.

## 9. Industry practice and case evidence

### Public evidence is not proof of a particular internal defect

Public cases often establish that traceability or process controls matter without disclosing the exact BOM row, routing, or calendar that failed.
The examples below therefore separate documented facts from analytical inference.
They do not claim that any named company used the fictional data or suffered the numerical losses in this lesson.

### FDA food traceability and lot-level records

The FDA's Food Traceability Rule page explains additional traceability record requirements for certain foods on the Food Traceability List, including records linked to key data elements and critical tracking events.
Applicability, exemptions, and compliance timing must be checked against the current FDA material and the current law.
The FDA states on its rule page that Congress directed FDA not to enforce the rule before 20 July 2028.
The rule is an example of a public regulatory framework for tracing covered foods, not evidence that a specific plant's BOM caused a recall.

**Inference:** a manufacturer trying to identify affected production will need reliable links among material identifiers, lot events, and production records.
A static BOM alone describes intended relationships, while actual lot consumption and transformations need execution evidence.
The exact required records and events depend on the business activity and legal scope.
A planning analyst should involve the quality and compliance owners rather than assume that the planning item hierarchy meets all traceability obligations.

### Toyota and supplier visibility after the 2011 earthquake

Toyota's public corporate material describes the severe supply-chain disruption after the Great East Japan Earthquake in 2011 and subsequent work to strengthen its supply chain.
Toyota describes the RESCUE database as a way to visualize supply-chain information and support a faster response to disruptions.
That is a reported company practice, not independent proof that every supplier or material record is complete.
The public material does not establish that a specific manufacturing BOM defect caused the disruption.

**Inference:** multi-tier visibility depends on identifiers and relationship data that can be joined to products and decisions.
A company mapping supplier sites to parts must manage date, plant, and alternative-source context rather than assume a supplier name alone defines an interchangeable source.
A planning BOM and a supply-network map answer different questions, although a controlled mapping between them can support incident analysis.

### Boeing 737 MAX production and quality oversight

The U.S. Federal Aviation Administration (FAA) announced in 2024 that Boeing must develop a comprehensive action plan to address systemic quality-control issues.
The agency's public statement describes oversight and quality-control expectations; it does not identify the fictional data fields in this lesson as the cause of a specific event.
The FAA also announced that it would not grant Boeing production expansion for the 737 MAX until satisfied that quality-control issues were resolved.
These are agency statements about a named real case, not evidence of the effectiveness of any individual master-data dashboard.

**Inference:** a manufacturing change must connect engineering definition, work instructions, execution, inspection, and accountability.
An analyst who observes a mismatch between a planned routing and execution should treat it as a reason for the responsible quality and engineering teams to investigate.
The data comparison alone cannot determine airworthiness, causation, or regulatory compliance.

### Common operating pattern

Across industries, a practical control model usually assigns distinct responsibilities:

- Engineering defines product and process compatibility.
- Operations confirms how the process is actually executed.
- Quality defines inspection, release, and change-control obligations within its remit.
- Planning owns the use of approved inputs in schedules and material plans.
- Maintenance owns equipment availability events and coordinates the calendar.
- Procurement validates supplier, pack, and lead-time information.
- Data stewardship coordinates definitions, exceptions, and data lineage.
- The accountable business owner resolves trade-offs and accepts residual risk.

These are illustrative role allocations, not a claim that every firm uses these titles or the same authority matrix.
A smaller organization can combine roles if approvals and independent checks remain appropriate to its risks.

### Evidence limitations

Vendor implementation guides and ERP manuals can explain a product's configuration semantics.
They are not independent evidence that a firm's physical process is correct.
Company case studies can show what the company says it did, but may not disclose counterfactuals, failures, or complete costs.
Regulator notices establish what the regulator reported, within the notice's scope and date.
Peer-reviewed studies can support general relationships but do not automatically validate a specific plant's standards.
For each claim, retain a source, date, scope, and distinction between documented fact and interpretation.

## 10. Failure modes and red flags

The mitigation column names a control and decision; it is not a substitute for investigating the particular incident.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Missing plant extension | Active orders have no plant item record. | Global item exists but local settings do not. | Plan omits or misclassifies supply. | Hold affected calculation; owner approves plant setup. |
| Wrong unit conversion | Order quantities differ by pack-size multiples. | Cases interpreted as eaches. | Shortage or excess stock. | Validate item-specific conversion and direction. |
| BOM base quantity ignored | Requirements are a constant multiple of expected usage. | Component quantity is per 100 parents, not per parent. | Systematic overbuy or underbuy. | Store and test base quantity with units. |
| Wrong revision selected | Selected version is valid today but not at order start. | Current-state lookup ignores effectivity. | Wrong material or process. | Select using documented order date and preserve snapshot. |
| Unresolved alternatives | Multiple methods match with no priority. | A join chooses an arbitrary row. | Nondeterministic material and load. | Require approved selection rule or explicit exception. |
| BOM-routing mismatch | Component specification differs from tooling instructions. | Versions selected independently. | Line stoppage or quality risk. | Approve and select compatible pairs. |
| Overlapping effectivity | Same priority methods overlap unexpectedly. | Inconsistent change cutover. | Duplicate or ambiguous plans. | Review intervals and enforce context-aware overlap rules. |
| Duplicate BOM line | Repeated line ID or unexplained quantity spike. | Source duplication or many-to-many join. | Inflated gross requirements. | Check keys, occurrence semantics, and join cardinality. |
| BOM cycle | Explosion fails or repeatedly visits same item. | Parent appears in its own ancestry. | Infinite recursion or invalid plan. | Detect cycles and route to engineering; model special processes explicitly. |
| Missing component | Null component or zero after outer join. | Missing reference is coerced to zero. | Undetected shortage. | Quarantine exception; never default missing to zero. |
| Unapproved substitute | Planner replaces a component based only on stock. | Availability mistaken for interchangeability. | Quality or safety exposure. | Require approved substitution conditions and traceability. |
| Setup multiplied per unit | Setup load grows linearly with pieces. | Setup basis mistaken for run basis. | False overload and unnecessary overtime. | Record setup event count and hours per event. |
| Setup omitted | Many short orders show implausibly low load. | Only run time is calculated. | False spare capacity. | Add order- or batch-specific setup events. |
| Wrong time denominator | Rate differs by a factor of 60 or 100. | Minutes/hours or per-100/unit conversion error. | Large load error. | Store units and denominators; reconcile sample orders. |
| Stale standard | Actual times drift persistently by product and shift. | Old process method remains in routing. | Unreliable plan and variances. | Review representative observations and approve a revised standard. |
| Calendar omission | Maintenance order has no corresponding exception. | Systems do not reconcile planned downtime. | Capacity overstated. | Reconcile maintenance events and resource calendars. |
| Downtime double count | Capacity falls below a manually constructed shift total. | Overlapping exceptions subtracted separately. | False shortage and unnecessary rescheduling. | Union intervals and document exclusion rules. |
| Efficiency counted twice | Effective capacity and run standards both include the same loss. | Identical allowance applied on both sides. | Excessive reported load or shortage. | Define a single consistent loss treatment. |
| Resource-unit mismatch | Machine-hour load compared with labor-hour capacity. | Resource types collapsed into one number. | False feasibility. | Maintain separate resource pools and check both. |
| Historical overwrite | Old order appears to use a revision created later. | Current record replaces past state. | Misleading investigation and audit trail. | Retain snapshots and transaction-time history. |
| Incomplete downstream publish | Engineering shows a new revision while MES shows the old one. | Interface failure or delayed replication. | Wrong work instruction or plan. | Reconcile delivery and block unsafe release. |
| Closed ticket, open orders | Master row is corrected but old orders still carry bad data. | Correction not propagated or orders frozen. | Continuing exposure after closure. | Enumerate and disposition affected orders. |
| Quality metric gaming | Completeness improves while exceptions are excluded. | Denominator or scope changes quietly. | False assurance. | Publish rule coverage, denominator, and change history. |
| Uncontrolled emergency edit | Shared account changes a routing without approval. | Access exceeds authority. | Untraceable process change. | Log identity, use emergency approval, and review afterward. |

A red flag indicates where to investigate; it is not always proof of error.
Two component lines can be legitimate if they represent different operations or occurrences.
Two valid production methods can be legitimate if allocation is explicit and both are approved.
A time-standard deviation can be driven by mix, operator training, measurement changes, or an actual process change.
Preserve these distinctions when designing automated alerts.

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | How to test or respond |
|---|---|---|---|
| Selection uses production start date. | Gives a deterministic version lookup. | Local policy uses release date or order-specific frozen revision. | Read process policy and compare released orders with snapshots. |
| Valid intervals are half-open. | Makes adjacent dates nonoverlapping. | Source system uses inclusive end dates. | Inspect source documentation and boundary examples; translate explicitly. |
| Primary priority resolves alternatives. | Selects V2 over approved backup V3. | Quantity, customer, or machine qualification changes the choice. | Review complete selection predicates and approval evidence. |
| Expected yield is 97%. | Converts required good pumps to starts. | Product mix or process changes actual yield. | Review representative yields by method and period; use risk scenarios. |
| All components are consumed per start. | Applies BOM quantities to 100 starts. | Components are issued after an early rejection or recovered in rework. | Map issue point and actual consumption; change the calculation basis. |
| One setup covers the order. | Calculates 1.25 setup machine-hours. | Batch-size limit or split schedule creates multiple setups. | Check batch capacity and sequence; count actual setup events. |
| One machine is eligible. | Defines WC-A capacity and resource unit. | Backup machine is qualified or primary machine is unavailable. | Check qualification and resource-specific calendars. |
| Maintenance occupies one full scheduled hour. | Subtracts 1.00 machine-hour. | Event lies partly outside shift or overlaps another exclusion. | Intersect and union dated intervals. |
| Shift hours are net of breaks. | Avoids another subtraction. | Calendar hours include unpaid breaks. | Reconcile shift definition with local calendar owner. |
| No efficiency factor is applied. | Keeps load and capacity on nominal standard-hour basis. | Standards or calendar require an agreed performance adjustment. | Document loss definitions and apply each once. |
| The order is the only load in the bucket. | Allows a simple load-capacity comparison. | Other orders share WC-A on Monday. | Sum all compatible loads and inspect sequencing. |
| Seal stock is available and unreserved. | Makes illustrative netting possible. | Stock is quarantined, allocated, or unavailable by issue date. | Reconcile inventory status, reservations, and timing. |
| Whole cases contain 12 seals. | Converts net need to purchase quantity. | Supplier pack size or conversion changes by version. | Verify current supplier agreement and item conversion. |
| Components have no extra scrap allowance. | Avoids double counting the yield loss. | Separate handling scrap is material and approved. | Measure loss by step and use distinct, documented factors. |
| One operation represents the process. | Makes the hand example transparent. | Parallel operations, labor, tooling, or queues constrain the order. | Model each relevant resource and precedence relationship. |
| Master data are current at extraction. | Supports the example's planned selection. | Interfaces are delayed or a correction is backdated. | Compare source timestamps, interface status, and audit trail. |
| Expected output is deterministic arithmetic. | Enables a reproducible hand check. | Actual yield and processing times vary. | Use scenarios and execution monitoring; do not present a guarantee. |

A capacity surplus in one bucket does not prove that a finite sequence is feasible.
An approved method does not prove that all components have arrived or that workers are qualified.
An accurate BOM does not prove that the actual lot genealogy was captured.
A well-governed master dataset is necessary for a reliable plan, but it is not sufficient by itself.

## 12. Exercises

### Exercise 1 — Map the object graph

**Type:** recall and application.  
**Time:** 15 minutes.  
**Deliverable:** a diagram and a six-row ownership table.

A plant makes product X from housing H and motor M at work center W.
Its schedule uses a routing with assembly and test operations, and W has a maintenance exception on Friday.
Draw the order-to-version-to-BOM-and-routing relationships and show where the calendar attaches.
Name a plausible accountable role for each of item, BOM, routing, version, work center, and calendar.

**Acceptance criteria:**

- The diagram identifies a compatible BOM-routing selection, not two unrelated latest records.
- Component relationships attach to the BOM, and operations attach to the routing.
- The work center connects to the dated calendar.
- The ownership table assigns a business role and distinguishes approval from data entry.
- The response states that the precise role titles depend on local authority.

### Exercise 2 — Effectivity and ambiguity

**Type:** application.  
**Time:** 15 minutes.  
**Deliverable:** a version decision for each date with a one-sentence justification.

For item X at plant P, A is approved for `[2026-01-01, 2026-10-01)` with priority 1.
B is approved for `[2026-10-01, 2027-01-01)` with priority 1.
C is approved for `[2026-10-01, 2027-01-01)` with priority 2.
Selection uses the planned start date and the lowest numeric priority.
A fourth version D has the same dates and priority as B but is for plant Q.

Choose the method for 30 September 2026 and 1 October 2026 at plant P.
Then state what happens if an additional approved method E at plant P has the same dates and priority as B and no other selection rule exists.

**Acceptance criteria:**

- The response applies the half-open boundary correctly.
- It excludes the other plant and uses the documented priority.
- It flags the B/E tie rather than choosing a row arbitrarily.
- It does not treat C's existence alone as a defect.

### Exercise 3 — Material and load arithmetic

**Type:** calculation.  
**Time:** 20 minutes.  
**Deliverable:** starts, component requirement, load, and units with shown arithmetic.

An order needs 114 good units at a 95% expected yield.
Every start consumes 3 clips, and there is no additional component scrap.
One setup takes 0.80 machine-hours; run time is 0.04 machine-hours per start.
Assume a whole-unit start quantity, one setup, and a single batch.

**Acceptance criteria:**

- The response computes whole starts using the yield denominator.
- It calculates clip gross requirement from starts rather than good units.
- It adds setup once and distinguishes machine-hours from elapsed lead time.
- It states that expected yield does not guarantee the actual good quantity.

### Exercise 4 — Calendar and capacity

**Type:** calculation and critique.  
**Time:** 20 minutes.  
**Deliverable:** capacity, overload or surplus, and one unresolved feasibility question.

Two identical machines each have a net 8-hour shift.
One machine has a 1.5-hour maintenance event within its shift; the other has none.
There are no overlapping exceptions and no capacity factor.
Planned machine load is 15.2 machine-hours in the shift.

**Acceptance criteria:**

- The response calculates both machines' available time separately or equivalently.
- It reports the signed capacity difference with units.
- It does not subtract the 1.5-hour event from both machines.
- It explains why aggregate capacity alone does not prove a feasible sequence.

### Exercise 5 — Diagnose and contain defects

**Type:** critique.  
**Time:** 20 minutes.  
**Deliverable:** a ranked incident table with owner, containment, and durable control.

A report shows three issues: a supplier case conversion is absent, two compatible methods have the same priority and validity, and a routing standard differs from the last operation confirmation by 4%.
The first issue affects a released customer order, the second affects a proposed order, and the third has only one observation.
No safety or regulatory evidence is supplied.

**Acceptance criteria:**

- The response distinguishes missing data and ambiguity from a single statistical anomaly.
- It identifies which calculation or release decision to hold or escalate.
- It names appropriate business owners rather than assigning every decision to analytics.
- It avoids asserting that the 4% difference proves the standard is wrong.
- It explains any chosen priority and identifies information that could change it.

### Exercise 6 — Design an engineering change

**Type:** design.  
**Time:** 25 minutes.  
**Deliverable:** a change workflow and a one-page decision record.

A replacement motor requires a new fixture and a revised assembly instruction.
The old motor is held in stock and several orders are already released.
The supplier can deliver the replacement next month, but no approval is yet recorded.

**Acceptance criteria:**

- The workflow includes engineering, quality, planning, operations, and procurement decisions.
- The new BOM and routing are reviewed as a compatible method.
- The record states effective date logic, affected orders, old stock disposition, and approvals.
- It addresses publication to planning and execution systems and a first-order check.
- It distinguishes proposed actions from authorized releases and customer commitments.

### Exercise 7 — Reconstruct historical knowledge

**Type:** critique and design.  
**Time:** 20 minutes.  
**Deliverable:** a short investigation plan and two different historical questions.

On 20 October, a steward corrects a routing and backdates its valid-from date to 1 October.
A planner released an order on 10 October using the routing then visible in the system.
A current extract only shows the corrected routing.

**Acceptance criteria:**

- The response distinguishes what is now believed valid on 10 October from what the planner could see then.
- It identifies transaction-time history or an order snapshot as necessary evidence for the latter question.
- It does not assert the planner was negligent based on a later correction.
- It identifies how to contain affected open orders while evidence is gathered.

### Exercise 8 — Design a quality scorecard

**Type:** design.  
**Time:** 25 minutes.  
**Deliverable:** three measures with numerator, denominator, grain, owner, threshold, and action.

A plant wants to measure production-method ambiguity, missing unit conversions, and calendar coverage for a rolling four-week horizon.
Its item population includes inactive products and products not used in the horizon.
Management asks for one green/red score.

**Acceptance criteria:**

- Each metric has a defined eligible population and unit of counting.
- The ambiguity rule permits properly prioritized alternatives.
- Calendar coverage distinguishes a closed shift from missing calendar data.
- The response reports rule coverage and avoids treating inactive items as current exposure without justification.
- Each threshold triggers a named action and accountable owner.
- The response explains why an aggregate score cannot replace incident-level review.

## 13. Self-check answers

### Exercise 1 — Suggested map and ownership

The order identifies item, plant, quantity, and the selection date.
The selected production version points to one compatible BOM and routing.
The BOM points to housing H and motor M through separate component lines.
The routing points to assembly and test operations; the appropriate operation references work center W.
The calendar attaches to W and includes the Friday maintenance exception.

A plausible ownership table is:

| Object | Accountable business role | Typical supporting role |
|---|---|---|
| Item | Product or item-data owner | Data steward |
| BOM | Product engineering | Manufacturing engineering and quality |
| Routing | Manufacturing engineering | Operations supervisor |
| Production version | Manufacturing engineering or designated plant owner | Planning and quality |
| Work center | Operations resource owner | Maintenance and industrial engineering |
| Calendar | Plant scheduling or operations owner | Maintenance planner |

The precise approval authority must come from the site's responsibility matrix.
A steward can enter an approved change without being authorized to approve its engineering content.

### Exercise 2 — Boundary and alternative answers

On 30 September 2026, A is valid because the date is before its exclusive 1 October end.
B and C have not yet started.
The selected method is A.

On 1 October 2026, A is no longer valid and B and C are both valid at plant P.
The documented priority selects B because 1 is lower than 2.
D is not eligible because it belongs to plant Q.

If E is also valid at priority 1 for plant P, B and E tie.
The selection is ambiguous unless another approved predicate or allocation rule resolves it.
Escalate the affected order rather than relying on row order or the alphabetically larger ID.
C remains a legitimate backup if its use and qualification are documented.

### Exercise 3 — Material and load answers

Required starts are `ceil(114 / 0.95) = 120` units.
Expected good output is `120 × 0.95 = 114` units, not a guaranteed yield.

Clip gross requirement is `120 starts × 3 clips/start = 360 clips`.
No inventory, receipts, or lot rule was supplied, so a net purchase quantity cannot be determined.

Run load is `120 starts × 0.04 machine-hours/start = 4.80 machine-hours`.
Total load is `0.80 + 4.80 = 5.60 machine-hours`.
The setup is counted once because the question specifies one setup.
A second batch or interruption requiring another setup would change the answer.

### Exercise 4 — Calendar answer

Machine 1 has `8.0 - 1.5 = 6.5` machine-hours available.
Machine 2 has `8.0` machine-hours available.
Total capacity is `6.5 + 8.0 = 14.5` machine-hours.

The signed gap, defined as load minus capacity, is `15.2 - 14.5 = 0.7` machine-hours of overload.
A feasible response might move work, add approved time, or reduce load, but the exercise does not authorize any option.
Even if aggregate load were below capacity, eligibility, start times, maintenance timing, setup, and precedence could prevent a feasible sequence.

### Exercise 5 — Diagnosis rubric

A strong answer normally contains the missing conversion for the released order first, subject to any higher safety or compliance risk discovered during triage.
The buyer or item-data owner should verify the supplier pack definition; planning should hold the affected quantity decision until an approved conversion is available.
A durable control verifies conversion at the relevant item and supplier or pack context and flags missing values before order release.

The equal-priority methods should be escalated to manufacturing engineering and planning.
The team must either supply a documented selection predicate or correct an erroneous overlap.
A future validation rule should distinguish permitted alternatives from unresolved ties.

The 4% difference is a signal to investigate, not an automatic correction.
Manufacturing engineering can review comparable observations, quantity, setup, mix, and measurement basis before changing the standard.
A single confirmation is insufficient to infer a systematic bias.
An excellent answer states that actual customer, safety, or regulatory exposure could change the ranking.

### Exercise 6 — Change-design rubric

A complete workflow begins with a request identifying the new motor, fixture, instructions, and business reason.
Engineering evaluates compatibility and identifies the new BOM-routing pair.
Quality assesses qualification, inspection, traceability, and any required approvals.
Operations validates the fixture and work instruction, while procurement confirms the supplier and timing.
Planning identifies open orders, old inventory, demand, and capacity exposure.

The decision record states the effective-date rule, the status of the old motor, and whether existing orders remain on the old revision or receive an approved amendment.
It identifies the approvers and a distribution check for both planning and execution systems.
It includes a first-order review and a response if the new method fails qualification.
A weak answer simply edits the BOM or assumes old stock is interchangeable.
The customer's commitment changes only through the appropriate commercial process.

### Exercise 7 — Historical-knowledge answer

The first question is: under the current corrected record, what routing is considered valid for 10 October?
The second is: what routing was visible and selected when the planner released the order on 10 October?
Those questions can have different answers because transaction time and valid time differ.

The current extract alone cannot answer the second question.
Obtain an immutable change log, a historical extract, an order snapshot, or comparable retained evidence.
Contain open affected orders by identifying the routing they actually carry and asking engineering and quality for disposition.
Do not assign blame based on the later correction alone.

### Exercise 8 — Scorecard rubric

One ambiguity measure is `order contexts with multiple top-priority eligible methods / evaluated active order contexts`.
Its owner is the production-method owner; the action is to resolve the tie before the affected method is released.
Approved lower-priority backups do not count as ambiguous under the stated priority rule.

A conversion measure can be `required item-pack contexts missing an approved conversion / item-pack contexts requiring conversion`.
Its owner is the item-data or procurement owner; the action is to verify the supplier pack and hold affected calculations.
Do not count items that never require a conversion in the denominator merely to improve the score.

A calendar coverage measure can be `resource-days with an explicit valid calendar state / resource-days required by the horizon`.
The calendar contract must distinguish an explicitly closed day from an absent record.
Operations or scheduling owns missing resource-days and must resolve them before relying on capacity.
Each measure should also publish the share of the intended population actually evaluated.
Thresholds should reflect local risk: an unresolved safety-critical exception may require immediate containment even when the overall percentage is small.
A single aggregate color conceals the specific orders and resource intervals needing action.

## 14. Interview questions

### 1. What is the difference between an item and a BOM line?

A strong answer distinguishes an item identity from a parent-component relationship at a defined revision, occurrence, and effectivity.
A weak answer treats a component's description as a complete product structure.

### 2. How do you select a BOM and routing for an order?

A strong answer starts with plant, status, selection date, configuration, and approved compatibility, then applies documented priority or allocation.
A weak answer selects the latest revision of each independently.

### 3. Are overlapping valid production versions always bad?

A strong answer explains approved primary and backup methods and distinguishes intended overlap from unresolved ambiguity.
A weak answer either rejects every overlap or silently chooses the first record.

### 4. How would you detect a BOM unit defect?

A strong answer checks parent base quantity, component unit, conversion direction, fixed versus variable basis, and sample requirements against approved specifications.
A weak answer relies on populated fields without checking their meaning.

### 5. Why is setup not normally multiplied by output quantity?

A strong answer identifies setup as time per setup event and explains that batch splits or changeovers can create more than one event.
A weak answer assumes either one setup per unit or one setup for every conceivable order without inspecting the process.

### 6. What makes a capacity comparison invalid?

A strong answer mentions mismatched resource units, time buckets, resource eligibility, missing shifts, duplicated downtime, or double-counted losses.
A weak answer compares a weekly labor-hour total with a daily machine-hour load and calls the result feasible.

### 7. How do you handle a missing calendar row?

A strong answer consults the source contract to distinguish closed capacity from missing data and escalates uncertainty.
A weak answer assumes missing always means 24-hour availability or zero capacity without evidence.

### 8. How do you evaluate an apparently stale routing time?

A strong answer compares representative actuals on a consistent basis, investigates mix and measurement, and routes a proposed standard change for approval.
A weak answer overwrites the standard from a single fast or slow order.

### 9. How do you analyze an old order after a backdated correction?

A strong answer distinguishes valid time from transaction time and seeks order snapshots or audit history.
A weak answer uses today's record as proof of what the planner knew then.

### 10. How would you measure master-data quality?

A strong answer defines scope, grain, numerator, denominator, coverage, owner, action, and business exposure.
A weak answer reports a percentage of non-null fields without checking accuracy or decision relevance.

### 11. What should happen after an incorrect BOM is corrected?

A strong answer identifies affected open orders, inventory, downstream systems, and customer exposure, then documents disposition and verifies publication.
A weak answer closes the incident as soon as the master row changes.

### 12. What can a standards body or regulator tell you about this data model?

A strong answer distinguishes an integration or identification convention from a binding rule applicable to a specific product and jurisdiction.
A weak answer claims that using a standard identifier automatically proves regulatory compliance.

## 15. Key terms

Each term links to the detailed definition table and, where useful, the relevant procedure.

- [Alternative](#4-definitions-and-vocabulary) — approved choice, not necessarily additive consumption.
- [Base unit](#4-definitions-and-vocabulary) — reference unit for the specified item context.
- [BOM header](#4-definitions-and-vocabulary) — identified parent structure and revision context.
- [BOM line](#4-definitions-and-vocabulary) — component occurrence with quantity and conditions.
- [By-product](#4-definitions-and-vocabulary) — secondary output under local process rules.
- [Calendar](#step-7--construct-dated-work-center-capacity) — dated availability and exceptions for a resource.
- [Capacity](#step-7--construct-dated-work-center-capacity) — available resource time on a stated basis.
- [Co-product](#4-definitions-and-vocabulary) — additional intended process output.
- [Component](#step-5--check-the-bom-as-a-dated-directed-graph) — item consumed in a parent process.
- [Data owner](#step-9--govern-changes-and-exceptions) — role accountable for a business definition or decision.
- [Data steward](#step-9--govern-changes-and-exceptions) — role coordinating quality and controlled maintenance.
- [Effectivity](#step-4--select-the-approved-production-method) — conditions determining when a record applies.
- [Fixed quantity](#4-definitions-and-vocabulary) — quantity per order, batch, or another defined event.
- [Item](#step-3--validate-item-records) — identified material, intermediate, or product.
- [Lead time](#4-definitions-and-vocabulary) — modeled time between defined events.
- [Lot-size rule](#4-definitions-and-vocabulary) — rule for determining replenishment quantity.
- [Operation](#step-6--check-the-routing-and-its-resource-basis) — process step with a resource and time basis.
- [Order snapshot](#3-the-core-mental-model) — retained evidence of what a particular order selected.
- [Parent](#step-5--check-the-bom-as-a-dated-directed-graph) — item made using component relationships.
- [Phantom](#4-definitions-and-vocabulary) — logical assembly handled by configured planning rules.
- [Plant item](#step-3--validate-item-records) — item settings scoped to a manufacturing site.
- [Production version](#step-4--select-the-approved-production-method) — approved BOM-routing combination or equivalent rule.
- [Quantity per](#4-definitions-and-vocabulary) — component quantity for a stated parent base.
- [Routing](#step-6--check-the-routing-and-its-resource-basis) — approved operation sequence or network.
- [Run time](#step-6--check-the-routing-and-its-resource-basis) — processing time per stated quantity.
- [Scrap allowance](#4-definitions-and-vocabulary) — explicitly modeled loss with a defined formula.
- [Setup time](#step-6--check-the-routing-and-its-resource-basis) — resource time per setup event.
- [Status](#step-4--select-the-approved-production-method) — workflow or approval state.
- [Traceability](#data-contracts-and-provenance) — ability to connect relevant product and process evidence.
- [Transaction time](#3-the-core-mental-model) — time a system recorded a fact or correction.
- [Unit conversion](#step-3--validate-item-records) — scoped, directional relationship between units.
- [Valid time](#3-the-core-mental-model) — period a fact applies in the business process.
- [Variable quantity](#4-definitions-and-vocabulary) — consumption proportional to a stated quantity.
- [Work center](#step-7--construct-dated-work-center-capacity) — defined capacity-owning resource or group.
- [Yield](#step-b--calculate-required-assembly-starts) — good output divided by input to a specified step.

## 16. Sources

Sources below are public starting points accessed as references for this educational lesson; check the current edition and applicable jurisdiction before operational use.
The standards may require purchase for full normative text.

1. International Society of Automation, [ISA-95 Standard: Enterprise-Control System Integration](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard).
   Primary standards-body overview of the series and its integration scope; it does not validate a plant's records.
2. GS1, [EPCIS and CBV](https://www.gs1.org/standards/epcis).
   Primary standards-body material on visibility-event exchange and associated vocabularies.
3. GS1, [GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/).
   Primary guidance on trade-item identification and GTIN assignment rules.
4. U.S. National Archives, [21 CFR Part 210 — Current Good Manufacturing Practice in Manufacturing, Processing, Packing, or Holding of Drugs](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-C/part-210).
   Current electronic Code of Federal Regulations; evaluate scope and effective text for the relevant activity.
5. U.S. National Archives, [21 CFR Part 211 — Current Good Manufacturing Practice for Finished Pharmaceuticals](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-C/part-211).
   Current regulatory text for the applicable U.S. finished-pharmaceutical context.
6. U.S. Food and Drug Administration, [Quality Management System Regulation (QMSR)](https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr).
   FDA explanation of the medical-device quality management regulation and its effective date.
7. U.S. Food and Drug Administration, [Food Traceability Final Rule](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods).
   FDA summary of covered foods, key events and data, exemptions, and current compliance information.
8. Toyota Motor Corporation, [Five Years on: Toyota's Efforts to Build a Disaster Resilient Future](https://global.toyota/en/detail/11373994), 11 March 2016.
   Corporate account of its disaster-response approach and RESCUE supply-chain database; the company's description is not independent assurance of complete visibility.
9. U.S. Federal Aviation Administration, [Updates on Boeing 737-9 MAX Aircraft](https://www.faa.gov/newsroom/updates-boeing-737-9-max-aircraft), including the 28 February 2024 update.
   Primary agency account of the quality-control action plan, not proof of any particular master-data defect.
10. U.S. Federal Aviation Administration, [FAA Halts Boeing MAX Production Expansion to Improve Quality Control](https://www.faa.gov/newsroom/faa-halts-boeing-max-production-expansion-improve-quality-control-also-lays-out-extensive), 24 January 2024.
    Primary agency statement on inspection and production-expansion restrictions at that time.

Source claims are limited to what the linked organization reports.
The fictional pump case, arithmetic, data design, and proposed controls are instructional constructions, not company evidence.

## 17. Links

- [Module 15 index](_index.md) — topic sequence and module position.
- [Manufacturing and production planning survey](../../../course/15_manufacturing_production_planning.md) — module-level planning cascade and examples.
- [Manufacturing basics](../../01_foundations/01_supply_chain_foundations/07_manufacturing-basics.md) — foundational context and terminology.
- [Production planning labs index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md) — related simulation and optimization exercises.
- [Production planning labs survey](../../../course/30_simulation_optimization_production_labs.md) — exercises that can use governed input data.
- [Planning cascade topic](01_planning-cascade.md) — sibling topic connecting the planning levels.
- [MRP explosion topic](04_mrp-explosion.md) — sibling topic for gross-to-net and time-phased calculations.
- [Capacity requirements topic](05_capacity-requirements.md) — sibling topic for load-versus-capacity decisions.

Sibling topic links follow the module index; availability depends on the expansion status shown there.
