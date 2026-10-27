# Manufacturing: BOM, Routing, MRP, Capacity

> Module 01 — Supply Chain Foundations for Data Scientists
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 03, 04, 05, 06
> Version 2026.5

---

## 1. Why this topic exists

Every supply chain plan eventually becomes a work order on a machine. A forecast becomes a
production plan, the plan becomes a master schedule, the master schedule explodes through a bill of
materials into component orders, and those orders are released to a shop floor with a finite number
of hours in the week. Most of the analytical content of Modules 06 to 20 sits above this layer.
Most of the consequences land here.

The gap between the plan and the plant is where operational analytics quietly fails. A demand model
can be excellent and still produce a schedule the plant cannot execute, because the model never
touched the two structures that determine feasibility: the bill of materials (BOM) and the routing.
A dashboard can report overall equipment effectiveness (OEE) to two decimal places and still
mislead, because the number is not comparable across plants.

Three examples show the shape of the problem.

1. **The forecast improves and the schedule gets worse.** A statistical model reduces forecast
   error at the finished-good level. The planner responds by cutting safety stock. Nothing changes
   in the BOM, where a component has been obsolete for two revisions and still carries a scrap
   factor. The plant now has a better demand signal and the same material surprises.
2. **The OEE league table.** A group publishes monthly OEE by plant. The single-product line
   reports 84%; the high-mix line reports 61%. Both are accurate. Management concludes the second
   plant is badly run and cuts its improvement budget.
3. **The impossible date.** Sales promises a date the planning system accepts. The system offset it
   backwards through fixed planned lead times and never checked capacity, so the start date it
   generated falls in the past. The order is expedited, and the expedite consumes capacity another
   order needed.

Each is a data-semantics failure, not a modelling failure. The fix is knowing what a BOM is, what a
routing is, what material requirements planning (MRP) actually computes, and what capacity planning
can and cannot tell you.

Topic 03 introduced the BOM as a graph with time on its edges and worked an explosion with scrap
and effectivity. This topic takes the same structures and asks the manufacturing question: given
demand for end items, what must be bought or built, when must it start, and can the plant do it in
the week the plan asks for.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** make-to-stock, make-to-order, assemble-to-order, engineer-to-order, repetitive,
   process, and project manufacturing, and **state** what each implies for planning data.
2. **Place** a planning problem at the right layer — sales and operations planning (S&OP), master
   production scheduling (MPS), MRP, rough-cut capacity planning (RCCP), capacity requirements
   planning (CRP), finite scheduling — and **explain** what that layer can decide.
3. **Explode** a two-level BOM with on-hand inventory, scrap, and lot-sizing rules, and **offset**
   the resulting orders backwards through lead times to obtain release dates.
4. **Compute** a load profile for a work centre in standard hours, **compare** it with rated and
   demonstrated capacity, and **identify** the overloaded week.
5. **Explain** why MRP generates infeasible dates, naming at least five mechanisms.
6. **Calculate** OEE from its components, **diagnose** which component drives a loss, and
   **explain** why it is unsafe as a cross-plant benchmark.
7. **Choose** a lot-sizing rule for a stated cost structure, and **explain** when the economic order
   quantity is the wrong answer.
8. **Map** a manufacturing data question to an ISA-95 level and **state** the system of record.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture in your head: a demand signal at the top, exploding downward through a product
structure, with a capacity check standing to the side and a feedback loop returning from the floor.

```text
                        CUSTOMER ORDERS / FORECAST
                                   |
                                   v
   +---------------------------------------------------------------+
   |  LEVEL 0  MPS      which end items, which week, how many       |
   +---------------------------------------------------------------+
             |  explode down the BOM            ^  confirm up
             v                                  |
   +-------------------------+       +-------------------------+
   |  LEVEL 1..n  MRP        |       |  CAPACITY CHECK         |
   |  gross -> net -> offset | ----> |  RCCP on the MPS        |
   |  lot size, scrap, dates | <---- |  CRP on the MRP orders  |
   +-------------------------+       +-------------------------+
             |                                  |
             v                                  v
   +---------------------------------------------------------------+
   |  FINITE SCHEDULE   sequence orders on machines, honour setups  |
   +---------------------------------------------------------------+
             |
             v
   +---------------------------------------------------------------+
   |  SHOP FLOOR / MES  dispatch, confirm, backflush, OEE, genealogy |
   +---------------------------------------------------------------+
             |
             +---- actual start, finish, scrap, downtime, yield ------+
                          (feedback, per shift or per day)
```

Five properties explain most of what follows.

**Property 1 — Material explodes downward, dates are offset backwards, capacity is checked last.**
MRP answers "what and when to order", not "can we build it". The capacity check is a separate
calculation, often run afterwards, often weekly, and often ignored when it disagrees.

**Property 2 — The BOM and the routing are two different graphs over the same items.** The BOM says
what a product is made of. The routing says how: which operations, in what sequence, at which work
centre, with what setup and run time. They are maintained by different people on different cycles
and routinely disagree. An item on the BOM with no operation is a component never consumed; an
operation consuming an item not on the BOM is a requirement MRP will never generate.

**Property 3 — MRP plans quantities. Only a schedule plans time.** MRP produces dated orders in
weekly or daily buckets, assuming each order owns its lead time exclusively and capacity is
infinite. A finite scheduler produces a sequence: order A then order B on machine 3, with a
90-minute changeover between them. The first is a plan. The second is something a person can run.

**Property 4 — Capacity is a rate, and the constraint sets the rate.** A work centre has hours per
week, adjusted for shifts, staffing, maintenance, and yield. Output is set by the constraint, and
the constraint moves when mix changes. A plan that loads every work centre to 95% has planned for
no protection anywhere.

**Property 5 — Every level adds rounding and uncertainty.** A lot-size rule at level 1 changes the
demand seen at level 2. A scrap factor at level 2 changes the quantity bought at level 3. Two
percent at each of four levels is not two percent; it compounds.

### 3.1 The layers, and what each one may decide

| Layer | Question it answers | Horizon | Bucket | Typical owner | Typical system |
|---|---|---|---|---|---|
| Demand plan / S&OP | What will the market want, in aggregate | 3–18 months | Month | Demand and supply planning | Planning suite, spreadsheet |
| MPS | Which end items, which week, how many | 1–3 months | Week | Master scheduler | ERP |
| MRP | Which components, how many, released when | 1–6 months, at least cumulative lead time | Day or week | Material planner | ERP |
| RCCP | Can the MPS be built at the critical work centres | 1–3 months | Week | Master scheduler | ERP or spreadsheet |
| CRP | Can the MRP orders be built at every work centre | 1–8 weeks | Day or week | Production planner | ERP, APS |
| Finite scheduling | In what sequence, on which machine, with which setup | 1 day–4 weeks | Minute to shift | Scheduler | APS, MES |
| Execution | What the operator or machine does next | Now–shift | Real time | Supervisor | MES, SCADA |

The table tells you which question a number can answer. An MPS number cannot tell you whether a
component is short. An MRP number cannot tell you whether the plant can build it. A finite schedule
cannot tell you whether demand was real.

### 3.2 The five questions every production plan must answer

1. **What** is to be made — item, revision, and customer or stock destination.
2. **How many** — quantity, unit of measure, and the scrap or yield assumption behind it.
3. **By when** — the due date, and whether it is a customer commitment or an internal target.
4. **With what** — components, capacity, tooling, people, and assumptions about availability.
5. **At what cost** — setup, run, overtime, expedite, and the working capital carried while open.

Most planning arguments are disagreements about which of the five is binding. A sales director
argues about question 3, a plant manager about question 4, a finance partner about question 5. The
plan is where those arguments are supposed to be resolved, and usually where they are hidden.

### 3.3 Where analytics enters

| Analytics task | Layer it acts on | Decision it changes | Failure if the manufacturing model is wrong |
|---|---|---|---|
| Demand forecasting | Demand plan | Aggregate volume and mix | Forecasts items the plant cannot make in that mix |
| Safety stock setting | Inventory policy | Buffer by item and level | Ignores component commonality and lot-size effects |
| Production scheduling optimisation | Finite scheduling | Sequence and machine assignment | Optimises a model of the plant, not the plant |
| Capacity planning | RCCP / CRP | Shifts, overtime, outsourcing | Uses rated capacity; misses the real constraint |
| OEE and downtime analytics | Execution | Maintenance and improvement priorities | Compares incomparable assets; hides the bottleneck |
| Yield and scrap analytics | Execution and BOM | Scrap factors and purchase quantities | Treats scrap as random when it is mix-dependent |
| Lead-time prediction | MRP parameters | Planned lead times and promise dates | Learns a queue that the plan itself creates |
| BOM and master data quality | All layers | Whether any of the above is valid | Models garbage accurately |

The last row is not a joke. The most valuable manufacturing analytics project in many companies is
not a scheduler and not a digital twin. It is a BOM comparison report that finds the lines nobody
has looked at since the last engineering change.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Bill of materials (BOM) | The structured list of components, quantities, and relationships that make one parent item | Treated as a flat parts list | Item numbers with quantity per parent |
| BOM level | Depth of an item below the end item | Confused with low-level code | Integer, 0 at the end item |
| Low-level code | The lowest level at which an item appears in any BOM | Computed once and never again | Integer |
| Phantom assembly | A logical grouping exploded through but never stocked | Treated as a stocked item with a lead time | Item flagged as phantom |
| Effectivity date | The date from which a BOM revision or line is valid | Ignored when exploding historical orders | Date range |
| Scrap factor | Expected loss at a BOM level, as a share of input | Applied by multiplying by (1 + scrap) | Percentage of input |
| Yield | Good output divided by input for a process | Used interchangeably with scrap | Percentage |
| Substitute / alternate | An item that can replace another under stated conditions | Modelled as an extra BOM line, doubling demand | Item plus ratio plus priority |
| Routing | The ordered operations required to make an item | Treated as documentation, not data | Operation sequence 10, 20, 30 |
| Operation | One step in a routing, performed at one work centre | Confused with a work order | Sequence number within a routing |
| Work centre | A grouping of machines or people that provides capacity | Confused with a single machine | Work centre code |
| Setup time | Time to prepare a work centre for a run, independent of quantity | Omitted from capacity calculations | Hours per lot |
| Run time | Time per unit produced, once set up | Confused with cycle time including queue | Hours per unit |
| Queue time | Time an order waits at a work centre before processing | Omitted because it is not on the routing | Hours or days |
| Planned lead time | The lead time the planning system uses to offset dates | Assumed to be the actual lead time | Days or weeks |
| Actual lead time | Observed elapsed time from release to completion | Reported as a mean, hiding the tail | Days, as a distribution |
| Cumulative lead time | Longest path through BOM and routings to the end item | Confused with the end item's own lead time | Weeks |
| Master production schedule (MPS) | The dated plan of end items to be produced | Confused with the forecast | Quantity by item by week |
| Material requirements planning (MRP) | Explosion of the MPS through the BOM into net, dated orders | Used to mean the whole planning system | Quantity by item by date |
| MRP II | MRP extended to capacity, shop floor, and finance | Used as a synonym for ERP | None |
| Enterprise resource planning (ERP) | Transactional system of record for orders, inventory, finance | Assumed to be a planning engine | None |
| Rough-cut capacity planning (RCCP) | Capacity check of the MPS at critical work centres | Run on MRP orders | Standard hours per week |
| Capacity requirements planning (CRP) | Capacity check of MRP planned orders at all work centres | Used interchangeably with RCCP | Standard hours per week |
| Finite scheduling | Sequencing orders so no capacity is exceeded | Confused with MRP | Sequence and timestamps |
| Advanced planning and scheduling (APS) | Constraint-based planning and scheduling, usually above ERP | Used as a marketing term for any scheduler | None |
| Load | Capacity demanded by planned or released orders | Reported without the routing version used | Standard hours per period |
| Rated capacity | Theoretical capacity at nominal shifts, staffing, speed | Used as if achievable | Hours per period |
| Demonstrated capacity | Capacity actually achieved, measured from history | Ignored in favour of rated capacity | Hours per period |
| Bottleneck | The resource whose capacity limits throughput | Assumed to be the busiest-looking machine | Work centre code |
| Drum-buffer-rope (DBR) | Constraint sets the beat, buffer protects it, rope releases work | Treated as a software product | None |
| Overall equipment effectiveness (OEE) | Availability × performance × quality | Used as a single cross-plant benchmark | Percentage |
| Availability | Uptime divided by planned production time | Inflated by reclassifying stops as planned | Percentage |
| Performance | Actual output rate divided by ideal rate | Allowed to exceed 100% because the ideal is wrong | Percentage |
| Quality | Good units divided by total units | Measured before rework | Percentage |
| Lot-for-lot | Order exactly the net requirement each period | Assumed always cheapest | Quantity |
| Fixed order quantity (FOQ) | Order a constant quantity when a requirement exists | Set once and never revisited | Quantity |
| Economic order quantity (EOQ) | Order size minimising order plus holding cost | Applied to lumpy dependent demand | Quantity |
| Period order quantity (POQ) | Order covering a fixed number of periods of demand | Confused with EOQ in periods | Quantity or periods |
| Wagner-Whitin | Dynamic programming method giving optimal lot sizes for known demand | Used where demand is not known | Quantity per period |
| Changeover | Work to convert a resource from one product to another | Confused with setup on the same product | Minutes |
| Sequence-dependent setup | Changeover time that depends on the product just made | Averaged away in planning | Matrix of minutes by pair |
| Backflush | Automatic component consumption on confirmation, from the BOM | Used where actual consumption varies | Quantity |
| Work order | Authorisation to make a quantity of an item | Confused with a planned order | Order number with dates |
| Available-to-promise (ATP) | Uncommitted supply that can be promised to a new order | Computed without capacity | Quantity by date |
| Customer order decoupling point (CODP) | Point where a specific customer order becomes the driver | Confused with the push–pull boundary | BOM stage |
| Manufacturing execution system (MES) | System that manages and records execution at ISA-95 level 3 | Assumed to be a reporting layer for ERP | None |
| ISA-95 | IEC 62264 standard defining enterprise-to-control integration | Treated as a software architecture | Levels 0 to 4 |
| Takt time | Available production time divided by demand rate | Confused with cycle time | Seconds per unit |

Two deserve expansion, because the damage from confusing them is large and repeated.

**Planned versus actual lead time.** The planning system offsets dates by a number in the item
master, usually a policy input set once and reviewed rarely. Actual elapsed time is an output of
congestion, queueing, and prioritisation. When utilisation rises, actual lead time rises and the
planned value does not move. The result is a plan that promises dates the plant cannot meet, and
then blames the plant.

**Rated versus demonstrated capacity.** Rated capacity is what the work centre could do at nominal
speed, staffing, and shifts. Demonstrated capacity is what it did. The difference is not waste; it is
the accumulated effect of changeovers, minor stops, quality losses, absenteeism, and maintenance.
Planning against rated capacity is planning against a number nobody has seen since commissioning.

## 5. The mechanics, step by step

This section follows one planning cycle from demand to shop floor, naming what each step computes
and which data it depends on, so that you can read a planning report and say which step produced
which number.

### Phase A — Demand becomes a master schedule

**Step 1. Independent demand is collected.**

- What happens: customer orders and forecasts combine into demand for end items.
- Data: sales orders, forecast, intercompany demand, service-part demand.
- Key branch: **firm or forecast?** Inside the demand time fence orders are firm and forecasts
  should be ignored. A model that mixes them without a flag learns the fence, not the market.

**Step 2. Supply is netted against demand to produce the MPS.**

- What happens: on-hand finished goods, scheduled receipts, and firm orders are netted; the
  remainder becomes the MPS quantity for a week.
- Data: inventory position, open production orders, ATP.
- Key branch: **end items or planning bills?** In assemble-to-order the MPS is often at family or
  option-group level, and the configuration is decided later. Forecast error is then measured at a
  different level from the one being scheduled.

**Step 3. The MPS is checked against key resources (RCCP).**

- What happens: MPS quantities multiply by a bill of resources — hours per unit at critical work
  centres — and are compared with capacity by week.
- Data: bill of resources, work centre capacity, shift calendar.
- Key branch: **is RCCP run at all?** In many installations it is skipped, and the first capacity
  signal arrives when the floor cannot start an order.

**Step 4. The MPS is frozen inside the time fence.**

- What happens: changes inside the fence need escalation; changes outside are absorbed by
  re-planning.
- Data: time fence policy in weeks, ideally at least the cumulative lead time.
- Key branch: **is the fence shorter than the cumulative lead time?** If it is, the plan promises
  dates for material that has not been ordered. This defect appears in no report.

### Phase B — The BOM explodes

**Step 5. Items are ordered by low-level code.**

- What happens: each item gets the lowest level at which it appears in any BOM; explosion proceeds
  from level 0 downward.
- Data: all active BOMs, including alternates and effectivities.
- Key branch: **when was the code last recalculated?** A stale code processes an item before all
  parent requirements are known, producing a shortage the next run "discovers" and expedites.

**Step 6. Gross requirements are calculated.**

- What happens: each parent planned order release is multiplied by the BOM quantity per, exploding
  through phantoms.
- Data: BOM quantities, parent order quantity, phantom flags.
- Key branch: **which revision applies on that date?** Effectivity is a date range, not a status.
  Exploding a historical order with the current revision is a silent error.

**Step 7. Net requirements are calculated.**

- What happens: on-hand inventory, scheduled receipts, and safety stock are subtracted from gross.
- Data: inventory by item and location, open orders, safety stock policy.
- Key branch: **is inventory netted at the right location?** Netting plant stock against a
  requirement at another plant produces a plan that looks feasible and ships air freight.

**Step 8. Lot size is applied, and scrap grossed up.**

- What happens: the net requirement is adjusted for scrap and rounded to the lot rule, producing a
  planned order.
- Data: scrap factor, lot-size rule, minimum order quantity, rounding value.
- Key branch: **is scrap a share of input or a multiplier on output?** For a share of input the
  correct form is `net / (1 − scrap)`. Multiplying by `(1 + scrap)` understates the requirement,
  and the error grows with the scrap rate.

**Step 9. Dates are offset backwards.**

- What happens: the receipt date moves back by the planned lead time to give the release date.
- Data: planned lead time by item, or routing-based lead time in a more advanced system.
- Key branch: **does the release date fall in the past?** If so, MRP has created an impossible
  instruction, which will be reported as past due and expedited — a capacity decision made by
  exception handling rather than planning.

### Phase C — Capacity is tested

**Step 10. Planned orders become load.**

- What happens: each order is multiplied by the routing's setup and run times per operation,
  producing standard hours by work centre by period.
- Data: routing, work centre, setup time, run time, order quantity.
- Key branch: **is load computed on the routing version that will actually be used?** A plan built
  on the old routing understates load on the operation that was added.

**Step 11. Load is compared with capacity (CRP).**

- What happens: load is compared with available capacity per period, giving utilisation and an
  overload quantity.
- Data: work centre calendar, shifts, staffing, efficiency and utilisation factors.
- Key branch: **which capacity?** Rated flatters the plan. Demonstrated tells the truth about last
  month and is the better input unless a change is planned.

**Step 12. The plan is adjusted, or it is not.**

- What happens: the planner levels the MPS, adds a shift, subcontracts, or accepts the overload and
  lets the floor resolve it.
- Data: cost of each option, labour agreements, supplier capacity.
- Key branch: **was the adjustment recorded?** An overload resolved by informal overtime is
  invisible next month, so it is "discovered" again.

### Phase D — The schedule is sequenced

**Step 13. Orders are released to the shop floor.**

- What happens: planned orders become work orders with operations, released to the first work
  centre.
- Data: planned order, routing, BOM, release policy.
- Key branch: **is release paced by the constraint or by the due date?** Releasing everything due
  floods the floor with work in process (WIP), which lengthens queue time, which makes the next
  plan worse.

**Step 14. Orders are sequenced at each resource.**

- What happens: a dispatch rule or optimiser chooses the operation order, accounting for setups,
  due dates, and material availability.
- Data: queue contents, setup matrix, priority rules, operator skills.
- Key branch: **is the setup matrix real?** A sequence-dependent model built on estimated
  changeovers produces a sequence the operators ignore, because they know better.

**Step 15. The schedule is published and regenerated.**

- What happens: the dispatch list is issued, execution deviates, and the schedule is rebuilt at a
  stated frequency.
- Data: execution feedback, machine status, material availability.
- Key branch: **how often?** Too often produces nervousness and distrust; too rarely produces a
  schedule that no longer matches the floor.

### Phase E — Execution feeds back

**Step 16. Operations are confirmed.**

- What happens: start and finish times, quantities, scrap, and downtime are recorded against the
  work order.
- Data: operator entry, machine signals, barcode or radio-frequency identification scans.
- Key branch: **real time or shift end?** Shift-end confirmation destroys intra-shift visibility,
  which is exactly when the planner needs it.

**Step 17. Components are consumed, by backflush or by issue.**

- What happens: components are deducted automatically at confirmation using BOM quantities, or
  explicitly by issue to the order.
- Data: BOM, scrap factor, actual issue records.
- Key branch: **does actual consumption differ from BOM consumption?** If it does, backflush hides
  the difference in an inventory adjustment, and the BOM error survives another cycle.

**Step 18. Performance is measured and fed back.**

- What happens: OEE, yield, adherence, and cycle time are computed and compared with plan.
- Data: confirmations, downtime reasons, quality results.
- Key branch: **is the loss taxonomy stable?** If reason codes change quarterly, the trend describes
  the coding practice, not the plant.

### 5.1 Why MRP produces impossible dates

MRP produces exactly the dates it was asked to produce, under assumptions that are usually false.
Six mechanisms account for most of the gap.

1. **Infinite capacity.** MRP assumes every work centre can process all load assigned to it. It does
   not sequence, so it cannot see two orders needing the same machine in the same week.
2. **Fixed planned lead times.** Lead time is an input, not an output. The same order takes longer
   in a busy month, and MRP cannot represent this.
3. **Backward scheduling from an unachievable date.** If the due date is earlier than the cumulative
   lead time, the start date is in the past. The system reports a past-due order rather than
   refusing the date.
4. **Lot sizing and rounding at every level.** A requirement of 235 becomes an order of 250, which
   becomes a component requirement of 250, which becomes a purchase of 275. Each rounding is
   rational locally and inflates the plan globally.
5. **Scrap and yield treated as constants.** Scrap depends on mix, run length, operator, and
   material. One factor per item is an average that is wrong in both directions at once.
6. **No supplier capability check.** MRP assumes a purchase order can be filled on the requested
   date. Supplier capacity, minimum order quantities, and transport frequency are not in the model.

### 5.2 Three lead times that never agree

| Lead time | Where it comes from | Typical value | Used for |
|---|---|---|---|
| Planned | Item master or routing | 10 working days | Offsetting MRP dates |
| Scheduled | Current finite schedule | 14 working days | Committing to the next order |
| Actual, recent | Confirmations, last 12 weeks | 19 working days, with a long tail | What the customer experiences |

They diverge because the planned value is a policy, the scheduled value reflects current queue, and
the actual value includes every disruption that has already happened. An analyst who trains a
lead-time model on actuals and writes the output into the planned lead-time field has built a
feedback loop: longer observed lead times cause earlier releases, which raise WIP, which raises
queue time, which lengthens observed lead times.

## 6. Worked example

Meridian Pumps makes a centrifugal pump skid, item `PS-100`, at a single plant. All figures are
illustrative but internally consistent. The bucket is one week. ISO week numbering is used, so week
20 begins Monday 11 May 2026.

### 6.1 The product structure

```text
   PS-100  pump skid                      LEVEL 0   make, final assembly
     |
     +--(1 EA)---- PUMP-END               LEVEL 1   make, subassembly
     |               |
     |               +--(1 EA)---- IMP-07      LEVEL 2   buy, impeller
     |               +--(2 EA)---- SEAL-02     LEVEL 2   buy, mechanical seal
     |               +--(1 EA)---- CASING-09   LEVEL 2   buy, casing
     |
     +--(1 EA)---- BASE-01                LEVEL 1   make, fabricated base
     +--(1 EA)---- MOTOR-15               LEVEL 1   buy, electric motor
```

Low-level codes: `PS-100` = 0, `PUMP-END` = 1, `BASE-01` = 1, `MOTOR-15` = 1, `IMP-07` = 2,
`SEAL-02` = 2, `CASING-09` = 2. Explosion proceeds 0, then 1, then 2.

### 6.2 Planning data

| Item | Level | Source | Planned lead time | Lot-size rule | Scrap | On hand |
|---|---:|---|---:|---|---:|---:|
| PS-100 | 0 | Make | 1 week | Lot-for-lot | 0.0% | 0 |
| PUMP-END | 1 | Make | 2 weeks | Fixed 50 | 2.0% | 10 |
| BASE-01 | 1 | Make | 1 week | Lot-for-lot | 1.0% | 0 |
| MOTOR-15 | 1 | Buy | 4 weeks | Fixed 25 | 0.0% | 5 |
| IMP-07 | 2 | Buy | 3 weeks | Fixed 30 | 3.0% | 20 |
| SEAL-02 | 2 | Buy | 5 weeks | Fixed 50 | 1.0% | 40 |
| CASING-09 | 2 | Buy | 3 weeks | Fixed 25 | 1.5% | 0 |

No scheduled receipts, and safety stock is zero for every item so the arithmetic stays visible.
Demand: one customer order for **240 units of PS-100, due in week 20**.

### 6.3 The explosion

Convention: scrap is a share of input, so gross requirement is `net / (1 − scrap)`, rounded up to a
whole unit; the lot rule is then applied, and the result is the planned order.

| Item | Gross req | Week | On hand | Net | Scrap | Grossed up | Lot rule | Planned order | Release wk | Receipt wk |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| PS-100 | 240 | 20 | 0 | 240 | 0.0% | 240.00 | L4L | 240 | 19 | 20 |
| PUMP-END | 240 | 19 | 10 | 230 | 2.0% | 234.69 | 50 | 250 | 17 | 19 |
| BASE-01 | 240 | 19 | 0 | 240 | 1.0% | 242.42 | L4L | 243 | 18 | 19 |
| MOTOR-15 | 240 | 19 | 5 | 235 | 0.0% | 235.00 | 25 | 250 | 15 | 19 |
| IMP-07 | 250 | 17 | 20 | 230 | 3.0% | 237.11 | 30 | 240 | 14 | 17 |
| SEAL-02 | 500 | 17 | 40 | 460 | 1.0% | 464.65 | 50 | 500 | 12 | 17 |
| CASING-09 | 250 | 17 | 0 | 250 | 1.5% | 253.81 | 25 | 275 | 14 | 17 |

Read it as a chain. The PS-100 order of 240, released in week 19, becomes the gross requirement for
its three level-1 children in week 19. The PUMP-END planned order of 250, released in week 17,
becomes the gross requirement for its three level-2 children in week 17.

`SEAL-02` is worth checking by hand, because it carries both a quantity per and a scrap factor:

```text
gross requirement   = 2 per skid x 250 skids        = 500 seals
net of on-hand      = 500 - 40                      = 460
grossed up for 1%   = 460 / 0.99                    = 464.65
rounded up          = 465
lot size 50         = 465 / 50 = 9.3 -> 10 lots     = 500
release date        = week 17 receipt - 5 weeks     = week 12
```

### 6.4 Two rounding effects that matter

**Lot sizing at one level changes demand at the next.** PUMP-END's net requirement was 230 and its
grossed-up requirement 234.69. Rounding to a fixed lot of 50 produced 250 — 8.7% more than needed.
Every level-2 requirement is computed from 250, not from 235. The extra subassemblies consume
components, and if they are not built they sit as WIP. Component demand is not a simple multiple of
end-item demand.

**Scrap compounds downward.** The end item had 0% scrap, but by level 2 the purchase carries the
accumulated effect: 500 seals cover 250 skids at 2 per skid, of which 1% is expected to be scrapped
in subassembly, of which 2% of subassemblies are expected to be scrapped before they reach the skid.
The purchase is 8.7% above the naive `240 × 2 = 480`, and every point is traceable to a BOM level.

### 6.5 Lead-time offsetting and the promise

The release dates are not independent. They form the cumulative lead time: the longest path from a
purchased component to the finished item.

| Path | Weeks consumed | Cumulative |
|---|---:|---:|
| SEAL-02 release (wk 12) to receipt (wk 17) | 5 | 5 |
| PUMP-END release (wk 17) to receipt (wk 19) | 2 | 7 |
| PS-100 release (wk 19) to receipt (wk 20) | 1 | 8 |

The cumulative lead time is **8 weeks**. If a customer asks on Monday of week 12 for delivery in
week 20, the plant can just do it, and only if every purchase order is placed that day. If the same
customer asks in week 15 for week 20, the answer is no: the seals cannot arrive before week 20, so
the skid cannot be assembled before week 21, and the earliest realistic promise is week 22 once
queue time is included.

This is the number that belongs in a promise-date policy. The sales team's system probably shows the
`PS-100` lead time of **1 week**, because that is the item master value for the item being quoted.
The difference between 1 week and 8 weeks is the difference between a promise and a failure.

### 6.6 The capacity check

Final assembly runs at work centre `WC-ASSY`: four stations, two shifts, 37.5 hours per station per
week. Rated capacity is 300 standard hours per week. Demonstrated capacity over the last 26 weeks is
280 hours per week, or 93.3% of rated.

The `PS-100` routing at `WC-ASSY` has one operation: setup 8 hours per lot, run 1.35 hours per unit.

```text
load, week 19   = 8 + (240 x 1.35)
                = 8 + 324
                = 332 standard hours

utilisation vs demonstrated = 332 / 280 = 118.6%
utilisation vs rated        = 332 / 300 = 110.7%

capacity in units, demonstrated = (280 - 8) / 1.35 = 201.5 -> 201 units
shortfall                       = 240 - 201 = 39 units
hours of overload               = 332 - 280 = 52 hours
```

The week is overloaded under either definition. Every other week in the horizon carries no `PS-100`
load at all, because the MPS put all 240 units in a single week.

Two other work centres are worth checking, because the answer is not always the obvious one:

| Work centre | Load week | Setup h | Run h | Total load | Demonstrated capacity | Utilisation |
|---|---:|---:|---:|---:|---:|---:|
| WC-ASSY (final assembly) | 19 | 8.0 | 324.0 | 332.0 | 280 | 118.6% |
| WC-SUB (subassembly) | 17 | 4.0 | 162.5 | 166.5 | 220 | 75.7% |
| WC-FAB (base fabrication) | 18 | 2.0 | 194.4 | 196.4 | 200 | 98.2% |

The overload is at final assembly, not at the work centre that looks busiest in aggregate.
`WC-FAB` at 98.2% is not overloaded, but it has no protection: any absence or tool failure in week
18 delays the base and therefore the skid. A planner who reports only "which work centre is over
100%" will miss this.

> **Note:** this is a single-product capacity check. A real CRP run aggregates every order at every
> work centre. The structure of the answer does not change; only the number of rows does. The
> teaching point is that the capacity check is a separate calculation the material plan never
> performs for you.

### 6.7 What the planner can actually do

| Option | What it does | Effect on the 240 | Incremental cost |
|---|---|---|---|
| Overtime in week 19 | Adds 52 hours across four stations, 13 each | All 240 complete end of week 19, on time | 52 h × €38/h = €1,976 |
| Split build across weeks 19–20 | 201 units week 19, 39 units week 20 | 201 on time, 39 one week late | Chargeback 39 × €250 = €9,750 |
| Expedite PUMP-END by one week | Release week 16, assemble weeks 18–19 | All 240 on time | 6% premium on €129,050 of parts = €7,743 |
| Level the MPS across weeks 19–20 | 120 per week, two setups | All 240 by end of week 20; 120 late | Extra setup 8 h ≈ €304 plus late-delivery exposure |

Overtime is the cheapest way to protect the date. The expedite is four times more expensive because
it pays a premium on the entire level-2 bill rather than on the hours actually short. The split
build is most expensive here because the chargeback is large, but it would be cheapest if the
customer accepted a one-week slip at no penalty.

There is a fifth option not on the list because it is not the planner's to take alone: sell 201 and
lose the rest. It is worth naming, because in many companies the plan is quietly executed this way
and the shortfall is reported as a "supply issue".

## 7. Code, query, or calculation

This topic is calculation-first. The table in section 6.3 is the centrepiece, and the block below is
the smallest complete statement of what MRP does, so that you can reproduce it by hand or in a
spreadsheet. It is not production code.

```text
for each item in ascending low-level code:
    for each period t in the horizon:
        gross[t]   = sum over parents p of
                       planned_release[p][t + offset(p)] x qty_per(p -> item)
        available  = on_hand + scheduled_receipts[t] - safety_stock
        net[t]     = max(0, gross[t] - available)
        if net[t] > 0:
            qty    = ceil( net[t] / (1 - scrap) )        # scrap is a share of input
            qty    = lot_rule(qty)                       # L4L, FOQ, POQ, EOQ, optimal
            receipt[t]      = qty
            release[t - LT] = qty                        # backward offset
            if t - LT < current_period: flag("past due")

# capacity check on the resulting orders
load[wc][t] = sum over orders o at wc in t of ( setup[o] + qty[o] x run_time[o] )
utilisation = load / demonstrated_capacity
overload    = max(0, load - demonstrated_capacity)
```

Three checks on any implementation:

- **Units.** `qty_per` must be in the same unit as the inventory record, not the purchase unit. A
  BOM in kilograms and an inventory record in drums is wrong by the drum factor, and it looks
  plausible.
- **Scrap convention.** If the source defines scrap as a share of output, the formula is
  `net × (1 + scrap)`. The two differ by `scrap²` — negligible at 1%, material at 10%. Document it.
- **Rounding.** Round up before applying the lot rule, not after. Rounding a lot multiple down
  creates a shortage that the next run fixes by expediting.

## 8. Data, standards, and regulatory context

Manufacturing data is governed less by trade regulation than by an architecture standard, a KPI
standard, and a set of quality-system rules. Knowing which is which tells you where a number comes
from and whether you may compare it with another number.

### 8.1 ISA-95 and the level model

ISA-95, published internationally as IEC 62264, defines the integration between enterprise systems
and control systems. Its level model is the reference architecture for "which system owns what". The
2025 revision of Part 1 (ANSI/ISA-95.00.01-2025) restates the models and terminology; Parts 2 to 8
cover object models, activity models, and information exchange profiles.

| Level | Name | Time horizon | Data it owns | Typical system |
|---|---|---|---|---|
| 0 | Physical process | Instant | The process itself | Equipment |
| 1 | Sensing and manipulation | Milliseconds to seconds | Sensor readings, actuator states | Sensors, programmable logic controllers |
| 2 | Monitoring and supervisory control | Seconds to hours | Setpoints, alarms, supervisory control | SCADA, distributed control systems |
| 3 | Manufacturing operations management | Seconds to days, shifts | Work orders as executed, dispatch, confirmations, genealogy, equipment state, quality results | MES, laboratory systems |
| 4 | Business planning and logistics | Days to months | Customer orders, material master, financial inventory, production plan, purchasing | ERP |

The standard is principally concerned with the level 3 to level 4 interface. That interface is where
most manufacturing analytics disputes originate, because both sides believe they own the production
quantity.

Three consequences:

- **Time semantics differ.** Level 3 works in seconds, minutes, shifts, and days; level 4 in days,
  weeks, and months. "Real time" at level 4 often means "after the nightly interface", which is a
  day late at level 3.
- **The unit of record differs.** Level 3 records what happened on a specific order at a specific
  work centre; level 4 records an aggregated inventory movement. They reconcile only if the
  aggregation rule is documented.
- **The identifier differs.** An ERP material number and an MES material number are frequently not
  the same string, and the mapping is maintained by hand.

> **Note:** ISA-95 is an architecture model, not a software product and not a legal requirement. A
> company can be fully conformant in its interfaces and still have terrible data. The standard tells
> you where to look for the system of record; it does not tell you whether the record is right.

### 8.2 The exchange and KPI standards

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| ANSI/ISA-95 / IEC 62264 | Enterprise-to-control integration | Level definitions, object models, activity models, exchange profiles | B |
| B2MML | XML implementation of ISA-95 | Message schemas for operations definition, schedule, and performance | B |
| OPC UA (IEC 62541) and OPC 10030 | Industrial interoperability and ISA-95 mapping | Machine information model; shared equipment and material identity | B |
| MTConnect | Machine tool data | Read-only machine and process data model | B |
| ISO 22400-1 and -2 | Manufacturing KPIs | Definitions and formulas for OEE and related indicators | B |
| ISA-88 / IEC 61512 | Batch control | Recipe hierarchy, process model, batch production records | B |
| GS1 GTIN and GLN | Material and location identity | What the item and the site are called across systems | B |

Two matter most for analytics. ISO 22400-2 defines OEE in terms of time elements — planned operation
time, planned unit time, actual unit time, actual productive time — and specifies the formula for
each KPI. If your OEE does not match those definitions, it is a local metric wearing a standard's
name. B2MML makes the level 3 to level 4 interface queryable: it defines message types for operations
schedule and operations performance, which is how a confirmation becomes an analyzable event rather
than a screen.

ISA-88 matters when the product is made in batches. It defines the recipe hierarchy — general, site,
master, and control recipes — and the process model of process, stage, operation, and action. A batch
record is a legal record in regulated industries, and the recipe is the object that links the BOM to
the equipment.

### 8.3 Regulated manufacturing and records

| Requirement | Where it applies | What it constrains | Grade |
|---|---|---|---|
| ISO 9001:2015 | Global, voluntary certification | Quality records and traceability of change | B |
| IATF 16949 | Automotive supply chain | Part approval, traceability, change control | B |
| FDA 21 CFR Part 11 | United States, regulated products | Electronic records and signatures | B |
| EU GMP Annex 11 | European Union, medicinal products | Computerised systems in GMP environments | B |
| EU Medical Device Regulation | European Union, devices | Device traceability and unique device identification | B |

### 8.4 What is convention rather than requirement

- **"World-class OEE is 85%."** Conventionally traced to Nakajima (1984), from availability above
  90%, performance above 95%, quality above 99%. A useful aspiration, a poor benchmark.
- **"BOM accuracy must be 98%."** From the Oliver Wight Class A checklist, a consulting framework,
  not a standard.
- **"Schedule attainment must be 95%."** Same source, same status.
- **A five-day week with two shifts** is a calendar convention. Capacity is meaningless without the
  calendar, and many reports omit it.
- **Standard cost** is an accounting convention, not a physical quantity. Variances measure the
  standard as much as the plant.

## 9. Industry practice and case evidence

### 9.1 MRP is universal and spreadsheets are not going away

| Source | Sample and date | Finding | Grade |
|---|---|---|---|
| Qlector, survey of large Slovenian manufacturers | 2025 | ERP in all, MES in three of four, MRP "universal"; 50% still use Excel or paper as a planning supplement; bad planning estimated at 5–10% of annual revenue | D |
| Capterra, MRP software research | Published April 2026, about 1,850–2,070 buyer interactions | 23% of manufacturers rely on spreadsheets for planning; 25% use manual methods; 37% cite planning inefficiency as the trigger to switch | D |
| HSO, Technology Maturity in Manufacturing 2026 | UK, August 2026 | ERP widely used but Excel "almost as prominent"; 6% rate systems excellent; 88% say systems are not well integrated | D |
| Rootstock, 2026 State of Manufacturing Technology | January 2026, n = 520 | Talent shortage now the fastest-rising barrier to digital transformation (33%); 48% expect AI to require reskilling | D |
| FIR (RWTH Aachen), Future of Production Planning | Survey February–June 2025 | Data maintenance, not technology or acceptance, is the biggest bottleneck; planning becomes "more automated, but not autonomous" | D |

### 9.2 Data accuracy determines whether MRP delivers anything

- A 1995 study in the *Journal of Operations Management* used alternating conditional expectation to
  analyse MRP benefits and found **data accuracy to be the critical determinant** of operational
  efficiency, customer service, and coordination benefits, with nonlinear effects and little gain
  from partial integration (Grade A).
- Cremers (2016), a TU/e thesis at VDL Nedcar, measured a 12,255-line BOM and found a **5.5% data
  error rate against a target below 0.5%**, with correction work consuming about 3.3 hours per day
  across material-handling gates (Grade C, single-company thesis).
- Stip and van Houtum (2019) showed that incorrect service BOMs cause under- and over-forecasting of
  spare-parts demand, and that a multi-source alerting method flagged errors with **95% precision**,
  cutting spare-part non-availabilities by roughly 4–5% per year (Grade A).

The arithmetic of why 95% BOM accuracy is not enough is simple. In the worked example, `SEAL-02` has
a quantity of 2 per skid. A BOM that says 1.9 — a 5% error on one line — produces:

```text
required, correct    = 250 x 2.0 = 500 seals
required, with error = 250 x 1.9 = 475 seals
after netting 40 and 1% scrap = (475 - 40) / 0.99 = 439.4 -> 440 -> lot 50 -> 450
shortfall against the correct requirement = 500 - 450 = 50 seals
units that cannot be built = 50 / 2 = 25 skids (10% of the order)
```

A 5% error on one line stops 10% of the order, because the component is used twice per unit and lot
sizing rounded the shortage in the wrong direction. This is the mechanism behind the convention that
BOM accuracy is held to 98% while other master data is held to 95% (Grade D for the threshold, A for
the mechanism).

### 9.3 Schedule nervousness is a measured phenomenon

- Blackburn, Kropp, and Millen (1986, *Management Science*) compared five dampening strategies and
  found that **freezing the schedule and change-cost penalties dominate**, while safety stock and
  lot-for-lot below level 0 are not cost-effective when nervousness comes from rolling re-planning
  (Grade A).
- Kadipasaoglu and Sridharan (1995, *Journal of Operations Management*) found freezing the master
  schedule most effective and least context-sensitive, once the freeze covers the cumulative lead
  time (Grade A).
- Li and Disney (2017, *IJPR*) reframed nervousness as period-to-period variability in future orders
  to suppliers and linked it to the bullwhip effect, showing that order-up-to policies with
  predictable-future adjustments reduce both (Grade A).
- Xue, Davis, Kanet, and Cantrell (2002) found no universal parameter setting that improves
  instability, cost, and service simultaneously (Grade A).

For an analyst, the implication is that **plan stability is a metric in its own right**. A scheduler
that cuts cost 2% while changing 40% of the sequence every day has not improved the system; it has
moved cost into expediting, overtime, and lost trust.

### 9.4 APS and MRP: what the evidence supports

- Vieira et al. (2021), a systematic literature review of APS, traces the evolution from MRP to MRP
  II to ERP to APS and concludes that APS **complements rather than replaces** ERP. It reports
  persistent difficulties: heavy dependence on human expertise, continued spreadsheet reliance
  especially in smaller firms, low data accuracy, high implementation cost, and weak integration
  with AI and Industry 4.0 practice so far (Grade A).
- Hvolby and Steger-Jensen (2010, *Computers in Industry*) concluded that APS functionality is
  "radically improved compared to MRP and ERP" but stressed the need for expert users to configure
  many conflicting planning parameters (Grade A).
- The 2025 vendor material is consistent in framing and should be read as marketing: MRP uses
  infinite capacity and fixed lead times; APS is finite-capacity and constraint-based; APS is the
  answer (Grade D).

### 9.5 Theory of constraints: evidence and limits

- Thürer, Stevenson, Silva, and Qu (2017, *IJPE*) compared drum-buffer-rope with workload control
  across job-shop and flow-shop configurations at 90% bottleneck utilisation. Workload control did
  better at low and moderate bottleneck severity; **DBR outperformed all workload-control methods
  under severe or very severe bottlenecks** (Grade A).
- The same paper cites Mabin and Balderstone's review of more than 80 TOC implementations, in which
  about **80% reported lead-time and due-date improvements** (Grade A for the review; the cases are
  self-reported).
- Watson and Patti (2008, *IJPR*) found DBR beat kanban/JIT on output and lead time with less
  inventory in a simulated cell with unplanned downtime (Grade A).
- Umble, Umble, and Murakami (2006, *IJPR*) documented a textbook DBR implementation at Hitachi Tool
  Engineering, with improvements across WIP, lead time, on-time delivery, and inventory turnover
  (Grade C, single case).

### 9.6 OEE: the benchmark that should not be a benchmark

| Claim | Source | Status |
|---|---|---|
| 85% is world class | Conventionally traced to Nakajima (1984): >90% availability, >95% performance, >99% quality | Convention, not a standard |
| Typical OEE is far below 85% | Ljungberg 55%; Ylipää et al. 51.5%; Hedman et al. 65% average, 70% median, as reported in Bengtsson et al. (2020) | Grade A (secondary reporting) |
| OEE is not a good cross-company or cross-machine comparison | Bengtsson, Andersson, and Ekström (2020) | Grade A |
| Distribution peaks at 55–60%; about 6% of organisations exceed 85% annually | Evocon, 50+ countries, mid-2023 to mid-2024 | Grade D |
| True share above 85% is closer to 3% once definitional blind spots are corrected | Evocon, same source | Grade D |
| Performance above 100% signals a lowered speed baseline | Evocon, May 2025 | Grade D, technically correct |
| A single plant-wide OEE aggregate is invalid; only like assets may be compared | LNS Research | Grade D |

The mechanism is not statistical subtlety. Both numerator and denominator are defined locally. A
plant that classifies changeover as "planned" removes it from availability. A plant that uses the
fastest observed cycle as the ideal rate reports performance below 100%; a plant using the datasheet
rate may report above 100%. A high-mix plant has structurally more changeover and therefore lower
availability than a single-product plant making the same family.

### 9.7 Adoption, and the 2026 framing

| Claim | Source | Grade |
|---|---|---|
| 54% of small and medium plants globally still use pen and paper or spreadsheets as their de facto MES; about 8% use a commercial MES | IoT Analytics, MES Market Report 2025–2031, December 2025 | D |
| 93% of large manufacturers have an MES, but only 23% have it fully integrated and 28% enterprise-wide | Rockwell Automation, State of Smart Manufacturing, 2025, 1,560 respondents | C |
| 48% of German manufacturers use digital twins in production, up from 33% in 2022 | Bitkom Research, April 2025, 552 companies | C |
| 32% report operational digital twins and 32% piloting; 65% report bidirectional data exchange | Zech, Clark, and Breu, HICSS 2025, 103 responses | A |
| About 41% of firms are piloting digital twins, 20% report full integration | Manufacturing IT/OT Trend Report 2025, via PEX Network | D |

The gap between "has an MES" and "has an integrated MES" is the gap that matters. An MES that
confirms at shift end and exports a flat file nightly is, for modelling purposes, a daily batch
source with a one-day lag, whatever the architecture diagram says.

The 2026 vendor and analyst material is optimistic about autonomous scheduling and cautious about
its maturity, and the peer-reviewed literature supports the caution. Gartner's 2026 manufacturing
material frames the direction as semi-autonomous AI agents, software-defined products, and
closed-loop digital twins by 2030, and argues that scaling "physical AI" is gated by IT/OT
integration rather than model capability (Grade D). Shabur (2026), a PRISMA review of 118 studies in
*IET Collaborative Intelligent Manufacturing*, finds agentic methods mature in diagnostics and
localised execution but a structural **deployment gap in real-time prescriptive decision-making**,
reporting that **68% of production agents run ten steps or fewer before human intervention and 74%
still rely primarily on human evaluation**, driven by explainability, safety verification, and
cybersecurity gaps (Grade A). The same review names hybrid agent-plus-digital-twin architectures as
the fastest-growing convergence point, with OPC UA, ISA-95, and MTConnect as enabling standards.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| MRP dates treated as promises | Sales quotes the item master lead time, not the cumulative lead time | MRP offsets backwards and never checks feasibility | Late orders, expediting, lost margin | Publish cumulative lead time by family; gate promises on it |
| BOM accuracy accepted at 95% | Shortages on high-quantity-per lines | Error compounds through quantity per and lot rounding | 10% of an order unbuildable from a 5% line error | Hold BOM accuracy at 98%; audit high-usage lines first |
| Phantom treated as stocked | A phantom appears with an inventory balance | MRP explodes through it but someone created a stock record | Fictitious inventory, phantom lead time | Flag phantoms; exclude from counting and netting |
| Scrap convention inverted | Purchases consistently 2–5% short at high scrap | `× (1 + scrap)` used where `÷ (1 − scrap)` is correct | Chronic small shortages | Document the convention; test at 10% scrap |
| Infinite capacity accepted | Every work centre planned above 100% | MRP does not sequence or constrain | Plan infeasible on day one | Run CRP weekly; treat overload as a decision |
| Routing and BOM out of sync | An operation consumes material not on the BOM | Separate maintenance cycles and owners | Unplanned material, wrong cost and load | Reconcile both at every engineering change |
| Planned lead time never updated | Actual lead time drifts up for years | Policy input, not measured output | Systematic lateness; mis-sized safety stock | Refresh quarterly from confirmations |
| Queue time omitted | Lead time is close to run time on paper | Routing records only setup and run | Plan understates lead time by 80% or more | Measure queue separately; model lead time as a distribution |
| OEE compared across plants | One plant always "wins" | Different definitions, mix, structural losses | Wrong investment, demotivated teams | Compare within asset and mix; decompose into A, P, Q |
| OEE above 90% or performance above 100% | Implausibly good numbers | Ideal cycle time set too slow; stops reclassified | Metric loses credibility | Audit the ideal rate and the loss taxonomy annually |
| Lot sizing ignores changeover | Many small lots on a setup-dominated resource | EOQ balances order and holding cost, not changeover | Capacity consumed by setups | Use sequence-aware lot sizing where setup dominates |
| Backflush hides yield loss | Recurring unexplained inventory adjustments | Components consumed at BOM quantity regardless of actual | BOM error persists; variance unexplained | Cycle-count high-value components; reconcile consumption |
| ERP and MES both own production quantity | Two "production this week" numbers | Both levels record output | Reporting disputes; no trusted number | Assign system of record per noun; publish the RACI |
| Schedule regenerated too often | Planners ignore the schedule | Nervousness from rolling re-planning | Expediting, overtime, lost trust | Freeze inside a fence; measure schedule stability |
| Capacity measured in units with changing mix | "Capacity" moves without investment | Units are not homogeneous across products | Wrong load, wrong promises | Measure load in standard hours |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The BOM is complete and current | Every explosion depends on it | Engineering changes not propagated; alternates missing | Sample build: compare issued components to BOM |
| The routing describes the operations actually performed | Load calculation depends on it | Process drift; unrecorded rework | Observe a run; compare to routing |
| Scrap is a stable proportion of input | A single factor is used per item | Mix-dependent yield; supplier changes | Plot scrap by lot size and period |
| Planned lead times approximate reality | Dates are offset by them | Congestion; priority changes; supplier delay | Compare planned against confirmed |
| Capacity is measurable in standard hours | Load and capacity must share a unit | Mix effects; learning; inaccurate standards | Re-time a sample; compare standard to actual |
| The planning bucket is fine enough | Weekly buckets hide intra-week overload | Short cycle times; daily shipping | Check whether a week's load can be sequenced |
| On-hand inventory is accurate and unallocated | Netting depends on it | Unrecorded issues; allocation; consignment | Cycle counts; allocation report |
| Demand inside the fence is firm | Freezing is the main anti-nervousness device | Expedite culture; sales overrides | Count fence violations per month |
| The constraint is stable over the horizon | TOC and capacity planning depend on it | Mix change moves the bottleneck | Load profile by work centre over time |

One limit of the framework are worth stating plainly.

**MRP assumes a deterministic world and is used in a stochastic one.** Every quantity in the
explosion is a point estimate. Demand, yield, lead time, and capacity are distributions. MRP's answer
is the centre of a distribution it never computes. That is acceptable for planning and unacceptable
for promising, which is why the promise date belongs to a separate process that accounts for
variability.

## 12. Exercises

### Exercise 1 — Classify the archetype (recall and application, 30 minutes)

For each, name the manufacturing archetype, state where the customer order decoupling point sits,
and name one critical and one nearly irrelevant planning datum: (a) a bakery supplying supermarkets;
(b) a maker of pressure vessels built to customer specification; (c) a laptop manufacturer offering
three processor, two memory, and two storage options; (d) a specialty chemical plant running one
product for eleven months.

**Deliverable:** a four-row table with archetype, CODP, critical datum, irrelevant datum.
**Acceptance criteria:** the CODP is at a named BOM stage, not "at the warehouse"; the critical datum
is specific (for example "option take rates", not "demand"); each justification is one sentence.

### Exercise 2 — Explode a three-level BOM (calculation, 45 minutes)

Using section 6.2, change `SEAL-02` to 3 per skid and `PUMP-END` scrap to 4%. Recompute the full
explosion and the release dates.

**Deliverable:** the completed table for all seven items.
**Acceptance criteria:** scrap as a share of input; rounding up before the lot rule; release weeks
consistent with lead times; `SEAL-02` arithmetic shown longhand.

### Exercise 3 — Find the overload (calculation, 40 minutes)

Using section 6.6, add a second product `PS-200` at `WC-ASSY`: 60 units in week 20, setup 10 hours,
run 2.1 hours per unit. Recompute load for weeks 19 and 20 and identify the new constraint.

**Deliverable:** a load table for both weeks with utilisation against rated and demonstrated
capacity, and a one-paragraph recommendation.
**Acceptance criteria:** both capacity definitions used; overload in hours and in units; the
recommendation names the constraint it relaxes and its approximate cost.

### Exercise 4 — Compute OEE and decompose it (calculation, 30 minutes)

A line runs two shifts of 7.5 hours, five days a week. It loses 4.5 hours to breakdowns and 3 hours
to changeovers that are classified as planned. It produces 8,400 units against an ideal rate of 140
units per hour. 210 units are rejected and 140 are reworked and sold as good.

**Deliverable:** availability, performance, quality, and OEE, with the definition of planned
production time stated, plus a second calculation treating changeover as unplanned.
**Acceptance criteria:** each component with numerator and denominator; the difference between the
two OEE values explained; rework treatment stated.

### Exercise 5 — Diagnose the impossible date (critique, 40 minutes)

A planner reports that MRP generated 340 past-due orders this week, up from 40 last week. Write a
diagnosis.

**Deliverable:** three plausible mechanisms ranked by likelihood, the query you would run for each,
and the one you would check first.
**Acceptance criteria:** at least one mechanism is a data artifact rather than a capacity problem;
each names the specific field or table involved; the first check is justified.

### Exercise 6 — Choose a lot size (design, 45 minutes)

An item has annual demand of 12,000 units, ordered in 12 equal monthly batches, setup cost €180 per
order, and holding cost €4.50 per unit per year. Demand is level. Compute EOQ, compare it with the
current policy, and state when EOQ would be the wrong rule.

**Deliverable:** EOQ, current order quantity, annual cost under each, and three invalidating
conditions.
**Acceptance criteria:** costs with units; setup plus holding included; conditions specific to
dependent demand, capacity, or shelf life.

### Exercise 7 — Map the data to ISA-95 (application, 40 minutes)

For a plant you know or can research, list six data objects (work order, confirmation, downtime
event, material issue, quality result, schedule) and assign each to an ISA-95 level and a system of
record.

**Deliverable:** a six-row table with object, level, system of record, time granularity, direction of
flow.
**Acceptance criteria:** exactly one system of record per object; granularity in units; at least one
object where the level is contested in practice.

### Exercise 8 — Price the options (design, 60 minutes)

Using section 6.7, change the chargeback to €80 per unit per week and the overtime premium to €52 per
hour. Re-rank the four options.

**Deliverable:** a revised options table, a recommendation, and the break-even chargeback at which
the ranking changes.
**Acceptance criteria:** all options costed on the same basis; the break-even computed, not
estimated; the recommendation states what would reverse it.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
PUMP-END: gross 240 week 19, on hand 10, net 230
          230 / (1 - 0.04) = 239.58 -> 240
          lot 50 -> 250 (unchanged from the base case)
          release week 17, receipt week 19

SEAL-02:  gross = 3 x 250 = 750 in week 17
          net = 750 - 40 = 710
          710 / 0.99 = 717.17 -> 718
          lot 50 -> 750
          release week 12, receipt week 17

IMP-07:    gross 250, on hand 20, net 230, / 0.97 = 237.11 -> 238, lot 30 -> 240
CASING-09: gross 250, / 0.985 = 253.81 -> 254, lot 25 -> 275
```

**Exercise 3 arithmetic.**

```text
week 19 load = 8 + 240 x 1.35  = 332.0 h   (PS-100 only)
week 20 load = 10 + 60 x 2.1   = 136.0 h   (PS-200 only)
pair capacity, demonstrated    = 560 h
332 / 280 = 118.6%   (week 19)      136 / 280 = 48.6%   (week 20)
```

**Exercise 4 arithmetic.**

```text
scheduled time = 2 x 7.5 x 5 = 75 h per week
planned production time, changeover planned = 75 - 4.5 - 3.0 = 67.5 h
availability   = 67.5 / 67.5 = 100.0%   (changeover removed as planned)

ideal output   = 67.5 x 140 = 9,450 units
performance    = 8,400 / 9,450 = 88.9%
quality        = (8,400 - 210) / 8,400 = 97.5%   (rework counted as good)

OEE = 1.000 x 0.889 x 0.975 = 86.7%
```

With changeover treated as unplanned, planned production time becomes 70.5 hours, availability is
`70.5 / 75 = 94.0%`, performance is `8,400 / (70.5 × 140) = 85.1%`, and OEE is
`0.940 × 0.851 × 0.975 = 78.0%`. The nine-point swing between the two conventions is the whole
argument of section 9.6.

If a submission reports performance above 100%, the ideal rate has been set too low — the standard
red flag. Performance of 88.9% with 100% availability usually means the ideal rate is a datasheet
value the line has never sustained.

**Exercise 6 arithmetic.**

```text
EOQ = sqrt(2 x D x S / H) = sqrt(2 x 12,000 x 180 / 4.50)
    = sqrt(960,000) = 979.8 -> 980 units

current policy = 12,000 / 12 = 1,000 units per order

annual cost at EOQ   = (12,000 / 980) x 180 + (980 / 2) x 4.50
                     = 2,204 + 2,205 = 4,409
annual cost at 1,000 = 12 x 180 + 500 x 4.50 = 2,160 + 2,250 = 4,410
```

**Exercise 1 model answer.**

(a) Make-to-stock, decoupling point at finished goods in the regional distribution centre; critical
datum is shelf life and daily demand variability; nearly irrelevant is the customer's specific
configuration. (b) Engineer-to-order, decoupling point at raw material or design release; critical
datum is engineering release date and long-lead purchased items; nearly irrelevant is a statistical
forecast of finished units. (c) Assemble-to-order, decoupling point at option inventory; critical
datum is option take rates and component commonality; nearly irrelevant is the forecast of a specific
configuration. (d) Process, continuous or campaign batch, decoupling point at the bulk tank; critical
datum is campaign length and changeover cost; nearly irrelevant is unit-level order tracking.

**Exercise 5 model answer.**

Mechanisms, ranked: (1) a master-data change that shortened planned lead times across many items,
moving receipts earlier and creating past-due releases; (2) a large order or demand spike that pulled
requirements into the near term; (3) a real capacity loss, such as a work centre down for a week,
which caused slippage the next run re-planned backwards. Check the master-data change log first: a
jump from 40 to 340 in one week is more consistent with a parameter change than with demand. The
query compares planned lead time by item between the two MRP runs.

**Exercise 7 rubric.**

A strong answer assigns work order release to level 4 (ERP) and the work order as executed to level 3
(MES), states that the material issue is level 3 but inventory valuation is level 4, and identifies
production quantity as the contested object because both levels record it. A weak answer assigns
everything to "MES" or everything to "ERP" and does not distinguish the planning document from the
execution record.

A strong audit reconstructs the ideal rate and shows whether it is a datasheet, best-ever, or
theoretical value; states whether planned stops include changeover, maintenance, and breaks; and
states whether rework counts as good. It then concludes comparability from those three definitions
plus mix. A weak audit reports the number, compares it with 85%, and stops.

**Exercise 8 model answer.**

Overtime becomes `52 × 52 = €2,704`; the split build becomes `39 × 80 = €3,120`; the expedite remains
€7,743. The ranking changes: overtime and split build are now close, and the break-even chargeback is
where `39 × c = 2,704`, or `c = €69.33` per unit per week. Below that, the split build is cheaper. The
levelled MPS remains cheapest on direct cost but moves 120 units a week late, so it is preferable
only if the customer accepts the slip.

## 14. Interview questions

**1. "Walk me through what MRP actually computes."**

A strong answer gives the four steps in order: gross requirements from parent orders, netting against
on-hand and scheduled receipts, lot sizing and scrap gross-up, backward date offsetting. It states
that capacity is not part of the calculation and lead times are inputs. A weak answer says "it works
out what materials you need" and cannot describe offsetting.

**2. "Why does MRP produce dates the plant cannot meet?"**

A strong answer gives at least four mechanisms from section 5.1 — infinite capacity, fixed lead
times, backward scheduling from unachievable dates, lot sizing and rounding, constant scrap, no
supplier constraint — and notes that MRP is correct given its assumptions. A weak answer says "bad
data" and stops.

**3. "What is the difference between RCCP and CRP?"**

A strong answer says RCCP checks the MPS at critical work centres using a bill of resources over a
medium horizon, while CRP checks MRP planned orders at all work centres using routings and offsets
over a short horizon, and that CRP is more accurate because it includes lot sizes and timing. A weak
answer says one is "rough" and one is "detailed" without naming the data each uses.

**4. "A plant reports 92% OEE. Is that good?"**

A strong answer refuses the question as posed: it depends on the asset, the mix, the definitions of
planned time, ideal rate, and quality, and on whether 92% is on the constraint. It notes that
performance above 100% or OEE above 90% is a red flag. A weak answer says "yes, above the 85%
benchmark."

**5. "How would you decide whether to buy an APS system?"**

A strong answer asks whether the constraint is real, stable, and identifiable; whether routing, setup,
and BOM data are accurate enough to model it; and whether the problem is sequencing or master data.
It notes that APS complements ERP, and that the academic evidence on benefits is thinner than the
vendor evidence. A weak answer says APS is better because it is finite-capacity.

**6. "What is a phantom assembly and why does it matter?"**

A strong answer defines it as a logical grouping exploded through but never stocked, notes that it
should carry no lead time and no inventory, and describes the failure when it is treated as stocked.
A weak answer confuses it with a substitute.

**7. "Why is 95% BOM accuracy not good enough?"**

A strong answer does the arithmetic: a 5% error on a line with quantity per above one, combined with
lot rounding, can stop a double-digit share of an order, and errors compound through levels. It cites
the 98% convention and notes it is a consulting benchmark. A weak answer says "95% sounds fine."

**8. "What is schedule nervousness and what do you do about it?"**

A strong answer defines it as instability in the timing of planned orders under rolling re-planning,
names the remedies (freezing inside a fence at least as long as the cumulative lead time, change-cost
penalties, stable forecasts), and cites the finding that safety stock and lot-for-lot are not
effective remedies. A weak answer says "the schedule changes a lot" with no mechanism.

**9. "How would you measure whether a scheduler is working?"**

A strong answer separates schedule adherence, stability, service, WIP, and overtime, and notes that a
scheduler can improve cost while destroying stability. It proposes measuring sequence change rate and
expedites. A weak answer measures only on-time delivery.

**10. "What data would you need to build a capacity model, and what would you check first?"**

A strong answer lists routing, work centre, setup and run times, calendar, staffing, demonstrated
capacity, and order book, and says it would first check that routing revisions match BOM revisions
and that standard times are current. A weak answer lists machines and orders.

**11. "Where does OEE sit in ISA-95, and why does that matter?"**

A strong answer places it at level 3, notes that it is computed from MES confirmations and ISO 22400
time elements, and explains that the level 4 view is aggregated and later, so an OEE computed from
ERP data is a different metric. A weak answer places it in "the dashboard."

## 15. Key terms

- **BOM** — the structured component list, with quantities, scrap, and effectivity.
- **Low-level code** — the deepest level at which an item appears, driving explosion order.
- **Phantom** — an assembly exploded through but never stocked or counted.
- **Scrap factor** — expected loss as a share of input, applied as `net / (1 − scrap)`.
- **Routing** — ordered operations with work centre, setup, and run time.
- **Work centre** — the capacity-providing resource load is measured against.
- **Setup time** — fixed preparation per lot; **run time** — time per unit.
- **Queue time** — waiting before processing, often 80–90% of lead time.
- **Planned lead time** — the policy value used to offset dates.
- **Cumulative lead time** — the longest BOM-plus-routing path to the end item.
- **MPS** — the dated plan of end items by week.
- **MRP** — the explosion of the MPS into net, dated component orders.
- **RCCP** — capacity check of the MPS at critical work centres.
- **CRP** — capacity check of MRP orders at all work centres.
- **Finite scheduling** — sequencing orders so no resource is over-allocated.
- **APS** — constraint-based planning and scheduling, usually above ERP.
- **Load** — capacity demanded, usually in standard hours per period.
- **Rated capacity** — theoretical capacity at nominal conditions.
- **Demonstrated capacity** — capacity actually achieved, from history.
- **Bottleneck** — the resource that limits throughput.
- **TOC / DBR** — improving through the constraint; drum-buffer-rope scheduling.
- **OEE** — availability × performance × quality.
- **Lot-for-lot / FOQ / EOQ / POQ / Wagner-Whitin** — lot-sizing rules from exact to optimal.
- **Changeover** — conversion work between products, often sequence-dependent.
- **Backflush** — automatic component consumption at confirmation, from the BOM.
- **ATP** — uncommitted supply available to promise.
- **CODP** — the point where a customer order becomes the production driver.
- **ISA-95 / ISA-88** — enterprise-control integration levels; batch control and recipes.
- **MES** — the level-3 system that records and manages execution.

## 16. Sources

**Peer-reviewed and academic**

- Blackburn, J. D., Kropp, D. H., and Millen, R. A. (1986). "A Comparison of Strategies to Dampen
  Nervousness in MRP Systems." *Management Science*, 32(4), 413–429. DOI: 10.1287/mnsc.32.4.413.
- Kadipasaoglu, S. N., and Sridharan, V. (1995). "Alternative approaches for reducing schedule
  instability in multistage manufacturing under demand uncertainty." *Journal of Operations
  Management*, 13(3), 193–211.
- Li, Q., and Disney, S. M. (2017). "Revisiting rescheduling: MRP nervousness and the bullwhip
  effect." *International Journal of Production Research*, 55(7), 1992–2012.
- "An analysis of material requirements planning (MRP) benefits using alternating conditional
  expectation (ACE)." (1995). *Journal of Operations Management*. DOI: 10.1016/0272-6963(95)00005-D.
- Vieira, A. et al. (2021). "Advanced Planning and Scheduling (APS) Systems: A Systematic Literature
  Review." *Advances in Transdisciplinary Engineering*. DOI: 10.3233/atde210118.
- Hvolby, H.-H., and Steger-Jensen, K. (2010). "Technical and industrial issues of Advanced Planning
  and Scheduling (APS) systems." *Computers in Industry*. DOI: 10.1016/j.compind.2010.07.009.
- Thürer, M., Stevenson, M., Silva, C., and Qu, T. (2017). "Drum-buffer-rope and workload control in
  high-variety flow and job shops with bottlenecks." *International Journal of Production
  Economics*. DOI: 10.1016/j.ijpe.2017.03.025.
- Watson, K. J., and Patti, A. (2008). "A comparison of JIT and TOC buffering philosophies on system
  performance with unplanned machine downtime." *International Journal of Production Research*,
  46(7). DOI: 10.1080/00207540600972943.
- Umble, M., Umble, E., and Murakami, S. (2006). "Implementing theory of constraints in a traditional
  Japanese manufacturing environment: the case of Hitachi Tool Engineering." *International Journal
  of Production Research*. DOI: 10.1080/00207540500381393.
- Bengtsson, M., Andersson, J., and Ekström, P. (2020). "Misconceptions Within the Use of Overall
  Equipment Effectiveness." *Advances in Transdisciplinary Engineering*. DOI: 10.3233/atde200141.
- Stip, J., and van Houtum, G.-J. (2019). "On a method to improve your service BOMs within spare
  parts management." *International Journal of Production Economics*. DOI: 10.1016/j.ijpe.2019.08.001.
- Atan, Z. et al. (2017). "Assemble-to-order systems: A review." *European Journal of Operational
  Research*. DOI: 10.1016/j.ejor.2017.02.029.
- Peeters, K., and van Ooijen, H. (2020). "Hybrid make-to-stock and make-to-order systems: a
  taxonomic review." *International Journal of Production Research*.
  DOI: 10.1080/00207543.2020.1778204.
- Zoller, K., and Robrade, A. (1988). "Dynamic lot sizing techniques: Survey and comparison."
  *Journal of Operations Management*, 7(3–4), 125–148.
- Jans, R., and Degraeve, Z. (2008). "Modeling industrial lot sizing problems: a review."
  *International Journal of Production Research*, 46(6), 1619–1643.
- Andriolo, A., Battini, D., Grubbström, R. W., Persona, A., and Sgarbossa, F. (2014). "A century of
  evolution from Harris's basic lot size model." *International Journal of Production Economics*,
  155, 16–38.
- Ying, K.-C., Pourhejazy, P., and Lin, B.-H. (2025). "Scheduling with sequence-dependent setup times
  in short-term production planning: A main path analysis-based review." *Operations Research
  Perspectives*. DOI: 10.1016/j.orp.2025.100340.
- Duplaga, E. A., Hahn, C. K., and Watts, C. A. (1996). "Evaluating capacity change and setup time
  reduction in a capacity-constrained, joint lot sizing situation." *International Journal of
  Production Research*. DOI: 10.1080/00207549608905001.
- Shabur, M. A. (2026). "Autonomous AI Agents in Industrial Operations: A Systematic Review of
  Production Scheduling, Predictive Maintenance and Supply Chain Optimisation." *IET Collaborative
  Intelligent Manufacturing*. DOI: 10.1049/cim2.70083.
- Zech, P., Clark, A., and Breu, R. (2025). "An Empirical Analysis of Digital Twin Adoption."
  *Proceedings of the 58th Hawaii International Conference on System Sciences*.
  DOI: 10.24251/hicss.2025.748.

**Standards and official**

- ANSI/ISA-95.00.01-2025, *Enterprise-Control System Integration — Part 1: Models and Terminology*
  (IEC 62264). ISA.
- ISA-95 Parts 2–8, including ISA 95.00.08 *Information Exchange Profiles*. ISA.
- ISA-88 / IEC 61512, *Batch Control*, Parts 1–4 and technical reports. ISA.
- ISO 22400-1:2014 and ISO 22400-2, *Key performance indicators for manufacturing operations
  management*. ISO.
- OPC 10030, *OPC Unified Architecture — Common Object Model: ISA-95*. OPC Foundation.
- B2MML, Business To Manufacturing Markup Language. MESA International.
- ISO 9001:2015; IATF 16949; FDA 21 CFR Part 11; EU GMP Annex 11.

**Industry, analyst, and vendor**

- Gartner, "Agentic AI Use-Case Assessment for Manufacturing Enterprises," 17 September 2026.
- Gartner, "Scale Physical AI From Pilot to Production by Architecting Integration Across IT/OT
  Systems," 6 August 2026.
- Gartner, "Manufacturing Predicts 2026: Digital Twins, AI Agents, and the Race to Autonomous
  Operations," webinar from 29 January 2026.
- IoT Analytics, *MES Market Report 2025–2031*, December 2025.
- Rockwell Automation, *State of Smart Manufacturing Report*, 2025 (1,560 respondents, 17 countries).
- Bitkom Research, survey of 552 manufacturing companies on digital twins, April 2025.
- Capterra, MRP software research, published 20 April 2026.
- Qlector, survey of large Slovenian manufacturers, 2025.
- HSO, *Technology Maturity in Manufacturing 2026*, reported August 2026.
- Rootstock, *2026 State of Manufacturing Technology Survey*, January 2026, n = 520.
- FIR e.V. (RWTH Aachen), *Future of Production Planning* study, survey February–June 2025.
- Evocon, "Decoding World-Class OEE: A Global Benchmark Analysis from 50+ Countries," 2025, and
  "3 Ways OEE Gets Manipulated," May 2025.
- LNS Research, "Overall Equipment Effectiveness (OEE) as a Metric — Don't Misuse It."
- Oliver Wight, Class A checklist performance measures, as reproduced in Correll, J., *Integrated
  Planning and Control — The Lost Art of Managing the Supply Chain*.
- Baudin, M. (2018). "The BOM Rap." Practitioner essay on BOM accuracy practice.
- Cremers, R. (2016). *Data quality improvement in a production environment*. MSc thesis, Eindhoven
  University of Technology (VDL Nedcar case).

**Note on citations.** Survey and vendor figures are reported as the sources state them and are not
independently verified. Where a figure is single-company, vendor, or self-reported, the text says so
and grades it. No statistic here is presented as an industry average unless a source claims one.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`06_cost-and-working-capital.md`](06_cost-and-working-capital.md)
- Next topic: [`08_transport-and-incoterms.md`](08_transport-and-incoterms.md)
- Product structure and BOM explosion: [`03_network-entities.md`](03_network-entities.md)
- Planning cadence and horizons: [`04_planning-processes.md`](04_planning-processes.md)
- Survey file: [`../../../course/01_supply_chain_foundations.md`](../../../course/01_supply_chain_foundations.md)
- Deeper treatment of production planning: [`../../../course/15_manufacturing_production_planning.md`](../../../course/15_manufacturing_production_planning.md)
- Return to the part map: [`../_index.md`](../_index.md)
