# Policy Families — (s,Q), (R,S), min-max, and periodic review in operational terms

> Module 10 — Inventory Optimization and Multi-Echelon Planning
> Topic 06 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 10 topics 02, 04, and 05
> Version 2026.5

---

## 1. Why this topic exists

Every stocked item in every company is governed by one of four rules. The rule decides when
somebody looks at the item and how much they order when they look. Almost nobody can name the
rule their own system runs.

This is not a failure of intelligence. It is a failure of labelling. A retail replenishment
system calls its policy "min-max" and stores a minimum and a maximum. An enterprise resource
planning (ERP) suite calls the same thing a "reorder point" and a "maximum inventory". A
distribution planner calls it "two weeks of cover". A consultant's slide calls it "(R,S)". Four
vocabularies, one object, and no field on any of the screens that says which of the four
mathematical families is actually executing.

The consequence is measurable. Consider three ways the same item goes wrong, all of which look
like a forecasting problem from the outside.

1. **A periodic policy is buffered as if it were continuous.** The buffer is computed over the
   lead time. The system only gets a chance to order once a week. The exposure is the lead time
   plus the review period, and the item runs two to five points of service below target,
   permanently, with no incident large enough to investigate.
2. **A reorder point is used as an order-up-to level.** Somebody sets a target of 600 units and
   a separate quantity of 200. The system reads the 600 as a threshold, orders 200 when the
   position falls to it, and the position now sits 400 units above where the buffer calculation
   assumed it would. Inventory is 40 per cent high and nobody can find the 40 per cent, because
   every individual number looks defensible.
3. **A minimum contains a buffer and a maximum contains another one.** The minimum was computed
   as lead-time demand plus safety stock. The maximum was set by rule of thumb at twice the
   minimum. A separate safety stock field exists because the item is critical. Three buffers,
   one physical item, and a working-capital line that finance queries every quarter.

None of these is a modelling error. Each is a specification error, and each is fixed by writing
down what the policy is rather than by fitting anything.

There is a second reason the topic matters, specific to analytics work. The policy family
determines which quantities are parameters and which are outputs. In an (s,Q) system the order
quantity is a parameter and the order frequency is an outcome. In an (R,S) system the order
frequency is a parameter and the order quantity is an outcome. An analyst who does not know
which way round the system runs will build a model that optimises a number the system cannot
accept, and the recommendation will be silently rounded, capped, or dropped by the planning
engine. This happens constantly and is almost never diagnosed, because the engine does not
report that it ignored you.

Finally, the policy family is where the cost-service frontier of topic 01 becomes concrete. The
frontier is a curve; a policy is a point on it. Choosing a family is choosing the shape of the
curve you are allowed to move along. Topic 04 derived the buffers and topic 05 defined the
service measures those buffers are set against. This file does not re-derive either. It takes
the buffer as given and asks what rule the buffer sits inside, what else that rule forces, and
how to state the whole thing so that the next person can audit it.

> **Note:** This file uses "policy" in the inventory-control sense — a rule mapping the observed
> inventory position to a replenishment decision. It is not a company policy document and it is
> not a planning process. The process that runs the rule is treated in Module 10 topic 08 and in
> the survey file.

## 2. Learning outcomes

By the end of this topic you can:

1. **Place** any named replenishment rule in a two-dimensional space of review timing (continuous
   or periodic) against order sizing (fixed quantity or variable to a target), and **name** the
   family it belongs to.
2. **Distinguish** a reorder point from an order-up-to level functionally — one is a trigger, the
   other a target — and **explain** the two opposite errors that follow from confusing them.
3. **Reparameterise** between min-max and (s,S), **state** the exact conditions under which
   (s,Q) and (s,S) coincide, and **identify** where the families genuinely diverge.
4. **Compute** the protection period implied by a policy configuration, **show** where it is
   hidden inside a stored parameter, and **quantify** the buffer difference between a continuous
   and a periodic policy at the same service target.
5. **Reconstruct** the policy a system is actually running from its order history and field
   layout, rather than from the policy label, and **justify** the reconstruction with two
   independent tests.
6. **Explain** why transport calendars, minimum order quantities (MOQs), order multiples and
   supplier cut-offs force periodic review in practice, and **price** the consequence of
   accepting the forced review period.
7. **Diagnose** the case in which continuous review cannot be implemented because the inventory
   position is not observable at the required frequency, and **state** the effective review
   period the organisation is really running.
8. **Write** a policy specification in the seven-part standard (family, review interval, trigger,
   target, multiple, protection period, service measure) and **list** what must be recomputed
   when any single parameter changes.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Every replenishment rule answers two questions and only two: **when do I look**, and **how much
do I take**. The answers are independent, and crossing them gives four usable combinations. Hold
that as a two-by-two grid and most of the confusion in this topic disappears.

```text
                          HOW MUCH TO ORDER
                    fixed quantity          variable, to a target
                 +----------------------+--------------------------+
                 |        (s, Q)        |         (s, S)           |
   CONTINUOUS    |  trigger: position   |  trigger: position <= s  |
   (event-       |  <= s                |  target:  S              |
   driven)       |  order:  Q           |  order:   S - position   |
                 |  protection: L       |  protection: L           |
                 +----------------------+--------------------------+
   WHEN TO       |        (R, Q)        |         (R, S)           |
   REVIEW        |  fixed lot on a      |  target: S, every R      |
                 |  calendar; uncommon  |  protection: R + L       |
   PERIODIC      |  but real in         |                          |
   (calendar-    |  make-to-stock       |  add a trigger s and you |
   driven)       |  environments        |  have (R, s, S)          |
                 +----------------------+--------------------------+
```

Three readings of the grid are worth carrying.

**Read the columns as "who chooses the quantity".** In the left column the buyer chooses a lot
size and the system chooses when. In the right column the system chooses the lot size — it is
whatever closes the gap to the target — and the buyer chooses the target. This is why an
optimisation that recommends an order quantity is meaningful in the left column and meaningless
in the right one. In an order-up-to system the quantity is a consequence, not a decision.

**Read the rows as "how long am I blind".** Continuous review means the position is examined at
every transaction, so the exposure between a decision and its correction is the lead time.
Periodic review means the position is examined every `R` periods, so the exposure is `R + L`.
That single extra term is the entire cost difference between the rows, and it appears in the
buffer, not in the formula that computes the buffer.

**Read min-max as a renaming, not a family.** Min-max is the top-right cell with the parameters
called "min" and "max" instead of `s` and `S`. It is exactly (s,S). Any claim that min-max is a
separate, simpler, or more robust method is a claim about the user interface, not about the
mathematics. The distinction matters because the (s,S) literature supplies optimality conditions
that the min-max vocabulary hides.

### 3.1 The timeline, and where the review period hides

The protection period is the span a buffer must cover. It is the only quantity in this topic
that changes physical meaning when you move between rows of the grid, and it is almost always
stored inside a parameter rather than shown as a field.

```text
  (s, Q) — continuous review: exposure is L

   position
      ^
   s+Q|--+                                +--
      |  | \                             / |
      |  |  \                           /  |
    s |--+---+----------+--------------+---+--- trigger fires at s
      |  |    \         |             /
    0 |--+-----+--------+------------+--------> time (weeks)
         0     2        6           10
         ^     ^        ^            ^
      order  arrives  order       arrives
         |<-L->|
         |< exposure = L >|


  (R, S) — periodic review: exposure is R + L

   position
      ^
    S |--+              +--              +--
      |   \            /   \            /
      |    \          /     \          /
    0 |-----+--------+-------+---------+-----> time (weeks)
          0 R      R+L      2R      2R+L
          ^  ^       ^       ^         ^
      review |    arrival  review   arrival
             |<-L->|
      |<----- exposure = R + L ------>|
```

In the lower diagram, an order placed at week 0 arrives at week 2. The next order is not placed
at week 2; it is placed at week `R`, which in the drawing is week 4. So if demand in weeks 1 to
6 is high, nothing can be done about it until week 6. The buffer has to cover not two weeks but
six, and the extra four weeks are a calendar artifact, not a supply-chain fact.

Two practical consequences follow and they are the ones that get missed.

**The protection period is inside the parameter.** In `(s, Q)` the number `s` equals mean demand
times `L` plus a buffer over `L`. In `(R, S)` the number `S` equals mean demand times `R + L`
plus a buffer over `R + L`. A planner reading `S = 629 kits` sees a target. They do not see a
six-week horizon, a 95 per cent service target and a 22-unit weekly standard deviation. The
parameter is an opaque encoding of four decisions.

**The trigger and the target are different kinds of object.** The trigger is a threshold on the
inventory position; the target is a level the position is raised to. A threshold is compared
against; a level is subtracted from. Swapping them does not produce a slightly wrong policy. It
produces a policy that orders at the wrong time, in the wrong amount, or both, and the resulting
inventory can be either systematically high or systematically empty.

### 3.2 Why the families diverge, stated in one line each

| Difference | Mechanism | Where you see it |
|---|---|---|
| Higher buffer under periodic review | Protection period `R + L` instead of `L` | Safety stock rises with `sqrt((R+L)/L)` |
| Higher cycle stock at low review frequency | One order per `R` periods means `d * R / 2` average cycle stock | Average inventory rises linearly in `R` |
| Deterministic ordering cost under `(R,S)` | Exactly `52 / R` orders per year | Budget predictability; contract volumes |
| Ordering cost is stochastic under `(s,Q)` | Order count depends on demand realisation | Freight spend varies with demand |
| Response latency under `(R,S)` | Nothing can be ordered between reviews | Stock-outs cluster at the end of the cycle |
| Undershoot under `(s,Q)` | A lumpy demand can jump the position past `s` | Order quantity is `Q`, so the gap is never closed |
| Self-correction under `(s,S)` and `(R,S)` | Ordering to a level closes any gap | Undershoot is absorbed automatically |

The last row is the one practitioners rarely state and it explains a great deal. A fixed-quantity
policy cannot repair an undershoot, because the quantity is fixed. An order-up-to policy repairs
it by construction, because the quantity is the gap. This is why (s,S) dominates (s,Q) once
demand arrives in lots rather than units, and it is the practical reason min-max survives in
retail even though it carries no theoretical advantage over the fixed-quantity rule it replaced.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Replenishment policy | A rule mapping inventory position to a replenishment decision | Used to mean the planning process that runs the rule | Family name plus parameters |
| Review | The act of examining the inventory position and deciding whether to order | Confused with the forecast refresh | Event |
| Review period (R) | Interval between successive replenishment decisions | Assumed to equal the system's batch job frequency | Days or weeks |
| Continuous review | Position examined at every transaction, so a decision can be taken at any moment | Assumed because the system runs nightly | Enumeration |
| Periodic review | Position examined only at fixed intervals | Assumed because the planner meets weekly | Enumeration |
| Reorder point (ROP, s) | Inventory position at which a replenishment is triggered | Set equal to safety stock | Units |
| Order-up-to level (S) | Target inventory position immediately after ordering | Reported as the safety stock | Units |
| Fixed order quantity (Q) | A constant lot size ordered whenever the trigger fires | Confused with the economic order quantity (EOQ) | Units |
| (s,Q) policy | Continuous review, order `Q` when position reaches `s` | Called "two-bin" without checking the bin sizes | `s` and `Q` in units |
| (s,S) policy | Continuous review, order up to `S` when position reaches `s` | Called "min-max" and treated as a different method | `s` and `S` in units |
| (R,S) policy | Periodic review, order up to `S` every `R` periods | Modelled with protection period `L` | `R` in periods, `S` in units |
| (R,s,S) policy | Periodic review, order to `S` only if position is at or below `s` | Modelled as `(R,S)` because the trigger "rarely binds" | `R`, `s`, `S` |
| Min-max | The practitioner name for `(s,S)`, with min = `s` and max = `S` | Treated as a distinct policy family | Two levels in units |
| Base-stock policy | Order up to `S` every period; the `K = 0` limit of `(s,S)` | Used as a synonym for `(R,S)` at any `R` | `S` in units |
| Protection period | The span a buffer must cover: `L` under continuous review, `R + L` under periodic | Assumed equal to the lead time | Days or weeks |
| Inventory position | On-hand plus on-order minus backorders | Computed on on-hand alone | Units |
| On-hand | Physical stock available at the location, as recorded | Assumed equal to physical reality | Units |
| On-order | Quantity ordered and not yet received | Omitted when the position is computed from the warehouse system | Units |
| Scheduled receipts | Firm inbound supply with a date, including transfers and production | Treated as on-order without dates | Units with dates |
| Backorder | Demand accepted but not yet served | Treated as equivalent to a lost sale | Units |
| Undershoot | The amount by which the position falls below `s` before the trigger is noticed | Ignored in fixed-quantity policies | Units |
| Cycle stock | Stock that exists because orders are placed in lots | Counted as safety stock | Units |
| Safety stock | Buffer above expected demand over the protection period | Set as the reorder point | Units |
| Order multiple | The lot increment in which orders must be placed | Applied after the buffer calculation instead of inside it | Units per increment |
| Minimum order quantity (MOQ) | The smallest quantity a supplier will accept | Treated as a rounding detail | Units |
| Lot size key | The ERP parameter that selects a lot-sizing rule | Read without checking which rule it selects | Code (system-specific) |
| Planning time fence | A horizon inside which the planning engine will not automatically change orders | Confused with the review period | Days |
| Safety lead time | Planned time added so material arrives before it is needed | Used as a substitute for safety stock without repricing | Days |
| Material requirements planning (MRP) | A time-phased planning method driven by bills of material and forecasts | Used as a synonym for any replenishment logic | Method |
| MMP | The materials management and planning module of an ERP suite; in some suites the reorder-point fields live here | Assumed to mean a specific policy | Module name (confirm locally) |
| Demand driven material requirements planning (DDMRP) | A buffer-zone method that places decoupling points and sizes buffers from average daily usage and decoupled lead time | Described as min-max with three levels | Method |
| Decoupled lead time | The lead time a DDMRP buffer is sized to protect | Confused with total lead time | Days |
| Service measure | The quantity the policy is set to achieve: cycle service level, fill rate, or a time-based measure | Left unstated | Enumeration plus target |
| Cycle service level (CSL) | Probability that a replenishment cycle ends without a stock-out | Reported as fill rate | Proportion |
| Fill rate | Expected fraction of demand units served from stock | Reported as CSL | Proportion |
| Days of cover | Coverage expressed in time rather than units | Converted to units without stating the demand rate | Days |
| Time bucket | The period the planning engine aggregates into | Changed without recomputing the buffer | Day, week, month |
| Dampener period | A window in which the engine suppresses small suggested quantity changes | Confused with a review period | Days |
| Double buffering | Two or more buffer layers accumulating on the same physical item | Invisible in any single field | Units |
| Policy string | A one-line specification of family, review interval, trigger, target, multiple, protection period and service measure | Replaced by a screenshot | Text |

Four of these rows carry most of the operational risk, so they are expanded here.

**Reorder point versus order-up-to level.** A reorder point is a threshold that the position must
fall to before anything happens. An order-up-to level is a target that the position is raised to
once something happens. They have the same unit, the same order of magnitude and completely
different jobs. The test that separates them in a live system is the order quantity: if the
quantity is the same every time, the stored number above it is a target and the stored number
below it is a trigger. If the quantity varies and lands the position on a constant level, that
level is the target and the trigger is the point at which the varying quantity started.

**Min-max versus (s,S).** These are the same rule. The reason to say so explicitly is that
min-max invites two errors that the algebraic notation discourages. The first is setting min
equal to the safety stock, because "min" sounds like "the least we should hold" rather than "the
level at which we act". The second is deriving max as a multiple of min, which conflates a
trigger level with a quantity and produces a band width — and therefore an order quantity — that
no cost calculation ever produced.

**Protection period versus lead time.** These are equal only in the top row of the grid. Under
periodic review the protection period exceeds the lead time by exactly one review interval, and
the buffer must cover the difference. Topic 04 quantified that difference; here the point is
that the protection period is not a field in any system. It is an inference from the family and
two other fields, and it is the first thing to reconstruct when auditing a policy.

**Undershoot.** In a fixed-quantity policy the order is placed at whatever position the trigger
was noticed at, which may be well below `s` if demand arrives in lots. The order of `Q` is then
measured from that lower point, so the cycle ends lower than intended. Order-up-to policies are
immune because the quantity is defined as the gap to `S`. The practical consequence is that an
(s,Q) policy carrying a buffer sized exactly for lead-time variability will stock out
periodically, and the fix is either an explicit undershoot allowance added to `s` or a move to
(s,S).

## 5. The mechanics, step by step

The procedure is written as an audit you perform on a live configuration, because that is the
task that pays. It has four phases: establish what the system actually runs, parameterise the
family you have chosen, reconcile the parameters to the physical constraints that will override
them, and put the result under change control. Each step states its input, its output, and the
branch where analysts diverge.

### Phase A — Establish what the system actually runs

**Step 1. Read the item master, not the policy label.**

- Input: the item record at the stocking location, including every field whose name suggests
  ordering: reorder point, minimum, maximum, safety stock, safety time, lot size, order multiple,
  minimum order quantity, planning time fence, time bucket.
- Output: a list of populated fields and their values.
- Branch: **which fields are populated?** A record with a reorder point and a maximum is running
  an order-up-to rule. A record with a reorder point and a fixed quantity is running (s,Q). A
  record with all three is running something nobody has documented, and step 5 decides which
  field wins.

**Step 2. Determine whether the engine can order on a non-calendar date.**

- Input: a month of proposed order dates from the planning run, or, if the engine is not
  available, the order release dates in the purchase order history.
- Output: the distribution of days between consecutive orders.
- Branch: **is the interval constant?** A constant interval means the engine or the process is
  periodic, whatever the policy field says. A variable interval means the engine is reacting to
  position. A bimodal distribution — many intervals at exactly 7 days, some at 14 — means a
  weekly cut-off is overriding a continuous rule.

**Step 3. Reconstruct the decision rule from order history.**

This is the decisive test and it needs only two columns: the inventory position at the moment of
each order, and the quantity ordered.

| Pattern in the scatter of quantity against position at order | Family | What the slope means |
|---|---|---|
| Quantity constant, position at order varies | `(s,Q)` or `(R,Q)` | Slope zero; the trigger is the maximum observed position at which an order occurred |
| Quantity falls as position rises, landing on a horizontal line | `(s,S)` or `(R,S)` | Slope −1; the intercept is `S` |
| Quantity falls with slope −1, but some reviews produce no order | `(R,s,S)` or `(s,S)` with a high trigger | Slope −1; the trigger is the highest position at which no order was placed |
| Quantity constant at a level and no orders between dates | `(R,Q)` | Slope zero, with a calendar rule |

- Input: twelve months of order lines with receipt dates and inventory snapshots.
- Output: a scatter, a fitted slope, and an intercept, with the number of observations stated.
- Branch: **is the reconstructed intercept stable over the year?** A drifting intercept means the
  target is being recomputed from a moving forecast, which is a different object from a stored
  constant and must be reported as such.

**Step 4. Measure the realised review interval.**

- Input: order release timestamps, not proposed dates.
- Output: the median interval between releases, in days, and the share of intervals within one
  day of that median.
- Branch: **does the realised interval match the configured one?** A system configured for daily
  review but constrained by a Tuesday supplier cut-off has a realised review period of seven
  days. The protection period follows behaviour, not configuration, and this measurement is the
  only way to know which.

**Step 5. Decide which field wins when the configuration is contradictory.**

- Input: the field list from step 1 and the reconstructed rule from step 3.
- Output: a statement of the effective rule, written as a policy string (step 19).
- Branch: **is there a field that is populated but never read?** Legacy fields are common: a
  maximum left over from a previous system, a safety stock that the lot-size logic ignores, a
  reorder point that only applies to a planning type nobody uses. The reconstruction in step 3
  tells you which fields are live. Fields that are populated but not live are the single largest
  source of phantom policy documentation.

> **Note:** Steps 1 to 5 can be done in an afternoon for one item and they routinely overturn
> the answer the organisation gives verbally. Do them before accepting any statement of the form
> "we run a reorder point system".

### Phase B — Parameterise the family you have chosen

**Step 6. Fix the protection period explicitly.**

- Input: the review period `R` from step 4, the lead time `L` from the supplier record or the
  realised receipt history, both in the same unit.
- Output: `P = L` for continuous review, `P = R + L` for periodic review, in days or weeks.
- Branch: **is the lead time the administrative lead time or the realised one?** The realised
  lead time includes order transmission, supplier processing, transit, customs and goods receipt
  posting. The last of these is frequently one to three days and is the difference between a
  buffer that works and one that is short by a day of demand.

**Step 7. Compute the buffer over the protection period.**

- Input: `P`, the demand mean and standard deviation per period, the lead-time standard deviation
  if the lead time is random, and the service factor implied by the named service measure.
- Output: a buffer in units.
- Branch: **has the service measure been named?** If the target is a fill rate, the multiplier
  depends on the order quantity and therefore on the policy family. Topic 04 derives the
  conversion; the point here is that you cannot compute step 7 before you have chosen the family,
  because the family determines `Q`.

**Step 8. Compute the trigger and the target for the chosen family.**

```text
   (s, Q):   s = d * L                    + buffer over L
             Q = chosen lot size          (EOQ, pallet multiple, contract volume)

   (s, S):   s = d * L                    + buffer over L
             S = s + Q_target             (the EOQ shortcut; not optimal, widely used)

   (R, S):   S = d * (R + L)              + buffer over (R + L)
             order at each review to S

   (R,s,S):  s = d * L                    + buffer over L
             S = d * (R + L)              + buffer over (R + L)
             order to S at review only if position <= s
```

- Input: the buffer from step 7, the demand rate, `P`.
- Output: two numbers with units, and the family they belong to.
- Branch: **does the family compute `s` over `L` or over `R + L`?** In `(R,s,S)` the trigger `s`
  is often computed over `L` on the argument that the review will catch it, while the target `S`
  is computed over `R + L`. That combination is defensible only if the trigger is checked against
  the position *at the review*, in which case the trigger should also be over `R + L`. Using `L`
  for the trigger and `R + L` for the target creates a band whose lower edge is one review period
  too low, and the item will stock out at the end of long cycles.

**Step 9. Apply order modifiers inside the parameterisation, not after it.**

- Input: the order multiple, the MOQ, the pallet and layer quantities, the supplier's price
  breaks.
- Output: an effective `Q` or an effective `S` that is realisable.
- Branch: **does the multiple exceed the model's order quantity?** If the EOQ is 90 units and the
  MOQ is 500, the policy's order quantity is 500 and the cycle length is 500 divided by the demand
  rate. That cycle length is now a review period in all but name, and the protection period must
  be recomputed with it. Applying the MOQ as a rounding step at the end leaves the buffer sized
  for a cycle that does not exist.

**Step 10. Check the trigger against the modifier ceiling.**

- Input: `s` from step 8, the effective order quantity from step 9, and the storage capacity at
  the location.
- Output: a statement of whether the policy can physically execute.
- Branch: **does `s + Q` exceed the location's capacity?** If it does, the effective policy is a
  capacity-constrained order-up-to rule, the target is the capacity, and the quantity is
  whatever fits. The buffer has been silently converted into a fill-the-shelf rule, and the
  service calculation no longer applies.

**Step 11. Name the service measure and record the target with it.**

- Input: the service level agreement, the internal service definition, or the category policy.
- Output: one of cycle service level, fill rate, or a time-based measure, with a number and a
  measurement window.
- Branch: **is the measure the one the customer contract names?** A CSL buffer against a fill-rate
  contract over-buffers; the reverse under-buffers. Topic 05 owns the definitions and this step
  only records which one applies.

**Step 12. Check that the parameters are internally consistent across the item's locations.**

- Input: the same item's records at every stocking location.
- Output: a table of family, `R`, `s`, `S`, and service target by location.
- Branch: **is a different family running at each location for the same item?** This is common in
  distribution: a central warehouse on periodic review because it receives consolidated trucks,
  stores on continuous review because they are scanned at the till. That is often correct, and it
  must be documented, because it means the item has two protection periods and two buffers, and
  a transfer between locations changes both.

### Phase C — Reconcile to physical constraints

**Step 13. Establish the inbound transport calendar.**

- Input: delivery windows, consolidation schedules, milk-run routes, container departure days.
- Output: the set of days on which a receipt is physically possible.
- Branch: **is the calendar finer or coarser than the policy's review period?** If the only
  inbound movement is a Tuesday truck, the review period is seven days regardless of the policy
  field. Accepting this is not a compromise; it is the correct parameterisation.

**Step 14. Establish the supplier's order acceptance pattern.**

- Input: order cut-off times, minimum order value, order-day restrictions, production campaign
  cycles.
- Output: the effective review period imposed by the supplier.
- Branch: **does the supplier run campaigns?** A supplier that produces the item every six weeks
  imposes a six-week effective review period on the buyer, and no amount of daily position
  checking changes it. The buffer must cover the campaign cycle.

**Step 15. Establish the lot-size ladder.**

- Input: case pack, layer, pallet, container, and any contractual full-truckload quantity.
- Output: the set of admissible order quantities.
- Branch: **is the ladder coarser than the cycle?** When one pallet is four weeks of demand, the
  order quantity is four weeks of demand and the review period is at least four weeks. The
  policy becomes periodic by arithmetic, not by choice.

**Step 16. Check whether the position is observable at the assumed frequency.**

Continuous review assumes that on-hand, on-order and backorders are known at the moment of every
transaction. Test each of the three:

- **On-hand.** Is every receipt posted at the moment of physical arrival, or at the end of the
  shift, or the next morning? Is every issue transacted at the point of pick, or backflushed at
  period end? Is the count accurate, or is the record a book value that drifts between cycle
  counts?
- **On-order.** Are purchase orders and production orders visible in the same system, or does
  the buyer hold a spreadsheet? Are transfer orders between locations in scope?
- **Backorders.** Is unmet demand recorded as a backorder, or does it disappear as a lost sale?

- Output: the effective observation lag, in days, for each of the three components.
- Branch: **is the lag greater than zero?** Then the system is running periodic review with a
  review period equal to the lag. An organisation that posts receipts overnight is running a
  daily-review policy and should set `P = 1 day + L`, not `L`. This is the most common hidden
  review period in practice and it is invisible in every configuration screen.

**Step 17. Check that lead times are non-crossing.**

- Input: receipt dates against order dates for the last twelve months, per supplier and lane.
- Output: the share of order pairs whose arrival order does not match their placement order.
- Branch: **do orders cross?** If two orders are outstanding and the later one can arrive first,
  the position is still computable but the demand accumulated over the effective lead time is
  not, and the classical fixed-quantity analysis — which assumes a single outstanding order —
  does not apply. Crossing is common with ocean freight and uncommon with a single domestic
  supplier. Where it occurs, the honest response is to treat the system as periodic with a review
  period equal to the receipt consolidation interval, or to model the lead-time distribution
  explicitly (topic 07).

### Phase D — Change control and reporting

**Step 18. Write the recomputation chain for every parameter.**

- Input: the parameters from Phase B and the constraints from Phase C.
- Output: a dependency list, in the form "if this changes, these must be recomputed".
- Reasoning: the parameters are not independent. A change to the review period changes the
  protection period, which changes the buffer, which changes the target, which changes the
  average inventory, which changes the storage requirement. Changing one field without the rest
  is how a policy drifts out of specification while every individual value looks plausible.

| Parameter changed | Must be recomputed | Must also be checked |
|---|---|---|
| Review period `R` | Protection period, buffer, target `S`, cycle stock | Transport plan, storage capacity, order frequency cost |
| Lead time `L` or its variance | Protection period, trigger `s`, target `S` | Safety lead time field, MOQ against the new cycle, contract dates |
| Service target or measure | Service factor or loss-function multiplier, buffer, `s` and `S` | Whether the measure matches the contract |
| Order quantity `Q` | Cycle stock, order frequency, undershoot allowance in `s` | MOQ, multiples, pallet ladder, transport mode |
| Order multiple or MOQ | Effective `Q`, cycle length, protection period if the cycle changed | Storage capacity, price breaks |
| Demand rate | `d * P` term in `s` and `S`, buffer if the standard deviation scales with the mean | Forecast horizon, seasonality |
| Demand standard deviation | Buffer, `s` and `S` | The estimation window and exclusions |
| Unit of measure | Everything | Case pack conversion, price per unit |
| Family | Every parameter, and the reporting string | Planning engine capability, buyer behaviour |

**Step 19. Write the policy string.**

The reporting standard is a single line with seven parts. Anything less is a screenshot.

```text
   (family, review interval, trigger, target, multiple, protection period, service measure)
```

A completed example:

```text
   (R, s, S) | review: weekly, Tuesday 06:00 position snapshot (R = 7 d)
             | trigger: s = 278 kits at position
             | target:  S = 638 kits
             | multiple: 40 kits (pallet layer); MOQ 120 kits
             | protection: R + L = 7 + 14 = 21 d
             | service: CSL 95% per cycle, measured on kit demand, 52-week window
```

Every element is necessary. Remove the review interval and the protection period is
underdetermined. Remove the multiple and the order quantity cannot be reproduced. Remove the
service measure and the buffer is an unfalsifiable number.

**Step 20. Set the review cadence for the parameters themselves.**

- Input: the volatility of demand, lead time and cost for the item.
- Output: a re-estimation interval, and the trigger events that force an out-of-cycle review.
- Branch: **is the parameter stored as a quantity or derived from a forecast?** A stored quantity
  goes stale silently as the demand rate moves. A derived level changes every time the forecast
  changes, which is correct but produces order volatility that buyers find unsettling. The
  specification must say which one the system does, because the two fail differently.

**Step 21. Test the proposed change against history.**

- Input: twelve months of demand and receipts, and the old and new parameters.
- Output: simulated service, average inventory, and order count under both.
- Method: replay the demand series against both policies, assuming orders are received after the
  realised lead time. Report the three metrics side by side, not a single blended score.
- Branch: **does the new policy improve service without increasing inventory, or the reverse?**
  If it improves both, one of the two simulations is wrong; the trade-off is real and a result
  that escapes it is a bug.

**Step 22. Stage the change and set the checkpoint.**

- Input: the change, the effective date, the owner, and the expected effect from step 21.
- Output: a change record and a review date four to eight weeks later.
- Method: change one parameter set at a time per item family, not the whole catalogue at once.
  Record the before and after values, the recomputation evidence, and the measured service and
  inventory at the checkpoint.
- Branch: **what would make you reverse it?** State the reversal condition before the change.
  A policy change without a reversal condition is a permanent decision made on one week of
  analysis.

### 5.1 The four families, compared on the dimensions that matter

| Dimension | `(s, Q)` | `(s, S)` | `(R, S)` | `(R, s, S)` |
|---|---|---|---|---|
| Trigger | Position `<= s` | Position `<= s` | Every review | Position `<= s` at review |
| Quantity | Fixed `Q` | `S - position` | `S - position` | `S - position`, or zero |
| Protection period | `L` | `L` | `R + L` | `R + L` plus skipped reviews |
| Parameters stored | 2 | 2 | 2 | 3 |
| Order frequency | Outcome | Outcome | Parameter (`1/R`) | Outcome, bounded by `1/R` |
| Order quantity | Parameter | Outcome | Outcome | Outcome |
| Undershoot | Not corrected | Corrected | Corrected | Corrected |
| Response latency | Zero | Zero | Up to `R` | Up to `R` |
| Position observability needed | At every transaction | At every transaction | At each review | At each review |
| Optimality result | EOQ under certainty | Scarf (1960) under fixed cost | Optimal given a review calendar | No classical result; heuristic |
| Typical system home | Stockroom, two-bin | Retail min-max, distribution | Consolidated inbound, MRP | ERP with both fields populated |

The table's last two rows are the ones to argue about in a design review. `(R,S)` is optimal
conditional on the review calendar being fixed, which is a strong condition that the business,
not the modeller, supplies. `(R,s,S)` has no classical optimality result and is nevertheless the
most common configuration in ERP systems, because it is what you get when a reorder-point field
and a maximum field are both populated by different people at different times.

## 6. Worked example

One item, four policies, and the arithmetic that separates them. The item is synthetic and the
numbers are constructed to be internally consistent, but every step is reproducible by hand.

### 6.1 The item card

| Quantity | Value | Unit |
|---|---:|---|
| Item | 7712-A | identifier |
| Description | Actuator service kit, two valve families | — |
| Location | DC North | identifier |
| Unit of measure | kit | eaches |
| Mean demand (`d`) | 90 | kits per week |
| Standard deviation of demand | 22 | kits per week |
| Lead time (`L`) | 2.0 | weeks |
| Service target | 95 per cent cycle service level | proportion |
| Service factor (`z`) | 1.645 | dimensionless |
| Unit cost | EUR 42.00 | currency per kit |
| Holding rate | 24 | per cent per year |
| Holding cost per kit per year | EUR 10.08 | currency per kit per year |
| Ordering cost (`K`) | EUR 140.00 | currency per order |
| Economic order quantity | 360 | kits per order |
| Order multiple | 40 | kits per increment |
| Minimum order quantity | 120 | kits |
| Annual demand | 4,680 | kits per year |

The economic order quantity (EOQ) is `sqrt(2 * 140 * 4,680 / 10.08) = 361`, rounded down to 360
because 360 is nine pallet layers of 40. Topic 02 owns the derivation; here the number is an
input that fixes the order frequency at about thirteen orders a year.

### 6.2 Four parameterisations of the same item

| Family | Parameters | Protection period | Buffer | Trigger or target |
|---|---|---|---:|---|
| `(s, Q)` | `s = 231`, `Q = 360` | `L = 2` weeks | 51 kits | Trigger 231 |
| `(s, S)` | `s = 231`, `S = 591` | `L = 2` weeks | 51 kits | Trigger 231, target 591 |
| `(R, S)`, weekly | `R = 1`, `S = 333` | `R + L = 3` weeks | 63 kits | Target 333 |
| `(R, S)`, four-weekly | `R = 4`, `S = 629` | `R + L = 6` weeks | 89 kits | Target 629 |

The arithmetic, all with `z = 1.645` and `sigma = 22`:

```text
   (s, Q)   s = 90 * 2      + 1.645 * 22 * sqrt(2)   = 180 + 51.2 = 231.2 -> 231
   (s, S)   S = 231 + 360                                            = 591
   (R, S)   S = 90 * 3      + 1.645 * 22 * sqrt(3)   = 270 + 62.7 = 332.7 -> 333
   (R, S)   S = 90 * 6      + 1.645 * 22 * sqrt(6)   = 540 + 88.6 = 628.6 -> 629
```

Two things are worth pausing on. First, the buffer grows with the square root of the protection
period, so moving from continuous to four-weekly review raises the buffer from 51 to 89 kits —
a factor of 1.73 — with no change to demand, lead time or service target. Second, the *target*
grows much more than the buffer, because the target also contains the expected demand over the
longer protection period. The 629 is 540 of cycle demand plus 89 of buffer; only the 89 is a
response to uncertainty.

### 6.3 Average inventory and annual cost at matched order frequency

The comparison that matters holds the order frequency constant, so that the only difference is
the policy family. Matching `(s,Q)` at `Q = 360` to `(R,S)` at `R = 4` weeks gives thirteen orders
a year in both cases.

| Component | `(s, Q)` | `(R, S)`, `R = 4` | Difference |
|---|---:|---:|---:|
| Cycle stock (kits) | 180 | 180 | 0 |
| Safety stock (kits) | 51 | 89 | +38 |
| Average on-hand (kits) | 231 | 269 | +38 |
| Holding cost (EUR/year) | 2,328 | 2,711 | +383 |
| Ordering cost (EUR/year) | 1,820 | 1,820 | 0 |
| Total relevant cost (EUR/year) | 4,148 | 4,531 | +383 |

The 38-kit difference is exactly `z * sigma * (sqrt(R + L) - sqrt(L)) = 36.19 * (2.449 - 1.414) =
37.5`, rounded. It is the price of the review period, isolated from every other effect, and it
is 16.3 per cent of the continuous-review average inventory and 9.2 per cent of total relevant
cost.

> **Note:** The 38 kits are an annual cost of EUR 383 on one item. The number is small, and that
> is the honest finding. Across an 800-item catalogue with a similar structure the same effect
> is around EUR 300,000 a year, which is large enough to matter and small enough that no single
> item will ever be investigated for it.

### 6.4 The same comparison at a weekly review

A weekly review is the most common configuration in practice, and it changes the picture in a
direction most people do not expect.

| Family | Orders per year | Average on-hand (kits) | Holding (EUR/year) | Ordering (EUR/year) | Total (EUR/year) |
|---|---:|---:|---:|---:|---:|
| `(s, Q)`, `Q = 360` | 13 | 231 | 2,328 | 1,820 | 4,148 |
| `(R, S)`, `R = 1` | 52 | 108 | 1,086 | 7,280 | 8,366 |
| `(R, S)`, `R = 4` | 13 | 269 | 2,711 | 1,820 | 4,531 |
| `(R, S)`, `R = 8` | 6.5 | 474 | 4,778 | 910 | 5,688 |

Weekly review has the *lowest* average inventory of the four and the *highest* total cost, by a
wide margin. It buys responsiveness with order frequency and pays for it in ordering cost. This
is why "review more often" is not a service improvement and why the review period must be chosen
against the ordering cost, not against a feeling about freshness.

Eight-weekly review shows the opposite failure: cycle stock rises to 360 kits because each order
covers eight weeks, and average inventory nearly doubles. The order quantity and the review
period are the same decision viewed from two sides, and neither can be set without the other.

### 6.5 The protection period hiding in the parameter

Ask four people what the buffer is on this item and you will get four numbers, all defensible
and none comparable.

| Person reads | Number | What it actually is |
|---|---:|---|
| "The reorder point is 231" | 231 kits | Lead-time demand plus buffer |
| "The safety stock is 51" | 51 kits | Buffer over `L` |
| "The max is 629" | 629 kits | Target including six weeks of demand |
| "We hold six weeks of cover" | 540 kits | Cycle demand only, no buffer at all |

The reason this matters is that all four statements can appear in the same meeting and each
will be defended with the same confidence. Only the second is a buffer. Only the third is a
target. The first is a trigger and the fourth is a coverage convention that contains no
uncertainty at all.

### 6.6 When the review period costs real money

The premium is driven by the ratio of the review period to the lead time, not by either
absolutely. Holding `L = 2` weeks fixed and varying `R`:

| `R` (weeks) | `P = R + L` | Buffer (kits) | Buffer versus continuous | Premium as share of buffer |
|---:|---:|---:|---:|---:|
| 1 | 3 | 63 | +12 | +24 per cent |
| 2 | 4 | 72 | +21 | +41 per cent |
| 4 | 6 | 89 | +38 | +75 per cent |
| 8 | 10 | 114 | +63 | +124 per cent |
| 13 | 15 | 140 | +89 | +175 per cent |

And holding `R = 4` fixed and varying `L`:

| `L` (weeks) | `P = R + L` | Buffer (kits) | Buffer versus continuous | Premium as share of buffer |
|---:|---:|---:|---:|---:|
| 1 | 5 | 81 | +45 | +124 per cent |
| 2 | 6 | 89 | +38 | +75 per cent |
| 4 | 8 | 102 | +30 | +41 per cent |
| 8 | 12 | 125 | +18 | +17 per cent |
| 16 | 20 | 162 | +10 | +7 per cent |

Read the two tables together and the rule is simple. **The review period matters most when it is
long relative to the lead time.** For a domestic item with a two-day lead time and a weekly
review, the policy family is nearly irrelevant to the buffer, because the review period dominates
the exposure either way. For an imported item with a twelve-week lead time and a monthly review,
the family is also nearly irrelevant, because the lead time dominates. The family matters in the
middle: short-to-medium lead times with weekly or fortnightly reviews, which is exactly where
most fast-moving consumer items live.

### 6.7 The min-max double buffer, priced

A configuration that produces systematic overstock, assembled from three individually reasonable
decisions.

| Layer | Field | Value (kits) | What it was meant to be |
|---|---|---:|---|
| 1 | Reorder point (min) | 231 | Correct trigger for continuous review |
| 2 | Maximum inventory | 720 | "Eight weeks of cover", set by a planner |
| 3 | Safety stock | 89 | Buffer over `R + L`, set independently |
| | Effective target | 809 | Max plus separately replenished safety stock |
| | Effective order quantity | 578 | 809 minus 231 |

The effective policy is `(s, S)` with `s = 231` and `S = 809`.

```text
   average on-hand = (S - s) / 2 + (s - d * L)
                   = (809 - 231) / 2 + (231 - 180)
                   = 289 + 51
                   = 340 kits
   correct average = 231 kits
   excess          = 109 kits, or 47 per cent
   annual cost     = 109 * 10.08 = EUR 1,099 per year
```

Three separate errors are stacked. The maximum is a coverage rule with no cost basis. The safety
stock is a second buffer computed over a protection period the maximum already covers. And the
band width of 578 kits is an order quantity that no EOQ calculation produced, which triples the
cycle stock relative to the 360 kit order that was intended. Removing any one layer reduces the
excess; removing all three returns the item to 231 kits.

> **Note:** The opposite error is more common than the double buffer and much worse. It occurs
> when a planner sets min equal to the safety stock, on the reading that "min" means "the least
> we should hold". With `s = 51` and `S = 591`, the item orders when the position reaches 51 kits
> against a two-week lead time of 180 kits, and stocks out in every cycle. The reorder point is
> understated by 78 per cent. Both errors come from the same vocabulary failure and neither is
> visible in a single field.

### 6.8 What the four policies cost when they are run correctly

| Question | `(s, Q)` | `(s, S)` | `(R, S)` `R = 4` | `(R, s, S)` |
|---|---|---|---|---|
| Average on-hand (kits) | 231 | 231 | 269 | 269 |
| Orders per year | 13 | 13.4 | 13 | 12 |
| Protection period (weeks) | 2 | 2 | 6 | 6 to 10 |
| Service at the stated target | 95 per cent CSL | 95 per cent CSL | 95 per cent CSL | 95 per cent CSL if the trigger binds as assumed |

Run correctly, all four hit the service target. The differences are in what they store, how often
they order, and — critically — how quickly they can respond to a surprise. The last column is the
one to watch: the effective protection period of `(R,s,S)` is only `R + L` if the trigger fires at
every review. If it fires in half the reviews, the effective period is about `2R + L`, and the
buffer is sized for a period the item does not have.

## 7. Code, query, or calculation

The simulation below is the whole argument of section 6.3 written as twelve weeks of arithmetic.
Demand is held constant at 90 kits a week so that the only source of difference is the review
discipline. Both policies place exactly the same orders, in the same quantity, at the same
average frequency. Only the timing differs.

```text
wk  demand | (s,Q) on-hand  order | (R,S) on-hand  order
 1      90 |           411      0 |           269    360
 2      90 |           321      0 |           179      0
 3      90 |           231    360 |           449      0
 4      90 |           141      0 |           359      0
 5      90 |           411      0 |           269    360
 6      90 |           321      0 |           179      0
 7      90 |           231    360 |           449      0
 8      90 |           141      0 |           359      0
 9      90 |           411      0 |           269    360
10      90 |           321      0 |           179      0
11      90 |           231    360 |           449      0
12      90 |           141      0 |           359      0
----------------------------------------------------------
mean         |           276      |           314
orders       |             3      |             3
```

Parameters: `s = 231`, `Q = 360`, `S = 629`, `R = 4` weeks, `L = 2` weeks, both starting at their
own steady state. Read the columns as follows.

- The `(s,Q)` system orders in week 3, when the position reaches exactly 231, and again in weeks
  7 and 11. Its on-hand runs from 411 down to 51 and back.
- The `(R,S)` system orders in weeks 1, 5 and 9, on the calendar, regardless of position. Its
  on-hand runs from 449 down to 89 and back.
- Both place 3 orders of 360 kits: 1,080 kits against 1,080 kits of demand. Identical throughput,
  identical order count, 38 kits more stock.

Two things the table deliberately does not show. First, both columns sample on-hand at the start
of the week, which overstates the time average by about half a week of demand (45 kits) for both.
The 38-kit difference between the columns is unaffected by that bias, which is why it is reported
as a difference and not as an absolute. Second, the `(s,Q)` system is allowed to trigger exactly
at 231 because demand is treated as unit-sized and continuous; with lot-sized demand the position
would undershoot and the real `(s,Q)` would need a higher trigger. Section 6.7 shows what happens
when that correction is omitted.

Expected output: 276 kits against 314 kits, a difference of 38, which is 16.3 per cent of the
continuous-review average and matches `z * sigma * (sqrt(6) - sqrt(2))` to the nearest unit.

## 8. Data, standards, and regulatory context

No standard defines a replenishment policy. There is no ISO, ANSI or IEC document that specifies
whether an item should run `(s,Q)` or `(R,S)`, what the review period should be, or how a min-max
pair should be derived. What the standards constrain is the **input data** and the
**auditability** of the parameters, and those are the parts that are usually missing.

### 8.1 What the standards constrain

| Standard or instrument | What it governs | What it constrains here | Grade |
|---|---|---|---|
| GS1 GTIN | Product identity | The item dimension of the policy; a pack-size change creates a new identifier and a new parameter set | B |
| GS1 GLN | Location identity | The location dimension; a re-coded warehouse silently splits one parameter set into two | B |
| EPCIS 2.0 (GS1) | Event data: what, where, when, why | Receipt and issue timestamps, which determine the realised lead time and the observation lag in step 16 | B |
| ISO 8000 series | Data quality | Completeness and consistency of the item, location and period keys that the parameters are attached to | B |
| ISO 22400-2 | Manufacturing operations key performance indicators | How a service or inventory indicator is defined and reported; does not choose the policy family | B |
| Service level agreement | Contract | Names the service measure and target the policy must satisfy, and therefore the multiplier | B |
| IAS 2 / ASC 330 | Inventory measurement | Inventories at the lower of cost and net realisable value; a buffer is a policy quantity, not an accounting category | B |
| Internal control over financial reporting | Listed-company governance | Change control over parameters that materially affect reported inventory | B |

### 8.2 What is convention rather than requirement

- "Min equals two weeks of cover" and "max equals twice min" are habits, not rules. Neither is
  derived from a cost calculation in any source consulted here.
- A weekly review is an artefact of a paper-based planning era and of the weekly planning meeting.
  It has no regulatory standing and it is frequently longer than the lead time.
- The choice between cycle service level and fill rate is a commercial decision recorded in a
  contract, not a technical one. Two companies with identical inventory and identical targets can
  report service levels four to nine points apart because they chose different measures.
- "Continuous review" in a system description usually means "the batch job runs nightly", which
  is periodic review with `R = 1` day. This is the single most widespread mislabelling in the
  field and it is not a standardisation problem, it is a documentation problem.

### 8.3 The audit trail

The parameter set is a controlled record in any listed company, because it determines reported
inventory value. Two practices follow and neither is usually implemented.

**A parameter change log.** Who changed which field, from what to what, on what date, with what
recomputation, and what was the measured effect at the checkpoint. Without it, a service failure
eight months later cannot be attributed, and the investigation restarts from zero.

**A reconstruction test at audit.** Take ten items, reconstruct the policy from order history as
in step 3, and compare with the documented policy. In most organisations the two disagree on at
least one item in ten, usually because a field was changed by a buyer and never documented. The
test takes a day and is the only reliable evidence that the documentation describes the system.

## 9. Industry practice and case evidence

### 9.1 What the research literature establishes

| Study or source | Design | Finding | Grade |
|---|---|---|---|
| Scarf (1960), *Mathematical Methods in the Social Sciences* | Analytical, dynamic programme | With a fixed cost per order and convex holding and shortage costs, the optimal policy is `(s,S)` | A |
| Singha, Buddhakulsomsiri and Parthanadee (2017), *Math. Problems in Engineering* | Analytical and numerical, Thai healthcare distribution | Once undershoot is modelled explicitly, continuous and periodic review perform almost identically in cost; a practical rule is to solve the continuous problem and add the expected undershoot to the reorder point | A |
| Gürler, Alp and Büyükkaramikli (2014), *Transportation Research Part E* | Analytical, one warehouse and many retailers with a third-party trucking contract | Periodic review is competitive with continuous review for large chains, large trucks and low demand rates; the periodic premium ranges from a few per cent to roughly 20 per cent depending on the configuration | A |
| Cachon (2001), *Operations Research* | Exact evaluation, two-echelon batch ordering | Policies optimised under continuous-review assumptions perform poorly when implemented under periodic review | A |
| Prak, Teunter and Riezebos (2015), *EJOR* | Analytical, hybrid review | Counting stock periodically while allowing orders continuously yields large savings against pure periodic ordering when the fixed ordering cost is zero | A |
| Hollier, Mak and Yiu (2005), *IJPR* | Analytical and heuristic, lumpy demand | `(s,S)` with a maximum issue quantity restriction outperforms naive `(s,Q)` on lumpy demand items | A |
| Ward (1978), *Management Science* | Analytical, compound demand | Reorder points computed for unit-sized demand are systematically too low when demand arrives in lots | A |

The Singha result is the most useful and the least known. It says that the apparent cost advantage
of continuous review is largely an artefact of comparing a correct continuous policy against a
periodic one, or of ignoring the undershoot that a lumpy fixed-quantity policy suffers. Correct
both, and the two rows of the grid converge on cost. That does not make the rows equivalent — the
response latency and the observability requirement remain different — but it removes the cost
argument that is usually deployed first.

The Gürler result supplies the counterweight. When inbound transport is contracted in trucks, the
truck capacity and the demand rate interact with the review period, and periodic review becomes
the better choice for large systems with large vehicles. The mechanism is not subtle: a truck
that must be filled imposes a lot size, and a lot size imposes a cycle, and a cycle is a review
period.

### 9.2 Where (s,S) comes from, and why the result matters

The dynamic inventory problem with a fixed ordering cost was formulated in the early 1950s, and
`(s,S)` optimality was proved first under restrictive conditions on the demand distribution.
Scarf's 1960 contribution was to prove it generally, using a property he called K-convexity: a
function is K-convex if adding a fixed amount `K` and shifting the argument never improves on the
linear interpolation between two points. The value function of the inventory problem is K-convex
at every stage, and a K-convex value function has a single interval of positions at which ordering
is optimal. That is what rules out complicated policies with several order and no-order regions.

The practical value of the theorem is not the proof. It is that **`(s,S)` is the right shape
whenever there is a fixed cost per order and the cost of being wrong is convex**. Both conditions
hold for most stocked items. The theorem also tells you what breaks it: if the holding and
shortage cost is not convex — for example, if there is a large fixed penalty for any stock-out,
or a quantity discount that makes the cost function non-convex — the single-interval structure can
fail, and the optimal policy may have several trigger levels. Very few operational systems can
represent such a policy, which is a reason to check the cost structure before assuming the
two-parameter rule is adequate.

> **Note:** The antecedents and later generalisations — the 1951 formulation, the 1952 and 1953
> optimality results under restricted demand, Veinott's weakening of the hypotheses, and the
> generalisation of K-convexity to several dimensions — were identified from secondary summaries
> in the sources consulted and are not cited individually here. Scarf (1960) is the reference to
> quote.

### 9.3 How the policy appears in real systems

The same four families appear under different names in every major suite. The table maps the
fields, not the marketing.

| System | Where the policy lives | Fields that matter | Grade |
|---|---|---|---|
| SAP ERP | MRP type on the material master | Reorder point `MINBE`, safety stock `EISBE`, lot size key `DISLS`, planned delivery time `PLIFZ`, planning time fence `FXHOR` | D |
| Oracle Fusion Cloud | Item planning attributes | Inventory planning method (reorder point, min-max, not planned), min-max quantities, safety stock planning method, order modifiers | D |
| Microsoft Dynamics 365 Business Central | Reordering policy on the item or stock-keeping unit card | Reordering policy, reorder point, maximum inventory, safety stock quantity, safety lead time, time bucket, order multiple, minimum and maximum order quantity | D |
| Infor SyteLine | Reorder point field on purchased items | Reorder point, order multiple, lead time; generates a fixed-quantity order when on-hand plus on-order is at or below the reorder point | D |
| Rootstock | MRP policies tab on the item master | Planning policy, policy quantity, minimum and maximum quantity, safety stock quantity, supply multiplier | D |
| QAD | Item-site planning record | Time fence, safety time, reorder point, order policy, minimum and maximum order | D |
| Retail replenishment suites | Store ordering parameters | Order point, order-up-to level, presentation minimum, case pack | D |

Three observations from the table are worth more than the table.

**The field names encode a theory, and the theory is often (s,S).** Business Central's "Maximum
Qty." reordering policy is a reorder point plus a maximum, which is `(s,S)`. Oracle's "min-max"
planning method is `(s,S)`. SyteLine's reorder point with an order multiple is `(s,Q)` rounded.
The label on the screen is not the label in the literature, and the mapping has to be done field
by field.

**Safety stock is a separate field in every one of them, and it interacts with the level.** In
Business Central, a safety stock breach triggers a backward-scheduled refill order while a
reorder point breach triggers a forward-scheduled exception order — two different mechanisms
operating on the same item. That is a defensible design and it is also exactly how double
buffering arises when both fields are populated by different people.

**Reorder-point logic and time fences do not always combine.** A SAP Community question reports
that a planning time fence cannot be set on a reorder-point MRP type, returning message 61839,
with safety stock suggested as the workaround. That is a community report about one release and
should be verified against the current documentation before it is relied on, but the structural
point stands: reorder-point planning and time-phased planning are different engines, and a
configuration that assumes both are active may be running only one.

### 9.4 Why min-max survives in retail

Min-max is theoretically a renaming of `(s,S)` and therefore neither better nor worse than the
optimal policy shape. Its persistence in retail is explained by four conditions that hold
together in that setting and rarely hold together elsewhere.

1. **Ordering cost per line is small relative to holding cost.** With electronic ordering and
   consolidated deliveries, the fixed cost `K` is low, so the optimal band width is narrow and
   the difference between a cost-optimal band and a plausible one is small.
2. **Lead times are short.** With a two-day lead time the undershoot and the review period both
   matter less, so a rule that ignores them is not badly wrong.
3. **Shelf capacity is a real constraint.** The maximum is not only an inventory target; it is
   the number of units the fixture holds. Making that ceiling an explicit field is operationally
   useful in a way that an abstract order-up-to level is not.
4. **The rule is computable by the person who runs it.** A store manager can set a minimum and a
   maximum. They cannot run an EOQ, and they should not have to.

The failure is not min-max. It is min-max used without the two disciplines the algebra implies:
min is a trigger that contains lead-time demand, not a buffer; max is a target that already
contains the buffer, so a separate safety stock field on top of it is a second buffer. Get those
two right and min-max is a perfectly good `(s,S)`.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Periodic policy buffered over the lead time only | Nobody can state the review period the buffer assumes | Protection period set to `L` | 12 to 55 per cent short buffer depending on `R/L` | Publish `R`, `L`, `P` on the item card |
| Reorder point used as an order-up-to level | Order quantity equals the position at order time, every cycle | Target read as a threshold | Position sits one order quantity too high | Reconstruct the rule from order history |
| Min set equal to safety stock | Stock-outs in every cycle on items with a long lead time | Trigger level contains no lead-time demand | Reorder point understated by 60 to 80 per cent | Test: is min greater than `d * L` |
| Max derived as a multiple of min | Max divided by min is exactly 2.0 across the catalogue | Band width set by rule of thumb, not by cost | Order quantity and cycle stock both arbitrary | Recompute the band from EOQ or from the service target |
| Safety stock added on top of a target that already contains it | Effective ceiling exceeds the configured maximum | Two buffer layers, two owners | 30 to 60 per cent excess inventory | One buffer per item; delete the redundant field |
| Review period longer than anyone believes | Order release dates cluster on one weekday | The calendar overrides the policy | Effective protection period longer than documented | Measure realised intervals from release timestamps |
| Continuous review claimed with an overnight posting lag | Receipts appear in the system the morning after arrival | Position is stale by up to a day | Hidden review period, buffer short by a day of demand | Set `R` to the observation lag and recompute |
| MOQ larger than the modelled order quantity | Order quantity is constant at the MOQ | Cycle length set by the supplier | Protection period longer than the buffer assumes | Recompute `P` with the realisable cycle |
| `(R,s,S)` trigger that rarely fires | Many reviews produce zero orders | Effective protection period is `2R + L` or more | Buffer sized for a period the item does not have | Compute the trigger-firing rate and the implied period |
| Order quantity capped by storage | The same item is expedited every quarter | Capacity binds before the target is reached | Service failures that look like demand spikes | Report the capacity constraint; treat as fill-the-shelf |
| Parameters changed without recomputation | Reorder point updated, target unchanged | Fields are edited independently | Band width, and therefore quantity, drifts | Change log with the dependency chain in step 18 |
| Policy family differs between locations and is undocumented | A transfer between two DCs changes the item's behaviour | Two protection periods, two buffers | Service discontinuity at the transfer | Document the family per location in the policy string |

The three failures at the top of the table account for most of the damage, and they share a
signature: every individual number is defensible and the combination is not. That is why the
audit in section 5 starts from order history rather than from the configuration. The
configuration records intent; the order history records behaviour; only the second can be
audited.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The review discipline is the one the policy assumes | Determines whether the protection period is `L` or `R + L` | A periodic policy buffered as continuous is short by 12 to 55 per cent | Measure the realised interval between order releases |
| The inventory position is observable at the assumed frequency | Continuous review has no meaning without it | Hidden review period equal to the posting lag | Compare receipt timestamps with posting timestamps |
| Demand arrives in units, not lots | Makes `(s,Q)` equivalent to `(s,S)` with `Q = S - s` | Undershoot; the fixed quantity never closes the gap | Distribution of order-to-order demand increments |
| At most one order is outstanding, or orders do not cross | The classical fixed-quantity analysis assumes it | Position is computable but lead-time demand is not | Match arrival order against placement order |
| The lead time is known and stationary | The trigger or target is computed from it | Buffer sized for a lead time that no longer exists | Control chart of realised lead times |
| The service measure is the one the contract names | The multiplier depends on it | Systematic over- or under-buffer at the same nominal target | Read the contract; compute both multipliers |
| The order size is a free choice | Both columns of the grid assume it | Cycle length, and therefore protection period, is externally imposed | Check MOQ, multiples, pallet and truck quantities |
| Cost per order is fixed and known | Justifies the band width in `(s,S)` and the lot size in `(s,Q)` | The band is arbitrary; the order quantity is arbitrary | Activity-based cost of a purchase order line |
| Holding cost does not depend on the quantity held | Makes the cost function linear and convex | Scarf's single-interval structure can fail; multi-trigger policies become optimal | Check for storage step costs and quantity discounts |
| The item is buffered individually | Justifies a per-item parameter set | Substitution, pooling or component commonality change the effective variance | Check for substitutes and shared components |

Two limits deserve statement. **The family is a shape, not a solution.** Knowing that an item
runs `(s,S)` tells you which two numbers to compute and which two to leave alone. It tells you
nothing about whether the numbers are right, and a correctly shaped policy with a wrong standard
deviation is worse than a wrongly shaped one with a good one, because it looks more rigorous.
**The taxonomy stops at the boundary of the planning engine.** A policy that the engine cannot
represent — a multi-trigger rule, a state-dependent target, a rule that depends on the age of the
stock — is not a policy the organisation has, whatever the design document says.

## 12. Exercises

### Exercise 1 — Classify four configurations (recall and application, 30 minutes)

Four item records: (A) reorder point 400, fixed order quantity 250; (B) minimum 180, maximum 720,
safety stock 0; (C) order point 260, order-up-to 640, review day Tuesday; (D) maximum inventory
700, safety stock 90, no reorder point.

**Deliverable:** a family for each, the protection period implied, and the parameter that would
have to change for the other family to be the correct description.
**Acceptance criteria:** each answer names the two dimensions of the grid; the implied protection
period is stated in days with its inputs; configuration D is identified as ambiguous and the
ambiguity explained.
**Time estimate:** 30 minutes.

### Exercise 2 — Reconstruct the rule from history (application, 45 minutes)

Using any order history with at least 20 lines, plot order quantity against inventory position at
the time of order.

**Deliverable:** the scatter, the fitted slope, the implied target, and the implied trigger.
**Acceptance criteria:** the inventory position is computed as on-hand plus on-order minus
backorders; the number of observations and the period are stated; the family is named from the
scatter rather than from the policy label.
**Time estimate:** 45 minutes.

### Exercise 3 — Four parameterisations (calculation, 40 minutes)

For an item with demand 60 units per week, standard deviation 15, lead time 3 weeks, order
quantity 300, target service 95 per cent CSL: compute `(s,Q)`, `(s,S)`, `(R,S)` at `R = 2` weeks,
and `(R,s,S)` at `R = 2` weeks.

**Deliverable:** a four-row table with protection period, buffer, trigger, target, average
on-hand and orders per year.
**Acceptance criteria:** units at every step; the average on-hand formula stated for each family;
one sentence on which family stores least and why.
**Time estimate:** 40 minutes.

### Exercise 4 — The double buffer audit (critique, 40 minutes)

An item runs with reorder point 300, maximum inventory 900 and safety stock 120. Demand is 75
units per week and the lead time is 2 weeks.

**Deliverable:** the effective target, the effective band width, the average on-hand, and the
excess against a correctly configured `(s,S)` at the same order frequency.
**Acceptance criteria:** each buffer layer identified with the field that created it; the
non-additivity of the layers acknowledged; a recommendation that names the field to change.
**Time estimate:** 40 minutes.

### Exercise 5 — Review period against the transport calendar (design, 40 minutes)

A supplier ships on a dedicated truck every Thursday. The item has a 10-day lead time, weekly
demand of 200 units and a standard deviation of 40.

**Deliverable:** the effective review period, the protection period, the target under `(R,S)`,
and the cost of the constraint against a hypothetical daily-review continuous policy.
**Acceptance criteria:** the realised interval is derived from the truck schedule, not from the
policy field; the cost difference is in currency per year with the holding rate stated; one
condition under which the constraint should be challenged commercially.
**Time estimate:** 40 minutes.

### Exercise 6 — When continuous review cannot be implemented (diagnosis, 35 minutes)

A distributor claims continuous review on imported items with a 9-week ocean lead time, receipts
posted at month end, two or three orders outstanding at any time, and no visibility of the
forwarder's consolidation schedule.

**Deliverable:** a diagnosis naming each failed observability condition, the effective review
period, and the policy family the organisation is actually running.
**Acceptance criteria:** all three position components addressed; the crossing-order problem
distinguished from the posting-lag problem; a recommendation that states what would have to
change for continuous review to become implementable.
**Time estimate:** 35 minutes.

### Exercise 7 — Write the policy string (design, 35 minutes)

Take an item you can inspect and write its policy in the seven-part standard.

**Deliverable:** the one-line policy string plus a note on which element was hardest to establish.
**Acceptance criteria:** all seven elements present; every number carries a unit; the protection
period is shown as arithmetic from `R` and `L`; the service measure names its window.
**Time estimate:** 35 minutes.

### Exercise 8 — The change protocol (application, 30 minutes)

The lead time on an item falls from 6 weeks to 3 weeks after a supplier moves to a regional
warehouse.

**Deliverable:** the recomputation chain, in order, with the effect on each dependent parameter.
**Acceptance criteria:** at least six parameters traced; the effect on the MOQ and on the storage
requirement included; one secondary effect that is easy to miss identified.
**Time estimate:** 30 minutes.

### Exercise 9 — Migration plan (design, 50 minutes)

A retailer runs min-max in stores and a weekly order-up-to at the distribution centre. Service is
acceptable and inventory is 30 per cent above the category benchmark.

**Deliverable:** a two-page plan identifying which of the four families each location should run,
what would be gained, and what would be lost.
**Acceptance criteria:** the recommendation distinguishes the store and the DC; the protection
period at each location is stated; at least one element of current practice is defended rather
than replaced; the plan names the measurement that would confirm the gain.
**Time estimate:** 50 minutes.

## 13. Self-check answers

**Exercise 1.** A is `(s,Q)`: trigger 400, fixed 250, protection `L`. B is `(s,S)` / min-max:
trigger 180, target 720, protection `L`. C is `(R,s,S)`: weekly review, trigger 260, target 640,
protection `R + L` when the trigger fires. D is ambiguous: a maximum with a safety stock and no
reorder point is either `(R,S)` with the safety stock embedded in the maximum, or `(s,S)` with the
trigger left at zero. The ambiguity is resolved by order history, not by the record. For A to
become `(s,S)`, the fixed quantity field must be replaced by a target above the trigger.

**Exercise 2.** A correct scatter has slope near zero and a vertical cluster of positions for
`(s,Q)`, or slope near minus one and a horizontal cluster of landed positions for `(s,S)`. The
target is the mean of the landed positions; the trigger is the highest position at which no order
was placed. A submission that reads the policy label and stops has not done the exercise. A
submission that reports a slope of minus one but a drifting intercept has found a target being
recomputed from a forecast, which must be reported as a different object.

**Exercise 3.** With `z = 1.645` and `z * sigma = 24.68`:

```text
(s,Q)      P = 3   buffer 42.7   s = 180 + 43 = 223   Q = 300   avg = 150 + 43 = 193   17.3 orders/yr
(s,S)      P = 3   buffer 42.7   s = 223   S = 523             avg = 150 + 43 = 193   17.3 orders/yr
(R,S) R=2  P = 5   buffer 55.2   S = 300 + 55 = 355           avg = 60 + 55 = 115    26 orders/yr
(R,s,S)    P = 5   buffer 55.2   s = 223   S = 355             avg = 115 if s binds   24 orders/yr
```

`(R,S)` at two weeks stores least, because the order frequency is doubled and the cycle stock
halves. It pays for that in ordering cost, exactly as in section 6.4.

**Exercise 4.** The effective target is 900 plus 120, or 1,020 kits, on the assumption that the
safety stock replenishes separately. The band width is 720 kits against a demand of 75 a week,
so the cycle is 9.6 weeks. Average on-hand is `360 + (300 - 150) = 510` kits. A correctly
configured `(s,S)` at the same order frequency would have a band of 720 and a trigger of
`150 + 1.645 * sigma * sqrt(2)`, so the excess is the 120-kit safety layer plus the difference
between the specified trigger and the correct one. The layers are not strictly additive: the
safety-stock layer lifts the ceiling, and the trigger layer shifts the whole sawtooth, so their
combined effect on the average is less than the sum. The field to change is the safety stock,
because the maximum already contains a buffer.

**Exercise 5.** The truck forces `R = 7` days. With a 10-day lead time, `P = 17` days or 2.43
weeks, against 1.43 weeks under continuous review. At `z * sigma = 65.8` per week, the buffers are
`65.8 * sqrt(2.43) = 102.6` and `65.8 * sqrt(1.43) = 78.7` kits, a difference of 24 kits, or about
EUR 240 a year at a holding cost of EUR 10 per kit per year. The constraint is worth challenging
commercially only if the item's holding cost is high enough that a second weekly delivery is
cheaper than 24 kits of buffer, which for most items it is not.

**Exercise 6.** Three conditions fail. Receipts posted at month end give an observation lag of up
to 31 days, so the position is not observable continuously and the effective review period is one
month regardless of the policy field. Two or three outstanding orders with no guaranteed arrival
sequence means orders cross, so lead-time demand over the effective lead time is undefined. The
forwarder's consolidation schedule is invisible, so the lead time is not merely random but
unobserved until it happens. The organisation is running periodic review with `R` between one
month and the consolidation interval, and its buffer should be sized over `R + L` with the realised
lead-time distribution from topic 07. Continuous review becomes implementable only with
transaction-level receipt posting, visible in-transit milestones, and a booking rule that prevents
crossing.

**Exercise 7.** A strong string reads, for example: `(s,S) | continuous against a daily position
snapshot (effective R = 1 d) | trigger 223 kits | target 523 kits | multiple 25 kits, MOQ 100 |
protection L = 21 d | fill rate 97 per cent per cycle, 26-week window`. A weak string gives two
numbers and a family name. The hardest element is usually the review interval, because the honest
answer is the observation lag rather than the configured frequency.

**Exercise 8.** In order: protection period falls from 6 to 3 weeks; buffer falls by
`sqrt(3/6) = 0.707`; trigger and target recomputed; cycle stock unchanged if the order quantity is
unchanged, but the order frequency rises because the same quantity now covers less time; MOQ must
be rechecked against the shorter cycle, since an MOQ that was 40 per cent of the old cycle is now
80 per cent of the new one; storage requirement falls with the target; the transport plan may need
more frequent, smaller shipments. The easy-to-miss secondary effect is the supplier's minimum
order value, which may now bind on every order rather than occasionally.

**Exercise 9.** A strong plan defends the store min-max, because the ordering cost is low, the
lead time is short and the maximum encodes shelf capacity — all four conditions in section 9.4.
It changes the DC to a documented `(R,S)` at the consolidation frequency with a stated protection
period, and it deletes any safety stock field that duplicates the maximum. The gain is measured
by average on-hand at constant service, not by service alone. A weak plan moves everything to one
family and calls the resulting service change an improvement.

## 14. Interview questions

**1. "What is the difference between a reorder point and an order-up-to level?"**

A strong answer says one is a trigger compared against the position and the other is a target the
position is raised to, and gives the order-history test that separates them. A weak answer says
they are both inventory levels, which is true and useless.

**2. "Is min-max the same as (s,S)?"**

A strong answer says yes, exactly, with min = `s` and max = `S`, and explains that the persistence
of the name is a user-interface fact rather than a theoretical one. A weak answer treats them as
different methods.

**3. "When would you choose periodic over continuous review?"**

A strong answer names the constraints that make the choice for you — transport calendar, supplier
campaigns, order cut-offs, consolidated inbound, the cost of monitoring — and adds that the
periodic premium is small when `R` is short relative to `L`. A weak answer says periodic is
simpler.

**4. "What is the protection period and where does it live?"**

A strong answer gives `L` or `R + L` and says it lives inside `s` or `S`, not in a field. A weak
answer gives the formula but cannot say where in the system it is stored.

**5. "Why does a fixed-quantity policy suffer undershoot and an order-up-to policy not?"**

A strong answer explains that the quantity is a constant so the gap created by a lumpy demand is
never closed, while the order-up-to quantity is defined as the gap. A weak answer has not heard
of undershoot.

**6. "How would you find out which policy a system is really running?"**

A strong answer describes reconstructing the rule from order quantity against inventory position,
plus measuring the realised review interval from release timestamps. A weak answer opens the
configuration screen and reads the label.

**7. "What happens if min is set equal to the safety stock?"**

A strong answer says the item orders when the buffer is already exhausted, so it stocks out every
cycle, and gives the arithmetic: min must exceed lead-time demand. A weak answer says it would
order too late, without a magnitude.

**8. "How does a transport constraint change the policy?"**

A strong answer says the delivery frequency becomes the review period, which extends the
protection period and therefore the buffer, and that the correct response is to accept the
constraint and reparameterise rather than to model a review that cannot happen. A weak answer
treats it as a rounding issue.

**9. "When can continuous review not be implemented?"**

A strong answer names the three observability conditions — on-hand, on-order, backorders — plus
crossing orders, and says the effective review period is the observation lag. A weak answer says
continuous review needs good data, without saying which data or what happens without it.

**10. "What must be recomputed when the review period changes?"**

A strong answer gives the chain: protection period, buffer, target, average inventory, storage,
order frequency and cost, and the MOQ check. A weak answer says the buffer, and stops.

## 15. Key terms

Twenty-six terms carry the whole topic. If you can define these without the file in front of you,
you can read any replenishment configuration you are handed and say what it will do.

| Term | One-line definition |
|---|---|
| Continuous review | Inventory position is observed at every transaction, so the trigger can fire at any moment |
| Periodic review | Inventory position is observed only at scheduled review instants, so the trigger is evaluated on a calendar |
| Review period (`R`) | The interval between consecutive review instants, in days or weeks |
| Review interval in practice | Whatever the batch job interval is, regardless of what the configuration screen calls the policy |
| Reorder point (`s`) | A threshold on the inventory position; an order is triggered when the position is at or below it |
| Order-up-to level (`S`) | A target the inventory position is raised to at the moment of ordering |
| Min | The name a min-max system gives to its trigger; operationally a reorder point |
| Max | The name a min-max system gives to its target; operationally an order-up-to level |
| `(s,Q)` | Continuous review, fixed order quantity; the trigger fires, a constant `Q` is ordered |
| `(s,S)` | Continuous review, variable quantity; the trigger fires, the position is raised to `S` |
| `(R,S)` | Periodic review, variable quantity; at each review the position is raised to `S` |
| `(R,s,S)` | Periodic review with an in-period trigger as well as a review-instant target |
| Inventory position | On-hand plus on-order minus backorders; the quantity the trigger is compared against |
| On-hand | Physical stock in the location, excluding goods in transit and goods committed elsewhere |
| On-order | Supply that has been released but not yet received; the field most often missing from a manual calculation |
| Backorder | Demand accepted but not served, held as a negative in the position |
| Undershoot | The amount by which the position at the trigger sits below `s`, caused by demand arriving in lots larger than one unit |
| Protection period | The span the buffer must cover: lead time, plus the review period where one exists |
| Review-period increment | The extra buffer required solely because the trigger can only fire on a calendar |
| Safety stock field | A separately configured quantity in most systems, distinct from the level or the trigger |
| Planning time fence | A horizon inside which the system will not reschedule supply; constrains what a policy can achieve |
| Time bucket | The granularity at which reviews and receipts are netted, usually a day or a week |
| Lot size key | The configuration that decides whether the quantity is fixed, lot-for-lot, or a multiple |
| Order multiple | A quantity the order must be a whole multiple of, usually a pallet or a case pack |
| Minimum order quantity (MOQ) | A floor on the order quantity imposed by the supplier rather than by the policy |
| Double buffer | Two overlapping protections set by different people, typically a min-max band and a safety stock field |
| Reconstruction test | Rebuilding the policy from order history and comparing it with the documented parameter set |

Two of these terms are doing more work than the rest. **Inventory position** is the quantity the
policy actually operates on, and almost every manual calculation that goes wrong has used on-hand
instead. **Undershoot** is the correction that makes fixed-quantity continuous review behave the
way the formulas claim, and it is absent from most textbook treatments of `(s,Q)`.

## 16. Sources

Sources are grouped by what they can support. Grades follow Module 28: A peer-reviewed or audited,
B regulatory or official, C company case or investor claim, D vendor marketing or product
documentation, E press summary or anecdote.

### 16.1 Analytical and empirical literature — grade A

| Source | What it supports |
|---|---|
| Scarf, H. (1960), *Mathematical Methods in the Social Sciences* | The optimality of `(s,S)` under a fixed ordering cost and convex holding and shortage costs |
| Singha, K., Buddhakulsomsiri, J. and Parthanadee, P. (2017), *Mathematical Problems in Engineering* | Continuous and periodic review converging in cost once undershoot is modelled |
| Gürler, Ü., Alp, O. and Büyükkaramikli, N. (2014), *Transportation Research Part E* | The periodic-review premium under contracted trucking, from a few per cent to roughly 20 per cent |
| Cachon, G. (2001), *Operations Research* | The poor performance of continuous-review-optimised policies implemented under periodic review |
| Prak, D., Teunter, R. and Riezebos, J. (2015), *European Journal of Operational Research* | Hybrid review: periodic counting with continuous ordering |
| Hollier, R., Mak, K. and Yiu, K. (2005), *International Journal of Production Research* | `(s,S)` against `(s,Q)` under lumpy demand with a maximum issue quantity |
| Ward, J. (1978), *Management Science* | Reorder points computed for unit-sized demand being too low when demand arrives in lots |

> **Note:** Journal, volume and page details are reported as they appeared in the sources consulted.
> Where a search returned a title and year but not a full citation, the entry is given without
> volume and page numbers rather than reconstructed. Nothing in this section is a claim about a
> source that was not read at least in abstract.

### 16.2 Standards and official instruments — grade B

| Instrument | What it supports |
|---|---|
| GS1 GTIN | Product identity as the item dimension of a policy |
| GS1 GLN | Location identity as the location dimension |
| GS1 EPCIS 2.0 | Event data supplying receipt and issue timestamps |
| ISO 8000 series | Data quality of the keys the parameters attach to |
| ISO 22400-2 | Definitions of manufacturing and inventory KPIs; not a policy prescription |
| IAS 2 and ASC 330 | Inventory measurement at the lower of cost and net realisable value |
| Service level agreements | The contractual service measure and target the policy is set against |

The standards group supports the data and auditability claims in section 8 and nothing about which
policy family to choose. No standard consulted prescribes a family, a review period, or a min-max
derivation, and none should be cited as if it did.

### 16.3 Product documentation — grade D

| Product | Used only for |
|---|---|
| SAP ERP material master | The names and roles of the reorder-point and lot-size fields |
| Oracle Fusion Cloud item planning attributes | The names of the planning-method and min-max fields |
| Microsoft Dynamics 365 Business Central | The reordering-policy options and the separate safety stock and safety lead time fields |
| Infor SyteLine | The reorder point and order multiple fields on purchased items |
| Rootstock | The planning policy and policy quantity fields |
| QAD | The item-site planning record fields |

Grade D is correct here even though the documentation is accurate: it establishes what a named
product documents about its own configuration, and nothing about industry practice, adoption, or
the frequency of any configuration error. No claim in this file rests on a vendor document for
anything other than a field name.

### 16.4 Community and anecdotal material — grade E

| Source | Used only for |
|---|---|
| SAP Community discussion of planning time fences on reorder-point MRP types, message 61839 | The existence of a reported incompatibility between reorder-point planning and time fencing in one release |

This is a single community report about a single release. It is cited because the structural point
it illustrates is real and is corroborated by the engine design, not because the report itself is
evidence of how often the combination fails.

### 16.5 What is inference rather than finding

Three statements in this file are the author's inference from field observation and are not
sourced to any study:

1. That "continuous review" usually means a nightly batch in practice.
2. That most organisations fail the reconstruction test on at least one item in ten.
3. That double buffering arises most often because two fields were set by two different people.

No source consulted measures any of these. They are stated as patterns worth testing against your
own system, and the tests are given in section 12 so the claim can be checked rather than believed.

## 17. Links

Within this module:

- Module index: [`_index.md`](_index.md)
- The cost-service frontier the policy is optimised against: [`01_cost-service-frontier.md`](01_cost-service-frontier.md)
- EOQ and the lot-size key: [`02_eoq-limits.md`](02_eoq-limits.md)
- Newsvendor, the periodic-review case with no replenishment: [`03_newsvendor.md`](03_newsvendor.md)
- Safety stock and the protection period: [`04_safety-stock.md`](04_safety-stock.md)
- The service measures the parameters are set against: [`05_service-metrics.md`](05_service-metrics.md)
- Lead-time demand distributions, which supply the buffer the trigger is built on: [`07_lead-time-demand.md`](07_lead-time-demand.md)
- Multi-echelon policies, where a trigger at one level depends on another: [`08_multi-echelon.md`](08_multi-echelon.md)
- Pooling and postponement, which change the item dimension of a policy: [`09_pooling-postponement.md`](09_pooling-postponement.md)
- Segmentation and DDMRP, which reparameterise rather than replace these families: [`10_segmentation-shelf-life.md`](10_segmentation-shelf-life.md)

Elsewhere in the program:

- Part II map: [`../_index.md`](../_index.md)
- Intermittent and compound demand, which changes what the trigger sees: [`../08_hierarchical_intermittent_probabilistic_forecasting/06_compound-poisson-hurdle.md`](../08_hierarchical_intermittent_probabilistic_forecasting/06_compound-poisson-hurdle.md)
- Demand sensing, which shortens the observation lag inside the lead time: [`../09_demand_sensing_promotions_npi/01_demand-sensing-signals.md`](../09_demand_sensing_promotions_npi/01_demand-sensing-signals.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file: [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
