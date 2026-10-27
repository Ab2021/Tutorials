# Case Pack, MOQ, and Integer Rounding

> Module 01A — Foundations: Units, Time, and Assumptions
> Topic 03 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01A.01, 01A.02

---

## 1. Why this topic exists

An optimiser returns 1,183.4 units. The number is correct. It is also not an instruction
anybody can carry out. Somebody must turn it into cases, and the moment they do, the plan
stops being the plan the optimiser evaluated. This topic is about that moment.

Three constraints get conflated in that conversion, and they are not the same kind of object:

- **Case pack** — a physical fact about the product. Units come in indivisible groups because
  somebody built a carton. The constraint lives at the level of the unit.
- **Order multiple or rounding value** — a system setting. A field in an enterprise resource
  planning (ERP) system says quantities must be a multiple of some number. The constraint lives
  at the level of the transaction record.
- **Minimum order quantity (MOQ)** — a commercial or physical floor. A supplier will not open
  a production line, a steriliser, or a truck for less. The constraint lives at the level of the
  contract.

A case pack of 250, an order multiple of 1,000, and an MOQ of 1,000 can produce four different
order quantities depending on which field a system applies first and how it treats the others.
The predecessor topic established that a case is a count bundled by an artefact and that
rounding is a one-way operation performed several times per cycle. This topic takes the next
step: it separates the three constraints, shows what each one does arithmetically, and shows
why their effects compound rather than cancel.

Four failures motivate the treatment.

**Failure one: the fractional plan is reported as the plan.** A solver returns 4.8 cases. The
output is reviewed and approved, and nothing asks whether 4.8 cases can be picked, packed,
shipped, or invoiced. The plan is arithmetically optimal and operationally empty.

**Failure two: one field is asked to do three jobs.** A company stores `1,000` as a minimum order
quantity and configures the system to round to that field. Weekly demand of 1,200 becomes 2,000,
because the system rounds *up to a multiple of* 1,000 rather than *up to at least* 1,000. The
buyer is then told the supplier demanded it.

**Failure three: rounding compounds through the bill of materials (BOM).** The rounded finished
good, not the forecast, drives component requirements, and each component is rounded again to its
own pack and MOQ. In section 6 one component's residue is 580 kg against a 330 kg safety stock.

**Failure four: nobody owns the direction.** Round up and inventory accumulates. Round to nearest
and small orders vanish. Round down and service fails where slack is thinnest. The rule was set
years ago, in another system, by someone who has left.

None of these is a modelling failure. Each is a representation failure: the model represents a
quantity the operation cannot produce. A data scientist who cannot compute the cost of rounding
cannot tell whether an optimisation result is worth implementing, nor defend a fractional answer
to the planner who must place the order.

**Where it leads.** Topic `02_container-loading.md` handled the top of the unit hierarchy and
asked which transport constraint binds. This topic handles the integer step that sits between
a continuous plan and a purchase order. Topic `04_calendars-weeks-time-zones.md` handles the
other discretisation: time. A plan is executable only when it is discrete in both dimensions,
in whole units and on real dates, and the two discretisations interact.

## 2. Learning outcomes

By the end of this topic you can:

1. **Separate** case pack, order multiple, and MOQ into three constraints with different owners,
   different levels of action, and different arithmetic, and **explain** why storing them in one
   field produces predictable errors.
2. **Compute** the order quantity for a given net requirement under ceiling, floor, and
   round-to-nearest, and **state** the residue and its sign for each.
3. **Derive** the expected residue per order for each rounding direction, and **explain** why
   round-up bias accumulates while round-to-nearest produces chronic shortfalls on small orders.
4. **Simulate** a twelve-week inventory profile under three lot-size policies and **compare**
   average inventory, holding cost, ordering cost, and service exposure.
5. **Distinguish** additive from multiplicative rounding error across a cascade, and **compute**
   the compounded overage for a stated number of levels and per-level error rates.
6. **Trace** rounding through a BOM, showing how a finished-good rounding factor propagates to
   every component and why the resulting errors are correlated rather than independent.
7. **Compare** a rounding residue with a computed safety stock and **decide** which dominates.
8. **Price** a rounding decision as extra holding cost against stockout risk and freight
   inefficiency, and **state** the break-even condition.
9. **Identify** the ratio of lot size to demand as the governing quantity for whether rounding
   is cheap or expensive, and **classify** an item as rounding-tolerant or rounding-sensitive.
10. **Report** a fractional optimum alongside its rounded implementation and the gap between
    them, so that the cost of integrality is visible rather than hidden.

Outcomes 2 to 7 are exercised quantitatively in section 12 and worked in section 13.

## 3. The core mental model

Hold three pictures. The first is a set of constraints acting at three different levels. The
second is an inventory sawtooth whose shape is decided by the rounding rule. The third is a
multiplication ladder in which every round-up multiplies the quantity that the next level sees.

### 3.1 Three constraints, three levels

```text
   LEVEL OF ACTION            CONSTRAINT              WHO OWNS IT        WHAT IT DOES
   ---------------------------------------------------------------------------------------
   unit / product             case pack               packaging eng.     makes 1,183.4
      |                       (divisibility)                             unrepresentable
      v
   order / transaction        order multiple          planning / IT      makes 1,250 into
      |                       or rounding value                          a legal 2,000
      v
   contract / supply          MOQ                     procurement /      makes a small
      |                       (commercial floor)      supplier           order impossible
      v
   transport / network        layer, pallet, truck     logistics          makes a case lot
                                                                         into a truck
```

Read the diagram downward and each row adds a constraint of a different kind. Read it upward and
each row may be applied before or after its neighbours, with different results. The most common
design error is to collapse rows two and three into one field, because both are "the smallest
quantity you can order". An order multiple is a *grid*: 1,000, 2,000, 3,000. An MOQ is a
*threshold*: 1,000 and anything above it, including 1,250. A system that stores an MOQ in a
rounding field implements a grid where the contract specified a threshold, and over-orders on
every requirement that sits between two grid points.

For most items the case pack is the smallest increment, the MOQ larger, and the transport unit
larger still. When the order is different — a case pack of 1 with an MOQ of 10,000 — the item is
a different kind of problem.

### 3.2 The sawtooth, and what decides its shape

```text
   inventory
   (units)
     ^
     |        /|        /|        /|        /|     round-up: lot 1,250, demand 1,200
     |      /  |      /  |      /  |      /  |     drift +50 per order
     |    /    |    /    |    /    |    /    |     floor rises
     |  /      |  /      |  /      |  /      |
     +-------------------------------------------> weeks
        1   2     3   4     5   6     7   8

     ^
     |\        /\        /\        /\              lot-for-lot with MOQ floor:
     | \      /  \      /  \      /  \             order 1,000, demand 800
     |  \    /    \    /    \    /    \            drift +200 per order,
     |   \  /      \  /      \  /      \           then a skipped order
     +-------------------------------------------> weeks

     ^
     |  /\      /\      /\      /\                  fixed lot 2,500, demand
     | /  \    /  \    /  \    /  \                 1,200 per week
     |/    \  /    \  /    \  /    \
     +-------------------------------------------> weeks
```

The sawtooth is the signature of a lot-size policy, and its three parameters are the lot size,
the demand rate, and the reorder point. Rounding changes the first parameter and therefore the
shape. Three shapes matter:

- **Round-up drift.** A lot slightly above demand leaves a residue on every order, so inventory
  ratchets upward rather than sawtoothing around a level. The annual drift is the residue times
  the number of orders.
- **MOQ sawtooth.** When the MOQ exceeds demand per order, the system orders the MOQ, consumes
  part of it, and orders again. Average inventory is roughly half the MOQ regardless of demand,
  so a slow-moving item carries the same average stock as a fast-moving one.
- **Fixed-lot sawtooth.** A fixed lot larger than demand per cycle produces deep cycles with a
  long tail and an average of roughly half the lot. Efficient when the lot is a transport or
  production unit, wasteful when it is not.

### 3.3 The cascade, and why the errors multiply

```text
   forecast demand        1,200 units/week
        |  round up to case pack 250
        v
   finished-good order    1,250 units        factor 1.0417
        |  x usage 0.35 kg per unit
        v
   component requirement    437.5 kg
        |  round up to 1,000 kg pallet
        v
   component order        1,000 kg           factor 2.286
        |  x scrap allowance 1.02
        v
   purchased quantity     1,020 kg           factor 1.020
```

Read the factors, not the quantities. Each level multiplies the previous quantity by a factor
greater than one, and the factors compose: `1.0417 × 2.286 × 1.020 = 2.429`, so the purchased
quantity is 2.43 times the true requirement, not 1.04 + 1.29 + 0.02 times it. Additive intuition
understates the result by a wide margin whenever any single step has a large factor.

Three properties follow, and they are the core of the mental model.

**Property 1 — the optimum is a bound, not a plan.** A continuous optimum is the best achievable
if quantities were divisible. It is a lower bound on cost. The integer plan is feasible and
therefore an upper bound. The difference between them is the price of integrality, and it is
knowable: compute both, report both. A planner who is shown only the fractional number will
distrust analytics, and a planner who is shown only the rounded number will never know what the
rounding cost.

**Property 2 — rounding direction is a policy with an asymmetric cost.** Ceiling is safe for
service and expensive for inventory. Floor is the reverse. Round-to-nearest is unbiased in the
mean over large quantities and catastrophic on quantities below half an increment, where it
rounds to zero. The asymmetry is not a defect of any one rule; it is the reason the rule must be
chosen per item class rather than inherited globally.

**Property 3 — rounding cost scales with the ratio of lot size to demand.** A case pack of 1 is
free. A case pack of 250 against a weekly demand of 1,200 costs 4.2% per order. A pallet of
1,000 kg against a weekly requirement of 420 kg costs 138% on the first order. The same
absolute increment is negligible for one item and ruinous for another, so no universal rounding
rule can be correct.

### 3.4 Where the fractional plan is still useful

The continuous optimum is not an error to be corrected. It is the benchmark against which the
rounded plan is judged, and the input to a rounding procedure. Carry it through the pipeline as
a labelled lower bound, alongside the executable upper bound, and report the gap between them.
If the gap is 1%, the rounding policy is not worth arguing about. If the gap is 25%, the policy
is a bigger lever than the forecast, and the meeting should be about the policy.

## 4. Definitions and vocabulary

The fourth column names the unit or coding system in which the term is normally expressed,
because a definition without a unit is the defect this module exists to prevent.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Case pack | The number of base units inside one shipping case | Confused with the order multiple | Integer, base units per case |
| Order multiple | The increment in which a quantity may be ordered; legal quantities are integer multiples of it | Confused with MOQ, which is a floor rather than a grid | Units, stated in a named unit of measure |
| Rounding value | The ERP field that rounds a calculated quantity up to a multiple of a stated number | Set equal to the case pack when the case pack changes | Units |
| Rounding profile | A scaled rounding rule with thresholds, so the increment changes at stated quantities | Assumed to be a single constant | Threshold and increment pairs |
| MOQ | Minimum order quantity: the smallest quantity a supplier will accept or produce | Implemented as a multiple rather than a floor | Units, or kg, stated per order and per period |
| Minimum lot size | The smallest production or purchase lot the planning system will propose | Used as a synonym for MOQ across an ownership boundary | Units |
| Maximum lot size | The largest quantity the system will place in one proposal; excess is split | Ignored, so one proposal violates a storage or shelf-life limit | Units |
| Lot size | The quantity produced or ordered in one lot | Used to mean the demand for a period | Units |
| Lot sizing | The decision rule that converts a net requirement into a lot size | Treated as a forecasting problem | Rule or algorithm |
| Lot-for-lot (L4L) | Order exactly the net requirement each period | Assumed to be optimal because it minimises inventory | Units per period |
| Fixed order quantity | A policy in which every order is the same stated quantity | Confused with a fixed lot multiple | Units per order |
| Fixed lot multiple | A policy in which orders are integer multiples of a stated quantity | Confused with fixed order quantity | Units, as a multiple |
| Period order quantity | A policy that orders enough for a fixed number of periods | Computed without regard to the case pack | Units per interval |
| Economic order quantity | The order size minimising ordering plus holding cost under stated assumptions | Quoted as optimal when demand is lumpy or a MOQ binds | Units, usually fractional |
| Net requirement | Gross requirement minus available inventory and scheduled receipts | Confused with gross requirement | Units per bucket |
| Gross requirement | Total demand for an item in a period, before netting | Treated as the order quantity | Units per bucket |
| Planned order | A system proposal that has not been converted to a purchase or production order | Treated as a commitment | Units, with a date |
| Order proposal | The output of the lot-size calculation in a planning run | Assumed to respect every constraint | Units |
| Integer constraint | A requirement that a decision variable take a whole-number value | Assumed to be a minor technicality | Integer |
| Continuous relaxation | The same model with the integer constraint removed | Reported as the answer | Real number |
| Integrality gap | The difference between the relaxed and integer optimum, as a ratio or a percentage | Ignored in reporting | Ratio or percent |
| Ceiling | Rounding up to the next multiple of the increment | Assumed to be the only safe rule | Integer |
| Floor | Rounding down to the previous multiple of the increment | Used without stating the service consequence | Integer |
| Round-half | Rounding to the nearest multiple, with a stated tie rule | Assumed to be unbiased at small quantities | Integer |
| Rounding residue | The quantity added or removed by one rounding operation | Called waste, overbuild, or safety stock | Base units or kg |
| Compounded residue | The accumulated effect of several rounding operations in sequence | Computed by adding the individual residues | Base units or kg |
| Divisibility | The property that a quantity can be split into whole units without loss | Assumed for every unit in the hierarchy | Boolean at a stated granularity |
| Granularity | The smallest representable increment at a stated level | Confused with precision | Units |
| Executable plan | A plan whose every quantity can be picked, packed, shipped, and invoiced as stated | Confused with a feasible plan, which may still violate a contract | Units, dates, and locations |
| Rounding exposure ratio | Lot size divided by demand over the same period; the governing quantity for rounding cost | Never computed | Dimensionless |
| Days of supply | Inventory divided by average daily demand | Quoted without the demand basis | Days |
| Cycle stock | Inventory that exists because supply arrives in lots | Called safety stock | Units |
| Safety stock | Inventory held to cover variability in demand or lead time | Used to absorb rounding residue | Units |
| Sawtooth | The inventory profile produced by periodic replenishment in lots | Described as a trend | Units over time |
| Order batching | Combining requirements into larger orders to reduce order frequency | Confused with case-pack rounding | Units per order |
| Break-even lot size | The lot size at which the cost of ordering equals the cost of holding | Used outside its cost assumptions | Units |
| All-units discount | A price schedule in which the discount applies to every unit once a threshold is crossed | Confused with an incremental discount | Currency per unit, by tier |
| Set-up cost | The cost incurred once per production run | Treated as a constant when it depends on the changeover | Currency per run |
| Changeover time | Time lost converting a line from one item to another | Ignored in lot sizing | Minutes or hours |
| Full pallet | A pallet quantity that cannot be broken without repacking | Used as an order multiple without checking the case pack | Cases per pallet |
| Truckload | The quantity at which a vehicle is full by weight or cube | Treated as a rounding increment for a single item | kg or m³ |
| Price of integrality | The cost difference between the continuous optimum and the best integer plan | Reported as rounding error | Currency per period |

Four rows carry most of the confusion.

**Order multiple versus MOQ.** An order multiple defines a lattice; an MOQ defines a half-line.
For a requirement of 1,200 with a multiple of 1,000 the answer is 2,000; with an MOQ of 1,000 it
is 1,200, subject only to the case pack. A system with one field cannot express both, and the
direction of the error is always upward, because rounding up satisfies the constraint.

**Rounding residue versus safety stock.** Both sit above requirement and answer different
questions. Safety stock covers variability and should shrink when lead time and demand
variability fall; residue covers indivisibility and does not. When a company says its safety
stock is "about three weeks", check whether the number describes a buffer or absorbs pack
rounding — the remedies differ.

**Integrality gap versus rounding error.** The gap is a property of the model: how much worse the
best integer solution is than the continuous one. The rounding error is a property of one
heuristic: how much worse a specific rounded quantity is than the best integer one. A good
procedure has a small error even when the gap is large, so a claim about one is not a claim about
the other.

**Cycle stock versus safety stock versus residue.** Cycle stock exists because the lot exceeds
one period of demand, safety stock because the future is uncertain, residue because the lot
cannot be any size. A plan reporting one number for "inventory above requirement" cannot say
which lever would reduce it.

## 5. The mechanics, step by step

This section converts a requirement into an executable order in eleven steps, grouped into four
phases. Each step names the arithmetic, the data that should carry it, and the failure that
occurs when the step is skipped or merged with a neighbour.

### Phase A — Establish which constraint binds

**Step 1. State the requirement in base units, with its time bucket.**

- Arithmetic: the net requirement is gross requirement minus on-hand minus scheduled receipts,
  in base units per bucket. It is not rounded.
- Data representation: a net requirement line with item, location, bucket, quantity, and unit.
- If skipped: the requirement is rounded at entry, and every later comparison measures the
  rounded number against a rounded plan.

**Step 2. Enumerate the applicable constraints separately, with units.**

- Arithmetic: none yet. This is an inventory of constraints, not a calculation.
- Data representation: one row per constraint, with a type (divisibility, grid, threshold,
  transport), a value, a unit, an owner, and an effective date.
- If skipped: the constraints are discovered one at a time, in the order the system happens to
  apply them, and the effective rule is whatever the configuration file says.

**Step 3. Determine which constraint binds at the current demand level.**

- Arithmetic: compute the requirement as a fraction of each constraint's increment. The case
  pack binds if the fractional part is non-zero. The MOQ binds if the requirement is below it.
  The transport unit binds if the lot exceeds the vehicle or pallet capacity.
- Data representation: a binding-constraint flag on the plan line, with the constraint named.
- If skipped: the planner is told the order is 2,000 because "the system rounded it", and cannot
  tell whether the cause was the pack, the grid, or the contract.

The binding test is worth stating as a small table, because most disputes about order quantity
are disputes about which row applies.

| Requirement | Case pack 250 | Order multiple 1,000 | MOQ 1,000 | Result and binding constraint |
|---:|---|---|---|---|
| 1,200 | 1,250 | 2,000 | 1,200 | 2,000 if the grid applies; 1,250 if only the pack applies |
| 1,200 | 1,250 | none | 1,000 | 1,250; the pack binds, the MOQ does not |
| 800 | 1,000 | none | 1,000 | 1,000; the pack and the MOQ coincide |
| 300 | 500 | none | 1,000 | 1,000; the MOQ binds and is twice the pack-rounded need |
| 12,000 | 12,000 | none | 1,000 | 12,000; the transport unit may bind above this |

**Step 4. Decide the rounding level and round once.**

- Arithmetic: choose the level whose increment is largest among those that must be satisfied
  at the point of order, and round there. Rounding at every level compounds.
- Data representation: a `rounded_at_level` attribute on the plan line.
- If skipped: the pack, the grid, and the pallet each round independently, and the residues add
  instead of the largest one governing.

### Phase B — Apply the rounding policy

**Step 5. Apply the rule and record the residue with its sign.**

- Arithmetic: `rounded = rule(requirement / increment) × increment`; `residue = rounded −
  requirement`. The sign matters: positive is inventory, negative is shortfall.
- Data representation: exact quotient, rounded quantity, rule identifier, residue, and the
  residue expressed as a percentage of requirement.
- If skipped: only the rounded quantity survives, and the residue is attributed to forecast
  error at the next review.

**Step 6. Compute the expected residue per order for the chosen rule.**

- Arithmetic: for ceiling with a uniformly distributed fractional part, the expected residue is
  half an increment. For floor it is minus half an increment. For round-half it is zero in
  expectation and unbounded in variance as the requirement falls below half an increment.
- Data representation: a policy parameter per item class, not a global default.
- If skipped: the policy is chosen on intuition, and the intuition is usually "round up, it is
  safer", which is true for service and false for cash.

**Step 7. Compute the number of rounding events per year.**

- Arithmetic: `events = annual demand / lot size`. The annual residue is the residue per event
  times the number of events, which is why rounding a weekly order costs more than rounding an
  annual one.
- Data representation: an annualised residue quantity and value per item.
- If skipped: a residue that looks small per order is never multiplied out, and a 50-unit
  weekly residue is reported as negligible while amounting to 2,600 units a year.

### Phase C — Cascade across levels

**Step 8. Explode the BOM from the rounded quantity, not the forecast.**

- Arithmetic: component requirement equals rounded parent quantity times usage, plus scrap.
  The parent's rounding factor applies to every component.
- Data representation: a requirement line that records which parent quantity it was derived
  from, including whether that parent was rounded.
- If skipped: the planner explodes the forecast, orders components for the forecast, and
  discovers at build time that the rounded parent needs more.

**Step 9. Round each component at its own level, record the factor, and compare the residue with
the safety stock.**

- Arithmetic: each component's order is its requirement times its own factor, and the factors
  multiply along the path. If the residue per replenishment exceeds safety stock, the buffer is
  dominated by rounding and no longer covers variability.
- Data representation: a rounding factor and a cumulative factor on each BOM path, and the
  residue and safety stock reported side by side in the same unit.
- If skipped: the compounding is invisible, each step looking reasonable in isolation, and the
  company raises safety stock to fix service without addressing the cause.

### Phase D — Price the decision and report it

**Step 10. Convert the residue into holding cost, service exposure, and freight effect.**

- Arithmetic: extra holding cost is residue times unit cost times holding rate times time held.
  Service exposure is the probability that a floor rule's shortfall exceeds the buffer. Freight
  effect is the cost of the partial pallet or container.
- Data representation: a cost line per mechanism, not one aggregated "rounding cost".
- If skipped: the rounding decision is argued in units, which nobody can compare with cash.

**Step 11. Report the fractional optimum, the rounded plan, and the gap.**

- Arithmetic: `gap = (cost of rounded plan − cost of continuous optimum) / cost of continuous
  optimum`.
- Data representation: two labelled rows in the plan output, with the rounding policy named.
- If skipped: the organisation cannot tell whether to invest in a better solver or a better
  pack configuration, and usually invests in the solver.

## 6. Worked example

The example is constructed to be internally consistent and is not drawn from a named company.
Every parameter is stated so that the arithmetic can be reproduced.

### 6.1 The item and its constraints

A purchased component used in a medical device assembly. Demand is level at 1,200 units per
week, 62,400 units per year, with no seasonality, for the purpose of isolating the rounding
effect.

| Parameter | Value | Note |
|---|---:|---|
| Weekly demand | 1,200 units | Level, deterministic in this example |
| Annual demand | 62,400 units | 1,200 × 52 |
| Case pack | 250 units | Physical, set by the carton |
| MOQ | 1,000 units | Contractual, per order |
| Pallet layer | 48 cases | 12,000 units per layer |
| Unit standard cost | €12.00 | For holding cost |
| Holding rate | 26% per year | €3.12 per unit per year |
| Weekly holding cost | €0.06 per unit | 12.00 × 0.26 / 52 |
| Ordering cost | €180 per order | Buyer time, receiving, inspection |

> **Note:** the pallet layer is stated as 48 *cases*, not 48 units. A literal layer of 48 units
> cannot be built from 250-unit cases, and a master-data field reading "48" without a unit is
> exactly the defect the predecessor topic described. If the field meant units, the pack and the
> layer are mutually inconsistent and one is wrong.

The MOQ of 1,000 is below the weekly requirement of 1,200, so it does not bind in the base case.
It is retained because section 6.6 shows what happens when demand falls and it does.

### 6.2 The continuous benchmark

Before any rounding, compute the economic order quantity (EOQ) with the standard square-root
formula, which assumes constant demand, a fixed cost per order, and holding cost proportional to
quantity and time.

```text
EOQ = sqrt(2 × D × K / h) = sqrt(2 × 62,400 × 180 / 3.12)
    = sqrt(22,464,000 / 3.12) = sqrt(7,200,000) = 2,683.28 units
annual cost = sqrt(2 × D × K × h) = sqrt(2 × 62,400 × 180 × 3.12)
    = sqrt(70,087,680) = EUR 8,371.90 per year
```

The EOQ is 2,683.28 units, or 10.73 cases, ordered 23.25 times a year, every 2.24 weeks. Pro
rata over twelve weeks, its cost is `8,371.90 × 12/52 = €1,932.00`. This is the lower bound. No
executable policy can beat it, and the question is how far each policy sits above it.

### 6.3 Policy A — lot-for-lot with an MOQ floor, rounded to the case

The requirement each week is 1,200. The MOQ of 1,000 does not bind. The case pack does:
`1,200 / 250 = 4.8`, which rounds up to 5 cases, or 1,250 units. The policy orders 1,250 every
week.

The residue is 50 units per order, or 4.17% of the requirement. Over 52 weeks that is 2,600
units, or 2.17 weeks of demand. Because every week behaves the same way, the twelve-week profile
is an arithmetic progression: the closing balance is `50n` after week `n`, so it runs 50, 100,
150, and so on to 600 at week 12. The average of an arithmetic progression is the mean of its
endpoints, `(50 + 600) / 2 = 325 units`, which agrees with `(50 + 100 + ... + 600) / 12`.

Holding cost is `325 × €0.06 = €19.50 per week`, or `€234.00` over twelve weeks. Ordering cost
is `12 × €180 = €2,160.00`. Total is **€2,394.00**, which is 23.9% above the benchmark.

Two observations matter more than the total. First, the drift is monotonic: inventory is not
cycling around a level, it is climbing by 50 units a week. In a real system the planner would
eventually skip an order or reduce one, and that correction is an unrecorded manual
intervention. Second, the policy holds almost no cycle stock — average 325 units is 0.27 weeks
of demand — so it has no buffer at all against variability. It is cheap in inventory only
because it is expensive in orders and fragile in service.

### 6.4 Policy B — the EOQ rounded up to the case

Round the fractional EOQ up to a whole case: `2,683.28 / 250 = 10.73`, so 11 cases, or 2,750
units. Order 2,750 whenever the opening inventory cannot cover the week.

| Week | Opening | Receipt | Demand | Closing | Order? |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 2,750 | 1,200 | 1,550 | Yes |
| 2 | 1,550 | 0 | 1,200 | 350 | No |
| 3 | 350 | 2,750 | 1,200 | 1,900 | Yes |
| 4 | 1,900 | 0 | 1,200 | 700 | No |
| 5 | 700 | 2,750 | 1,200 | 2,250 | Yes |
| 6 | 2,250 | 0 | 1,200 | 1,050 | No |
| 7 | 1,050 | 2,750 | 1,200 | 2,600 | Yes |
| 8 | 2,600 | 0 | 1,200 | 1,400 | No |
| 9 | 1,400 | 0 | 1,200 | 200 | No |
| 10 | 200 | 2,750 | 1,200 | 1,750 | Yes |
| 11 | 1,750 | 0 | 1,200 | 550 | No |
| 12 | 550 | 2,750 | 1,200 | 2,100 | Yes |

Six orders of 2,750, or 16,500 units against a demand of 14,400. The closing inventory total is
16,400, so the average is `16,400 / 12 = 1,366.67 units`. Holding cost is
`1,366.67 × €0.06 = €82.00` per week, or `€984.00` over twelve weeks. Ordering cost is
`6 × €180 = €1,080.00`. Total is **€2,064.00**, or 6.8% above the benchmark.

The 6.8% is the rounding penalty: the difference between the best integer policy and the
continuous optimum, under a level-demand assumption. It is close to the classic bound for
power-of-two lot sizing, which is about 6% in the worst case over an infinite horizon. The
coincidence is not exact and should not be over-read, but it illustrates the useful rule that
rounding a well-chosen lot size is usually cheap, while rounding a badly chosen one is not.

### 6.5 Policy C — a fixed order quantity of 5,000 units

Suppose logistics imposes a minimum consignment of 5,000 units, or the buyer orders monthly.
The policy orders 5,000 whenever stock cannot cover the week, in weeks 1, 5, and 9. The closing
balance then runs `3,800, 2,600, 1,400, 200` and repeats, because `5,000 − 4 × 1,200 = 200`.

Three orders of 5,000, or 15,000 units. The closing inventory total is 26,400, so the average is
`26,400 / 12 = 2,200 units`. Holding cost is `2,200 × €0.06 = €132.00` per week, or `€1,584.00`
over twelve weeks. Ordering cost is `3 × €180 = €540.00`. Total is **€2,124.00**, or 9.9% above
the benchmark.

Policy C is cheaper than Policy A and dearer than Policy B on this cost basis. It carries more
inventory and fewer orders, and it has a service advantage that the cost comparison does not
show: its minimum stock before replenishment is 200 units, so it absorbs a demand spike of up to
3.5 weeks. Policy A absorbs none.

### 6.6 The comparison, and the constraint that was not binding

| Policy | Lot size | Orders in 12 wk | Units ordered | Avg inventory | Holding | Ordering | Total | vs benchmark |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Continuous EOQ | 2,683.28 | 5.4 | 14,400 | — | — | — | €1,932.00 | 0.0% |
| A: L4L, case-rounded | 1,250 | 12 | 15,000 | 325.00 | €234.00 | €2,160.00 | €2,394.00 | +23.9% |
| B: EOQ, case-rounded | 2,750 | 6 | 16,500 | 1,366.67 | €984.00 | €1,080.00 | €2,064.00 | +6.8% |
| C: fixed 5,000 | 5,000 | 3 | 15,000 | 2,200.00 | €1,584.00 | €540.00 | €2,124.00 | +9.9% |

Now change one parameter to make the MOQ bind. Suppose demand falls to 800 units per week while
the MOQ stays at 1,000.

- The case pack forces 4 cases, or 1,000 units, so the MOQ of 1,000 is satisfied exactly and the
  residue is 200 units per order, 25% of requirement.
- Ordering weekly, that residue is 10,400 units a year against annual demand of 41,600.
- Average closing inventory over twelve weeks is `(200 + 400 + ... + 2,400) / 12 = 1,300 units`,
  1.63 weeks of demand held with no variability justification.

This is the MOQ sawtooth: the average inventory is roughly half the order quantity and is
independent of demand. Halve the demand again to 400 a week and the policy still orders 1,000,
so the average inventory stays near 500 units while demand falls, and the item's days of supply
double. A slow-moving item with a large MOQ carries inventory in proportion to the MOQ, not to
its demand, which is why MOQ problems concentrate in the tail of the item portfolio and why they
are invisible in aggregate reports.

### 6.7 The BOM cascade

Take the base case and explode one level. The finished good is the component above; it is built
from two purchased parts.

| Component | Usage per FG unit | Requirement at 1,200 FG | Pack | MOQ | Order | Factor |
|---|---:|---:|---|---:|---:|---:|
| X | 2.00 units | 2,400 units | 500 units per case | 1,000 | 2,500 | 1.042 |
| Y | 0.35 kg | 420 kg | 1,000 kg per pallet | 1,000 kg | 1,000 | 2.381 |

Component X: `2,400 / 500 = 4.8`, so 5 cases, or 2,500 units. Residue 100 units, factor
`2,500 / 2,400 = 1.0417`.

Component Y: `420 / 1,000 = 0.42`, so 1 pallet, or 1,000 kg. Residue 580 kg, factor
`1,000 / 420 = 2.3810`. The MOQ and the pallet coincide at 1,000 kg, so the constraint that binds
is the pallet, and the MOQ merely confirms it.

Now apply the finished-good rounding too. Ordered at 1,250 rather than 1,200, the component
requirement is `1,250 × 0.35 = 437.5 kg`, still one pallet, so the component factor is unchanged
at this granularity. But the compounded factor from forecast to component order is
`(1,250 / 1,200) × (1,000 / 437.5) = 1.0417 × 2.2857 = 2.381`: 4.2% from the finished good and
128.6% from the component, whose product is 138.1%, not the 132.8% that addition gives.

The general identity is worth memorising. For factors `f_i = 1 + e_i`:

```text
product of factors = (1 + e_1)(1 + e_2)...(1 + e_n)
                    = 1 + sum(e_i) + sum over pairs(e_i e_j) + ... + product(e_i)

for three levels with e = 0.0417, 1.2857, 0.020:
additive estimate  = 1 + 0.0417 + 1.2857 + 0.020 = 2.347
multiplicative     = 1.0417 × 2.2857 × 1.020 = 2.429
difference         = 0.082, or 8.2 percentage points
```

The difference grows with the size of the individual errors, which is why addition is acceptable
for a chain of 2% pack roundings and unacceptable for any chain containing a MOQ several times
the requirement.

### 6.8 When the residue exceeds the safety stock

The safety stock for component Y is computed from demand variability and lead time:

```text
safety stock = z × sigma × sqrt(L)
             = 1.65 × 200 kg/week × sqrt(1 week)
             = 330 kg
```

The rounding residue on one order is 580 kg and the average cycle stock is about 500 kg; both
exceed the 330 kg buffer. The consequence is precise: protection against a one-standard-deviation
demand spike is smaller than the quantity added by the act of ordering. The company believes it
holds 330 kg of protection and in fact holds a rounding artifact of 580 kg, consumed whether or
not demand requires it and replenished next order regardless of the buffer.

The remedy is not to raise the safety stock, which increases the inventory the residue already
dominates, but to order Y in a way that matches consumption: aggregate orders across the items
sharing the pallet, negotiate a lower MOQ, or use a smaller pack.

### 6.9 The pallet and truck cascade, and why it is not applied to one item

The finished good's case pack gives 5 cases per week. The pallet layer is 48 cases, or 48 weeks
of demand; a four-layer pallet is 48,000 units, or 40 weeks; a 26-pallet truck is 1,248,000
units, or 20 years. The arithmetic is correct and the conclusion is absurd: no policy should
round this item to a truck. The reason is the rounding exposure ratio, lot size divided by
annual demand: 20.0 for the truck against `1,250 / 62,400 = 0.02` for the weekly case pack, which
is why the pack costs 4.2% per order rather than 2,000%.

The rule follows: transport units are rounded at the level of the *consignment*, which contains
many items, not at the level of the *item*. A truck constrains the shipment, not the
stock-keeping unit, and applying a vehicle constraint to one item is the category error that
produces the twenty-year order — usually because a planner copied a "full truckload" field into
an item-level rounding value.

## 7. Code, query, or calculation

One block, and it is pseudocode rather than a program, because the arithmetic is the point.
The procedure makes three things explicit that a standard rounding function hides: which
constraint applies, the residue with its sign, and the annualised exposure.

```text
apply_policy(requirement, case_pack, order_multiple, moq, rule):
    q = requirement
    if moq is not null and q < moq: q = moq          -- threshold, not a grid
    if order_multiple is not null:                   -- grid, applied to the result
        q = rule(q / order_multiple) * order_multiple
    if case_pack is not null:                        -- divisibility, applied last
        q = ceil(q / case_pack) * case_pack
    residue = q - requirement
    annual_events = annual_demand / q
    annual_residue = residue * annual_events
    return q, residue, residue / requirement, annual_residue
```

Three notes. The order of the last two operations is a policy decision: applying the pack last
guarantees divisibility but can break the order multiple, and the reverse also holds. The `rule`
argument is `ceil`, `floor`, or `round_half`, passed explicitly so no default is inherited
silently. And the annualised residue uses the lot size, not the requirement, because the number
of orders is `D / Q` — a difference that matters when the residue is large.

A useful test set: requirement 1,200 with pack 250 returns 1,250 under ceiling and 1,000 under
floor; requirement 100 with pack 250 returns 250 under ceiling, 0 under floor, and 0 under
round-half, which shows why round-half is unsafe for small orders; requirement 0 returns 0
without a division error; requirement 1,250 returns 1,250 with zero residue under every rule.

## 8. Data, standards, and regulatory context

There is no international standard that defines a minimum order quantity. MOQ is a commercial
term, written into contracts and represented in software fields whose semantics are set by the
vendor. That absence is the analytical problem: a case pack has a physical referent that can be
counted on a dock, an MOQ has only a clause, and an order multiple has only a configuration
setting.

### 8.1 What standards do govern

| Standard or system | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN Management Standard | Product identity at each packaging level | A change in case quantity requires a new identifier at that level | B |
| UN/CEFACT Recommendation 20 | Unit-of-measure codes | The unit in which the quantity is expressed | B |
| ANSI X12 850 / UN/EDIFACT ORDERS | Purchase order messages | The quantity field and its unit code, but not whether the quantity is legal under a packing rule | B |
| GS1 EANCOM data element 6411 | Unit of measure in EDI | Points implementers to Recommendation 20 | B |
| ISO 8000-110 | Master-data quality | Syntactic and semantic quality of attributes, including pack data | B |
| FDA unique device identification (UDI) | Medical device identification | Device identifier and production identifier at the packaging level | B |

The GTIN rule bites hardest here. A case pack is not a parameter that can be edited in place; a
change creates a new trade item at the case level. Historical analysis spanning the change joins
two different products, and a rounding rule keyed on the old pack is applied to a quantity that
no longer exists physically.

### 8.2 What enterprise systems actually store

Field names differ across vendors, and the differences determine whether a quantity is a floor,
a grid, or a fixed value.

| System and field | Semantics | Level | Grade |
|---|---|---|---|
| SAP rounding value | Rounds the calculated quantity up to a multiple of the value | Transaction | D |
| SAP rounding profile | Scaled rounding with thresholds, for example 2→5 and 32→40; configured, not delivered by default | Transaction | D |
| SAP minimum lot size | Lower bound on the order proposal | Transaction | D |
| SAP maximum lot size | Upper bound; excess is split across proposals | Transaction | D |
| SAP "last lot exact" setting | Allows the final lot in the horizon to ignore the rounding value | Horizon | D |
| Oracle fixed order quantity | Every order is the stated quantity, and the interval varies | Transaction | D |
| Oracle fixed lot multiple | Orders are integer multiples of the stated quantity | Transaction | D |
| Oracle minimum order quantity | A floor applied before rounding | Contract | D |
| Oracle rounding | Applied last, after the other modifiers | Transaction | D |

The Oracle documentation states an application order: fixed days supply, fixed order quantity,
fixed lot multiple, minimum order quantity, maximum order quantity, then rounding. The SAP
documentation describes a different set of interactions, including the horizon exception. Both
are vendor documentation, not independent evidence. Grade D, 2026.

> **Note:** the existence of a documented precedence order is the evidence that matters here.
> If a vendor with a mature planning product needs six ordered modifiers and a horizon exception
> to express these constraints, then a data model with a single `min_order_qty` column is not a
> simplification of the problem. It is a different and weaker problem.

Two consequences follow. A query that reads one field named `moq` cannot know whether the engine
treated it as a floor or a grid, so the reconstructed quantity will disagree with the engine on
some items with nothing in the field name to explain which. And the same item can carry a
different effective constraint in two systems — ERP purchase orders and planning-tool proposals
— with the divergence visible only when both are compared for the same requirement.

### 8.3 Regulatory constraints that are not MOQs but behave like them

Three regulatory mechanisms create a quantity floor without being commercial MOQs, and they are
frequently conflated with one.

- **Validated batch size.** In pharmaceutical and some medical-device manufacturing, the batch
  size is part of the validated process. Changing it is a change-control event requiring
  supporting data, so the production lot size is not a free planning variable even when demand
  would justify a smaller lot. Grade B for the existence of the requirement; the specific
  obligations vary by jurisdiction and product class.
- **Minimum sterilisation or process load.** A steriliser, autoclave, or oven has an economic
  minimum load, and running it below that load wastes a cycle. The constraint is physical and
  applies to the load, not to the item.
- **Minimum analytical or stability batch.** A stability programme may require a minimum batch
  to generate meaningful data, which fixes a floor for the product rather than for the order.

These differ from an MOQ in one respect that matters: they are properties of the process, not of
the commercial relationship. A supplier can renegotiate an MOQ; a validated batch size changes
by technical file submission. A model that stores both in one field cannot tell which lever is
available.

### 8.4 What is convention rather than requirement

- "Order in full cases" is a convention of the channel, not a physical necessity: a distribution
  centre that breaks bulk is a design choice with a cost.
- "MOQ equals one pallet" is a supplier's habit in many categories, not a law, and renegotiating
  it is a legitimate output of the analysis in section 6.
- "One order per week per item" is an artefact of the planning calendar, not of the product, and
  is the subject of Topic 04.

An analyst who can separate these conventions from requirements can propose changing one. An
analyst who cannot will reproduce the current policy in a model and call the result optimal.

## 9. Industry practice and case evidence

### 9.1 Case-pack rounding as a measured cause of amplification

The most directly relevant case is Tesco's automated store ordering system, reported by Potter
and Disney in a 2010 POMS conference paper. The system used an order-up-to policy in which
orders were multiples of a fixed case size, and the study reports that daily distribution-centre
workload variability was amplified substantially relative to sales — the figure given is 185%.
Adding a proportional controller reduced order variability to about 75% of sales variability and
was associated with an annual saving of roughly £28 million. The paper notes that case-pack
rounding is worst for low-volume, long-shelf-life products, where demand can be as low as one
case per week. Grade C, conference case study, company-reported magnitudes.

Three features transfer. The mechanism is the one in section 3: a fixed increment applied to a
requirement comparable to or smaller than itself. The effect was measured at the distribution
centre's workload rather than the item, which is where the operational cost appears. And the
remedy paired a control policy with a pack-size argument — the rule and the increment are both
levers.

### 9.2 The batch-size effect, quantified by simulation

Holland and Sodhi (2004) simulate a supply chain in which orders must be integer multiples of a
batch size and report that the increase in order variance over a baseline is proportional to the
square of the batch size and to the variance of order deviations. Grade A, peer-reviewed.

The square term is the useful part: doubling the order increment roughly quadruples the
distortion. That is consistent with section 3.3, where the residue scales linearly with the
increment while its variability scales with the increment relative to demand, and the two
compound. So the largest pack in a portfolio does disproportionate damage, and a pack-reduction
programme should start with the items whose increment is largest relative to demand, not those
with the largest total volume.

### 9.3 Breaking bulk as the alternative

Ketzenberg, Metters, and Vargas (2002) study case-pack ordering against breaking bulk at the
distribution centre for grocery products, using dynamic programming and a 77-product simulation.
The paper links case-pack granularity to inventory, shelf space, and profit, and treats breaking
bulk as the option that buys granularity at the cost of handling. Grade A, peer-reviewed.

The framing matters: the case pack is not a fixed constraint of nature but a constraint with a
price, and the price can be paid elsewhere — in the DC, in labour, or in the pack itself. An
analysis that treats the pack as immovable has conceded the most valuable degree of freedom.

### 9.4 MOQ in the inventory literature

The MOQ literature is unambiguous that the classical economic order quantity is often
inapplicable. Alnahhal and colleagues (2024), reviewing the state of the art in *Sustainability*,
state that suppliers may enforce an MOQ much larger than the EOQ, and that when the EOQ falls
below the MOQ the calculation is pointless because the MOQ governs. Grade A, peer-reviewed review.

The technical treatment is harder than practitioner discussion suggests. Zhao and Katehakis
(2006) show the optimal policy under an MOQ is non-monotone, which is why simple rules are used.
Kiesmüller, de Kok, and Dabia (2011) propose an (R, S, Qmin) policy for periodic review whose
costs fall mostly within 1% of the complex alternative, with parameters computable in a
spreadsheet. Grade A for both. The reading: an MOQ changes the shape of the optimal policy, but a
well-chosen simple policy is close enough that the argument should be about the MOQ itself.

### 9.5 The cost of rounding a lot size

Bertazzi (2003) analyses rounding the continuous economic lot size to a discrete set of reorder
intervals. The classic power-of-two result gives a worst-case cost ratio of about 1.06, and the
paper examines how naive rounding performs against that bound. Grade A, peer-reviewed. This is the
formal version of section 6.4: rounding a lot near a good value costs a few percent, while
rounding one chosen for another purpose costs far more, since the cost function is flat only near
its minimum.

### 9.6 MOQ and expiry in pharmaceutical distribution

The clearest empirical link between MOQ and a physical consequence comes from a 2022 Metropolia
thesis by Mäkelä, an ABC-XYZ analysis of 321 pharmaceutical stock-keeping units for a Finnish
company. MOQ is identified among the causes of scrapped expired stock, alongside forecast error,
mandatory reserve stocks, and declining sales; all fifteen of the top expiry-cost batches were in
the volatile-demand or low-sales classes. Grade C, single-company thesis.

The mechanism is the one in section 6.6: when the MOQ exceeds demand per cycle, the order is set
by the MOQ rather than by consumption, days of supply rise as demand falls, and for a product
with a shelf life the excess becomes waste. The thesis adds the structural version: long lead
times and high MOQs in primary manufacturing cannot respond to short-term demand changes, so a
slow-moving, volatile product faces expiry and shortage at once.

### 9.7 What the evidence does not establish

No source located for this topic reports a representative base rate for rounding cost across a
population of companies. The Tesco figures are one case with company-reported magnitudes; the
simulation results establish mechanisms under stated assumptions; the vendor documentation
describes fields, not outcomes. Treat the evidence as establishing direction and mechanism, and
measure the magnitude in your own data. The one robust number is arithmetic: a residue of `r`
per order on `N` orders a year is `r × N` units a year, and no case study is needed to compute it.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Case pack and MOQ stored in one field | Orders are always a round multiple of the MOQ | The field is applied as a grid rather than a threshold | Over-order on every requirement between grid points | Separate fields with separate semantics and owners |
| Fractional quantity written to the order | Purchase orders with decimal quantities for discrete goods | The planning output is copied without a rounding step | Supplier rejects or silently rounds; invoice mismatch | Round at the plan-to-order boundary and record the rule |
| Ceiling applied at every level of a cascade | Plan exceeds forecast by more than the largest single increment | Independent round-ups that do not cancel | Chronic overbuild; working capital; markdown or expiry | Round once at the binding level; carry exact fractions below it |
| Round-to-nearest on small requirements | Slow movers show zero orders followed by large ones | A requirement below half an increment rounds to zero | Chronic stockouts on the tail, invisible in aggregate service | Use ceiling or a MOQ floor for low-volume items |
| MOQ larger than demand per cycle | Days of supply rise as demand falls | Order quantity set by the MOQ, not by consumption | Excess inventory; expiry for perishable goods | Aggregate orders across items; renegotiate; reduce pack |
| Rounding residue larger than safety stock | Safety stock rises but service does not improve | Residue dominates the buffer and is replenished regardless | Buffer no longer covers variability; cost without service | Compare residue and safety stock explicitly; fix the lot, not the buffer |
| BOM exploded from the forecast, not the rounded parent | Component shortages at build time | Components bought for 1,200 while the line builds 1,250 | Line stoppage; expediting cost | Explode from the rounded parent quantity |
| Rounding factors added instead of multiplied | Reported overage is below the observed purchase value | Additive intuition applied to a multiplicative process | Understated cost; wrong policy choice | Compute cumulative factors as a product |
| Transport unit used as an item-level rounding value | One item's order equals a truck or pallet for months of demand | Vehicle constraint applied to a stock-keeping unit | Absurd lot sizes; obsolescence; storage overflow | Round consignments, not items, at the transport level |
| Horizon exception such as "last lot exact" | Most orders respect the rule; the final one does not | Horizon-level exception in the planning engine | Reconciliation breaks at the horizon boundary | Document the exception; test the final bucket explicitly |
| Residue never attributed | Overbuild is "normal" and no report shows its cause | The rounded quantity overwrites the exact one | No feedback, no learning, no owner | Persist exact, rounded, rule, and residue on every line |

Three of these deserve expansion.

**One field for two constraints** is the most common and the most expensive, because the error
is systematic and directional. It is also the easiest to detect: compare the distribution of
order quantities with that of requirements. If every order is a multiple of a round number while
the requirements are not, the field is being used as a grid.

**Residue larger than safety stock** most often produces a wrong remedy: the service problem is
real, so the obvious response is to raise safety stock, which worsens inventory without fixing
service because the buffer was never binding. Compare both numbers in the same unit first.

**The transport unit as an item-level increment** produces absurd results that are usually
caught, but the near-miss is common: a "full pallet" increment applied to an item whose weekly
demand is a fraction of a pallet. The order is not twenty years of demand but twenty weeks, and
nobody notices because the item is cheap. Test the exposure ratio, not the currency value.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The case pack is a fixed integer over the horizon | Rounding needs a constant increment | Pack changes, promotional packs, supplier substitution | Compare pack records by effective date |
| Demand is level enough for a single lot size | The worked example uses a constant rate | Seasonality, promotions, lumpy demand | Compute the coefficient of variation by item |
| The MOQ applies per order, not per period | Determines whether aggregation across items helps | Contractual minimum per quarter or per year | Read the clause; do not infer from the field |
| The order multiple applies after the MOQ | The arithmetic depends on the order of operations | Vendor-specific precedence, horizon exceptions | Reproduce the engine's output on a test item |
| Unit cost and holding rate are constant | Converts quantity to cash | Volume discounts, perishability, obsolescence | Segment the holding rate where material |
| Ordering cost is independent of lot size | EOQ and break-even arithmetic | Fixed costs that scale with quantity | Check whether receiving cost varies with order size |
| Lead time is zero or constant in the comparison | Isolates the rounding effect | Real lead times, variability, and review periods | Add lead time and re-simulate; the ranking may change |
| Bill-of-material usage rates are exact | Component requirements are computed from them | Scrap, yield loss, substitution | Compare planned and actual issues |
| Component rounding is independent across items | Simplifies the portfolio effect | A common parent rounding factor correlates them | Compute the correlation of residues across components |
| The integer plan and the continuous optimum use the same cost basis | The gap is meaningful | Costs that exist only in the integer plan, such as freight | Include the incremental costs in both |

Two limits are worth stating plainly.

**This topic treats the constraints as given.** It does not ask whether the pack or the MOQ is
well designed, beyond noting in section 9 that both are negotiable. The design question — what
pack size should exist for this item — belongs with the packaging and sourcing material.

**The arithmetic assumes the only source of indivisibility is the unit.** In practice a
production run has a minimum economic size independent of the pack, a truck has a minimum charge,
and a supplier has a minimum campaign. Each behaves like an MOQ with a different owner. The
methods apply to all of them, but the data model must keep them apart, for the same reason it
keeps the case pack apart from the order multiple.

## 12. Exercises

### Exercise 1 — Audit the three constraints (recall and application, 40 minutes)

Choose one item you can obtain data for. Identify its case pack, its order multiple or rounding
value, and its MOQ. Do not accept a single field as the answer to all three.

**Deliverable:** a table of constraint, value, unit, owner, and effective date; the order quantity
for a stated requirement under ceiling, floor, and round-half; one sentence naming the binding
constraint.
**Acceptance criteria:** three separate rows with units; the binding constraint named; at least
one constraint marked unknown, with the question that would resolve it.

### Exercise 2 — Rounding direction arithmetic (calculation, 45 minutes)

Item A has a weekly requirement of 1,450 units and a case pack of 500. Item B has a weekly
requirement of 180 units and a case pack of 500. For each item and each rule, compute the order
quantity, the residue in units, the residue as a percentage, and the annualised residue.

**Deliverable:** a six-row table, one row per item and rule, with all arithmetic shown.
**Acceptance criteria:** the sign of every residue stated; the round-half result for Item B
explained rather than reported; annualisation on orders per year, not on the requirement.

### Exercise 3 — Simulate three policies (calculation, 60 minutes)

Weekly demand is 1,000 units, the case pack is 300, the MOQ is 1,000, holding cost is €0.05 per
unit per week, and ordering cost is €150. Simulate twelve weeks under (A) lot-for-lot with the
MOQ floor rounded to the case, (B) a fixed lot of 2,400 and (C) a fixed lot of 4,000, both ordered
when opening stock cannot cover the week.

**Deliverable:** three week-by-week tables with opening, receipt, demand, and closing; average
closing inventory, holding, ordering, and total for each; a service comment for each.
**Acceptance criteria:** every closing balance reconciles; the lowest-cost policy is not
recommended without a service argument; at least one policy is shown to end a cycle at zero.

### Exercise 4 — Cascade the BOM (application, 50 minutes)

Finished-good demand is 1,000 units per week and its case pack is 120. One component has a usage
of 1.5 units per finished good, a case pack of 700 units, and an MOQ of 2,000 units.

**Deliverable:** the rounded finished-good quantity, the component requirement derived from it,
the component order, each factor, the compounded factor, the additive estimate, and the
difference in percentage points.
**Acceptance criteria:** the component requirement derived from the rounded parent; factors
multiplied rather than added; the MOQ and the pack checked separately to see which binds.

### Exercise 5 — Residue against safety stock (calculation, 35 minutes)

A component has a weekly requirement of 600 kg, a pack and MOQ of 2,000 kg, demand standard
deviation of 250 kg per week, a two-week lead time, and a service factor of 1.28.

**Deliverable:** safety stock, residue on one order, the ratio, and average cycle stock, all in
kilograms, plus one paragraph on which quantity governs the inventory held.
**Acceptance criteria:** safety stock uses the square root of lead time; residue distinguished
from cycle stock; the conclusion states what would have to change for the buffer to govern.

### Exercise 6 — Break-even and negotiation (design, 45 minutes)

An item has a weekly requirement of 300 units, a case pack of 250, and a supplier MOQ of 2,500.
Unit cost is €40, the holding rate is 25% per year, and a stockout costs an estimated €12 per
unit short. Find the largest MOQ you would accept before holding cost exceeds avoided stockout
cost.

**Deliverable:** the arithmetic, the break-even MOQ, and a negotiation position with a specific
ask.
**Acceptance criteria:** holding cost annualised; the stockout term probability-weighted rather
than certain; the ask expressed as a number the supplier can accept or reject.

### Exercise 7 — Classify a portfolio (design, 60 minutes)

Take twenty items with demand and pack data. Compute the rounding exposure ratio, lot size divided
by annual demand, using the case pack as the lot where no other data exists; sort the items and
propose a policy per quartile.

**Deliverable:** a table of twenty items with annual demand, lot size, ratio, unit cost, and
proposed policy; a sorted list; a policy per quartile.
**Acceptance criteria:** consistent units; at least one item with a ratio above 0.5 whose policy
is not "round up"; the recommendation distinguishes cheap rounding from a pack that should be
challenged.

### Exercise 8 — Report the integrality gap (design, 40 minutes)

Design the output format for a planning run that must show both the continuous optimum and the
executable plan.

**Deliverable:** a field list with types and units, a mock output for the worked example in
section 6, and the one-line summary a planner would read.
**Acceptance criteria:** both quantities labelled as executable or not; the gap in cost and
units; the rounding policy named; the summary cannot be mistaken for a plan.

### Exercise 9 — Review a data model (critique, 45 minutes)

A schema has one field, `min_order_qty`, on the item master, applied by the planning engine as a
rounding value. Propose a replacement.

**Deliverable:** revised fields with names, units, semantics, and owners; a migration note; a
test that would catch the current behaviour.
**Acceptance criteria:** at least three distinct fields; each semantic stated in one sentence;
effective dating addressed; the test reproduces the over-order between grid points.

### Exercise 10 — Diagnose the case (critique, 40 minutes)

A distributor reports inventory up 14% and fill rate down from 96% to 92%, blaming forecast error
and supplier delays. Order quantities cluster at 1,000, 2,000, and 3,000 units while weekly
requirements range from 250 to 1,600, and safety stock was raised twice.

**Deliverable:** a one-page diagnosis naming the most likely mechanism, two alternatives, and
two discriminating measurements.
**Acceptance criteria:** the mechanism stated in arithmetic rather than adjectives; one
alternative is a measurement artefact; the measurements can be run next week.

## 13. Self-check answers

### Exercise 2 arithmetic

Item A, requirement 1,450, pack 500. Item B, requirement 180, pack 500.

```text
Item A   annual demand 1,450 x 52 = 75,400 units
ceiling     1,450 / 500 = 2.90 -> 3 cases = 1,500; residue +50   (+3.45%)
floor       1,450 / 500 = 2.90 -> 2 cases = 1,000; residue -450  (-31.03%)
round-half  2.90 -> 3 cases = 1,500;              residue +50   (+3.45%)
annual      ceiling 52 x 50 = +2,600; floor 52 x 450 = -23,400 units

Item B   annual demand 180 x 52 = 9,360 units
ceiling     180 / 500 = 0.36 -> 1 case = 500;    residue +320  (+177.78%)
floor       0.36 -> 0 cases = 0;                 residue -180  (-100.00%)
round-half  0.36 -> 0 cases = 0;                 residue -180  (-100.00%)
annual      ceiling 52 x 320 = +16,640 units                   (+177.78%)
```

Item B is the case that matters. Round-half does not round to the nearest *case* in any useful
sense when the requirement is below half a case; it rounds the order to zero, which is
indistinguishable from a decision not to stock the item. A system that applies round-half
globally will silently stop ordering every slow mover whose weekly requirement is below half a
pack, and the resulting stockouts will be attributed to demand volatility.

> **Note:** the expected residue under ceiling is half an increment only if the fractional part
> is uniformly distributed. Item A's fractional part is 0.90, so the residue is 50, not 250.
> Never substitute the expected value for the observed one in an audit; use it only to size a
> policy before the data exists.

### Exercise 3 arithmetic

Policy A orders `ceil(1,000 / 300) = 4` cases, or 1,200, every week. Policy B orders 2,400 when
opening stock is below 1,000. Policy C orders 4,000 on the same rule.

The closing balances, week 1 to 12, are:

- A: 200, 400, 600, ... 2,400 — an arithmetic progression, sum 15,600.
- B: 1,400, 400, 1,800, 800, 2,200, 1,200, 200, 1,600, 600, 2,000, 1,000, 0 — sum 13,200.
- C: 3,000, 2,000, 1,000, 0 repeating — sum 18,000.

Policy A: average closing `15,600 / 12 = 1,300 units`; holding `1,300 × 0.05 = €65.00` per week,
or €780.00; ordering `12 × 150 = €1,800.00`; total **€2,580.00**. Policy B: average 1,100;
holding €660.00; ordering €750.00; total **€1,410.00**. Policy C: average 1,500; holding €900.00;
ordering €450.00; total **€1,350.00**.

Policy C is the cheapest and the worst: it ends every four-week cycle at exactly zero, so any
week above 1,000 units produces a stockout before the next receipt. Policy B costs €60 more and
also ends at zero, but its minimum position is 200 units. Policy A costs €1,230 more and
accumulates 2,400 units of unintended stock, 2.4 weeks of demand. Recommending C on cost alone
ignores service; recommending A ignores the drift. The defensible answer is B, with a note that a
reorder point above zero is required in practice.

### Exercise 4 arithmetic

```text
FG requirement          1,000 units
FG case pack              120 units
1,000 / 120 = 8.333   -> 9 cases = 1,080 units        factor 1.0800
component requirement   1,080 x 1.5 = 1,620 units
component case pack       700 units
MOQ                     2,000 units
1,620 / 700 = 2.314   -> 3 cases = 2,100 units        factor 1.2963
MOQ check               2,100 >= 2,000, so the MOQ does not bind

compounded factor       1.0800 x 1.2963 = 1.4000
additive estimate       1 + 0.0800 + 0.2963 = 1.3763
difference              2.37 percentage points
```

The purchased quantity is 2,100 units against a true requirement derived from the forecast of
1,500 units, which is 40% above requirement. An analyst who adds the two rounding percentages
reports 37.6% and understates the overage by 2.4 points, which on a component with a €30 unit
cost and 52 weekly orders is material.

### Exercise 5 arithmetic

```text
safety stock = 1.28 x 250 x sqrt(2) = 1.28 x 250 x 1.4142 = 452.5 kg
residue      = 2,000 - 600 = 1,400 kg on the first order
ratio        = 1,400 / 452.5 = 3.09
cycle stock  = 2,000 / 2 = 1,000 kg on average
```

The residue on a single order is 3.09 times the safety stock, and the average cycle stock is
more than twice it. The buffer does not govern the inventory the company holds; the lot size
does. To make the buffer govern, the order quantity would have to fall below roughly 900 kg,
which is impossible under a 2,000 kg pack and MOQ, so the only available levers are a smaller
pack, a renegotiated MOQ, or aggregation of this component's order with others that share the
pack.

### Design rubrics

**Exercise 1.** Strong separates the three constraints and marks one as unknown, since one is
usually undocumented; weak reports three values copied from one screen, which usually means one
field is doing three jobs.

**Exercise 6.** Strong computes the annual holding cost of the MOQ residue, states stockout cost
as `probability × units short × cost per unit`, and converts the comparison into a maximum
acceptable MOQ, noting that ordering less often lowers the annual residue even at a fixed MOQ.
Weak compares the residue cost with the unit price, a category error.

**Exercise 7.** Strong sorts by the rounding exposure ratio and finds that the expensive items
are not the expensive units; it proposes ceiling only where service requires it and challenges
the pack where the ratio exceeds about 0.5. Weak applies one rule to all twenty items.

**Exercise 8.** Strong labels the continuous number a bound and the integer number the plan,
names the policy, and gives the gap in cost and in units. Weak shows two unlabelled numbers.

**Exercise 9.** Strong has three fields with distinct semantics, effective dating, and a test
that reproduces the over-order. Weak renames the field and keeps the behaviour.

**Exercise 10.** Likely mechanism: a rounding field of 1,000 applied as a grid, so requirements
between grid points are rounded up and the excess is held, while items far below the grid are
replenished late and stock out. Raising safety stock then adds inventory without service, since
the buffer is not binding. Alternatives to test: a genuine demand shift and a changed fill-rate
definition. Discriminating measurements: the distribution of order quantities against
requirements, and fill rate by item class split at the grid points.

## 14. Interview questions

**1. "What is the difference between a case pack, an order multiple, and an MOQ?"**

Strong: the case pack is physical, the multiple a system grid, the MOQ a contractual floor, and a
1,200 requirement becomes 1,250, 2,000, and 1,200 respectively. Weak: calls all three "the
smallest amount you can order", the error being tested.

**2. "Why is a fractional plan a problem if the model is correct?"**

Strong: separates mathematical optimality from executability, calls the fractional answer a
valid lower bound, and reports bound, plan, and gap together. Weak: says the model is wrong,
confusing a representation limit with an error.

**3. "Which rounding rule would you choose, and why?"**

Strong: chooses per item class, notes that ceiling protects service and accumulates inventory
while floor does the reverse, and that round-half is unsafe below half an increment because it
rounds to zero. Weak: "round up to be safe", with no cost and no condition.

**4. "A requirement of 180 units meets a case pack of 500. What happens under each rule?"**

Strong: ceiling gives 500 (+178%), floor and round-half give zero, and the zero is the dangerous
case because the item stops being ordered. Weak: gives only the ceiling answer.

**5. "How does rounding compound through a bill of materials?"**

Strong: the rounded parent drives the component requirement, each component has its own factor,
the factors multiply, and an example shows the product exceeding the sum; the errors are
correlated through the shared parent, so they do not average out. Weak: says errors add up.

**6. "When is rounding free and when is it expensive?"**

Strong: names lot size divided by demand as the governing ratio, gives a case pack of 1 as the
free end and a 10,000-unit MOQ against 500 units a week as the expensive end, and adds that
holding cost and shelf life decide the currency consequence. Weak: says rounding is always
small, or always large.

**7. "You find a rounding residue larger than the safety stock. What do you do?"**

Strong: the buffer is not the binding quantity, so raising safety stock will not fix service;
the levers are the pack, the MOQ, and order aggregation, compared in the same unit. Weak:
recommends raising safety stock, the intuitive and wrong response.

**8. "How would you detect that an MOQ field is being used as an order multiple?"**

Strong: compares the distribution of order quantities with the distribution of requirements,
looks for clustering at multiples of a round number, and checks over-ordering between grid
points. Weak: reads the field name, which does not state the semantics.

**9. "A planner says the supplier demands orders of 2,000 when the need is 1,200. What do you
check?"**

Strong: checks whether 2,000 is an MOQ or a rounding value, reads the contract, and tests whether
1,250 would be accepted — a supplier asking for "minimum 2,000" states a threshold, which 1,250
satisfies. Weak: accepts the account and redesigns policy around a constraint that may not exist.

**10. "How would you reduce rounding cost across a portfolio without changing any supplier
contract?"**

Strong: aggregates orders across items that share a pack or supplier, orders less often where
shelf life allows, moves the rounding level from pallet to case, and challenges pack sizes for
the highest-ratio items, quantifying each with its service trade-off. Weak: orders more often,
which adds rounding events and can raise the annual residue.

## 15. Key terms

- **Case pack** — the number of base units in one shipping case.
- **Order multiple** — the increment defining legal order quantities; a grid.
- **Rounding value** — the enterprise-system field that rounds a quantity up to a multiple.
- **Rounding profile** — a scaled rule with thresholds at which the increment changes.
- **MOQ** — minimum order quantity; a threshold, not a grid.
- **Minimum lot size** — the smallest lot the planning system will propose.
- **Lot size** — the quantity produced or ordered in one lot.
- **Lot-for-lot** — ordering exactly the net requirement each period.
- **Fixed order quantity** — a policy of ordering the same quantity every time.
- **Fixed lot multiple** — a policy of ordering integer multiples of a stated quantity.
- **Economic order quantity (EOQ)** — the lot size minimising ordering plus holding cost under
  stated assumptions.
- **Net requirement** — gross requirement less available stock and scheduled receipts.
- **Integer constraint** — a requirement that a decision variable be a whole number.
- **Continuous relaxation** — the model with the integer constraint removed.
- **Integrality gap** — the ratio between the integer and relaxed optima.
- **Price of integrality** — the cost difference between the continuous optimum and the best
  executable plan.
- **Ceiling** — rounding up to the next multiple.
- **Floor** — rounding down to the previous multiple.
- **Round-half** — rounding to the nearest multiple with a stated tie rule.
- **Rounding residue** — the quantity added or removed by one rounding operation.
- **Compounded residue** — the multiplicative accumulation of several rounding operations.
- **Divisibility** — the property that a quantity can be split into whole units without loss.
- **Executable plan** — a plan whose quantities can be picked, packed, shipped, and invoiced.
- **Rounding exposure ratio** — lot size divided by demand over the same period.
- **Cycle stock** — inventory that exists because supply arrives in lots.
- **Safety stock** — inventory held against variability in demand or lead time.
- **Sawtooth** — the inventory profile produced by periodic replenishment in lots.

## 16. Sources

**Peer-reviewed and academic**

- Alnahhal, M., and colleagues (2024). "Economic Order Quantity: A State-of-the-Art in the Era of
  Uncertain Supply Chains." *Sustainability*, 16, 5965. Suppliers may enforce an MOQ much larger
  than the EOQ, and the EOQ calculation is pointless when the MOQ binds. Grade A.
- Bertazzi, L. (2003). "Rounding off the optimal solution of the economic lot size problem."
  *International Journal of Production Economics*, 81–82, 385–392. Rounding to discrete reorder
  intervals; power-of-two worst case near 1.06; naive rounding. Grade A.
- Holland, W., and Sodhi, M. S. (2004). "Quantifying the effect of batch size and order errors on
  the bullwhip effect using simulation." *International Journal of Logistics: Research and
  Applications*. Order variance rises with the square of the batch size. Grade A.
- Ketzenberg, M., Metters, R., and Vargas, V. (2002). "Quantifying the benefits of breaking bulk
  in retail operations." *International Journal of Production Economics*. Case-pack ordering
  against breaking bulk; 77-product simulation. Grade A.
- Kiesmüller, G. P., de Kok, A. G., and Dabia, S. (2011). "Single item inventory control under
  periodic review and a minimum order quantity." *International Journal of Production
  Economics*, 133(1), 280–285. The (R, S, Qmin) policy, mostly within 1% of the complex
  alternative. Grade A.
- Zhao, Y., and Katehakis, M. N. (2006). "On the structure of optimal ordering policies for
  stochastic inventory systems with minimum order quantity." *Probability in the Engineering and
  Informational Sciences*, 20, 257–270. Non-monotone optimal policy under an MOQ. Grade A.
- Potter, A., and Disney, S. M. (2010). "Removing bullwhip from the Tesco supply chain." POMS
  proceedings. Case-size multiples in automated store ordering; workload variability 185%;
  order variability reduced to about 75% of sales; saving of roughly £28 million a year. Grade C.

**Vendor and system documentation**

- SAP Learning, "Understanding the Lot Size Calculation"; SAP Help documentation on lot-for-lot
  order quantity; SAP support notes 3135343 and 3213460. Rounding value, rounding profile,
  minimum and maximum lot size, and the "last lot exact" horizon exception. Grade D, retrieved
  2026.
- Oracle, "Order Modifiers" (MRP and Supply Chain Planning Help) and "Item Attributes and Order
  Modifiers for Supply Planning" (Fusion Cloud Supply Planning 26b/26c); Oracle Inventory
  Optimization documentation on cycle stock. Fixed order quantity, fixed lot multiple, minimum
  and maximum order quantity, rounding, and the documented precedence order. Grade D, 2026.

**Standards and official**

- GS1, *GTIN Management Standard*, current version. A change in case quantity requires a new
  identifier at that level. Grade B.
- UN/CEFACT, Recommendation 20, *Codes for Units of Measure Used in International Trade*, and
  Recommendation 21 on packaging types. Grade B.
- GS1 EANCOM data element 6411; ANSI X12 850 and UN/EDIFACT ORDERS for purchase order quantity
  fields. Grade B.
- ISO 8000-110 on master-data quality, and FDA unique device identification requirements for
  medical devices, are named at the level of the standard; no clause numbers are asserted here.
  Grade B for the existence of the requirements.

**Company and thesis evidence**

- Mäkelä, K. (2022). "Optimizing the inventory system of a pharmaceutical company: A case
  study." Metropolia University of Applied Sciences thesis. ABC-XYZ analysis of 321 SKUs; MOQ
  among the identified causes of scrapped expired stock; all fifteen top expiry-cost batches in
  the volatile or low-sales classes. Grade C.

**Note on citations.** Vendor documentation is reported as the vendor states it and is not
independently verified; the field semantics may differ between releases. The Tesco and
pharmaceutical figures are company or thesis reported and are not base rates. Where a number
could not be confirmed from a retrieved source, the text gives the mechanism and omits the
number rather than quoting precision it cannot support.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_container-loading.md`](02_container-loading.md)
- Next topic: [`04_calendars-weeks-time-zones.md`](04_calendars-weeks-time-zones.md)
- Related: [`../01_supply_chain_foundations/07_manufacturing-basics.md`](../01_supply_chain_foundations/07_manufacturing-basics.md)
- Survey file: [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
- Return to the part map: [`../_index.md`](../_index.md)
