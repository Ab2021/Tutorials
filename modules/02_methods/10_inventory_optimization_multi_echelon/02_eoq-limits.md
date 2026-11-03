# EOQ and Its Limits — setup, holding, and the assumptions that rarely hold in practice

> Module 10 — Inventory Optimization and Multi-Echelon Planning
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 10 Topic 01 (the cost-service frontier)
> Version 2026.5

---

## 1. Why this topic exists

The economic order quantity (EOQ) is the oldest result in inventory theory and still the first
formula a planner is taught. It answers one question: given a steady demand for a single item,
what order quantity minimises the sum of the cost of ordering and the cost of holding? Its
answer is a square root. Its derivation fits on half a page. Its assumptions are so strong that
almost no real item satisfies all of them.

It survives anyway, and understanding why is more useful than memorising it. Two properties keep
it alive. First, the total cost curve around the optimum is extraordinarily flat: an order
quantity wrong by a factor of two costs about 25 per cent of the controllable cost, and an input
wrong by 10 per cent moves the answer by about 5 per cent. Second, it produces a defensible
number in about a minute, which is often all a meeting allows. Those two properties mean the
formula is usually approximately right in a domain where "approximately right" is genuinely
useful.

The damage comes from somewhere else. It comes from treating the output as a policy, from
treating the ordering cost as a measurement rather than an allocation, and from applying a
stationary-demand model to items whose demand is lumpy, seasonal, or intermittent. A planner who
computes an EOQ of 973 units for a part that sells 24,000 units a year has done something
harmless. A planner who reports that number to three significant figures, in a system that
orders in pallets of 480, on an item with a 60-day shelf life and a supplier minimum of 5,000,
has produced a number that is wrong in three independent ways and cannot be acted on.

**Who is blocked by this.** Almost every inventory analytics project begins with a request to
"set the order quantities". The honest answer is usually that order quantity is the least
important of the four variables in play:

| Variable | What it controls | Typical influence on cost and service |
|---|---|---|
| Order quantity | Cycle stock and order frequency | Modest; flat basin around the optimum |
| Reorder point | When replenishment starts | Large; sits in the tail of a distribution |
| Safety stock | Protection against variability | Large; directly trades cost against service |
| Review and replenishment design | Who decides, how often, on what signal | Largest; determines whether any of the above is stable |

EOQ addresses the first row. Topics 04 and 06 of this module address the second, third, and
fourth. This topic exists so that you can use the first row correctly, and so that you can say
precisely what it does not answer when someone asks you to "just run the EOQ".

**The relationship to Topic 01.** The cost-service frontier is the general statement: every
additional point of service has a marginal cost, and the curve is convex. EOQ is the smallest
possible instance of that idea. It has exactly two cost terms, one decision variable, and no
service dimension at all. Studying it in isolation is useful precisely because it is so small:
you can see the whole trade-off, and you can see exactly which part of the real problem has been
left outside the model.

**The relationship to the rest of the module.** Topics 03 through 10 replace parts of this model
one at a time. The newsvendor replaces constant demand with a distribution over a single season.
Safety stock replaces the missing variability term. Policy families replace "how much" with a
joint rule for how much and when. Multi-echelon replaces the single-item assumption with a
network. Each replacement is a relaxation of one line in the assumption ledger in section 11.
The ledger is therefore the most important page in this file.

## 2. Learning outcomes

By the end of this topic you can:

1. **Derive** the EOQ formula from the two cost terms, and **explain** in one paragraph why the
   answer contains a square root rather than a linear or cubic relationship.
2. **Compute** an EOQ by hand, state every input with its unit and its source, and **verify** the
   result using the equal-cost identity rather than by re-doing the arithmetic.
3. **Quantify** the cost penalty of a quantity that is not optimal, using the penalty ratio, and
   **state** the penalty at half and at double the optimum without a calculator.
4. **Evaluate** an all-units quantity discount schedule and an incremental discount schedule, and
   **explain** why the optimum may sit at a break point rather than where the derivative is zero.
5. **Diagnose** which of the twelve assumptions in section 11 a given application violates, and
   **predict** the direction in which the resulting order quantity is biased.
6. **Distinguish** an order quantity from a reorder point, and **explain** why a system that
   reports only the first is not an inventory policy.
7. **Critique** an ordering cost figure, identifying whether it is a marginal cost, an allocated
   overhead, or a managerial choice, and **describe** the behaviour each produces.
8. **Decide** when the economic lot size idea applies at all — repetitive manufacturing,
   container or pallet quantities, supplier minimums — and when it should be replaced by a
   different method.

Each outcome maps to an exercise in section 12 and an answer in section 13.

## 3. The core mental model

Hold one picture: two cost lines crossing, and a total cost curve whose bottom is almost flat.

```text
  annual
  cost
  (EUR/yr)
     ^
     |   \                                    /
     |    \                                  /   holding cost  H x Q/2
     |     \                                /    (rises with Q)
     |      \                              /
     |       \                            /
     |        \                          /
     |         \                        /
     |   ordering\                    /
     |   cost     \                  /
     |   D x S / Q \                /
     |   (falls as  \              /
     |    1/Q)       \            /
     |                \          /
     |                 \        /
     |                  \      /
     |                   \    /
     |                    \  /
     |                     \/                total cost  TC(Q)
     |                  .-''\__          .-'
     |              .-''       \______.-'      <- flat basin
     +----|--------|-------|--------------|---------> Q  (units per order)
         0.5Q*   0.75Q*    Q*           1.5Q*      2.0Q*
          |<---------- +-50% ---------->|
          |        |       |              |         |
        +25.0%   +4.2%    0.0%          +8.3%     +25.0%   cost penalty
```

Read the diagram in four passes.

**Pass 1 — the two lines.** Ordering cost per year is annual demand divided by order quantity,
multiplied by the cost of one order: `D x S / Q`. Each order is bigger, so you place fewer of
them. The line falls as 1/Q. Holding cost per year is average inventory multiplied by the cost
of holding one unit for a year. Under constant demand the average inventory is exactly half the
order quantity, so the line is `H x Q / 2`, which rises linearly. One line falls, one rises, and
the sum has a single minimum.

**Pass 2 — why the square root.** At the minimum the two lines are equal, because if they were
not equal you could move Q slightly toward the cheaper side and reduce the total. Set them equal:

```text
  D x S / Q  =  H x Q / 2
  2 x D x S  =  H x Q^2
  Q^2        =  2 x D x S / H
  Q*         =  sqrt( 2 x D x S / H )
```

That is the whole derivation. The square root appears because one term scales as Q and the other
as 1/Q; their product is constant, so the balance point is the geometric mean of the two
"natural" quantities, `2DS` and `H`. A useful dimensional check: `D` is units per year, `S` is
currency per order, `H` is currency per unit per year. The ratio `D x S / H` has units of
units squared, so its square root is in units. If your formula does not yield units, your inputs
are on inconsistent time bases.

**Pass 3 — the flatness.** Substitute `Q = k x Q*` into the total cost and divide by the cost at
the optimum. Everything cancels except `k`:

```text
  TC(k Q*) / TC(Q*)  =  (1/2) x ( k + 1/k )
```

This is the single most important equation in the topic, because it is independent of demand,
cost, and currency. It says that being wrong about the quantity costs the same in every business
in the world. At `k = 2` the penalty is 25 per cent. At `k = 0.5` the penalty is also 25 per cent.
At `k = 1.25` it is 2.5 per cent; at `k = 0.75` it is 4.2 per cent. Note the asymmetry: a 25 per
cent undershoot costs the same as a 33 per cent overshoot, because the curve is convex and
steeper on the left. Undershooting is the more expensive direction, which is why the rounding
convention in most planning systems is to round up.

**Pass 4 — what the picture does not contain.** There is no service level in this diagram. There
is no stock-out, no lead time, no variability, no shelf life, no pallet, no truck, no supplier
minimum, no capacity. The diagram is a statement about cost only, for one item, in a world where
demand arrives as a smooth flow. Every one of those missing features is a section of this file.

### 3.1 The second picture: quantity and timing are different objects

The most common confusion in practice is between the order quantity and the reorder point. They
are separate decisions, computed by separate models, and stored in separate fields.

```text
  inventory
  position
     ^     /\        /\        /\        <- each peak is Q
     |    /  \      /  \      /  \
     |   /    \    /    \    /    \
     |  /      \  /      \  /      \
     | /        \/        \/        \
     +------------------------------------> time
     |           \         \         \
     | ROP -------\---------\---------\---  order placed when
     |             \         \         \    position hits ROP
     +------------------------------------>
        |<- lead ->|                        time
        |<------- order cycle Q/D ------->|
```

The sawtooth has two parameters, and they answer two questions:

- **Q (the height of each tooth)** answers "how much". EOQ, or a discount analysis, or a pallet
  multiple, sets it.
- **ROP (the level at which the tooth triggers a new order)** answers "when". It is expected
  demand during the lead time, plus a safety stock that depends on the variability of demand and
  lead time and on the service target.

A system configured with Q but no ROP will order at arbitrary times. A system configured with a
ROP but a Q inherited from a legacy default will order at the right time in the wrong quantity.
Neither is a policy. The pairing is called an `(r, Q)` policy: order Q whenever the inventory
position falls to r. Topic 06 covers the full family. Here the point is narrower: **EOQ is half
of a policy and the less consequential half.**

### 3.2 The third idea: the answer is a heuristic, not a measurement

An EOQ of 973.3 units is not a fact about the world in the way a weight is. It is the output of
four inputs, at least two of which are estimates and one of which is often a managerial
allocation.

The discipline: whenever you present a lot size, present three numbers — the computed optimum,
the quantity the operation can execute, and the cost penalty of the difference. If the penalty is
under 1 per cent, say so and stop the analysis. Most of the time it is.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Economic order quantity (EOQ) | The order quantity that minimises ordering plus holding cost for one item under constant demand | Used as the name of a policy rather than of a quantity | Units per order |
| Order quantity (Q) | Units bought or produced in one replenishment | Confused with the reorder point | Units per order |
| Optimal order quantity (Q*) | The minimiser of the modelled total cost | Reported to four significant figures on estimated inputs | Units per order |
| Annual demand rate (D) | Units consumed or sold per year | Replaced with shipments, or with a forecast | Units per year |
| Ordering cost (S) | The incremental cost of placing and receiving one order | Set equal to an allocated overhead rate per order line | Currency per order |
| Setup cost | The incremental cost of one changeover in production | Treated as a fixed technological constant | Currency per changeover |
| Holding cost (H) | The cost of keeping one unit in stock for one year | Quoted only as a percentage, with no components | Currency per unit per year |
| Holding rate (i or h) | Holding cost expressed as a fraction of unit cost per year | Assumed uniform across a product family | Fraction per year |
| Unit cost (C) | Purchase price or standard cost of one unit | Confused with landed cost | Currency per unit |
| Landed cost | Unit cost including freight, duty, insurance, and fees | Reported without duty | Currency per unit |
| Delivered duty paid (DDP) | Incoterm under which the seller delivers cleared for import | Treated as a synonym for landed cost | Incoterm code plus place |
| Total variable cost (TC) | Ordering plus holding cost per year, excluding purchase spend | Reported including purchase spend, which dominates everything | Currency per year |
| Purchase spend (D x C) | Annual value of the units bought | Omitted when comparing discount options | Currency per year |
| Cost penalty ratio | `TC(Q) / TC(Q*)` for any quantity Q | Reported as a currency saving without stating the baseline | Dimensionless |
| Flat basin | The region around Q* where large quantity errors cost little | Misread as "the inputs do not matter" | Quantity range |
| Cycle stock | Inventory that exists because orders are batched | Called excess stock, or confused with safety stock | Units |
| Average cycle stock | Q/2 under constant demand and instantaneous receipt | Applied unchanged to lumpy demand | Units |
| Order cycle length | `Q / D`, expressed in days or years | Confused with lead time | Days |
| Order frequency | `D / Q` orders per year | Confused with review frequency | Orders per year |
| Time supply | Order quantity divided by average daily demand | Quoted without stating the demand basis | Days of cover |
| Replenishment lead time (L) | Time from order release to availability for use | Used as if it equalled transit time | Days |
| Reorder point (ROP) | Inventory position at which a replenishment is triggered | Computed as demand times lead time, omitting safety stock | Units |
| Safety stock | Buffer held against variability in demand or lead time | Set by rule of thumb and attributed to the EOQ model | Units |
| Inventory position | On-hand plus on-order minus backorders | Replaced by on-hand, which causes duplicate orders | Units |
| Sawtooth diagram | The triangular inventory pattern under fixed-quantity ordering | Drawn as smooth when receipts are gradual | Diagram |
| All-units discount | One unit price applies to the whole order once a break point is reached | Modelled as if the discount applied only to marginal units | Price schedule |
| Incremental discount | The lower price applies only to units above each break point | Assumed to be the common supplier format | Price schedule |
| Minimum order quantity (MOQ) | The smallest quantity a supplier will accept | Treated as a target rather than a floor | Units |
| Economic production quantity (EPQ) | The lot size minimising setup and holding cost when output is gradual | Used interchangeably with EOQ | Units per production run |
| Changeover | Work between the last good unit of one product and the first good unit of the next | Counted only as machine time | Minutes |
| Dynamic lot sizing | Lot sizing where demand varies by period, as in Wagner–Whitin and successors | Applied to stationary demand where EOQ is adequate | Units per period |
| Stock-keeping unit (SKU) | A unique item at a unique location | Used to mean "product" | Identifier |
| Parameter elasticity | Percentage change in Q* per percentage change in one input | Assumed to be one | Dimensionless |

Three pairs in that table do most of the damage when confused.

**Ordering cost versus setup cost.** These are the same symbol in the formula and different
objects in the world. Ordering cost is the administrative and transactional cost of a purchase
order: buyer time, approval, transmission, receiving, inspection, invoice matching. Setup cost
is the cost of a changeover: lost production time, scrap at start-up, labour, consumables. They
share a mathematical role — a fixed charge per replenishment — and nothing else. A purchasing
organisation with an automated three-way match has an ordering cost near zero, while a plant with
a four-hour changeover has a setup cost in the thousands. Using one organisation's number for the
other is a common and expensive shorthand.

**Holding cost versus holding rate.** `H` is currency per unit per year. `i` is a fraction of unit
cost per year. They are related by `H = i x C`. The distinction matters because `C` changes when
a discount is taken, so `H` changes too, and because a percentage is not auditable while a list
of components is.

**Cycle stock versus safety stock.** Cycle stock exists because you ordered in a batch. Safety
stock exists because the future is uncertain. Cycle stock scales with Q; safety stock scales with
the standard deviation of lead-time demand and the service target. A company that reduces Q
reduces cycle stock and leaves safety stock untouched. A company that reports "inventory
reduction" after a lot-size change without decomposing the two is reporting half a result.

## 5. The mechanics, step by step

This section walks the calculation from raw inputs to an executable quantity. The purpose is not
to reach the formula — that takes three lines — but to make visible the eight judgement calls
that sit between a data table and a number someone will act on.

### Phase A — assembling the four inputs

**Step 1. Establish annual demand from demand, not from orders.** Use consumption or sales at
item-location level, at least 24 months where available, and add back an estimate of unmet demand
where stock-outs occurred. `D` computed from shipments equals sales, so slow movers get
quantities that perpetuate their own stock-outs. State the observation window and any
de-seasonalisation.

**Step 2. Separate the trend from the level.** EOQ assumes a constant rate. If the item is growing
30 per cent a year, the annual average is stale by the time the policy loads. Recompute quarterly,
or use the forecast rate, and say which you used.

**Step 3. Build the ordering cost from components.** Time each activity that occurs once per
order and not once per unit: requisition, approval, PO creation and transmission, supplier
confirmation, receiving and put-away, inspection, invoice matching. Multiply by a loaded labour
rate and add any fixed freight or transaction charge, giving `S` in currency per order. Without
this, `S` becomes whatever the finance allocation says, which is usually a percentage of spend and
therefore correlated with the quantity you are trying to choose.

**Step 4. Build the holding cost from components.** Sum cost of capital, storage, insurance and
tax, obsolescence and shrinkage, and handling into `H` in currency per unit per year, or into `i`
as a fraction of unit cost; state which, and recompute `H` whenever `C` changes. A single corporate
rate applied to a perishable fashion item and to a steel fastener is wrong in both directions,
because the true rates differ by an order of magnitude.

> **Note:** There is no universal holding rate. Figures in the high teens to low thirties per
> cent per year are commonly quoted, but the spread within a single company's portfolio usually
> exceeds the spread between companies. Build the number; do not import it.

**Step 5. Decide whether unit cost is price or landed cost.** If `H` derives from `C`, both must
use the same cost basis. An ex-works price with a holding rate that includes in-transit inventory
understates `H` for imports. Under delivered duty paid (DDP) the seller's price already contains
freight and duty; under free carrier (FCA) it does not, and the difference can exceed 20 per cent.

### Phase B — deriving and checking the answer

**Step 6. Write the total cost function and name every term.**

```text
  TC(Q) = D x S / Q  +  H x Q / 2  +  D x C
          \_________/    \_______/    \___/
           ordering      holding     purchase
           (falls)       (rises)     (constant in Q)
```

The third term does not depend on `Q`, so it drops out of the derivative. It is nevertheless the
largest term by far — typically 95 to 99 per cent of the total — and this is why lot-size
decisions rarely move a profit and loss statement as much as their advocates claim.

**Step 7. Differentiate, set to zero, and confirm a minimum.**

```text
  dTC/dQ = -D x S / Q^2 + H / 2 = 0
  Q*     = sqrt( 2 x D x S / H )
  d2TC/dQ2 = 2 x D x S / Q^3 > 0   for all Q > 0
```

The second derivative is positive everywhere, so the function is strictly convex and the
stationary point is the unique global minimum. There is no boundary case in the unconstrained
problem. Every boundary case in practice comes from a constraint the model does not contain —
a minimum, a maximum, a break point, or a package size.

**Step 8. Verify with the equal-cost identity rather than by repeating the arithmetic.**

At the optimum, ordering cost equals holding cost:

```text
  D x S / Q*  =  H x Q* / 2  =  sqrt( D x S x H / 2 )
  TC(Q*)      =  sqrt( 2 x D x S x H )
```

If your computed ordering cost and holding cost differ by more than a rounding error, you have
made an arithmetic mistake, used inconsistent time bases, or used a different `H` in the two
terms. This check catches more errors than any other single step, and it takes five seconds.

### Phase C — testing the answer against the world

**Step 9. Convert the quantity into days of cover.** `Q / D x 365`. This is the number that
reveals whether the answer is usable. A quantity that represents 90 days of supply on an item
with a 60-day shelf life is not an order quantity; it is a write-off schedule. Compare the cover
against four things: the lead time, the review period, the shelf life, and the supplier's
production cadence.

**Step 10. Round to a handling unit, and price the rounding.** Take the pallet quantity, case
pack, container capacity, and inner-pack multiple; round to the nearest feasible multiple in the
direction the operation can execute; recompute the penalty ratio. Output the executable quantity
and its penalty in per cent and in currency per year. Skip it and you get a business case built on
973 units and an operation that orders 960, with nobody able to reconcile the two.

**Step 11. Check the transport and capacity constraints.** Does the quantity fill a handling unit
sensibly, since half a pallet costs almost as much to move as a full one? Does the implied order
frequency fit the dock, with `D/Q` orders per year times receiving hours per order inside
available hours? For production, does the run length fit the window between changeovers?

**Step 12. Evaluate any discount schedule in full, including the purchase cost.** This is the one
step where the purchase term cannot be dropped, because it changes between alternatives. The
procedure is in section 5.2.

**Step 13. State what the calculation did not answer.** The quantity is now known. The reorder
point, the safety stock, the review frequency, and the service level are not. Write that down.
An EOQ delivered without that sentence will be read as a complete policy.

### Phase D — sensitivity and documentation

**Step 14. Run the ±50 per cent quantity test.** Compute the total variable cost at `0.5Q*`,
`0.75Q*`, `1.25Q*`, and `1.5Q*`. Report the penalties. If the maximum penalty in that range is
under about 5 per cent, the quantity decision is not worth further analytical effort, and you
should say so.

**Step 15. Run the parameter error test.** Because `Q*` scales as the square root of `D` and `S`
and as the inverse square root of `H`, a parameter error of `e` produces a quantity error of
approximately `e/2` for small `e`. The cost penalty is second order in the quantity error, so a
10 per cent parameter error produces roughly 0.25 per cent cost penalty; a factor-of-four error
produces a factor-of-two quantity error and a 25 per cent penalty. This is the fourth-root result
in plain language: cost is remarkably insensitive to parameter uncertainty.

**Step 16. Document provenance.** For each of `D`, `S`, `H`, `C`, record the value, the unit, the
source, the date, and the owner. An EOQ without provenance is a number that cannot be defended
when the buyer, the controller, and the supplier each arrive with a different one.

### 5.1 What each assumption failure does to the answer, one at a time

The value of the assumption ledger is that failures have directions. This table takes each
assumption, breaks it alone, and states the effect on the recommended quantity.

| Assumption broken | Model that handles it | Effect on the recommended quantity | What actually goes wrong |
|---|---|---|---|
| Demand is constant | Dynamic lot sizing; newsvendor | Quantity is right on average and wrong in every period; over-orders in troughs and under-orders in peaks | Service fails in the peak, cash is trapped in the trough |
| Replenishment is instantaneous | EPQ with finite production rate | EPQ is larger by `1 / sqrt(1 - d/p)`, where d is demand rate and p is production rate | A single EOQ lot cannot be produced in one block; the run is interrupted |
| Lead time is known and constant | Stochastic lead-time models | Q is barely affected; the reorder point is badly affected | Stock-outs appear even though the quantity was "optimal" |
| No quantity discounts | Discount lot sizing (section 5.2) | The optimum moves to a break point, often several times the EOQ | Purchase savings dwarf the holding cost change |
| No shortages permitted | Backorder models | Allowing planned backorders increases Q by `sqrt((H+B)/B)` where B is the backorder cost | The firm is leaving a cheaper mix of stock-out and holding on the table, or is pretending it never stocks out |
| Unit cost is fixed | Price-dependent cost models | If price falls with Q, both Q and H move; H falls, which reinforces the larger quantity | Treating a discount as a pure purchase saving and forgetting that holding cost per unit also falls |
| Horizon is infinite | Finite-horizon lot sizing | The final order before a demand end should be smaller than the EOQ | Leftover stock at the end of a season, a contract, or a product life |
| No capacity limit | Capacitated lot sizing | If capacity binds, Q is capped and the cost is the constrained optimum | The plan is infeasible; the schedule is levelled by expediting or overtime |
| Ordering cost is a known marginal constant | Activity-based or engineered cost | Q moves as the square root of S, so an inflated S inflates Q | Larger lots justified by a cost that exists only in the allocation |
| Holding cost is proportional to Q | Stepwise or non-linear holding cost | With warehouse space priced in blocks, the cost curve becomes a staircase and has several local minima | A "wrong" quantity that fits the existing space can beat the theoretical optimum |
| Items are independent | Joint replenishment models | Ordering several items together lowers the effective S per item and raises the optimal frequency | Missed consolidation; more inbound freight and receiving events than necessary |
| Demand and costs are stationary | Rolling re-estimation | Parameters drift; a policy not refreshed becomes progressively wrong in one direction | The policy is quietly abandoned rather than revised |

One row deserves expansion, because it is the most counter-intuitive. **Finite lead time damages
the timing, not the quantity.** Lead time does not appear in the quantity derivation at all: with
constant demand and a known lead time you order the same quantity and place the order earlier.
Once either is variable, the quantity remains approximately right and the timing becomes the
source of all service failure. That is why Topic 04 matters more than this one for service
outcomes, and why an organisation that has "done EOQ" has not addressed service at all. A second
point follows: a discount changes `H` as well as `C`, so updating the purchase price without
updating the holding rate always biases against the discount.

### 5.2 Quantity discounts: all-units versus incremental

A supplier price schedule is not one function. It is a set of tiers, and the arithmetic depends
on whether the discount applies to every unit or only to the units above the break.

**All-units (also called all-quantity).** Once the order reaches the break, every unit is priced
at the lower tier. The purchase cost function jumps down at each break. This is the more common
commercial format, and the one that produces large discontinuities.

**Incremental (also called marginal or discounted).** The first block of units is priced at the
first tier, the next block at the second, and so on. The purchase cost function is continuous and
piecewise linear. The incentive to jump is much weaker, because the saving applies only to the
marginal units.

The all-units procedure is a four-step search, not a derivative.

1. For each price tier `k`, compute `Q_k* = sqrt(2 x D x S / H_k)` with `H_k = i x C_k`.
2. Discard any `Q_k*` that does not fall inside its own tier's quantity range. It is not a
   feasible candidate.
3. The candidate set is: every feasible `Q_k*`, plus every break point at or above the lowest
   feasible quantity. The optimum is at a break point whenever the feasible interior minimum in
   a higher tier is infeasible, which is the normal case.
4. Evaluate the full annual cost, including `D x C_k`, at every candidate, and take the minimum.

The reason the optimum can sit at a break point is that the cost function is discontinuous. A
derivative finds stationary points of a continuous function. At a break point the function jumps
down, and the left-hand limit and the right-hand value differ. The minimum can therefore be at a
point where the derivative is undefined, and no amount of differentiation will find it.

The practical implication is worth stating plainly: **in a discount environment, the EOQ is
usually the wrong answer, and it is the wrong answer in the direction of ordering too little.**
The EOQ computed in the top tier is frequently infeasible, and the correct quantity is the next
break point up. Section 6 works this through with numbers, and the answer there is a quantity
5.1 times the EOQ.

### 5.3 The ordering cost problem

`S` is the most fragile input in the formula, and it is the one that most often becomes a
management instrument rather than a measurement. Three distinct things are called "ordering
cost" in practice.

| What it is | How it is derived | Behaviour it produces |
|---|---|---|
| Engineered marginal cost | Time study of the activities that occur once per order | Stable, auditable, appropriate for the formula |
| Allocated overhead | Total procurement department cost divided by number of purchase orders | Correlated with order frequency, hence with Q; creates a feedback loop |
| Managerial parameter | Chosen to produce a desired lot size, container fill, or changeover frequency | Whatever the chooser wants; the formula becomes a justification device |

The allocated-overhead case is the most common in large organisations, and it has a perverse
property. If `S` is total procurement cost divided by order count, then ordering less often
raises `S` per order — because the same department cost is spread over fewer orders. Feeding that
higher `S` back into the formula produces a still larger recommended quantity, which further
reduces order count, which raises `S` again. The loop does not converge to an economic optimum;
it converges to whatever the department's fixed cost is divided by the smallest order count the
organisation will tolerate. This is a real dynamical failure, not a theoretical curiosity, and
it is one of the strongest arguments for using an engineered `S` and treating departmental
overhead as a fixed cost outside the model.

The opposite perversion is deliberate. A planner who wants larger batches — to reduce
changeovers, to fill containers, or to hit a supplier's discount tier — can inflate `S` until the
formula recommends the desired quantity. This is sometimes done honestly and transparently ("we
set S to reflect the true cost of a changeover including lost capacity, which the accounting
system does not capture") and sometimes not. Your job as an analyst is to notice when `S` has
been reverse-engineered from a desired answer, because the resulting quantity then carries no
independent information.

> **Note:** There is a defensible case for a large `S` in a capacity-constrained plant. If a
> changeover consumes scarce capacity, its true cost includes the contribution margin of the
> production displaced, which accounting systems rarely capture. Topic 11 treats the capacity
> economics; here the point is that a large `S` must be justified in components, not asserted.

### 5.4 What EOQ is not

Four negative definitions, each preventing a specific mistake. It is not a policy: a policy
specifies quantity and timing together, plus the response to a stock-out and the review cadence,
and EOQ supplies one number. It is not a service model: it contains no service level, so two
items with identical EOQ can have entirely different service outcomes. It is not a forecast: it
summarises history and degrades quietly when the rate changes. It is not a total cost model: it
omits purchase spend, transport, quality, obsolescence, supplier risk, and the capital that funds
the cycle. The dishonesty is not in the two terms; it is in presenting a two-term result as a
total.

## 6. Worked example

An industrial distributor stocks a stainless steel ball valve, which we will call SKU 4471. The
item is a steady seller: it is not seasonal, it is not promoted, and it has sold between 1,800
and 2,200 units a month for three years. The distributor buys from a single European
manufacturer on 30-day terms, and the manufacturer has just issued a new price schedule.

All figures are illustrative but internally consistent, in euros (EUR), and use a one-year time
base.

### 6.1 The inputs

| Input | Value | Unit | Source | Note |
|---|---:|---|---|---|
| Annual demand `D` | 24,000 | units/yr | Three-year sales, no stock-outs recorded | 2,000/month average |
| Ordering cost `S` | 180.00 | EUR/order | Engineered: 2.5 h at 52 EUR/h plus 50 EUR freight fixed charge | Excludes departmental overhead |
| Unit cost `C` | 38.00 | EUR/unit | Current invoice, ex-works | Freight is in `S`, so `C` is consistent |
| Holding rate `i` | 0.24 | fraction/yr | 9% capital, 6% storage and handling, 4% insurance and tax, 5% obsolescence and shrinkage | Built from components |
| Holding cost `H` | 9.12 | EUR/unit/yr | `i x C = 0.24 x 38.00` | Recomputed if `C` changes |
| Lead time `L` | 21 | days | Supplier confirmation to dock | Constant in this example |

The two contested inputs deserve a note. `S` is engineered, not allocated: 2.5 hours of buyer,
receiving, and invoice-matching time plus the fixed component of inbound freight, with
departmental overhead excluded because it does not change when one more order is placed. `i` is
built from four components; the 5 per cent obsolescence figure is the most contestable, since the
item has no shelf life but the assortment turns over. It is a judgement, and it is stated as one.

### 6.2 The calculation

```text
  Q* = sqrt( 2 x D x S / H )
     = sqrt( 2 x 24,000 x 180.00 / 9.12 )
     = sqrt( 8,640,000 / 9.12 )
     = sqrt( 947,368.42 )
     = 973.33 units
```

Check with the equal-cost identity:

```text
  ordering cost = D x S / Q* = 24,000 x 180 / 973.33 = 4,438.36 EUR/yr
  holding cost  = H x Q* / 2 = 9.12 x 486.67         = 4,438.36 EUR/yr
  total variable cost                                = 8,876.71 EUR/yr
```

The two are equal to the cent, so the arithmetic is right.

Now translate the quantity into operating terms:

```text
  orders per year   = D / Q* = 24,000 / 973.33 = 24.66
  order cycle       = 365 / 24.66             = 14.8 days
  days of cover     = 973.33 / 65.75 per day  = 14.8 days
  average cycle stock = Q*/2                  = 486.67 units
```

Fourteen point eight days of cover against a 21-day lead time is a warning sign, not an error:
with a 21-day lead time the next order is placed before the previous one arrives, so there are
routinely two orders in the pipeline. That is exactly the situation in which the reorder point
does the work and the quantity does not.

### 6.3 The stakes, stated honestly

The total annual cost of the item, including purchase spend:

```text
  purchase spend  = 24,000 x 38.00 = 912,000.00 EUR/yr
  variable cost   =                   8,876.71 EUR/yr
  total           =                 920,876.71 EUR/yr
  variable share  = 8,876.71 / 920,876.71 = 0.96%
```

The quantity decision governs less than one per cent of the cost of this item. Suppose the
distributor currently orders monthly, 2,000 units at a time. What does that cost?

```text
  at Q = 2,000:  ordering = 24,000 x 180 / 2,000 = 2,160.00
                 holding  = 9.12 x 1,000         = 9,120.00
                 variable total                  = 11,280.00
  at Q* = 973:   variable total                  =  8,876.71
  annual saving from moving to the EOQ          =  2,403.29
  saving as a share of purchase spend           =  0.26%
```

Two thousand four hundred euros a year on a 912,000 euro item: this is the honest scale of the
lot-size decision for a single SKU with well-behaved demand. It is worth capturing across a
portfolio of thousands of items, not worth a six-month programme for any one of them, and
certainly not worth arguing about whether the answer is 960 or 973.

### 6.4 The flat basin, in numbers

The penalty ratio removes all the parameters, so the following table applies to every item in
this file and to every item in your company:

| `Q / Q*` | Total variable cost (EUR/yr) | Penalty versus optimum | Interpretation |
|---:|---:|---:|---|
| 0.50 | 11,095.89 | +25.00% | Ordering half as often as you should is expensive |
| 0.64 | 9,775.47 | +10.12% | The edge of the practically flat region |
| 0.75 | 9,246.57 | +4.17% | Barely worth correcting |
| 0.90 | 8,926.06 | +0.56% | Noise |
| 1.00 | 8,876.71 | 0.00% | Optimum |
| 1.10 | 8,917.28 | +0.46% | Noise |
| 1.25 | 9,098.61 | +2.50% | Still small |
| 1.50 | 9,616.50 | +8.33% | Starting to matter |
| 1.56 | 9,769.00 | +10.05% | The other edge of the flat region |
| 2.00 | 11,095.89 | +25.00% | Doubling is as bad as halving |

Read the table as a licence and as a warning at the same time. As a licence: any quantity
between roughly 0.75 and 1.35 of the optimum costs under 5 per cent, so there is no reason to
defend the decimal places. As a warning: the curve turns up sharply beyond a factor of two, and
the asymmetry means that under-ordering is punished at lower percentage error than over-ordering.
A 36 per cent undershoot costs 10 per cent; a 56 per cent overshoot costs the same.

### 6.5 The discount schedule changes everything

The manufacturer's new schedule is all-units:

| Tier | Quantity range (units) | Unit price (EUR) |
|---:|---|---:|
| 1 | 1 – 999 | 38.00 |
| 2 | 1,000 – 4,999 | 36.50 |
| 3 | 5,000 and above | 35.20 |

Because the unit price changes, `H` changes with the tier:

```text
  H1 = 0.24 x 38.00 = 9.1200 EUR/unit/yr
  H2 = 0.24 x 36.50 = 8.7600 EUR/unit/yr
  H3 = 0.24 x 35.20 = 8.4480 EUR/unit/yr
```

Step 1 of the procedure — compute an EOQ inside each tier:

```text
  Q1* = sqrt(2 x 24,000 x 180 / 9.1200) = sqrt(947,368)  = 973.3   feasible (1-999)
  Q2* = sqrt(2 x 24,000 x 180 / 8.7600) = sqrt(986,301)  = 993.1   NOT feasible (below 1,000)
  Q3* = sqrt(2 x 24,000 x 180 / 8.4480) = sqrt(1,022,727)= 1,011.3 NOT feasible (below 5,000)
```

Steps 2 and 3 — the candidate set is the one feasible optimum plus the break points that make
the higher tiers reachable: 973.3, 1,000, and 5,000.

Step 4 — evaluate the full annual cost at each candidate:

| Candidate | Ordering (EUR/yr) | Holding (EUR/yr) | Purchase (EUR/yr) | Total (EUR/yr) |
|---:|---:|---:|---:|---:|
| 973.3 units at 38.00 | 4,438.36 | 4,438.36 | 912,000.00 | 920,876.72 |
| 1,000 units at 36.50 | 4,320.00 | 4,380.00 | 876,000.00 | 884,700.00 |
| 5,000 units at 35.20 | 864.00 | 21,120.00 | 844,800.00 | 866,784.00 |

The optimum is 5,000 units, at a break point, 5.1 times the EOQ. Holding cost rises from 4,438
to 21,120 EUR a year, an increase of 16,682, and the purchase saving is 67,200. The net gain over
the tier-2 option is 17,916 EUR a year, and over the unconstrained EOQ it is 54,093 EUR a year.

Three lessons follow, and they are the lessons of the whole topic.

**The EOQ is not just suboptimal here; it is irrelevant.** No perturbation of 973.33 gets you to
5,000. The decision has changed character: it is no longer a balancing of two continuous costs
but a comparison of three discrete packages.

**The quantity decision now matters.** Fifty-four thousand euros a year is 5.9 per cent of
purchase spend, against the 0.26 per cent available from tuning the quantity inside tier 1. The
commercial term dominates the inventory arithmetic by a factor of about twenty.

**The answer has a feasibility problem the model cannot see.** Five thousand units is 76 days of
cover. For a valve with no shelf life that is acceptable; for a 60-day shelf life it is
impossible; for a product refreshed every 18 months it is a bet on the refresh date. The discount
analysis says 5,000. Only the physical context says whether 5,000 exists as an option.

### 6.6 The same schedule under incremental pricing

Suppose the same three tiers applied incrementally, so that the price applies only to units above
each break. The average price at 5,000 units becomes:

```text
  999 units at 38.00        =  37,962.00
  4,000 units at 36.50      = 146,000.00
  1 unit at 35.20           =      35.20
  total for 5,000 units     = 183,997.20
  average unit price        =      36.80
```

Compare with the all-units average of 35.20. The incremental schedule delivers a 3.2 per cent
average price reduction at the 5,000-unit point, where the all-units schedule delivers 7.4 per
cent. The purchase saving at 5,000 units falls from 67,200 to roughly 28,800 EUR a year, while
the extra holding cost of 16,682 is unchanged. The jump still pays, but by a much smaller margin,
and the optimum may well move back toward the EOQ or to the 1,000-unit break.

This is why the first question about any discount is not "how big is it" but "is it all-units or
incremental". Two schedules with identical numbers produce different quantities, different
inventory, and different working capital. A price list that does not say which convention applies
is an incomplete document.

## 7. Code, query, or calculation

The following is a hand calculation with a sensitivity check, in Python. It is deliberately short:
the arithmetic is the point, not the library.

```python
# EOQ with a sensitivity check. Units: units/yr, EUR, EUR/unit/yr.
from math import sqrt
D, S, C, i = 24_000, 180.0, 38.0, 0.24
H = i * C
Q = sqrt(2 * D * S / H)
var = lambda q, h=H: D * S / q + h * q / 2
base = var(Q)
print(f"Q*={Q:,.1f} units, cycle={Q / D * 365:.1f} days, variable cost={base:,.2f}")
for k in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    print(f"k={k:<5} Q={k * Q:8,.1f}  TC={var(k * Q):9,.2f}  penalty={(var(k * Q) / base - 1) * 100:6.2f}%")
for q, c in ((1_000, 36.50), (5_000, 35.20)):
    print(f"tier candidate Q={q:,}: variable={var(q, i * c):,.2f}  purchase={D * c:,.2f}")
# Expected output (hand-checked, not generated here):
# Q*=973.3 units, cycle=14.8 days, variable cost=8,876.71
# k=0.5   Q=   486.7  TC=11,095.89  penalty= 25.00%
# k=0.75  Q=   730.0  TC= 9,246.57  penalty=  4.17%
# k=1.0   Q=   973.3  TC= 8,876.71  penalty=  0.00%
# k=1.25  Q= 1,216.7  TC= 9,098.61  penalty=  2.50%
# k=1.5   Q= 1,460.0  TC= 9,616.50  penalty=  8.33%
# k=2.0   Q= 1,946.7  TC=11,095.89  penalty= 25.00%
# tier Q=1,000: variable=8,700.00 spend=876,000.00 | Q=5,000: variable=21,984.00 spend=844,800.00
```

Three checks before you trust any such output.

- Confirm that ordering cost and holding cost are equal at `Q*`. If they are not, `H` is being
  applied inconsistently, usually because one term uses an annual rate and the other a monthly
  one.
- Confirm that the penalty at `k = 0.5` and at `k = 2.0` are identical. If they are not, the
  penalty is being computed against a baseline that includes purchase spend, which obscures
  everything.
- Confirm that the tier candidates use the holding rate implied by their own unit price. In the
  output above, tier 2 at 1,000 units has a *lower* variable cost than the tier-1 optimum, which
  surprises people. It is correct: the cheaper unit carries a cheaper holding cost.

## 8. Data, standards, and regulatory context

There is no standard that prescribes an EOQ. No regulator requires an order quantity, no
accounting standard defines one, and no interoperability standard carries one. That absence is
itself the important fact: the formula is a management convention, and everything it depends on
is defined locally. What standards do constrain is the *inputs* and the *units*, and those
constraints are where the real data work sits.

### 8.1 Identification and packaging

| Standard or convention | What it governs | What it constrains for EOQ |
|---|---|---|
| GS1 GTIN | Trade item identification | Defines whether `D` is counted in eaches, cases, or pallets; a case-level GTIN changes `D` by the pack factor and `Q` by its square root |
| GS1 SSCC | Logistics unit identification | Defines the pallet or case as a handling unit, which is the rounding granularity for `Q` |
| GS1 package measurement rules | Dimensions and weights per level of the packaging hierarchy | Determines pallet quantity and therefore the feasible set of order quantities |
| UN/CEFACT and EDI order messages (X12 850, EDIFACT ORDERS) | Purchase order content | Determines where the ordered quantity lives and whether it is the quantity the model recommended |
| Incoterms 2020 | Transfer of cost and risk | Determines whether `C` includes freight and duty, hence whether `H` is consistent with `C` |
| ISO 8000 | Data quality for master data | A convention for assessing whether item master attributes are fit for purpose |

The packaging hierarchy is the constraint most often missed. A distributor that sells in eaches,
ships in cases of 12, and receives in pallets of 480 has three legitimate units of measure. An
EOQ of 973 units is 81.1 cases and 2.03 pallets. Rounding to 960 units (80 cases, 2 pallets) costs
0.01 per cent. Rounding to 1,440 (3 pallets) costs 7.8 per cent. The choice of rounding direction
is therefore worth more than most of the modelling refinement in this topic.

### 8.2 Cost definitions and the accounting boundary

The holding cost in the model and the inventory carrying cost in the accounts are not the same
object, and the difference is systematic.

- IAS 2 *Inventories* and ASC 330 *Inventory* govern which costs may be capitalised into
  inventory; both exclude most storage costs unless storage is necessary to the production
  process, and both exclude abnormal waste. The carrying value therefore omits much of the
  economic holding cost.
- The EOQ `H` needs the economic cost: capital, storage, handling, insurance, tax, obsolescence,
  shrinkage, and the opportunity cost of space. Some are period expenses in the accounts and none
  appears as a per-unit-per-year rate in any standard ledger.
- The ordering cost `S` has no accounting definition. It is either engineered or allocated, and
  section 5.3 explains why the allocated version is dangerous.

> **Note:** The two systems answer different questions. The accounts answer "what did we
> recognise as expense this period". The model answers "what changes if we place one more order".
> A cost that is fixed in the accounts can still be marginal for the decision, and a cost that is
> allocated per order can be entirely fixed in reality.

### 8.3 Physical and safety limits that bound the quantity

Several regulated limits can override any economic calculation, and they act as hard upper
bounds rather than costs:

| Constraint | Where it applies | Effect on `Q` |
|---|---|---|
| Shelf life and expiry rules | Food, pharmaceuticals, cosmetics | Caps the time supply; `Q/D x 365` must sit well inside shelf life |
| Dangerous goods package limits (ADR, IATA, IMDG) | Chemicals, batteries, aerosols | Caps units per package and per consignment, which can bind below the EOQ |
| Controlled-substance and precursor rules | Pharmaceuticals, certain chemicals | Caps the quantity that may be held at a site |
| Storage licence limits | Flammables, explosives, refrigerants | Caps total site inventory, which caps `Q` |
| Temperature-controlled capacity | Cold chain | Caps `Q` by pallet positions in the cold room, not by economics |

These are binding requirements, not conventions. An EOQ that violates one of them is not
suboptimal; it is non-compliant, and the analysis should have applied the cap before computing
anything.

### 8.4 What is convention rather than requirement

- The classic EOQ assumption list is a pedagogical device, not a definition; different texts list
  eight, nine, or ten. Rounding order quantities up is a sensible default, justified by the
  asymmetry of the penalty curve, not by any standard.
- The "annual holding rate of 20 to 25 per cent" is a rule of thumb with no regulatory basis.
- ABC classification thresholds are inherited conventions, not derived from EOQ theory.

## 9. Industry practice and case evidence

### 9.1 Where the formula is genuinely used

The idea is applied, honestly and usefully, in four settings.

**Repetitive manufacturing.** A plant that runs the same part number every few weeks, with a
known changeover and level demand, is close to the model's assumptions. Even here the correct
model is the economic production quantity (EPQ), because output is gradual: the EPQ adds a factor
of `1 / sqrt(1 - d/p)`, which is 1.41 for a line running at twice demand.

**Container, pallet, and case quantities.** The handling unit often determines the quantity, and
the analysis reduces to choosing how many units per order: not "what is the EOQ" but "is two
pallets better than one, and is three better than two". The penalty ratio answers that in a line.

**Supplier minimum order quantities (MOQs).** When the MOQ exceeds the EOQ it becomes the floor,
and the analysis becomes an evaluation of forced overstock: how many days of cover does the
minimum imply, and is that compatible with shelf life and working capital?

**A sanity check on inherited parameters.** Many enterprise resource planning (ERP) systems carry
lot-size fields unreviewed for a decade. Comparing a fresh EOQ with the stored value is a cheap
way to find items where the quantity is wildly out of line, even when the EOQ is not adopted.

### 9.2 Documented cases, with their limits

| Case | What was done | Reported result | Grade | Caveat |
|---|---|---|---|---|
| DAF Trucks press shop lot sizing (Coppens, TU/e master's thesis, 2013) | Capacitated lot sizing with practical round-up to pallet and pack quantities, plus a maximum lot size for flexibility | 26% cost reduction versus the ERP method; the maximum-lot constraint cost 6.7%; the warehouse volume constraint cost 10.4% | C | Single company, thesis-reported, not independently audited |
| Wexiödisk lot sizing study (Ringius and Nilsson, 2024) | ABC classification, then EOQ on the three highest-value items, rounded to packaging | Lot sizes of 3–4 units versus a prior fixed batch of 6; approximately 47% inventory value reduction on those items | C | Student thesis, three items, short observation window |
| AllAboutLean practitioner analysis (Roser) | Argues lot size is usually set by machine batch size, changeover time, container size, and customer order size rather than by a cost optimum | No quantitative result claimed | E | Practitioner blog; a reasoned argument, not evidence |
| Industrial distributor discount example in section 6 | All-units schedule, three tiers | Optimum at a break point, 5.1 times the EOQ; the purchase saving dominates the holding change by roughly four to one | — | Constructed illustration, internally consistent, not a company case |

The two theses point the same way from different angles. Both found that the value came from
*changing* the lot size away from an unreviewed default, and both found the final quantity was
set by a physical constraint — pallet, pack, or warehouse volume — rather than by the cost
optimum. The DAF case quantifies what constraints cost: 6.7 per cent for a maximum lot size that
preserves flexibility, 10.4 per cent for a warehouse volume limit. That is a useful calibration:
a constraint that costs 10 per cent is worth arguing about; one that costs 0.5 per cent is not.

### 9.3 The history, which explains the formula's persistence

The square-root formula was published by Ford Whitman Harris in 1913 in *Factory, The Magazine
of Management*, under the title "How Many Parts to Make at Once". Harris was a Westinghouse
engineer with no education beyond high school, and the paper is three pages long. It was then
effectively lost: Erlenkotter's archival work documents that for decades the paper was cited with
an incorrect year, could not be located, and was only properly rediscovered in the late 1980s.
R. H. Wilson popularised the formula in a 1934 *Harvard Business Review* article on stock control,
which is why many textbooks call it the Wilson formula; German-language literature calls it the
Andler formula, after Kurt Andler's 1929 treatment.

The result has been independently rediscovered many times, which is evidence about its
naturalness: the same balance of a `1/Q` term against a linear term appears in electrical
engineering as Kelvin's law for economic conductor size, and in any domain where a fixed cost per
event trades against a holding cost per unit. A compact timeline: Harris 1913 (derivation, then
lost), Andler 1929, Wilson 1934 (popularisation), Hadley and Whitin 1963 (the all-units discount
procedure), Erlenkotter 1988 to 1990 (rediscovery), Drake and Marley 2014 (centennial review).

The persistence is not intellectual inertia. A formula that is approximately right, computable on
the back of an envelope, and whose error costs little survives because it is useful. The problem
is that its usefulness has been over-generalised: it is a sizing heuristic for a stable,
high-volume item, and it has been applied to every item in every catalogue.

### 9.4 What practitioners argue about

Three positions recur, and all three contain something true.

**EOQ is obsolete.** Deterministic demand never occurs, ordering cost is an accounting fiction,
and modern systems plan from a forecast, a lead time, and a service target, none of which appear
in the formula. Correct about the assumptions, wrong about the role: nobody proposes EOQ as the
replenishment engine for a stochastic item.

**EOQ is fine as a starting point.** The flat basin makes the formula robust and the real work is
in the constraints. True for stable high-volume items, progressively less true as variability
rises; for intermittent demand the EOQ can be 15 to 25 per cent too small.

**The ordering cost is the real problem.** The output is only as good as its most contestable
input, `S` is usually an allocation, and an inflated `S` will justify whatever batch size
management wants. The most useful of the three, because it locates the failure in the organisation
rather than the mathematics. Section 5.3 elaborates it.

What the positions agree on: the quantity is a small lever, the constraints are large, and
presenting a single number without its feasibility check is the actual professional error.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| False precision on estimated inputs | A slide shows "Q* = 973.3 units" with no input provenance | The formula returns a real number, so it looks like a measurement | Credibility collapse when someone audits an input | Round to the handling unit; report the penalty of the rounding |
| Demand computed from shipments | `D` equals the sum of order quantities, and the stock-out log is not joined | Shipments equal demand minus unmet demand | Systematic under-sizing, self-reinforcing stock-outs | Reconstruct demand; add unmet demand back |
| Time-base mismatch | Implied order cycle is 0.4 days or 120 days | `D` monthly, `H` annual, or `S` per order line | Answer wrong by a factor of 3.5 or 12 | Check the dimensions of `D x S / H` before computing |
| Holding cost quoted only as a percentage | "We use 20 per cent" and nobody can list the components | The percentage hides capital, storage, and obsolescence in one number | Wrong rate for perishables and for slow movers, in opposite directions | Build `H` from components; publish the list |
| Ordering cost is an allocation | `S` is identical for a 2 EUR part and a 2,000 EUR part | Departmental cost divided by order count | Feedback loop: fewer orders raise `S`, which recommends fewer orders | Use engineered `S`; treat departmental overhead as fixed |
| Ordering frequency exceeds receiving capacity | Implied orders per year times receiving hours exceeds available dock hours | EOQ optimises cost, not throughput | Recommendations the warehouse cannot execute; the policy is quietly ignored | Add a capacity constraint; re-solve as a constrained problem |
| Every order sits exactly at the MOQ | Order quantity distribution is a spike at the minimum | The MOQ exceeds the EOQ, and the system floors silently | Chronic overstock on slow movers; expiry and obsolescence | Compute days of cover implied by the MOQ; flag items where it exceeds shelf life |
| Discount schedule ignored | Procurement negotiates a new price and planning is not re-run | Lot-size logic lives in a different system from the price list | Leaving 5 per cent of spend on the table | Re-run lot sizing whenever a price schedule changes |
| Cycle stock exceeds shelf life | `Q/D x 365` greater than half the shelf life | EOQ has no expiry term | Write-offs, which cost more than the ordering saving | Cap `Q` by shelf life; use Topic 10 methods |
| EOQ presented as a policy | A document titled "our EOQ policy" with no reorder point, no safety stock, no review rule | The model is mistaken for a decision system | Service failures that nobody connects to the lot size | State the four policy components explicitly; defer to Topics 04 and 06 |
| One `S` and one `H` for an entire catalogue | The same parameters on an A item and a C item | Convenience, or a single corporate standard | Fast movers over-ordered, slow movers under-ordered | Segment parameters; at minimum by ABC and by shelf-life class |
| Capacity-constrained plant uses accounting `S` | Changeover consumes bottleneck capacity but `S` contains only labour and scrap | The true cost includes displaced contribution margin | Systematic under-sizing of lots, and lost throughput | Use an engineered `S` that includes opportunity cost; document the basis |

## 11. Assumptions and limits

| Assumption | Why the model needs it | What breaks if it fails | Direction of bias on Q | How to test it |
|---|---|---|---|---|
| Demand is deterministic and constant | The cost function is written for a single rate `D` | Lumpy, seasonal, or trending demand | Right on average, wrong every period; no simple direction | Plot the demand series; compute the coefficient of variation; test for trend |
| Demand rate is known without error | `D` enters the square root directly | Forecast or history error | `Q` biased by half the percentage error in `D`, in the same direction | Compare forecasts with actuals at the item level |
| Lead time is known and constant | The quantity derivation does not use `L`, but feasibility does | Variable lead time | `Q` unaffected; the reorder point is badly biased | Plot lead time by lane; compute its standard deviation |
| Replenishment is instantaneous | Average inventory is `Q/2` | Gradual production or staged receipts | EOQ too small; EPQ is larger by `1/sqrt(1-d/p)` | Compare production rate with demand rate |
| Unit price is independent of quantity | `C`, and hence `H`, are constants | Quantity discounts or freight breaks | Optimum moves to a break point, usually upward | Obtain the actual price schedule; check whether it is all-units or incremental |
| No shortages are permitted | The model has no backorder term | Planned backorders, or stock-outs that occur anyway | Allowing backorders increases `Q`; ignoring them makes the model look better than reality | Compare recorded service with the model's implied service |
| Ordering cost is a known marginal constant | `S` enters the square root | Allocation, joint orders, or managerial choice | `Q` moves with the square root of the error in `S`, same direction | Ask whether `S` changes when one more order is placed |
| Holding cost is proportional to quantity | `H x Q/2` is linear | Space priced in blocks, stepwise handling | Cost curve becomes a staircase; local minima appear | Compare storage cost with occupied pallet positions |
| Items are independent | One item, one order | Joint replenishment, shared containers | Effective `S` per item is lower, so the true optimal frequency is higher | Count how many items arrive on the same vehicle or order |
| Horizon is infinite | No terminal condition | Seasonal end, contract end, product end of life | Final order should be smaller than `Q*` | Identify the end date and compute the residual stock |
| No capacity or handling constraint binds | The optimum is interior | Truck, pallet, dock, or machine capacity | `Q` capped; the constrained optimum is at the cap | Check all four constraints explicitly |
| Parameters are stationary | One computation, one policy | Drift in demand, cost, or terms | Policy becomes progressively wrong in one direction and is abandoned | Re-estimate quarterly; track the parameter series |

Four of these limits are the ones a competent analyst is expected to raise unprompted.

**The deterministic demand assumption is the strongest and the least defensible.** Demand is
never a smooth flow. Under stochastic demand the EOQ is systematically *smaller* than the
optimal quantity in the corresponding stochastic problem, because it ignores that a larger order
reduces the number of cycles exposed to stock-out risk. Reported quantity errors are in the
region of 15 to 25 per cent with cost errors of a few per cent — treat the EOQ as a lower bound
for a stochastic item, not a target.

**The ordering cost is a construct more often than a measurement.** It is the input most likely
to be set by the person who will use the output. The test is one question: *if we placed one more
order this year, what cost would the company actually incur?* If the answer is "none, the
department is fixed", then `S` is not an ordering cost and the formula is allocating, not
deciding.

**The model has no service dimension and no horizon.** It cannot say whether a quantity is
compatible with a 95 per cent cycle service level (CSL), because no service level is in it; and
it cannot see the peak season, the announced price rise, or the June discontinuation. The cost-
minimising quantity and the service-achieving quantity are answers to different questions.

The formula takes ten minutes to learn and years to learn when not to use.

## 12. Exercises

### Exercise 1 — Compute and verify (application, 30 minutes)

A distributor sells 7,300 filter cartridges a year at 14.00 EUR each, ordering at 95 EUR per order
with a 22 per cent holding rate.

**Deliverable:** `Q*`, ordering cost, holding cost, total variable cost, order cycle in days, and
the equal-cost check.
**Acceptance criteria:** inputs with units; ordering and holding equal within 1 EUR; the cycle
stated in days; one sentence on an 18 per cent holding rate.
**Time:** 30 minutes.

### Exercise 2 — The flat basin (application, 40 minutes)

Using the item from Exercise 1, compute the total variable cost and the penalty ratio at `Q/Q*`
of 0.5, 0.75, 1.0, 1.25, 1.5, and 2.0.

**Deliverable:** a table of quantity, cost, and penalty in per cent, plus a one-paragraph reading.
**Acceptance criteria:** penalties at 0.5 and 2.0 identical; the 0.75 versus 1.25 asymmetry
explained; a stated range within which you would not correct the quantity.
**Time:** 40 minutes.

### Exercise 3 — The discount decision (application, 60 minutes)

A supplier offers all-units pricing: 1–499 units at 22.00 EUR, 500–1,999 at 21.20 EUR, 2,000 and
above at 20.40 EUR. Annual demand is 9,000 units, ordering cost is 120 EUR, holding rate 25 per
cent. Repeat assuming the tiers are incremental.

**Deliverable:** both analyses, the candidate set in each case, the chosen quantity, and the
annual cost difference between the conventions.
**Acceptance criteria:** holding cost recomputed per tier; every candidate evaluated including
purchase cost; infeasible per-tier EOQs identified as infeasible; the two conclusions compared.
**Time:** 60 minutes.

### Exercise 4 — Diagnose the parameter (critique, 45 minutes)

A colleague presents an EOQ of 41,000 units for a component with annual demand of 120,000 units,
an ordering cost of 2,400 EUR, and a holding cost of 0.34 EUR per unit per year. The purchasing
manager says the ordering cost is "the procurement department budget divided by the number of
purchase orders".

**Deliverable:** a one-page critique naming at least three problems, the direction of each bias,
and a corrected calculation with stated assumptions.
**Acceptance criteria:** the allocation feedback loop named and explained; the corrected `S`
justified in components; the new quantity stated in days of cover.
**Time:** 45 minutes.

### Exercise 5 — Derive, do not memorise (derivation, 40 minutes)

Starting from `TC(Q) = D x S / Q + H x Q / 2`, derive `Q*`, prove the stationary point is a
minimum, and derive the penalty ratio `TC(kQ*)/TC(Q*) = (k + 1/k)/2`.

**Deliverable:** a half-page derivation with every step shown and every unit checked.
**Acceptance criteria:** the dimensional check performed on `D x S / H`; the second-derivative
argument stated; the cancellation shown, with a note on why the result is parameter-free.
**Time:** 40 minutes.

### Exercise 6 — Audit S and H (design and fieldwork, 60 minutes)

For one real item in an organisation you can access, reconstruct the ordering cost from the
activities performed and the holding cost from its components. Compare each with the figure the
planning system currently uses.

**Deliverable:** two component tables, the reconstructed values, the system values, and a short
note on the difference.
**Acceptance criteria:** at least four activities in `S`; at least four components in `H`; the
capital rate's source identified; the effect on `Q*` computed as a percentage.
**Time:** 60 minutes.

### Exercise 7 — Production rather than purchase (application, 45 minutes)

A line produces a bracket at 4,000 units per week against demand of 1,800 units per week. Setup
takes 3.5 hours at 90 EUR per hour including lost contribution. Annual demand is 90,000 units,
unit cost is 6.40 EUR, and the holding rate is 20 per cent.

**Deliverable:** the EOQ, the EPQ, and the ratio between them, with the interpretation.
**Acceptance criteria:** the `d/p` factor computed explicitly; both quantities stated in weeks of
demand; one sentence on why the single-lot EOQ is not producible here.
**Time:** 45 minutes.

### Exercise 8 — The honest one-pager (design and writing, 40 minutes)

Write a one-page memo to an operations director who has asked you to "set the EOQs for the
catalogue", which holds 4,000 SKUs: 300 stable high-volume items, 2,100 moderate movers, and
1,600 intermittent items.

**Deliverable:** a memo with a recommendation, a segmentation, and an explicit statement of what
the analysis will and will not deliver.
**Acceptance criteria:** the recommendation distinguishes the three segments; it states that
quantity is one of four policy variables; at least one quantitative expectation carries a basis;
the limitations are stated before the benefits.
**Time:** 40 minutes.

## 13. Self-check answers

**Exercise 1.** `H = 0.22 x 14.00 = 3.08 EUR/unit/yr`, so `Q* = sqrt(2 x 7,300 x 95 / 3.08) =
sqrt(450,325) = 671.1 units`. Ordering `= 1,033.3 EUR/yr` and holding `= 3.08 x 335.6 = 1,033.6
EUR/yr`, total `2,066.9 EUR/yr`, cycle `671.1 / 20.0 = 33.6 days`. At 18 per cent, `Q* = 741.9`, a
10.5 per cent increase costing about 0.5 per cent: not sensitive to a plausible holding-rate error.

**Exercise 2.** The penalties are 25.00, 4.17, 0.00, 2.50, 8.33, and 25.00 per cent. The values at
0.5 and 2.0 are identical because the formula is symmetric in `k` and `1/k`; 0.75 costs 4.17 per
cent and 1.25 costs 2.50 per cent because a 25 per cent undershoot is a 33 per cent overshoot. A
defensible range for not correcting is roughly 0.8 to 1.3, where the penalty stays under 5 per cent.

**Exercise 3.** `H` per tier is 5.50, 5.30, and 5.10 EUR. `Q1* = sqrt(2,160,000 / 5.50) = 626.7` is
infeasible (above the tier-1 range), `Q2* = 638.4` is feasible, and `Q3* = 650.8` is infeasible
(below 2,000). The candidate set is 499, 638.4, and the break point at 2,000; all-units totals are
about 201,537, 194,184, and 189,240 EUR, so 2,000 wins by about 4,944 EUR. Under incremental
pricing the average price at 2,000 units is about 21.40 EUR rather than 20.40, and the advantage
shrinks to roughly 1,380 EUR — still in favour of the break point, but by a margin a small change
in the holding rate could erase.

**Exercise 4.** Three problems. First, `S = 2,400` is an allocation: budget divided by order count,
so placing fewer orders raises `S` and recommends a still larger quantity. Second, 2,400 EUR is
implausible for a component with 120,000 units of annual demand unless every order consumes most
of a working week. Third, the resulting `41,000/120,000 x 365 = 125` days of cover is a
working-capital decision disguised as an inventory calculation. An engineered `S` of 150 to 300
EUR gives `Q* = sqrt(2 x 120,000 x 250 / 0.34) = 13,283` units, or 40 days of cover.

**Exercise 5.** The derivation is in section 3, pass 2, and section 5, steps 6 and 7. Dimensional
check: `D` [units/yr] x `S` [EUR/order] / `H` [EUR/(unit x yr)] = [units^2]. The penalty ratio
follows by substituting `Q = kQ*` and using `2DS = H Q*^2`, which cancels every parameter.

**Exercise 6.** A good submission finds an ordering cost well below the system figure, because
the system figure usually includes allocated overhead, and a holding cost above it for perishable
items, because the system rate omits obsolescence. The effect on `Q*` is the square root of the
ratio: if reconstructed `S` is half the system value, `Q*` falls by 29 per cent.

**Exercise 7.** `H = 1.28 EUR/unit/yr`, `S = 3.5 x 90 = 315 EUR`, so `EOQ = sqrt(2 x 90,000 x 315 /
1.28) = 6,655` units. With `d/p = 0.45`, `1 - d/p = 0.55` and `EPQ = 6,655 / 0.7416 = 8,974` units,
35 per cent larger, because the line consumes inventory while producing it.

**Exercise 8.** A strong memo segments: EOQ and close relatives for the 300 stable items;
constrained lot sizing with MOQ and shelf-life caps for the moderate movers; a different method
for the 1,600 intermittent items, where the EOQ is biased low (Module 08). It states that the
deliverable is a quantity recommendation and a cost penalty, not a policy, and that reorder points
and safety stocks are separate work. It gives one quantified expectation — say a 0.2 to 0.5 per
cent reduction in variable cost across the stable segment — and says plainly that this is small. A
weak memo promises an inventory reduction percentage without a baseline.

## 14. Interview questions

**1. "Derive the EOQ formula."**

A strong answer writes the two cost terms, sets them equal, solves, and notes that the square
root arises because one term is proportional to `Q` and the other to `1/Q`, with a positive
second derivative confirming a minimum. A weak answer recites `sqrt(2DS/H)` and cannot say where
the 2 comes from.

**2. "How sensitive is the EOQ to bad inputs?"**

A strong answer gives two levels: quantity scales as the square root of the inputs, so a 10 per
cent input error moves `Q*` by about 5 per cent, and the cost penalty is second order and smaller
still — doubling `Q` costs 25 per cent, a 25 per cent error about 2.5 per cent. A weak answer says
"it's very sensitive, you need good data" without quantifying.

**3. "What happens to the EOQ if demand is not constant?"**

A strong answer says the quantity becomes right on average and wrong in every period, with the
service failure concentrated in the peak, and names dynamic lot sizing as the replacement. A weak
answer says "the formula doesn't apply" and stops.

**4. "A supplier offers a volume discount. How do you decide?"**

A strong answer asks first whether the discount is all-units or incremental, then recomputes `H`
per tier, computes a per-tier EOQ, discards the infeasible ones, and evaluates full annual cost —
including purchase spend — at every feasible optimum and every reachable break point, noting that
the optimum often sits at a break point where the derivative does not exist. A weak answer
compares the discount percentage with the holding cost percentage.

**5. "What is the ordering cost, and how would you measure it?"**

A strong answer defines it as the incremental cost of placing one more order and lists the
activities — requisition, PO, expediting, receiving, inspection, invoice matching — noting that
allocated overhead is not an ordering cost and explaining the feedback loop that results from
using it. A weak answer says "it's the cost of placing an order, from the finance system".

**6. "Why might an inflated setup cost be defensible?"**

A strong answer explains that in a capacity-constrained plant a changeover consumes scarce
capacity, so its true cost includes the contribution margin of displaced production, which the
accounting system does not capture, and insists the inflation be justified in components. A weak
answer either rejects any adjustment as manipulation or accepts it without justification.

**7. "What is the difference between an order quantity and a reorder point?"**

A strong answer says the quantity is the height of the sawtooth and the reorder point is the level
that triggers replenishment; they come from different models, and the quantity is robust to error
while the reorder point is sensitive because it sits in the tail of a distribution. A weak answer
treats the reorder point as "the EOQ divided by something".

**8. "Does the EOQ have any remaining role?"**

A strong answer says yes, narrowly: a sizing heuristic for stable, high-volume items, a way to
choose between one, two, or three pallets, and a diagnostic against unreviewed system parameters.
It says the formula is not a policy. A weak answer either defends it as a general method or
dismisses it entirely.

**9. "Where does the EOQ break down worst?"**

A strong answer names intermittent demand, short shelf life, and MOQ-dominated items, and
explains that in each case a constraint or a distribution replaces the smooth-flow assumption. It
notes that the failure is not a wrong number but an irrelevant one. A weak answer says "when
demand is uncertain" without distinguishing the kinds of uncertainty.

**10. "You are asked to set order quantities for 4,000 SKUs. What do you do first?"**

A strong answer segments before computing: stable high-volume items get the formula, constrained
items get a constrained comparison, intermittent items get a different method, and a sample is
audited for parameter quality. It also asks who owns the reorder point and the safety stock. A
weak answer starts computing EOQs for all 4,000 and reports the total inventory reduction.

## 15. Key terms

- **EOQ** — economic order quantity; the cost-minimising lot size under constant demand.
- **Q*** — the optimal order quantity for a given set of inputs.
- **D** — annual demand rate, in units per year.
- **S** — ordering or setup cost, in currency per order.
- **H** — holding cost, in currency per unit per year.
- **i** — holding rate, as a fraction of unit cost per year; `H = i x C`.
- **C** — unit cost, purchase price or standard cost.
- **TC(Q)** — total annual variable cost: ordering plus holding.
- **Equal-cost identity** — at the optimum, ordering cost equals holding cost.
- **Penalty ratio** — `(k + 1/k)/2`, the cost of ordering `k` times the optimal quantity.
- **Flat basin** — the region around the optimum where quantity errors cost little.
- **Elasticity** — the percentage change in `Q*` per percentage change in an input; one half for
  `D`, `S`, and `H`.
- **Cycle stock** — inventory that exists because orders are batched; `Q/2` on average.
- **Sawtooth** — the inventory pattern under fixed-quantity ordering.
- **Reorder point** — the inventory position at which replenishment is triggered.
- **Inventory position** — on-hand plus on-order minus backorders.
- **Safety stock** — the buffer against variability; not part of the EOQ model.
- **All-units discount** — one price applies to the whole order at or above a break point.
- **Incremental discount** — the lower price applies only above the break point.
- **Break point** — the quantity at which the unit price changes.
- **MOQ** — minimum order quantity; a floor, not an optimum.
- **EPQ** — economic production quantity; the lot-size analogue with gradual output.
- **Changeover** — the work between the last good unit of one product and the first of the next.
- **Wagner–Whitin** — the dynamic programming method for lot sizing with time-varying demand.
- **Andler formula** — German-language name for the EOQ formula.
- **Wilson formula** — common attribution to R. H. Wilson, who popularised the result in 1934.

## 16. Sources

**Peer-reviewed and archival**

- Harris, F. W. (1913). "How Many Parts to Make at Once." *Factory, The Magazine of Management*,
  10(2), 135–136, 152. The original derivation; details as reported by Erlenkotter (1990). Grade A.
- Erlenkotter, D. (1989). "An Early Classic Misplaced: Ford W. Harris's Economic Order Quantity
  Model of 1915." *Management Science*, 35(7), 898–900. DOI: 10.1287/mnsc.35.7.898. Grade A.
- Erlenkotter, D. (1990). "Ford Whitman Harris and the Economic Order Quantity Model."
  *Operations Research*, 38(6), 937–946. DOI: 10.1287/opre.38.6.937. Grade A.
- Wilson, R. H. (1934). "A Scientific Routine for Stock Control." *Harvard Business Review*,
  13(1), 116–128. Popularisation, not derivation. Grade A for the historical record.
- Dobson, G. (1988). "Sensitivity of the EOQ Model to Parameter Estimates." *Operations Research*,
  36(4), 570–574. DOI: 10.1287/opre.36.4.570. Grade A.
- Pentico, D. W., and Drake, M. J. (2011). "A Survey of Deterministic Models for the EOQ and EPQ
  with Partial Backordering." *European Journal of Operational Research*, 210(3), 449–458. Grade A.
- Andriolo, A., Battini, D., Grubbström, R. W., Persona, A., and Sgarbossa, F. (2014). "A Century
  of Evolution from Harris's Basic Lot Size Model." *International Journal of Production
  Economics*, 155, 38–60. Grade A.
- Drake, M. J., and Marley, K. A. (2014). "A Century of the EOQ." In *Handbook of EOQ Inventory
  Problems*, Springer. DOI: 10.1007/978-1-4614-7639-9_1. Grade A.
- Asadabadi, M. R. (2016). "A Revision on Cost Elements of the EOQ Model." *Studies in Business
  and Economics*, 11(1). DOI: 10.1515/sbe-2016-0001. Grade A.
- Sangeladji, M., and Damooei, J. (2009). "Batch-Level Costs in Activity-Based Costing."
  *International Business and Economics Research Journal*, 8(9). DOI: 10.19030/iber.v8i9.3167.
  Grade A, lower-tier journal.
- Teunter, R. H., Babaï, M. Z., and Syntetos, A. A. (2025). "Simple Economic Order Quantity
  Heuristics for Stochastic Inventory Control." *IMA Journal of Management Mathematics*.
  DOI: 10.1093/imaman/dpaf035. Grade A.

**Teaching material, standards, and official**

- Schönsleben, P., *Integral Logistics Management*, section 11.4, Open Education Platform for
  Management Education (ETH Zurich). Sensitivity analysis with the 64 and 156 per cent figures.
  Grade B.
- MIT OpenCourseWare, ESD.273J *Logistics and Supply Chain Management*, lecture 2. Grade B.
- Bilkent University IE375 course notes, chapter 4, quantity discount procedures. Grade B.
- Rossetti, M. D., *Analysis of Inventory Systems*, chapter 8. Grade B.
- IAS 2 *Inventories*; ASC 330 *Inventory*. Grade B.
- GS1 General Specifications, including GTIN, SSCC, and package measurement rules. Grade B.
- Incoterms 2020, International Chamber of Commerce. Grade B.
- Aro-Gordon, S., and Gupte, J. (2016). "Revisiting Classic Economic Order Quantity Model."
  ICOMAR conference proceedings. Grade C.

**Company cases, theses, and practitioner analysis**

- Coppens, T. (2013). Capacitated lot sizing in a make-to-order press shop. Master's thesis,
  Eindhoven University of Technology, with DAF Trucks. Grade C.
- Ringius, J., and Nilsson, M. (2024). Lot size determination and order planning at Wexiödisk.
  Master's thesis, Linnaeus University. Grade C.
- Perkins, D. (2004). "Incorporating the Opportunity Cost of Setups into Production-Related
  Decisions." *Management Accounting Quarterly*. Grade C.
- Vercio, A., and Shoemaker, J. (2007). "ABCs of Batch Processing." *Journal of Accountancy*.
  Grade C.
- Roser, C. "How to Determine Your Lot Size, Part 1." AllAboutLean.com. Grade E; a reasoned
  practitioner argument, not evidence.

**Vendor material, cited only as evidence of practice**

- NetSuite, "Minimum Order Quantity (MOQ): Formula, Tips, and Benefits." Grade D.
- Imperia SCM, "Minimum Order Quantity: Its Real Impact on Inventory." Grade D.
- Logio, "What is Minimum Order Quantity." Grade D.
- EOQCalculator.com, quantity discount variant and Wilson formula history pages. Grade D.

> **Note on citations.** Historical details follow Erlenkotter's archival work; where the
> publication year of Harris's article is given differently in secondary sources, the text says
> 1913 and notes the discrepancy. No figure in this file is presented as a company result unless
> the case is named and graded. The worked example in section 6 is constructed and internally
> consistent, not reported.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic, the cost-service frontier:
  [`01_cost-service-frontier.md`](01_cost-service-frontier.md)
- Part map: [`../_index.md`](../_index.md)
- Survey file: [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
- Part I index: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Forecast error metrics and cost-weighted evaluation:
  [`../07_time_series_forecasting/09_error-metrics-cost.md`](../07_time_series_forecasting/09_error-metrics-cost.md)
- Intermittent demand and the compound Poisson hurdle model:
  [`../08_hierarchical_intermittent_probabilistic_forecasting/06_compound-poisson-hurdle.md`](../08_hierarchical_intermittent_probabilistic_forecasting/06_compound-poisson-hurdle.md)

Sibling topics in this module that are not yet written are listed here as plain text, not links:

```text
03_newsvendor.md                (planned)  newsvendor and the critical fractile
04_safety-stock.md              (planned)  safety stock under demand and lead-time uncertainty
05_service-metrics.md           (planned)  CSL, fill rate, OTIF
06_policy-families.md           (planned)  (s,Q), (R,S), min-max, periodic review
07_lead-time-demand.md          (planned)  lead-time demand distributions
08_multi-echelon.md             (planned)  multi-echelon inventory, MEIO, DRP
09_pooling-postponement.md      (planned)  pooling and postponement
10_segmentation-shelf-life.md   (planned)  segmentation, DDMRP, shelf life
```
