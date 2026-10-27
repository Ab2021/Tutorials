# Inventory Types and Service Metrics

> Module 01 — Supply Chain Foundations for Data Scientists
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 03, 04
> Version 2026.5

---

## 1. Why this topic exists

A company's balance sheet reports inventory as one number. The business manages at least six
different kinds of stock, each created by a different decision, each costing money for a
different reason, and each owned by a different role. The single number is an accounting
residual. The decisions are plural. Most inventory arguments are really arguments about which
kind of stock the speaker means.

Two distributors can report the same 45 days of inventory and be in completely different
conditions. The first holds 45 days of cycle stock on a stable product with predictable demand;
it is over-ordering and could free cash by ordering more often. The second holds 45 days made up
of 15 days of safety stock on a declining line, 20 days of stock that will never sell, and 10
days of pipeline it does not need because it pays for expedited freight. The first has a
working-capital problem. The second has a product problem, a forecasting problem, and a
disposal problem wearing a working-capital costume. The metric is identical. The interventions
are not.

This distinction became operational in 2024–2026. The pandemic build-up, the 2022–2024
destocking, and the 2025 tariff frontloading moved inventory onto the board agenda, where
"reduce it" becomes a target and targets applied to an undifferentiated number produce
predictable damage. Sections 9 and 10 give the evidence and the damage.

The second half of the topic is service, because inventory is the cost of service and service is
the reason inventory exists. "Service level" is not one metric. It is a family of at least eight
metrics that answer different questions and can differ from one another by twenty points on the
same stock. A company can report 95% cycle service level and deliver a 78% fill rate, and both
numbers can be correct. Section 6 works that example numerically.

**Why this matters to a data scientist specifically.** You will be asked to "optimise inventory"
or "predict stock-outs." Both requests are underspecified in a way that decides whether the work
is useful. Optimise which inventory: cycle, safety, pipeline, or obsolete? Predict stock-outs at
what grain: SKU-location-day, order line, or customer? Measured how: a cycle that ends short, a
unit that cannot ship, an order that arrives incomplete? The metric you accept becomes the
objective function, and the model is judged on whichever metric the business already reported.

Two recurring failures show the shape of the problem:

1. **An inventory reduction that removes the cheapest stock.** A programme targets 15% inventory
   reduction; the quickest lever is order quantity, so halving the lot size halves cycle stock.
   The balance sheet improves within a quarter, ordering and freight costs rise, and the
   programme reports the win without the cost. Safety stock, in parameters nobody has reviewed
   since 2019, is untouched.
2. **A service metric that measures the wrong event.** A supplier reports 97% on-time in-full
   (OTIF) against its own confirmation date and 89% against the retailer's must-arrive-by date.
   Both are computed correctly. The retailer's chargebacks are based on the second.

Each failure is a classification failure before it is a modelling failure. This topic supplies
the classification.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** every line in an inventory extract into one of the stock types in section 3.1, and
   **name** the decision and the role that created it.
2. **Apply** Little's law to a lane and to a whole network, and **use** it to explain why
   inventory cannot be reduced permanently without reducing flow time or throughput.
3. **Compute** safety stock, reorder point, cycle stock, and pipeline inventory for a single
   stock-keeping unit (SKU) under a stated service target, with units shown at every step.
4. **Distinguish** cycle service level (CSL), fill rate, ready rate, OTIF, OTIF with
   documentation (OTIF-D), and perfect order, and **state** which one a given customer promise
   actually measures.
5. **Explain** why CSL and fill rate diverge, in which direction, and **predict** the direction
   from the ratio of order quantity to lead-time demand variability.
6. **Choose** a demand distribution for a slow-moving item and **justify** the choice against the
   normal approximation, using over-dispersion and intermittency evidence.
7. **Diagnose** an aged or at-risk position, **recommend** a write-down or disposal decision, and
   **critique** an inventory-reduction proposal for whether it removes redundant stock or merely
   the stock that is easiest to measure.

Each outcome maps to an exercise in section 12, with grading criteria or a model answer.

## 3. The core mental model

Hold one picture in your head: **inventory is time made visible.** A unit of stock is not a
quantity. It is a duration that has been purchased, stored, and placed on a balance sheet. The
quantity is the shadow; the duration is the object.

```text
   ONE LANE, ONE SKU, MEASURED IN WEEKS OF FLOW

   supplier          transit              DC               customer
   +--------+   +----------------+   +---------+      +----------+
   |        |-->|    PIPELINE    |-->|  CYCLE  |----->|          |
   |        |   |  R x transit   |   |  Q / 2  |      |          |
   +--------+   +----------------+   +---------+      +----------+
                                           |
                                      +---------+
                                      | SAFETY  |   held, not planned to be used
                                      |  z x sL |   consumed only in the tail
                                      +---------+
                                           |
                                      +---------+
                                      | AT-RISK |   same shelf, weakening demand
                                      | OBSOLETE|   no future flow, still carries cost
                                      +---------+

   total flowing inventory   I = R x T
   T = transit + Q/(2R) + SS/R

   the last two boxes add to I but not to T: dead stock has no flow time
```

The picture has five properties worth memorising.

**Property 1 — Every unit of inventory is a duration.** Divide any inventory quantity by its
throughput and you get time. This is not a modelling convenience; it is an identity, and it is
the single most useful calculation in the topic. A warehouse holding 4,712 units that ships 1,000
units per week is holding 4.7 weeks, whatever the label on the bin says.

**Property 2 — Each type answers a different question, so a single inventory target is
incoherent.** Cycle stock answers "how much to buy at once?" Safety stock answers "how much
variability can we absorb?" Pipeline answers "how much is in motion?" A target applied to the
total without specifying the composition is a target applied to an average of unrelated
decisions. Section 3.1 gives the full mapping.

**Property 3 — Inventory is a residual of decisions made elsewhere.** Nobody decides to hold
safety stock. Somebody decides a service target, somebody estimates variability, and the safety
stock is the arithmetic consequence. Nobody decides to hold pipeline inventory; somebody chooses
a mode and an Incoterm. When you are asked to reduce inventory, you are being asked to change
one of those upstream decisions, and the first analytic move is to find which one.

**Property 4 — Service metrics count different things, and the count is a policy choice.** A
stock-out can be counted as a failed cycle, a missing unit, a missed order, a late delivery, or a
day without stock. Each count produces a different number from the same event. Choosing the
metric is therefore choosing what the organisation promises, which is a commercial decision, not
a measurement decision.

**Property 5 — Inventory has an age, and the age changes the type.** A unit starts as cycle
stock, becomes safety stock if it is not consumed, becomes excess when demand falls below plan,
becomes at-risk when its remaining shelf life or season runs short, and becomes obsolete when no
reasonable prospect of sale remains. The unit does not move. Its classification moves. An
inventory report that does not carry age cannot see this transition, which is why write-downs
arrive as surprises.

### 3.1 The taxonomy, with the decision that creates each type

| Type | Question it answers | Main cost it incurs | Decision that creates it | Owner role |
|---|---|---|---|---|
| Cycle stock | How much do we buy or build at once? | Carrying plus ordering or setup | Order quantity, lot size, minimum order | Replenishment planner or buyer |
| Safety stock | How much variability can we absorb? | Carrying; service risk if cut | Service target plus variability estimate | Inventory policy owner |
| Pipeline (in-transit) | How much is moving between nodes? | Capital in transit, freight, insurance | Mode, lane, Incoterm, transit time | Logistics or transport manager |
| Decoupling stock | Where do two stages stop waiting for each other? | Carrying at the buffer | Buffer placement between stages | Production or network planner |
| Anticipation (seasonal) | How much do we build before a known peak or shutdown? | Carrying, obsolescence if the peak misses | Build-ahead plan | S&OP or master scheduler |
| Hedge stock | How much do we buy before a price, tariff, or supply shock? | Price risk, storage, obsolescence | Forward-buy decision | Procurement with treasury |
| Maintenance and spare (MRO) | How much uptime are we insuring? | Carrying plus downtime risk | Criticality and asset policy | Maintenance and reliability |
| Consignment | Who owns stock sitting at the point of use? | Supplier capital, administration | Contract terms | Sourcing or commercial |
| Work in process (WIP) | How much work sits between operations? | Value added, cycle time, quality risk | Batch size and line design | Manufacturing engineering |
| Obsolete and at-risk | What will never sell, or soon will not? | Full write-down plus storage | Phase-out and write-down decision | Product management with finance |

Two rows deserve a note because they are routinely misclassified.

**Pipeline is not the same as in-transit.** In-transit inventory is goods physically moving under
a document. Pipeline inventory is all the inventory required by a lane's flow time, including
goods awaiting consolidation and goods awaiting receipt. The pipeline is the mean lead-time
demand; in-transit is the part visible on a tracking screen. Counting only the tracked part
understates the pipeline by the dwell at both ends, which is often half the total.

**Obsolete is not the same as excess.** Excess stock is more than the policy says you need, and
it may still sell. Obsolete stock has no reasonable prospect of sale or use. At-risk stock is
between the two: still sellable, but only at a discount, only within a window, or only if a
forecast that has been wrong before turns out right. The accounting treatment differs for each,
and so does the correct action. Step 12 of section 5 treats the transition.

### 3.2 Little's law, and what it forbids

Little's law states that for a stable system, the average number of units in the system equals
the arrival rate multiplied by the average time a unit spends in it:

```text
I = R x T
I = inventory (units)
R = throughput (units per week)
T = flow time (weeks)
```

The law is an identity, not an approximation, and it holds for any stable system regardless of
the distribution of arrivals or service times. Little's original proof is Little (1961); the
fiftieth-anniversary review (Little, 2011) is the readable version. It applies to a queue, a
factory, a distribution lane, a receivables ledger, and a hospital ward. It does not require
independence, Poisson arrivals, or any other modelling assumption. It requires stability.

Three consequences follow, and they are the reason the law belongs in a foundations topic.

**Consequence 1 — Inventory reduction is a time problem.** If throughput is fixed, a 20%
inventory reduction requires a 20% flow-time reduction. There is no third lever. A programme
that cuts inventory without touching time must be doing one of four things: cutting service,
shifting the stock to another party, liquidating a one-time surplus, or mis-measuring. Those are
the only options, and each has a cost that the inventory saving does not show.

**Consequence 2 — Lead-time reduction dominates lot-size reduction.** Cutting a week from a
three-week lead time removes a week of pipeline at full throughput, plus the safety stock that
the extra week of uncertainty required. Cutting the order quantity in half removes half the cycle
stock and nothing else. On the worked SKU in section 6, a one-week lead-time cut removes 24% of
total inventory. Halving the order quantity removes 11% and increases ordering cost by more than
the carrying saving.

**Consequence 3 — The law does not apply to dead stock.** Obsolete inventory has no flow, so it has
no finite flow time. It sits in `I` and contributes nothing to `R x T`. A flow time computed from
total inventory divided by throughput on a company with significant dead stock overstates `T` and
makes the operation look slower than it is. Strip the obsolete stock first, then apply the law.

> **Note:** Little's law requires a stable system measured over a period long enough for the
> averages to settle. A seasonal build is not stable: inventory rises for twelve weeks and falls
> for eight. Applying `I = R x T` inside a build season produces a flow time that describes the
> build, not the year. State the measurement window and check whether it contains a build.

### 3.3 The inventory-reduction trap

Most inventory programmes target a percentage reduction on total inventory, delegate it to
planners, and measure success at quarter end. The planners respond rationally. They cut the only
component they control directly: the order quantity. Cycle stock falls immediately, visibly, and
with a computable saving. Safety stock is a parameter nobody in the room owns, and cutting it
means accepting a lower service promise. Pipeline is a logistics budget. Obsolete stock needs a
finance decision and a write-down that hurts this quarter.

So the programme removes cycle stock, reports the saving, and leaves the other types untouched.
Order frequency, freight, and receiving costs then rise, and service either holds or degrades. If
it degrades, expediting begins, and expediting is the most expensive inventory in the company:
inventory bought at a premium after the need is known.

The diagnostic question is not "how much inventory do we hold?" It is "which type, created by
which decision, owned by whom?"

### 3.4 The service metric family in one table

| Metric | Unit of count | Question it answers | Typical owner |
|---|---|---|---|
| Cycle service level (CSL) | Replenishment cycle | How often does a cycle end without a stock-out? | Inventory planner |
| Fill rate | Demand units | What share of demanded units shipped from stock? | Operations or customer service |
| Ready rate | Time periods | In what share of periods was stock positive? | Operations |
| Stock-out rate | Events or SKUs | How often was an item unavailable? | Category or supply |
| Backorder rate | Order lines or units | What share of demand was deferred rather than lost? | Customer service |
| Lost-sales rate | Units or value | What share of demand disappeared? | Commercial |
| OTIF | Orders or order lines | Was the order complete and on time? | Key account or logistics |
| OTIF-D | Orders plus documents | Was it also documented correctly? | Compliance or finance |
| Perfect order | Orders | Was it on time, complete, undamaged, documented? | Cross-functional |

The metrics are not ranked; each is correct for a different promise. A retailer charging suppliers
cares about OTIF at the order line, a hospital about the ready rate of a critical item, and a
planner setting a reorder point about CSL, because CSL is what the safety-stock formula controls.
Using a customer-facing metric to set an internal policy parameter, or the reverse, is the second
most common error in this topic.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Inventory | Goods owned or controlled and held for sale, use, or transformation | Treated as a single homogeneous quantity | Units and currency |
| Cycle stock | Inventory that exists because orders arrive in lots larger than one unit | Treated as waste rather than a lot-size consequence | Units |
| Safety stock | Buffer held to cover variability in demand and lead time | Treated as a fixed policy constant, never reviewed | Units |
| Pipeline inventory | Inventory required by the flow time of a lane, including dwell | Confused with goods physically in transit | Units |
| In-transit inventory | Goods owned but not yet received at the destination node | Counted by shipment document rather than by ownership | Units |
| Decoupling stock | Buffer between two stages so each can run at its own rate | Confused with safety stock | Units |
| Anticipation stock | Stock built ahead of a known peak, promotion, or shutdown | Treated as excess when the peak is late | Units |
| Hedge stock | Stock bought ahead of an expected price, tariff, or supply event | Confused with speculative trading | Units and currency |
| MRO stock | Maintenance, repair, and operations spares held to protect asset uptime | Managed with production inventory policies | Units |
| Consignment stock | Goods held at a customer or user site but owned by the supplier until use | Counted as owned inventory, or ignored entirely | Units |
| Work in process (WIP) | Partially converted material between operations | Omitted from inventory reporting because it is "not finished" | Units, equivalent units |
| Obsolete stock | Inventory with no reasonable prospect of sale or use | Used loosely for anything old | Units and currency |
| At-risk stock | Inventory whose value is likely to decline before it can be sold | Reported as healthy because it is not yet written down | Units, currency, remaining shelf life |
| Inventory position | On-hand plus on-order minus backorders | Confused with on-hand | Units |
| Reorder point (ROP) | Inventory position at which a replenishment order is triggered | Set from lead-time demand alone, with no safety stock | Units |
| Order quantity | The lot size ordered or produced each cycle | Assumed to equal demand | Units |
| Lead-time demand | Demand expected during the replenishment lead time | Computed on the wrong time bucket | Units |
| Cycle service level (CSL) | Probability that a replenishment cycle ends without a stock-out | Reported to customers as service level | Proportion |
| Fill rate | Share of demanded units that ship from stock | Computed on value rather than units | Proportion |
| Ready rate | Share of periods in which stock on hand is positive | Confused with fill rate | Proportion |
| Lost-sales rate | Share of demand that disappears when stock is unavailable | Assumed to be zero in most models | Proportion |
| OTIF | On-time in-full: share of orders or lines complete and within the delivery window | Reported without stating date basis, grain, or exclusions | Proportion |
| OTIF-D | OTIF plus correct documentation (proof of delivery, labels, invoice) | The "D" is also used for "delivery" in some vendors' material | Proportion |
| Perfect order | Product of on-time, complete, damage-free, and documentation-accurate rates | Computed as an average of the four rates | Proportion |
| Days inventory outstanding (DIO) | Inventory value divided by daily cost of goods sold | Computed on revenue, which understates the days | Days |
| Days of supply | Inventory quantity divided by average daily demand | Computed on plan rather than actual demand | Days |
| Weeks of cover | Inventory divided by weekly demand | Applied to lumpy items where it is meaningless | Weeks |
| Shrink | Inventory loss from theft, damage, error, and fraud | Confused with obsolescence | Units and currency |
| Remaining shelf life | Time until an item's usable or saleable life ends | Confused with days on hand | Days |
| Net realisable value (NRV) | Estimated selling price less costs to complete and sell | Confused with market price | Currency per unit |
| Write-down | Reduction of inventory's carrying value to NRV | Confused with a markdown, which is a price change | Currency |
| Forecast | A probabilistic statement about the future | Treated as a commitment | Units with a distribution |
| Plan | A decision about what the company will do | Treated as a forecast | Units with a schedule |
| Target | An aspiration set by management | Treated as a plan | Units or percent |
| Commitment | A promise made to a customer or supplier | Treated as a forecast | Contract |

Four of these deserve expansion.

**CSL versus fill rate.** CSL counts replenishment cycles. Fill rate counts units. A cycle in
which the company runs out on the last day and misses three units is a failed cycle for CSL and a
99.7% fill rate if the cycle shipped 1,000 units. A cycle in which the company runs out early and
misses 400 units is also a failed cycle, and CSL cannot tell the two apart. That is the entire
divergence, and section 6 turns it into arithmetic.

**OTIF, OTIF-D, and perfect order.** OTIF is a joint test on time and completeness. OTIF-D adds a
documentation test. Perfect order usually adds damage. The critical detail is that these are
*joint* tests at the order or line level, not products of independent probabilities. Multiplying
`on-time% x in-full%` assumes the two are independent; when they are correlated (a late shipment
is often also an incomplete one), the product is wrong. Compute a flag per line and average the
flags.

**Write-down versus markdown.** A markdown is a price reduction. A write-down is an accounting
adjustment to carrying value. A company can mark down prices for a year without writing down
inventory, and write down inventory without changing any price.

## 5. The mechanics, step by step

This section builds a policy for one SKU, then extends it to a portfolio. The sequence matters:
each step depends on the one before, and skipping a step produces a number that looks precise and
is not.

### Step 1 — Classify the item before computing anything

Establish four attributes: demand pattern (smooth, seasonal, intermittent, lumpy), value per unit,
lifecycle position (new, growth, mature, declining, end-of-life), and criticality (what happens
if it is unavailable). The attributes select the policy. A lumpy, high-value, end-of-life item
should not receive the same treatment as a smooth, low-value, mature one, and no formula can
recover from applying the wrong policy to the wrong class.

### Step 2 — Fix the time bucket and the unit of measure

Choose the bucket (day, week, month) and state it everywhere. Demand variance depends on the
bucket: a weekly standard deviation is not seven daily standard deviations but the daily standard
deviation multiplied by the square root of seven, and only if demand is independent across days.
Mixing buckets is the most common source of a safety stock wrong by a factor of two.

Choose the unit: each, case, pallet, kilogram, litre. Then state the conversion. A policy
computed in cases and applied in eaches is wrong by the case pack, which is often 12 or 24.

### Step 3 — Estimate the demand distribution, not just the mean

For each item, estimate the mean and the variance over a window that excludes structural breaks.
Then test whether a normal distribution is defensible. Two diagnostics:

- **Coefficient of variation (CV) above roughly 0.5.** The normal distribution will place
  non-trivial probability on negative demand. If `sigma` exceeds `mu/2`, compute
  `Phi(-mu/sigma)` and look at the number. If it is above 5%, the normal is not describing the
  item.
- **Intermittency.** If more than about a quarter of periods have zero demand, the item is
  intermittent and the variance-to-mean ratio (VMR) is the relevant statistic. A VMR near 1 is
  consistent with a Poisson process; a VMR well above 1 indicates over-dispersion, for which the
  negative binomial is the standard alternative.

The practical version of section 9.4: a normal safety stock on a lumpy item systematically
under-protects, because the tail that causes stock-outs is longer than the normal allows.

### Step 4 — Estimate the lead-time distribution, not just the mean

Lead time is a random variable with a mean, a variance, and occasionally a bimodal shape (normal
transit versus expedited). Safety stock protects against the variability of demand *during* the
lead time, which combines both sources. If demand and lead time are independent:

```text
sigma_L = sqrt( L x sigma_d^2 + mu_d^2 x sigma_LT^2 )
```

where `L` is the mean lead time in periods, `sigma_d` the per-period demand standard deviation,
`mu_d` the per-period mean demand, and `sigma_LT` the lead-time standard deviation. Many
implementations use only the first term, correct only when lead time is deterministic. For an
imported item with a four-week nominal lead time and a two-week standard deviation, the second
term usually dominates.

### Step 5 — Choose the service metric and the target, and name the owner

This is a commercial decision, not an analytical one. Use the following branch.

| If the promise is... | Measure | Typical target | Consequence of the choice |
|---|---|---|---|
| "We almost never run out" (internal) | CSL | 90–98% by ABC class | Controls the safety-stock formula directly |
| "We ship almost every unit ordered" (customer) | Fill rate | 95–99% | Depends on lot size as well as buffer |
| "The item is on the shelf when the patient arrives" | Ready rate | 98–99.5% | Sensitive to review frequency |
| "The order arrives complete and on time" (retail) | OTIF or OTIF-D | Contractual, often 90–98% | Joint test; chargebacks attach |
| "The order is perfect" (premium channel) | Perfect order | 90%+ | Four joint tests multiply |

Write the target down with its metric name, its grain, and its date basis. "95% service" is not a
target; it is a word.

### Step 6 — Compute safety stock under the chosen distribution

Under a normal approximation with continuous review:

```text
SS = z x sigma_L
z  = inverse normal of the CSL
```

Under a fill-rate target, `z` is not the inverse normal of the fill rate. It solves:

```text
G(z) = (1 - beta) x Q / sigma_L
```

where `G` is the unit normal loss function, `G(z) = phi(z) - z x (1 - Phi(z))`, and `beta` is the
fill rate. There is no closed-form inverse; solve numerically. This is the single formula most
often implemented incorrectly, usually by substituting the fill rate into the CSL formula.

### Step 7 — Compute the reorder point or the order-up-to level

Continuous review: `ROP = mu_L + SS`. Periodic review: the order-up-to level covers the review
period plus the lead time, so `S = mu_(L+R) + z x sigma_(L+R)`. A continuous-review formula in a
periodic-review system under-protects by the review period, a month of exposure for a monthly
ordered item.

### Step 8 — Set the order quantity from the real constraints

The economic order quantity (EOQ) is a starting point, not an answer. In practice the quantity is
constrained by case pack, pallet, truckload, minimum order quantity, supplier production run,
shelf life, or a contractual commitment. Compute the EOQ, then find the binding constraint; the
quantity you use determines cycle stock, half the order quantity on average.

### Step 9 — Compute cycle stock and pipeline

Cycle stock averages `Q/2` under steady demand. Pipeline averages `mu_d x L`, the mean lead-time
demand, which is the same quantity as `mu_L` in the reorder point. The reorder point is therefore
pipeline plus safety stock, which is why a planner can describe it as "cover the lead time plus a
buffer" and be exactly right.

### Step 10 — Reconcile the four numbers

For each SKU, four quantities must agree: units on hand, value on the balance sheet, days of
cover, and flow time implied by Little's law. If they do not, one of the following is almost
always true: obsolete stock is in the balance but not the flow; consignment stock is present but
not owned; goods in transit are owned but not in the warehouse system; or the demand rate behind
cover is the plan rather than actual. Reconcile first: optimising unreconciled data optimises the
error.

### Step 11 — Measure realised service ex post and close the loop

Compute the realised metric from transaction data, not from the policy, and compare it with the
target. If realised CSL is materially below target, the variability estimate was too low, the
lead time was longer than modelled, or the review was not executed; each has a different fix. A
service number never measured against the policy that set it is a number nobody is accountable
for.

### Step 12 — Manage the lifecycle

Every item has a phase-in and a phase-out. At phase-in there is no history, so start conservative
and narrow as data accumulates. At phase-out, stop replenishing at a computed last-buy quantity
and start the at-risk clock. The last-buy quantity is not the forecast; it is the forecast plus
the committed minimum minus on-hand and on-order, and it is where obsolete stock is most often
created by arithmetic rather than by demand.

## 6. Worked example

Two SKUs at one distribution centre (DC) of a industrial parts distributor. All figures are
illustrative and internally consistent. Currency is euro (EUR); the time bucket is one week; the
review is continuous.

### 6.1 Parameters

| Parameter | SKU A (fast mover) | SKU B (lumpy spare) |
|---|---|---|
| Mean weekly demand | 1,000 units | 15 units |
| Weekly demand standard deviation | 250 units | 45 units |
| Demand pattern | Smooth, CV 0.25 | Zero in 90% of weeks, 150 in 10% |
| Lead time | 3 weeks, deterministic | 8 weeks, deterministic |
| Order quantity | 2,000 units (2 weeks) | 12 units (case pack) |
| Unit cost | EUR 40 | EUR 2,500 |
| Service target | 95% CSL | 95% CSL |
| Ownership in transit | Buyer owns from supplier dock (FCA) | Buyer owns from supplier dock (FCA) |

SKU B's demand pattern is worth checking: zero 90% of the time and 150 the other 10% gives a mean
of 15 and a standard deviation of `sqrt(0.1 x 150^2 - 15^2) = sqrt(2,250 - 225) = 45`. The
numbers are consistent, and the item is genuinely lumpy rather than merely low-volume.

### 6.2 SKU A under a 95% CSL target

```text
mu_L    = 1,000 x 3                = 3,000.0 units
sigma_L = 250 x sqrt(3)            =   433.0 units
z       = 1.645  (for 95% CSL)
SS      = 1.645 x 433.0            =   712.3 units
ROP     = 3,000.0 + 712.3          = 3,712.3 units
cycle   = 2,000 / 2                = 1,000.0 units
pipeline= 1,000 x 3                = 3,000.0 units
total   = 1,000.0 + 712.3 + 3,000  = 4,712.3 units
```

Value at EUR 40 per unit: `4,712.3 x 40 = EUR 188,492`. At a 25% annual carrying cost rate, that
is `EUR 47,123` per year. Annual demand is 52,000 units, so 26 orders per year; at EUR 250 per
order, ordering cost is `EUR 6,500`. Total relevant cost, excluding shortage, is `EUR 53,623` per
year.

Little's law check: `4,712.3 / 1,000 = 4.712 weeks` of flow time. The components agree: 3.0 weeks
of pipeline, 1.0 week of cycle stock (`Q/(2R)`), and 0.712 weeks of safety stock (`SS/R`).

Now compute the fill rate at this CSL:

```text
G(1.645) = phi(1.645) - 1.645 x (1 - 0.95)
         = 0.10311 - 0.08223
         = 0.02088

E[US]    = 433.0 x 0.02088          = 9.04 units short per cycle
fill     = 1 - 9.04 / 2,000         = 0.9955  (99.55%)
```

At 95% CSL, this SKU delivers a 99.55% fill rate. The safety stock prevents 95% of cycles from
ending short, and when a cycle does end short it is short by about nine units out of two
thousand. The customer barely sees it.

### 6.3 The same SKU under a 95% fill-rate target

Now invert the question. What safety stock does a 95% fill-rate target require?

```text
G(z) = (1 - 0.95) x 2,000 / 433.0   = 0.2309
z    = 0.399   (solved numerically)
CSL  = Phi(0.399)                   = 0.655  (65.5%)
SS   = 0.399 x 433.0                = 172.8 units
ROP  = 3,000 + 172.8                = 3,172.8 units
```

A 95% fill-rate target costs 173 units of safety stock instead of 712, and accepts stock-outs in
34.5% of cycles. Each stock-out averages about 100 units, which is 5% of a 2,000-unit cycle. The
balance-sheet saving is `539.5 x 40 = EUR 21,580`, and the annual carrying saving is about
`EUR 5,395`.

This is the counterintuitive result that most inventory training skips: **for a fast-moving item
with a large lot relative to variability, a fill-rate target is cheaper than the equivalent CSL
target.** The reason is that CSL is a conservative count when a cycle is large. One failed cycle
out of twenty sounds like good service, but if each failure costs only nine units, the customer
experiences 99.55% service.

### 6.4 SKU B, where the gap reverses and reaches 78%

SKU B is the case this topic exists to explain. Same target, 95% CSL.

```text
mu_L    = 15 x 8                   =  120.0 units
sigma_L = 45 x sqrt(8)             =  127.3 units
SS      = 1.645 x 127.3            =  209.4 units
ROP     = 120.0 + 209.4            =  329.4 units
cycle   = 12 / 2                   =    6.0 units
pipeline= 15 x 8                   =  120.0 units
total   = 6.0 + 209.4 + 120.0      =  335.4 units
```

Fill rate at that policy:

```text
E[US] = 127.3 x 0.02088            = 2.66 units short per cycle
fill  = 1 - 2.66 / 12              = 0.7785  (77.8%, reported as 78%)
```

**CSL 95%, fill rate 78%.** Both numbers come from the same safety stock, lead time, and demand
history. The mechanism is lot size. A cycle is 12 units, so a 2.66-unit shortfall costs 22% of the
cycle. Annual demand is 780 units over about 65 cycles; the expected shortfall accumulates to
roughly 173 units, or 22% of demand. The implied shortage in a failed cycle exceeds the order
quantity, which is a sign the normal loss approximation is straining; treat 78% as the metric gap,
not as a forecast of a particular shortage.

To hit a 95% fill rate on SKU B, the policy needs:

```text
G(z) = 0.05 x 12 / 127.3           = 0.00471
z    = 2.214
CSL  = 0.9866  (98.66%)
SS   = 2.214 x 127.3               = 281.9 units
```

So the direction of the gap is not a universal law; it depends on the ratio of order quantity to
lead-time demand variability. For a 95% target the metrics coincide at
`Q / sigma_L = G(1.645) / 0.05 = 0.4176`; below that CSL flatters, above it CSL understates.
SKU A has a ratio of 4.62, SKU B 0.094.

### 6.5 Two interventions, priced

Take SKU A and evaluate two proposals that a cost-reduction programme might make.

**Proposal 1: halve the order quantity to 1,000 units.** Cycle stock falls from 1,000 to 500, a
500-unit reduction worth `EUR 20,000` at cost, `EUR 5,000` per year in carrying at 25%. Orders
rise from 26 to 52 per year; at EUR 250 each, ordering cost rises by `EUR 6,500`. Net effect:
the company spends EUR 6,500 to save EUR 5,000, and reports the inventory reduction. Fill rate at
95% CSL falls to `1 - 9.04/1,000 = 99.10%`. This is the inventory-reduction trap in one
calculation: the saving is visible on the balance sheet and the cost is in a different budget.

**Proposal 2: cut the lead time from three weeks to two.** Pipeline falls from 3,000 to 2,000;
`sigma_L` falls to `250 x sqrt(2) = 353.6`; safety stock falls to 581.6; total inventory falls to
`2,000 + 581.6 + 1,000 = 3,581.6`, a reduction of 1,130.7 units (24.0%). The saving is
`EUR 45,228` at cost and `EUR 11,307` per year in carrying, and it is permanent. It requires
commercial work with the supplier, which is why it is less popular than proposal 1.

The ranking is instructive: the intervention with the largest saving is the one that changes
time, not the one that changes quantity.

## 7. Code, query, or calculation

One compact implementation, because the arithmetic in section 6 is the point and the code is only
a way to repeat it without transcription error.

```python
import numpy as np
from scipy.stats import norm

def policy(mu_w, sd_w, L, Q, csl=0.95, unit_cost=40.0, carry=0.25):
    mu_l, sd_l = mu_w * L, sd_w * np.sqrt(L)
    z = norm.ppf(csl)
    ss = z * sd_l
    loss = norm.pdf(z) - z * (1 - csl)        # unit normal loss G(z)
    fill = 1 - sd_l * loss / Q
    total = Q / 2 + ss + mu_l                 # cycle + safety + pipeline
    return dict(ss=round(ss, 1), rop=round(mu_l + ss, 1), fill=round(fill, 4),
                flow_weeks=round(total / mu_w, 2),
                carry_eur=round(total * unit_cost * carry))

print(policy(1000, 250, 3, 2000))   # ss 712.3, fill 0.9955, 4.71 weeks, EUR 47123
print(policy(15, 45, 8, 12, unit_cost=2500))   # ss 209.4, fill 0.7785, 22.36 weeks
```

Expected output matches section 6: `fill 0.9955` for SKU A and `fill 0.7785` for SKU B. To solve
the inverse problem (a fill-rate target, unknown `z`), replace the closed form with a one
dimensional root find on `G(z) - (1 - beta) * Q / sd_l`. Do not pass `beta` to `norm.ppf`; that
is the error the function is written to expose.

## 8. Data, standards, and regulatory context

Inventory and service measurement are governed by a mixture of accounting standards, voluntary
identification standards, private contracts, and benchmarking conventions. Knowing which is which
tells you how much freedom you have.

### 8.1 The governing instruments

| Instrument | Governs | What it constrains | Grade |
|---|---|---|---|
| IAS 2 *Inventories* | International accounting | Measurement at the lower of cost and net realisable value; write-down and reversal | B |
| ASC 330 *Inventory* | United States accounting | Measurement rules including lower of cost or market for relevant methods | B |
| Incoterms 2020 | Contract convention | The point where cost, risk, and usually title transfer | B |
| GS1 GTIN | Product identification | The trade item that is ordered, shipped, and counted | B |
| GS1 SSCC | Logistics unit identification | The pallet or case identity used in receiving and service measurement | B |
| GS1 GLN | Location identification | Party and place identity across order and event data | B |
| EPCIS 2.0 and CBV 2.0 (ISO/IEC 19987, 19988:2024) | Event data | What happened to which object, where, when, and why | B |
| Retailer compliance programmes | Private contract | OTIF targets, windows, chargebacks, dispute rules | C |
| APQC Open Standards Benchmarking | Voluntary benchmark | Metric definitions and peer medians | C |

Two observations follow. Only the accounting standards and the Incoterms have legal force in the
ordinary sense; the GS1 standards are voluntary but contractually required, which makes them
effectively mandatory in retail. And the retailer programmes are contracts, not regulations: they
bind because the supplier signed, and they change when the retailer decides, as Walmart's OTIF
target did in 2024.

### 8.3 Retailer service measurement, as it stands in 2025–2026

Retail compliance programmes define service in ways that differ from the textbook metrics, and
the differences are material.

| Programme | On-time requirement | In-full requirement | Penalty | Grade |
|---|---|---|---|---|
| Walmart, prepaid suppliers | 90% on time | 95% in full | 3% of cost of goods sold per non-compliant case | C |
| Walmart, collect suppliers | 98% collect-ready | 95% in full | As above | C |
| Walmart historical target | 98% OTIF from September 2020 | Same | Reduced effective 1 February 2024 | C |
| Target perfect order | 100% on time | 95% fill rate | Up to 5% of cost of goods sold | C |

Three details change the meaning of these numbers. Walmart measures against a must-arrive-by date
(MABD) window that varies by department, and an early arrival counts as a failure as well as a
late one. Partial shipments fail the in-full test. Scores are published weekly in the retailer's
supplier portal, and disputes usually must be raised within about ninety days. A supplier
reporting "98% OTIF" in 2025 is therefore either quoting a target that was reduced in 2024,
quoting the collect-ready metric, or using its own definition.

> **Note:** The commonly repeated claim that Walmart requires 98% OTIF is out of date. The target
> was reduced to 90% on-time and 95% in-full for prepaid suppliers effective 1 February 2024,
> while the 98% figure survives as the collect-ready requirement. If a document you are reading
> quotes 98% without this qualification, check its date.

Target's programme uses a different structure again: a 95% fill-rate target and a 100% on-time
target, with fines of up to 5% of cost of goods sold and additional charges for advance ship
notice and barcode accuracy introduced in May 2025. The two retailers are measuring different
things with similar-sounding language, which is why a supplier's "service level" cannot be
compared across customers without the programme definition.

### 8.4 What changed in GS1 and what did not

A search of GS1's published standards and 2025 releases found no change to retail service
measurement definitions. The relevant 2025 movement was elsewhere:

- EPCIS and CBV 2.0 are aligned with ISO/IEC 19987 and 19988:2024 (third edition), adding a
  sensor dimension, new business steps including sampling and sensor reporting, and additional
  dispositions.
- GS1 US published EPCIS recommendations for Food Safety Modernization Act section 204 critical
  tracking events, release 2.0, in May 2025.

Neither redefines OTIF, fill rate, or perfect order. If a trading partner tells you that GS1 has
redefined service measurement, ask for the document. The likely answer is that the partner has
changed its own programme and attributed the change to a standard.

### 8.5 Benchmark definitions are conventions, not standards

APQC, formerly the American Productivity and Quality Center, publishes open standards benchmarks
that many companies use as peer comparisons. Two of its definitions matter here:

- **Inventory days of supply** is average inventory at standard cost divided by annual cost of
  goods sold over 365, or equivalently 365 divided by the inventory turns rate.
- **Perfect order performance** is the product of four rates: on-time delivery, complete orders,
  damage-free orders, and accurate documentation.

The perfect-order definition is a good example of a convention that looks like a standard. APQC's
own illustration multiplies `0.98 x 0.97 x 0.99 x 0.82 = 77.16%`. The arithmetic is correct. The
modelling assumption behind it, that the four dimensions are independent, is an assumption and
not a fact. Section 10 treats the consequences.

## 9. Industry practice and case evidence

### 9.1 The 2024–2026 inventory cycle

The environment in which this topic is being read matters, because it determines which
inventory arguments are socially acceptable. The pandemic build-up, the 2022–2024 destocking, and
the 2025 tariff frontloading produced three different inventory problems in five years.

| Indicator | Reading | Date | Source | Grade |
|---|---|---|---|---|
| Total business inventory-to-sales ratio | 1.37, down from 1.40 a year earlier | November 2025 | U.S. Census Bureau, MTIS | B |
| Manufacturers inventory-to-sales ratio | 1.56, down from 1.58 | November 2025 | U.S. Census Bureau, MTIS | B |
| Retailers inventory-to-sales ratio | 1.28, down from 1.31 | November 2025 | U.S. Census Bureau, MTIS | B |
| Wholesalers inventory-to-sales ratio | 1.28, down from 1.32 | November 2025 | U.S. Census Bureau, MTIS | B |
| Total business inventories | USD 2,678.3bn, up 1.2% year on year | November 2025 | U.S. Census Bureau, MTIS | B |
| Manufacturers inventory-to-sales, not adjusted | 1.63, against 1.50 in September | November 2025 | FRED series MNFCTRIRNSA | B |
| Eurozone firms reporting finished goods "too large" | Lowest share since March 2022 | June 2025 | ING research note | D |
| World merchandise trade growth | 0.9% in 2025; 1.8% forecast for 2026, cut from 2.5% | August 2025 | WTO | B |
| United States inventory drawdown | About USD 128bn between Q2 2025 and Q2 2026 | September 2026 | SMBC research note | D |
| Retail container imports | Record 2.47m TEU forecast for July 2026 | July 2026 | NRF forecast via FreightWaves | C |

Read the table carefully, because two rows will mislead you if combined. The Census ratio of 1.37
is seasonally adjusted; the FRED manufacturing series at 1.63 is not, and its path from 1.50 in
September contains seasonal movement as well as real change. Do not subtract one from the other.
Use one series, state its adjustment, and compare like with like.

The substantive story is that the aggregate overhang was largely worked off by late 2025. The WTO
attributed 2025 trade resilience partly to United States import frontloading and inventory
accumulation, and warned that the stock would "have to be unwound at some point." The 2026
picture is a drawdown followed by cautious restocking, with tariff timing as the swing factor; a
record import month in July 2026 is frontloading again, so the cycle has changed phase rather
than ended.

### 9.2 Write-down evidence from 2025

| Company | Event | Stated cause | Grade |
|---|---|---|---|
| American Eagle Outfitters | USD 75m inventory write-down, May 2025; revenue down 5% to USD 1.1bn; operating loss about USD 85m | Merchandising missteps, promotions, excess spring and summer stock | C |
| Puma | Discounting to clear high inventory after front-loading shipments ahead of tariffs; planned United States price increases for Q4 2025 | Tariff-driven frontloading plus falling sales | C |
| Macy's | Markdowns on early spring goods, order cancellations and delays, full-year EBITDA guidance cut | Tariff cost uncertainty | C |
| Abercrombie & Fitch | Deliberate clearance to reset Abercrombie-brand inventory in Q2 2025; about USD 5m tariff impact and 170 basis points of annual margin pressure | Excess inventory plus tariffs | C |
| Multiple retailers | Stockpiling ahead of tariffs, with holiday markdown risk if demand weakens | Frontloading | E |

The pattern is consistent across the cases: frontloading creates hedge stock, demand softens or
the tariff lands differently from the plan, and the hedge becomes at-risk stock within one season.
The write-down is the accounting recognition of a classification that changed months earlier. An
organisation that had measured remaining shelf life and sell-through rate would have seen the
transition. One that measured days on hand would not, because days on hand can look healthy while
sell-through collapses.

### 9.3 Benchmark levels and what they are worth

| Benchmark | Value | Sample | Source | Grade |
|---|---|---|---|---|
| Inventory days of supply, median | 45.6 days | 6,951 companies | APQC Open Standards Benchmarking | C |
| Cash-to-cash cycle time, median | 50.0 days (25th 38.0, 75th 65.0) | 15,617 companies | APQC Open Standards Benchmarking | C |
| Inventory carrying cost, median | 10.0% of average inventory value | 6,468 companies | APQC Open Standards Benchmarking | C |
| Forecast accuracy by value shipped | 95% top, 88% median, 80% bottom performers | Not stated in the public summary | APQC | C |

These are self-reported member data, which means the sample over-represents large firms with
benchmarking programmes and the definitions are applied by the respondents. Use them to locate a
company in a range, not to set a target. The carrying-cost figure needs caution: 10.0% is a
realised cash-outflow measure, while the 20–25% used in planning models embeds a full cost of
capital and an obsolescence allowance. Eightx (2025, a vendor compilation) puts the planning
convention at 20–30% and perishable food at 30–40%.

The Institute of Business Forecasting and Planning publishes a forecast-error benchmark report,
updated in 2025, with breakdowns by industry, SKU level, and horizon. The figures sit behind
membership or purchase. This curriculum does not quote them, because a number that has not been
read is a number that cannot be cited. If you need the benchmark, obtain the report.

### 9.4 Evidence on distribution choice for slow movers

The theoretical claim in section 5, step 3 has empirical support, and it is worth knowing the
shape of it.

- Rossetti and Ünlü (2011) evaluated normal, gamma, Poisson, and negative binomial distributions
  for lead-time demand under intermittent demand with `(r, Q)` policies. Gamma and negative
  binomial performed best for high-variability items. They recommend the negative binomial when
  the variance-to-mean ratio exceeds about 1.1, and Poisson only when it is close to 1.
- Persson and colleagues (2017) simulated safety stocks for intermittent demand and found that
  analytical formulas based on the normal distribution systematically under-estimated the stock
  needed. Policies calculated for a 95% target produced realised service levels in the range of
  roughly 50% to 75%.
- Snyder, Ord, and Beaumont (2010) found average variance-to-mean ratios of about 1.9 to 2.3 in
  slow-moving item data, and an unrestricted negative binomial outperformed Poisson and Croston's
  method on predictive likelihood.

The practical implication is that a normal safety stock on a lumpy item is a known under-estimate,
and a CSL target will not be met. SKU B in section 6 is the diagnostic: with mean lead-time demand
120 and standard deviation 127, the normal places about 17% probability on negative demand, which
is impossible. That number alone should stop the calculation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Cutting cycle stock and calling it waste reduction | Inventory falls, order count rises, freight and receiving costs rise | Lot size is the only lever the planner controls directly | Saving booked in one budget, cost in another | Price ordering and freight into the proposal |
| Reporting CSL as fill rate | Planner reports 95%, customers report shortages | The two metrics count different units | Service promise quietly broken | Publish both, with grain and denominator |
| Fill rate computed on value | Expensive slow movers dominate the number | Value weights units by price | Fast, cheap items look better served than they are | Compute on units and on lines |
| OTIF as a product of on-time and in-full | Number is stable and slightly optimistic | Assumes independence of two correlated events | Wrong baseline for chargeback exposure | Compute a joint flag per line |
| OTIF measured to the wrong date | Internal score exceeds the customer's score | Confirmed date versus requested or must-arrive-by date | Chargebacks arrive as a surprise | State date basis and reconciliation rule |
| Pipeline excluded from inventory position | Recurring phantom shortages and duplicate orders | Ownership in transit not modelled | Over-ordering and expediting | Model ownership by Incoterm per lane |
| Weeks of cover applied to lumpy items | "Twelve weeks of cover" on an item ordered every ten weeks | Cover is a ratio; lumpy demand is not continuous | False comfort, then a stock-out | Use lead-time demand distribution |
| Normal approximation on slow movers | Safety stock set, service target still missed | Right-skewed, over-dispersed demand | Systematic under-protection | Test CV and VMR; use gamma or negative binomial |
| Ageing measured in days on hand | Old stock looks young because demand is low | Days on hand conflates age with movement | Write-downs arrive late | Track batch age and remaining shelf life |
| Write-down deferred to year end | At-risk bucket grows quarter after quarter | Nobody wants the current-period charge | Larger charge later, worse decisions meanwhile | Review at-risk monthly with finance |
| Record inaccuracy tolerated | Cycle counts show persistent variance | Receiving lag, location errors, unrecorded scrap | Policy computed on a false position | Measure record accuracy before optimising |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Demand is stationary over the estimation window | Mean and variance are estimated from history | Promotions, price changes, seasonality, new competition | Plot mean and variance by period; test for breaks |
| Lead time is independent of order quantity | Safety stock uses a fixed lead-time distribution | Capacity constraints, congestion at peak | Compare lead time by order size and season |
| Shortages are backordered and later filled | The loss-function fill rate assumes units are deferred | Lost sales in retail and e-commerce | Compare unfulfilled orders with later repeat demand |
| Demand arrives in unit-sized transactions | The standard loss function assumes single-unit demand | Pallet or case orders cause undershoot at the reorder point | Compare order sizes with the review increment |
| The review policy matches the formula | Continuous and periodic formulas differ by the review period | Monthly ordering with a continuous-review formula | Check the actual ordering calendar |
| The service target is exogenous | The formula takes the target as given | Target should be derived from cost of shortage | Compute the implied cost of service |
| Inventory records are accurate | Every calculation uses the recorded position | Receiving lag, misplacement, unrecorded scrap | Cycle-count a sample and measure variance |
| Demand is observed, not censored by stock-outs | History must reflect true demand | Sales lost during stock-outs vanish from the data | Reconstruct demand during stock-out periods |

Two limits follow. The taxonomy is analytical, not accounting: the ledger will not reproduce the
split, and a reconstruction is a method, not an audited figure. And the metrics count events, not
consequences: 78% fill on a line-stopping spare part is a different business event from 78% on a
substitutable fashion item. Shortage cost is a separate model, treated in Module 06.

## 12. Exercises

### Exercise 1 — Classify the stock (recall and application, 30 minutes)

Take ten inventory lines from any company you can describe, or use these: 2,000 units of a
seasonal item bought for a December promotion; 400 units on a vessel from Asia; 150 units of a
spare part last consumed fourteen months ago; 5,000 units of a stable fastener; 80 units of a
discontinued colour; 300 units held at a customer's site under a supplier-owned agreement; 1,200
units of a raw material bought ahead of a announced price increase; 60 units of WIP between two
press operations; 900 units with 45 days of remaining shelf life; 25 units of a critical
production input with one week of cover.

**Deliverable:** a table with the type, the decision that created it, and the owning role for
each line.
**Acceptance criteria:** at least six types are used; each line names a role rather than a
department; one line is ambiguous and the ambiguity is stated with the evidence that would
resolve it.

### Exercise 2 — Little's law (calculation, 25 minutes)

A DC ships 12,000 units per week and holds 60,000 units of flowing inventory. Compute the flow
time. The transport team proposes a lane change that removes one week of transit with throughput
unchanged. Compute the inventory reduction and the annual carrying saving at EUR 25 per unit and a
22% rate.

**Deliverable:** flow time in weeks, the new inventory level, the unit and currency saving.
**Acceptance criteria:** units shown at each step; the assumption of constant throughput stated;
the answer distinguishes the pipeline reduction from any safety-stock effect.

### Exercise 3 — Safety stock and fill rate (calculation, 45 minutes)

A SKU has mean weekly demand 400 units, weekly standard deviation 120 units, lead time 4 weeks,
order quantity 1,600 units, and a 97.5% CSL target. Compute lead-time demand, its standard
deviation, safety stock, reorder point, and the resulting fill rate. Then compute the safety stock
required for a 98% fill-rate target, and compare.

**Deliverable:** the full calculation with units, and a one-paragraph interpretation of the
difference.
**Acceptance criteria:** correct `z` for CSL (1.96); correct loss-function use for fill rate;
both safety stocks in units; the comparison states which target is more expensive and why.

### Exercise 4 — Explain the 95/78 gap (diagnosis, 30 minutes)

Using the SKU B numbers from section 6.4, write an explanation for a commercial director of why
the planner reports 95% service and the customer experiences 78%. State the single ratio that
drives the gap and three changes that would close it, with the cost or constraint attached to
each.

**Deliverable:** one page, no formulas beyond arithmetic.
**Acceptance criteria:** the ratio `Q / sigma_L` is identified; all three changes are feasible
and one is rejected with a reason; the explanation does not blame the planner.

### Exercise 5 — Metric audit (critique, 45 minutes)

A supplier reports 96% OTIF. You learn that it measures against its own confirmation date, at
order level, includes partial shipments as in-full if the remainder ships within five days, and
excludes intercompany orders. Reconstruct what the customer's experience might be, and state the
direction of the bias in each of the four choices.

**Deliverable:** a metric dictionary entry for the supplier's definition and for the likely
customer definition, plus a bias table.
**Acceptance criteria:** numerator, denominator, date basis, grain, and exclusions stated for
both; each of the four choices has a direction and a mechanism; the estimated gap is a range, not
a point.

### Exercise 6 — Ageing and write-down (calculation and judgement, 40 minutes)

A batch of 1,000 units cost EUR 30 each. Of these, 400 units are more than 180 days old with
remaining shelf life of 60 days. Expected realisable value is EUR 12 per unit after selling costs.
Carrying cost is 25% per year. Compute the write-down, the annual carrying cost of holding the
batch, and the break-even probability of selling at full price that would justify holding rather
than disposing now.

**Deliverable:** write-down in currency, carrying cost in currency per year, and a break-even
probability.
**Acceptance criteria:** NRV used correctly; the break-even accounts for the time value of the
delay; the recommendation states who must approve it.

### Exercise 7 — Policy for a lumpy item (design, 60 minutes)

Design a replenishment policy for an item with mean weekly demand 8 units, standard deviation 30
units, lead time 10 weeks, unit cost EUR 900, and a customer who will accept backorders. Choose
the distribution, the service metric, the order quantity, and the review frequency.

**Deliverable:** a policy specification with parameters and justifications.
**Acceptance criteria:** the distribution choice is justified against the normal using CV and
VMR; the service metric matches the customer's actual tolerance; the order quantity states its
binding constraint; the policy states how it will be validated after three months.

### Exercise 8 — Pipeline ownership (application, 30 minutes)

A lane carries 1,000 units per week with a three-week transit. Unit cost is EUR 40. Compute the
pipeline value. State who holds it on the balance sheet under FCA origin and under DDP
destination, and compute the effect on the buyer's DIO if annual cost of goods sold is EUR 104m
and other inventory is EUR 12m.

**Deliverable:** pipeline value, ownership statement for each Incoterm, and DIO under each.
**Acceptance criteria:** the DIO difference is computed, not asserted; the ownership statement
names the point of transfer; one sentence on why the commercial term matters to the metric.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
flow time    = 60,000 / 12,000            = 5.0 weeks
transit cut  = 1 week
inventory    = 12,000 x 4.0               = 48,000 units
reduction    = 12,000 units
saving       = 12,000 x 25 x 0.22         = EUR 66,000 per year
```

The saving is entirely pipeline, assuming the safety stock does not change. If the lane change
also reduces lead-time variability, safety stock falls as well and the total saving is larger. If
the change increases variability, the safety-stock effect offsets part of the pipeline saving.
State which assumption you used.

**Exercise 3 arithmetic.**

```text
mu_L    = 400 x 4                  = 1,600.0 units
sigma_L = 120 x sqrt(4)            =   240.0 units
z       = 1.96 for 97.5% CSL
SS      = 1.96 x 240               =   470.4 units
ROP     = 1,600 + 470.4            = 2,070.4 units
G(1.96) = 0.05844 - 1.96 x 0.025   = 0.00944
E[US]   = 240 x 0.00944            = 2.266 units per cycle
fill    = 1 - 2.266 / 1,600        = 0.99858  (99.86%)
```

For a 98% fill-rate target:

```text
G(z) = 0.02 x 1,600 / 240          = 0.13333
z    = 0.746  (solved numerically)
SS   = 0.746 x 240                 = 179.0 units
CSL  = Phi(0.746)                  = 0.772  (77.2%)
```

The 97.5% CSL target costs 470 units of safety stock; the 98% fill-rate target costs 179, because
the lot is 1,600 units against a lead-time standard deviation of 240. The company is buying
protection it does not need if the promise is unit-based. A common wrong answer uses `z = 2.054`,
the inverse normal of 0.98, giving 493 units and the wrong conclusion.

**Exercise 4 model answer.**

The gap is driven by `Q / sigma_L = 12 / 127.3 = 0.094`. A cycle is twelve units against a
lead-time demand standard deviation of 127, so a shortage that is small in absolute terms is large
relative to the cycle. Three changes would close it: increase the order quantity to roughly 500
units, which reduces exposure frequency at the cost of 488 units of cycle stock; cut the lead time
from eight weeks to two, which reduces `sigma_L` to 63.6 and improves both metrics; or change the
customer promise to a CSL or ready-rate commitment. Rejecting the first change is defensible if
shelf life or cash constrains the lot size, but the reason must be stated.

**Exercise 5 model answer.**

Four biases, all in the supplier's favour. Measuring against the confirmation date rather than the
requested date removes the supplier's own acceptance delay from the clock. Order-level measurement
lets one late line hide inside a large order, or vice versa. Counting a partial shipment as
in-full after five days converts a failure into a success with a lag. Excluding intercompany
orders removes the shipments the supplier controls most directly. A plausible range for the
customer's experience is 86% to 92%; the correct answer is a range with reasoning, not a point.

**Exercise 6 model answer.**

Write-down is `400 x (30 - 12) = EUR 7,200` on the aged portion; the remaining 600 units are
carried at cost. Annual carrying cost of the batch is `1,000 x 30 x 0.25 = EUR 7,500`, of which
the aged portion is EUR 3,000. Holding the aged units sixty more days costs about
`3,000 x 60/365 = EUR 493` plus the risk of zero recovery. Break-even probability solves
`p x 30 = 12 + 493/400`, giving `p = 44%`. Below that, disposal now is the better financial
decision. The write-down needs finance approval; disposal usually needs commercial approval too.

**Exercise 7 rubric.**

A strong answer rejects the normal: mean 8 against standard deviation 30 gives a CV of 3.75, so
the normal places substantial probability on negative demand. It uses a negative binomial, a
compound Poisson model, or simulation. It chooses fill rate or backorder rate rather than CSL,
because the customer accepts backorders. It notes that a ten-week lead time at a mean of 8 implies
a pipeline of 80 units, which dominates cycle stock, so lead-time reduction is the highest-value
intervention. It states the binding constraint on order quantity and a validation plan.

**Exercise 8 arithmetic.**

```text
pipeline value = 1,000 x 3 x 40            = EUR 120,000
annual COGS    = EUR 104,000,000; daily COGS = 284,931
DIO, buyer owns (FCA): (12,000,000 + 120,000) / 284,931 = 42.5 days
DIO, seller owns (DDP): 12,000,000 / 284,931            = 42.1 days
difference     = 0.4 days
```

The difference is small because the pipeline is small relative to total inventory; where most
inventory is imported goods in transit, it can reach several days.

## 14. Interview questions

**1. "What kinds of inventory does a company hold, and why does each exist?"**

Strong answer contains the taxonomy with a decision attached to each type: cycle from lot size,
safety from variability, pipeline from transit, anticipation from a known peak, hedge from a price
or supply event, and obsolete from a lifecycle that ended. Weak answer reveals a list of synonyms
("raw material, WIP, finished goods") with no owner or mechanism.

**2. "Can a company have 95% cycle service level and a 78% fill rate?"**

Strong answer contains yes, the mechanism (CSL counts cycles, fill rate counts units), and the
ratio that governs the gap, `Q / sigma_L`. Weak answer reveals a belief that the two are
interchangeable or that one of the numbers must be wrong.

**3. "A CFO wants inventory down 20% by year end. What do you ask?"**

Strong answer contains which type, which owner, and which lever: lot size, lead time, service
target, or disposal. It applies Little's law to show that a permanent cut needs a flow-time cut.
Weak answer reveals an immediate list of SKUs to cut, with no distinction between safety and
cycle stock.

**4. "How does lead-time reduction compare with lot-size reduction?"**

Strong answer contains the arithmetic: lead time removes pipeline at full throughput plus the
safety stock that the uncertainty required, while lot size removes only `Q/2` and raises ordering
cost. Weak answer reveals a preference for whichever is easier to implement, without pricing
either.

**5. "How would you set safety stock for a slow-moving item?"**

Strong answer contains a distribution test (CV, VMR, proportion of zero periods), a rejection of
the normal where over-dispersed, and a gamma or negative binomial or simulation approach. Weak
answer reveals `z x sigma x sqrt(L)` applied without checking whether the normal fits.

**6. "Why is OTIF a problematic single metric?"**

Strong answer contains the definitional switches: grain, date basis, tolerance window, exclusions,
and whether it is a joint test or a product. Weak answer reveals a belief that OTIF is standard
across retailers.

**7. "What is the difference between excess, at-risk, and obsolete stock?"**

Strong answer contains sellability and time: excess is above policy but sellable, at-risk is
sellable within a window at a discount, obsolete has no reasonable prospect of sale. Weak answer
reveals that the three are used as synonyms for "old."

**8. "When should inventory be written down?"**

Strong answer contains the accounting rule (lower of cost and net realisable value), the fact that
NRV is an estimate, and the operational triggers: remaining shelf life, sell-through collapse,
phase-out, and contractual returns. Weak answer reveals "at year end, when finance asks."

**9. "What is pipeline inventory and who owns it?"**

Strong answer contains the definition (flow time times throughput, including dwell), the
distinction from tracked in-transit goods, and the role of the Incoterm in determining ownership.
Weak answer reveals that pipeline is assumed to be the same as goods on a truck.

**10. "How would you validate an inventory position before optimising on it?"**

Strong answer contains reconciliation of units, value, cover, and implied flow time, plus tests
for obsolete stock, consignment, goods in transit, and censored demand. Weak answer reveals a
check of row counts and nulls only.

## 15. Key terms

- **Cycle stock** — inventory created by ordering or producing in lots; averages `Q/2`.
- **Safety stock** — buffer against demand and lead-time variability.
- **Pipeline inventory** — inventory required by a lane's flow time; equals mean lead-time demand.
- **Decoupling stock** — buffer that lets two stages run at different rates.
- **Anticipation stock** — build-ahead for a known peak, promotion, or shutdown.
- **Hedge stock** — forward buy against price, tariff, or supply risk.
- **Consignment stock** — supplier-owned goods at the point of use.
- **WIP** — work in process between operations.
- **Obsolete stock** — no reasonable prospect of sale or use.
- **At-risk stock** — value likely to decline before sale.
- **Inventory position** — on-hand plus on-order minus backorders.
- **Reorder point** — inventory position that triggers replenishment.
- **CSL** — probability a replenishment cycle ends without a stock-out.
- **Fill rate** — share of demanded units shipped from stock.
- **Ready rate** — share of periods with positive stock.
- **OTIF** — on-time in-full, definition-dependent.
- **OTIF-D** — OTIF plus correct documentation.
- **Perfect order** — on time, complete, undamaged, and correctly documented.
- **DIO** — inventory value divided by daily cost of goods sold.
- **Little's law** — `I = R x T` for a stable system.

## 16. Sources

**Peer-reviewed and foundational**

- Little (1961), "A Proof for the Queuing Formula: L = λW," *Operations Research* 9(3), 383–387,
  DOI 10.1287/opre.9.3.383; and Little (2011), "Little's Law as Viewed on Its 50th Anniversary,"
  *Operations Research* 59(3), 536–549.
- Rossetti and Ünlü (2011), "Evaluating the Lead Time Demand Distribution for (r, Q) Policies
  Under Intermittent Demand." https://cavern.uark.edu/~rossetti/conference_papers/iie281.pdf
- Persson et al. (2017), "Using Simulation to Determine the Safety Stock Level for Intermittent
  Demand," Winter Simulation Conference. https://www.informs-sim.org/wsc17papers/includes/files/318.pdf
- Snyder, Ord and Beaumont (2010), "Forecasting the Intermittent Demand for Slow-Moving Items."
  https://www2.gwu.edu/~forcpgm/2010-003.pdf

**Standards, official statistics, and regulatory**

- IAS 2 *Inventories* (IFRS Foundation); ASC 330 *Inventory* (FASB); Incoterms 2020 (ICC).
- GS1, EPCIS and CBV Implementation Guideline, EPCIS 2.0 and CBV 2.0 aligned with ISO/IEC 19987
  and 19988:2024. https://www.gs1.org/standards/epcis-and-cbv-implementation-guideline/current-standardd
- GS1 US, "EPCIS Recommendations for FSMA 204 Critical Tracking Events," Release 2.0, May 2025.
- U.S. Census Bureau, "Manufacturing and Trade Inventories and Sales, November 2025."
  https://www2.census.gov/mtis/historical/mtis2511.pdf
- FRED series MNFCTRIRNSA. https://fred.stlouisfed.org/series/MNFCTRIRNSA/
- WTO, "Frontloading, measured responses cushion tariff impact in 2025 but risk high for 2026,"
  8 August 2025. https://www.wto.org/english/news_e/news25_e/tfore_08aug25_e.htm

**Company, investor, and benchmark sources**

- APQC Open Standards Benchmarking, inventory days of supply.
  https://www.apqc.org/resources/benchmarking/open-standards-benchmarking/measures/inventory-days-supply
- APQC, cash-to-cash cycle time. https://www.apqc.org/resources/benchmarking/open-standards-benchmarking/measures/cash-cash-cycle-time-days
- APQC, perfect order performance. https://www.apqc.org/resources/benchmarking/open-standards-benchmarking/measures/perfect-order-performance
- APQC, inventory carrying cost. https://www.apqc.org/what-we-do/benchmarking/open-standards-benchmarking/measures/inventory-carrying-cost-percentage
- Institute of Business Forecasting and Planning, "Benchmarking Forecast Errors," 2025 update;
  figures gated and not quoted here. https://ibf.org/knowledge/reports/research-report13
- Walmart supplier compliance, as reported by Forbes (1 February 2024) and 5G Sales (2025).
  https://www.forbes.com/sites/stevebanker/2024/02/01/walmart-lowers-its-shipping-requirements-but-suppliers-still-struggle/
- Target compliance programme, SPS Commerce help centre.
  https://help.supplypike.com/en/articles/8300994-understanding-target-s-compliance-program

**Vendor and analyst material (Grade D unless stated)**

- Gartner, "Reduce Obsolete Inventory by Addressing 2 Key Root Causes," 18 February 2025
  (https://www.gartner.com/en/documents/6184755); "Beyond Assessment: A CSCO Guide to Sustained
  Inventory Health," 22 January 2026 (https://www.gartner.com/en/documents/7360130).
- Eightx, "Inventory Carrying Cost by Vertical: 2025 Benchmarks."
  https://eightx.co/blog/average-inventory-carrying-cost-by-vertical
- Cart.com, "Inventory management metrics," May 2025. https://cart.com/blog/inventory-management-metrics
- Kvantis Atlas, "Service level and fill rate."
  https://www.kvantisgroup.com/en/atlas/topics/service-rate-fill-rate.html
- Orkestra, "OTIF: Meaning, Formula, and How to Improve It."
  https://www.orkestrascs.com/resources/guides/otif
- MIT CTL.SC1x, "Inventory Models for Probabilistic Demand," annotated course notes.
  https://courses.edx.org/asset-v1:MITx+CTL.SC1x_1+2T2015+type@asset+block/w7l1_ContinuousReviewPolicies_ANNOTATED_v14.pdf

**Press and trade**

- Retail Dive, "American Eagle Outfitters pulls guidance, writes down $75M in inventory," May 2025;
  Reuters, "Puma inventory headache highlights US tariff dilemma for retailers," 25 July 2025.
- *Wall Street Journal*, "Retailers Stockpiled to Avoid Tariffs. The Holidays Will Put That to the
  Test," October 2025.
- Supply Chain Dive, "5 supply chain management trends to watch in 2026," January 2026;
  FreightWaves, "2M+ import containers to set new record, say retailers," July 2026;
  SMBC, "Early Signs of Q3 Inventory Restocking," September 2026.

**Note on citations.** Benchmark and survey figures are reported as the sources state them and are
not independently verified. Where a figure is gated, out of date, or inconsistent with another
series, the text says so rather than quoting precision it cannot support.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`04_planning-processes.md`](04_planning-processes.md)
- Next topic: [`06_cost-and-working-capital.md`](06_cost-and-working-capital.md)
- Related: [`07_manufacturing-basics.md`](07_manufacturing-basics.md)
- Foundations: [`01_three-flows-one-network.md`](01_three-flows-one-network.md)
- Survey file: [`../../../course/01_supply_chain_foundations.md`](../../../course/01_supply_chain_foundations.md)
- Deeper treatment of inventory optimisation:
  [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
- Return to the part map: [`../_index.md`](../_index.md)

