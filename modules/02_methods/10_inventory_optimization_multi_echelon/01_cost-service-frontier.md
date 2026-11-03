# The Cost-Service Frontier — the trade-off curve and the marginal cost of the next point of service

> Module 10 — Inventory Optimization and Multi-Echelon Planning
> Topic 01 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 07 topic 09; Module 08 topics 07 and 10
> Version 2026.5

---

## 1. Why this topic exists

Inventory exists to buy service. Every unit of safety stock is a purchase, made with a currency
the company understands, in exchange for a reduction in the probability that a customer is told
no. The exchange rate between the two is the single most useful object in inventory management,
and almost nobody computes it.

Instead, organisations set a number. Ninety-five per cent is the number in most companies, in
most enterprise resource planning (ERP) systems, and in most textbooks. It arrives as a default,
survives as a convention, and is defended as a policy. When you ask where it came from, the
answer is usually that it was there when the person arrived.

The number is not the problem. The problem is that it is stated as a target when it is really a
position on a curve, and the curve is not the same for any two stock-keeping units (SKUs). On a
cheap, fast-moving item with a stable demand history, the ninety-fifth percentile of demand
costs a few hundred euros a year. On an expensive, slow-moving item with a long lead time, the
same nominal target costs tens of thousands. Setting both to 95 per cent is not a policy. It is
a refusal to make a decision, expressed in the grammar of one.

Three failures recur, and this topic exists to name and fix them.

**The target failure.** A service level is treated as an input when it is an output. A target is
a statement about what the company wants; a frontier is a statement about what the company can
have and at what price. An organisation that sets the target without the price is choosing
blind, and then discovering the price in the working-capital review three months later, where
it is attributed to demand.

**The uniformity failure.** "Ninety-five per cent everywhere" looks fair, looks simple, and is
measurably the most expensive way to hit any aggregate service objective. The reason is
arithmetic rather than philosophy: the marginal cost of the next point of service differs
across items by one or two orders of magnitude, and a uniform target spends the budget in the
wrong places by construction. The empirical literature on this is not ambiguous (Section 9).

**The authority failure.** A curve drawn from eighteen months of demand history, a normal
assumption that the tail does not support, and a holding-cost rate that finance has never
reviewed, carries the visual authority of a physical law. It will be used to justify a budget.
A frontier drawn from bad data is worse than no frontier, because no frontier invites a
judgement and a bad one forecloses it.

**Who is blocked.** The inventory analyst asked to justify a service target to a finance partner
who wants to know what the money buys. The planner who suspects that one item is grossly
over-served and another is starved, and has no vocabulary for saying so. The data scientist who can
produce a calibrated demand distribution and is asked for "the 95th percentile" as though the
percentile were the decision. The supply chain director defending a working-capital number to a
board and a service promise to a customer, with the same word used for both.

**What this topic is not.** The critical fractile at the level of a single point forecast is
Module 07 topic 09, and it is not re-derived here. The decision use of quantiles, the service
level families, and the distinction between cycle service level and fill rate are Module 08
topic 10, and they are referenced rather than re-taught. What this topic adds is the layer
above both: the curve those objects trace out, the slope of that curve, and what happens when
you have to allocate one budget across many curves at once.

> **Note:** This module is where a single decision becomes a system of policies. Everything here
> assumes you can already produce a defensible distribution for one item. If you cannot, the
> frontier will be a confident drawing of a shape you have not measured.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** a cost-service frontier for a named item from its demand distribution and holding
   cost, and **label** the flat region, the knee, and the steep tail.
2. **Compute** the marginal cost of the next point of service, both as a band average and as a
   local slope, and **explain** why the two differ.
3. **Derive** the implied ratio of shortfall cost to holding cost that any stated service target
   encodes, and **state** what that ratio means commercially.
4. **Convert** a frontier between three service coordinates — probability, fill rate, and cost —
   and **show** how the same nominal number lands at different points on the curve.
5. **Demonstrate** that a change in lead time shifts the whole curve rather than a point on it,
   and **state** the condition under which the curve's shape is invariant.
6. **Allocate** a fixed inventory budget across a portfolio of items by equalising marginal
   rates, and **quantify** the cost of a uniform target against that allocation.
7. **Specify** the data required to draw a frontier and **identify** the four ways a frontier
   drawn from those data can be wrong.
8. **Write** a one-page justification of a service target that states the price of the choice
   and the evidence behind the curve.

Each outcome maps to an exercise in section 12 and to an answer or rubric in section 13.

## 3. The core mental model

Hold one picture: a curve, not a number. The horizontal axis is service. The vertical axis is
the annual cost of holding the inventory that delivers it. The curve starts flat, bends upward
at a knee, and rises without bound as service approaches certainty.

```text
   incremental safety-stock holding cost, EUR per year
        ^
  4,000 |                                                     ,
        |                                                  ,
        |                                               ,
  3,000 |                                           ,
        |                                        ,
        |                                    ,         steep tail:
  2,000 |                             *  <--+           EUR 540 per point
        |                        ,     knee            at 0.99
        |                  ,
  1,200 |          ,  <---- marginal cost
        |     ,            at 0.95 = EUR 140 per point
        | ,  <---- marginal cost
        |/          at 0.90 = EUR 82 per point
      0 +---|--------|--------|--------|--------|--------|----> CSL
          0.80     0.90     0.95     0.98     0.99    0.995   1.00
          <---- flat ---->  <-knee->  <--- steep tail ---->   |
                                                               v
                                                    vertical asymptote
```

The picture has five properties worth memorising, because they organise everything that follows.

**Property 1 — The frontier is a curve of achievable pairs, not a target.** Every point on it is
a policy the company could run. The target is a choice of one point. The curve does not tell you
which point; it tells you what each point costs, which is the information the choice requires.

**Property 2 — The flat region is real, and it is where most items live.** Below roughly 85 per
cent cycle service level (CSL), the cost of the next point is small in absolute terms for most
items, and the binding constraint is not money. The CSL is the probability that a replenishment
cycle ends without a stock-out. In the flat region, an extra half-point of service costs less
than the administrative effort of arguing about it.

**Property 3 — The knee is where the decision is, and its position is set by a cost ratio.** The
curve bends where the cost of one more unit of stock begins to exceed the expected value of the
stock-out it prevents. That point is the fractile `b/(b+h)`, where `b` is the shortfall cost per
unit and `h` is the holding cost per unit per period. A 95 per cent target is therefore not a
service convention. It is an implicit assertion that a stock-out costs nineteen times what
holding a unit costs. That assertion is either true for the item or it is not.

| Stated CSL | Implied ratio `b/h` | What the target asserts |
|---:|---:|---|
| 0.80 | 4 : 1 | A shortfall is four times as painful as a leftover |
| 0.90 | 9 : 1 | Nine times |
| 0.95 | 19 : 1 | Nineteen times |
| 0.98 | 49 : 1 | Forty-nine times |
| 0.99 | 99 : 1 | Ninety-nine times |
| 0.999 | 999 : 1 | The item is effectively unspeakable to run out of |

**Property 4 — The steep tail is a purchase, not a policy.** Above roughly 98 per cent, each
additional point is a discrete commercial decision with a price tag, and the price rises
geometrically. The frontier has a vertical asymptote at 100 per cent: perfect service costs
infinitely much, because the required safety stock grows without bound even though the
probability it protects against shrinks to nothing.

**Property 5 — There are two frontiers, and they are different objects.** The item frontier says
what service costs for one SKU. The portfolio frontier says what service costs for a budget
allocated across many SKUs with different curves. The allocation rule for the second is not
equal service, and the difference between the two is the subject of Section 5, Phase D. A
company that has drawn only item frontiers and then applied one target to all of them has
skipped the harder half of the problem.

> **The one-sentence version.** A service target is a price, and a frontier is the price list.
> An organisation that sets targets without a price list is negotiating with itself.

## 4. Definitions and vocabulary

The service families themselves are defined in Module 08 topic 10. What is given here is the
frontier vocabulary, with the unit and the coding case that decides whether the number you
compute is the number you meant.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Cost-service frontier | The set of achievable (service, cost) pairs for an item or portfolio under a stated policy family | Called a "trade-off curve" and then treated as a straight line | Currency per year against a stated service measure |
| Marginal cost of service | The additional annual cost of raising service by one unit (one percentage point, or one unit of fill) at a stated point | Computed as an average across a wide band and reported as a local rate | Currency per percentage point per year |
| Local slope | The derivative of the cost curve at a service level, `dC/dα` | Confused with the band average | Currency per unit of service |
| Band average | The cost difference between two service levels divided by the width of the band | Reported as though it were the marginal cost at either end | Currency per percentage point |
| Service level | A stated probability, proportion, or ratio describing how well demand is met | Used without naming the family | Must carry family, horizon, population, unit |
| Cycle service level (CSL) | Probability that a replenishment cycle ends without a stock-out | Reported as a fill rate | Proportion per cycle |
| Fill rate | Fraction of demanded units served from stock | Treated as a quantile at the same numeric level | Proportion of units, stated window |
| Ready rate | Fraction of time that stock is available | Treated as identical to CSL | Proportion of time |
| Cost-measured service | Service expressed as expected unmet demand or expected cost of shortfall | Treated as not being a service measure at all | Units per period, or currency per period |
| Safety stock | Buffer held above expected protection-period demand | Treated as a policy constant rather than a quantile offset | Units |
| Cycle stock | Inventory that exists because orders are batched | Added to safety stock when computing the frontier | Units |
| Holding cost rate | Annual cost of carrying one euro of inventory, as a fraction | Quoted without saying what it includes | Proportion per year, exclusions stated |
| Carrying cost per unit (`h`) | Annual cost of holding one unit, equal to unit cost times the holding cost rate | Computed on selling price rather than cost | Currency per unit per year, price year stated |
| Shortfall cost (`b`) | Cost of one unit of unmet demand, at the moment it is unmet | Stated as gross margin alone | Currency per unit short, price year stated |
| Implied cost ratio | The `b/h` value encoded by a stated service target | Never computed, so never challenged | Dimensionless ratio |
| Economic service level | The service level at which marginal holding cost equals marginal shortfall cost avoided | Described as "the optimal service level" as though it were the only defensible one | Proportion, at a stated cost model |
| Critical fractile | `b/(b+h)`, the cost-minimising quantile of a single-period order | Re-derived here; it is Module 07 topic 09 | Dimensionless, 0 to 1 |
| Shadow price of service | The marginal cost of the binding service or budget constraint in an optimisation | Reported as an accounting cost | Currency per unit of constraint |
| Budget constraint | A cap on the holding cost or inventory value the policy may consume | Treated as a soft preference rather than a hard limit | Currency per year |
| Lagrange multiplier | The rate at which the objective improves per unit of relaxed constraint | Used without an interpretation | Currency per unit of constraint |
| Marginal rate of substitution | The rate at which service can be traded between two items at constant total cost | Confused with the marginal cost of either item | Ratio of marginal costs |
| Knee of the curve | The service region where marginal cost begins to rise sharply | Located by eye rather than by the cost ratio | Proportion, with the implied `b/h` stated |
| Tail probability | `1 − F(x)`, the probability of exceeding a level | Quoted as "the service level" when it is its complement | Proportion |
| Standard normal density (`φ`) | The height of the normal density at `z` | Confused with the cumulative distribution | Dimensionless per standard deviation |
| Standard normal loss (`L`) | `φ(z) − z(1−Φ(z))`, expected exceedance in standard deviations | Unknown outside specialist teams | Dimensionless |
| Safety factor (`z`) | Standard deviations between the policy level and the mean | Quoted without the probability it encodes | Dimensionless |
| Protection period | Lead time plus review period over which stock must cover demand | Replaced by lead time alone in periodic review | Days or weeks |
| Lead-time demand (LTD) | Total demand realised between order placement and receipt | Computed on the planning bucket rather than the lead time | Units over the protection period |
| Coefficient of variation (CV) | Standard deviation divided by the mean | Compared across different time buckets | Dimensionless, bucket stated |
| Item frontier | Cost-service curve for one SKU at one location | Assumed to generalise to a category | Currency per year per SKU |
| Portfolio frontier | Cost-service curve for a set of SKUs under one budget | Assumed to be the sum of item frontiers with one target | Currency per year against an aggregate measure |
| Efficient allocation | An allocation where no item's service can be raised without lowering another's at the same budget | Confused with equal service | Vector of service levels |
| Uniform service policy | One target applied to every item in a population | Defended as fair or simple | Proportion, applied without differentiation |
| Service differentiation | Deliberately different targets across items, justified by cost or criticality | Described as inconsistent | Vector, with the rationale recorded |
| Price-criticality ratio (PCR) | A ranking quantity combining unit price and shortfall criticality | Unknown outside the inventory literature | Dimensionless, defined per study |
| ABC classification | Grouping items by annual value or volume, conventionally 80/15/5 | Treated as an optimal segmentation | Class label A, B, C |
| Censoring | Demand truncated by available stock and therefore unobserved | Forecast from sales and called it demand | Units, with a censoring flag |
| Stock-keeping unit (SKU) | The item at the finest level at which it is planned and stocked | Aggregated into a family and given the family target | Identifier |
| Economic order quantity (EOQ) | The order size minimising ordering plus holding cost | Used as a service lever | Units per order |
| On-time in-full (OTIF) | Fraction of orders delivered complete and on the promised date | Quoted without the definition of "on time" | Proportion of orders |

Four of these carry most of the confusion.

**Marginal and average are different numbers, and the difference grows with the band.** Moving
from 90 to 95 per cent costs an average of 105 euros per point in the worked example. Moving
from 99 to 99.5 costs an average of 719. The local slope at 99 is 540 per point, which is higher
than the 98-to-99 average of 393 and lower than the 99-to-99.5 average of 719. All three are
correct answers to different questions. The question "should we fund the next point" needs the
local slope; the question "should we fund this band" needs the band average.

**The item frontier and the portfolio frontier are not related by aggregation.** You cannot sum item
frontiers to get a portfolio frontier, because the budget constraint couples the items. The portfolio
curve is traced by re-optimising the allocation at each budget level, and it is generally better than
any uniform-target allocation at the same spend.

**Cost-measured service is still service.** Expressing service as expected unmet units per year,
or as expected shortfall cost, does not make it a non-service measure. It makes it a measure
whose units are comparable across items, which is exactly what the allocation problem needs.

## 5. The mechanics, step by step

Five phases and twenty-two steps, from the definition of the coordinate to a published curve
with a data vintage on it. The sequence matters: the coordinate is fixed before the distribution
is built, the distribution before the cost, the item curves before the portfolio allocation, and
the allocation before anybody is told what their new target is.

### Phase A — Fix the coordinate

**Step 1. Name the service family, and write down its denominator.**

A frontier is a curve in a coordinate system, and the coordinate has to be named before the
curve means anything. A CSL is a probability per cycle. A fill rate is a ratio of units over a
window. A ready rate is a ratio of time. A cost-measured service is currency or units per
period. Section 6.4 shows one item where the same word, "ninety-five", lands on two points of
the same cost curve that differ by a factor of 4.8 in annual holding cost.

The test is the one from Module 08 topic 10: hand the target sentence to two analysts and ask
them to compute it from the warehouse. If they produce different numbers, the coordinate is not
fixed. Where the contract names a different family from the internal target, draw two frontiers
and state which one governs the money.

**Step 2. Fix the population and the unit of allocation.**

The frontier is drawn for a population: one SKU at one location, a category, a customer segment, a
distribution centre. The population determines what "the service level" is an average of, and
averages across populations hide exactly the variation the frontier exists to expose.

The allocation unit is what the budget is spent on, usually the SKU-location. It is not the SKU,
because the same item at two locations has two demand distributions and two lead times; it is
not the category, because a category target cannot be executed. If the planning system allocates
at a coarser grain than demand varies at, draw the frontier at the coarser grain and state the
error rather than hiding it.

**Step 3. Fix the horizon: the protection period, not the forecast horizon.**

Holding cost is incurred continuously; service is realised over the interval during which the
policy cannot respond. That interval is the protection period: lead time plus review period,
inflated by information delay. The distribution must cover it and the cost must be charged over
it. A frontier drawn on weekly demand and applied to a four-week protection period is not a
frontier for this item. Under continuous review the protection period is the lead time; under
periodic review it is lead time plus review period. Do not use one convention for both.

**Step 4. Decide whether service is an input or an output.**

| Mode | Service is | The question asked | What the frontier supplies |
|---|---|---|---|
| Target | An input | "What does 95 per cent cost, and is it worth it?" | The price of a stated promise |
| Optimisation | An output | "Given this budget, what service do we get, and where?" | The efficient allocation |
| Constraint | An input with a floor | "Meet these floors at minimum cost" | The cost of the floors, and where they bind |
| Diagnostic | An output, unconstrained | "Where are we spending badly today?" | The gap between current and efficient |

Most organisations claim target mode and operate in diagnostic mode without saying so. Target
mode needs a shortfall cost or an approved promise; diagnostic mode needs neither, only a
demonstration that the current allocation is dominated. If nobody can state a shortfall cost,
you are in diagnostic mode whether you admit it or not. Say so.

### Phase B — Build the cost side

**Step 5. Build the carrying cost per unit, and state what it excludes.**

| Component | What it covers | Typical magnitude | Why it matters to the frontier |
|---|---|---|---|
| Capital charge | Cost of the cash tied up | 5–12 per cent of unit cost per year | Scales the whole curve; does not change its shape |
| Physical holding | Storage, handling, insurance, tax | 3–8 per cent | Often a step function, not a rate |
| Risk | Obsolescence, shrinkage, damage, expiry | 2–20 per cent, item-dependent | Rises with quantity held, so it steepens the tail |
| Total, as a convention | Usually quoted as one rate | 18–30 per cent | The single number hides the third row |

The rate is a management estimate, not an accounting quantity. International Accounting
Standard 2 (IAS 2) governs what may be capitalised into inventory value; it does not specify a
carrying rate, and no standard does. This is the honest position: `h` is a modelling parameter
whose build-up must be shown, because it scales every number on the vertical axis. If the item
has a shelf life, do not use a single rate; use a rate that rises with the quantity held.

**Step 6. Separate cycle stock from safety stock, and price only the part that moves.**

The frontier plots the cost of the inventory that changes when service changes: the safety
component. Cycle stock depends on the order quantity and the economic order quantity (EOQ), not
on the service level. The obvious implementation, `cost = h × total inventory`, is correct as a
total and wrong as a frontier: the cycle component shifts the curve up without changing its
slope, and you will attribute the intercept to service. If the order quantity is itself set by a
service target, as in some vendor agreements, the separation fails and the frontier must be
re-derived with `Q` endogenous.

**Step 7. Write the cost function, in both forms.**

The incremental form is `C(α) = h·z(α)·σ`, cycle stock excluded. The total form is
`C(α) = h·(μ + z(α)·σ)`. Use the incremental form for the frontier and the total form for the
budget. They differ by `hμ`, which is large and constant. Reporting the total form while discussing
marginal decisions inflates every number by the cycle stock and makes the marginal cost look
smaller as a share of the total, a rhetorical effect rather than an economic one.

**Step 8. Add the costs that are not linear in stock.**

- **Storage capacity.** Warehouse space is bought in blocks. The curve jumps where a new aisle,
  rack, or chamber is required, and a frontier that is smooth through a capacity boundary is
  wrong at exactly the point where the investment decision sits.
- **Obsolescence and expiry.** The probability of writing off a unit rises with how long it
  sits. Safety stock held against a 99th percentile sits longer than stock held against a 90th,
  so its effective holding cost per unit is higher and the linear rate understates the tail.
- **Insurance and tax.** Year-end inventory assessment is real and it makes observed cost
  discontinuous in a way the model does not know about.

If the item is perishable, the linear frontier is not an approximation. It is a different model,
covered in Module 10 topic 10.

**Step 9. Price the shortfall side, or state that you cannot.**

| Component | Who owns the estimate | Typical difficulty |
|---|---|---|
| Lost contribution margin | Commercial finance | Easy if the sale is genuinely lost |
| Expedite or recovery premium | Operations | Moderate; depends on substitute supply |
| Contractual penalty or service credit | Legal and account management | Easy where a clause exists |
| Goodwill, escalation, and future share | Nobody | Difficult, political, often the largest |

Where no estimate exists, invert the question: state the target, compute the implied `b/h` ratio,
and ask whether it is plausible. A 99 per cent target on a commodity item implies a 99-to-1 ratio:
a stock-out would cost ninety-nine times a leftover. Sometimes that is true; often it is not, and
the ratio is a better provocation than a cost estimate nobody believes. If the components span a
factor of three, carry the range through the analysis rather than picking the midpoint.

### Phase C — Trace the curve

**Step 10. Tabulate the safety factor on a service grid.**

Use 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999. The grid is not arbitrary: the top three points
are where the priced decisions live, and a grid that stops at 0.99 cannot show the asymptote. Read
`z` from the inverse normal distribution function, not from memory. The common error is reading
`z = 2` for 95 per cent, which is the two-sided value and corresponds to 97.7 per cent one-sided.

**Step 11. Convert to units and to money.**

Multiply `z` by `σ` for safety stock in units, then by `h` for currency per year. Keep both columns
visible: the units column is what the warehouse can act on, the money column is what the decision is
about. State whether `σ` is over the protection period and whether `h` is annual. A mismatch here
produces a curve wrong by the ratio of the periods, often four to thirteen.

**Step 12. Compute the marginal cost twice: as a band average and as a local slope.**

The band average between two service levels is the cost difference divided by the width; the local
slope is the derivative. In the normal case the derivative is `dC/dα = hσ/φ(z)`, where `φ` is the
standard normal density, since `dz/dα = 1/φ(z)` by the inverse function theorem and `C = hσz`.

Report both. The band average answers "what does this programme cost"; the local slope answers
"what does the next point cost". In the steep tail they converge because the band is narrow. In
the flat region they diverge, because the slope changes little across a wide band and the
average is a fair summary. The middle is where they differ most and where most arguments happen.

**Step 13. Read the three regions, and locate the knee from the cost ratio.**

| Region | Approximate CSL range | Behaviour | What governs the decision |
|---|---|---|---|
| Flat | 0.50–0.85 | Marginal cost low and slowly rising | Administrative effort, not money |
| Knee | 0.85–0.97 | Marginal cost rising steeply | The cost ratio `b/h` |
| Steep tail | 0.97–1.00 | Marginal cost rising geometrically | An explicit purchase decision |
| Asymptote | 1.00 | Cost unbounded | Nothing; perfect service is not purchasable |

The knee is not a visual feature. It is the point where the marginal cost of the next point
begins to exceed the marginal value of the shortfall it prevents, and its position is `b/(b+h)`.
The ranges shift with the cost ratio; the ordering does not. If the current target sits far
above the knee and nobody can state why, that is a finding, not a problem with the analysis.

**Step 14. Check the shape assumptions before you trust the shape.**

| Assumption | What it produces | How to test it | What happens if it fails |
|---|---|---|---|
| Normality of protection-period demand | Smooth convex curve | Compare empirical and fitted quantiles at 0.90, 0.95, 0.99 | Tail understated; curve too flat at the top |
| Independence across periods | `σ` scaling as the square root of the horizon | Estimate autocorrelation; compare with the aggregate | `σ` wrong, so the whole curve is scaled wrongly |
| Stationarity over the horizon | A fixed curve | Compare distributions across sub-periods | The curve describes a regime that has ended |

### Phase D — From one item to a portfolio

**Step 15. Draw every item curve in a common unit.**

Item frontiers are in currency per year against a service measure, and they need a common
vertical axis to be comparable. Two work: currency per unit of expected shortfall avoided, which
is what the budget actually buys; and currency per point of a common service measure, which is
what the organisation already tracks. Compute both, report the second, allocate on the first.
Where some items have no measurable shortfall cost, allocate those on a criticality ranking and
state the split.

**Step 16. Sort by marginal rate, and read the spread.**

The rate for item `i` at tail probability `p` is `h_i / (p_i · f_i · b_i)`, with `h` in currency
per unit per year, `f` in cycles per year, and `b` in currency per unit short. The item's `σ`
cancels between numerator and denominator, which is worth pausing on: two items with the same
cost structure and the same target have the same marginal rate regardless of how volatile their
demand is. Variability scales the quantity, not the price of the next point.

At a uniform 95 per cent target the five-item portfolio in Section 6.6 ranges from 0.037 to 5.76
euros per unit of shortfall avoided, a factor of 156. If the spread is below about 3:1, uniform
targets are close to efficient and the reallocation is not worth the organisational cost. State
that finding too.

**Step 17. Reallocate until the marginal rates equalise at the budget.**

Minimise total expected shortfall cost subject to a holding-cost budget. The first-order
condition sets marginal benefit per euro equal across items, which is a Karush-Kuhn-Tucker (KKT)
point; in practice it is found by bisection on the shadow price `λ`: set
`p_i = h_i / (λ f_i b_i)`, compute the holding cost, and adjust.

| What the rule says | What it does not say |
|---|---|
| Raise service on cheap-to-hold, high-penalty, frequently ordered items | How to treat items with no measurable penalty |
| Cut service on expensive, low-penalty, rarely ordered items | Whether a floor is politically required |
| Target a tail probability, not a service level | Whether the reported average service will look better or worse |

The last cell is the one that gets analysts into trouble. Section 6.6 shows an allocation that
cuts total cost by 34 per cent and lowers the average CSL by half a point. Both numbers are
correct, and the second is the one that will be quoted back at you.

**Step 18. Check the aggregate measures, and expect them to move the wrong way.**

| Measure | Weighting | Typical movement under efficient allocation |
|---|---|---|
| Aggregate fill rate | Units | Improves; high-volume items are often the cheap ones |
| Mean per-item CSL | Items, equally | Can fall; cutting one expensive item costs the average |
| Expected shortfall | Units per period | Falls sharply; it is the quantity being optimised |

Report all three. A single aggregate number is what caused the problem in the first place, and
if the only reported measure is the item average, the reallocation will be reversed at the next
review.

**Step 19. Record the targets, the shadow price, and the reason.**

Every target gets a row: item, target, family, protection period, `h`, `b` or the implied ratio,
marginal rate at the target, owner, effective date, approver. The shadow price goes at the top
of the table, because it is what makes the vector coherent and it is what will change next
quarter. An allocation without a recorded shadow price is indistinguishable from a set of
arbitrary but different numbers, which is how it will be described by anyone who disagrees.

### Phase E — Defend and maintain

**Step 20. Attach the data vintage to the curve.**

State the observation window, the number of observations in the tail, the fitting method, the
calibration test, and the date. If the tail has fewer than about twenty observations beyond the
highest plotted service level, say so on the figure. A curve without a vintage will be quoted in
two years as though it were still true.

**Step 21. Test the tail shape, and widen the curve where the data cannot support it.**

| Distribution feature | Effect on the frontier | What to do |
|---|---|---|
| Skewed, continuous (gamma, lognormal) | Steeper tail, higher cost at high service | Fit and compare; do not assume normal above 0.95 |
| Intermittent with an atom at zero | Staircase frontier with vertical jumps | Read empirical quantiles; report the jump sizes |
| Heavy-tailed, occasional very large demand | Tail cost several times the normal estimate | Use empirical or semi-parametric quantiles; bound the exposure |
| Regime-changing | The curve moves between regimes | Draw one curve per regime; do not average |

The staircase case breaks the calculus. When demand is discrete the frontier is a step function:
between support points, raising the target changes nothing and costs nothing, and at a support
point the cost jumps by the size of the atom. The marginal cost of "the next point" is either
zero or undefined, and the useful question becomes "what does the next step cost". An analyst
who reports a smooth slope on a discrete distribution has fitted a curve to a staircase and
hidden the steps, which are the decision.

**Step 22. Re-draw on a calendar, and diff the curves.**

The frontier is a maintenance object. A curve that has shifted up at constant shape means costs
or variability rose. A curve whose knee has moved left means the cost ratio fell. A curve that
has changed shape means the distribution changed, which is a demand question, not an inventory
question. If the curve has moved by more than the decision tolerance since the target was set,
the target is re-opened, not automatically re-set.

## 6. Worked example

A distributor of industrial components holds five items at one warehouse. We draw the frontier
for one of them in detail, then allocate a fixed budget across all five. The numbers are
constructed to be internally consistent and to isolate the economics; they are not drawn from a
published case. All money is euro at 2026 price levels, all rates are annual, and all service
levels are cycle service levels unless the text says otherwise.

### 6.1 The item and its parameters

Item A-114 is a bracket sold to original equipment manufacturers. It is cheap, fast-moving, and
smooth at the weekly grain.

| Parameter | Value | Unit | Basis |
|---|---:|---|---|
| Mean protection-period demand, `μ` | 1,000 | units | Four-week lead time, review weekly |
| Standard deviation, `σ` | 200 | units | Coefficient of variation 0.20 |
| Unit cost | 30.00 | EUR per unit | Landed cost, 2026 prices |
| Holding cost rate | 24 | per cent per year | 9 capital, 6 physical, 9 risk |
| Carrying cost, `h` | 7.20 | EUR per unit per year | 30.00 × 0.24 |
| Replenishment cycles, `f` | 13 | per year | Order quantity near mean demand |
| Order quantity, `Q` | 1,000 | units | Equal to mean demand, for the fill-rate case |
| Shortfall cost, `b` | 25.00 | EUR per unit short | Margin 18, expedite 4, goodwill 3 |

The product `hσ` is 1,440 euros per year per standard deviation. That single number sets the
vertical scale of the entire curve, and it is the number to check first when a frontier looks
wrong.

### 6.2 The curve

| CSL | `z` | Safety stock (units) | Incremental holding cost (EUR/yr) | Band cost (EUR per point) |
|---:|---:|---:|---:|---:|
| 0.80 | 0.8416 | 168.3 | 1,212 | — |
| 0.85 | 1.0364 | 207.3 | 1,492 | 56.1 |
| 0.90 | 1.2816 | 256.3 | 1,845 | 70.6 |
| 0.95 | 1.6449 | 329.0 | 2,369 | 104.8 |
| 0.98 | 2.0537 | 410.8 | 2,957 | 196.1 |
| 0.99 | 2.3263 | 465.3 | 3,350 | 392.5 |
| 0.995 | 2.5758 | 515.2 | 3,709 | 718.6 |
| 1.000 | ∞ | ∞ | ∞ | ∞ |

Read the last column as the average price of a point across each band. It rises from 70 euros
between 85 and 90 per cent to 719 euros between 99 and 99.5. The rise is not a modelling
artefact. It is what the normal tail does: the density falls away, so a fixed increment of
probability requires a growing increment of standard deviations.
### 6.3 The marginal cost, computed two ways

The band averages above answer "what does this programme cost". The local slope answers "what
does the next point cost now". The local slope is `hσ/φ(z)`.

| CSL | `φ(z)` | Local slope (EUR per unit of service) | Local slope (EUR per point) | Ratio to the 0.90 rate |
|---:|---:|---:|---:|---:|
| 0.80 | 0.28034 | 5,137 | 51.4 | 0.63 |
| 0.90 | 0.17538 | 8,211 | 82.1 | 1.00 |
| 0.95 | 0.10314 | 13,962 | 139.6 | 1.70 |
| 0.98 | 0.04861 | 29,624 | 296.2 | 3.61 |
| 0.99 | 0.02665 | 54,030 | 540.3 | 6.58 |
| 0.995 | 0.01446 | 99,585 | 995.9 | 12.1 |
| 0.999 | 0.00337 | 427,300 | 4,273 | 52.0 |

Two readings matter. **Between 90 and 99 per cent, the price of a point rises by a factor of
6.6.** That is the entire quantitative case against uniform high targets: the same nominal
increment costs seven times as much at the top as in the middle. **And at 99.9 per cent, the
price of a point is 52 times the price at 90 per cent.** The curve is not steep in the way a
straight line is steep. It is steep in the way an exponential is steep, and it is why the last
decimal place is a different kind of decision from the first.
### 6.4 The same item in a fill-rate coordinate

The coordinate matters as much as the number. Hold the item fixed and change only the family
the target is expressed in.

| Stated target | `z` | Safety stock (units) | Holding cost (EUR/yr) | Achieved CSL | Achieved fill rate |
|---|---:|---:|---:|---:|---:|
| CSL 90 per cent | 1.2816 | 256.3 | 1,845 | 90.0% | 99.06% |
| CSL 95 per cent | 1.6449 | 329.0 | 2,369 | 95.0% | 99.58% |
| CSL 99 per cent | 2.3263 | 465.3 | 3,350 | 99.0% | 99.93% |
| Fill rate 95 per cent | 0.3455 | 69.1 | 498 | 63.5% | 95.00% |
| Fill rate 99 per cent | 1.2560 | 251.2 | 1,809 | 89.6% | 99.00% |
| Fill rate 99.9 per cent | 2.1900 | 438.0 | 3,154 | 98.6% | 99.90% |

The fill rates are computed from `β = 1 − σL(z)/Q` with `Q = 1,000`, using the standard normal
loss function `L(z) = φ(z) − z(1−Φ(z))`. At `z = 1.6449`, `L = 0.020889`, so the expected
shortfall is `200 × 0.020889 = 4.18` units per cycle and the fill rate is `1 − 4.18/1,000`.

The pair to stare at is the first two rows against the fourth. **A 95 per cent CSL costs 2,369
euros a year. A 95 per cent fill rate on the same item costs 498.** The same word, the same
item, the same week, a factor of 4.8. The fill-rate target is cheaper because it is a far weaker
promise: it runs out on 36 per cent of cycles rather than 5, and it simply serves most of the
units when it does.

Neither is wrong. The point is that "ninety-five" does not locate you on the curve. It locates
you on a curve only once the family is named, and the two families put the same number at
opposite ends of the decision region.
### 6.5 Lead time moves the whole curve

Weekly demand has a standard deviation of 100 units. The protection period is the lead time in
weeks, review being effectively continuous at this volume. Under independence,
`σ_LTD = 100 × √L`.

| Lead time | `σ_LTD` (units) | Safety stock at 95% | Holding at 95% (EUR/yr) | Marginal cost at 95% (EUR/point) | Ratio to the 4-week case |
|---:|---:|---:|---:|---:|---:|
| 2 weeks | 141.4 | 232.6 | 1,675 | 98.7 | 0.707 |
| 4 weeks | 200.0 | 329.0 | 2,369 | 139.6 | 1.000 |
| 8 weeks | 282.8 | 465.2 | 3,349 | 197.4 | 1.414 |

The curve does not move to a different point. It is scaled vertically by `√L`. Every cost on it
changes by the same factor, and every marginal cost changes by the same factor. Halving the
lead time halves the price of the next point at 90 per cent, at 95 per cent, and at 99 per cent
alike.

That gives a testable invariance. Under the normal model with demand and lead time independent,
the ratio `C(0.99)/C(0.95)` is `2.3263/1.6449 = 1.414` at every lead time. If your computed
curves do not show that ratio, something other than lead time is changing between them, and you
should find out what before presenting.

The practical consequence is the important one. **Lead-time reduction is a multiplicative
lever; service targeting is a positional one.** Cutting lead time from eight weeks to four
buys the same proportional saving at every service level, which means it buys more absolute
money at the top of the curve, where the numbers are largest. That is why lead-time programmes
and service-target debates should be in the same conversation and usually are not.
### 6.6 The portfolio: five items, one budget

Now the harder problem. Five items share a holding-cost budget equal to what the company
currently spends on safety stock at a uniform 95 per cent target. The question is whether that
budget is being spent well.

**Parameters and the current position**

| Item | `σ` (units) | Unit cost (EUR) | `h` (EUR/unit/yr) | Cycles/yr | `b` (EUR/unit short) | SS at 95% (units) | Holding at 95% (EUR/yr) | Shortfall cost at 95% (EUR/yr) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-114 | 200 | 30 | 7.20 | 13 | 25 | 329.0 | 2,369 | 1,358 |
| B-207 | 60 | 180 | 43.20 | 6 | 25 | 98.7 | 4,263 | 188 |
| C-330 | 400 | 8 | 1.92 | 26 | 40 | 657.9 | 1,263 | 8,690 |
| D-401 | 15 | 900 | 216.00 | 2 | 900 | 24.7 | 5,331 | 564 |
| E-512 | 120 | 45 | 10.80 | 12 | 12 | 197.4 | 2,132 | 361 |
| **Total** | | | | | | | **15,358** | **11,161** |

Every shortfall figure uses `f × σ × L(1.6449) × b`, with `L(1.6449) = 0.020889`. For C-330:
`26 × 400 × 0.020889 = 217.2` units per year, times 40 euros, gives 8,690.

C-330 alone accounts for 78 per cent of the total expected shortfall cost, on 8 per cent of the
holding budget. That is the signature of a misallocation, and it is visible before any
optimisation.

**Marginal rates at the uniform target**

The rate is `h / (p × f × b)` with `p = 0.05` at every item, in euros of holding cost per euro
of shortfall avoided.

| Item | `h/(f·b)` | Rate at 95% | Rank |
|---|---:|---:|---:|
| A-114 | 0.02215 | 0.443 | 2 |
| B-207 | 0.28800 | 5.760 | 5 |
| C-330 | 0.001846 | 0.037 | 1 |
| D-401 | 0.12000 | 2.400 | 4 |
| E-512 | 0.07500 | 1.500 | 3 |

The spread is 156 to 1. The company is paying 5.76 euros to avoid one euro of expected
shortfall on B-207 while paying 3.7 cents for the same thing on C-330. No commercial argument
justifies that ordering; it is purely the arithmetic of a uniform target meeting non-uniform
cost structures.

**Equalising the rates at the budget**

Set `p_i = h_i / (λ f_i b_i)` for a common shadow price `λ`, and solve for the `λ` that spends
15,358 euros. The solution is `λ ≈ 1.82`.

| Item | Optimal `z` | Optimal CSL | Holding (EUR/yr) | Shortfall cost (EUR/yr) | Change in holding |
|---|---:|---:|---:|---:|---:|
| A-114 | 2.254 | 98.8% | 3,246 | 271 | +877 |
| B-207 | 1.006 | 84.3% | 2,606 | 741 | −1,657 |
| C-330 | 3.093 | 99.9% | 2,375 | 91 | +1,112 |
| D-401 | 1.512 | 93.5% | 4,897 | 761 | −434 |
| E-512 | 1.744 | 95.9% | 2,260 | 273 | +128 |
| **Total** | | | **15,384** | **2,137** | **+26** |

The budget is met to within 0.2 per cent, which is the bisection tolerance rather than a
modelling choice. Total cost falls from `15,358 + 11,161 = 26,519` to `15,384 + 2,137 = 17,521`,
a reduction of 34 per cent at essentially the same spend. Expected shortfall falls by 81 per
cent.

Look at what the allocation does. It cuts B-207 from 95 to 84 per cent, because B-207 is
expensive to hold, ordered rarely, and has a modest shortfall cost. It raises C-330 to 99.9 per
cent, because C-330 is cheap to hold, ordered twice a month, and has the highest shortfall cost
in the portfolio. It leaves E-512 almost unchanged, which is the accident that its cost
structure happens to be near the shadow price.

### 6.7 What the numbers say, and what to say in the meeting

The item-averaged CSL falls from 95.0 to 94.5 per cent. Three items rise, two fall. If the only
reported service measure is the mean of per-item CSLs, the reallocation looks like a
deterioration.

Everything else improves. Expected shortfall falls 81 per cent. Total cost falls 34 per cent.
The worst-served item in volume terms, C-330, goes from one stock-out every five cycles to one
in a thousand. The unit-weighted fill rate across the portfolio rises.

That divergence is the argument to make in the meeting, and it needs to be made once, calmly,
with three numbers side by side rather than one.

| Measure | At uniform 95% | At equalised rates | Direction |
|---|---:|---:|---|
| Mean per-item CSL | 95.0% | 94.5% | Down |
| Expected shortfall (EUR/yr) | 11,161 | 2,137 | Down 81% |
| Total cost (EUR/yr) | 26,519 | 17,521 | Down 34% |
| Items below 90% CSL | 0 | 0 | Unchanged |
| Highest-cost item (D-401) service | 95.0% | 93.5% | Down |
| Highest-shortfall item (C-330) service | 95.0% | 99.9% | Up |

rests on the shortfall costs in the fourth column, and those are estimates. Section 6.6 shows
the sensitivity that matters — if B-207's shortfall cost were three times higher, its implied
tail probability at the same shadow price would fall from 15.8 to 5.3 per cent, moving its
optimal service from 84 to about 95 per cent, and the cut would disappear. That range should
be shown, not hidden, because a reader who finds it later will discount everything else.

## 7. Code, query, or calculation

The calculation that matters is small enough to do by hand, and doing it by hand is the point:
the numbers are arguable, and the argument is easier when the arithmetic is visible. The block
computes the holding cost and the marginal cost at four service levels for item A-114, and it
is the table in Sections 6.2 and 6.3 in a form you can reproduce on paper.

```text
HAND COMPUTATION -- cost of CSL on normal demand, one item, annual figures

given  mu = 1,000 units   sigma = 200 units   h = EUR 7.20 per unit per year
       z(a) = Phi^-1(a)   SS = z * sigma      cost = h * SS

  a      z        SS (units)   cost (EUR/yr)    band average (EUR/point)
 0.90   1.2816     256.3         1,845             --
 0.95   1.6449     329.0         2,369           (2,369-1,845)/5  = 105
 0.98   2.0537     410.8         2,957           (2,957-2,369)/3  = 196
 0.99   2.3263     465.3         3,350           (3,350-2,957)/1  = 393

local slope, from  dC/da = h * sigma / phi(z)

  a = 0.90:  7.20 * 200 / 0.17538 =  8,211 /unit = EUR   82 per point
  a = 0.95:  7.20 * 200 / 0.10314 = 13,962 /unit = EUR  140 per point
  a = 0.99:  7.20 * 200 / 0.02665 = 54,030 /unit = EUR  540 per point

check  the local slope is a tangent; the band average is a chord. The chord
       lies below the slope at its upper end and above it at its lower end,
       because the slope is rising. Both answer real questions; label which.
```

Four checks before trusting any version of this calculation.

- Confirm that `σ` is over the protection period. A weekly `σ` applied to a four-week lead time
  understates the curve by a factor of two.
- Confirm that `h` is annual and that the cost column is annual. Mixing a per-cycle `h` with an
  annual demand rate is the most common unit error in this work.
- Confirm that the safety stock excludes cycle stock. If it does not, the intercept is wrong and
  the slope is right, which is the hardest kind of error to notice.
- Confirm that `z` came from the inverse normal and not from a two-sided table. At 95 per cent
  one-sided, `z` is 1.645; the two-sided value is 1.96, which is 97.5 per cent one-sided.

> **Note:** The relation `dC/dα = hσ/φ(z)` is exact under normality and gives the marginal cost
> in currency per unit of probability. Multiply by 0.01 for currency per percentage point. For
> non-normal demand the same idea applies through the density of the fitted distribution at the
> chosen quantile, `dC/dα = h/f(F⁻¹(α))`, which is why a heavy tail is expensive: the density
> in the tail is small, so its reciprocal is large.

## 8. Data, standards, and regulatory context

No international standard defines a cost-service frontier, a marginal cost of service, or a
holding cost rate. That absence is informative rather than a gap: the object is a management
construct, and its inputs must be documented rather than certified. What exists constrains the
vocabulary, the valuation of the inventory, and the control environment around the parameters.

### 8.1 Vocabulary and measurement

| Standard | Governs | What it fixes | What it does not fix | Grade |
|---|---|---|---|---|
| ISO 3534-1 | Statistics vocabulary and symbols | Definitions of probability, distribution, quantile, expectation | Any service measure, any cost rate | B |
| ISO 80000-1 | Quantities and units | Unit conventions for money, time, dimensionless ratios | Which currency, which price year | B |
| ISO 8601 | Date and time representation | Week numbering, durations, intervals | Which week a demand observation belongs to | B |
| ISO 31000 | Risk management | Risk appetite as a stated willingness to accept risk for a reason | How to price a service level | B |
| ASCM SCOR Digital Standard | Supply chain reference framework | Metric definitions including perfect order fulfilment | Any binding requirement; a reference model | C |
### 8.2 Where the cost side becomes an accounting number

| Requirement | Where it applies | Effect on the frontier | Grade |
|---|---|---|---|
| IAS 2 *Inventories* | IFRS-reporting entities | Fixes what may be capitalised; write-downs raise the effective risk component of `h` | B |
| ASC 330 *Inventory* | US GAAP entities | Lower-of-cost-or-market logic; same effect on the risk component | B |
| IAS 37 *Provisions* | IFRS-reporting entities | A contractual shortfall penalty is a provision; its best estimate belongs in `b` | B |
| IFRS 15 / ASC 606 | Revenue | Unsold units generate no revenue, so the margin forgone is part of `b` | B |
| Local inventory tax rules | Jurisdiction-specific | Period-end assessment creates a sawtooth in observed holding cost | B |
| Transfer pricing rules | Multinational groups | The internal margin in `b` depends on the transfer price | B |

The practical consequence is that the accounting function will give you a defensible inventory
valuation and will not give you a carrying rate, because the rate is a management estimate with
no standard behind it. Ask for the components separately: the cost of capital, the warehouse
cost per unit stored, and the write-off rate by product family. A single blended percentage
quoted in a slide is a number with no owner.

### 8.3 Where the service side becomes a contractual number

| Instrument | What it fixes | Consequence for the frontier | Grade |
|---|---|---|---|
| Service level agreement (SLA) | Measure, threshold, window, exclusions, remedy | Fixes the family and the target; the frontier becomes a cost-of-compliance curve | B |
| Penalty and liquidated-damages clause | A sum per unit short or per late order | Adds directly to `b`, sometimes by more than the margin | B |
| Service credit schedule | A percentage of invoice value per missed window | Converts to `b` only after the order-value distribution is known | B |
| Vendor-managed inventory agreement | Who sets the level and how availability is measured | The frontier is delegated, often without the cost model going with it | B |
| Take-or-pay capacity contract | Committed volume and unused-capacity penalty | The reservation is a high quantile and the penalty is an overage cost | B |

### 8.4 Internal control over the parameters

| Requirement | Where | What it implies for the frontier | Grade |
|---|---|---|---|
| Sarbanes-Oxley Act 2002, s.404 | US-listed entities | The process that sets and changes service targets is a control | B |
| Federal Reserve SR 11-7 | US bank holding companies | Model inventory, validation, ongoing monitoring — by analogy only | B |
| UK Corporate Governance Code | UK-listed entities | Internal control and risk reporting over material judgements | B |
| Local statutory audit requirements | Most jurisdictions | Parameters material to inventory valuation are auditable | B |
### 8.5 What is convention rather than requirement

The 95 per cent service level is a policy convention in most companies, not a regulatory
requirement. The `z`-value table is a convention inherited from a normal approximation and a
printed appendix. The carrying rate of 20 to 25 per cent is an industry rule of thumb with no
standard behind it. The 80/15/5 ABC split is a historical convention, not an optimum. The
one-sided normal quantile at 1.645 is a mathematical fact; the choice to use 95 per cent is not.
An analyst who can separate requirement from convention can challenge a convention; one who
cannot treats every inherited parameter as a physical constant.

## 9. Industry practice and case evidence

### 9.1 The uniformity result, which is the empirical core of this topic

The claim that uniform service targets are costly is not an analytical conjecture. It has been
measured on large real datasets.

| Evidence | What it reports | Grade |
|---|---|---|
| Teunter, Babai and Syntetos (2010), *Production and Operations Management* | On three datasets of 4,799, 39,274 and 9,086 SKUs, ranking by demand value or volume and applying a fixed service level per class is "far from cost optimal"; a cost-based ranking on `b·D/(h·Q)` performs better, and demand-value ranking performs worst because it rewards expensive items over cheap fast movers | A |
| Teunter, Syntetos and Babai (2017), *European Journal of Operational Research* | Assigning every SKU the same fill rate equal to the system target is far from optimal; differentiation by price-criticality ratio reduces stock value by at least 27 per cent, and up to 70 per cent, while meeting or exceeding the system target | A |
| Rosling (2002), *Operations Research* | Holding plus nonlinear shortage cost rates as a function of inventory position; shortage-cost coefficients read as the shadow prices of fill-rate or ready-rate service constraints | A |
| Chopra, Reinhardt and Dada (2004), *Decision Sciences* | The normal approximation misstates required safety stock for skewed lead-time demand; exact calculation gives materially higher stock at high cycle service levels, and the error grows with the coefficient of variation | A |

The 27 per cent figure is the one to carry into a business case, because it is a stock-value
reduction at constant service, which is free money in the accounting sense. It comes from a
specific dataset and method, and it should be quoted as a direction and a magnitude rather than
as a forecast for your portfolio. The direction is robust across all four rows.

### 9.2 The shape result, and where the shape evidence is weaker

The convexity of cost in service level is asserted in every textbook and measured less often.
The applied and vendor literature states it plainly; the peer-reviewed literature tends to
state it as a property of specific models rather than as an empirical finding.

| Evidence | What it reports | Grade |
|---|---|---|
| Rolf et al. (2024), Hawaii International Conference on System Sciences, guaranteed-service model applied to automotive networks | Reports a concave-upward cost-versus-service curve and states that the marginal cost of raising service grows rapidly because of the near-quadratic relation between the two | B |
| MetricGate inventory calculator documentation | States that moving from 95 to 99 per cent adds roughly 40 to 50 per cent more safety stock, and 95 to 99.9 per cent can double or triple it | D |
| Georgia Tech Supply Chain and Logistics Institute teaching material | Plots inventory against service level at 6-week and 12-week lead times, showing the vertical shift | C |
| Umbrex inventory playbook | Recommends stepping service from 95 to 96 to 97 per cent and comparing marginal carrying cost against marginal stockout cost saved | D |

The second and fourth rows are vendor or advisory content and are graded accordingly. Their
value here is that they describe what practitioners do, not that they establish the shape. The
shape follows from the arithmetic in Section 6.3; what needs citation is the claim that real
portfolios are misallocated, and Section 9.1 supplies that.

### 9.3 What the field argues about

Three arguments recur, and they are not resolvable by more analysis.

**Uniformity versus differentiation.** The operations research literature is close to unanimous
that differentiation wins on cost. Practitioners defend uniformity on three grounds: it is
explainable to a customer, it is auditable, and it avoids a political process in which every
category manager argues for A status. The third ground is the real one. A differentiated policy
requires an owner who will defend a deliberately low target on an item, and most organisations
have no such role.

**Where the budget comes from.** Some organisations treat the inventory budget as a constraint
and optimise inside it. Others treat the service target as the constraint and let the budget
follow. The two produce different allocations when the budget is tight, and the same allocation
when it is not. Most companies run the second and are surprised by the first at the annual
planning round.

**Whether to fund service or lead time.** Section 6.5 shows that lead-time reduction is a
multiplicative lever and service targeting is a positional one. Logistics projects are usually
justified on freight or on on-time in-full (OTIF) performance, and the inventory saving they
also produce is left out of the business case because nobody computes the curve. This is one of
the most reliably missed value pools in supply chain analytics, and the reason is
organisational: the benefit accrues to a different budget line from the cost.

### 9.4 Practice in the systems

| Artifact | What it does | What it reveals about practice | Grade |
|---|---|---|---|
| Oracle Inventory Optimization, Inventory Budget Analysis | Compares budget value against achieved service level to check the budget is allocated well; companion pages cover service level versus profit and safety stock analysis | Commercial tools expose the frontier as a budget-versus-service report, so the concept is mainstream and the execution is not | D |
| ERP service-level fields | Typically one service level per item, one family, one number | The uniformity default is encoded in the schema, not chosen by a person | D |
| IEOM 2024 conference paper on dynamic safety stock adjustment | Proposes reallocating a fixed safety stock budget across ABC classes with linear programming and class-specific service bounds | Academic practice is moving toward explicit allocation under a budget constraint | C |

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Service target set without a price | Nobody in the room can say what the target costs per year | The frontier was never drawn, so the target is an aspiration | Money committed without a decision | Draw the curve and state the annual cost of the current target |
| Uniform target across a heterogeneous portfolio | Every item shows the same service level in the parameter table | One number applied to many cost structures | Budget spent on the cheapest-to-serve items instead of the most valuable | Compute marginal rates; report the spread |
| Family not named | Two reports give different service numbers for the same week and both are correct | CSL, fill rate, and ready rate computed by different teams | Arguments that never resolve | Metric dictionary with numerator, denominator, window |
| Marginal cost reported as a band average | The "cost of the next point" equals a five-point average | Chord used where a tangent is needed | Understates the price at the top of the curve | Show both columns; label which answers which question |
| Normal assumed in the tail | No calibration test above 0.95 appears anywhere | Fitted curve extrapolated past the data | Tail cost understated; the curve looks flatter than reality | Compare empirical and fitted quantiles at the decision level |
| Cycle stock inside the frontier | Holding cost at 0.50 service is large and positive | Total inventory plotted instead of safety stock | Slope right, intercept wrong, marginal cost misread | Separate the components; plot the incremental form |
| Lead time treated as a point parameter | Lead time appears as a single number in the curve's caption | Variability omitted from `σ_LTD` | Curve scaled wrongly, often by a factor | Use the full variance expression or simulate |
| Budget shifted to hit a target rather than a rate | Targets move after the budget review, not after a cost review | Allocation driven by affordability rather than efficiency | Ratchet across the portfolio | Re-optimise at the new budget; compare allocations |
| Shortfall cost inherited from a template | Every item in a category shares one `b` | Cost is a property of the commercial context, not the product | Wrong knee for most items | Segment the cost model and publish the segments |
| Frontier without a data vintage | The figure has no date, no window, no sample size | Snapshot quoted as a law | Stale curve used to justify a new commitment | Caption block with vintage and tail count |
| Smooth curve on discrete demand | The plotted line has no steps, but the demand histogram does | Continuous fit applied to integer demand | Decision points hidden between support values | Plot the staircase; report the jump sizes |
| Aggregate service reported alone | One number, usually an average across items, with no shortfall measure beside it | Averaging hides the distribution | Reallocation reversed at the next review | Report unit-weighted service and expected shortfall alongside |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The service coordinate is fixed and agreed | Every cost on the curve is conditional on the family | The curve describes a different promise from the one being managed | Write the six-slot service sentence; ask two analysts to compute it |
| Demand over the protection period is stationary | The curve is a property of one distribution | Promotions, trends, and listings move the whole curve | Compare distributions across sub-periods; flag breaks |
| The protection period is known | It sets the horizon of the distribution | Curve drawn at the wrong horizon, scaled wrongly | Compare modelled and observed order-to-receipt times |
| The distribution is adequate in the tail used | The decision reads the 0.90 to 0.999 region | Tail cost understated; curve too flat at the top | Calibrate at the decision level, with sample size |
| The holding cost rate is a fair average | It scales the vertical axis | The whole curve is mispriced, in the same direction everywhere | Show the build-up; test at the low and high ends of the range |
| Only safety stock varies with service | It justifies the incremental cost function | Cycle stock or pipeline stock also moves, so the slope is wrong | Decompose inventory; check which components respond to the target |
| The shortfall cost is estimable or the target is approved | The knee is placed by the ratio | Knee in the wrong place; allocation in the wrong direction | Compute the implied ratio and ask whether it is plausible |
| Items are independent in the portfolio problem | The allocation treats each item separately | Substitution and joint service requirements couple the items | Test substitution in the data; compute joint availability where a set is ordered together |
| The budget is the binding constraint | The shadow price exists and is positive | The allocation is unconstrained and the shadow price is zero | Check whether relaxing the budget changes the answer |
| The cost curve is convex | A unique interior optimum exists | Multiple local optima, or a corner solution | Check convexity numerically; inspect the staircase case |

Three limits are worth stating plainly.

**The frontier is a model of a decision, not a description of a company.** It says what service
would cost under a stated policy family with stated parameters. It does not say what the company
will achieve, because execution, substitution, and customer patience intervene between the
target and the realised service. Track that gap rather than assuming it away.

**The portfolio allocation assumes the shortfall costs are comparable.** They are, if they are
all in currency per unit short. They are not, if one is a contractual penalty and another is a
goodwill estimate that nobody believes. The allocation inherits whatever incommensurability the
inputs carry.

**The curve is a snapshot and it moves.** A frontier drawn in January is a statement about
January's demand distribution, January's lead times, and January's costs. By December it may be
a historical document being used as a policy instrument, which is the most expensive way to be
out of date.

## 12. Exercises

### Exercise 1 — Read a frontier (recall and application, 30 minutes)

Using the figure in Section 3, identify the flat region, the knee, and the steep tail, and state
the marginal cost marked at each of the three positions in units of euros per percentage point.

**Deliverable:** a one-page annotated copy of the figure with the three regions shaded and the
three marginal costs labelled with their units. **Acceptance criteria:** all three regions
bounded; each marginal cost labelled with currency and unit of service; a sentence stating which
of the three marks answers a "next point" question and which answers a "whole band" question.
**Time:** 30 minutes.

### Exercise 2 — Compute the marginal cost (calculation, 40 minutes)

An item has protection-period mean 800 units, standard deviation 150 units, and a carrying cost
of 9.60 euros per unit per year. Compute the safety stock and annual holding cost at 90, 95, 98
and 99 per cent CSL, the band average for each interval, and the local slope at each point.

**Deliverable:** one table with the `z`, safety stock, cost, band average, and local slope
columns. **Acceptance criteria:** all four `z` values correct to three decimals; band averages
computed against the preceding row; local slopes computed from `hσ/φ(z)`; one sentence explaining
why the local slope at 99 exceeds the 98-to-99 band average. **Time:** 40 minutes.

### Exercise 3 — Three coordinates on one item (calculation, 45 minutes)

For the item in Exercise 2 with an order quantity of 500 units, compute the holding cost of a 95
per cent CSL, a 95 per cent fill rate, and a 99 per cent fill rate.

**Deliverable:** a three-row table with `z`, safety stock, cost, and the service level achieved in
the other family. **Acceptance criteria:** the loss function shown at each step; the CSL and
fill-rate targets produce different quantities; a sentence stating which target is the stronger
promise and why. **Time:** 45 minutes.

### Exercise 4 — Lead time shifts the curve (calculation, 35 minutes)

Weekly demand has a standard deviation of 80 units. Compute the safety stock, the holding cost at
95 per cent CSL, and the local slope at 95 per cent for lead times of 3, 6 and 12 weeks under
independent demand.

**Deliverable:** a four-row table plus the ratio of each row to the 6-week case. **Acceptance
criteria:** `σ_LTD` computed as `80√L`; the ratios equal `√(L/6)` to two decimals; a sentence
stating whether the curve moved to a new point or changed scale. **Time:** 35 minutes.

### Exercise 5 — The implied cost ratio (analysis, 25 minutes)

For CSL targets of 0.80, 0.90, 0.95, 0.99 and 0.999, compute the implied ratio of shortfall cost
to holding cost, and state in one sentence what each target asserts about the item.

**Deliverable:** a five-row table with the ratio and the assertion. **Acceptance criteria:** the
ratio computed as `α/(1−α)`; each assertion phrased as a commercial claim rather than a
statistical one; a closing sentence on which of the five is defensible for a commodity item.
**Time:** 25 minutes.

### Exercise 6 — Portfolio allocation (calculation, 60 minutes)

Four items share a holding-cost budget. Parameters: W (`σ` 100, `h` 4.80, 12 cycles/yr, `b` 20);
X (`σ` 50, `h` 60.00, 4 cycles/yr, `b` 50); Y (`σ` 250, `h` 2.40, 24 cycles/yr, `b` 30); Z (`σ`
20, `h` 150.00, 3 cycles/yr, `b` 300). All are at a uniform 95 per cent CSL.

**Deliverable:** a table of marginal rates at 95 per cent, a ranking, and a direction of
reallocation for each item. **Acceptance criteria:** rates computed as `h/(p·f·b)` with `p =
0.05`; the ranking stated; the direction justified by the rate rather than by item value; a
sentence naming the item whose cost structure most resembles a candidate for a floor. **Time:**
60 minutes.

### Exercise 7 — Critique a frontier (critique, 40 minutes)

A colleague presents a frontier drawn on 18 months of weekly demand, fitted with a normal
distribution, using a 22 per cent carrying rate quoted as "standard", and plotted against total
inventory value rather than safety stock. Write a critique.

**Deliverable:** five findings, each naming the defect, its direction of bias, and the check that
would resolve it. **Acceptance criteria:** at least three of the five identify a direction of
bias rather than a missing item; the tail question is addressed with a specific test; the
critique endswith what can still be concluded from the curve. **Time:** 40 minutes.

### Exercise 8 — Specify the data pull (design, 45 minutes)

Write the specification for the dataset needed to draw a frontier for one product family.

**Deliverable:** a field list with source system, grain, and known defect for each field, plus a
statement of the minimum observation count at the highest service level plotted. **Acceptance
criteria:** at least twelve fields; censoring addressed explicitly; the cost fields separated
into components rather than a single rate; the tail-count rule stated as a number. **Time:** 45
minutes.

### Exercise 9 — The memo (design, 45 minutes)

Write a one-page memo to a finance partner justifying a change from a uniform 95 per cent target
to a differentiated allocation.

**Deliverable:** the memo, with the current cost, the proposed cost, the change in expected
shortfall, and the change in the reported service measure. **Acceptance criteria:** the reported
service measure that gets worse is stated rather than omitted; the shortfall costs are shown
with a range; the decision requested is one specific approval, not a direction of travel.
**Time:** 45 minutes.

## 13. Self-check answers

### 13.1 Exercise 1

The flat region runs from about 0.50 to 0.85, the knee from 0.85 to 0.97, and the steep tail
above 0.97. The three marks are 82 euros per point at 0.90, 140 at 0.95, and 540 at 0.99. The
mark at 0.90 is close to a band average because the curve is still gently sloped there; the mark
at 0.99 is a tangent and answers the "next point" question. A submission that labels all three
as band averages has not distinguished the two computations.

### 13.2 Exercise 2

With `h = 9.60` and `σ = 150`, `hσ = 1,440` euros per year per standard deviation, which is the
same product as the worked example. The four rows are therefore identical: at 0.90, `z` is
1.2816, safety stock 192.2 units, cost 1,845; at 0.95, `z` 1.6449, 246.7 units, 2,369; at 0.98,
`z` 2.0537, 308.1 units, 2,957; at 0.99, `z` 2.3263, 348.9 units, 3,350. The band averages are
104.8, 196.1 and 392.5 euros per point; the local slopes are 82.1, 139.6, 296.2 and 540.3.

That identity is the teaching point: the frontier depends on `hσ`, not on the two factors
separately. The local slope at 99 exceeds the 98-to-99 band average because the slope rises
across the interval and the average is a chord drawn under it.

### 13.3 Exercise 3

At a 95 per cent CSL, `z = 1.6449`, safety stock 246.7 units, cost 2,369, and the loss function
`L = 0.020889` gives `σL/Q = 150 × 0.020889 / 500 = 0.00627`, a fill rate of 99.37 per cent.

At a 95 per cent fill rate, `L(z) = 0.05 × 500 / 150 = 0.16667`, which gives `z = 0.607`, safety
stock 91.1 units, cost 874, and a CSL of 72.8 per cent.

At a 99 per cent fill rate, `L(z) = 0.03333`, `z = 1.508`, safety stock 226.2 units, cost 2,172,
and a CSL of 93.4 per cent.

A 95 per cent fill rate costs 874 against 2,369 for a 95 per cent CSL, a factor of 2.7 on the
same item, because it tolerates a stock-out on 27 per cent of cycles. The 99 per cent fill rate
costs 2,172 and delivers a 93.4 per cent CSL: close to the 95 per cent CSL in money, and still a
materially different promise.

### 13.4 Exercise 4

`σ_LTD = 80√L`. At 3 weeks, `σ` is 138.6, safety stock at 95 per cent 228.0 units, holding cost
1,642, and the local slope 97.9 euros per point. At 6 weeks, 196.0, 322.4, 2,321 and 138.5. At 12
weeks, 277.1, 455.8, 3,282 and 195.8.

The ratios are `√(3/6) = 0.707`, `1.000` and `√(12/6) = 1.414`, which are also the ratios of
every cost and every marginal cost to the 6-week case. The curve changed scale, not position,
and `C(0.99)/C(0.95)` remains 1.414 in all three rows.

### 13.5 Exercise 5

| CSL | Implied `b/h` | The assertion |
|---:|---:|---|
| 0.80 | 4 : 1 | Running out is inconvenient but survivable |
| 0.90 | 9 : 1 | Running out costs about a working day of margin |
| 0.95 | 19 : 1 | Running out is a serious commercial event |
| 0.99 | 99 : 1 | Running out threatens the relationship |
| 0.999 | 999 : 1 | The item must not fail |

For a commodity item with substitutes available within a day, 0.90 to 0.95 is the defensible
range and 0.999 is indefensible. The test is not the number but whether anyone can state the
right-hand column without laughing.

### 13.6 Exercise 6

The rates are `h / (0.05 · f · b)`: W `4.80 / 12.0 = 0.400`, X `60.00 / 10.0 = 6.000`, Y
`2.40 / 36.0 = 0.067`, Z `150.00 / 45.0 = 3.333`.

Ranking from cheapest to most expensive marginal service: Y, W, Z, X. Reallocate toward Y and W,
away from X and Z. Item X is the candidate for a floor: expensive to hold and rarely ordered,
where a criticality argument the `b` of 50 does not capture may justify protection. Note that Y,
the cheapest, has the highest variability of the four, because the rate depends on `h`, `f` and
`b` and not on `σ`.

### 13.7 Exercise 7

Five findings. **One:** eighteen months gives at most 78 weekly observations, so the 0.99
quantile rests on less than one observation; the tail is extrapolated, not estimated. Direction:
understates tail cost. Check: count observations above the highest plotted service level.
**Two:** a normal fit on weekly demand containing a promotion will understate the upper tail.
Direction: understates tail cost and the true curve is steeper. Check: empirical versus fitted
quantiles at 0.90, 0.95, 0.99. **Three:** a 22 per cent rate quoted as standard has no owner and
no build-up. Direction: unknown, probably understated for perishable or high-risk items. Check:
component build-up. **Four:** total inventory value includes cycle stock, so the intercept is
inflated and the marginal cost is understated as a share of the total. Direction: mislabels the
level, preserves the slope. Check: separate the two components. **Five:** no vintage and no
owner on the figure, which makes the curve unmaintainable. What survives: the shape and the
relative ranking of items, provided the tail is presented as illustrative.

### 13.8 Exercise 8

The field list should include: item identifier and location; demand quantity by period with a
censoring flag and a stock-out flag; promised and actual receipt dates for every replenishment;
order quantity and order date; unit cost at the stated price year; the capital charge rate;
warehouse cost per unit stored; write-off quantity and value by age bucket; the contractual
shortfall penalty per unit or per event; and the stated target with its family. Censoring is the
field most often omitted and the one that biases the curve most. The tail rule should be
explicit: at least twenty observations beyond the highest service level plotted, or the curve is
labelled extrapolated.

### 13.9 Exercise 9

A rubric rather than an answer. Strong memos open with the current annual holding cost and the
current expected shortfall, state the proposed allocation with the shadow price, and put the
deteriorating measure in the second paragraph rather than the appendix. They show shortfall
costs as ranges and say which item's allocation is sensitive to the range. They ask for one
decision: approval of a target vector and a review date. Weak memos open with the theory, quote a
single aggregate service number, omit the item whose service falls, and ask for support rather
than a decision.

## 14. Interview questions

**1. "What is a cost-service frontier?"**

Strong: a curve of achievable service-cost pairs for a stated policy family, with the axes and
their units named, and the target as a point on it. Weak: "the trade-off between service and
cost", which is a topic rather than an object.

**2. "Why does the curve steepen?"**

Strong: the normal density falls in the tail, so a fixed increment of probability needs a
growing increment of standard deviations; local slope `hσ/φ(z)`. Weak: "it gets harder at the
top", which restates the question.

**3. "Is 95 per cent a good service level?"**

Strong: not a service level until the family and item are named, and as a CSL it asserts a
19-to-1 shortfall-to-holding ratio, which is a commercial claim that must be true for the item.
Weak: yes, it is the industry standard.

**4. "How would you compute the marginal cost of the next point?"**

Strong: distinguish band average from local slope, give `hσ/φ(z)`, and say which question each
answers. Weak: divide total cost by the number of points.

**5. "Why is 95 per cent everywhere a bad policy?"**

Strong: the marginal rate differs across items by one or two orders of magnitude, so a uniform
target spends where it buys least; cite the measured stock-value reduction from
differentiation. Weak: it is unfair to some items, which is a different and weaker argument.

**6. "How do you allocate a fixed inventory budget across two hundred items?"**

Strong: equalise marginal rates at a shadow price, target tail probability rather than service
level, and expect the reported average service to fall. Weak: "ABC analysis", which is a
segmentation, not an allocation.

**7. "What is the difference between the item and portfolio frontiers?"**

Strong: the portfolio curve is traced by re-optimising under a budget constraint that couples
the items, so it cannot be obtained by summing curves or applying one target. Weak: the
portfolio frontier is the sum of the item frontiers.

**8. "What does a lead-time reduction do to the curve?"**

Strong: it scales the curve vertically by roughly the square root of the lead time under
independence, moving every point by the same factor, with the shape invariant under stated
conditions. Weak: it lowers inventory, which is true and does not distinguish a shift from a
movement along the curve.

**9. "What data do you need to draw one?"**

Strong: a demand distribution at the decision grain with censoring handled, a carrying cost with
its components, a shortfall cost or an approved target, a lead-time distribution, and a stated
tail observation count. Weak: "demand history and costs".

**10. "When is a frontier worse than nothing?"**

Strong: when it is drawn from a tail the data cannot support, because it carries visual authority
that forecloses the judgement a missing curve would have invited. Weak: never, more data is
always better.

**11. "How would you explain this to a finance partner?"**

Strong: lead with the annual cost of the current target and the fact that the same money buys
more service if allocated differently, with shortfall costs as a range. Weak: lead with the
distributional theory.

**12. "What do you do when nobody can estimate the stockout cost?"**

Strong: invert the question, state the target, compute the implied ratio, and ask whether it is
plausible, which converts an unanswerable question into an answerable one. Weak: use an industry
benchmark, which imports somebody else's commercial context.

## 15. Key terms

- **Cost-service frontier** — the curve of achievable service-cost pairs for a stated policy.
- **Marginal cost of service** — the annual cost of one more unit of service at a stated point.
- **Local slope** — the derivative `dC/dα`, a tangent at one point.
- **Band average** — a chord across an interval; the average price of a point in that band.
- **Knee** — the region where marginal cost begins to rise sharply; positioned by `b/h`.
- **Steep tail** — the region above roughly 0.97 CSL where each point is a purchase.
- **Vertical asymptote** — the unbounded cost of perfect service at 100 per cent.
- **Implied cost ratio** — `α/(1−α)`, the `b/h` a stated target encodes.
- **Economic service level** — where marginal holding cost equals marginal shortfall avoided.
- **Shadow price of service** — the marginal value of relaxing the budget or service constraint.
- **Efficient allocation** — no item's service can rise without another's falling at the budget.
- **Marginal rate** — cost per unit of shortfall avoided; the allocation coordinate.
- **Uniform service policy** — one target for every item; efficient only by coincidence.
- **Service differentiation** — deliberately different targets, justified by cost or criticality.
- **Safety stock** — the buffer the frontier prices; the part of inventory that moves with service.
- **Cycle stock** — inventory from batching; shifts the curve's intercept, not its slope.
- **Carrying cost (`h`)** — currency per unit per year; scales the whole curve.
- **Shortfall cost (`b`)** — currency per unit short; places the knee.
- **Protection period** — lead time plus review period; the horizon of the distribution.
- **Lead-time demand (LTD)** — demand realised between order placement and receipt.
- **Coefficient of variation (CV)** — standard deviation over mean; sets the quantity, not the rate.
- **Standard normal density (`φ`)** — the tail thinning that makes the curve steep.
- **Standard normal loss (`L`)** — expected exceedance; links CSL to fill rate.
- **Fill-rate frontier** — the same item plotted in a quantity coordinate; a different position.
- **Staircase frontier** — the shape under discrete demand; vertical jumps at support points.
- **Price-criticality ratio (PCR)** — a ranking quantity used in differentiated fill-rate research.
- **Parameter register** — the record of targets, owners, and the shadow price.
- **Data vintage** — the window, sample size, and fitting method behind a curve.

## 16. Sources

**Peer-reviewed and academic**

- Teunter, Babai and Syntetos (2010), "ABC Classification: Service Levels and Inventory Costs," *Production and Operations Management* 19(3), 343-352. DOI 10.1111/j.1937-5956.2009.01098.x. Grade A.
- Teunter, Syntetos and Babai (2017), "Stock keeping unit fill rate specification," *European Journal of Operational Research* 259(3), 917-925. DOI 10.1016/j.ejor.2016.11.017. Grade A.
- Chopra, Reinhardt and Dada (2004), "The Effect of Lead Time Uncertainty on Safety Stocks," *Decision Sciences* 35(1). DOI 10.1111/j.1540-5414.2004.02332.x. Grade A.
- Rosling (2002), "Inventory Cost Rate Functions with Nonlinear Shortage Costs," *Operations Research* 50(6), 1007-1017. DOI 10.1287/opre.50.6.1007.346. Grade A.
- Schneider (1981), "Effect of service-levels on order-points or order-levels in inventory models," *International Journal of Production Research*. DOI 10.1080/00207548108956694. The survey that codified the cycle-service, fill-rate and ready-rate families. Grade A.
- Soriano and Gross (1969), "The Effect of Reducing Leadtime on Inventory Levels - Simulation Analysis," *Management Science* 16(2), B61-B76. DOI 10.1287/mnsc.16.2.B61. Grade A.
- "Operations research models and methods for safety stock determination: A review," PMC7550265, 2020. Open-access systematic review; concludes that a Gaussian assumption can be fundamentally flawed for skewed or erratic demand. Grade A.
- Rolf et al. (2024), "Optimizing Safety Stock Placement in Large Real-World Automotive Supply Networks Using the Guaranteed-Service Model," *Proceedings of the 57th Hawaii International Conference on System Sciences*. Grade B.

**Official, standards, and regulatory**

- ISO 3534-1, *Statistics - Vocabulary and symbols - Part 1*. Grade B.
- ISO 80000-1, *Quantities and units - Part 1: General*; ISO 8601, *Date and time*. Grade B.
- ISO 31000:2018, *Risk management - Guidelines*. Grade B.
- IAS 2 *Inventories*; ASC 330 *Inventory*; IAS 37 *Provisions*; IFRS 15 / ASC 606 revenue. Grade B.
- Sarbanes-Oxley Act 2002, s.404; Federal Reserve SR 11-7 (by analogy only). Grade B.
- ASCM, *SCOR Digital Standard*. A reference model, not a regulation. Grade C.

**Industry, analyst, and practitioner**

- De Kok, "125 years of inventory management, 100 years of EOQ and 40 years of vLm," CWI institutional publication. Historical survey of the safety-stock formulary and the 95 per cent convention. Grade B.
- King, "The CodeCrack: Understanding safety stock and mastering its equations," *APICS Magazine*, July/August 2011. Practitioner statement of the `z`-table convention. Grade C.
- Georgia Tech Supply Chain and Logistics Institute, inventory planning teaching material on inventory versus service level at 6- and 12-week lead times. Grade C.
- Hartunian, "Four Steps to Service Level-Driven Inventory Planning," *Supply Chain Management Review*. Grade C.
- IEOM Society 2024 North American Conference proceedings, "Proposing a New Dynamic Safety Stock Adjustment System: Balancing Service Level Targets and Financial Constraints." Grade C.

**Vendor and advisory**

- Oracle, *Inventory Optimization User's Guide*, "Inventory Budget Analysis." Grade D.
- Umbrex, "Inventory Trade-Offs: Service, Cost and Working Capital." Grade D.
- MetricGate, inventory management and continuous-review calculator documentation. Grade D.

**Note on citations.** Figures attributed to vendor or advisory sources are reported as those sources state them and are not independently verified. Where a specific number is uncertain, the text says so rather than quoting precision it cannot support.

## 17. Links

- This module's index: [`_index.md`](_index.md)
- This part's index: [`../_index.md`](../_index.md)
- Prerequisite, quantiles and decisions: [`../08_hierarchical_intermittent_probabilistic_forecasting/10_quantiles-decisions.md`](../08_hierarchical_intermittent_probabilistic_forecasting/10_quantiles-decisions.md)
- Prerequisite, error metrics and cost: [`../07_time_series_forecasting/09_error-metrics-cost.md`](../07_time_series_forecasting/09_error-metrics-cost.md)
- Related, demand sensing and post-launch learning: [`../09_demand_sensing_promotions_npi/10_post-launch-learning.md`](../09_demand_sensing_promotions_npi/10_post-launch-learning.md)
- Part I index: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module survey file: [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
- Sibling 02, EOQ and its limits: [`02_eoq-limits.md`](02_eoq-limits.md)
- Sibling 03, the single-period case: [`03_newsvendor.md`](03_newsvendor.md)
- Sibling 04, safety stock: [`04_safety-stock.md`](04_safety-stock.md)
- Sibling 05, service metrics: [`05_service-metrics.md`](05_service-metrics.md)
- Sibling 06, policy families: [`06_policy-families.md`](06_policy-families.md)
- Sibling 07, lead-time demand distributions: [`07_lead-time-demand.md`](07_lead-time-demand.md)
- Sibling 08, multi-echelon and MEIO: [`08_multi-echelon.md`](08_multi-echelon.md)
- Sibling 09, pooling and postponement: [`09_pooling-postponement.md`](09_pooling-postponement.md)
- Sibling 10, segmentation, DDMRP, and shelf life: [`10_segmentation-shelf-life.md`](10_segmentation-shelf-life.md)
