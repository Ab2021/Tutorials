# Service Metrics: CSL, Fill Rate, and OTIF — definitions that are not interchangeable

> Module 10 — Inventory Optimization and Multi-Echelon Planning
> Topic 05 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 10 topics 01 and 04; Module 08 topic 10
> Version 2026.5

---

## 1. Why this topic exists

Two reports land in the same Monday meeting. The distribution centre reports a 96.4 per cent
service level for the month. The commercial team reports that the same distribution centre achieved
88.1 per cent on-time in-full (OTIF) against its largest customer. Forty minutes go into arguing
about which number is right. Neither analyst made an arithmetic mistake. They counted different
objects, over different windows, with different denominators, and nobody in the room can say what
either number means.

This is the normal condition of service measurement in supply chains, and it is more expensive than
it looks.

**Policy is sized against a number the business does not use.** A safety stock is computed at a 95
per cent cycle service level (CSL). The customer contract is written as a 98 per cent OTIF. Nobody
has shown that the first produces the second, and the two are not related by any formula.

**Improvement projects target the measure instead of the outcome.** A team reduces stockout cycles
by holding more stock of the two highest-volume items. CSL rises from 91 to 96 per cent. Unit fill
rate barely moves, because the remaining stockouts were on the long tail. The programme is declared
a success and the customer experience is unchanged.

**The measure is gamed, usually without anyone deciding to game it.** A warehouse measured on order
fill rate learns to hold small orders rather than ship them short. A supplier measured against the
customer's revised promise date learns that rescheduling is cheaper than expediting. A planner
measured on ready rate learns that one unit on the shelf produces a perfect score.

The purpose of this topic is not to pick the best measure. There is no best measure. The purpose is
to make you the person who can say, in one sentence, what a service number counts and what it
therefore cannot tell you.

> **Note:** This file defines the measures. It does not derive the buffer. Module 10 topic 04
> derives safety stock from a service measure; here the same conversion is treated as a measurement
> question. Topic 06 turns these measures into policy families. Availability as a driver of demand
> is Module 09 topic 09 and is not re-taught here. Network-level service trade-offs belong to
> Module 11.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** cycle service level (CSL), unit, line, and order fill rate, ready rate, order
   fulfilment lead time (OFLT), OTIF, and perfect order, each with numerator, denominator, and unit
   of account.
2. **Convert** a CSL target into an expected fill rate and a fill target into an implied CSL using
   the standard normal loss function, and **state** the three assumptions the conversion requires.
3. **Predict** whether the gap between CSL and fill rate will be large or small for an item, from
   the coefficient of variation of demand over a replenishment cycle.
4. **Compute** demand-weighted, line-weighted, and order-weighted versions of the same performance
   and **explain** why they can differ by more than eighty percentage points.
5. **Decompose** an OTIF figure into its definitional choices — date basis, window, early
   tolerance, in-full grain, cause attribution, period attribution — and **estimate** the range the
   same operation could report.
6. **Distinguish** a target, an achieved rate, and a measurement bias, and **identify** which moved
   when a reported number changes.
7. **Audit** a perfect order index for whether it is a product of component rates or a count, and
   **bound** the difference using the Fréchet inequalities.
8. **Name** the perverse incentive each measure creates and the counter-metric that neutralises it.
9. **Write** a seven-field service metric specification.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a ladder of counting objects. Every service measure climbs this ladder at a
different rung, and the rung determines what the number can and cannot see.

```text
  WHAT IS COUNTED          MEASURE                      DENOMINATOR
  ---------------------------------------------------------------------
  cycles                   cycle service level (CSL)    replenishment cycles
      |                    "did this cycle stock out?"
      v
  time                     ready rate                   periods of time
      |                    "was stock on hand?"
      v
  units                    unit fill rate               units demanded
      |                    "what share of units shipped?"
      v
  order lines              line fill rate               order lines
      |                    "what share of lines complete?"
      v
  orders                   order fill rate              orders
      |                    "what share of orders complete?"
      v
  deliveries in a window   OTIF                         deliveries promised
                           "on time AND complete?"
```

A cycle either stocked out or did not, so CSL throws away the depth of the shortfall. Units keep the
depth and throw away the customer. Lines keep the order structure and throw away quantity. Orders
keep the whole order and throw away partial credit. OTIF keeps the commercial promise and throws
away the inventory. Moving down the ladder makes the number more commercially meaningful and less
diagnostically useful.

### 3.1 The central algebra: quantile versus partial expectation

CSL asks a **probability** question. Fill rate asks a **magnitude** question. That difference
generates everything else in this file.

```text
   CSL  = P(D <= S) = Phi(z)          where z = (S - mu) / sigma
   E[(D - S)+] = sigma * L(z)
   L(z) = phi(z) - z * (1 - Phi(z))        standard normal loss function
   unit fill rate  beta = 1 - sigma * L(z) / Q
```

`Q` is demand per replenishment cycle, approximately the order quantity, and
`CV_cycle = sigma / Q`. Three consequences follow.

- **CSL is a quantile.** It depends on `z` alone, so two items at the same target carry the same
  buffer in standard deviations regardless of volume.
- **Fill rate is a partial moment scaled by `sigma / Q`.** It depends on `z` and on the ratio of
  demand variability to cycle demand.
- **The conversion is asymmetric.** CSL to fill is a direct evaluation. Fill to CSL requires
  inverting `L`, which has no closed form.

`L(z)` is the expected shortfall in units of `sigma` for a buffer placed `z` standard deviations
above the mean. At `z = 0`, `L(0) = phi(0) = 0.3989`, so a buffer at the mean still leaves a
shortfall of four tenths of a standard deviation per cycle.

| Target CSL | `z` | `L(z)` | Expected shortfall per cycle, in `sigma` |
|---|---:|---:|---:|
| 90% | 1.282 | 0.0474 | 0.047 |
| 95% | 1.645 | 0.0209 | 0.021 |
| 99% | 2.326 | 0.0035 | 0.004 |

The collapse of `L(z)` explains why high service is cheap in probability and expensive in inventory.
Moving from 90 to 95 per cent cuts expected shortfall by 56 per cent at a buffer cost of 0.36
standard deviations; moving from 95 to 99 cuts it by a further 83 per cent at a cost of 0.68.

### 3.2 The curve that explains the paradox

Fix CSL at 95 per cent, so `z = 1.645` and `L(z) = 0.0209` for every item. Fill rate then depends on
one variable: `beta = 1 - 0.0209 * CV_cycle`.

```text
  fill rate at a fixed 95% CSL, as a function of CV of demand per cycle

  100% |*
       | *
   98% |   *
       |      *
   95% |            *
       |                    *
   90% |                              *
       |                                        *
   80% |                                                  *
       +----+-----+-----+-----+-----+-----+-----+-----+----->
       0.05  0.1  0.25  0.5   1.0   2.0   3.0   5.0   10   CV_cycle

  beta = 1 - 0.0209 x CV        CV = sigma over the cycle / Q
```

| Item type | Mean demand per cycle | `sigma` per cycle | `Q` | `CV_cycle` | Fill at CSL 95% |
|---|---:|---:|---:|---:|---:|
| Fast, smooth | 4,800 | 620 | 4,800 | 0.13 | 99.73% |
| Fast, lumpy (three customers) | 4,800 | 2,100 | 4,800 | 0.44 | 99.08% |
| Slow, smooth | 120 | 15 | 120 | 0.13 | 99.73% |
| Slow, Poisson-like | 6 | 2.45 | 6 | 0.41 | 99.14% |
| Slow, lumpy | 4 | 10 | 4 | 2.50 | 94.78% |
| Very lumpy spare part | 1.2 | 6 | 1.2 | 5.00 | 89.56% |

**It is not slowness that produces the gap; it is lumpiness.** The slow, smooth item behaves exactly
like the fast, smooth one. The fast, lumpy item — three customers ordering in pallets — behaves
almost like a spare part. Under independent demand, `CV_cycle` falls roughly as
`1 / sqrt(mean demand per cycle)`, which is why fast movers sit on the left of the curve. Bulk
ordering, few customers, and intermittency move an item to the right regardless of volume.

| Item type | `CV_cycle` | `L(z)` for 95% fill | `z` | Implied CSL | Buffer in `sigma` |
|---|---:|---:|---:|---:|---:|
| Fast, smooth | 0.13 | 0.385 | 0.03 | 51.2% | 0.03 |
| Fast, lumpy | 0.44 | 0.114 | 0.76 | 77.7% | 0.76 |
| Slow, Poisson-like | 0.41 | 0.122 | 0.75 | 77.2% | 0.75 |
| Slow, lumpy | 2.50 | 0.020 | 1.67 | 95.2% | 1.67 |
| Very lumpy spare | 5.00 | 0.010 | 1.94 | 97.4% | 1.94 |

Five items, one contract clause reading "95 per cent fill rate", five buffers spanning 0.03 to 1.94
standard deviations. The fast mover needs essentially nothing: its order-up-to level sits at mean
cycle demand. Below a `CV_cycle` of about 0.125 the implied buffer turns negative, which is not an
error but a statement that the cheapest way to hit a fill target on a smooth high-volume item is to
accept frequent, trivial stockouts.

### 3.3 Perfect order as a product of conditional probabilities

```text
                          all orders in the period (80)
                                     |
                    on time? --------+-------- in full?
                    /       \                  /       \
               yes 95.0%   no 5.0%        yes 97.5%  no 2.5%
                   |                          |
                   +------- damage free? ----+
                            /        \
                       yes 99.0%    no 1.0%
                           |
                    documentation accurate?
                          /        \
                     yes 96.0%    no 4.0%
                          |
              product = 0.950 x 0.975 x 0.990 x 0.960
                      = 0.8803  ->  88.0% "perfect"
```

Four components at 95 per cent do not produce 95 per cent. They produce `0.95^4 = 0.8145`. Four at 97
per cent produce `0.97^4 = 0.8853`. A customer who receives four orders in five perfectly still
receives one defective order in five.

The product equals the true rate only if the four failure modes are independent across orders. They
rarely are, so use the Fréchet inequalities for hard bounds from the marginals alone:

```text
   1 - (0.05 + 0.025 + 0.01 + 0.04)  <=  P(perfect)  <=  min(0.95, 0.975, 0.99, 0.96)
              1 - 0.125  =  0.875                      =  0.950
                    87.5%      <=      P(perfect)      <=   95.0%
```

The independence value, 88.0 per cent, sits near the lower bound. That is not a coincidence: when
four things can each go wrong, most failing orders fail on exactly one dimension, so the "any
failure" probability is close to the sum of the marginal failure probabilities.

> **Note:** The multiplicative composite is the American Productivity and Quality Center (APQC)
> benchmarking form. The Supply Chain Operations Reference (SCOR) model's reliability metric
> RL.1.1 counts orders directly — perfect orders divided by total orders. The two are different
> estimators of the same quantity and agree only under independence.

### 3.4 Weighting changes the number, not the performance

A distributor ships ten orders in a week. Nine are small, ten units each. One is large, ten thousand
units. The large order ships nothing; the nine small orders ship complete.

```text
   unit-weighted fill rate    = (10,090 - 10,000) / 10,090 = 0.89%
   order-weighted fill rate   = 9 / 10                     = 90.0%
```

The same week, the same warehouse, the same customers. One measure says the operation failed almost
totally; the other says it succeeded nine times out of ten. Both are correct, and a company
benchmarking itself externally will not know which form the benchmark used.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Service level | Any quantified statement about the fraction of demand, cycles, orders, or time in which a supply obligation is met | Used as if it named one specific measure | Enumeration; must be qualified |
| Cycle service level (CSL) | Probability that demand over one replenishment cycle is met from stock, `P(D <= S)` | Reported as fill rate; reported without the cycle definition | Proportion per cycle |
| Fill rate (FR) | Fraction of demand met from stock without backorder or substitution | Reported without stating unit, line, or order grain | Proportion |
| Unit fill rate | Units shipped from stock divided by units demanded over a stated window | Called "the fill rate" when the contract means lines | Units / units |
| Line fill rate | Order lines shipped complete divided by total order lines | Confused with unit fill when lines are uniform | Lines / lines |
| Order fill rate | Orders shipped complete divided by total orders | Confused with OTIF, which adds a time test | Orders / orders |
| Case fill rate | Cases shipped complete divided by cases ordered | Used interchangeably with unit fill after unit conversion | Cases / cases |
| Ready rate | Fraction of time that stock on hand is positive | Treated as equivalent to fill rate | Time / time |
| Alpha service level | Synonym for CSL in the inventory literature; sometimes used for ready rate | Ambiguous across texts | Proportion |
| Beta service level | Synonym for fill rate in the inventory literature | Assumed identical to CSL | Proportion |
| Gamma service level | Time-weighted measure of backorder magnitude | Rarely computed; often claimed | Units × time |
| Normal loss function | `L(z) = phi(z) - z(1 - Phi(z))`, expected exceedance of a standard normal above `z` | Omitted when converting CSL to fill | Dimensionless, in `sigma` |
| Partial expectation | Expected value of a random variable above a threshold, `E[(D - S)+]` | Confused with the probability of exceeding the threshold | Units |
| Protection period | Review period plus lead time; the span the buffer covers | Assumed equal to lead time | Weeks or days |
| Coefficient of variation per cycle | `sigma` of demand over a replenishment cycle divided by cycle demand | Computed on annual demand | Dimensionless |
| Replenishment cycle | Interval between two consecutive replenishment arrivals at a stocking point | Defined differently by different reports | Weeks |
| Stockout | State in which on-hand stock cannot meet a demand that arrives | Counted per unit, per line, or per cycle without saying which | Event |
| Stockout cycle | Replenishment cycle containing at least one stockout | Counted as a stockout | Count |
| Expected shortfall | Expected units unserved per cycle, `sigma * L(z)` | Reported as a probability | Units per cycle |
| On-time delivery (OTD) | Fraction of deliveries arriving within the agreed window | Computed against different dates in different systems | Deliveries / deliveries |
| OTIF | Fraction of deliveries both on time and complete, per the contract's definition | Assumed to mean the same thing at every customer | Deliveries / deliveries |
| In-full | The quantity condition: everything ordered was delivered | Measured at order, line, or case grain without stating which | Proportion |
| On-time basis | Reference date used to judge punctuality: requested, committed, must-arrive-by, or current promise | Left unstated in most reports | Date field |
| Must-arrive-by date (MABD) | Retailer-specified latest acceptable arrival date | Treated as a delivery appointment | Date |
| Customer requested date (CRD) | Date the customer asked for at order placement | Replaced silently by a later confirmed date | Date |
| Delivery window | Interval within which arrival counts as on time, including early tolerance | Assumed to be the whole day | Hours or days |
| Early tolerance | How early a delivery may arrive and still count as on time | Assumed infinite by suppliers, zero by some retailers | Hours or days |
| Appointment slot | Booked dock time at a receiving location | Confused with the delivery window | Timestamp |
| Cause attribution | Rule assigning a late delivery to supplier, carrier, or customer | Assumed neutral; usually negotiated | Code |
| Period attribution | Rule deciding which reporting period a late order belongs to | Set by default in the reporting tool | Requested or delivered period |
| Order fulfilment lead time (OFLT) | Elapsed time from order receipt to delivery at the customer | Confused with OTD, which is a rate not a duration | Hours or days |
| Perfect order | Order meeting every component test: on time, in full, undamaged, correctly documented | Reported without the component list | Orders / orders |
| Perfect order index | Composite of component rates, either multiplied or counted | The two forms are compared as if identical | Proportion |
| Demand-weighted | Averaging with units as weights | Called "volume-weighted" | Units |
| Line-weighted | Averaging with order lines as weights | Called "order-weighted" | Lines |
| Order-weighted | Averaging with orders as weights | Called "unweighted" | Orders |
| Target | Level a contract or management commits to | Compared with an achieved rate from a different definition | Proportion |
| Achieved rate | Level a specified measurement system reports | Treated as ground truth | Proportion |
| Measurement bias | Systematic difference between an achieved rate and the quantity it purports to measure | Treated as a data-quality defect | Percentage points |
| Denominator management | Improving a rate by removing hard cases from the denominator | Described as an exclusion policy | Practice |
| Service metric specification | Seven-field statement of measure, denominator, weighting, window, exclusions, source, and target basis | Replaced by a metric name in a dashboard tooltip | Document |

Two rows deserve expansion, because each is where a real argument happens.

**Fill rate versus ready rate.** Ready rate is the fraction of *time* that stock is on hand, not the
fraction of demand served. An item with one unit permanently on the shelf and no demand has a 100
per cent ready rate and an undefined fill rate. The two coincide only under restrictive demand
assumptions — essentially Poisson arrivals, where the fraction of time in stock equals the fraction
of arrivals served, a property known as PASTA. For lumpy demand, a high ready rate can conceal a
poor fill rate, because stock is present during the many quiet periods and absent during the few
busy ones.

**OTD versus OTIF.** OTIF is always less than or equal to OTD, and the gap is the share of
deliveries that arrived on time but short. A supplier reporting 97 per cent OTD and 89 per cent OTIF
is telling you that eight points of its punctuality is bought by shipping incomplete orders on
schedule.

## 5. The mechanics, step by step

The procedure runs as a sequence because the choices compound: the wrong unit of account makes the
denominator wrong, the wrong denominator makes the rate wrong, and the wrong rate makes every
comparison with a target meaningless.

### Phase A — Fix the counting object before computing anything

**Step 1. Name the unit of account.** Input: the question and the contract that will consume the
answer. Output: cycles, time periods, units, cases, order lines, orders, or deliveries. Branch: is
the consumer asking "how often did we fail?" or "how much did we fail by?" The first wants a count
of cycles or orders, the second wants units; one metric cannot answer both.

**Step 2. Name the denominator explicitly, with its filter.** Input: the transaction table and the
intended population. Output: a written denominator, for example "all customer order lines released
in the period, excluding intercompany and sample orders". Branch: does the denominator include the
failures? `shipped / ordered` includes failures. `on-time / closed deliveries` excludes every
delivery still open, which is precisely the population that is late.

**Step 3. Name the weighting.** Input: the population and its size distribution. Output: unit-,
line-, order-, or time-weighted. Branch: is the population skewed? If a few orders carry most of the
volume, unit and order weighting diverge sharply, as section 3.4 shows.

**Step 4. Name the window and the attribution rule.** Input: the reporting calendar and the
available transaction dates. Output: a measurement window plus a rule for which period a late item
belongs to. Branch: requested date or delivered date? Step 20 shows the choice moving a rate by 7.8
points with no change in performance.

**Step 5. Name the source system and the extract date.** Input: enterprise resource planning (ERP),
warehouse management system (WMS), transport management system (TMS), electronic data interchange
(EDI) messages, customer portals. Output: a system name and a timestamp. Branch: do two systems
disagree about the same event? They will. The ERP records the requested ship date, the TMS the
actual departure, the customer's portal the receipt.

### Phase B — Compute the inventory measures

**Step 6. Define the replenishment cycle, and make the definition operational.** Input: the policy
family — continuous review `(s, Q)` or periodic review `(R, S)`. Output: a rule assigning every
demand observation to exactly one cycle. Branch: arrivals or orders? Under continuous review these
differ by the lead time; under periodic review they coincide. For continuous review, a cycle is
short if the inventory position — on hand plus on order minus backorders — went negative within it.

**Step 7. Count short cycles and divide.** Input: the cycle table with a short flag. Output:
`CSL = short-free cycles / total cycles`. Branch: what counts as short, and for how long? CSL is
binary, so a one-hour stockout affecting no order counts the same as a three-day one. That is the
measure's principal limitation.

**Step 8. Count units short and divide.** Input: demand and units unserved per cycle. Output:
`unit FR = 1 - sum(units short) / sum(units demanded)`. Branch: lost or backordered? If backordered,
an eventual-fill measure reports 100 per cent for any item that is eventually supplied, which is
nearly every item. Report first-attempt and say so.

**Step 9. Count lines and divide.** Input: order lines with a complete flag. Output:
`line FR = complete lines / total lines`. Branch: does a partial line count as a miss? It must, or
the measure degenerates into a unit fill rate with extra steps. A line ordered at 100 and shipped at
99 is a missed line.

**Step 10. Count orders and divide.** Input: orders with a complete flag, complete meaning every
line complete. Output: `order FR = complete orders / total orders`. Branch: is a partially shipped
order still an order? Some systems split a partial shipment into two orders, removing the failure
and creating a new order that succeeds. This is denominator management in its purest form and is
usually a system default.

**Step 11. Count time in stock, if ready rate is required.** Input: inventory snapshots at a stated
frequency. Output: `ready rate = periods with positive on-hand / total periods`. On-hand means
physically available and unallocated. Branch: what snapshot frequency? Daily snapshots miss intraday
stockouts; month-end snapshots measure month-end bookkeeping, not availability.

### Phase C — Convert between measures

**Step 12. Compute sigma over the protection period.** Input: `sigma_d`, review period `R`, lead
time `L`. Output: `sigma_P = sigma_d * sqrt(R + L)` under deterministic lead time, or the two-term
form with `mu_d^2 * sigma_L^2` when lead time is random. Branch: is the same `sigma_P` used on both
sides of the conversion? A conversion computed on two different sigmas is not a conversion.

**Step 13. Compute the cycle coefficient of variation.** Input: `sigma_P` and cycle demand `Q` —
`mu_d * R` under periodic review, the order quantity under continuous review. Output:
`CV_cycle = sigma_P / Q`. Branch: is `Q` the order quantity or mean cycle demand? They differ after
a promotion, during a ramp, or under a fixed lot size. Use the quantity actually ordered.

**Step 14. Apply the loss function, or invert it.** CSL to fill: `z = Phi^-1(CSL)`, look up `L(z)`,
then `beta = 1 - sigma_P * L(z) / Q`. Fill to CSL: `L_target = (1 - beta) * Q / sigma_P`, find `z`
such that `L(z) = L_target`, then `CSL = Phi(z)`. Branch: is the target near 100 per cent? The
inversion is delicate because `L(z)` approaches zero, so small errors in the target produce large
errors in `z`.

**Step 15. Check the model against the empirical distribution.** Input: the historical distribution
of demand over the protection period. Output: model-implied fill versus observed fill at the same
CSL. Branch: is the gap large enough to change the decision? Section 6 shows a three-point gap
caused by the normal distribution failing to represent deep, rare shortfalls. When the gap exceeds
two or three points, report both numbers.

### Phase D — Compute the delivery measures

**Step 16. Fix the on-time basis.** Input: the contract, the customer's compliance manual, the order
record. Output: one reference date — customer requested, supplier committed, must-arrive-by, or
current promise. Branch: who may move the date? TL 9000's on-time delivery guidance is explicit:
only a customer-initiated change resets the reference date; a supplier-initiated change does not,
even if the customer agrees. Without that rule, a supplier improves its score by rescheduling its
own promises.

**Step 17. Fix the window and the early tolerance.** Input: the contract clause or vendor guide.
Output: a window in hours or days, and whether early arrival counts. Branch: day, four-hour slot, or
appointment? A day-long window and a two-hour slot produce different rates from identical
deliveries. Early tolerance is equally consequential.

**Step 18. Fix the in-full grain.** Input: the order, the shipment, the customer's receiving rules.
Output: order, line, or case grain, and whether substitutions count. Branch: does a substitution
count as in-full? A customer who ordered blue and received green has received the quantity and not
the product. Most retailer programmes count a miss; many supplier systems count a fill.

**Step 19. Fix cause attribution.** Input: the delivery event log, appointment record, carrier
exception codes. Output: a rule assigning each late delivery to supplier, carrier, customer, or
force majeure. Branch: who hired the third party? A delay caused by a carrier the customer hired is
normally excluded; one caused by a carrier the supplier hired is normally included.

**Step 20. Fix period attribution.** Input: the reference date and the delivery date. Output: a rule
stating which reporting period a late order counts in. Branch: requested period or delivered
period? The difference is large and always flattering in the same direction.

```text
   100 orders requested in week 1. 90 arrive on time in week 1, 10 are late;
   8 of the late ones arrive in week 2, 2 arrive in week 3.

   requested-period attribution
     week 1:  90 on time / 100 requested            = 90.0%

   delivered-period attribution
     week 1:  90 on time / 92 delivered             = 97.8%
     week 2:   0 on time /  8 delivered             =  0.0%
```

The delivered-period rule reports 97.8 per cent for week 1 and a catastrophic week 2 that nobody
looks at, because week 2's denominator is small and the story has moved on. TL 9000 takes the
requested-period position explicitly, reporting an order in the month in which its requested date
falls.

**Step 21. Compute OFLT if the question is duration rather than rate.** Input: order receipt and
delivery timestamps. Output: a distribution of elapsed hours or days, with a stated percentile.
Branch: mean or percentile? A mean of 4.2 days can coexist with a 95th percentile of 19. The
customer experiences the percentile. State whether the clock starts at receipt, confirmation, or
release.

### Phase E — Composite and compare

**Step 22. Build the composite both ways.** Input: component flags per order. Output: a counted
perfect-order rate and a multiplied index from the same rows. Branch: are they close? If the counted
rate exceeds the product, failures are positively correlated; if it falls below, they are
dispersed. The direction is a diagnostic.

**Step 23. State the bounds.** Input: the four marginal rates. Output:
`1 - sum(failure rates) <= P(perfect) <= min(component rates)`. Branch: is the reported composite
inside the bounds? If not, the composite and the components are computed on different populations.
This is the fastest available audit of a perfect-order report.

**Step 24. Reconcile internal and external measures.** Input: the warehouse's dispatch measure and
the customer's receipt measure. Output: a bridge quantifying the gap and explaining it. Branch:
transport time, receiving delay, or definition? A shipment dispatched on time can arrive late; one
received on time can be posted late because the customer's system updates at end of shift.

### Phase F — Govern

**Step 25. Separate target, achieved, and bias.** Procedure: recompute the achieved rate under the
target's own definition, on the same data.

```text
   reported achieved     A   (the system's own definition)
   restated achieved     A*  (the target's definition, same data)
   target                T

   performance gap       A* - T
   measurement bias      A - A*
   target movement       T_t - T_(t-1)
```

**Step 26. Choose the design measure and the contract measure, and bridge them.** Input: the
contract, the policy engine, simulation capability. Output: two named measures plus a stated
conversion. Reasoning: safety stock formulas consume CSL or fill rate because both are properties
of a demand distribution; contracts consume OTIF because it is a promise about deliveries. There is
no closed-form map from OTIF to a buffer. Section 9.4 gives the two-ledger rule.

**Step 27. Write the specification.**

```text
   1. Measure and formula:      OTIF, on time and in full / deliveries promised
   2. Unit of account:          delivery (order-shipment), not line and not case
   3. Weighting:                order-weighted, unweighted by volume
   4. Window and attribution:   calendar week; requested-date attribution; +/- 1 day early
   5. In-full grain:            line complete, no substitution credit
   6. Source and extract:       TMS delivery events joined to ERP order lines, extract 2026-10-31
   7. Exclusions and target:    intercompany and sample excluded; target 98.0% per contract
                                schedule 4, which uses the customer's MABD, not our CRD
```

Field 7 is the one always missing and the one that makes the number comparable. It is not a
disclaimer. It is the list of conditions under which the number means what the reader thinks.

## 6. Worked example

One item, one distribution centre, twenty replenishment cycles. All figures are constructed to be
internally consistent and reproducible by hand.

### 6.1 The history

Item AX-118, a replacement seal kit at DC North. Weekly review, three-week lead time.

| Cycle | Demand | Short | Lines | Complete lines |
|---:|---:|---:|---:|---:|
| 1 | 118 | 0 | 12 | 12 |
| 2 | 126 | 0 | 12 | 12 |
| 3 | 109 | 0 | 12 | 12 |
| 4 | 131 | 0 | 12 | 12 |
| 5 | 122 | 0 | 12 | 12 |
| 6 | 114 | 0 | 12 | 12 |
| 7 | 143 | 0 | 12 | 12 |
| 8 | 97 | 0 | 12 | 12 |
| 9 | 128 | 0 | 12 | 12 |
| 10 | 148 | 38 | 12 | 3 |
| 11 | 119 | 0 | 12 | 12 |
| 12 | 134 | 0 | 12 | 12 |
| 13 | 111 | 0 | 12 | 12 |
| 14 | 124 | 0 | 12 | 12 |
| 15 | 92 | 0 | 12 | 12 |
| 16 | 137 | 41 | 12 | 2 |
| 17 | 105 | 0 | 12 | 12 |
| 18 | 121 | 0 | 12 | 12 |
| 19 | 116 | 0 | 12 | 12 |
| 20 | 138 | 41 | 12 | 2 |
| **Total** | **2,433** | **120** | **240** | **211** |

Units are units for demand and short, and count for lines. The line total reconciles as 17 clean
cycles at 12 lines each, plus 3, 2, and 2.

### 6.2 Four measures from one history

```text
   cycle service level = 17 short-free cycles / 20 cycles  = 85.00%
   unit fill rate      = (2,433 - 120) / 2,433              = 95.07%
   line fill rate      = 211 complete lines / 240 lines    = 87.92%
   order fill rate     = 70 complete orders / 80 orders    = 87.50%
```

The order count needs one more assumption: 80 orders of three lines each, four per cycle. In the
seventeen clean cycles all 68 orders are complete; of the twelve orders in the three stockout
cycles, two are complete.

Four numbers between 85.00 and 95.07 per cent, all describing the same item in the same month at the
same warehouse. The spread is 10.07 percentage points. Here CSL is the lowest measure, because it
counts every failing cycle equally regardless of depth. In another dataset — short, shallow
stockouts concentrated in multi-line orders — line fill can fall below CSL. There is no theorem
ordering the four.

### 6.3 What the model says

```text
   mean weekly demand  mu_d    = 2,433 / 20         = 121.65 units/week
   standard deviation  sigma_d = 14.70 units/week   (n - 1 divisor, n = 20)
   protection period   P       = R + L = 1 + 3      = 4.0 weeks
   sigma over P        sigma_P = 14.70 * sqrt(4)    = 29.40 units
   cycle demand        Q       = 121.65 units
   CV per cycle        sigma_P / Q                  = 0.242
```

At the observed CSL of 85 per cent, `z = 1.036`, so the implied buffer is `1.036 * 29.40 = 30.5`
units. The loss function gives `L(1.036) = 0.0779`, hence an expected shortfall of `2.29` units per
cycle and a predicted fill rate of

```text
   beta = 1 - (20 * 2.29) / 2,433 = 1 - 45.8 / 2,433 = 98.12%
```

The observed unit fill rate is 95.07 per cent. The model is 3.05 points optimistic. The conversion
is exact given its assumptions, so the discrepancy is a statement about the assumptions. Three fail
here.

1. **Normality over the protection period.** The empirical distribution has a long right tail
   (weeks of 148, 143, 138 against a mean of 122). `L(z)` treats shortfalls as frequent and shallow;
   here they are rare and deep — three cycles averaging 40 units, a third of a week's demand.
2. **Independence between demand and supply.** Two of the three stockout cycles were high-demand
   weeks, so the buffer was smallest exactly when it was needed.
3. **Sample size.** Twenty cycles with three failures gives a standard error on the CSL of roughly
   seven points, so 85 per cent is an estimate, not a measurement.

> **Note:** Do not conclude that the conversion is wrong. Conclude that it converts between two
> properties of a *distribution*, and inherits the error when the empirical distribution is not the
> assumed one. Report both numbers; a single number hides the only interesting fact.

### 6.4 The same month measured on OTIF

DC North promises 80 deliveries in the month. Seventy-six arrive within the customer's window; of
those, four are short. Two of the short deliveries also arrive outside the window.

```text
   OTD             = 76 / 80 = 95.0%
   in-full         = 78 / 80 = 97.5%
   OTIF (counted)  = 72 / 80 = 90.0%
   product of marginals = 0.950 * 0.975 = 92.6%
```

The counted rate is below the product, and the reason is mechanical. Being late and being short are
alternative failure modes here: when the last pallet is missing, the warehouse either ships short
on time or holds the delivery to ship complete. The two choices produce different failures on
different orders, so failures are negatively associated and the product overstates joint success by
2.6 points.

This is why a supplier can report 95 per cent OTD and 97.5 per cent in-full and still fail an OTIF
contract at 90 per cent. The three numbers are internally consistent. The contract tests the
intersection.

### 6.5 The restated number

The customer's scorecard reports 86.4 per cent for the same month. Nothing at the distribution
centre changed. Five definitional differences account for the gap.

| Difference | Our figure | Customer's figure | Effect |
|---|---|---|---|
| On-time basis | Our committed date | Customer's must-arrive-by date | Customer's is stricter |
| Early tolerance | None (early counts as late) | One day early allowed | Customer's is more lenient |
| In-full grain | Order complete | Case complete | Customer's is more lenient |
| Cause attribution | All delays counted | Retailer-caused excluded | Customer's is more lenient |
| Period attribution | Delivered week | Requested week | Customer's is stricter |

Four lenient changes and one strict change net to 3.6 points lower for the customer. The point is
not that one is fairer. It is that the argument in the Monday meeting is unwinnable until this
table exists.

## 7. Code, query, or calculation

The calculation below is sections 6.2 and 6.3 in a form that can be reproduced on paper or in any
language. It is deliberately not library code: a service function that returns 0.9507 without
showing which objects were counted is a function nobody can audit.

```text
Item AX-118, DC North, 20 replenishment cycles (weeks 1-20)

counted objects
  cycles            n_cyc  = 20
  short cycles      n_short =  3          CSL      = 17 / 20       = 85.00%
  units demanded    D      = 2,433
  units short       S      =   120        unit FR  = 2,313 / 2,433 = 95.07%
  order lines       L      =   240
  complete lines    L_ok   =   211        line FR  = 211 / 240     = 87.92%
  orders            O      =    80
  complete orders   O_ok   =    70        order FR = 70 / 80       = 87.50%

conversion at the observed CSL
  z         = Phi^-1(0.85)                = 1.036
  sigma_P   = 14.70 * sqrt(1 + 3)         = 29.40    (R = 1, L = 3, deterministic)
  L(z)      = phi(1.036) - 1.036 * 0.15   = 0.0779
  E[short]  = sigma_P * L(z)              = 2.29 units per cycle
  model FR  = 1 - 20 * 2.29 / 2,433       = 98.12%
  observed  = 95.07%                       gap = 3.05 points

weighting check on the same rows
  unit-weighted shortfall  = 120 / 2,433  =  4.93%  -> fill 95.07%
  line-weighted shortfall  =  29 /   240  = 12.08%  -> fill 87.92%
  order-weighted shortfall =  10 /    80  = 12.50%  -> fill 87.50%
```

Expected output: CSL 85.00, unit fill 95.07, line fill 87.92, order fill 87.50, model fill 98.12.
Four checks before you trust any implementation, including your own:

1. **Reconcile the short units.** `sum(short)` must be less than or equal to total demand and must
   match the difference between demand and units served. A shortfall larger than demand means
   double counting across overlapping cycles.
2. **Check complete lines against short units.** A cycle with a non-zero shortfall cannot have all
   lines complete. If it does, one of the two fields is wrong.
3. **Check the weighting denominators.** Lines per cycle times cycles must equal the line total,
   and orders per cycle times cycles the order total. A mismatch means the measures are computed on
   different populations.
4. **Recompute `L(z)` from `phi` and `Phi` directly.** Tables differ in the last digit, and an
   inversion near 99 per cent amplifies that into a visible buffer change.

## 8. Data, standards, and regulatory context

No binding regulation in this programme's scope defines fill rate, and none defines the conversion
between service measures. What standards constrain is the **definitional apparatus**: the reference
date, the event timestamp, the unit of account, and the semantics of the delivery commitment. Those
constraints are what make a service number auditable, and they are the part usually missing.

### 8.1 Measures frameworks

| Framework | What it defines | What it constrains here | Grade |
|---|---|---|---|
| SCOR Digital Standard (ASCM), RL.1.1 Perfect Customer Order Fulfilment | Orders perfect on in-full, commit-date delivery, documentation accuracy, condition | Counting rule: an order is perfect only if every line passes every test | B |
| APQC Open Standards Benchmarking, Perfect Order Performance (measure 101741) | Multiplicative composite of four component rates | The product form and its benchmark population | C |
| ISO 22400-2 | Key performance indicators for manufacturing operations | Naming and definition discipline; does not choose a service measure | B |
| TL 9000 (QuEST Forum), on-time delivery guidance | Reference date, early delivery, cause and period attribution | The most explicit published counting rules for OTD found in this research | B |

SCOR and APQC are both legitimate and are not the same estimator. A company that reports "perfect
order" without naming which it used has reported a number without a definition, and the two can
differ by two to four points on the same data.

### 8.2 Event and identity data

| Standard or instrument | What it governs | What it constrains here | Grade |
|---|---|---|---|
| GS1 GTIN | Product identity | Whether a substitution is the same item for in-full purposes | B |
| GS1 SSCC | Logistics unit identity | Whether a delivery matches at case or pallet grain | B |
| EPCIS 2.0 | Event data: what, where, when, why | The timestamp and location that make an on-time test possible | B |
| EDI 855 (PO acknowledgement) | Order confirmation | The committed date, one of the four on-time bases | B |
| EDI 856 (ASN) | Advance ship notice | Ship date and expected arrival used in OTD measurement | B |
| EDI 860 (PO change) | Order modification | Whether a date change is customer-initiated and therefore legitimate | B |
| Incoterms 2020 | Delivery point and risk transfer | What "delivered" means, hence which timestamp is the delivery event | B |
| IFRS 15 / ASC 606 | Revenue recognition at control transfer | Why the delivery definition also has a financial consequence | B |

The Incoterms row is easy to skip and should not be. A delivery under delivered at place (DAP) is
complete when goods arrive at the named place; under free carrier (FCA) the seller's obligation ends
when goods are handed to the carrier. An OTIF report that does not carry the Incoterm is comparing
two definitions inside one number.

### 8.3 Commercial and regulatory instruments

| Instrument | Where | Effect on service measurement | Grade |
|---|---|---|---|
| Retailer vendor compliance programmes (must-arrive-by rules) | North America, grocery and general merchandise | Sets the window, the in-full grain, and the chargeback; contractual, not a standard | C |
| UK Groceries Supply Code of Practice (GSCOP) | United Kingdom | Constrains how large retailers may vary delivery requirements and impose related charges | B |
| Good distribution practice (GDP) guidance for medicinal products | European Union | Requires delivery under controlled conditions with records; makes the delivery event evidentiary | B |
| Contractual service level agreements (SLAs) | Everywhere | The only instrument that actually binds a service number | B |

> **Note:** Regulatory instruments constrain the *evidence* of delivery, not the *rate*. A
> requirement that a temperature record accompany a shipment does not tell you whether it arrived
> on time; it tells you the timestamp and condition must survive an audit. That is a data-quality
> floor, not a metric definition.

### 8.4 What is convention rather than requirement

- A 95 per cent service target is a management choice. The `z` values that follow from it are
  arithmetic, not policy.
- The multiplicative perfect-order formula is an APQC benchmarking convention, not a law. The
  counted form is equally defensible and more auditable at order level.
- "Service level" without a qualifier is not a defined term. Every unqualified use in a document is
  a defect until the measure is named.
- Early delivery counting as a failure is a commercial choice. Some retailers enforce it because of
  dock congestion and shelf life; others do not care.

## 9. Industry practice and case evidence

### 9.1 The OTIF definition problem is documented, and unresolved

The most useful public artefact is a 2019 McKinsey and Company paper with the Trading Partner
Alliance, "Defining 'on-time, in-full' in the consumer sector". It reports a survey of major North
American retailers and consumer goods manufacturers and finds no industry consensus. As reported:

- A majority preferred a single combined metric over separate on-time and in-full measures.
- A majority preferred case-level measurement for "in full".
- A majority preferred the customer's requested delivery date as the on-time basis.
- **No consensus existed on the acceptable window.** Responses split across plus or minus four
  hours, plus or minus six hours, one day early, two days early, and other.

The paper's proposed standard — case quantity delivered by the requested date, measured to end of
working day, one-day early allowance, adjusted for retailer-caused appointment delays, counting
arrival at the facility rather than check-in — is a proposal, not an adopted standard. Grade C:
consultancy survey, self-reported, not independently verified. The value is the demonstration that
the ambiguity is real and recognised by the parties, not the percentages.

### 9.2 What a standardised definition looks like when one exists

TL 9000, the telecommunications quality management standard, publishes counting rules for on-time
delivery that are more explicit than anything in the general supply chain literature:

- The reference date is the customer's requested date, and the order is reported in the period in
  which that date falls.
- Early delivery counts as late unless the customer has authorised it, and authorisation may be
  contractual rather than per order.
- Only a customer-initiated change moves the requested date. A supplier-initiated change does not,
  even if the customer agrees.
- Delays caused by a customer-hired third party may be excluded; those caused by an
  organisation-hired third party are included.
- Delivery is counted by purchase order line item, and a line is on time only if its full quantity
  meets the date.

Grade B: a published industry standard with contractual force in its sector. The relevance is not
that every company should adopt TL 9000. It is that every rule answers a question most companies
leave unanswered, and each unanswered question is worth several points.

### 9.3 The measurement-basis effect dominates improvement programmes

| Source | Claim | Grade |
|---|---|---|
| TL 9000 guidance | Reference date and period attribution specified; supplier-initiated changes do not reset the clock | B |
| McKinsey / Trading Partner Alliance (2019) | No consensus on window; majority prefer requested date and case-level in-full | C |
| project44 (vendor) | On-time measured against carrier SLA, original estimated date, and promise date produce materially different scores | D |
| JTR Logistics (consultancy) | The same 100 shipments score 88%, 96%, or 98% depending on original versus revised appointments and exclusions | E |
| Summit Analytics (consultancy) | The date-basis switch alone can move an OTIF figure from about 95% to about 60% | D |
| GEODIS (carrier) | Retailer chargebacks reported at roughly 1.5–5% of cost of goods, varying by retailer | E |

The last four rows are vendor and consultancy claims and should be read as illustrations of
mechanism, not measurements. The mechanism is confirmed by the one Grade B source: **the definition
moves the number more than most operations move the number.** An improvement programme that gains
two points while the definition drifts five has lost ground and will report a gain.

### 9.4 The two-ledger rule

There is a structural asymmetry no measurement sophistication removes. Safety stock formulas consume
a service measure that is a property of a demand distribution — CSL or fill rate. Contracts consume
a measure that is a property of deliveries — OTIF. There is no closed-form mapping from the second to
the first, because OTIF depends on transport reliability, appointment availability, order
completeness, and cause attribution, none of which appear in the inventory model.

```text
   LEDGER 1  inventory policy          LEDGER 2  commercial promise
   --------------------------          --------------------------
   measure: CSL or fill rate           measure: OTIF, per contract
   set by:  policy, cost of capital    set by:  contract schedule
   used for: sizing the buffer         used for: chargeback, ranking
   frequency: per item, per cycle      frequency: per delivery, per week

   BRIDGE: simulate the policy over historical demand and lead times,
           apply the contract's OTIF definition to the simulated deliveries,
           report the implied OTIF and its confidence interval.
```

The bridge is a simulation, not a formula, and it carries a stated error. A company that claims a
95 per cent CSL "delivers" 98 per cent OTIF without having run the simulation is asserting a
relationship it has not measured.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| CSL reported as fill rate | Two reports differ by 5–10 points for the same week, both called "service level" | One counts cycles, the other units | Buffer sized against the wrong target | Name the measure in the metric title |
| Fill target converted with a CSL multiplier | One `z` table applied across the catalogue | Fill multiplier depends on `Q` | Over-buffer fast movers, under-buffer lumpy items | Convert through `L(z)` per item; state `Q` |
| Unit fill used where the contract means lines | Disputes about "in full" the supplier thought settled | A 99-unit line out of 100 counts as a miss | Chargebacks on orders the supplier believes it filled | Read the contract's grain; report both |
| Order-weighted average presented as demand-weighted | Rate stable while customer complaints rise | Small orders dominate the average | Systematic optimism | State weighting; compute the other as a check |
| Denominator excludes open or late orders | Service rises in a month with known delivery problems | Failures are still open, so uncounted | Rate improves as performance degrades | Freeze the population at order date |
| Delivered-period attribution | Week 1 excellent, week 2 catastrophic | Late orders roll into a later denominator | Failure escapes the period that caused it | Attribute to requested period |
| Supplier-initiated date changes reset the clock | OTIF stable while expedite spend rises | Lateness laundered by rescheduling | Score improves, experience does not | Only customer changes move the date |
| Perfect-order product presented as a count | Composite lies outside the Fréchet bounds of its components | Rates multiplied across different populations | Arithmetically impossible number | Compute both forms; check bounds |
| Ready rate used as a fill measure | 99% service on an item with chronic shortages | Time in stock is not demand served | Item looks healthy while orders fail | Measure fill; use ready rate diagnostically |
| Internal dispatch compared to external receipt target | Warehouse meets target, customer scorecard misses | Dispatch time is not arrival time | Blame allocated to the wrong party | Build the bridge in step 24 |
| Target compared to a rate from another definition | The gap is stable and unexplained for years | Definitions differ in window or grain | Perpetual underperformance no action fixes | Restate under the target's definition |
| Composite average of four components | A "service index" of 92% nobody can act on | Averaging hides the binding constraint | Investment goes to the wrong component | Decompose; report the weakest component |
| Sample orders left in the denominator | Rate moves when the sampling programme changes | Samples are not customer demand | Rate measures the sampling plan | Exclude and state the exclusion |
| Substitution counted as in-full | Supplier reports a fill the customer disputes | Quantity delivered, product not delivered | Dispute on every substitution | State the substitution rule |

Three deserve a note.

**Denominator excludes open orders** is the most common and least visible. It is rarely deliberate;
it is what happens when a query joins deliveries to orders and keeps the delivered rows. A late
order still in transit has no delivery row, so it is absent, so the failure does not exist. Freeze
the population from the order table and left-join the delivery, so a missing delivery is a failure
rather than a null.

**Delivered-period attribution** is the only failure here that improves the number as performance
worsens, reliably and in every organisation that uses it. Ask in the first week of any new role
which period a late order counts in. If the answer is "when it was delivered", the reported trend is
not a trend.

**Perfect-order product presented as a count** produces numbers that can be arithmetically
impossible. If a report shows a perfect-order rate below `1 - sum(failure rates)`, the composite and
the components are on different populations and the report should be withdrawn until resolved.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The replenishment cycle can be defined unambiguously | CSL counts cycles, so cycles must exist and be countable | CSL is not reproducible between analysts | Two analysts assign cycles independently and compare |
| Demand over the protection period is approximately normal | The loss-function conversion assumes it | Fill at a given CSL is misestimated, usually optimistically | Compare model fill with observed fill at the same CSL |
| Unmet demand is observed or estimable | Fill rate needs a shortfall numerator | Fill is computed on shipments only and shows nothing | Availability flags; backorder tables; lost-sale logs |
| The unit of account is consistent across the population | Otherwise the rate mixes denominators | Unit and case orders averaged together | Check the unit-of-measure field distribution |
| The on-time reference date is recorded and immutable | Otherwise lateness is rescheduled away | Score improves without performance change | Count orders with a revised date field |
| Delivery timestamps share one clock and time zone | An on-time test compares timestamps | A one-hour offset moves deliveries across the window boundary | Audit offsets by location |
| The contract's OTIF definition is written down | OTIF has no default meaning | Two parties compute different numbers from the same events | Obtain the clause; compare with the report method |
| Component failures are independent, if a product form is used | The product equals the joint rate only then | The composite is biased by the correlation | Compare the product with the counted rate |
| Order populations are comparable across periods | Trend comparison needs a stable denominator | Mix shift looks like performance change | Decompose change into rate and mix |
| The reported rate is first-attempt, not eventual | Eventual fill approaches 100% for any supplied item | Measure loses all discriminating power | State the rule; recompute both ways |
| Benchmark figures use the same definition | Benchmarks are comparisons | External comparison is meaningless | Ask the provider for the definition |

Two limits deserve statement.

**Every measure here is a ratio, and every ratio can be improved by shrinking its denominator.**
This is not a moral failing of the metric; it is an arithmetic property. The only defence is to fix
the population before measuring, state the exclusions, and audit the denominator as carefully as the
numerator.

**No measure here measures the customer's experience directly.** CSL, fill rate, and ready rate are
properties of an item at a location. Line and order fill are properties of a transaction. OTIF is a
property of a delivery. The customer experiences the conjunction of all of them over time, plus
price, product, and the last three orders they remember. A company can hit every target in this file
and still lose the account, and the measures will not have been wrong.

## 12. Exercises

### Exercise 1 — Four numbers, one history (calculation, 40 minutes)

A distribution centre reports 24 weekly cycles for one item: total demand 3,120 units, total
shortfall 187 units, 360 order lines of which 318 complete, 90 orders of which 74 complete. Nine
cycles contained at least one stockout.

**Deliverable:** CSL, unit fill, line fill, and order fill, each with numerator and denominator
written out.
**Acceptance criteria:** all four to two decimal places; the spread stated in percentage points; one
sentence explaining why the ordering of the four is not guaranteed in another dataset.
**Time estimate:** 40 minutes.

### Exercise 2 — Convert a fill target (calculation, 45 minutes)

Five items share a contract clause requiring 95 per cent fill rate.

| Item | `mu` per cycle | `sigma` per cycle | `Q` |
|---|---:|---:|---:|
| A | 5,000 | 650 | 5,000 |
| B | 5,000 | 2,200 | 5,000 |
| C | 40 | 6 | 40 |
| D | 5 | 12 | 5 |
| E | 1.5 | 7.5 | 1.5 |

**Deliverable:** for each, `CV_cycle`, required `L(z)`, implied `z`, implied CSL, and buffer in
standard deviations.
**Acceptance criteria:** inversion method stated; the item with the smallest buffer identified and
explained; one sentence on what a single catalogue-wide `z` would do to items D and E.
**Time estimate:** 45 minutes.

### Exercise 3 — The date-basis audit (critique, 60 minutes)

You are given 100 deliveries with five date fields: customer requested, our committed, customer
must-arrive-by, current promise, and actual arrival. Twenty have a revised promise date; 14 were
revised by us and 6 by the customer.

**Deliverable:** five OTD figures, one per basis, plus a table of how many deliveries change status
between bases.
**Acceptance criteria:** the two revision types distinguished; the current-promise figure identified
as the most flattering and the reason stated; a recommendation with a contractual justification.
**Time estimate:** 60 minutes.

### Exercise 4 — Window sensitivity (application, 40 minutes)

Using the same 100 deliveries, compute OTIF under four windows: same day; plus or minus four hours;
one day early to end of day; no early tolerance with a two-day late tolerance. Assume 12 deliveries
are short.

**Deliverable:** a four-row OTIF table with the window stated in hours, and the range.
**Acceptance criteria:** early and late tolerance stated separately for each; in-full grain held
constant and stated; the range in percentage points; one sentence on which window the contract most
likely specifies.
**Time estimate:** 40 minutes.

### Exercise 5 — Period attribution (calculation, 35 minutes)

In week 1, 250 orders are requested; 231 arrive on time in week 1; 15 of the remaining 19 arrive in
week 2 and 4 in week 3. Week 2 has 240 orders requested, of which 236 arrive on time.

**Deliverable:** week 1 and week 2 rates under requested-period and delivered-period attribution,
and the two-week aggregate under each.
**Acceptance criteria:** all four combinations shown; the direction of the bias stated; the
observation that the aggregate differs even though the events are the same.
**Time estimate:** 35 minutes.

### Exercise 6 — Perfect order bounds (calculation, 35 minutes)

A report shows on-time 94.0, in-full 96.5, damage-free 99.2, documentation 95.5, and a composite of
86.1 per cent.

**Deliverable:** the multiplicative product, the Fréchet bounds, and a verdict on whether 86.1 is
admissible.
**Acceptance criteria:** bounds computed from the marginals; comparison with the product; the
conclusion stated as one of "admissible, positive correlation", "admissible, negative correlation",
or "not admissible".
**Time estimate:** 35 minutes.

### Exercise 7 — Write the specification (design, 40 minutes)

Take one service number you have seen reported, at work or in a public filing, and write the
seven-field specification for it.

**Deliverable:** the seven fields, plus a note on which field you could not determine and what that
implies.
**Acceptance criteria:** every field filled or explicitly marked unknown; the denominator written as
a filter, not a noun; the target basis separate from the target value; source system and extract
date included.
**Time estimate:** 40 minutes.

### Exercise 8 — The perverse incentive audit (critique, 50 minutes)

For each of CSL, unit fill, line fill, order fill, ready rate, OTD, and OTIF, name the cheapest way
an operator could improve the number without improving the customer outcome, and the counter-metric
that would detect it.

**Deliverable:** a seven-row table with columns measure, gaming behaviour, detection signal, and
counter-metric.
**Acceptance criteria:** each behaviour plausible and specific, not cynical; at least two observed
personally or citable; the counter-metric computable from data the organisation already holds.
**Time estimate:** 50 minutes.

### Exercise 9 — Two-ledger design (design, 60 minutes)

Design the bridge between an inventory policy and an OTIF contract for an operation with 400
stock-keeping units (SKUs), weekly review, two-week lead time, 1,200 deliveries per month, and a
contract OTIF target of 98 per cent against must-arrive-by with a one-day early allowance.

**Deliverable:** a one-page design stating the policy measure, contract measure, simulation inputs,
output, and reported conversion error.
**Acceptance criteria:** simulation inputs named with their sources; the OTIF definition applied
inside the simulation named; conversion error in percentage points; a statement of what the design
cannot tell you.
**Time estimate:** 60 minutes.

## 13. Self-check answers

### Exercise 1

```text
CSL       = (24 - 9) / 24         = 15 / 24         = 62.50%
unit FR   = (3,120 - 187) / 3,120 = 2,933 / 3,120  = 94.01%
line FR   = 318 / 360                              = 88.33%
order FR  = 74 / 90                                = 82.22%
spread    = 94.01 - 62.50                          = 31.51 points
```

The ordering here is order fill below line fill below unit fill, with CSL far below all three
because nine stockout cycles out of twenty-four still left most demand served. In another dataset,
with deep stockouts concentrated in multi-line orders, line fill can fall below CSL. The ordering is
an empirical fact about the data, not a property of the measures.

### Exercise 2

```text
Item  CV     L(z) = 0.05/CV   z      CSL      buffer (sigma)
A     0.130  0.385           0.03   51.2%    0.03
B     0.440  0.114           0.76   77.7%    0.76
C     0.150  0.333           0.14   55.5%    0.14
D     2.400  0.0208          1.65   95.0%    1.65
E     5.000  0.0100          1.94   97.4%    1.94
```

Items A and C need almost no buffer to meet a 95 per cent fill target; their order-up-to level sits
at mean cycle demand and the expected shortfall is under one per cent of the cycle quantity. D and E
need 1.65 and 1.94 standard deviations. A catalogue-wide `z` of 1.645 would over-buffer A and C by
more than 1.5 standard deviations, roughly tripling their inventory, and under-buffer E by 0.30
standard deviations, which on a very lumpy item is the difference between 95 and about 90 per cent
fill.

### Exercise 3

A strong answer produces five figures and a transition table. The current-promise basis will be
highest because fourteen supplier-initiated revisions moved our own deadlines forward. The
recommendation should be the customer requested date or the must-arrive-by date, depending on the
contract, and should cite the TL 9000 rule that only customer-initiated changes move the reference
date. A weak answer picks the committed date because it is "the date we agreed", which is exactly
the basis the supplier controls.

### Exercise 4

A strong answer holds the in-full grain constant and reports something like: same day 74 per cent,
plus or minus four hours 79 per cent, one day early to end of day 86 per cent, no early tolerance
with two days late 81 per cent. Exact values depend on the arrival distribution, which the answer
should say. The range across windows is typically eight to fifteen points, larger than most annual
improvement programmes. The contract almost certainly specifies a narrow window, because narrow
windows transfer dock risk to the supplier, and the supplier's report almost certainly uses a wider
one.

### Exercise 5

```text
requested-period attribution
  week 1:     231 / 250        = 92.40%
  week 2:     236 / 240        = 98.33%
  two weeks:  467 / 490        = 95.31%

delivered-period attribution
  week 1:     231 / 231        = 100.00%
  week 2:     (236 + 15) / 255 =  98.43%
  two weeks:  482 / 490        =  98.37%
```

The delivered-period rule reports a perfect week 1, which is true by construction: every delivery
that landed in week 1 was on time, because the late ones landed later. The aggregate still differs,
95.31 against 98.37, because the nineteen late orders enter week 2's denominator as failures but
also enlarge it. The bias raises the early period, lowers the later one, and raises the aggregate
whenever the later period is larger.

### Exercise 6

```text
product  = 0.940 * 0.965 * 0.992 * 0.955
         = 0.90710 * 0.992 = 0.89984
         * 0.955           = 0.85935  ->  85.9%

lower bound = 1 - (0.060 + 0.035 + 0.008 + 0.045) = 1 - 0.148 = 85.2%
upper bound = min(0.940, 0.965, 0.992, 0.955)             = 94.0%
```

86.1 per cent lies inside [85.2, 94.0] and slightly above the product of 85.9, so it is admissible
and consistent with mild positive correlation. A reported 84.0 per cent would be inadmissible,
falling below the union bound, which could only arise from computing composite and components on
different populations.

### Exercise 7

A strong specification has seven filled fields and a denominator written as a filter. For example:
"OTIF, deliveries both on time and in full divided by deliveries promised in the calendar week;
delivery grain; order-weighted; plus or minus four hours around the appointment; requested-date
attribution; no substitution credit; TMS delivery events joined to ERP order lines, extract
2026-10-31; intercompany and sample excluded; target 97.0 per cent per contract schedule 4." A weak
answer writes "service level 97 per cent, monthly, from the warehouse system", which specifies one
of seven fields.

### Exercise 8

A strong table names: CSL gamed by expediting after the fact or redefining a short cycle; unit fill
by prioritising large orders and short-shipping small ones; line fill by splitting lines so fewer
fail; order fill by cancelling residual quantities and re-invoicing; ready rate by keeping one unit
permanently in stock; OTD by rescheduling promises; OTIF by declining or deferring orders that cannot
be filled completely. The counter-metrics are respectively: cycle-count audits; unit fill by
customer segment; lines per order trend; order cancellation codes; fill rate alongside ready rate;
count of revised dates; the ratio of OTIF to OTD, which rises when orders are declined. A weak
answer names only the well-known cases and supplies no computable counter-metric.

### Exercise 9

A strong design states: policy measure is fill rate at SKU-location level, because the buffer is per
item while the contract is per delivery; contract measure is the customer's OTIF against
must-arrive-by with a one-day early allowance; simulation inputs are SKU demand history, lead-time
distribution by lane, order-line composition from the last twelve months, and the transport
reliability distribution; output is an OTIF distribution with a median and an 80 per cent interval;
conversion error is the difference between simulated and observed OTIF over a backtest quarter, in
percentage points. What it cannot tell you: the effect of a disruption outside the historical range,
or the customer's willingness to accept a substitution. A weak answer computes OTIF by formula from
CSL and reports a single number with no backtest.

## 14. Interview questions

**1. "What is the difference between cycle service level and fill rate?"**

A strong answer says CSL is the probability that a cycle does not stock out, fill rate is the
expected fraction of demand served, and gives `beta = 1 - sigma_P * L(z) / Q`. It adds that the gap
depends on `sigma_P / Q`, so a 95 per cent CSL can be a 99.7 per cent fill rate or an 89.6 per cent
one. A weak answer says they are "different service metrics" without being able to say what makes
them differ.

**2. "Why does the fill-rate multiplier depend on the order quantity when the CSL multiplier does
not?"**

A strong answer explains that CSL is a quantile depending only on `z`, while fill rate is a ratio of
expected shortfall to cycle demand, and the same absolute shortfall spreads over a larger `Q`. It
notes that one `z` table cannot serve both measures. A weak answer says "because fill rate is
calculated differently" and stops.

**3. "A supplier reports 95 per cent OTD and 97.5 per cent in-full, but 90 per cent OTIF. Is that
consistent?"**

A strong answer says yes, and explains that the product of marginals is 92.6 per cent, so the
counted 90 per cent implies negative association — exactly what happens when the warehouse chooses
between shipping short on time and holding to ship complete. A weak answer treats the three numbers
as contradictory.

**4. "How would you measure service for an item that stocks out for two hours a week?"**

A strong answer distinguishes the measures and says CSL will call it a failure every week while fill
rate may be near 100 per cent, and that the right choice depends on whether any customer order was
affected. It may propose measuring affected order lines rather than time. A weak answer picks CSL
and reports 0 per cent.

**5. "Why is perfect order a product of four rates and not an average?"**

A strong answer explains that the customer experiences the conjunction, so the probability of all
four succeeding is the product under independence, and that four 95 per cent components give 81.5
per cent. It adds that the counted form is a different estimator and that the Fréchet bounds relate
them. A weak answer says "because all four have to be right", which is the intuition but not the
arithmetic.

**6. "Which period does a late order belong to?"**

A strong answer says the requested period, because otherwise a failure escapes the period that
caused it, and cites the TL 9000 rule as a standard that states this explicitly. A weak answer says
"whenever it was delivered" without recognising that this makes the metric improve as delivery
slows.

**7. "A customer's scorecard is four points below ours. How do you investigate?"**

A strong answer lists the definitional axes — date basis, window, early tolerance, in-full grain,
cause attribution, period attribution — and proposes recomputing on the customer's definition to
separate performance from definition. A weak answer assumes the customer is wrong or the data is
bad.

**8. "What is wrong with measuring the warehouse on fill rate?"**

A strong answer notes that fill rate is an outcome of forecasting, procurement, and inventory policy
as much as warehouse execution, so the warehouse is measured on variables it does not control. It
distinguishes internal dispatch measures from external delivery measures. A weak answer says fill
rate is a good measure and moves on.

**9. "How would you set a service target?"**

A strong answer separates three questions: what the contract requires, what the cost-service
frontier implies, and what the operation can achieve. It notes that safety stock formulas consume
CSL while contracts consume OTIF, and proposes the two-ledger structure with a simulation bridge. A
weak answer says "95 per cent is standard".

**10. "Can a 99 per cent fill rate be worse than a 95 per cent one?"**

A strong answer says yes, if the 99 per cent is on a fast mover with large cycle quantities and the
95 per cent is on a lumpy item where each failure loses a customer, and if the first is measured
eventually while the second is first-attempt. It notes the measures are comparable only when item,
grain, weighting, and window are held constant. A weak answer says higher is always better.

**11. "What single field would you add to a service dashboard?"**

A strong answer adds the denominator, written as a filter, because it determines whether two numbers
can be compared. It may also propose the target basis. A weak answer proposes a colour threshold or
a trend arrow.

## 15. Key terms

- **Service level** — a quantified statement about meeting a supply obligation; never meaningful
  without its measure.
- **Cycle service level (CSL)** — probability of no stockout in a replenishment cycle.
- **Fill rate** — fraction of demand met from stock; unit, line, case, or order grain.
- **Unit, line, case, order fill rate** — the same ratio at four different units of account.
- **Ready rate** — fraction of time stock on hand is positive.
- **Alpha, beta, gamma service levels** — textbook labels for cycle, fill, and time-weighted
  measures.
- **Normal loss function** — `L(z) = phi(z) - z(1 - Phi(z))`, expected exceedance per standard
  deviation.
- **Partial expectation** — `E[(D - S)+]`, the expected shortfall above a threshold.
- **Coefficient of variation per cycle** — `sigma` over a cycle divided by cycle demand.
- **Protection period** — review period plus lead time.
- **Stockout cycle** — a cycle containing at least one stockout.
- **Expected shortfall** — expected units unserved per cycle, `sigma * L(z)`.
- **On-time delivery (OTD)** — fraction of deliveries within the agreed window.
- **OTIF** — fraction of deliveries both on time and in full, per contract definition.
- **In-full grain** — order, line, or case level at which completeness is judged.
- **On-time basis** — requested, committed, must-arrive-by, or current promise date.
- **Must-arrive-by date (MABD)** — retailer-specified latest acceptable arrival date.
- **Customer requested date (CRD)** — date requested at order placement, resettable only by the
  customer under TL 9000 rules.
- **Delivery window and early tolerance** — the interval, and how early a delivery may arrive and
  still count.
- **Order fulfilment lead time (OFLT)** — elapsed time from order receipt to delivery.
- **Cause attribution** — rule assigning a late delivery to supplier, carrier, or customer.
- **Period attribution** — rule deciding which reporting period a late order belongs to.
- **Perfect order** — order meeting all of on-time, in-full, damage-free, and documentation tests.
- **Perfect order index** — composite of component rates, multiplied or counted.
- **Demand-weighted, line-weighted, order-weighted** — averaging schemes with different weights.
- **Denominator management** — improving a rate by removing hard cases from its denominator.
- **Two-ledger rule** — size the buffer with an inventory measure, judge the promise with OTIF, and
  bridge them by simulation.

## 16. Sources

**Peer-reviewed and academic (A)**

- Teunter, R. H., Syntetos, A. A., and Babai, M. Z. "Stock keeping unit fill rate specification."
  *European Journal of Operational Research*. DOI: 10.1016/j.ejor.2016.11.017. (A)
- Silver, E. A., Pyke, D. F., and Thomas, D. J. *Inventory and Production Management in Supply
  Chains*, 4th edition, CRC Press, 2016. Chapters on service measures and the loss function. (A)
- Zipkin, P. H. *Foundations of Inventory Management*, McGraw-Hill, 2000. Fill rate and the
  standard normal loss function. (A)
- Hopp, W. J., and Spearman, M. L. *Factory Physics*, 3rd edition, McGraw-Hill/Irwin, 2008. Service
  measures in production-inventory systems. (A)

**Standards and official (B)**

- ASCM, SCOR Digital Standard, reliability metric RL.1.1 Perfect Customer Order Fulfilment, with
  level-2 components RL.2.1 to RL.2.4. `scor.ascm.org`. (B)
- QuEST Forum, TL 9000 Quality Management System Requirements and Measurements Handbook; on-time
  delivery measurement Q&A at `tl9000.org`. Rules for reference date, early delivery, date changes,
  exclusions, and line-item grain. (B)
- ISO 22400-2, *Automation systems and integration — Key performance indicators for manufacturing
  operations — Part 2: Definitions and descriptions*. (B)
- GS1 General Specifications, including GTIN, SSCC, and EPCIS 2.0. (B)
- ANSI X12 EDI transaction sets 855, 856, and 860. (B)
- Incoterms 2020, International Chamber of Commerce. (B)
- IFRS 15 *Revenue from Contracts with Customers*; ASC 606. (B)
- UK Groceries Supply Code of Practice (GSCOP), regulated by the Groceries Code Adjudicator. (B)
- Good distribution practice guidance for medicinal products, European Commission. (B)

**Professional, benchmarking, and consulting (C)**

- APQC Open Standards Benchmarking, "Perfect order performance", measure 101741. Multiplicative
  formula of four component rates; benchmark median reported at 88.0 per cent. Definition as
  reported by the search result; the measure page returned an access error on direct retrieval, so
  the formula is cited as reported rather than quoted. (C)
- McKinsey and Company with the Trading Partner Alliance (2019), "Defining 'on-time, in-full' in the
  consumer sector". Survey of major North American retailers and manufacturers; no consensus on
  delivery window; proposed standard definition. (C)
- ILOS, "CSL and IFR — How to convert the service level?" Worked conversions in both directions,
  noting the loss-function inversion has no closed form. (C)
- Rossetti, M. D., *Analysis of Inventory Systems*, open textbook, chapter 8. Ready rate versus fill
  rate and the conditions under which they coincide. (C)

**Vendor and consultancy (D)**

- project44, "How to measure on-time performance, and why you are (probably) doing it wrong." (D)
- MetricGate, "Cycle Service Level vs Fill Rate Calculator". Loss-function formulation. (D)
- Kvantis Atlas, "Service level and fill rate". Conversion guidance and granularity caveat. (D)
- RightChain, "Fill Rate" knowledge base entry. Approximate relations between unit, line, and order
  fill rates. (D)
- Permute, "Fill rate: calculate units, lines, and complete orders". Grain definitions and common
  distortions. (D)

**Press and anecdote (E)**

- GEODIS, "On-Time, In-Full Delivery: Why It Matters to Retailers and Brands". Early arrivals as
  failures; retailer chargeback ranges reported. (E)
- JTR Logistics, "What Does On-Time Delivery Actually Measure?" The same shipments scoring
  differently under different date bases and exclusion rules. (E)

**Note on grading and unverified items.** Four items need flags. The APQC measure page could not be
retrieved directly (HTTP 403), so the formula and the 88.0 per cent median are reported as the
search result stated them and should be confirmed against a current APQC publication before
quotation. The McKinsey percentages are reported in aggregate because the source paper timed out on
direct retrieval; the qualitative finding — that no consensus exists on the delivery window — is
consistent across every secondary source consulted. The chargeback percentages attributed to GEODIS
are press-grade and vary by retailer, season, and contract; they are illustrative only. No source
consulted establishes an origin for the term "fill rate", and this file does not claim one.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic, the buffer these measures size: [`04_safety-stock.md`](04_safety-stock.md)
- The trade-off the service target sits on: [`01_cost-service-frontier.md`](01_cost-service-frontier.md)
- The order quantity that enters the fill conversion: [`02_eoq-limits.md`](02_eoq-limits.md)
- The single-period quantile, where service becomes a critical ratio: [`03_newsvendor.md`](03_newsvendor.md)
- Next in this module: [`06_policy-families.md`](06_policy-families.md) — how (s,Q) and (R,S)
  policies generate the cycles counted here.
- Later in this module: [`07_lead-time-demand.md`](07_lead-time-demand.md),
  [`08_multi-echelon.md`](08_multi-echelon.md),
  [`09_pooling-postponement.md`](09_pooling-postponement.md),
  [`10_segmentation-shelf-life.md`](10_segmentation-shelf-life.md).
- Quantiles and decisions: [`../08_hierarchical_intermittent_probabilistic_forecasting/10_quantiles-decisions.md`](../08_hierarchical_intermittent_probabilistic_forecasting/10_quantiles-decisions.md)
  — choosing the quantile from costs rather than from a target.
- Availability as a demand driver: [`../09_demand_sensing_promotions_npi/09_distribution-assortment.md`](../09_demand_sensing_promotions_npi/09_distribution-assortment.md)
  — why a stockout changes the demand you later measure.
- Part II map: [`../_index.md`](../_index.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file: [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
