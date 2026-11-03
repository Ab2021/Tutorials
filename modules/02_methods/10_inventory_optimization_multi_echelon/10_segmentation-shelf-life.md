# Segmentation, DDMRP, and Shelf Life — classifying items, decoupling demand, and stock that expires

> Module 10 — Inventory Optimization and Multi-Echelon Planning
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 10 topics 01, 04, and 06
> Version 2026.5

---

## 1. Why this topic exists

The first nine topics build a single policy. Topic 01 draws the cost-service frontier, topic 02
derives the economic order quantity (EOQ), topic 03 gives the single-period rule, topic 04 computes
the safety stock, topic 05 defines the service measures, topic 06 names the policy families, topic
07 supplies the lead-time demand distribution, topic 08 places the buffer across echelons, and
topic 09 pools the variance. Read in order, the chain ends with a specification that is
arithmetically correct and ready to deploy.

Deployed against a real catalogue, it fails. Not because the arithmetic is wrong, but because the
same arithmetic is applied to every item, at every level, with no reference to which item it is,
where the demand should be decoupled, or how long the stock lives. Three things stop one policy
from being a policy for everything.

- **Items differ in what they are worth and how they behave.** A high-value item with stable demand
  and a high-value item with erratic demand share a value class and need different treatment.
  Value alone does not tell you which.
- **Inventory is not the only way to decouple.** A buffer can sit at a component, a sub-assembly,
  a finished good or a regional depot. Where it sits changes what the upstream lead time means and
  therefore what the buffer has to cover. Demand driven material requirements planning (DDMRP) is
  the best-known attempt to make that placement an explicit decision rather than a by-product of
  the bill of materials.
- **Stock ages.** Every formula above assumes a unit is as good tomorrow as today. For anything
  with an expiry date, a policy that is optimal for durable stock systematically buys more than
  the item can sell before it dies.

**Segmentation is a resource-allocation device, not an optimisation.** It partitions a catalogue so
that attention, review frequency and analytical effort go where they change an outcome. It produces
no optimal class boundaries, because the objective it would optimise — management attention — is not
measurable in the same units as inventory. The honest claim is that it improves the allocation of
scarce planning effort, not the policy within a segment.

**DDMRP is a real idea wrapped in a thin evidence base.** The placement of decoupling points is
genuinely useful thinking. The buffer arithmetic is a reparameterisation of an order-up-to rule,
and the performance claims are largely produced by the organisations that sell the training and
the software. Section 9 grades those claims and does not repeat adoption or performance statistics
that cannot be traced to an independent source.

**Perishability is the one that changes the mathematics.** Ageing stock introduces a state variable
the classical policy does not have: the age distribution of what is on hand. A policy that ignores
it does not merely perform suboptimally. It can be guaranteed to produce expired stock, cycle
after cycle, while every service metric looks acceptable.

Three patterns show the shape, and each is a policy applied outside the domain in which it was
derived.

1. **The class that was never a class.** A distributor sets safety stock by ABC class — items
   grouped by annual consumption value — so A items get four weeks, B two, C one. The rule is
   defensible and wrong for the A item with a coefficient of variation of 1.1, driven by project
   orders. It is over-buffered in quiet quarters and under-buffered in loud ones, and the annual
   average looks correct. Nothing catches it, because the report groups by class.
2. **The buffer that moved the problem.** A manufacturer implements DDMRP at every level of a
   four-level bill of materials. Lead time falls, work in process rises, and total inventory is
   higher than before. The decoupling points were not chosen; they were defaulted.
3. **The order that could not be sold.** A chilled-food producer moves from weekly to four-weekly
   deliveries to cut freight cost. The order quantity is correct by every classical measure. The
   product has 28 days of shelf life and 12 days of lead time, so 16 days are saleable on receipt.
   A four-week order takes 28 days to sell. The last 12 days of every order expire.

Each is arithmetically correct inside a model whose domain was never checked. That is the subject
of this file.

> **Note:** This is the closing topic of Module 10. It assumes the buffer formula of topic 04, the
> policy families of topic 06 and the cost-service framing of topic 01, and does not re-derive
> them. Healthcare and pharmaceutical detail lives in Module 22; food and agriculture detail lives
> in Module 32.

## 2. Learning outcomes

By the end of this topic you can:

1. **Build** an ABC by value and an XYZ by variability classification, **state** the thresholds and
   why you chose them, and **explain** why the axes are independent.
2. **Assign** a policy to each of the nine cells and **collapse** them into a number of operating
   segments a planning team can actually run.
3. **Classify** demand as smooth, erratic, intermittent or lumpy using average demand interval
   (ADI) and squared coefficient of variation, and **connect** the class to the forecasting
   method-selection rule.
4. **Critique** an ABC-only segmentation, naming the item type it mishandles and the direction of
   the error.
5. **Compute** DDMRP red, yellow and green zones from average daily usage (ADU) and decoupled lead
   time (DLT).
6. **Reparameterise** a DDMRP buffer as an order-up-to policy, **compute** the implied service
   level, and **state** what the presentation adds.
7. **Grade** the evidence for DDMRP's claims about material requirements planning (MRP) nervousness
   and the bullwhip effect.
8. **Test** a policy for shelf-life feasibility — order quantity against saleable window, safety
   stock against turn, minimum order quantity (MOQ) against life — and **price** the waste.
9. **Apply** a newsvendor-with-salvage calculation to a fixed-life buy.
10. **Write** a closing policy statement naming segment, decoupling point, buffer derivation and
    expiry constraint.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Three partitions, one for each thing that stops a single policy from fitting everything.

- **Segmentation partitions the catalogue.** Which items deserve attention, and what kind.
- **Decoupling partitions the bill of materials.** Where the buffer sits, and what it covers.
- **Shelf life partitions the calendar.** How much stock can still be sold before its date.

The three are independent. An item can be high-value, decoupled at the component and perishable, or
low-value, decoupled at the finished good and durable. The policy is the intersection.

### 3.1 The two-axis matrix

ABC measures how much money flows through the item: annual consumption value, usually annual units
multiplied by unit cost. XYZ measures how predictable the flow is: the coefficient of variation
(CV) of demand over a stated bucket. Neither implies the other, which is the whole reason for
crossing them and the exact error an ABC-only segmentation makes.

```text
                        XYZ — DEMAND VARIABILITY (CV)
                  X  low CV       Y  medium CV      Z  high CV
               +---------------+---------------+---------------+
    A          | AX            | AY            | AZ            |
    high       | tight review, | short review, | daily review, |
    value      | low buffer,   | forecast-led, | lead-time     |
               | auto-replenish| seasonal build| reduction     |
               |               |               | before buffer |
               +---------------+---------------+---------------+
    B          | BX            | BY            | BZ            |
    medium     | periodic      | periodic      | min-max with  |
    value      | review,       | review,       | wider band,   |
               | standard      | standard      | monthly touch |
               | service       | service       |               |
               +---------------+---------------+---------------+
    C          | CX            | CY            | CZ            |
    low        | two-bin,      | two-bin,      | order on      |
    value      | annual count  | quarterly     | demand, or    |
               |               | count         | delist        |
               +---------------+---------------+---------------+
```

Read across a row and you see the cost of variability at a given value. AX can run a lean buffer
because demand is predictable; AZ cannot, and the cheapest response to AZ is usually to shorten the
lead time rather than enlarge the buffer. Read down a column and you see the cost of value at a
given variability: CX does not need a sophisticated policy, because the inventory saved by
optimising it is smaller than the salary of the person doing the optimising. The interesting cells
are off the diagonal: AZ justifies analytical effort, CX justifies neglect.

> **Note:** ABC and XYZ are class labels, not acronyms. There is no expansion of either in the
> inventory literature. A document that expands them is inventing an etymology.

### 3.2 The buffer as three zones

DDMRP presents a buffer as three stacked zones rather than a reorder point and an order quantity.
The zones are multiples of ADU times DLT, so the buffer scales with the item's consumption rate.

```text
  units
   ^
TOG|=============================================  top of green   6,720
   |             GREEN  — order quantity band      |
   |                                                |
TOY|---------------------------------------------  top of yellow  5,040
   |             YELLOW — lead-time demand          |
   |                                                |
TOR|---------------------------------------------  top of red     2,160
   |             RED    — safety                    |
   |                                                |
  0+--------------------------------------------->  time
        order when net flow <= TOY; raise to TOG

  The rule is an order-up-to rule with s = TOY and S = TOG,
  evaluated on a daily bucket. What DDMRP adds is where the
  buffer sits and how TOY and TOG are derived.
```

- **Yellow** is expected consumption during the decoupled lead time. It is not safety; it is the
  demand served while replenishment is in transit. Height: `ADU × DLT`.
- **Red** protects against variability in that consumption and in the lead time. It is the safety
  component, built from two judgement factors rather than a service level.
- **Green** is the order quantity band: the largest of the MOQ, demand over one order cycle, and
  the red base.

The critical observation: **the red zone is not a service level.** It is a number produced by
multiplying lead-time demand by two factors chosen from a published range. Nothing in the
arithmetic states a probability of not stocking out, and two companies using the same factors on
the same item obtain the same buffer while intending different service targets.

### 3.3 The ageing curve

For perishable stock the quantity that matters is not on-hand. It is on-hand that can still be
sold before its date, and that falls faster than a linear reading of the shelf-life clock suggests,
because the tail of a lot is the part that expires.

```text
  expected saleable
  share of a lot
   1.0 |********
       |        ****
   0.8 |            ****
       |                ***
   0.6 |                   **
       |                     *
   0.4 |                      *
       |                       *
   0.2 |                        *
       |                         *
   0.0 +-------------------------*----> days of demand cover
       0    4    8   12   16   20  24
                          ^
                  cover beyond the remaining
                  life is expected waste
```

Three consequences organise the rest of the file.

1. **Feasibility is a separate question from optimality.** A policy can be cost-optimal for durable
   stock and infeasible for perishable stock. Feasibility is checked by comparing the time a lot
   takes to clear with the saleable window at receipt, and it is a hard constraint outside the
   cost calculation.
2. **First-expired-first-out (FEFO) is necessary but not sufficient.** FEFO ensures the oldest
   stock leaves first. It does not ensure the oldest stock leaves in time. If the quantity on hand
   exceeds what can be sold within the remaining life, FEFO still produces waste, in a predictable
   quantity.
3. **The safety stock is the stock most likely to expire.** Safety stock is consumed last, so in a
   FEFO warehouse it is the residual that never turns in a normal period and ages at the full rate
   of the calendar.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Stock-keeping unit (SKU) | The lowest level at which stock is planned, ordered and counted, defined by item and location | Used to mean a product regardless of location | Item–location pair |
| ABC classification | A ranking of items by annual consumption value into three groups | Treated as an acronym with an expansion | Class label A, B or C |
| XYZ classification | A ranking of items by demand variability into three groups | Treated as a measure of forecast accuracy | Class label X, Y or Z |
| Annual consumption value | Annual units consumed multiplied by unit cost | Computed on revenue, which imports margin | Currency per year |
| Pareto principle | The observation that a minority of items often carries a majority of value | Treated as a law that fixes the 80/20 split | Proportion |
| Lorenz curve | Cumulative value plotted against cumulative item count | Confused with a Pareto chart of defect causes | Cumulative proportion |
| Coefficient of variation (CV) | Standard deviation of demand divided by mean demand over a stated bucket | Computed on a different bucket from the policy | Dimensionless ratio |
| Average demand interval (ADI) | Mean periods between non-zero demand occurrences | Computed with zero periods removed from the denominator | Periods |
| Smooth demand | Low ADI, low squared CV: frequent and regular in size | Assumed to describe all fast movers | Class label |
| Erratic demand | Low ADI, high squared CV: frequent but irregular in size | Confused with lumpy | Class label |
| Intermittent demand | High ADI, low squared CV: infrequent but regular in size | Confused with lumpy | Class label |
| Lumpy demand | High ADI, high squared CV: infrequent and irregular | Used as a synonym for any sparse demand | Class label |
| Segmentation | Partition of a catalogue into groups receiving different policies | Treated as an optimisation with an optimal group count | Group identifier |
| Policy cluster | A set of segments sharing one operating policy and one owner | Conflated with the nine matrix cells | Cluster identifier |
| Demand driven material requirements planning (DDMRP) | A buffer-zone method that places decoupling points and sizes buffers from ADU and DLT | Described as a replacement for MRP rather than a reparameterisation | Method name |
| Decoupling point | A location where a buffer breaks a dependency on upstream lead time | Confused with the push–pull boundary | Node or bill-of-material level |
| Average daily usage (ADU) | Average units consumed per day over a stated window | Computed over a window shorter than the demand cycle | Units per day |
| Decoupled lead time (DLT) | Cumulative lead time from a buffer back to the next buffered or purchased point | Confused with total cumulative lead time | Days |
| Lead time factor (LTF) | A multiplier on `ADU × DLT` that sizes the red base | Treated as a service-level parameter | Dimensionless, about 0.2 to 1.0 |
| Variability factor (VF) | A multiplier on the red base that sizes the red safety portion | Treated as a statistical estimate of demand variability | Dimensionless, about 0.2 to 1.0 |
| Red zone | The safety portion of a buffer, at the bottom | Reported as the safety stock without qualification | Units |
| Yellow zone | Expected demand over the decoupled lead time | Reported as safety stock | Units |
| Green zone | The order quantity band at the top of the buffer | Treated as a maximum inventory | Units |
| Top of yellow (TOY) | Cumulative height of red and yellow; the order trigger | Used as a target | Units |
| Top of green (TOG) | Cumulative height of all three zones; the replenishment target | Used as a capacity limit | Units |
| Net flow position (NFP) | On-hand plus on-order minus qualified demand | Computed as a classical inventory position | Units |
| Qualified demand | Past-due, due-today and order-spike demand counted against the position | Assumed to equal total open demand | Units |
| Order spike | A demand event large enough to be netted separately | Ignored, so a large order drains the buffer silently | Units and threshold |
| Inventory position | On-hand plus on-order minus all backorders | Computed on on-hand alone | Units |
| MRP nervousness | Frequent small rescheduling of open orders caused by minor plan changes | Used to mean any plan instability | Reschedules per period |
| Shelf life | The period from production during which a product remains saleable | Measured from receipt rather than production | Days |
| Remaining shelf life | Shelf life minus the age of the lot at the point of decision | Assumed constant for all stock of an item | Days |
| FEFO | Issuing stock in order of earliest expiry date | Treated as identical to FIFO | Pick rule |
| FIFO | Issuing stock in order of earliest receipt | Treated as identical to FEFO | Pick rule |
| LEFO | Last-expired-first-out; the effective rule where customers choose their own units | Confused with LIFO costing | Pick rule |
| Lot or batch | A quantity produced together and identified by one code | Treated as optional metadata | GS1 AI (10), up to 20 characters |
| Expiry date | The date after which a product must not be used or sold | Stored as free text | GS1 AI (17), six digits YYMMDD |
| Salvage value | Amount recovered from an unsold unit at the end of its window | Assumed zero, or assumed to equal cost | Currency per unit |
| Date-based allocation | Allocating stock by expiry date rather than by order priority alone | Applied only at dispatch, not at order acceptance | Allocation rule |

Four rows carry more risk than the rest.

**CV versus ADI.** CV measures dispersion in the size of demand events; ADI measures how often
they occur. An item can have low CV and high ADI — a component ordered every six weeks in a steady
quantity — or high CV and low ADI — a fast mover whose daily volume swings by a factor of three.
Different problems for the forecaster, different problems for the planner, which is why the
taxonomy is two-dimensional.

**FEFO versus FIFO.** FIFO issues by receipt date, FEFO by expiry date. They coincide for a single
supplier with a constant production-to-delivery lag, which is why the distinction is often
dismissed. They diverge when two lots of different ages arrive together, when a lot is held at a
different location, or when a supplier ships older stock against a newer order.

**Qualified demand versus backorders.** A classical position subtracts all backorders. A DDMRP
position subtracts past-due, due-today and spike demand only. The position therefore looks higher,
and the buffer is sized accordingly. Whether that is right depends on whether the qualification
threshold is calibrated, and in most implementations it is a default.

**Shelf life versus remaining shelf life.** Shelf life is a property of the product; remaining
shelf life is a property of the lot at the moment of decision. Every feasibility test uses the
second, and a data model storing only the first cannot run the test.

## 5. The mechanics, step by step

Three phases, one per partition. Each step states its input, output and branch.

### Phase A — Segmentation

**Step 1. Choose the value measure and state its basis.**

- Input: twelve to twenty-four months of issue history by SKU, and unit cost at a stated basis.
- Output: annual consumption value per SKU, in currency per year.
- Branch: **consumption or sales?** Consumption reflects the rate at which stock leaves and is the
  right measure for a replenishment policy. Sales value imports margin and ranks a high-margin slow
  mover above a low-margin fast mover that consumes more cash and space. State which you used.

**Step 2. Rank, cumulate, and read the curve before choosing thresholds.**

- Input: the value measure from step 1.
- Output: a Lorenz curve of cumulative value share against cumulative item share.
- Branch: **does the curve have a knee?** Place cut points where it bends, not at 80/15/5 by
  reflex. A catalogue with a weak knee has no meaningful A class; forcing one yields churn.

**Step 3. Assign ABC and record the decision.**

- Input: the cut points from step 2.
- Output: a class per SKU, plus threshold, date, measure and data window.
- Branch: **are the classes stable?** Re-run on the prior year. If more than a tenth of items
  change side, the boundary reflects noise rather than structure.

**Step 4. Compute variability on the bucket the policy will run at.**

- Input: demand history at the policy bucket or finer.
- Output: CV per SKU, with bucket, window and exclusions recorded.
- Branch: **which bucket?** CV falls as the bucket lengthens. Computing it monthly and applying the
  class to a weekly review understates variability for exactly the items reviewed most often.

**Step 5. Compute ADI and classify the demand pattern.**

- Input: the same history, including zero-demand periods.
- Output: ADI and squared CV per SKU, classified smooth, erratic, intermittent or lumpy.
- Branch: **are the zeros real?** A zero meaning "no demand" and a zero meaning "out of stock" are
  different observations. Counting stock-out periods as zero demand inflates ADI and deflates CV,
  making an intermittent item look smooth. This is the most consequential cleaning decision in the
  phase.

**Step 6. Cross the axes and collapse to operating clusters.**

- Input: ABC and XYZ classes per SKU.
- Output: a nine-cell matrix and a smaller set of policy clusters.
- Branch: **how many clusters can the team run?** Nine cells is the analytical answer; four to six
  is the operational answer for most teams, because each cluster needs an owner, a cadence, an
  exception queue and targets. Collapsing is a management decision and should be recorded as one.

**Step 7. Assign a policy, cadence, service target and buffer derivation per cluster.**

- Input: the clusters, and the policy families from topic 06.
- Output: one policy string per cluster in the seven-part standard.
- Branch: **is the service target uniform inside the cluster?** It should be. A cluster holding
  both a 99 per cent and a 90 per cent target is two clusters wearing one label.

**Step 8. Set the reclassification cadence and drift triggers.**

- Input: catalogue volatility and the cost of re-running the classification.
- Output: a calendar cadence plus event triggers — new product, discontinuation, promotion, supplier
  change.
- Branch: **quarterly or annual?** Quarterly is common and often too frequent, producing class
  churn that costs attention without changing decisions. Annual plus event triggers is usually
  enough. Whatever the cadence, the class must not change silently mid-cycle.

### Phase B — Decoupling and DDMRP

**Step 9. Select decoupling points deliberately.**

- Input: the bill of materials or distribution network, lead times and variability at each stage.
- Output: buffered locations, each with a stated reason.
- Branch: **why here and not one level up?** A defensible reason names one of three things: the
  upstream lead time is long relative to the promise, variability pools better here, or a batch or
  transport constraint creates a step change in cost. "The software put one here" is not a reason.

**Step 10. Compute average daily usage and state the window.**

- Input: historical consumption, a forward forecast, or a blend.
- Output: ADU in units per day, with method and window recorded.
- Branch: **past, forward or blended?** Historical ADU is stable and lags a trend. Forward ADU
  responds to a forecast and imports its error. Blended is the common compromise and its weighting
  is a judgement.

**Step 11. Compute the decoupled lead time correctly.**

- Input: lead time from the buffer back to the next buffer, purchased part or raw material.
- Output: DLT in days per item.
- Branch: **nominal or realised?** DLT is the longest unprotected path, not the sum of all stages.
  Using total cumulative lead time defeats the purpose of decoupling; using nominal rather than
  realised lead time produces a buffer short by the difference.

**Step 12. Choose the lead time and variability factors as judgements.**

- Input: the published ranges, the item's lead-time class and observed variability.
- Output: LTF and VF per item, with the basis recorded.
- Branch: **are the factors defaults?** A catalogue where every item carries 0.50 and 0.50 has been
  initialised, not parameterised. The factors are the entire safety content of the method.

**Step 13. Compute the zones.**

- Input: ADU, DLT, LTF, VF, MOQ and order cycle.
- Output: red, yellow, green and the cumulative tops, in units.
- Branch: **which green candidate binds?** Green is the largest of the MOQ, the demand over the
  order cycle and the red base. For a slow mover with a large MOQ the MOQ binds, so the green zone
  is a supplier constraint rather than a planning choice. Record which candidate bound.

**Step 14. Compute the net flow position.**

- Input: on-hand, on-order, past-due, due-today and qualified spikes.
- Output: NFP in units.
- Branch: **what counts as a spike?** Usually a multiple of ADU or a demand percentile. Set too
  low, ordinary demand is excluded and buffers look adequate; too high, the mechanism is inert.

**Step 15. Apply the rule and record the implied policy.**

- Input: the NFP and the zone tops.
- Output: an order quantity, and the equivalent classical policy.
- Branch: **is the rule order-up-to?** It is. When NFP is at or below TOY, the order raises it to
  TOG. That is `(s, S)` with `s = TOY` and `S = TOG` on a daily bucket. Writing it down lets you
  compare the buffer with the topic 04 formula, which is the only way to know what service level
  it implies.

**Step 16. Reconcile against capacity, service and shelf life.**

- Input: the average on-hand implied by the zones, storage capacity, any service target, and
  remaining shelf life.
- Output: a verdict — feasible, or a named binding constraint.
- Branch: **does the buffer survive contact with the warehouse?** A buffer whose top of green
  exceeds storage becomes a fill-the-shelf rule. One whose average on-hand exceeds what can be sold
  before expiry becomes a waste generator. Neither is visible in the zone arithmetic.

### Phase C — Shelf life

**Step 17. Classify the perishability and the clock.**

- Input: product specification, supplier date convention, transit time.
- Output: a lifetime model — fixed lifetime or continuous decay — and the clock start.
- Branch: **production or receipt?** For most regulated products the clock starts at production, so
  lead time consumes shelf life before arrival. Starting at receipt overstates the window by it.

**Step 18. Compute the saleable window at receipt.**

- Input: shelf life from production, and realised lead time plus any decoupling storage.
- Output: days of saleable life available on arrival.
- Branch: **is the lead time variable?** It is, so the window is a distribution. Use a low
  percentile, not the mean. Using the mean guarantees that half of receipts arrive with less life
  than planned.

**Step 19. Test the order quantity against the window.**

- Input: replenishment quantity and ADU.
- Output: days to clear, compared with the window.
- Branch: **does the lot clear before it expires?** The test is `Q / ADU <= window`, a hard
  constraint that overrides every economic calculation. When it fails, the options are to reduce
  the quantity, shorten the lead time, increase frequency, or accept a calculated waste — in that
  order of preference.

**Step 20. Test the safety stock against the turn.**

- Input: the safety stock or red zone, and its position in the issuance sequence.
- Output: expected age of the safety layer when consumed, and the share expected to expire.
- Branch: **does it ever turn?** A safety stock sized for 95 per cent cycle service level is
  consumed in roughly one cycle in twenty. If its shelf life is shorter than twenty cycles, it
  expires before use. The remedy is a shorter safety horizon, or accepting the expiry as the cost
  of the service level.

**Step 21. Test the MOQ and order cycle against the window.**

- Input: MOQ, order multiple, transport frequency and ADU.
- Output: days of cover implied by each constraint, against the window.
- Branch: **which binds first?** A four-weekly delivery with a large MOQ sets the effective order
  quantity, and therefore the clearing time, regardless of what the policy says. This is decided by
  a transport contract, not by a planner.

**Step 22. Choose the issuance discipline.**

- Input: the warehouse system's pick logic, layout and customer behaviour.
- Output: FEFO, FIFO, or a documented hybrid.
- Branch: **can FEFO be executed?** It requires the expiry date on the handling unit, held at lot
  level in the system, and a pick directed by it. On an open retail shelf the effective discipline
  is LEFO regardless of policy, and the rotation problem moves to the shelf.

**Step 23. Price the waste with a newsvendor-with-salvage calculation.**

- Input: selling price, unit cost, salvage, disposal cost, demand distribution over the window.
- Output: critical fractile, optimal single-period quantity, expected waste.
- Branch: **is salvage positive?** Markdown, animal feed and supplier returns produce positive
  salvage; destruction produces negative salvage because disposal adds to the loss. Omitting
  disposal cost is the most common error.

**Step 24. Set date-based allocation and record the lot.**

- Input: available stock by lot and expiry, open orders by required date, customer minimum-life
  requirements.
- Output: an allocation rule, and a lot-level transaction record supporting forward and backward
  traceability.
- Branch: **do customers specify a minimum remaining life?** Many retailers do, contractually, and
  that specification converts saleable stock into unsaleable stock at a stroke. It must be a field
  in the allocation, not an afterthought at dispatch.

## 6. Worked example

One item, through all three lenses, with arithmetic reproducible by hand.

### 6.1 The item and its catalogue position

A chilled sauce base in 10 kg bag-in-box units, supplied to food-service customers. The numbers are
constructed to be internally consistent and are not from a real company.

| Quantity | Value | Unit |
|---|---:|---|
| Average daily usage (ADU) | 240 | units per day |
| Standard deviation of daily demand | 72 | units per day |
| Coefficient of variation (daily) | 0.30 | dimensionless |
| Decoupled lead time (DLT) | 12 | days |
| Order cycle | 7 | days |
| Minimum order quantity (MOQ) | 1,200 | units |
| Unit cost | 4.20 | EUR per unit |
| Selling price | 9.50 | EUR per unit |
| Markdown recovery | 1.80 | EUR per unit |
| Disposal cost | 0.30 | EUR per unit |
| Shelf life from production | 28 | days |
| Annual usage | 87,600 | units per year |
| Annual consumption value | 367,920 | EUR per year |

The catalogue has 1,850 active SKUs and EUR 92.0m of annual consumption value. The A class is 180
SKUs carrying 80 per cent of value, which puts the A/B boundary near EUR 300,000. At EUR 367,920
this item is an A item, but only just. Its four-weekly CV is 0.22, below a 0.5 threshold, so the
segment is AX: high value, stable demand.

### 6.2 Four items with the same value and different variability

| Item | CV (4-weekly) | Segment | Policy assigned | Buffer basis |
|---|---:|---|---|---|
| 8821-K sauce base | 0.22 | AX | Weekly `(R,S)`, auto-release | Service-led, 98 per cent cycle service level (CSL) |
| 8834-T seasonal glaze | 0.55 | AY | Weekly review, pre-season build | Service-led with seasonal uplift |
| 8842-P bespoke marinade | 1.10 | AZ | Daily review, lead-time reduction | Lower target, plus dual source |
| 8851-R imported paste | 0.95 | AZ | Daily review, buffer at the port | DDMRP zones, high VF |

Items 8821-K and 8842-P are the same value class and would receive identical treatment under ABC
alone. At the same service target their buffers differ by roughly a factor of three, and the AZ
item's first recommendation is a shorter lead time rather than a larger buffer, because the buffer
required to cover a CV of 1.10 over twelve days is expensive and the lead time is negotiable.

### 6.3 The DDMRP buffer arithmetic

Item 8851-R is the one that runs on DDMRP, because its lead time is long and supply is
intermittent. Apply the same arithmetic to 8821-K so that section 6.4 compares like with like.
Factors: lead time 0.50 (medium), variability 0.50 (medium), cycle seven days, MOQ 1,200.

| Zone | Formula | Arithmetic | Result (units) |
|---|---|---|---:|
| Yellow | `ADU × DLT` | `240 × 12` | 2,880 |
| Red base | `ADU × DLT × LTF` | `2,880 × 0.50` | 1,440 |
| Red safety | `red base × VF` | `1,440 × 0.50` | 720 |
| Red | `red base + red safety` | `1,440 + 720` | 2,160 |
| Green | `max(MOQ, ADU × cycle, red base)` | `max(1,200, 1,680, 1,440)` | 1,680 |
| Top of red (TOR) | `red` | — | 2,160 |
| Top of yellow (TOY) | `TOR + yellow` | `2,160 + 2,880` | 5,040 |
| Top of green (TOG) | `TOY + green` | `5,040 + 1,680` | 6,720 |

Green is set by the order cycle, not the MOQ. If procurement later negotiates a larger MOQ, the
green zone changes and the buffer changes with it, and the change will not appear in any service
report. The average on-hand implied by the zones is approximated as the top of red plus half of the
yellow and green bands: `2,160 + (2,880 + 1,680) / 2 = 4,440` units, or 18.5 days of cover.

### 6.4 The same buffer as a classical policy

Topic 04's formula at a 95 per cent cycle service level with weekly review, protection period
`R + L = 19` days:

```text
   sigma over P   = 72 x sqrt(19)     = 313.8 units
   safety stock   = 1.645 x 313.8     = 516 units
   order-up-to S  = 240 x 19 + 516    = 5,076 units
   average on-hand = S - ADU x R / 2
                   = 5,076 - 840     = 4,236 units
```

The two averages are within five per cent: 4,440 units under DDMRP against 4,236 units under a
service-led `(R,S)` rule at 95 per cent. The interesting question is what service level the DDMRP
buffer implies. Invert it: an `(R,S)` policy with average on-hand 4,440 has `S = 4,440 + 840 =
5,280` and a buffer of `5,280 − 4,560 = 720` units. Dividing by the demand standard deviation over
the protection period gives `720 / 313.8 = 2.29`, a cycle service level of approximately 98.9 per
cent.

| Policy | Buffer derivation | Average on-hand | Implied CSL | Days of cover |
|---|---|---:|---:|---:|
| DDMRP zones, LTF 0.50, VF 0.50 | Judgement factors | 4,440 | ~98.9 per cent | 18.5 |
| `(R,S)`, weekly, 95 per cent CSL | Service target | 4,236 | 95.0 per cent | 17.7 |
| `(R,S)`, weekly, 99 per cent CSL | Service target | 4,450 | 99.0 per cent | 18.5 |

The buffer sits almost exactly on a 99 per cent service policy, which is defensible and completely
accidental. The honest assessment in one table: this is not a different method but an order-up-to
rule whose parameters come from judgement factors instead of a service level, presented as three
coloured zones. What it adds is the placement discipline and the visible split between lead-time
demand and protection, which a single target of 5,076 units hides.

### 6.5 The shelf-life constraint

Shelf life is 28 days from production and DLT is 12 days, so a lot arrives with 16 days of life.

| Quantity | Value | Unit |
|---|---:|---|
| Saleable window at receipt | 16 | days |
| Weekly order quantity | 1,680 | units |
| Days to clear a weekly order | 7.0 | days |
| Four-weekly order quantity | 6,720 | units |
| Days to clear a four-weekly order | 28.0 | days |
| Units expiring per four-weekly cycle | 2,880 | units |
| Expired share of the order | 42.9 | per cent |

At weekly ordering the policy is feasible: a lot clears in seven days against a sixteen-day window,
with nine days of margin for a late delivery or a short-dated lot. Move to four-weekly deliveries
and it becomes infeasible in a way no classical metric detects. Under FEFO with deterministic
demand the first `240 × 16 = 3,840` units sell and 2,880 expire: 37,440 units a year, or EUR
157,248 of goods plus EUR 11,232 of disposal, EUR 168,480 in total. Any freight saving has to
exceed that before the change breaks even.

### 6.6 The newsvendor check on the cycle buy

A single buy that must sell within a fixed window is a newsvendor problem. Net salvage is `1.80 −
0.30 = 1.50`; underage cost is `9.50 − 4.20 = 5.30`; overage cost is `4.20 − 1.50 = 2.70`. The
critical fractile is `5.30 / 8.00 = 0.6625`, corresponding to `z = 0.42`. Demand over the
seven-day cycle is approximately normal with mean 1,680 and standard deviation `72 × sqrt(7) =
190.5`, so `Q* = 1,680 + 0.42 × 190.5 = 1,760` units.

The three numbers are 1,200 (MOQ), 1,680 (green zone) and 1,760 (optimal cycle buy). The green zone
is the mean cycle demand and therefore orders to the fifty-percent point of the distribution, not
the critical fractile. This does not make the green zone wrong — it is an order quantity band in a
continuous system, not a single-period buy — but a planner should know which of the two
calculations produced the number in front of them.

Change the salvage assumption and the fractile moves sharply. With no salvage and disposal of 0.30,
net salvage is `−0.30`, overage cost is 4.50, the fractile falls to `5.30 / 9.80 = 0.54`, and the
optimal buy falls to about 1,700 units. A product that cannot be marked down, given to staff or
returned has a materially lower economic order quantity than one that can.

### 6.7 What the three lenses say together

Item 8821-K passes all four tests at its current configuration: AX segment, decoupled at the
finished good because the lead time is the constraint and the product is not postponable, a 4,440
unit average buffer equivalent to about 99 per cent CSL, and a weekly order that clears in seven
days against a sixteen-day window. Move one parameter — the delivery frequency — and it fails the
fourth test while still passing the first three. That is the shape of every shelf-life failure: the
policy remains optimal in the model and becomes infeasible in the warehouse.

## 7. Code, query, or calculation

One hand calculation covering the zones, the topic 04 comparison and the shelf-life test for item
8821-K. Every figure in sections 6.3 to 6.5 can be reproduced from it.

```text
ADU = 240 u/d;  DLT = 12 d;  LTF = 0.50;  VF = 0.50;  cycle = 7 d;  MOQ = 1,200 u
YELLOW = ADU x DLT                        = 240 x 12        = 2,880 u
RED    = ADU x DLT x LTF x (1 + VF)       = 1,440 x 1.50    = 2,160 u
GREEN  = max(MOQ, ADU x cycle, ADU x DLT x LTF)             = 1,680 u
TOR = 2,160;  TOY = 5,040;  TOG = 6,720 u
avg on-hand (zones) = TOR + (YELLOW + GREEN)/2 = 4,440 u

Topic 04 at 95% CSL:  P = R + L = 19 d;  sigma_P = 72 x sqrt(19) = 313.8 u
SS = 1.645 x 313.8 = 516 u;  S = 240 x 19 + 516 = 5,076 u
avg on-hand (R,S) = S - ADU x R/2 = 5,076 - 840 = 4,236 u
ratio = 4,440 / 4,236 = 1.048   (+4.8%)

Implied CSL of the zone buffer:  (4,440 + 840 - 4,560) / 313.8 = 2.29 -> ~98.9%

Shelf life 28 d - DLT 12 d = 16 d saleable at receipt
weekly Q 1,680 u -> 7.0 d to clear  -> feasible
4-weekly Q 6,720 u -> 28.0 d to clear -> 2,880 u expire per cycle (42.9%)
```

Expected output in order: yellow 2,880 units; red 2,160 units; green 1,680 units; tops 2,160, 5,040
and 6,720 units; average on-hand 4,440 units; safety stock 516 units; `(R,S)` average 4,236 units;
ratio 1.048; implied service approximately 98.9 per cent; window 16 days; four-weekly waste 2,880
units, or 42.9 per cent of the order.

Two sanity checks before trusting any such calculation. Confirm that green is at least the MOQ; if
not, the MOQ binds and the buffer is replenished in quantities the zones did not intend. Confirm
that the window is computed from production and not from receipt; if the source is a warehouse
receipt date, the window is overstated by the entire lead time and the feasibility test will pass
when it should fail.

## 8. Data, standards, and regulatory context

Nothing here is legal advice. The purpose is to state what each instrument constrains
operationally, so a data model can be built that satisfies it, and to separate requirement from
convention.

### 8.1 Identification and date coding

| Standard | What it governs | What it constrains here | Grade |
|---|---|---|---|
| GS1 GTIN, AI (01) | Trade item identity | The item dimension of the segment and buffer | B |
| GS1 AI (10) | Batch or lot number, up to 20 alphanumeric characters | The lot dimension every expiry and recall record keys on | B |
| GS1 AI (17) | Expiration date, six digits in YYMMDD form | The date FEFO sorts on and the feasibility test compares against | B |
| GS1 AI (15) | Best-before date, quality rather than safety | A separate field; conflating it with expiry over-constrains allocation | B |
| GS1 AI (16) | Sell-by date | Retail shelf-life rules that differ from the consumer date | B |
| GS1 AI (21) | Serial number | Instance-level traceability where lot level is insufficient | B |
| GS1 AI (7003) | Expiration date and time | Hour-level precision for short-life products | B |
| GS1 Global Traceability Standard 2.0 | Critical tracking events and key data elements | The event record that makes a recall possible | B |
| EPCIS 2.0 | Event data exchange | Sharing lot and date events between trading parties | B |

Two properties matter more than the codes.

**The date is a property of the lot, not the item.** A data model with an expiry column on the item
master stores one date and will be wrong for every lot except one. The schema requirement is a lot
table keyed by item, lot code, production date, expiry date and location. Almost every failure in
section 10 traces to a model that skipped this table.

**A lot number is not a unique identifier on its own.** It must be processed with the GTIN. Two
suppliers can use the same string for different products, and a manufacturer can reuse a string
across plants. The pair `GTIN + lot` is the batch-level unit of traceability.

### 8.2 Regulatory regimes that constrain operations

These regimes determine what data must exist and be retained. They do not prescribe a buffer size,
a review period or a service level, and no inventory policy should be called "compliant" on their
basis.

| Regime | Jurisdiction and scope | What it constrains operationally | Grade |
|---|---|---|---|
| EU Falsified Medicines Directive, Delegated Regulation 2016/161 | EU, prescription medicines | Unique identifier and tamper evidence; verification and decommissioning at dispensing, forcing lot and serial records | B |
| EU Good Distribution Practice, 2013/C 343/01 | EU, medicinal product distribution | Temperature control, stock rotation, segregation of expired stock, recall capability | B |
| U.S. Drug Supply Chain Security Act (DSCSA) | U.S., prescription drugs | Transaction information, lot-level and serialised tracing, verification of suspect product; enforcement phased | B |
| Regulation (EU) No 1169/2011 | EU, food information to consumers | "Use by" and "best before" marking, which fixes what the date field means | B |
| Regulation (EC) No 852/2004 and HACCP | EU, food hygiene | Hazard analysis and critical control points; batch identification and withdrawal capability | B |
| U.S. Food Traceability Rule, FSMA section 204 | U.S., foods on the traceability list | Key data elements at critical tracking events; lot-level records; compliance dates extended | B |
| EU Medical Device Regulation 2017/745, UDI | EU, medical devices | Unique device identification on label and record; lot or serial traceability by risk class | B |
| U.S. 21 CFR Part 830, UDI | U.S., medical devices | Device identifier and production identifier, including lot and expiry where applicable | B |

Read the table for what it implies about the data model rather than for compliance. Every row
requires that the organisation can answer, for a given lot, where it is, where it has been and when
it expires. That is the same record set the FEFO and feasibility tests need. A company that can
pass a recall test can run a shelf-life policy; one that cannot, cannot.

### 8.3 What is convention rather than requirement

- **80/15/5 is a convention.** It comes from a specific historical application and is not optimal
  for any particular business. Place the cut where the value curve bends.
- **The 0.5 and 1.0 CV thresholds are conventions.** Different sources use different bands, some a
  percentage rather than a ratio. What matters is that the threshold is stated and applied
  consistently.
- **The ADI 1.32 and squared-CV 0.49 cut-offs are derived, not arbitrary.** They come from a
  specific comparison of two forecasting methods and carry that scope. They are not a universal
  law of demand classification.
- **The DDMRP factor ranges are professional-body guidance.** Roughly 0.20 to 0.40 for long lead
  times, 0.41 to 0.60 for medium, 0.61 to 1.00 for short, with the same structure for variability.
  They are guidance from the method's custodians, not a standard and not a statistical result.
- **"FEFO is required" is usually a contractual or good-practice expectation,** not a statutory
  algorithm. The regulations require traceability and withdrawal capability; rotation is the
  operator's choice.

> **Note:** A regulator can ask you to demonstrate that you can withdraw a lot. A regulator cannot
> ask you to prove FEFO is optimal, because no regulation defines it. Confusing the two leads teams
> to over-invest in documentation and under-invest in the rotation logic that prevents waste.

### 8.4 Why date coding changed the problem

Before lot and date coding were routine, spoilage was an aggregate cost observed after the fact: a
quantity of value disappeared and was written off. There was no unit-level deadline, so there was
nothing to allocate against and nothing to trace. Date coding turned perishability from a cost
problem into a control problem. Once each unit carries a deadline, three things become possible:
allocation by remaining life rather than arrival order; feasibility testing at the point of order,
because the window is a number on the lot record; and attribution, so waste can be traced to the
order cycle, the MOQ, the transport mode or the forecast rather than to a general shrinkage line.

## 9. Industry practice and case evidence

### 9.1 Where ABC comes from

ABC classification was described by H. Ford Dickie, a manager at General Electric, in a 1951
article in *Factory Management and Maintenance* titled "ABC Inventory Analysis Shoots for Dollars,
Not Pennies." Dickie applied the Pareto principle and the Lorenz curve to inventory, and the method
spread through the quality movement into general management practice. Grade C for the citation: the
attribution is consistent across several secondary sources, including university teaching material
and reference works, but the 1951 article itself was not consulted for this file.

Two consequences matter. The method was designed as a **prioritisation device for a manual filing
system** — Dickie's stated benefits were better planning, lower administrative cost and motivated
control staff. It was never an optimisation, and the 80/20 split is an artefact of the cases he
observed rather than a derived result. The literature later extended ABC to multiple criteria —
Flores and Whybark (1986) as reported in the review literature — an admission that one value axis
is insufficient. The ABC-by-XYZ matrix is a simpler version of the same admission.

### 9.2 The demand-pattern taxonomy

The four-class scheme comes from Syntetos, Boylan and Croston (2005), "On the categorization of
demand patterns," *Journal of the Operational Research Society*, 56(5), 495–503. It crosses two
binary indicators — average demand interval and squared coefficient of variation of demand sizes —
at cut-offs of 1.32 and 0.49. The cut-offs were derived by comparing the mean squared error of
Croston's method against the Syntetos–Boylan approximation (SBA) on a large set of real
intermittent demand series from the automotive industry.

Why exactly four classes: because two binary indicators produce four combinations, and because the
comparison that generated the cut-offs produced two regions in each dimension. The scheme is not
four because demand has four kinds. It is four because the method-selection question has a
two-by-two answer, which is a different and more honest statement than the textbook one.

Kostenko and Hyndman (2006), "A note on the categorization of demand patterns," *JORS* 57(10),
1256–1257, showed the true boundary between the methods is non-linear and that the correct limiting
values are 4/3 and 0.5. They concluded the scheme effectively reduces to two categories. The
original authors replied accepting the correction and defending its practical simplicity. Grade A
for both.

The practical reading: the taxonomy is a **method-selection device for intermittent demand**, not a
universal segmentation. Using it to choose between Croston and SBA is what it was built for. Using
it to assign service levels, buffers or review frequencies is an extension the original work does
not support, and it is extremely common.

### 9.3 DDMRP: the claims and the evidence

The vendor claims are consistent: DDMRP reduces MRP nervousness by decoupling planning from
upstream variability, dampens the bullwhip by buffering at strategic points, and reduces inventory
while improving service. The first two are mechanisms that can be reasoned about. The third is an
empirical claim, and the evidence is thin.

| Source | Design | What it found | Grade |
|---|---|---|---|
| Miclo, Lauras, Fontanili, Lamothe and Milian (2015), "MRP vs. demand-driven MRP: towards an objective comparison," IEEE IESM | Discrete-event simulation | Notes that DDMRP's claim of managing uncertainty better than MRP "has never been scientifically demonstrated," and builds a simulation to compare them | A |
| Miclo, Lauras, Fontanili and Lamothe (2019), "Demand Driven MRP: assessment of a new approach to materials management," *IJPR* | Simulation against MRP II and Kanban | Concludes DDMRP represents a superior approach and warrants further study | A |
| Ihme and Stratton, "Evaluating Demand Driven MRP: a case based simulated study," Nottingham Trent University | Single-company case simulation, printing inks | Roughly 45 per cent fewer inventory alerts and 95 per cent fewer stock-outs, with no significant stock reduction; authors state the improvement was not fully attributable to DDMRP because of poor data and procedures, and the SKU sample was small | C |
| Butturi, De Rosa, Balugani and Gamberini (2021), "Understanding the DDMRP scope of application: a critical literature review," DAAAM | Critical review of 39 publications | Buffer positioning highly subjective; parameters empirically based and poorly documented; implementation costs material; most "case studies" are simulations, with only two real deployments identified | A |
| Thürer, Fernandes and Stevenson (2020), "Production planning and control in multi-stage assembly systems," *IJPR* | Simulation across assembly configurations | MRP performs worst; DDMRP strong under bottlenecks and converges with OPT under severe bottlenecks | A |

The pattern is the finding. The peer-reviewed work is almost entirely simulation. The one
case-based study reports large operational improvements with no inventory reduction and explicitly
disclaims attribution. The critical review finds two real deployments in 39 publications and
identifies buffer positioning as the method's weakest link. None of this says DDMRP does not work.
It says the evidence base is dominated by simulation and that the strongest claims in circulation
are not supported by independent field data.

> **Note:** Performance and adoption statistics in DDMRP vendor material are graded D and are not
> reproduced here. The figures above are reported only because they come from a named study with a
> stated design, and they carry that study's own caveats.

### 9.4 Why the arithmetic is conventional, and what survives

The rule is: when the net flow position is at or below the top of yellow, order up to the top of
green. Write `s = TOY` and `S = TOG`. The rule is exactly "order up to `S` when the position is at
or below `s`", which is the `(s, S)` policy of topic 06 on a daily review bucket.

The parameters are derived differently. A classical `(s, S)` derives `S` from lead-time demand plus
a buffer computed from a demand distribution and a service target. DDMRP derives the same structure
from `ADU × DLT` multiplied by two factors. Both produce two numbers in units. Only one states a
probability.

What DDMRP genuinely adds: an explicit placement decision with a stated rationale, rather than an
emergent property of the bill of materials; a visible buffer composition, so you can see how much
stock is lead-time demand and how much is protection, where a single `S` of 5,076 hides the split;
and a demand-qualification rule that is a deliberate statement about which demand the buffer
protects against. What it does not add is a service-level guarantee, an optimality result, or a
different mathematical structure. Stated that way it is defensible and useful. Stated as a
replacement for inventory theory, it is marketing.

### 9.5 Perishable inventory: what the literature establishes

| Source | Design | What it establishes | Grade |
|---|---|---|---|
| Nahmias (1982), "Perishable inventory theory: a review," *Operations Research* | Review of fixed-life and decaying inventory models | Age-dependent inventory has no simple optimal policy in general; oldest-first issuing is optimal under stated conditions | A |
| Chaudhary, Kulshrestha and Routroy (2018), "State-of-the-art literature review on inventory models for perishable products," *JAMR* | Review of 418 articles, 1990–2016 | Calls for more stochastic modelling; notes limited work on multi-item and multi-echelon perishable systems | A |
| Staff and Mustafee (2025), "Discrete-event simulation for effective perishable inventory management: a review," *SIMULATION* | Systematic review of 25 papers | Most models assume fixed lead times and fixed shelf lives; supply uncertainty and stochastic lifetimes are rarely modelled; only one study reported an actual implementation | A |
| Kandasamy and others (2025), "Analysis of key challenges to implementation of FEFO in perishable food supply chain," *Journal of Agriculture and Food Research* | PRISMA review plus DEMATEL and TISM | Thirteen barriers; storage management and expiry tracking are the foundational causal barriers, with cost and technology dependence significant | A |
| Solari, Lysova and Montanari (2025), "Reordering perishables: identifying the shelf life threshold," *Procedia Computer Science* | Simulation, periodic review, FEFO, Weibull demand | A critical shelf life of roughly 14 days beyond which perishability no longer significantly affects the optimal policy | A |

The Staff and Mustafee finding is the most useful for practice: the models assume fixed lead times
and fixed shelf lives, and the operations they describe do not. If the lead time is variable, the
saleable window is variable, and the feasibility test in step 19 is a distribution rather than a
comparison. The Solari finding is the most useful for scoping: below roughly two weeks of shelf
life, perishability dominates and must be modelled explicitly; above it, a classical policy is
approximately adequate.

The retail rotation finding is counter-intuitive and worth stating separately. Where customers
select their own units, the effective discipline is last-expired-first-out — customers take the
freshest stock — regardless of policy. FEFO minimises waste under top-down control, which is why it
works in food service, healthcare and distribution but less well on an open shelf.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| ABC applied without a variability axis | Every A item carries the same weeks of cover | Value and variability treated as correlated | High-value erratic items under-buffered, stable ones over-buffered | Add XYZ; price the difference for the top twenty items |
| Nine cells, nine policies, no owner | A policy document with nine rows; nobody can name their item's cell | Segmentation mistaken for an operating model | Policies exist on paper and default in the system | Collapse to four to six clusters with named owners |
| Stock-out periods counted as zero demand | ADI inflated; intermittent items classified smooth | Zero conflated with no demand | Wrong method selected; buffer sized for a demand pattern that does not exist | Flag stock-out days and exclude them from the denominator |
| Buffers at every bill-of-materials level | Lead time falls, work in process rises, total inventory rises | Decoupling points defaulted rather than chosen | The claimed inventory reduction is false in aggregate | Select decoupling points against the three criteria in step 9 |
| Shelf life measured from receipt | Warehouse expiry dates later than the supplier's | Clock start mis-specified | Window overstated by the whole lead time | Store production and expiry dates separately at lot level |
| Order quantity exceeds what can sell before expiry | `Q / ADU` greater than days remaining at receipt | Feasibility never tested | Systematic waste every cycle, invisible in service metrics | Run the step 19 test on every perishable item |
| Safety stock never turns | The oldest lot in the warehouse is always the safety layer | Safety stock designed without reference to age | Safety stock expires before it is used | Size the safety horizon below the shelf life, or price the waste |
| MOQ or delivery frequency sets the cycle | Order quantity constant at MOQ or truck capacity; cover exceeds the window | Transport constraint overrides the policy | Waste determined by a procurement contract | Make shelf life a constraint in transport negotiations |
| FEFO configured but not executable | Pickers override the suggested lot; short-dated stock sits in a far bay | Expiry not on the handling unit or not held at lot level | Rotation policy exists and waste continues | Put the date on the label, hold it at lot level, direct the pick |
| Waste measured in units rather than cost | Shrinkage reported as a percentage of units | High-value and low-value waste blended | The expensive waste hides inside an average | Report waste in currency by item and cause |

The segmentation rows share a cause: the classification was treated as a one-off analytical exercise
rather than an operating parameter with an owner and a cadence. The shelf-life rows share a
different one: the data model has no lot dimension, so the feasibility test cannot be run even where
somebody wants to run it.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| Annual consumption value is a good proxy for attention | The ABC axis rests on it | A high-margin, low-volume item is under-managed; a low-margin bulk item absorbs attention | Rank on consumption value and on gross margin; check the top twenty for disagreement |
| The bucket used for CV matches the policy bucket | CV is bucket-dependent | Variability understated, buffer too small, service below target | Recompute at two buckets and compare class assignments |
| The history is long enough to estimate CV and ADI | Both are sample statistics | A new item is classified on noise; a seasonal item on one season | Check observation counts, not calendar span |
| Classes are stable over the planning horizon | Policies are set per class | An item migrates and keeps its old policy | Re-run on the prior year and measure churn |
| ADU is stationary over the decoupled lead time | Zones scale linearly with ADU | Buffer sized for a demand rate that has moved | Plot ADU over twelve months by item |
| DLT is the genuine decoupled lead time | The buffer covers what the number says | Buffer sized for a chain that no longer exists, or short by the difference | Compare DLT with realised receipt intervals at the buffer |
| DDMRP factors encode the intended service level | They are used as if they did | Target and buffer are unrelated; the target is aspirational | Invert the buffer to an implied CSL |
| Shelf life and production date are recorded at lot level | Every feasibility and FEFO calculation depends on it | The window is unknown or overstated; recall impossible | Query the lot table for nulls and for expiry before production |
| FEFO is executable in the physical warehouse | The rotation policy assumes it | Waste continues while the policy claims otherwise | Audit pick confirmations against the suggested lot |

Three limits deserve statement. **Segmentation does not improve a policy; it allocates attention.**
No arrangement of items into classes makes any individual buffer better; the improvement comes from
spending effort where it changes an outcome, and that improvement is bounded by the quality of the
policy applied afterwards. **DDMRP's evidence base is simulation-heavy and partly vendor-produced,**
so this file treats its performance claims as unproven rather than established. **The perishable
models in the literature assume fixed lead times and fixed shelf lives,** and real operations have
neither — so a feasibility test identifies items that will certainly fail, not the precise waste of
items that pass.

## 12. Exercises

### Exercise 1 — Build the matrix (application, 45 minutes)

Take a transaction dataset with at least 200 SKUs and twelve months of history.

**Deliverable:** the nine-cell matrix with item counts and value share per cell, and a Lorenz curve
with cut points marked.
**Acceptance criteria:** value measure and basis stated; cut points justified from the curve; CV
bucket stated and matching a plausible policy cadence; one item identified that ABC alone would
treat wrongly.
**Time estimate:** 45 minutes.

### Exercise 2 — Two items, one class (design, 40 minutes)

Choose two A-class items from Exercise 1 with CVs at least a factor of two apart.

**Deliverable:** a two-column comparison of policy family, review cadence, service target, buffer
basis and first operational change.
**Acceptance criteria:** policies differ in at least three dimensions; buffer difference quantified
at the same service target; the more erratic item gets a lead-time intervention, not only a buffer.
**Time estimate:** 40 minutes.

### Exercise 3 — DDMRP zones by hand (calculation, 40 minutes)

ADU 150 units per day, DLT 9 days, lead time factor 0.40, variability factor 0.60, order cycle 5
days, MOQ 800 units.

**Deliverable:** red, yellow, green and the three cumulative tops, each with formula and units.
**Acceptance criteria:** the binding green candidate is named; arithmetic reproducible by hand; one
sentence states what the yellow zone represents physically.
**Time estimate:** 40 minutes.

### Exercise 4 — The equivalence test (critique, 50 minutes)

For Exercise 3, assume daily demand standard deviation of 45 units and weekly review.

**Deliverable:** the 95 per cent `(R,S)` policy, the average on-hand under each approach, the
implied service level of the DDMRP buffer, and a one-paragraph assessment.
**Acceptance criteria:** protection period stated; comparison on average on-hand, not buffer
height; implied CSL computed by inversion; assessment separates what DDMRP adds from what it
renames.
**Time estimate:** 50 minutes.

### Exercise 5 — Evidence audit (critique, 45 minutes)

Obtain a DDMRP vendor presentation, white paper or product page.

**Deliverable:** a table of claims with the evidence offered, a grade A to E, and a verdict.
**Acceptance criteria:** at least five claims; each grade justified in one sentence; mechanism
statements distinguished from empirical claims; the verdict separates "plausible" from
"demonstrated."
**Time estimate:** 45 minutes.

### Exercise 6 — Shelf-life feasibility (calculation, 35 minutes)

Shelf life 45 days from production, lead time 21 days, demand 80 units per day. The commercial team
proposes moving from weekly to fortnightly ordering. Unit cost EUR 3.10, disposal EUR 0.25.

**Deliverable:** saleable window, days to clear under each frequency, units expiring per cycle under
the fortnightly option, annual waste cost, and a recommendation.
**Acceptance criteria:** window computed from production; clearing time uses quantity and daily
rate; annual figure uses the correct cycle count; recommendation states the condition under which
the margin is exhausted.
**Time estimate:** 35 minutes.

### Exercise 7 — Newsvendor with salvage (calculation, 40 minutes)

Using Exercise 6: selling price EUR 7.40, unit cost EUR 3.10, markdown recovery EUR 1.20, disposal
EUR 0.25. Demand over the window is normal, mean 1,120, standard deviation 180.

**Deliverable:** critical fractile, optimal quantity, and the effect of setting salvage to zero.
**Acceptance criteria:** net salvage includes disposal cost; underage and overage stated separately;
the zero-salvage case computed and compared.
**Time estimate:** 40 minutes.

### Exercise 8 — Rotation diagnosis (critique, 40 minutes)

A warehouse reports FEFO compliance of 98 per cent by pick confirmation and a spoilage rate that has
not fallen in two years. Short-dated stock is stored in the far bay because it arrived last.

**Deliverable:** at least three mechanisms by which both facts can be true, and the measurement that
would separate them.
**Acceptance criteria:** each mechanism specific to a physical or system design; at least one is a
measurement artefact; the next measurement is named.
**Time estimate:** 40 minutes.

### Exercise 9 — The integrated policy statement (design, 60 minutes)

Take one real item.

**Deliverable:** a one-page statement of segment, decoupling point, buffer derivation and expiry
constraint, with every parameter, unit, derivation and owner.
**Acceptance criteria:** all four parts present; buffer derivation states whether it is service-led
or factor-led and gives the implied service level; expiry constraint is an inequality with numbers;
each part names what would force a review.
**Time estimate:** 60 minutes.

## 13. Self-check answers

**Exercise 1.** A correct submission states the value measure (consumption value, not sales value),
places cut points from the curve, and reports value share per cell. The most common error is a CV
computed on monthly data applied to a weekly policy, which compresses AY and AZ into AX. Nine cells
with roughly equal item counts means percentile cut points were used, producing a segmentation with
no economic content.

**Exercise 2.** A strong answer gives the stable item a service-led buffer at a high target with
weekly review, and the erratic item a shorter review, a lower per-cycle target backed by a
lead-time reduction, and possibly a second source. At the same 95 per cent CSL the erratic item's
buffer is larger by roughly the ratio of the CVs. A weak answer gives both a larger buffer.

**Exercise 3.**

```text
yellow   = 150 x 9                       = 1,350 units
red base = 1,350 x 0.40                  =   540 units
red safe = 540 x 0.60                    =   324 units
red      = 540 + 324                     =   864 units
green    = max(800, 150 x 5 = 750, 540)  =   800 units  (MOQ binds)
TOR = 864;  TOY = 2,214;  TOG = 3,014 units
```

Yellow is the expected consumption during the nine days it takes to replenish — the demand the
buffer must serve while an order is in transit. Green is set by the MOQ, so a supplier constraint,
not the order cycle, determines the replenishment quantity, and that should be visible in the
policy statement.

**Exercise 4.** With `sigma_d = 45`, `R = 7`, `L = 9`, `P = 16`: `sigma_P = 45 × sqrt(16) = 180`;
`SS = 1.645 × 180 = 296`; `S = 150 × 16 + 296 = 2,696`; average `= 2,696 − 525 = 2,171` units. The
DDMRP average is `864 + (1,350 + 800) / 2 = 1,939` units. Inverting: `S_equiv = 1,939 + 525 =
2,464`; expected demand over `P` is 2,400; buffer is 64; `z = 64 / 180 = 0.36`, giving a CSL of
about 64 per cent.

The DDMRP buffer is smaller than a 95 per cent policy here, because the MOQ binds green and the
factors are low. The implied service level is far below what most businesses would assume from a
buffer described as protecting the item. The assessment should say the zones are internally
consistent and that the service level they produce is an output nobody chose — the strongest form
of the critique.

**Exercise 5.** A strong audit finds most claims are mechanism statements — "buffers absorb
variability", "decoupling reduces nervousness" — which are plausible and untested, and that the
empirical claims cite the method's own case material. It grades those D and says none would support
a capital decision without a pilot. A weak audit grades everything D without distinguishing
mechanism from evidence, which is unhelpful because the mechanism is the part that is sound.

**Exercise 6.** Window is `45 − 21 = 24` days. Weekly `Q = 560`, clearing in 7.0 days; fortnightly
`Q = 1,120`, clearing in 14.0 days. Both are feasible, which is the point: the proposal is not
obviously wrong and the analysis has to be done rather than assumed. The fortnightly margin is ten
days, which is the tolerance for one late delivery or a demand dip. A strong answer states that
margin, states the condition under which it is exhausted — a single late delivery consumes most of
it — and recommends a remaining-life specification on receipt rather than rejecting the change.

**Exercise 7.** Net salvage `1.20 − 0.25 = 0.95`. `Cu = 4.30`, `Co = 2.15`, `CF = 4.30 / 6.45 =
0.667`, `z = 0.43`, `Q* = 1,120 + 0.43 × 180 = 1,197` units. With salvage zero and disposal still
0.25, net salvage is `−0.25`, `Co = 3.35`, `CF = 4.30 / 7.65 = 0.562`, `z = 0.156`, `Q* = 1,148`
units. The 49-unit difference is about 4 per cent of the order and is entirely attributable to
whether the product can be marked down.

**Exercise 8.** Three mechanisms. First, pick confirmation measures adherence to the system's
suggested lot, and the system may be suggesting by arrival date, so compliance with a FIFO
suggestion is reported as FEFO. Second, the far bay is distant, so pickers override the suggestion
and the override may not be recorded as a deviation. Third, spoilage may be measured at write-off
rather than at expiry, so stock that expires in the bay is attributed to the disposal period. The
measurement that separates them is the age distribution at issue: if FEFO were working, mean age at
issue would fall. A pick-confirmation percentage cannot show that, because it measures adherence to
a suggestion rather than an outcome.

**Exercise 9.** A strong statement reads, for example: "Segment AX, weekly review, auto-release.
Decoupling point at the finished good because the 12-day lead time is the constraint and the
product is not postponable. Buffer service-led at 98 per cent CSL, `R + L = 19` days, `S = 5,290`
units, implied average 4,450 units. Expiry constraint `Q / 240 <= 16` days, so `Q <= 3,840` units,
which binds at any order cycle longer than 16 days." Every part has a number and a unit, and the
constraint is an inequality a planner can check. A weak statement gives the segment and the buffer
and leaves expiry as a note to "monitor shelf life."

## 14. Interview questions

**1. "What is ABC analysis and what does it not tell you?"**

A strong answer defines it as a ranking by annual consumption value — the letters are labels, not an
acronym — and says at once that it says nothing about variability. A weak answer recites 80/20.

**2. "Why cross ABC with XYZ instead of using one axis?"**

A strong answer says the axes measure independent things — money flow and predictability — and that
value determines how much attention is justified while variability determines what kind. It gives
AZ and CX as the cells that justify the extra axis. A weak answer says more granularity is better.

**3. "How many segments should a planning team run?"**

A strong answer separates the matrix from the operating model: nine cells is the analysis, four to
six clusters is what a team can own, and the collapse is recorded. A weak answer says nine.

**4. "What are the four demand pattern classes and where do the thresholds come from?"**

A strong answer names smooth, erratic, intermittent and lumpy, gives ADI and squared CV as the
axes, and says the 1.32 and 0.49 cut-offs came from comparing Croston and the SBA, with the
Kostenko–Hyndman correction that the boundary is non-linear. A weak answer recites four names.

**5. "What are the three DDMRP zones and what is each for?"**

A strong answer gives yellow as lead-time demand, red as protection with a base and a safety
component, and green as the order band set by the largest of MOQ, cycle demand and red base, with
the formulas. A weak answer describes them as high, medium and low.

**6. "Is DDMRP a new method?"**

A strong answer says the rule is `(s,S)` with `s = TOY` and `S = TOG`, so the arithmetic is
conventional; what is new is the placement discipline, the visible buffer composition and the
demand-qualification rule. A weak answer calls it revolutionary or rebranded min-max.

**7. "What evidence supports DDMRP's performance claims?"**

A strong answer says the literature is simulation-dominated; the case-based study found operational
improvement without inventory reduction and disclaimed attribution; a critical review found two real
deployments in 39 publications; and vendor material is graded D. A weak answer cites a percentage.

**8. "How would you find the service level a DDMRP buffer implies?"**

A strong answer inverts the buffer: compute the implied average on-hand, convert it to an equivalent
order-up-to level, subtract expected demand over the protection period, divide by the demand
standard deviation, and read the normal CDF. A weak answer says it has no service level.

**9. "Why is FEFO not the same as FIFO?"**

A strong answer says one sorts on expiry and the other on receipt; they coincide only when the
production-to-delivery lag is constant, and diverge when lots of different ages arrive together or a
supplier ships older stock. A weak answer says they are the same.

**10. "When is a policy infeasible for a perishable item?"**

A strong answer gives the test: order quantity divided by daily usage greater than the saleable
window at receipt, where the window is shelf life minus lead time, and notes this is a hard
constraint that overrides the cost calculation. A weak answer says "when it expires."

**11. "How does salvage value change the order quantity?"**

A strong answer gives the critical fractile `(p − c) / (p − s)`, notes that disposal cost makes net
salvage negative and lowers the fractile further, and quantifies the effect. A weak answer says
salvage increases the order quantity without giving the mechanism.

**12. "A vendor claims 30 per cent inventory reduction from DDMRP. What do you ask?"**

A strong answer asks what the baseline was, whether service was held constant, whether decoupling
points were chosen or defaulted, whether total inventory or only buffered levels fell, and whether
the result is a simulation. A weak answer asks for the case study.

## 15. Key terms

- **ABC classification** — ranking items by annual consumption value into three groups.
- **XYZ classification** — ranking items by demand variability into three groups.
- **Annual consumption value** — annual units consumed multiplied by unit cost.
- **Lorenz curve** — cumulative value share against cumulative item share.
- **Coefficient of variation (CV)** — standard deviation divided by mean, over a stated bucket.
- **Average demand interval (ADI)** — mean periods between non-zero demand occurrences.
- **Smooth, erratic, intermittent, lumpy** — the four demand classes from crossing ADI and CV.
- **Segmentation** — partitioning a catalogue so different items receive different policies.
- **Policy cluster** — a set of segments sharing one operating policy and one owner.
- **Decoupling point** — a location where a buffer breaks a dependency on upstream lead time.
- **DDMRP** — demand driven material requirements planning; buffer zones at decoupling points.
- **ADU** — average daily usage, in units per day.
- **DLT** — decoupled lead time, from a buffer to the next buffered or purchased point.
- **Lead time factor (LTF)** — multiplier on `ADU × DLT` sizing the red base.
- **Variability factor (VF)** — multiplier on the red base sizing the red safety portion.
- **Red, yellow, green zones** — safety, lead-time demand, and order quantity band.
- **TOR, TOY, TOG** — top of red, yellow and green; the cumulative zone heights.
- **Net flow position** — on-hand plus on-order minus qualified demand.
- **Qualified demand** — past-due, due-today and spike demand counted against the position.
- **Order spike** — a demand event large enough to be netted separately.
- **MRP nervousness** — frequent small rescheduling caused by minor plan changes.
- **Shelf life** — the period from production during which a product remains saleable.
- **Remaining shelf life** — shelf life minus the lot's age at the decision point.
- **FEFO, FIFO, LEFO** — issue by expiry, by receipt, or by newest-expiry where customers choose.
- **Lot or batch** — a quantity produced together and identified by one code.
- **Salvage value** — amount recovered from an unsold unit at the end of its window.
- **Critical fractile** — the service level at which the newsvendor order is optimal.
- **Date-based allocation** — allocating stock by expiry rather than by order priority alone.

## 16. Sources

Grouped by what they can support. Grades: A peer-reviewed or audited, B regulatory or official, C
company case or investor claim, D vendor marketing or product documentation, E press or anecdote.

### 16.1 Peer-reviewed and analytical — grade A

| Source | What it supports |
|---|---|
| Syntetos, Boylan and Croston (2005), "On the categorization of demand patterns," *JORS* 56(5), 495–503 | The four-class taxonomy and the ADI 1.32 / squared CV 0.49 cut-offs |
| Kostenko and Hyndman (2006), "A note on the categorization of demand patterns," *JORS* 57(10), 1256–1257 | The non-linear Croston/SBA boundary and the reduction to two categories |
| Miclo, Lauras, Fontanili, Lamothe and Milian (2015), IEEE IESM | That DDMRP's uncertainty claim had not been scientifically demonstrated |
| Miclo, Lauras, Fontanili and Lamothe (2019), *IJPR* | Simulation evidence that DDMRP outperforms MRP II and Kanban in tested configurations |
| Butturi, De Rosa, Balugani and Gamberini (2021), DAAAM Proceedings | Buffer positioning subjectivity, weak parameter documentation, two real deployments in 39 publications |
| Thürer, Fernandes and Stevenson (2020), *IJPR* | MRP performing worst; DDMRP strongest under bottlenecks |
| Nahmias (1982), "Perishable inventory theory: a review," *Operations Research* | Age-dependent inventory and the absence of simple optimal policies |
| Chaudhary, Kulshrestha and Routroy (2018), *JAMR* | State of perishable inventory modelling across 418 articles |
| Staff and Mustafee (2025), *SIMULATION* | Fixed lead time and shelf life assumptions; scarce real implementations; FEFO, FIFO and LEFO in practice |
| Kandasamy and others (2025), *Journal of Agriculture and Food Research* | Thirteen barriers to FEFO, with storage and expiry tracking foundational |
| Solari, Lysova and Montanari (2025), *Procedia Computer Science* | A critical shelf life of roughly 14 days beyond which perishability stops dominating |

### 16.2 Standards and regulatory — grade B

| Instrument | What it supports |
|---|---|
| GS1 General Specifications, AIs (01), (10), (15), (16), (17), (21), (7003) | Item, lot and date coding, including YYMMDD expiry |
| GS1 Global Traceability Standard 2.0 | Critical tracking events and key data elements |
| EPCIS 2.0 | Event data exchange for lot and date events |
| EU Delegated Regulation 2016/161 | Verification and decommissioning forcing lot and serial records |
| EU Good Distribution Practice, 2013/C 343/01 | Stock rotation, segregation of expired stock, recall capability |
| U.S. Drug Supply Chain Security Act | Transaction information and lot-level tracing |
| Regulation (EU) No 1169/2011 | "Use by" and "best before" date marking |
| Regulation (EC) No 852/2004 and HACCP | Batch identification and withdrawal capability |
| U.S. Food Traceability Rule, FSMA section 204 | Key data elements at critical tracking events |
| EU MDR 2017/745 and U.S. 21 CFR Part 830 | Unique device identification including lot and expiry |

### 16.3 Historical and case material — grade C

| Source | What it supports |
|---|---|
| Dickie (1951), "ABC inventory analysis shoots for dollars, not pennies," *Factory Management and Maintenance* 109(6), 92–94 | The origin of ABC classification at General Electric |
| Ihme and Stratton, Nottingham Trent University | A single-company case simulation reporting fewer alerts and stock-outs with no significant stock reduction, attribution disclaimed |
| Pareto (1896), *Cours d'économie politique*; Lorenz (1905) | The distributional observations ABC was built on |
| Flores and Whybark (1986) | The extension of ABC to multiple criteria, as reported in the review literature |

The Dickie citation is grade C because the attribution rests on several secondary sources rather
than on the 1951 article itself.

### 16.4 Vendor and product documentation — grade D

| Source | Used only for |
|---|---|
| Demand Driven Institute published factor ranges | The LTF and VF bands quoted as guidance, not as a standard |
| Microsoft Dynamics 365 DDMRP buffer profile documentation | Zone definitions and the net flow position calculation |
| IFS Cloud DDMRP documentation | Zone definitions and decoupled versus unprotected lead time |
| DDMRP buffer calculator and explainer pages | The worked structure of a zone calculation, cross-checked against ERP documentation |

Grade D is correct even where the documentation is accurate: it establishes what a vendor documents
about its own implementation and nothing about adoption or performance. No claim in this file about
DDMRP's effectiveness rests on a vendor source.

### 16.5 Inference rather than finding

Four statements are the author's analysis, not sourced to any study: that a team can operate four
to six policy clusters; that the buffer for item 8821-K is equivalent to roughly 99 per cent
service, which follows from section 6.4 and is true only for those parameters; that date coding
converted perishability into a control problem; and that the green zone orders near the median
rather than at the critical fractile.

## 17. Links

Within this module:

- Module index: [`_index.md`](_index.md)
- The cost-service frontier the policy is optimised against: [`01_cost-service-frontier.md`](01_cost-service-frontier.md)
- EOQ and the lot-size constraints that set the green zone: [`02_eoq-limits.md`](02_eoq-limits.md)
- Newsvendor and the critical fractile used in section 6.6: [`03_newsvendor.md`](03_newsvendor.md)
- Safety stock, whose formula is compared against the zones: [`04_safety-stock.md`](04_safety-stock.md)
- Service metrics, which define what an implied CSL means: [`05_service-metrics.md`](05_service-metrics.md)
- Policy families, which own the `(s,S)` rule DDMRP reparameterises: [`06_policy-families.md`](06_policy-families.md)
- Lead-time demand distributions, which supply the comparison's standard deviation: [`07_lead-time-demand.md`](07_lead-time-demand.md)
- Multi-echelon inventory optimisation (MEIO), which decides where the buffer sits across levels: [`08_multi-echelon.md`](08_multi-echelon.md)
- Pooling and postponement, which change what a decoupling point can achieve: [`09_pooling-postponement.md`](09_pooling-postponement.md)

Elsewhere in the programme:

- Part II map: [`../_index.md`](../_index.md)
- Category management, which owns commercial range segmentation: [`../17_assortment_category_retail_analytics/_index.md`](../17_assortment_category_retail_analytics/_index.md)
- Aggregation and consistency, which governs the bucket used for CV: [`../08_hierarchical_intermittent_probabilistic_forecasting/02_aggregation-consistency.md`](../08_hierarchical_intermittent_probabilistic_forecasting/02_aggregation-consistency.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file: [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)

Sector detail lives in Module 22 (healthcare and pharmaceutical operations) and Module 32 (food and
agriculture), named here rather than linked because neither is yet written.

What you now have at the end of Module 10: a frontier to price service against, a lot-size rule and
its limits, a single-period rule, a buffer formula, service measures that are not interchangeable,
four policy families and a way to write one down, a lead-time demand distribution, a multi-echelon
placement logic, a pooling argument, and the three things that stop any of it being applied
uniformly. A policy is only as good as its segmentation, its distribution of buffers, and its
handling of the item's own ageing.
