# Lead-Time Definitions

> Module 01A — Foundations: Units, Time, and Assumptions
> Topic 05 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01A.04

---

## 1. Why this topic exists

Lead time is the most quoted and least defined number in supply chain. It appears in contracts,
in item master records, in safety-stock formulas, in customer promises, in capacity plans, and
in every delivery scorecard. In a typical organisation it is one word covering at least five
different durations, two different epistemics, and one distribution that nobody has plotted.

Topic 04 discretised time. This topic gives the first of the two quantities a planner actually
needs — how long a replenishment takes — the same treatment: a stated start, a stated end, a
stated unit, and a stated shape. Topic 11 will take the second quantity, demand, and put the two
together.

Three short cases set the shape of the problem.

**One.** A buyer negotiates a 14-day contracted lead time. The supplier can ship in 5. The
14 sits in the enterprise resource planning (ERP) system as the item's planned lead time,
drives the reorder point, and holds nine days of pipeline inventory that the business does not
need. Nobody removes it, because the contract says 14 and because the number has been "right"
for six years. It is not a measurement. It is a policy with a signature on it.

**Two.** Two analysts compute on-time delivery for the same week and get 91% and 78%. Both are
correct. One measured against the date the supplier confirmed at order acknowledgement; the
other measured against the date the customer originally asked for. The gap is not error. It is
the difference between two legitimate questions, and the organisation has never decided which
one it is answering.

**Three.** A safety-stock model uses a 30-day expected lead time and the demand variability
during those 30 days. The lead time has a standard deviation of 8 days. The model reports a 95%
cycle service level and the operation delivers something near 40%. No parameter was mistyped.
The model answered a question about a supply chain in which lead time is a constant, and the
company does not operate one.

Each case is a definitions failure, not a modelling failure. The arithmetic in all three is
correct. What is wrong is the referent of the word.

**Why five names matter.** The five durations that get called "lead time" are measured by
different people, in different systems, against different clocks, and each handoff between two
of them is where a day or two quietly disappears. A report that sums three of the five and
calls the result the lead time will be internally consistent and systematically short. Worse,
it will be short by an amount that looks like a small rounding error and behaves like a
structural bias.

**Why a data scientist should care.** Four things you will build depend on this definition: the
safety-stock and reorder-point calculation in Topic 01.05 and Module 10, the forecast horizon
and lag structure of every time-series feature in Module 07, the feature set of any
supplier-delay model in Module 14, and every "on-time" service metric in Modules 19 and 25. A
lead time that is wrong by 15% moves a reorder point by more than 15% of lead-time demand, and
an omitted lead-time variance moves safety stock by a factor of five to ten. Neither error
announces itself.

**Where this leads.** Topic 06 turns to the other unit that silently mixes: currency. Topic 11
takes the lead-time distribution introduced here and pairs it with the demand distribution.
Module 10 uses both to set inventory policy. The vocabulary in section 4 is the one the rest of
the program assumes.

## 2. Learning outcomes

By the end of this topic you can:

1. **Name** the five durations commonly called lead time, and for each **state** the start
   event, the end event, the owner of the record, and the unit it is normally expressed in.
2. **Distinguish** nominal, actual, promised, and planned lead time, and **say** which of them
   may enter a safety-stock calculation and which may not.
3. **Identify**, from a pair of timestamps and a system description, which measurement window a
   reported lead time actually covers, and **estimate** the days lost at each handoff.
4. **Compute** the mean, standard deviation, median, and 95th percentile of a lead-time sample,
   and **explain** why the mean alone is the wrong summary for a right-skewed variable.
5. **Derive** the standard deviation of lead-time demand from demand and lead-time variability,
   and **state** the independence assumption the derivation requires.
6. **Quantify** the safety stock added by the lead-time variability term at plausible parameter
   values, and **identify** the lead-time standard deviation at which that term overtakes the
   demand term.
7. **Explain** why a normal fit to a right-skewed lead time understates the upper tail, and
   **demonstrate** the effect on the 95th and 99th percentiles.
8. **Distinguish** on-time-to-request from on-time-to-promise from on-time-to-current-promise,
   and **compute** all three from the same order history.
9. **Diagnose** a lead-time bias in a planning system, separating a contracted policy from a
   measured capability, and **propose** the evidence that would distinguish them.
10. **Write** a lead-time specification for a dataset: stage boundaries, timestamps, unit,
    calendar basis, estimation window, and distribution family, each with a test.

Outcomes 4 to 8 are exercised with arithmetic in section 12 and answered in section 13.

## 3. The core mental model

Hold three pictures. The first is a chain of five durations with a timestamp at each joint.
The second is three numbers wearing the same name. The third is a histogram with a hard left
wall and a long right arm.

### 3.1 One chain, five durations, five owners

```text
FROM DECISION TO AVAILABLE STOCK — five durations, five owners, five clocks

  DECISION          HANDOFF            HANDOFF            HANDOFF            STOCK
  order raised ---> PO sent ---------> supplier --------> goods -----------> put away
                                     confirms            arrive             complete
  |                 |                  |                  |                  |
  |<-- 2.4 d ------>|<-- 0.9 d ------->|<- 17.0 d -------->|<-- 11.2 d ------>|<- 2.7 d ->|
  (buyer)           (EDI)              (supplier)         (carrier)          (DC)
```

Read the diagram left to right and each joint is a handoff between two organisations' systems.
The duration between two joints is owned by whoever holds the record at that stage, and it is
measured on that party's clock, in that party's unit, rounded by that party's reporting habit.

| Window | From | To | Mean in the example | Who owns the record |
|---|---|---|---:|---|
| Order processing | Requisition raised | Purchase order transmitted | 2.4 d | Buyer, ERP |
| Order transmission | PO transmitted | Supplier confirmation | 0.9 d | EDI gateway, supplier |
| Supplier lead time | Confirmation | Goods ready or shipped | 17.0 d | Supplier |
| Transit time | Shipped | Arrived at destination dock | 11.2 d | Carrier |
| Receiving and put-away | Arrived | Available to promise | 2.7 d | Distribution centre |
| **Total replenishment** | Requisition raised | Available to promise | **34.2 d** | Nobody |

The last row is the one that matters and the one with no owner. It is the sum, and it is the
only figure that belongs in a reorder point. The nominal figure in the item master in this
example is 30 days, which matches none of the six rows: it is close to the supplier lead time
(31.0 days measured from confirmation), which is where it probably came from, and it is 4.2
days short of the truth.

Three properties of this chain are worth memorising.

**Property 1 — the sum of the parts is not the total.** If each stage owner rounds up, the sum
overstates. If each stage is measured from its own start and the boundaries are not shared, the
total understates. Both happen in the same company, in different item categories.

**Property 2 — the days vanish at the joints, not in the stages.** The 2.4 days of requisition
queue and the 0.9 days of transmission sit outside every supplier-reported lead time. They are
visible only in the buyer's ERP timestamps, which nobody exports. Measuring "supplier lead
time" from confirmation excludes 3.3 days, about 10% of the total, consistently.

**Property 3 — the clock changes at every joint.** The buyer works in working days, the supplier
confirms in calendar days, the carrier reports departure in local time, and the DC confirms
put-away at end of shift. Topic 04's four clocks reappear as five stage clocks, and the
conversion is where a Friday shipment becomes a Monday receipt.

### 3.2 Three numbers wearing one name

```text
   NOMINAL                    ACTUAL                     PROMISED
   what the contract,         what the receipts          what someone told
   the item master, or        show, measured             a customer, with
   the catalogue says         between two named          a date attached
                              timestamps
        |                          |                          |
        |  a policy or a           |  a sample from a         |  a commitment,
        |  capability claim        |  population              |  and an outcome
        |                          |                          |
        v                          v                          v
   "14 days"                  "mean 34.2, sd 8.3,         "arrives Friday"
                               p95 51, n = 45"

   Enters planning as        The only one that may       Never an input to a
   a parameter.              enter a safety-stock        lead-time distribution;
                             calculation.                it is what you are judged on.
```

The distinctions that matter:

- **Nominal** is a single number with no variance attached. It is what the contract says, what
  the supplier's published lead time says, or what somebody typed into the item master in 2019.
  It may be a capability (the supplier could do 5 days) or a policy (the contract allows 14).
  Either way it is not a measurement.
- **Actual** is a sample. It has a mean, a spread, a shape, and a sample size. It is the only
  one of the three that carries information about the distribution of future lead times, and
  even it carries that information only if the process has not changed since the sample.
- **Promised** is a date, not a duration, though a duration can be inferred from it. It is
  endogenous: the promise is usually set from the nominal plus a buffer, so using the promise
  to estimate the distribution is circular.

A fourth word is used in planning systems and deserves its own line. **Planned** lead time is
the offset the material requirements planning (MRP) engine uses to schedule supply. It is
neither nominal nor actual; it is a control parameter that someone set, and it can be longer
than both if the planner padded it.

> **Note:** A contract lead time longer than the supplier's capability is not automatically
> waste. It may buy allocation priority, protect against expedite premiums, or reflect a
> service level the supplier has agreed to guarantee. The error is not the number. The error is
> recording a policy in a field that the planning engine reads as a capability, and then
> computing safety stock as though the extra days were uncertain in the ordinary way.

### 3.3 The distribution behind the average

```text
  45 RECEIPTS, ORDER CREATION TO AVAILABLE, CALENDAR DAYS

  days    count  histogram
  20-24     2    ##
  25-29    12    ############
  30-34    14    ##############
  35-39     8    ########
  40-44     4    ####
  45-49     2    ##
  50-54     1    #
  55-59     1    #
  60-64     1    #
                 +----+----+----+----+----+
                  0    5   10   15

  mean 34.2 d   median 32 d   p95 51 d   max 61 d
  five fastest mean 24.6 d; five slowest mean 52.2 d
  normal fit at the same mean and sd: p95 = 47.7 d, 3.3 d short
  normal fit places 7.0% of its mass below the fastest observation ever seen
```

Four properties of this shape recur across every lane, supplier, and industry where somebody
has actually plotted the data.

**A hard left wall.** There is a physical minimum: the fastest possible production plus the
fastest possible transit. No observation can fall below it. The wall is not a statistical
artefact; it is a constraint. A normal fit ignores it and assigns probability to lead times
that cannot occur.

**A mode near the wall.** Most receipts cluster within a few days of the best achievable time,
because most orders are unremarkable. This is why the median is usually well below the mean.

**A long right arm.** Congestion, customs inspection, quality holds, carrier rollover,
allocation shortfalls, and weather produce a tail that is not a scaled copy of the body. It is
a different mechanism operating occasionally. The tail is where the stock-outs live.

**A different tail in every period.** The left wall is stable; the right arm moves with
freight markets, port congestion, and supplier load. This is why a lead-time distribution
estimated on a calm year will under-protect in a tight one, and why the estimation window is a
modelling decision with a service consequence.

### 3.4 Why the variance term is not a rounding error

The standard deviation of demand during the lead time, for independent demand and lead time, is

```text
sigma_LTD = sqrt( L x sigma_d^2 + mu_d^2 x sigma_LT^2 )
```

where `L` is the mean lead time in days, `sigma_LT` the lead-time standard deviation in days,
and `mu_d` and `sigma_d` the mean and standard deviation of daily demand. Topic 01.05 writes
the same quantity as `sigma_L`; this topic uses `sigma_LTD` to keep it distinct from the
lead-time standard deviation.

The two terms under the root are directly comparable, both in units. The demand term is
`sqrt(L) x sigma_d`. The lead-time term is `mu_d x sigma_LT`. They are equal when

```text
sigma_LT* = ( sigma_d / mu_d ) x sqrt(L)  =  CV_d x sqrt(L)
```

For the item in section 6, `CV_d` is 0.20 and `L` is 34.2 days, so the crossover is at a
lead-time standard deviation of **1.2 days**. Beyond 1.2 days — that is, for essentially every
imported or made-to-order item in existence — the lead-time variability term is the larger of
the two.

At a standard deviation of 2 days, which is 5.8% of a 34-day mean and sounds negligible, the
lead-time term is 66 units against the demand term's 39 units, and safety stock at a 95% cycle
service level rises from 63 units to 125. Two days is not a rounding error. It is the whole
calculation.

## 4. Definitions and vocabulary

The fourth column names the unit or coding system, because a definition without one is the
defect this module exists to prevent.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Lead time | Elapsed time between a defined start event and a defined end event in a supply process | Used with neither event named | Calendar days, unless stated |
| Nominal lead time | The single figure stated in a contract, catalogue, or item master | Treated as a measurement of what happens | Days; often working days |
| Actual lead time | The observed elapsed time on a completed order, between two named timestamps | Computed from whichever two dates are convenient | Calendar days, with the timestamps named |
| Promised lead time | The duration implied by a date someone committed to a customer | Used as an input to a demand or safety-stock model | Days, derived from a date pair |
| Planned lead time | The offset the planning engine uses to schedule supply | Assumed to equal the actual lead time | Days, in the planning calendar |
| Quoted lead time | The figure a supplier offers before an order exists | Treated as a commitment | Days, usually calendar |
| Contractual lead time | The lead time the supply agreement obliges the supplier to meet | Confused with the supplier's capability | Days, plus remedy terms |
| Capability lead time | The fastest lead time the process can reliably deliver when unconstrained | Treated as the planning figure | Days, with a stated reliability |
| Order processing time | Requisition raised to purchase order transmitted | Omitted from the total | Working days, usually |
| Order transmission time | PO transmitted to supplier acknowledgement or confirmation | Merged into supplier lead time | Hours to days |
| Supplier lead time | Confirmation to goods ready for despatch or shipped | Started at PO creation instead | Calendar days |
| Production lead time | Release to production to completion of the last operation | Used for purchased items | Working days or calendar days |
| Manufacturing lead time | Theoretical or observed time from order release to finished goods | Confused with flow time | Working days |
| Transit time | Despatch from origin to arrival at destination | Used to mean door-to-door | Calendar days, by mode |
| Receiving time | Arrival at dock to complete unloading and inspection | Omitted from the total | Hours to days |
| Put-away time | Inspection complete to stock available to promise | Assumed to be zero | Hours |
| Replenishment lead time | Order release to stock availability at the ordering location | Equated with supplier lead time | Calendar days |
| Total replenishment lead time | Requisition raised to available to promise, across all stages | Reported by summing rounded stage means | Calendar days |
| Cumulative lead time | The longest path from raw material to finished good, across tiers | Confused with the item's own lead time | Calendar days |
| Order-to-delivery cycle time | Customer order receipt to customer acceptance | Compared with internal measures without adjustment | Calendar days |
| Dwell time | Time an order waits for a reason the supplier does not control | Counted as supplier slowness | Calendar days |
| Queue time | Time an order waits for capacity or for a prior operation | Included in touch time | Working hours or days |
| Touch time | Time work is actually performed on the order | Confused with lead time | Minutes or hours |
| Lead-time demand | Demand expected to occur during the replenishment lead time | Computed on the review period instead | Units |
| Lead-time variability | The standard deviation of the lead-time distribution | Omitted, or replaced by a range | Days |
| Lead-time bias | A systematic difference between the lead time used and the lead time experienced | Treated as random error | Days, signed |
| Safety lead time | Time added to a planned lead time to absorb timing uncertainty | Confused with safety stock | Days |
| Safety stock | Inventory held to cover variability in demand and lead time | Treated as a policy constant | Units |
| Reorder point | Inventory position at which replenishment is triggered | Set from mean lead-time demand alone | Units |
| Cycle service level (CSL) | Probability of not stocking out within a replenishment cycle | Reported as fill rate | Proportion |
| Fill rate | Fraction of demand units served from stock | Reported as CSL | Proportion |
| Service factor (z) | The standard normal quantile corresponding to a CSL | Chosen without naming the metric | Dimensionless |
| On-time delivery (OTD) | Fraction of orders delivered by a stated date basis | Reported without stating the basis | Proportion |
| On-time in full (OTIF) | Fraction of orders delivered complete and by a stated date | Definition varies by customer | Proportion |
| On-time to request | Delivery by the date the customer asked for at order placement | Confused with on-time to promise | Proportion |
| On-time to promise | Delivery by the date the supplier confirmed | Used to flatter performance | Proportion |
| Requested date | The delivery date the customer asked for | Overwritten by the confirmation | Date |
| Promised date | The delivery date the supplier committed to | Silently updated on reschedule | Date |
| Original promised date | The first date the supplier committed to, retained separately | Not stored, so cannot be measured | Date |
| Must-arrive-by date (MABD) | The latest acceptable arrival date set by a retailer | Treated as a request rather than a constraint | Date |
| Reschedule | A change to a committed date, by either party | Counted as on-time after the change | Date pair, with initiator |
| Expedite | Acceleration of an order beyond its planned lead time | Used to correct a planning error, repeatedly | Premium cost plus days saved |
| Lead-time compression | A structural reduction in a stage duration | Confused with expediting | Days removed |
| Pipeline inventory | Goods owned but not yet available at the destination | Omitted from the inventory position | Units |

Six rows carry most of the damage.

**Nominal versus actual.** The nominal is a claim; the actual is a sample. Organisations
routinely validate the nominal by checking that the actual "roughly matches", which is a test
with no power, because a nominal that is 10% high will pass it in most months. The correct test
is a distributional comparison: does the nominal sit inside the central 50% of the actual
distribution, and is the actual's 95th percentile inside the buffer the nominal implies?

**Replenishment versus supplier lead time.** The supplier lead time excludes order processing,
transmission, and put-away — 3.3 days out of 34.2 in the example, roughly 10%. The
replenishment lead time is what the reorder point needs. Systems label both "lead time", and
the field name rarely tells you which.

**Safety lead time versus safety stock.** Safety lead time is a time buffer; safety stock is a
quantity buffer. They are not interchangeable, and the literature does not give one answer.
Safety stock hedges quantity uncertainty better; safety lead time hedges timing uncertainty
better; and safety lead time has a documented pathology, which is that the shop floor learns
to consume it (section 9.3).

**On-time to request versus on-time to promise.** Two different measurements of two different
things. Request-date performance measures the promise the customer heard; promise-date
performance measures the promise the supplier made itself. The second is always better, and
the gap between them is a measure of how much the supplier padded at confirmation.

**Promised date versus original promised date.** Most systems overwrite the promised date when
an order is rescheduled. Without a separate field for the original, on-time performance is
measured against a moving target, and every reschedule improves the score without changing
anything physical. Section 5, step 14 treats this.

**Dwell time versus supplier lead time.** A pre-order placed 90 days ahead and shipped in 10
days has a 90-day gross cycle time and a 10-day process time. SCOR distinguishes the two for
exactly this reason; a dashboard that reports the gross figure as supplier performance
penalises the customer's own forward buying.

## 5. The mechanics, step by step

This section turns a stack of order and receipt records into a lead-time distribution that can
be used in a planning calculation. It is fourteen steps in four phases. Each step names the
rule, the data it needs, and the failure that occurs when it is skipped.

### Phase A — Classify before you measure

**Step 1. Decide which of the three numbers you have been given.**

- Rule: every lead-time figure is nominal, actual, or promised. Ask for the two events it spans
  and the document it came from. A contract number is nominal. A receipt history is actual. A
  date quoted to a customer is promised.
- Data: the source document identifier alongside the number.
- If skipped: a nominal figure and an actual sample get averaged together, which produces a
  number that is neither a policy nor a measurement and cannot be defended in either role.

**Step 2. Write down the measurement window, as a pair of events.**

- Rule: "lead time" is meaningless until both endpoints are named. The pair must be
  unambiguous: requisition raised, PO transmitted, supplier confirmation, despatch, arrival at
  dock, receipt posted, put-away complete. Pick two and keep them for every observation.
- Data: two timestamp columns, both event times (Topic 04), not record times.
- If skipped: the window drifts. Early records are measured from despatch because that was the
  only date available; later records are measured from confirmation because the EDI feed
  arrived. The series shows a step change that is entirely a data-model change.

**Step 3. Fix the unit and the calendar basis.**

- Rule: state whether the figure is calendar days or working days, and convert to one basis
  before any arithmetic. Convert with a real working-day calendar for the relevant jurisdiction,
  not with a 5/7 multiplier.
- Data: a calendar table (Topic 04) with working-day flags per site.
- If skipped: a 20-working-day lead time becomes 20 calendar days, an understatement of eight
  days, and the error is largest around holiday periods, which is precisely when lead times
  are longest.

> **Note:** Supplier lead times are almost always quoted in calendar days. Internal process
> times are almost always managed in working days. The two conventions meet at the supplier
> boundary, which is also the boundary where the largest duration sits. A conversion error
> there is multiplied by the largest number in the chain.

### Phase B — Measure the five durations

**Step 4. Measure order processing.**

- Rule: requisition created to PO transmitted, per order. This is the buyer's queue: approval
  thresholds, budget checks, batch release schedules, and the Friday-afternoon problem.
- Data: requisition timestamp, PO transmission timestamp, both from the buyer's ERP.
- If skipped: the queue is attributed to the supplier, the supplier is judged slow for a delay
  it never caused, and the actual fix — an approval threshold — is never found.

**Step 5. Measure order transmission and acknowledgement.**

- Rule: PO transmitted to supplier confirmation. This includes the outbound queue, any manual
  re-keying, and the supplier's order-entry batch.
- Data: PO transmission timestamp, order acknowledgement timestamp (ANSI X12 855 or EDIFACT
  ORDRSP), plus the confirmation date stated inside the message.
- If skipped: the difference between "we sent it Tuesday" and "they saw it Thursday" is
  invisible, and so is the manual re-keying that causes it.

**Step 6. Measure supplier lead time.**

- Rule: confirmation to goods ready or shipped. State which end you use, because "ready" and
  "shipped" differ by the despatch queue. Prefer the ship event if you can get the carrier's
  timestamp; otherwise state that you are measuring readiness.
- Data: supplier confirmation timestamp, advance ship notice (ASN) despatch timestamp, carrier
  pickup event.
- If skipped: the despatch queue — often one to three days at a busy supplier — is invisible,
  and the supplier's on-time despatch performance cannot be separated from its production
  performance.

**Step 7. Measure transit time.**

- Rule: carrier pickup to arrival at the destination dock. Not "shipped" to "delivered to the
  customer", and not despatch date to invoice date.
- Data: carrier pickup scan, port or hub events, arrival scan at destination.
- If skipped: the receiving delay is folded into transit, and the distribution centre is never
  identified as a source of lateness even when it is one.

**Step 8. Measure receiving and put-away.**

- Rule: arrival at dock to stock available to promise. Two sub-events: receipt posted, and
  put-away confirmed. Use put-away, because that is when a planner can allocate the stock.
- Data: gate-in timestamp, receipt posting timestamp, put-away confirmation timestamp.
- If skipped: phantom stock-outs. Goods are physically on the shelf and invisible to the
  system, so the planner orders again.

**Step 9. Assemble the total and reconcile it.**

- Rule: total replenishment lead time is the sum of the five on one basis. Then reconcile
  against an independent measure — at minimum, the difference between requisition date and
  put-away date on a sample of recent orders — and explain any gap larger than a day.
- Data: a stage-level fact table with one row per order and one column per stage duration.
- If skipped: the sum of five measured stages and the end-to-end elapsed time will disagree,
  and nobody will know which is wrong.

The five stages and their typical handoff losses:

| Stage | Start event | End event | Typical handoff loss | Where it hides |
|---|---|---|---|---|
| Order processing | Requisition raised | PO transmitted | 0.5–3 d | Approval queue, batch release |
| Transmission | PO transmitted | Supplier confirmation | 0.2–2 d | Outbound queue, manual re-key, ack batch |
| Supplier | Confirmation | Ready or shipped | 5–60 d | Despatch queue at the end |
| Transit | Picked up | Arrived at dock | 1–45 d | Customs, rollover, port dwell |
| Receiving | Arrived | Available to promise | 0.5–3 d | Unload queue, inspection, put-away scan |

### Phase C — Characterise the distribution

**Step 10. Plot the sample before summarising it.**

- Rule: a histogram with at least eight bins, plus a time series of lead time by order date.
  Look for bimodality (normal versus expedited), a step change (process or supplier change),
  and a heavy tail.
- Data: one observation per completed order, with the order date.
- If skipped: a bimodal distribution gets a mean that describes neither mode, and a supplier
  with two routings looks like one supplier with high variance.

**Step 11. Estimate on a stated, defensible window.**

- Rule: state the window, the number of observations, and why the window ends where it does.
  Exclude structural breaks. Report the observation count, not the calendar span.
- Data: the sample plus an exclusion log.
- If skipped: 45 observations over six months and 45 observations over four years are treated
  as the same evidence, and a supplier who changed plant mid-year is modelled as one process.

**Step 12. Test the shape and choose a family.**

- Rule: compare the empirical quantiles with a normal fit. If the sample has a floor and a long
  right arm, the normal will be poor in both tails. Fit a lognormal, gamma, or Weibull as well,
  and test with a Kolmogorov–Smirnov or Anderson–Darling statistic.
- Data: the sample and the fitted parameters.
- If skipped: the normal's 95th percentile is used to size a buffer, and it is a day or more
  below the empirical 95th percentile in exactly the cases where it matters.

### Phase D — Use it in a decision

**Step 13. Put the distribution into the reorder point, not the mean.**

- Rule: reorder point equals mean lead-time demand plus the service factor times
  `sigma_LTD`. Both terms use the actual distribution. Never substitute the nominal.
- Data: the fitted or empirical parameters, plus the service metric and target.
- If skipped: the calculation is internally consistent and systematically short, and the
  shortfall scales with the lead-time variance, which is largest where supply is hardest.

**Step 14. Separate policy from capability in the master data.**

- Rule: carry at least two fields — the contractual or policy lead time, and the measured
  capability distribution. Make the planning engine read the one you have chosen, and label
  the choice.
- Data: item master fields with their definitions documented, plus a review date.
- If skipped: a negotiated buffer becomes a physical constant, and every year it survives adds
  another year of evidence that it was "right".

Two of these steps are skipped in almost every organisation: step 3 (the calendar basis) and
step 13 (the variance term). Step 14 is skipped in every organisation that has never had an
argument about it.

## 6. Worked example

Meridian Components is a fictional manufacturer of industrial pumps. It buys a cast housing
from a single supplier in another country. All figures are constructed to be internally
consistent.

### 6.1 The setting

The item master records a **nominal supplier lead time of 30 days**. Procurement believes this
is the contract figure. The buyer who set it in 2019 says it came from the supplier's quotation.
Nobody has tested it since.

Forty-five receipts are available, covering 14 months. Each has a requisition date, a PO
transmission date, a supplier confirmation date, a carrier pickup, a gate-in, a receipt posting,
and a put-away confirmation. The sample below is the total replenishment lead time measured
from requisition raised to put-away complete, in calendar days:

```text
22 24 25 26 26 27 27 27 28 28 28 29 29 29 30 30 30 30 31 31 31
32 32 32 33 33 34 34 35 35 35 36 36 37 38 39 40 41 42 44 46 48
51 55 61
```

### 6.2 Descriptive statistics

| Statistic | Value | Note |
|---|---:|---|
| Observations | 45 | Receipts, 14 months, no exclusions |
| Mean | 34.16 d | 1,537 / 45 |
| Median | 32 d | Below the mean, as expected for a right skew |
| Standard deviation | 8.25 d | Sample standard deviation, denominator 44 |
| Coefficient of variation | 0.24 | 8.25 / 34.16 |
| 95th percentile | 51 d | Nearest-rank, `ceil(0.95 x 45)` = 43rd value |
| Maximum | 61 d | One observation, an 11-day customs hold |
| Mean of five fastest | 24.6 d | Physics floor is near here |
| Mean of five slowest | 52.2 d | 1.89 times as far from the mean as the fast arm |

The arithmetic for the two moments, shown so it can be checked:

```text
   sum(x)        = 1,537
   sum(x^2)      = 55,493
   mean          = 1,537 / 45                     = 34.1556
   SS_dev        = 55,493 - 45 x 34.1556^2        = 55,493 - 52,497 = 2,996
   sample var    = 2,996 / 44                     = 68.09
   sample sd     = sqrt(68.09)                    = 8.25
   p95 (nearest) = 43rd smallest value            = 51
```

Three observations about the shape. The mean exceeds the median by 2.2 days, which is the
signature of a right skew and is the first thing to check before using a mean in a calculation.
The five slowest observations sit 18.0 days above the mean while the five fastest sit 9.6 days
below it, a ratio of 1.89 — the tail is not a mirror image of the body. And the maximum is 61
days, 26.8 days above the mean and well beyond the normal fit's 99th percentile of 53.4.

### 6.3 What a normal fit does to the tail

Fit a normal with the sample mean and standard deviation and compare the quantiles:

| Quantile | Empirical | Normal fit | Difference |
|---|---:|---:|---:|
| 5th | 25 d | 20.6 d | normal is 4.4 d too low |
| 50th | 32 d | 34.2 d | normal is 2.2 d too high |
| 95th | 51 d | 47.7 d | normal is 3.3 d too low |
| 99th | 61 d (max) | 53.4 d | normal is 7.6 d too low |

The pattern is diagnostic. The normal is too high in the middle and too low in both tails,
because it is putting mass below the physical floor. Specifically, `Phi((22 - 34.16) / 8.25)`
= `Phi(-1.474)` = 7.0%: the fit assigns a 7% probability to a lead time faster than the fastest
delivery ever observed on this lane.

The consequence for a service calculation is not symmetric. Safety stock is set from an upper
quantile. A buffer sized on the normal's 95th percentile is 3.3 days of demand short of the
empirical 95th, every cycle. At the item's demand rate of 32.88 units per day, that is about
108 units.

> **Note:** The normal fit is not "wrong" in the sense of a computational error, and for a
> fast-moving item with a coefficient of variation below about 0.4 the error in the safety
> stock calculation may be modest (Tyworth and O'Neill, 1997). The problem here is the
> combination that is common in practice: a modest CV with an upper-tail service target. The
> tail is exactly where the fit is worst and exactly where the decision is made.

### 6.4 Safety stock under three policies

Take annual demand of 12,000 units, or 32.88 units per day, with a daily demand standard
deviation of 6.6 units (coefficient of variation 0.20). Target cycle service level is 95%, so
`z = 1.645`.

```text
   (a) nominal, zero variance       SS = 1.645 x 6.6 x sqrt(30)          =  59 units
   (b) mean actual, zero variance   SS = 1.645 x 6.6 x sqrt(34.16)       =  63 units
   (c) mean and sd actual           SS = 1.645 x sqrt(34.16 x 6.6^2
                                              + 32.88^2 x 8.25^2)
                                       = 1.645 x sqrt(1,488 + 74,121)
                                       = 1.645 x 275.1                  = 453 units
```

| Policy | Lead time used | Safety stock | Reorder point | Realised CSL |
|---|---|---:|---:|---:|
| (a) Nominal, deterministic | 30 d fixed | 59 | 1,046 | 39% |
| (b) Mean actual, deterministic | 34.16 d fixed | 63 | 1,187 | 59% |
| (c) Mean and standard deviation | 34.16 d, sd 8.25 | 453 | 1,577 | 95% |

The reorder points and realised service levels follow from the lead-time demand distribution,
which has mean 34.16 x 32.88 = 1,123 units and standard deviation 275 units. Policy (a) places
the reorder point at 30 x 32.88 + 59 = 1,046, which is 77 units below the mean of the
distribution it is supposed to cover. The realised cycle service level is
`Phi((1,046 - 1,123) / 275)` = `Phi(-0.28)` = 39%.

Policy (b) corrects the mean and leaves the variance. It recovers 20 points of service and is
still 36 points short of target. Policy (c) is the only one that meets the target.

Three things follow.

**The nominal error costs 20 points of service.** Using 30 days instead of 34.16 shifts the
reorder point down by 4.16 x 32.88 = 137 units. That is the pure mean effect, and it is large
because the item has high daily demand.

**The variance error costs 36 points.** Adding the lead-time term raises safety stock from 63
to 453 units, a factor of 7.2. Of the 275.1 units of `sigma_LTD`, the lead-time term contributes
272.3 and the demand term 38.6 — 98% of the variance in lead-time demand comes from lead-time
variability, not demand variability.

**The buffer is not free and may be the wrong lever.** 453 units at a unit cost of, say, EUR 40
is EUR 18,100 of additional inventory. The alternative is to attack the lead-time variance:
if the 11-day customs hold is a documentation problem that can be removed, the 95th percentile
falls and so does the buffer. Section 6.6 returns to this.

### 6.5 Where the days vanish

Now measure the same 45 orders from the supplier confirmation date instead of the requisition
date:

| Stage | Mean duration | Cumulative from requisition | Cumulative from confirmation |
|---|---:|---:|---:|
| Order processing (requisition to PO sent) | 2.4 d | 2.4 | — |
| Transmission (PO sent to confirmation) | 0.9 d | 3.3 | — |
| Supplier production (confirmation to shipped) | 17.0 d | 20.3 | 17.0 |
| Transit (shipped to dock) | 11.2 d | 31.5 | 28.2 |
| Receiving and put-away (dock to available) | 2.7 d | 34.2 | 30.9 |

The supplier-reported lead time is 30.9 days. The true replenishment lead time is 34.2. The
difference is **3.3 days**, which is the buyer's approval queue plus the EDI transmission and
acknowledgement. It is 9.6% of the total and it belongs to the buyer, not the supplier.

And 30.9 rounds to 31, which a hurried analyst writes down as 30. That is where the nominal
almost certainly came from: somebody measured the supplier-controlled portion once, rounded
it, and the figure entered the master data without its measurement window attached.

The safety-stock consequence of the 3.3 days alone is small, because the variance term
dominates: policy (c) with 30.9 days instead of 34.16 gives 453 units, unchanged at this
precision. But the reorder point consequence is not small:

```text
   ROP at 30.9 d  = 30.9 x 32.88 + 453 = 1,016 + 453 = 1,469 units
   ROP at 34.16 d = 34.16 x 32.88 + 453 = 1,123 + 453 = 1,576 units
   difference      = 107 units, 9.5% of mean lead-time demand
```

**The mean governs the reorder point; the variance governs the safety stock.** A 3.3-day error
in the mean barely moves the buffer but moves the trigger point by more than a week of demand.
This is why the two error types must be diagnosed separately. A model that gets the variance
right and the window wrong will hold the correct buffer at the wrong level, and it will run
out on a schedule.

### 6.6 What the analyst should recommend

Fix the master data to 34.2 days with the measurement window recorded; replace the scalar with
the distribution; attack the tail rather than the mean, because the customs hold on the five
slowest receipts is worth more than any improvement at the mode; and re-open the contract
conversation with the distribution in hand.

## 7. Code, query, or calculation

One block. It computes the statistics in section 6 and the three safety-stock policies, so the
arithmetic can be checked rather than trusted.

```python
import math, statistics as st

x = [22,24,25,26,26,27,27,27,28,28,28,29,29,29,30,30,30,30,31,31,31,
     32,32,32,33,33,34,34,35,35,35,36,36,37,38,39,40,41,42,44,46,48,
     51,55,61]                                  # days, requisition -> put-away

mean_L, sd_L = st.mean(x), st.stdev(x)           # 34.16, 8.25
p95 = sorted(x)[math.ceil(0.95 * len(x)) - 1]    # nearest-rank, 51
mu_d, sd_d, z = 32.88, 6.6, 1.645                 # units/day, units/day, 95% CSL

ss_nominal = z * sd_d * math.sqrt(30)                                # 59
ss_mean    = z * sd_d * math.sqrt(mean_L)                            # 63
ss_stoch   = z * math.sqrt(mean_L * sd_d**2 + mu_d**2 * sd_L**2)     # 453
rop        = mu_d * mean_L + ss_stoch                                # 1,577

share = (mu_d**2 * sd_L**2) / (mean_L * sd_d**2 + mu_d**2 * sd_L**2) # 0.980

print(f"n={len(x)} mean={mean_L:.2f} sd={sd_L:.2f} p95={p95}")
print(f"SS {ss_nominal:.0f} / {ss_mean:.0f} / {ss_stoch:.0f}  ROP={rop:.0f}")
print(f"lead-time share of variance = {share:.1%}")
# n=45 mean=34.16 sd=8.25 p95=51
# SS 59 / 63 / 453  ROP=1577
# lead-time share of variance = 98.0%
```

Two checks before you trust an implementation. Compute `p95` by more than one method, because
nearest-rank, linear interpolation, and a fitted distribution differ by two or three days on a
sample of 45 and the buffer moves with the choice. And confirm that the lead-time term uses
`mu_d^2 x sd_L^2`: the dimension check catches the common error, since the term must be in
units squared.

## 8. Data, standards, and regulatory context

There is no international standard that fixes what "lead time" means, which is the root of the
problem in section 1. There are standards that fix the *measurement* of related metrics, and
standards that fix the *documents* in which dates travel. Knowing which governs which is what
lets you tell a definition you may choose from a definition you have inherited.

### 8.1 Metrics that embed a lead-time definition

| Instrument | What it defines | Where the definition bites | Grade |
|---|---|---|---|
| ASCM SCOR, RS.1.1 Customer order fulfilment cycle time | Clock starts at order receipt, ends at customer acceptance | Gross measure includes customer-caused dwell time; SCOR separates a net process-time variant for benchmarking | C |
| APQC customer order cycle time | Order placement to delivery, calendar days | Reported median 12.0 days, 25th percentile 5.0, 75th 25.0, n = 12,074 | C |
| TL 9000 on-time delivery | Delivery against the customer requested date | Quoted lead time and available-to-promise have no bearing; early delivery counts as a miss; a supplier-initiated change still counts as late | B |
| Retailer OTIF scorecards | Delivery against requested date, case fill, arrival basis | No common definition across retailers; the customer's scorecard is the definition that binds | C |

The TL 9000 position is the most explicit published statement of the request-date principle,
and it is worth quoting in substance because it settles an argument that recurs in every
commercial review: if the supplier asks to move the date and the customer agrees, delivery on
the new date is still late, because the change was not customer-initiated. Under that rule,
rescheduling cannot improve a score.

APQC's figure is useful as a calibration point, not as a target. A median of 12 calendar days
from order placement to delivery across 12,074 organisations describes a population dominated
by stocked, domestically sourced items. An imported make-to-order component at 34 days is not
an outlier; it is a different population.

### 8.2 Documents that carry the dates

| Standard or document | What it carries | Why it matters for lead-time measurement | Grade |
|---|---|---|---|
| ANSI X12 855 (PO acknowledgement) | Supplier's confirmed dates and quantities | The confirmation timestamp and the promised date are different events in the same message | B |
| ANSI X12 856 (ASN) | Despatch date, contents, and often a scheduled arrival | The only supplier-side source for the ship event if the carrier feed is unavailable | B |
| EDIFACT ORDRSP / DESADV | European equivalents of 855 and 856 | Same distinction, different segments | B |
| GS1 EPCIS 2.0 | Event data: what, where, when, why | Makes the physical event time shareable across parties, which is what an actual lead time needs | B |
| Incoterms 2020 | The point at which delivery and risk transfer | Fixes where transit ends for contractual purposes, which is not always your dock | B |
| SAP `VDATU` / `EDATU`; JD Edwards `SDDRQJ` / `SDOPDJ` / `SDPPDJ` | Requested date; confirmed date; original promised date | JD Edwards keeps the original promise in a separate field specifically so on-time can be measured against it | D |

The last row is the practical heart of section 8. A separate "original promised date" field is
the strongest available evidence that the reschedule problem is old. Where it is not populated,
on-time performance is measured against a date that moves, and no analysis recovers the original.

### 8.3 What is convention rather than requirement

- **"Lead time equals production plus transit."** A convention. It omits order processing,
  transmission, and put-away, which together are commonly 5–15% of the total.
- **A single lead-time number per item.** A convention inherited from paper planning cards,
  where a distribution could not be written down.
- **The 30-day month, the 4-week period, the ISO week.** Conventions, as Topic 04 established.
- **SCOR's five processes and its metric hierarchy.** A reference framework, not a regulation.
  Useful for vocabulary; not binding on any company.
- **The choice of request date over promise date.** A commercial and contractual choice.
  TL 9000 fixes it for one industry; nothing fixes it in general.

An analyst who can name which of these is a requirement can challenge the rest. An analyst who
cannot will treat a 2019 item master entry as a physical constant.

## 9. Industry practice and case evidence

### 9.1 What receipts show about the shape

Three empirical studies have looked at large samples of actual lead times, and they agree on
the shape.

| Study | Sample | Finding on shape | Grade |
|---|---|---|---|
| Nielsen, Michna, Sørensen and Do (2017) | 98,219 deliveries across 50 products | Lead times highly right-skewed (skewness above 1) and highly kurtotic; median coefficient of variation 1.18 to 1.41; mostly independent across orders | A |
| Gudum (2002), Copenhagen Business School | About 90,000 orders at a Danish wholesaler, plus a simulation | Delays are strongly right-skewed with long tails; beta, Weibull, and gamma fit best; normal and uniform fit worst, with normal squared error around seven times the best fit | B |
| Tate (2023), University of Arkansas | About 2.87 million administrative and production lead times, DLA data | Lognormal best overall; gamma better for short production lead times; exponential for the outlier component | B |

Two readings matter here.

**A coefficient of variation near 1 is close to exponential.** At CV = 1 the gamma shape
parameter is 1 and the distribution is exponential: the mean and the standard deviation are
equal, the median is 69% of the mean, and the 95th percentile is three times the mean. That is
the shape of a lead time dominated by delay events rather than by a stable process. The worked
example's CV of 0.24 is mild by comparison, and it is a single supplier stage rather than a
total replenishment lead time. Total lead times, which compound several stages each with its
own tail, are more skewed than any one stage.

**The normal is the worst of the common choices, not the most convenient.** All three studies
report it as the poorest fit. It is nonetheless the default in every planning system that
carries a mean and a standard deviation, because two parameters are what the software accepts.
The disagreement worth naming is on independence: Gudum reports autocorrelation in successive
deliveries, Nielsen and colleagues report mostly independent observations. If consecutive lead
times are positively autocorrelated, a stock-out this cycle predicts a higher probability next
cycle, and the independence assumption behind the safety-stock formula understates the buffer.

### 9.2 What the literature says about the variance term

| Claim | Evidence | Grade | Caveat |
|---|---|---|---|
| Lead-time variance can dominate demand variance in lead-time demand | Nielsen et al. (2017): lead-time variance contributed about 35% more to lead-time demand variance than demand variance did | A | One wholesaler; the balance depends on the item's demand CV |
| Omitting autocorrelation understates safety stock | Prak, Teunter and Syntetos (2017): traditional normal-based safety stocks up to 30% too low; achieved service up to 10 points below target, 1,000,000 replications | A | Requires positively autocorrelated forecast errors over the lead time |
| The normal approximation can reverse the sign of the lead-time variability effect | Chopra, Reinhardt and Dada (2004): for cycle service levels roughly 50–63%, reducing lead-time variability increases required safety stock under gamma lead times | A | Narrow CSL band; effect shrinks as demand CV falls |
| The normal approximation is unwarranted in general | Eppen and Martin (1988): stock-out probabilities can be "egregiously in error" | A | Analytic; the size of the error is case-specific |
| The normal approximation is adequate for fast-moving items | Tyworth and O'Neill (1997): robust for demand CV below about 40% across seven industry groups, though contingency stock errors can exceed 25% | A | Cost impact small because safety stock is a minor share of logistics cost |
| Cost penalties from the normal approximation can be serious even at low CV | Lau and Lau (2003), in a `(Q, R)` system | A | Contradicts the prevailing "normal is fine" view |

The apparent contradiction between the Chopra and the Tyworth results is worth understanding
rather than resolving. Tyworth asks whether the *total cost* is close; Chopra asks whether the
*service level* is close. For a fast-moving item with cheap holding cost, a 10% inventory
error is a small cost error. For a service-level target, the same error is a contractual
failure. The question you are answering determines which result applies.

> **Note:** These studies disagree in the range of cycle service levels where most companies
> actually operate. Firms targeting 97–99% fill rates commonly run at cycle service levels of
> 50–70%, because a large lot size relative to lead-time demand converts a modest CSL into a
> high fill rate. That is precisely the band in which Chopra and colleagues find the normal
> approximation directionally wrong. Check which service metric your target names before
> applying either result.

### 9.3 Planned lead times and the self-fulfilling bias

A separate literature concerns the lead time *inside* the planning system, which is the
planned offset rather than the observed duration. Its finding is uncomfortable.

- **St. John (1985)** simulated an MRP system and found total manufacturing cost 41% higher at
  the maximum lead-time allowance than at the minimum, with cost rising roughly exponentially
  once queue time exceeds 80–90% of the allowance. The mechanism is that a longer allowance
  permits more queueing, which lengthens the actual time, which appears to justify the
  allowance.
- **Marlin (1986)** examined planned lead-time accuracy and found that both too-long and
  too-short planned lead times raise available inventory, and that a historical average
  outperformed work-content-based methods when work content varied between orders.
- **Karmarkar's work on job shops** names the behavioural mechanism: assembly departments
  inflate lead-time offsets to guarantee part availability, so the offsets ratchet upward and
  the actual times follow them. The consequences listed are higher work in process, early
  arrivals that masquerade as safety stock, larger finished-goods buffers, distorted priority
  signals, and delayed detection of quality problems.

This is section 1, case one, seen from the production side. A lead time that is a policy is not
automatically wrong, but it is a decision that must be revisited against evidence. Once a
padded figure is in the master data, the organisation loses the ability to tell whether the
padding is still needed, because the only measurements it takes are consistent with it.

### 9.4 Requested versus promised in practice

The choice of date basis is not academic. McKinsey and the TPA surveyed 24 major North
American retailers and consumer goods manufacturers in 2019 and found no industry-standard
OTIF definition. On-time was variously the retailer's requested date, the manufacturer's
committed date, or a delivery appointment slot. Sixty-seven percent of respondents preferred
the requested delivery date, on the reasoning that the original request is what drives
on-shelf availability, and that measuring against a self-set promise date lets a supplier
flatter its own numbers (Grade C).

Three conventions coexist:

| Convention | Date used | What it measures | Failure mode |
|---|---|---|---|
| On-time to request | Customer's requested date at order placement | Whether the customer's need was met | Punishes the supplier for a date it never agreed to |
| On-time to (original) promise | First confirmed date | Whether the supplier kept its word | Measures the promise, not the need |
| On-time to current promise | Most recent promised date | Nothing stable | Every reschedule improves the score |

The third is not a measurement. It is a measurement of the rescheduling process. It appears in
dashboards because it is the only date most systems retain, not because anyone chose it.

The practical resolution used by mature programmes is to report both request-date and
original-promise performance, and to report the gap between them as a separate number. The gap
measures how much the supplier padded at confirmation, which is a leading indicator of
capacity stress.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Nominal lead time used as the actual | Service misses target while every parameter "looks right" | Contract or catalogue figure carries no variance and may be a policy | Reorder point set below mean lead-time demand; chronic stock-outs | Compare nominal with the central 50% of the empirical distribution |
| Lead-time variance omitted | Safety stock far below the level observed deliveries imply | Planning system accepts a mean only | Service 20–40 points below target in high-variance lanes | Compute `sigma_LTD` outside the system; load the result |
| Measurement windows differ across records | Lead time shows a step change with no process change | Early records measured from despatch, later from confirmation | A trend that is entirely a data-model artefact | Fix the endpoint pair per item; store it as metadata |
| Order processing excluded from the total | Supplier blamed for lateness the buyer caused | Buyer's approval queue is invisible in supplier data | Wrong remediation; supplier relationship damage | Measure all five stages from the buyer's own timestamps |
| Working days and calendar days mixed | Lead times are suspiciously short in December | 5/7 conversion applied as a constant | Understated by up to 40%; worse around holidays | Convert with a real working-day calendar |
| Normal fit used for the upper tail | p95 buffer short by a day or more every cycle | Right skew puts mass below the physical floor | Systematic under-protection where it matters | Fit lognormal or gamma; use empirical quantiles |
| Rescheduled orders counted as on-time | On-time improves while customers complain | Current promised date overwrites the original | Metric decoupled from physical reality | Retain original promised date; report both bases |
| Lead time measured only on on-time orders | Variance implausibly small | Late receipts excluded as "exceptions" | Tail truncated; buffer sized on the body | Include all receipts; label exclusions |
| Partial receipts dropped | Lead time understated for multi-drop orders | First receipt taken as completion | Missing days at the end of the chain | Define completion as full quantity available |
| Cancelled and re-issued orders double counted | Observation count exceeds order count | Rescheduling creates a new order record | Inflated sample, distorted mean | Deduplicate on the commercial order line |
| Lead time pooled across items and suppliers | Distribution is bimodal but nobody looks | One average per category | Mean describes neither mode; buffer wrong for both | Segment by lane and supplier; test for bimodality |
| Drift not monitored | Distribution estimated once, used for years | No review cadence | Buffer calibrated to a process that no longer exists | Quarterly review; control chart on lead time |

**The window mismatch is the hardest to detect.** It produces no error message, only a plausible
series with a step in it, and the step is usually attributed to the supplier, because the
supplier is the party the series is supposed to describe. The test is to plot lead time by
order date alongside the fraction of records whose window includes order processing. If the
two move together, the step is a data-model change.

**The reschedule failure is the most corrosive.** It does not merely bias a number; it teaches
the organisation that the number is not to be trusted, which is worse than not having it. Once
planners know that on-time performance is computed against a moving date, they stop using it
to prioritise, and the metric loses its only operational function.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The distribution is stationary over the estimation window | A sample can only describe one process | Supplier change, plant move, route change, freight market shift | Plot lead time by order date; test for a break |
| Demand and lead time are independent | The variance formula omits the covariance term | Allocation, rationing, or congestion: high demand lengthens lead time | Correlate order quantity with lead time by period |
| Successive lead times are independent | Safety stock per cycle is computed from one distribution | Capacity effects and backlog carry-over | Autocorrelation function of the lead-time series |
| Observations are a random sample of completed orders | The mean and variance estimate the process | Late orders excluded as exceptions; partial receipts dropped | Count receipts against orders shipped; reconcile |
| Stored timestamps are event times, not record times | The window must measure the physical process | End-of-shift posting, batch interfaces, backfill | Compare receipt timestamp with gate-in timestamp on a sample |
| The five stages are exhaustive and non-overlapping | The total is their sum | Re-work loops, split shipments, storage before despatch | Reconcile the sum against end-to-end elapsed time |
| The calendar conversion is constant | Working-day conversion has to be a function | Holiday periods, shift patterns, regional calendars | Compare conversion factors across months |
| The service target maps to a known quantile | The z-value comes from the chosen metric | Confusing CSL with fill rate; non-normal lead-time demand | State the metric; verify the quantile against the fit |
| The nominal is a policy or a capability, and you know which | The two require different treatments | Contract renegotiation, supplier capacity change | Compare the 50th percentile with the contract |

Two limits of the whole framing deserve stating.

**A lead-time distribution is a description, not a forecast.** It tells you what the process
did under the conditions in the sample. It does not tell you what it will do in a port strike.
The distribution is the right input to a buffer calculation and the wrong input to a disruption
plan; Module 14A treats the second problem.

**The distribution is estimated from survivors.** Every method here uses receipts. Orders that
were cancelled, that never arrived, or that are still open contribute nothing. In a period of
severe shortage the observed distribution is truncated on the right precisely when the true
distribution has shifted right. That is the opposite of what the buffer calculation assumes,
and it is the mechanism behind the most expensive class of stock-out.

## 12. Exercises

### Exercise 1 — Decompose a lead time (recall and application, 30 minutes)

Take any purchase order you can inspect, or construct one, and identify the five stage
boundaries from the documents available.

**Deliverable:** a six-row table of events with timestamps and source system, plus the five
durations in calendar days.
**Acceptance criteria:** every stage has both endpoints; estimated endpoints are marked with
the estimation basis; the sum reconciles against end-to-end elapsed time, with any gap
explained.
**Time:** 30 minutes.

### Exercise 2 — Descriptive statistics (calculation, 35 minutes)

Twenty receipts, in calendar days, requisition to put-away:

```text
12 13 14 14 15 15 16 16 17 18 18 19 20 21 23 25 28 32 38 47
```

The nominal lead time is 15 days.

**Deliverable:** mean, median, standard deviation, coefficient of variation, and 95th
percentile, each with its formula or method, plus one sentence on what the nominal would lead
a planner to believe.
**Acceptance criteria:** the percentile method is named; the denominator for the standard
deviation is stated; the comparison with the nominal is quantified in days and in percent.
**Time:** 35 minutes.

### Exercise 3 — Three policies (calculation, 40 minutes)

Using the twenty receipts from Exercise 2: daily demand 50 units, daily demand standard
deviation 10 units, target cycle service level 95% (`z = 1.645`).

**Deliverable:** safety stock and reorder point under (a) the nominal with zero variance,
(b) the mean actual with zero variance, and (c) the mean and standard deviation; the ratio
between (c) and (a).
**Acceptance criteria:** the formula is written before it is evaluated; the lead-time share of
`sigma_LTD` is computed; the arithmetic is reproducible from the stated numbers.
**Time:** 40 minutes.

### Exercise 4 — The two-day question (calculation, 25 minutes)

An item has a 30-day mean lead time, daily demand of 30 units with a standard deviation of 5,
and a lead-time standard deviation of 2 days. Target CSL is 95%.

**Deliverable:** safety stock with and without the lead-time variance term, the ratio between
them, and the lead-time standard deviation at which the two terms under the square root are
equal.
**Acceptance criteria:** the terms are compared in units, not days; the crossover is derived
algebraically; one sentence states where the supply chain team should spend its effort.
**Time:** 25 minutes.

### Exercise 5 — Timestamp pair audit (audit, 40 minutes)

Take a lead-time field from any system you can inspect. Determine which two events it spans,
whether both are event times, and whether the basis is calendar or working days.

**Deliverable:** a one-page finding with the evidence for each determination and a list of
queries that would silently produce wrong answers.
**Acceptance criteria:** the window is established from data rather than from the field name;
at least one query is shown to be wrong under the finding; historical repair is separated from
going-forward policy.
**Time:** 40 minutes.

### Exercise 6 — Request versus promise (application, 45 minutes)

| Order | Requested | Promised at confirmation | Actual | Current promise |
|---|---:|---:|---:|---:|
| 1 | 20 | 22 | 21 | 22 |
| 2 | 25 | 25 | 25 | 25 |
| 3 | 18 | 20 | 20 | 20 |
| 4 | 30 | 30 | 33 | 34 |
| 5 | 15 | 18 | 17 | 18 |
| 6 | 22 | 22 | 22 | 22 |
| 7 | 28 | 28 | 35 | 36 |
| 8 | 12 | 15 | 15 | 15 |
| 9 | 24 | 24 | 23 | 24 |
| 10 | 19 | 21 | 27 | 28 |

**Deliverable:** on-time performance on all three bases, plus the two gaps between them, each
attributed to a named mechanism.
**Acceptance criteria:** the date basis is stated for each calculation; all three figures come
from the same rows; padding at confirmation is distinguished from laundering at reschedule.
**Time:** 45 minutes.

### Exercise 7 — Policy or capability (design, 40 minutes)

A supplier's contract specifies 21 days. Its last twelve deliveries averaged 14 days with a
standard deviation of 2. Design the analysis that would tell you whether the 21 is a policy
buffer or a capability limit.

**Deliverable:** a one-page design naming the data required, three hypotheses, and the test
that would distinguish them.
**Acceptance criteria:** at least one hypothesis is deliberate pacing to the contract; the
design distinguishes allocation-constrained from capacity-constrained supply; each outcome
names the action that follows.
**Time:** 40 minutes.

### Exercise 8 — Write the specification (design, 35 minutes)

Write the lead-time specification for one item in a dataset you know.

**Deliverable:** a half-page specification covering start event, end event, unit, calendar
basis, estimation window, observation count, distribution family, and review cadence, each
with the test that would falsify it.
**Acceptance criteria:** both endpoints name a system and a field; the calendar basis is
explicit; the window has a justification rather than a duration; every test is executable
against the data.
**Time:** 35 minutes.

## 13. Self-check answers

**Exercise 2 — descriptive statistics.**

```text
   n = 20, sum = 421, sum(x^2) = 10,421
   mean       = 421 / 20                       = 21.05 d
   SS_dev     = 10,421 - 20 x 21.05^2         = 10,421 - 8,862.05 = 1,558.95
   variance   = 1,558.95 / 19                 = 82.05
   sd         = sqrt(82.05)                   = 9.06 d
   CV         = 9.06 / 21.05                  = 0.43
   median     = (17 + 18) / 2                 = 17.5 d
   p95        = ceil(0.95 x 20) = 19th value  = 38 d
```

The nominal of 15 days sits below all but two observations and 5.4 days below the median. A
planner using it would believe the item can be replenished in about the time of its best-ever
delivery, which is not a plan but a hope. The mean overstates the typical delivery by 3.6 days
and the median understates it by 9.1; the mean is the right centre for a reorder point and the
median is the right description of a typical delivery.

**Exercise 3 — three policies.**

```text
   (a) nominal 15 d, zero variance
       SS  = 1.645 x 10 x sqrt(15)                    = 63.7  -> 64 units
       ROP = 15 x 50 + 64                            = 814 units
   (b) mean 21.05 d, zero variance
       SS  = 1.645 x 10 x sqrt(21.05)                = 75.5  -> 75 units
       ROP = 21.05 x 50 + 75                        = 1,128 units
   (c) mean 21.05 d, sd 9.06 d
       SS  = 1.645 x sqrt(21.05 x 100 + 2,500 x 82.05)
           = 1.645 x sqrt(2,105 + 205,125)
           = 1.645 x 455.2                          = 748.8 -> 749 units
       ROP = 21.05 x 50 + 749                       = 1,802 units

   ratio (c)/(a)               = 749 / 64          = 11.7
   lead-time share of variance = 205,125 / 207,230 = 99.0%
```

A buffer of 749 units against lead-time demand of 1,053 units is plausible for a lead-time
coefficient of variation of 0.43 at a 95% target, and it is also a signal. At this level of
lead-time variability the item's service level is governed by supplier reliability, not by
forecast quality, and the correct investment is in the supplier rather than in the buffer.

**Exercise 4 — the two-day question.**

```text
   demand term        = sqrt(30) x 5             = 27.39 units
   lead-time term     = 30 x 2                   = 60.00 units

   SS deterministic   = 1.645 x 27.39             = 45.1 -> 45 units
   SS stochastic      = 1.645 x sqrt(30 x 25 + 900 x 4)
                      = 1.645 x sqrt(750 + 3,600)
                      = 1.645 x 65.95             = 108.5 -> 108 units

   ratio              = 108 / 45                 = 2.4

   crossover: mu_d x sigma_LT = sqrt(L) x sigma_d
              sigma_LT* = (sigma_d / mu_d) x sqrt(L)
                        = (5 / 30) x 5.477       = 0.913 d
```

A lead-time standard deviation of under one day doubles this item's safety stock. A team
choosing between a forecasting project and a supplier-reliability project should compute this
crossover first, because it determines which project moves inventory.

**Exercise 6 — request versus promise.**

| Basis | Hits | Performance |
|---|---:|---:|
| On-time to request | 2, 6, 9 | 3/10 = 30% |
| On-time to original promise | 1, 2, 3, 5, 6, 8, 9 | 7/10 = 70% |
| On-time to current promise | all ten | 10/10 = 100% |

The request-to-promise gap is 40 points. It measures how much the supplier padded each
commitment at acknowledgement: orders 1, 3, 5, and 8 were promised two to three days later than
requested, and three of the four then delivered inside the padded window. That is not
dishonesty; it is a supplier pricing its own uncertainty into the promise. It is also invisible
in any report that uses the promise date.

The promise-to-current gap is 30 points and is a different mechanism. Orders 4, 7, and 10 were
rescheduled after confirmation and all three then met the revised date. Under the TL 9000 rule,
a supplier-initiated change that the customer accepts still counts as late. Under a naive
current-promise calculation all three become hits, and the metric has been converted from a
measure of delivery into a measure of administrative agility.

**Exercise 1 rubric.** A strong answer names the system and field for each endpoint, marks
inferred endpoints, and reconciles the stage sum against end-to-end elapsed time, treating a
gap above one day as a finding. A weak answer produces six dates and sums them.

**Exercise 5 rubric.** A strong audit establishes the window by test: does the field equal the
difference between two candidate timestamps; does it move when the record is edited; does it
shift across a month boundary consistent with working days. A weak audit reports the field name
and its data type.

**Exercise 7 rubric.** A strong design names three hypotheses: pacing to the contract,
allocation-constrained, and capacity-constrained. Pacing predicts a distribution truncated just
below 21 with little mass below 14; capability predicts one centred on 14 with a right tail. A
weak design proposes to ask the supplier and stops.

**Exercise 8 rubric.** A strong specification reads like a contract: "start
`requisition.created_at_utc` in ERP; end `putaway.confirmed_at_utc` in WMS; unit calendar days;
window 2025-01-01 to 2026-03-31, 45 observations; family gamma, shape 17.2, scale 1.99; review
quarterly." It states what would falsify each element. A weak specification states a number
and a unit.

## 14. Interview questions

**1. "What is lead time?"**

A strong answer refuses the question as posed and asks for the two events. It then gives the
five-stage decomposition and notes that the sum has no owner. A weak answer says "the time
between ordering and receiving", which is correct for one of the five definitions and silently
excludes the other four.

**2. "What is the difference between nominal, actual, and promised lead time?"**

A strong answer defines all three, says which belongs in a safety-stock calculation (actual, as
a distribution), and explains why the promise is endogenous. A weak answer treats them as
synonyms.

**3. "How would you measure supplier lead time?"**

A strong answer names both endpoints, the system holding each, the calendar basis, and the
exclusion policy. It notes that measuring from confirmation excludes the buyer's own queue and
that this is a stated choice. A weak answer says "from PO to receipt", which spans a different
window from every supplier report the company receives.

**4. "Why is the average lead time not enough?"**

A strong answer gives the right skew, the physical floor, the long tail, the mean-median gap,
and the consequence: the upper quantile that sets the buffer is not recoverable from the mean.
A weak answer says "there is variance" without saying what it does to a decision.

**5. "Write the safety stock formula with stochastic lead time."**

A strong answer writes `z x sqrt(L x sigma_d^2 + mu_d^2 x sigma_LT^2)`, states the independence
assumption, and notes that the second term is in units squared. A weak answer writes
`z x sigma_d x sqrt(L + sigma_LT)`, which is dimensionally plausible and wrong.

**6. "A two-day standard deviation on a thirty-day lead time. Is that small?"**

A strong answer converts both terms to units, compares them, and computes the crossover
`CV_d x sqrt(L)`. It concludes that two days is small only if daily demand variability is large
relative to mean demand, which it usually is not. A weak answer says 2/30 is 6.7% and therefore
small.

**7. "Why does the normal distribution fail for lead times?"**

A strong answer gives the floor, the skew, and the tail, and shows it with a number: the normal
assigns several percent of its mass below the fastest delivery ever observed. It may add that
for fast-moving items the cost error is modest while the service error is not. A weak answer
says "lead times are not normal" without a mechanism.

**8. "How would you compute on-time delivery?"**

A strong answer names the date basis, grain, arrival-versus-ship basis, tolerance, and
exclusion policy, and says the customer's scorecard binds when one exists. It proposes
reporting both request-date and original-promise performance. A weak answer says "actual minus
promised, less than or equal to zero".

**9. "Why keep the original promised date?"**

A strong answer explains that the current promise is overwritten on reschedule, so a metric
built on it measures the rescheduling process, and that the original is the only date against
which the supplier's word can be tested. A weak answer says "for audit purposes".

**10. "Policy lead time versus capability lead time?"**

A strong answer defines capability as what the process can reliably do and policy as what has
been agreed, notes that a policy longer than capability can be rational, and says the error is
recording a policy in a field the planning engine reads as a capability. It proposes the test:
compare the 50th percentile of actuals with the contract. A weak answer treats the contract
figure as the truth.

**11. "How would you detect that a lead-time field is wrong?"**

A strong answer proposes four tests: compare the field against the difference between two
candidate timestamps; plot lead time by order date and look for a step; check whether the value
moves when a record is edited; compare working-day and calendar-day conversions across a
holiday. A weak answer says "check for outliers".

## 15. Key terms

- **Lead time** — elapsed time between a named start event and a named end event.
- **Nominal lead time** — the single figure in a contract, catalogue, or item master.
- **Actual lead time** — the observed duration on completed orders, between two named
  timestamps.
- **Promised lead time** — the duration implied by a committed delivery date.
- **Planned lead time** — the offset the planning engine uses to schedule supply.
- **Capability lead time** — what the process can reliably deliver, unconstrained by policy.
- **Order processing time** — requisition raised to PO transmitted.
- **Transmission time** — PO transmitted to supplier confirmation.
- **Supplier lead time** — confirmation to goods ready or shipped.
- **Transit time** — carrier pickup to arrival at destination.
- **Receiving and put-away time** — arrival to stock available to promise.
- **Replenishment lead time** — order release to availability, across all five stages.
- **Cumulative lead time** — the longest path from raw material to finished good.
- **Order-to-delivery cycle time** — customer order receipt to customer acceptance.
- **Dwell time** — waiting time the supplier does not control; excluded from process time.
- **Lead-time demand** — demand expected during the replenishment lead time; units.
- **Lead-time variability** — the standard deviation of the lead-time distribution; days.
- **`sigma_LTD`** — standard deviation of demand during the lead time; units.
- **Lead-time bias** — a signed, systematic difference between the lead time used and the one
  experienced.
- **Safety lead time** — a time buffer added to a planned offset.
- **Safety stock** — a quantity buffer against variability; distinct from safety lead time.
- **Reorder point** — mean lead-time demand plus safety stock; units.
- **Service factor (z)** — the standard normal quantile for a stated service metric.
- **Cycle service level (CSL)** — probability of no stock-out in a replenishment cycle.
- **Fill rate** — fraction of demand units served from stock; not the same as CSL.
- **On-time to request** — delivery by the customer's original requested date.
- **On-time to promise** — delivery by the supplier's first confirmed date.
- **Current promised date** — the most recent committed date; a rescheduling artefact.
- **Must-arrive-by date (MABD)** — a retailer's latest acceptable arrival date.
- **Lead-time compression** — a structural reduction in a stage duration, not an expedite.

## 16. Sources

**Peer-reviewed and academic**

- Chopra, S., Reinhardt, G., and Dada, M. (2004). "The Effect of Lead Time Uncertainty on
  Safety Stocks." *Decision Sciences*, 35(1). Grade A.
- Eppen, G. D., and Martin, R. K. (1988). "Determining Safety Stock in the Presence of
  Stochastic Lead Time and Demand." *Management Science*, 34(11). DOI:
  10.1287/mnsc.34.11.1380. Grade A.
- St. John, R. (1985). "The cost of inflated planned lead times in MRP systems." *Journal of
  Operations Management*. DOI: 10.1016/0272-6963(85)90002-6. Grade A.
- Marlin, P. G. (1986). "Manufacturing lead time accuracy." *Journal of Operations
  Management*. DOI: 10.1016/0272-6963(86)90024-0. Grade A.
- Nielsen, P., Michna, Z., Sørensen, B. B., and Do, N. A. D. (2017). "Lead Times — Their
  Behavior and the Impact on Planning and Control in Supply Chains." *Management and
  Production Engineering Review*. Grade A.
- Prak, D., Teunter, R., and Syntetos, A. (2017). "On the calculation of safety stocks when
  demand is forecasted." *European Journal of Operational Research*. DOI:
  10.1016/j.ejor.2016.06.035. Grade A.
- Tyworth, J. E., and O'Neill, L. (1997). "Robustness of the normal approximation of lead-time
  demand in a distribution setting." *Naval Research Logistics*, 44(2). Grade A.
- Lau, A. H.-L., and Lau, H.-S. (2003). "Nonrobustness of the normal approximation of
  lead-time demand in a (Q, R) system." *Naval Research Logistics*. DOI: 10.1002/nav.10053.
  Grade A.
- Tadikamalla, P. R. (1984). "A comparison of several approximations to the lead time demand
  distribution." *Omega*, 12(6), 575–581. Grade A.
- Ioannou, G., and Dimitriou, S. (2012). "Lead time estimation in MRP/ERP for make-to-order
  manufacturing systems." *International Journal of Production Economics*. Grade A.

**Working papers and theses**

- Gudum, C. K. (2002). "On the distribution of lead time delays in supply chains." PhD thesis,
  Copenhagen Business School. Grade B.
- Karmarkar, U. S. "Controlling WIP and Leadtimes in Job Shops." Working paper, William E.
  Simon Graduate School of Business, University of Rochester. Grade B.
- Tate, W. (2023). "Lead Time Distribution Modeling for Supply Chains with a Large Number of
  Items." Honors thesis, University of Arkansas. Grade B.

**Standards and official**

- ASCM, SCOR Digital Standard, RS.1.1 Customer Order Fulfilment Cycle Time. Grade C — a
  voluntary reference framework, not a regulation.
- TL 9000, QuEST Forum, "Q&A — On-time Delivery — OTD." Request-date basis; early delivery
  counts as a miss; supplier-initiated changes remain late. Grade B.
- ANSI X12 855 and 856; EDIFACT ORDRSP and DESADV. Grade B.
- GS1 EPCIS 2.0. Grade B.
- Incoterms 2020, International Chamber of Commerce. Grade B.

**Industry survey and benchmark**

- McKinsey and Company with TPA, "Defining 'on-time, in-full' in the consumer sector," 2019.
  Twenty-four retailers and manufacturers; 67% prefer the requested date. Grade C.
- APQC, "What Is Customer Order Cycle Time?" Median 12.0 days, n = 12,074. Grade C.

**Vendor documentation**

- Summit Analytics, "OTIF: Definition & Switches." SAP and JD Edwards date-field semantics.
  Grade D.
- IFS Cloud documentation, "Lead Time." Stage decomposition across separate calendars.
  Grade D.
- NetSuite, "Lead Time Defined." Practitioner component breakdown. Grade D.

**Note on citations and grades.** The 41% cost figure from St. John (1985) comes from a
simulation whose parameters are not reproduced in the abstract, and the 35% variance
contribution from Nielsen et al. (2017) is specific to one wholesaler's product set. Neither is
quoted here as a base rate. Vendor sources are cited for field semantics, not effectiveness.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_calendars-weeks-time-zones.md`](04_calendars-weeks-time-zones.md)
- Next topic: [`06_currency-and-fx.md`](06_currency-and-fx.md)
- Related: [`11_demand-lead-time-distributions.md`](11_demand-lead-time-distributions.md)
- Related: [`../01_supply_chain_foundations/05_inventory-and-service.md`](../01_supply_chain_foundations/05_inventory-and-service.md)
- Survey file: [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
- Return to the part map: [`../_index.md`](../_index.md)

