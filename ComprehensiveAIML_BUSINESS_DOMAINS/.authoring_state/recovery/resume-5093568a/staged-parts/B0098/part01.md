---
title: "Fulfilment Promise Metrics"
topic: "Omnichannel fulfilment promise accuracy, fill, split fan-out, and recovery economics"
collection: "retail-02-omnichannel-loyalty-fulfilment"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
file_id: "03"
version: "2.0"
source_status: "OPEN"
owner_role: "retail analytics practitioner"
prerequisite_sheets: ["01_Omnichannel_Event_Taxonomy", "02_Customer_Identity_Controls"]
---

# Fulfilment Promise Metrics

Every number in this sheet is either **derived** (the algebra is shown and can be re-checked
independently) or **ILLUSTRATIVE** (a constructed dataset used to exercise the arithmetic). No
figure here is a measurement of a named retailer, a claim about a live deployment, or a benchmark.
The frozen dataset used throughout is called **D-1000** and is defined completely in Section 1.6, so
that a reader who wants to re-run any calculation has every input in hand. Where a quantity depends
on a distribution we do not have, the text says so and marks the curve as constructed.

## 1. File Header

### 1.1 Purpose and the decision this sheet serves

A fulfilment promise is the date or time window a customer is shown at checkout, plus the implicit
promise that the order arrives complete. Every metric on this sheet exists to answer one of four
operational questions: did we keep the promise we actually made, to whom did we fail to keep it,
what did the failure cost, and which lever reduces failures at acceptable cost. A metric that cannot
be attached to one of those four questions is decoration and is excluded.

The sheet is written for the analyst or product owner who has to defend a number in a trading
review. That defence has three parts: the numerator and denominator must refer to the same grain and
the same frozen promise version; the observation window must be mature enough for the outcome to be
complete; and the comparator must be the policy that would have applied without the change. Failing
any one of the three makes the number unquotable regardless of how plausible it looks.

### 1.2 Scope boundary

In scope: delivery promise accuracy for parcels, pickup readiness for buy-online-pickup-in-store
(BOPIS), line and unit fill, split-shipment fan-out, cancellation timing, promise-buffer trade-offs,
capacity gating, and the cost arithmetic that turns a late line into a money figure.

Out of scope, deliberately: warehouse labour scheduling models, carrier contract negotiation,
route optimisation, returns disposition, and legal interpretation of consumer delivery rules. Those
interact with promises but each needs its own treatment and its own evidence basis. A reader looking
for a shipping-rate model or a demand forecast will not find it here.

Also out of scope: any claim that a particular promise length is optimal. The frontier in Section 6
is constructed from a stated transit assumption; a real retailer would estimate it from its own
carrier data, and the shape would differ.

### 1.3 Prerequisites

The reader should be comfortable with ratios and their denominators, with the difference between an
entity and an event (an order-line is an entity, a parcel scan is an event), with timestamp
semantics including timezone and cut-off, and with the idea of a cohort maturing over time. No
probability theory beyond the product rule and the union bound is assumed; both are derived where
used in Sections 4 and 10.

### 1.4 Conventions used in this file

All display mathematics is inside fenced `text` blocks and uses a single plain-text convention:
`FORMULA [ID]`, `WHERE`, `CONSTRAINT`, `NUMERICAL EXAMPLE`, `INTERPRETATION`. Symbols are defined at
first use with meaning, unit, and valid range. Rates are reported as percentages to two decimals
when the denominator exceeds one thousand and to one decimal otherwise, because a denominator of
ninety-four does not support two decimal places of precision.

Dates are event dates in the market's local calendar. Money is US dollars. Durations are in days
unless stated as hours or minutes. A "day" of promise lead time always means calendar days from the
order cut-off, never business days, because consumer promise pages are written in calendar days and
mixing the two creates a one-to-two day metric error that looks like a fulfilment regression.

### 1.5 Evidence ledger

| Claim class | Basis | How to re-check |
|---|---|---|
| Funnel counts in D-1000 | constructed | additive checks in 1.6 |
| Every rate in Sections 4, 5 | derived arithmetic | recompute from 1.6 inputs |
| Split fan-out penalty | derived under independence | 4.2.4 and 5.4 |
| Promise frontier in Section 6 | constructed curve | stated generating assumption |
| Cost per late line | constructed unit costs | 5.6 break-even algebra |
| Power and MDE figures | derived from binomial variance | 10.5 |
| Carrier and store behaviour | not measured | no claim made |

The ledger matters because a promise metric travels further than the analysis that produced it. Six
months after publication, a slide quoting "84% OTIF" will be read as a performance claim unless the
sheet that produced it states which parts were invented. Everything labelled ILLUSTRATIVE here is
safe to copy for teaching and unsafe to copy into a board pack as a result.

### 1.6 The frozen dataset D-1000

D-1000 is one market, 13 consecutive ISO weeks ending 2026-09-27, one apparel-and-home retailer with
a central distribution centre (DC) and 42 stores. The unit of record is the **order-line-promise**:
one product on one order, with one promise version as shown at checkout. An order with three
products produces three records; this is deliberate, because fill and promise accuracy are line
properties and order-level aggregation destroys the ability to attribute a failure.

Population construction, with every step additive:

```text
ILLUSTRATIVE dataset D-1000, window 2026-06-29 .. 2026-09-27 (13 weeks, 91 days)

A. checkout promises issued .................. 1,000 order-line records
B. cancelled before warehouse release ........    31  (customer-initiated)
C. cancelled after release, never shipped ....    17  (picked short / no stock)
D. entered fulfilment  F = A - B - C .........   952
     D.1 fulfilled complete ...................   902
     D.2 fulfilled partial ....................    34
     D.3 failed after pick, written off .......    16
   check: 902 + 34 + 16 = 952
E. orders involved ............................   812
F. units ordered .............................. 1,240
     F.1 units shipped on D.1 lines ........... 1,100
     F.2 units ordered on D.2 lines ...........    96
     F.3 units shipped on D.2 lines ...........    22
     F.4 units ordered on D.3 lines ...........    44
     F.5 units shipped on D.3 lines ...........     0
   check: 1,100 + 96 + 44 = 1,240
```

Channel assignment, which matters for every later comparison because the channels have different
promise mechanics:

```text
G. lines by originating node
     G.1 distribution centre (parcel) .........   362
     G.2 ship-from-store (parcel) .............   252
     G.3 BOPIS pickup (no parcel) .............   338
   check: 362 + 252 + 338 = 952

H. rescue lines (DC shortfall shipped from a store, additional to G.2)
   H.1 count ...................................    12
   H.2 these are parcel-bearing records created mid-flow, not checkout channel
```

The twelve rescue lines are separated out because they are the single most common accounting trap
in this domain: they are real parcels with real promises, they are not in the checkout channel mix,
and a naive channel split therefore loses twelve records and every downstream percentage drifts.

```text
I. parcel fan-out
     I.1 parcel-bearing lines = G.1 + G.2 + H.1 = 362 + 252 + 12 = 626
     I.2 BOPIS-free lines (pickup only) ......   326   (= 952 - 626)
     I.3 lines shipped as a single parcel ....   532   (626 - 94)
     I.4 lines split across two parcels ......    94
     I.5 of I.4, DC-origin ...................    52
     I.6 of I.4, store-origin ................    42
   check: 532 + 94 = 626;  52 + 42 = 94
J. parcel count = 362 + 52 (DC) + 252 + 12 + 42 (store)
             = 414 + 306 = 720
   check: 532 single-parcel lines contribute 532; 94 split lines contribute 188;
          532 + 188 = 720
```

Outcomes, the numbers every later formula consumes:

```text
K. late parcels ..................  81   (DC 29 of 414; store 52 of 306)
L. late parcel-bearing lines .....  71   (52 single-parcel, 9 one-parcel-only, 10 both)
M. BOPIS not ready by promise ....  20   of 338
N. late lines  L + M .............  91
O. lines on time among D  ........  861  (952 - 91), 90.44%
P. D.1 lines on time ............. 840
Q. D.2 lines on time .............  21
R. OTIF (complete and on time) ... 840 of 1,000 = 84.00%
S. re-promised lines (promise moved after checkout) .. 28
T. inventory record accuracy ..... 96.2%
```

The two promise-adjacent facts that a reader usually asks for next: 28 lines had their promise moved
after checkout, and inventory record accuracy is 96.2%. Both are used in Sections 8 and 4.3 as
sensitivity levers, never as corrections to the headline.

### 1.7 How to read the rest of the sheet

Section 2 gives the mental model in one diagram. Section 3 fixes the vocabulary, because half of all
disputes about fulfilment numbers are definitional. Section 4 derives the formulas and is the
authoritative part; if a formula here conflicts with a dashboard elsewhere, this sheet wins only
after the dashboard's denominator is shown to differ. Section 5 works one coherent example end to
end with an independent arithmetic re-check. Sections 6 and 7 convert the algebra into pictures and
selection rules. Section 8 is the diagnostic core and is where a practitioner should start when a
number moves without explanation. Section 9 covers the cost of computing the metrics, Section 10
collects the results worth memorising, Section 11 is the interview set, and Section 12 is the card.

## 2. Concept Map / Mental Model

The whole topic compresses into one pipeline with three gates. Promises are issued, the promise is
frozen, outcomes accrue against the frozen promise, and a metric is computed only over records that
passed all three gates. The gates are not bureaucratic; each one corresponds to a class of error
that silently inflates or deflates the final number.

```text
   checkout promise issued
            |
            v
   [GATE 1: promise frozen]  ---- fail ---> promise drift pool (Section 8.1)
            |
            v
   [GATE 2: line enters fulfilment] ---- fail ---> cancelled pool (Section 3.4)
            |
            v
   [GATE 3: outcome mature]  ---- fail ---> open / censored pool (Section 8.2)
            |
            v
   outcome label (on time / late / partial / failed)
            |
            +--> line metrics  (fill, OTP, split penalty)
            +--> order metrics (OTIF, parcels per order)
            +--> money         (late-line cost, lever break-even)
```

The mental model that makes the formulas stick is **a promise is a conjunction of two independent
obligations**: the order must arrive complete, and it must arrive by the date shown. Those
obligations fail through different mechanisms at different rates, so combining them naively
(measuring only completeness, or only lateness) overstates performance by the width of the gap
between them. In D-1000 that gap is 4.24 percentage points, computed in Section 4.1.3.

The second mental model is **fan-out**. A promise is made to a line, but the physical work happens
in parcels. One line can become two parcels, and each parcel is an independent chance to be late.
Splitting a line therefore does not divide the risk, it compounds it. Section 4.2.4 derives the
compounding exactly and Section 5.4 shows it reproducing the observed penalty to within one point.

The third mental model is **the promise is a quantile, not a mean**. A promise date is chosen so
that a high fraction of deliveries land on or before it. Adding buffer along a chain does not add
quantiles; the p95 of a sum of two random transit legs is not the sum of the two p95s. Section 4.4.1
gives the counterexample, and it is the single most common arithmetic error in promise design.

```text
   concept            unit of record        failure it protects against
   ---------------    -------------------    ------------------------------------
   frozen promise     line x version         re-anchoring after the fact
   eligibility        line                   denominator inflation by cancellation
   maturity           line                   right-censoring at snapshot edge
   fan-out            parcel                 understated split penalty
   buffer             day                    percentile mis-addition
   capacity           pick per day           promise issued beyond throughput
```

Read the table by column, not by row. The left column is what you measure, the middle column is the
grain that makes the measurement legal, and the right column is the specific defect that appears
when you get the grain wrong. A practitioner who can recite the middle column from memory will catch
most bad numbers before computing them.

## 3. Foundational Definitions

### 3.1 Promise, promised date, and promise version

The **promise** is the delivery or pickup commitment shown to the customer at the moment of purchase
confirmation, including its date or window. The **promised date** is the last instant at which
delivery or readiness still satisfies the promise; a parcel delivered at 14:00 on the promised date
is on time, and one delivered at 09:00 the next morning is late by one day regardless of how the
carrier rounds.

**Promise version** is a monotonically increasing integer attached to each promise record. Version 1
is what the customer saw at checkout. Version 2 and above exist only when the promise was
subsequently moved, for example after a stock-out forced a re-source. Metrics on this sheet always
use the version that was shown at checkout, because that is the only version the customer
experienced. Using the latest version measures the retailer's ability to negotiate with its own
customers after the fact.

### 3.2 Line, order, unit, parcel

Four grains, and every metric must declare which one it uses.

| Term | Definition | Count in D-1000 |
|---|---|---|
| unit | one physical item of one SKU | 1,240 ordered |
| line | one SKU on one order, one promise | 1,000 issued, 952 fulfilled |
| order | one checkout transaction | 812 |
| parcel | one physical shipment with one tracking number | 720 |

A single order can produce several lines and several parcels, and the mapping between them is
many-to-many once splitting and rescue are allowed. This is why fill rate needs three denominators
and why quoting "fill rate" without a grain is meaningless.

### 3.3 Fill: three legitimate definitions

**Line fill rate** is complete lines divided by fulfilled lines: 902 / 952 = 94.75%. **Unit fill
rate** is shipped units divided by ordered units: 1,122 / 1,240 = 90.48%. **Order fill rate** is
orders shipped complete divided by orders: 726 / 812 = 89.41%. All three are correct and they differ
by up to 5.3 points on identical data because partial lines concentrate in multi-unit orders.

The mistake to avoid is averaging line fill rates weighted by nothing, or assuming unit fill can be
recovered from line fill. It cannot: two datasets with identical line fill can differ by ten points
in unit fill if one has partial lines on ten-unit orders and the other on two-unit orders.

### 3.4 Cancellation timing and the eligibility boundary

**Pre-release cancellation** happens before the warehouse or store begins picking. It is usually
customer-initiated and carries no promise failure, because no promise was broken; the customer
changed their mind. **Post-release cancellation** happens after picking began, typically because the
inventory record was wrong or the item was damaged. It is a promise failure in substance even though
no parcel moved.

In D-1000, 31 lines are pre-release and 17 are post-release. The post-release seventeen are inside
the fulfilled denominator of 952 as the 16 failed-plus-one-partial-failure residual; the point is
that they must appear *somewhere*, and Section 4.1.3 quantifies what happens to the headline when a
retailer moves them out of the denominator instead.

### 3.5 On time, in full, and OTIF

**On time** is a binary label per line: the last delivery or readiness event at or before the
promised date. **In full** is a binary label per line: all ordered units shipped in the first
despatch attempt, with no backorder and no partial. **OTIF** is the conjunction of the two labels,
and it is only defined once both definitions and the denominator are stated. Three retailers can
each report "OTIF" and be measuring three different things.

### 3.6 Split shipment and fan-out

A **split shipment** is a line whose units were despatched in more than one parcel, or a line whose
single order was despatched from more than one node. **Fan-out** is the number of parcels a line
generates; in D-1000 the mean is 720 / 626 = 1.15 parcels per parcel-bearing line, and 94 of 626
lines have fan-out 2.

Fan-out is the hidden variable in every promise comparison. Store-origin fulfilment has higher
fan-out than DC fulfilment because a single store rarely holds a whole multi-line order, so a naive
comparison of store versus DC promise accuracy measures the sourcing topology as much as the
operational quality.

### 3.7 Promise buffer and promise slack

**Buffer** is the padding between the earliest feasible delivery and the promised date. **Slack** is
the realised difference between the promised date and the actual delivery, which can be negative.
Buffer is a design choice made before the order ships; slack is an outcome observed after. A
retailer with generous buffer will show positive mean slack and mediocre fill, and a retailer with
tight buffer will show near-zero slack and frequent breaches. Neither is better without the cost
side, which Section 5.6 supplies.

### 3.8 Maturity, censoring, and the snapshot edge

A record is **mature** when its outcome is determined and cannot change: the parcel was delivered, or
the promised date passed without delivery, or the order was closed. A record is **open** when the
promised date is in the future relative to the snapshot. Open records are right-censored: including
them in the numerator as failures understates performance, and excluding them without stating the
exclusion overstates it by an unknown amount.

### 3.9 Recovery and the late-line cost

**Recovery** is any intervention after a breach: apology credit, expedited replacement, pickup
re-slot, or proactive notification before the breach occurs. The **late-line cost** is the expected
money lost per breached line, combining the credit actually issued with the handling labour of the
recovery contact. Section 5.6 uses $10.40, constructed as an $8.00 credit plus $2.40 of handling,
and every break-even in this sheet is expressed in those units so a reader can substitute their own.

Takeaway for the section: define the grain, the version, the eligibility boundary, and the cost unit
before computing anything. Four definitions, then arithmetic.
