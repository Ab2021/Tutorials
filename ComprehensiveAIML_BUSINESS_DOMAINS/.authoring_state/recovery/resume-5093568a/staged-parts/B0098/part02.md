## 4. Core Formulas with Full Derivation

This section is the authoritative algebra for the sheet. Each substantive result is presented in the
five-part anatomy: formula, where each symbol is defined with units and range, the constraint that
must hold for the formula to be legal, a numerical example from D-1000, and an interpretation that
says what the number licenses you to do. Every subsection closes with an explicit validity
condition, because a formula quoted outside its validity condition is the most common way a correct
derivation becomes an incorrect decision.

### 4.1 Base case: OTIF as a conjunction on a frozen denominator

#### 4.1.1 Fill rate at three grains

```text
FORMULA [F-01]: line_fill  = L_complete / L_fulfilled
                unit_fill  = U_shipped  / U_ordered
                order_fill = O_complete / O_total
WHERE: L_complete [lines, integer >= 0]  complete first-attempt lines
       L_fulfilled [lines, integer > 0]  lines that entered picking
       U_shipped [units, integer >= 0]   units actually despatched
       U_ordered [units, integer > 0]    units promised on those lines
       O_complete [orders, integer]      orders with every line complete
       O_total [orders, integer > 0]     orders with at least one fulfilled line
CONSTRAINT: all three ratios use the same window, the same market, and the same
            exclusion list; U_shipped and U_ordered must refer to the same lines.
NUMERICAL EXAMPLE (D-1000): 902/952 = 0.947479 = 94.75% line fill
                            1,122/1,240 = 0.904839 = 90.48% unit fill
                            726/812 = 0.894089 = 89.41% order fill
INTERPRETATION: the spread of 5.34 points between line and order fill, and 4.26 points
between line and unit fill, is not rounding; it is partial lines on multi-unit orders.
```

The spread is diagnosable. Line fill 94.75% and unit fill 90.48% differ by 5.34 points; the missing
units live on 34 partial lines that together ordered 96 units and shipped 22. Those 34 lines ordered
2.82 units each on average versus 1.30 units per line across the whole dataset, so partial failures
are concentrated exactly where they hurt most in unit terms. A retailer that reports only line fill
is reporting the most flattering of the three numbers on data where partial lines are heavy.

Validity condition for F-01: the three ratios are comparable only inside one dataset. Do not compute
the spread between a line fill from one system and a unit fill from another; the 5.34-point gap above
is a property of D-1000's order composition, not a general constant, and a grocery dataset with
basket sizes of twenty units would show a far wider gap.

#### 4.1.2 On-time rate for parcel-bearing lines

```text
FORMULA [F-02]: otp_lines = LP_on_time / LP_parcel
WHERE: LP_on_time [lines, integer]  parcel-bearing lines whose last parcel arrived on or before
                                   the checkout promised date
       LP_parcel  [lines, integer] parcel-bearing lines with a mature outcome
CONSTRAINT: the promised date is the version-1 date; a line re-promised before despatch
            is counted against version 1, and a line whose promise moved after despatch
            is counted against the date the customer last saw before despatch.
NUMERICAL EXAMPLE: 555 of 626 parcel-bearing lines on time -> 555/626 = 88.66%
    decomposition: single-parcel lines 532 with 52 late -> 480/532 = 90.23%
                   split lines 94 with 19 late          ->  75/94  = 79.79%
INTERPRETATION: the split-line rate is 10.44 points below the single-parcel rate,
so the blended 88.66% is a mixture of two very different populations, not one rate.
```

The blended figure for parcel-bearing lines is 88.66%, and every later section treats that number as
a mixture rather than a property. This matters for the store-versus-DC comparison in 4.3.2: nodes
differ in their share of split lines, so a comparison of node-level on-time rates without a split
adjustment is partly a comparison of sourcing topology.

Validity condition for F-02: the denominator must contain only lines whose outcome is determined.
A parcel still in transit on the snapshot date is not late, and it is not on time either; it belongs
in the open pool of 8.2 until it resolves.

#### 4.1.3 OTIF and the chain identity

OTIF is the product of three conditional rates, and this identity is the most useful single
relationship in the sheet because it converts an opaque headline into three addressable losses.

```text
FORMULA [F-03]: OTIF = (L_fulfilled/L_issued) x (L_complete/L_fulfilled) x (OT|complete)
WHERE: L_issued     [lines]  checkout promises issued in the window
       L_fulfilled  [lines]  subset that entered picking (excludes cancellations)
       L_complete   [lines]  subset of those fulfilled complete
       OT|complete  [fraction] P(on time | fulfilled complete)
CONSTRAINT: the three factors must come from one cohort traced end to end; multiplying a
            survival rate from one report by a fill rate from another is invalid.
NUMERICAL EXAMPLE: (952/1,000) x (902/952) x (840/902) = 0.952 x 0.9475 x 0.9313
                   = 0.84 exactly -> 84.00% OTIF on the issued denominator
INTERPRETATION: the losses are 4.80 pts (cancellation), 5.00 pts (incomplete), and
6.20 pts (lateness given complete). Lateness, not stock, is the largest single loss.
```

Three properties of F-03 are worth memorising. First, the losses are additive in percentage points
of the issued base: 4.80 + 5.00 + 6.20 = 16.00, and 100.00 - 16.00 = 84.00. That additivity is exact,
not approximate, because the three factors are sequential conditional rates on a telescoping
product. Second, the denominator choice changes the headline by a large and easily confused amount:
840/952 = 88.24% is "OTIF among lines we actually shipped", which is 4.24 points higher than the
issued-base 84.00%. Third, the identity shows where to intervene: improving lateness-given-complete
from 93.13% to 96.00% moves OTIF by 0.952 x 0.9475 x 0.0287 = 2.59 points, while improving fill from
94.75% to 97.00% moves it by 0.952 x 0.0225 x 0.9313 = 2.00 points. Lateness is the bigger lever on
this dataset, and that conclusion is readable directly from the multiplicative structure.

```text
COUNTEREXAMPLE [C-01]: the denominator swap is not a rounding choice
Report A: 840/952 = 88.24% "OTIF" (fulfilled base)
Report B: 840/1,000 = 84.00% "OTIF" (issued base)
Both use identical numerator 840 and identical window. The 4.24-point gap is entirely
the 48 lines that never shipped (31 pre-release cancellations + 17 post-release).
If the retailer excludes post-release cancellations from the base to "measure operational
quality fairly", it must say so; the customer who received nothing experienced a failure.
```

Validity condition for F-03: the numerator of the last factor must be a subset of L_complete, and
L_complete a subset of L_fulfilled. A pipeline that computes OTIF by joining an on-time flag to a
separate fill table and taking an inner join silently drops the 16 failed lines, which is precisely
how a retailer reports 88% while shipping nothing on 48 of 1,000 orders.

### 4.2 General case: promise cost, fan-out, and utility

#### 4.2.1 Expected cost per late line

```text
FORMULA [F-04]: C_late = c_credit + c_handling + c_expedite * P(expedite | late)
WHERE: c_credit [currency/line]     credit or refund actually issued for a breach
       c_handling [currency/line]   fully loaded labour for one recovery contact
       c_expedite [currency/line]   cost of a replacement or upgrade when offered
       P(expedite|late) [fraction]  share of breaches that trigger expediting
CONSTRAINT: costs are marginal, not average; do not allocate fixed warehouse overhead
            to a late line, or the lever comparison in 4.2.3 becomes uninterpretable.
NUMERICAL EXAMPLE: 8.00 + 2.40 + 6.00 x 0.00 = $10.40 per late line
    total exposure: 91 late lines x $10.40 = $946.40 over the 13-week window
INTERPRETATION: the exposure is small relative to the parcel base, which is why
expedite-everything policies fail the break-even test in 5.6 rather than the moral one.
```

The choice of marginal costing is deliberate and consequential. If a retailer allocates $4.10 of
facility overhead to every line, the late-line cost rises to $14.50 and every intervention in
Section 5 suddenly looks profitable, including ones that cost more than the failures they prevent.
The right test is always incremental: does this lever change the cash the retailer would otherwise
spend, and does it change the failures?

Validity condition for F-04: c_credit must be the amount actually issued, not the published policy
maximum. A policy that offers $8 but is applied to 60% of breaches has an expected credit of $4.80,
and using $8 overstates the value of prevention by two thirds.

#### 4.2.2 Parcels per order and the fan-out distribution

```text
FORMULA [F-05]: fanout_mean = N_parcels / N_parcel_lines
                E[late | fanout=k] = 1 - (1 - p)^k   under independent legs
WHERE: N_parcels [parcels]  physical shipments in the window
       p [fraction]         marginal probability that one parcel misses its date
       k [integer >= 1]     parcels generated by one line
CONSTRAINT: independence across the parcels of one line. This constraint is false in
practice (Section 4.4.2) and the formula therefore understates the split penalty.
NUMERICAL EXAMPLE: fanout_mean = 720/626 = 1.1502
    p = 52/532 = 0.097744 for single-parcel lines
    predicted split lateness = 1 - (1 - 0.097744)^2 = 0.18593 = 18.59%
    observed split lateness  = 19/94 = 20.21%
INTERPRETATION: independence explains 18.59 of the 20.21 observed points; the residual
1.62 points is the correlation between parcels of the same order.
```

This is the sheet's central quantitative result: **the observed split penalty is almost entirely
mechanical**. A retailer that splits 15% of its parcel-bearing lines will show a lateness rate on
those lines roughly double its single-parcel rate, with no operational failure at all. The temptation
is to attribute the 20.21% to poor coordination between nodes; the arithmetic says otherwise, and the
discriminating test in 8.4 confirms the mechanical explanation before any process change is
proposed.

Validity condition for F-05: p must be estimated on single-parcel lines of the same node mix. If p is
taken from the blended rate of 11.34% (71 late lines of 626), the independence prediction becomes
21.40% and over-explains the observed 20.21%, which would hide the true residual. Always calibrate p
on the population where fan-out is one.

#### 4.2.3 Utility of a recovery queue with capacity

```text
FORMULA [F-06]: select the top min(K, N) candidates ranked by u_i = P(prevent_i) * C_late - c_act_i
WHERE: P(prevent_i) [fraction]  probability the action prevents the breach
       C_late [currency]        cost of the breach per F-04
       c_act_i [currency]       cost of the action for candidate i
       K [actions/window]       review or contact capacity
       N [candidates]           lines flagged as at risk before the promise date
CONSTRAINT: u_i must be computed from information available before the promise date;
            a candidate flagged after delivery has u_i = 0 by definition.
NUMERICAL EXAMPLE: P(prevent) = 0.60, C_late = $10.40, c_act = $2.50
    u = 0.60 x 10.40 - 2.50 = 6.24 - 2.50 = $3.74 per action
    with K = 15 and 22 candidates: max value = 15 x 3.74 = $56.10 per window
INTERPRETATION: the queue is worth running, but the expected value of the fifteenth
action is the same as the first only if all candidates share P(prevent); they do not,
so rank by u_i and stop when u_i < 0, not when the queue is empty.
```

The stopping rule matters more than the ranking. A queue that runs to capacity K regardless of u_i
will spend $2.50 on candidates whose prevention probability is 0.10, producing 1.04 - 2.50 = -$1.46
each. In D-1000 terms, 22 candidates at K=15 means 7 are left untouched; if those 7 are the
lowest-utility ones, the queue is correctly designed, and if the ranking is by order value rather
than by u_i, it is not.

Validity condition for F-06: P(prevent_i) must be estimated on held-out data. A prevention
probability fit on the same lines it later scores is an in-sample artefact, and in this domain the
artefact is large because late lines share visible pre-breach signals (a pick delay, a carrier scan
gap) that are themselves caused by the thing being predicted.

#### 4.2.4 Why fan-out compounds rather than divides

The intuitive error is to suppose that splitting a line across two nodes halves the promise risk,
because each node handles less work. The algebra gives the opposite sign.

```text
FORMULA [F-07]: P(line on time | k parcels) = (1 - p)^k  under independent legs
WHERE: p [fraction] per-parcel lateness; k [integer] parcels for the line
CONSTRAINT: identical p across parcels. If store parcels are systematically worse
(p_store > p_dc), then P = (1 - p_dc)^a (1 - p_store)^b for a DC and b store parcels.
NUMERICAL EXAMPLE: p = 0.0977, k = 1 -> 90.23% on time
                   p = 0.0977, k = 2 -> 0.9023^2 = 81.41% on time
                   observed split-line on time = 75/94 = 79.79%
INTERPRETATION: one extra parcel costs 8.8 points of promise reliability at this
base rate. Fan-out is a reliability multiplier below one, never a divisor.
```

Restating with the node split makes the mechanism explicit and produces the number used in the
store-versus-DC comparison. Of the 94 split lines, 52 are DC-origin splits and 42 are store-origin
splits. Using the node-specific single-parcel rates of 0.9300 for DC and 0.8301 for store, a
two-parcel DC line is on time with probability 0.9300 squared = 86.49%, while a two-parcel store line
is on time with probability 0.8301 squared = 68.91%. The store penalty is 17.6 points, roughly double
the DC penalty of 6.5 points, and that difference is a pure consequence of the worse underlying
parcel reliability, not of extra coordination difficulty.

Validity condition for F-07: the legs must be distinct physical parcels. A line shipped as two
parcels from the same DC on the same truck is positively correlated, so its true joint reliability is
higher than the product. The product is therefore an upper bound on lateness, which is exactly the
direction that makes it safe as a screening rule and unsafe as a forecast.

### 4.3 Variants

#### 4.3.1 Direct standardisation for node comparison

```text
FORMULA [F-08]: r_std(A) = sum_c w_c * r_c(A),   w_c = share of cohort c in the standard population
WHERE: r_c(A) [fraction]  on-time rate of node A within promise cohort c
       w_c [fraction, sums to 1]  cohort weights of the standard population
       c in {same-day, 1-day, 2-day, 4-day}
CONSTRAINT: cohorts must be defined identically in both nodes and the standard
            weights must come from a population where both nodes contribute.
NUMERICAL EXAMPLE (common support = 1-day and 2-day cohorts only):
    DC:    1-day 206/214 on time, 2-day 157/168 on time
    store: 1-day 134/152 on time, 2-day  62/70  on time
    DC rate on support    = (206+157)/382 = 95.03%
    store rate on support = (134+ 62)/222 = 88.29%
INTERPRETATION: on the promise cohorts both nodes actually serve, the gap is 6.74
points, not the crude 9.99 points from the full-mix comparison.
```

The 3.25 points between the crude and standardised gaps is pure mix. The store carries an 84-parcel
same-day cohort at 69.05% on time; the DC carries a 32-parcel four-day cohort at 68.75% on time.
Both are difficult promises, so their presence in unequal proportions distorts the crude comparison
in both directions at once. Restricting to common support is the cleaner of the two adjustments and
needs no weight vector, which is why it is the default recommendation here.

Validity condition for F-08: direct standardisation requires that each cohort exists in both
populations with a reasonable count. With 32 and 84 records the four-day and same-day cohorts are
too thin to serve as their own strata, which is exactly why they are excluded rather than weighted.

#### 4.3.2 Weighted average of rate over mixed populations

```text
FORMULA [F-09]: r_blend = sum_c n_c r_c / sum_c n_c
WHERE: n_c [records]  count in cohort c; r_c [fraction] cohort rate
CONSTRAINT: n_c and r_c must come from the same snapshot; a rate recomputed later
            against a backfilled cohort list is not the same quantity as the original.
NUMERICAL EXAMPLE: (118 x 0.7797 + 402 x 0.9055 + 268 x 0.9291 + 164 x 0.9512) / 952
    = (92.00 + 364.01 + 248.99 + 156.00) / 952 = 861.00/952 = 90.44%
INTERPRETATION: the blend equals the sum of late lines divided by the total, which is
the arithmetic identity that must hold; if it does not, one cohort rate is miscomputed.
```

This formula earns its place because it is also the check. The weighted sum of cohort rates must
reproduce the aggregate count, and the arithmetic above returns exactly 861 on-time lines, matching
the direct count of 952 - 91. Any dashboard where the weighted cohort rates do not reconcile to the
headline count has a join defect, a double-count, or a stale cohort table.

Validity condition for F-09: the cohort partition must be exhaustive and disjoint. In D-1000 the four
promise cohorts sum to 118 + 402 + 268 + 164 = 952 exactly, and their late counts 26 + 38 + 19 + 8 =
91 exactly. A cohort table that sums to 950 has two lines in no bucket, and those two will appear in
some downstream join as either on-time or deleted depending on the join type.

#### 4.3.3 Pickup readiness for BOPIS

```text
FORMULA [F-10]: pickup_ready = ready_by_promise / pickup_records
                pickup_conversion = collected / pickup_records
WHERE: ready_by_promise [records]  orders marked ready at or before the promised slot
       collected [records]        orders actually picked up before the hold expiry
       pickup_records [records]   BOPIS orders with a mature slot
CONSTRAINT: readiness must be timestamped by the "ready for pickup" scan, not by the
sms notification; notification delay is a separate failure with a separate owner.
NUMERICAL EXAMPLE (D-1000): ready 318/338 = 94.08%
                            collected 272/338 = 80.47%
INTERPRETATION: readiness looks strong next to the parcel rate, but 66 of 338 orders
were never collected, which is a holdings problem, not a promise problem.
```

The two numbers answer different questions and must not be combined into one "BOPIS success" figure.
Readiness is a store labour and inventory accuracy metric; collection is a customer behaviour and
hold-window metric. A store that is 100% ready and collects 70% is holding stock for orders that will
never arrive, and the holding cost is real even though every promise was honoured.

Validity condition for F-10: records whose hold window has not yet expired are open, not failures.
With a 7-day hold and a 13-week window, the last 7 days of orders are necessarily immature, and
including them as uncollected would understate the conversion rate by roughly 7/91 = 7.7%.

#### 4.3.4 Censoring-adjusted rate

```text
FORMULA [F-11]: r_adj = favorable_mature / (eligible_mature + w * eligible_open)
WHERE: favorable_mature [records]  resolved on time
       eligible_mature [records]   resolved late, or on time, outcome known
       eligible_open [records]     promise date not yet reached at snapshot
       w [fraction, 0..1]          assumed eventual success share of the open pool
CONSTRAINT: w must be bounded by the historical success rate of comparable cohorts;
            reporting r_adj without stating w is reporting an undisclosed assumption.
NUMERICAL EXAMPLE: D-1000 has 861 favourable of 952 mature and 0 open, so
    r_adj = 861/952 = 90.44% and w is immaterial.
  Illustrated instability with the same dataset plus 40 open records:
    w = 0.93 -> (861 + 0.93 x 40)/992 = 898.20/992 = 90.54%
    w = 0.50 -> (861 + 0.50 x 40)/992 = 881.00/992 = 88.81%
    w = 0.00 -> (861 + 0.00 x 40)/992 = 861.00/992 = 86.79%
INTERPRETATION: an open pool of only 4% of the base spans 3.75 points of headline
depending on a weight nobody measured, so report the mature rate plus the open count.
```

D-1000 is frozen at a cut-off where every issued promise has resolved, so the censoring adjustment is
theoretically required and numerically inactive. That is deliberate: the dataset is designed to teach
the formula without letting the formula's free parameter contaminate the teaching numbers. A live
dashboard does not have that luxury, and the honest convention is to print "90.44% on 952 mature
records, 0 open" rather than a single adjusted number whose weight nobody can audit.

Validity condition for F-11: the open pool must be defined by promise date, not by order date. An
order placed 2 days before the snapshot with a 4-day promise is open; an order placed 10 days before
with a same-day promise is resolved and late. Defining openness by order age rather than promise date
mixes the two populations and makes w meaningless.

### 4.4 Limits and counterexamples

#### 4.4.1 Percentiles do not add

```text
FORMULA [F-12, INCORRECT]: p95(total) = sum over legs of p95(leg)
COUNTEREXAMPLE [C-02]: two transit legs, each uniform on {1, 2, 3} days
    leg p95 = 3 days, so the naive sum is 6 days
    the true sum distribution is 9 equally likely pairs:
      2(1) 3(2) 4(3) 5(2) 6(1)
    cumulative share at 4 days = 6/9 = 66.67%
    true p95 of the sum = 6 days, only because the support is bounded at 6
    with four legs (81 outcomes, sums 4..12) true p95 = 11 while the naive
    sum of marginal p95s is 12 and the mean is 8
CONSTRAINT: the naive sum is exact only when one leg has zero variance.
INTERPRETATION: for four legs the naive rule over-promises by one full day; the error
grows with the number of legs and with leg variance, and it always errs late.
```

The four-leg case is the one that bites. A chain of pick, line-haul, local depot, and last mile has
four legs, and adding the four 95th percentiles gives 12 days where the true 95th percentile of the
sum is 11 days. Over-promising by a day is a commercial error in the wrong direction: the promise
page shows a date the operation will miss 5% of the time even when every leg behaves as designed.

```text
CHECK on the four-leg distribution: counts by sum, sums 4..12
   4:1  5:4  6:10  7:16  8:19  9:16  10:10  11:4  12:1   total 81
   cumulative through 8: 50/81 = 61.73%
   cumulative through 9: 66/81 = 81.48%
   cumulative through 10: 76/81 = 93.83%
   cumulative through 11: 80/81 = 98.77%
   so p95 = 11 days and P(sum > 8) = 31/81 = 38.27%
```

That last figure is the useful one for buffer design. With four independent legs each uniform on
{1,2,3}, a promise of 8 days is breached 38.27% of the time. A planner who reasons "each leg is at
most 3 days, so 4 legs need at most 12, and 8 gives me a day of slack" has mis-specified slack: the
mean is 8, so a promise of 8 is at the centre of the distribution, not near its edge.

Validity condition for F-12: the additivity holds when the legs are deterministic or when the
promise target is the mean rather than a quantile. Most real promise engines quote a quantile, which
is why this counterexample applies to almost every live system.

#### 4.4.2 Independence fails in the observed data

```text
COUNTEREXAMPLE [C-03]: split lines are late more often than the product predicts
    single-parcel lateness p = 52/532 = 9.7744%
    product prediction for a 2-parcel line = 1 - (1-p)^2 = 18.593%
    observed 2-parcel lateness = 19/94 = 20.213%
    residual = 1.62 points, which is 8% of the observed rate
CONSTRAINT VIOLATED: the two parcels of one line share a despatch date, a carrier
network, and often a weather or backlog event, so their lateness is positively
correlated. Positive correlation makes the product optimistic about on-time.
INTERPRETATION: the product is a lower bound on the split penalty; quote it as
"at least" and use the observed rate for capacity and cost work.
```

The direction of the error is worth stating precisely, because it is easy to get backwards. Positive
correlation means the two parcels fail together more often than independence allows, which raises
the probability that at least one fails. So the independent product *understates* lateness, and the
formula's promise-reliability output is an optimistic bound. Using it to size a buffer would leave
the buffer slightly too thin, which is the opposite of what a safety-margin calculation should do.

Validity condition for F-07 and F-05 revisited: both are screening tools. They are legitimate for
answering "is the split penalty mechanical or operational", and illegitimate for answering "how many
late lines will we have next quarter". The first question tolerates a bound; the second does not.

#### 4.4.3 Simpson reversal between nodes and cohorts

```text
COUNTEREXAMPLE [C-04]: a node that wins every cohort can lose the aggregate
    Node P: cohort X  16/20 = 80.00%;  cohort Y 76/80 =  95.00%;  total 92/100 = 92.00%
    Node Q: cohort X  86/95 = 90.53%;  cohort Y  5/ 5 = 100.00%;  total 91/100 = 91.00%
    Q is better inside X by 10.53 points and inside Y by 5.00 points, yet Q's
    aggregate of 91.00% sits a full point below P's 92.00%.
CONSTRAINT VIOLATED: aggregates compared across populations with different cohort
mixes. P serves 80 of its 100 records in easy cohort Y; Q serves 95 of its 100 in
the harder cohort X, so P's advantage is a routing statement, not a quality one.
INTERPRETATION: the aggregate ranking is an artefact of the mix. Any node league
table that does not stratify by promise cohort is quotable only as a mix statement.
```

This is the exact trap that the crude 9.99-point store-versus-DC gap falls into. Store fulfilment
serves 84 same-day parcels at 69.05% on time; the DC serves 32 four-day parcels at 68.75%. Neither
node is uniformly better; their promise engines route different work to them, and the aggregate
difference is partly a statement about routing policy. The fix is F-08's common-support restriction,
which yields a 6.74-point gap that is quotable.

Validity condition for aggregate node comparison: the cohorts must have similar mixes, or the
comparison must be stratified, or the reader must be told explicitly that the number is a mix
statement. There is no fourth option, and computing a confidence interval does not rescue the
comparison because the bias is in the estimand, not in the sampling error.

#### 4.4.4 Capacity-constrained promises

```text
FORMULA [F-13]: feasible promise d is quoted only if expected demand by d <= K_d
WHERE: K_d [picks/day]  committed pick capacity for day d, net of forecast absence
       expected demand by d [picks/day]  promised volume scheduled to pick on day d
CONSTRAINT: capacity must be committed capacity, not nameplate; a DC rated at 1,400
            picks/day that reliably delivers 340 after absence and breaks is not 1,400.
NUMERICAL EXAMPLE: a peak day carries 1,180 promised lines against 340 committed
    picks -> utilisation 347% and 3.47 days of accrued backlog
INTERPRETATION: a promise the capacity cannot serve is not a promise; it is a queue
with a date attached, and its breach rate is a capacity fact rather than a model error.
```

The 3.47 days of accrued work is the number to carry into a trading review. It says that no amount
of carrier performance or store discipline recovers the day; the work simply does not fit, and the
promise is structurally unkeepable. The correct response is to lengthen the promise for the affected
days, which is a commercial decision, not an analytics one, and the analytics contribution is to
state the demand-capacity crossing date early enough for marketing to act on it.

Validity condition for F-13: demand must already be promised demand, not forecast demand. A promise
engine that quotes from forecast demand before the orders exist is running the same test with an
error term attached, and in a promotion week that error term dominates.

Closing validity statement for Section 4: F-01 through F-11 are estimands that require a frozen
denominator, a stated grain, and mature outcomes; F-05 and F-07 are bounds rather than forecasts;
F-08 is the only legitimate way to compare nodes with different promise mixes; and F-12 is a warning
rather than a formula, because the operation it describes is wrong. A practitioner who can state
which of those four sentences applies to a given number has passed the main test of this sheet.
