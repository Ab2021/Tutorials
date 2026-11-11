## 5. Worked Numerical Example

The example traces one coherent decision through the whole chain so that no step is orphaned: a
retailer wants to know whether to shorten the promise from 2 days to 1 day on a subset of orders, and
the analysis has to convert that commercial idea into a measurable delta with a cost attached. Every
figure is ILLUSTRATIVE from D-1000 and every arithmetic step is re-checked independently at 5.7.

### 5.1 The decision and its comparison

The candidate: quote a 1-day promise instead of the current 2-day promise for single-SKU order lines
that carry the same-day cohort's characteristics. The comparator: keep the 2-day promise for those
lines, i.e. the frozen baseline. The estimand: change in OTIF percentage points on the issued base,
plus the change in late-line cost, plus the change in margin lost to abandonment. The decision rule:
adopt only if the net money change is positive and the OTIF change is distinguishable from zero at
the sample size available.

### 5.2 Step 1 — establish the cohort split

The four promise cohorts partition the 952 fulfilled lines. Recomputing the partition and its late
counts independently:

```text
cohort      lines   late   late rate
same-day      118     26     22.03%
1-day         402     38      9.45%
2-day         268     19      7.09%
4-day         164      8      4.88%
----------------------------------------
total         952     91      9.56%
```

Two checks on the table. First, 118 + 402 + 268 + 164 = 952, so the partition is exhaustive. Second,
26 + 38 + 19 + 8 = 91, matching the directly counted late lines, so no line is double-counted or
dropped. The late rate falls monotonically with promise length, which is the expected signature of a
common operational slack: a longer promise absorbs the same underlying variability.

The same-day cohort is the outlier and the reason it is the candidate. At 22.03% it is 2.3 times the
1-day rate; even allowing for the small denominator, that gap is not sampling noise. Its composition
explains it: 61 of the 118 are single-SKU orders and 57 are multi-SKU. Splitting on that dimension:

```text
same-day cohort decomposition
    single-SKU lines   61   late 19   31.15%
    multi-SKU lines    57   late  7   12.28%
    check 61 + 57 = 118;  19 + 7 = 26
```

The 18.87-point internal gap is larger than the gap between any two promise cohorts, which says the
promise length is not the operative variable inside same-day; the operative variable is whether the
order needed more than one pick from more than one location under a same-day deadline. That is a
routing fact, and it is the fact the candidate policy targets.

### 5.3 Step 2 — convert the cohort gap into a projected delta

The candidate moves same-day single-SKU lines to a 1-day promise. The projection assumes those lines
then breach at the 1-day cohort's rate of 38/402 = 9.4527%, which is the only defensible naive
estimate without an experiment. Recomputing:

```text
current late lines in the affected subset = 19 of 61
projected late lines at the 1-day rate    = 61 x 0.094527 = 5.766 -> 6 lines
lines saved (count arithmetic)            = 19 - 6 = 13 lines
projected new late total                  = 91 - 13 = 78 lines
projected new on-time complete lines      = 840 + 13 = 853
projected new OTIF                        = 853/1,000 = 85.30%
```

The 13-line saving is worth 13 x $10.40 = $135.20 at the F-04 unit cost. Note precisely which
component moves: the 61 affected lines are already inside the 952, so the denominator is unchanged at
1,000, while the 19 late lines fall to 6 and the on-time-complete numerator rises from 840 to 853.
OTIF therefore goes from 84.00% to 85.30%, a gain of 1.30 points. Only the numerator moves, and a
projection that instead moved the denominator would understate the gain by roughly 0.4 points.

### 5.4 Step 3 — cost the abandonment side

Shortening the promise is not free. A 1-day promise on an order that previously showed 2 days loses
conversion; the constructed assumption is 3% of the 61 affected orders abandon, which is 1.83 orders
per 13-week window. At a constructed gross margin of $12.00 per order, that is a margin loss of
1.83 x 12.00 = $21.96.

```text
net value = late-line saving - abandoned margin
          = 13 x 10.40 - 1.83 x 12.00
          = 135.20 - 21.96
          = 113.24 per 13-week window
```

The net is positive at $113.24, but the magnitude is small enough that the conclusion is
uncomfortable: the entire policy is worth about eleven late lines a quarter, and any unmodelled cost
of $114 per quarter reverses it. Two such unmodelled costs are obvious. The first is the contact
centre: a shorter promise generates more "where is my order" calls, and 40 calls at $3.00 each is
$120, which flips the sign. The second is the promise engine's own coverage: shortening the promise
on 61 lines may pull work into the same-day pick window, raising the utilisation in F-13.

### 5.5 Step 4 — test whether the delta is measurable

The policy promises a 1.30-point OTIF gain on a base of 1,000 lines per 13 weeks. With a binomial
standard error at p = 0.84:

```text
SE(p) = sqrt(p(1-p)/n) = sqrt(0.84 x 0.16 / 1,000) = sqrt(0.0001344) = 0.011593
95% CI on a single 1,000-line quarter: 84.00% -/+ 1.96 x 1.1593 = 81.73% .. 86.27%
```

A single quarter's interval is 4.54 points wide, so a 1.30-point improvement is invisible. This is
the most important structural fact about promise metrics: **the noise floor of a weekly OTIF number
is larger than most improvements anyone proposes**. The minimum detectable effect for a two-arm
comparison at 80% power and 5% two-sided significance, using the standard factor 2.8016:

```text
MDE(n per arm) = 2.8016 x sqrt(2 p (1-p) / n)
 n =  1,000  ->  MDE = 4.59 pts
 n =  4,000  ->  MDE = 2.30 pts
 n =  6,500  ->  MDE = 1.80 pts
 n = 13,000  ->  MDE = 1.27 pts
```

At 1,000 lines per week per arm, a 13-week test gives 13,000 per arm and an MDE of 1.27 points, just
under the 1.30-point effect. So the test is only barely powered with a full quarter of data on both
arms, and it is not powered at all if the arms are unbalanced. Any plan that runs the test for four
weeks and reads off the difference is reading noise: at 4,000 per arm the MDE is 2.30 points and the
observed gain of 1.30 is inside it.

### 5.6 Step 5 — break-even against two rejected alternatives

The candidate is not the only lever available. Two others were costed with the same method, and both
fail the break-even test, which is why the recommendation is the candidate and not them.

```text
LEVER A: upgrade all store-origin parcels to a 2-day express service
    volume 720 parcels (the 13-week total, not a weekly rate)
    cost   720 x $2.80 = $2,016.00 for the window
    best case: express removes every store late parcel, i.e. 52 lines
    value  52 x $10.40 = $540.80
    net    -$1,475.20  -> reject

LEVER B: consolidate all 94 split lines into single parcels
    cost   94 x $6.10 = $573.40 for the window
    best case: split lines late fall to the single-parcel rate
      94 x 0.097744 = 9.19 -> 9 late lines, down from 19
      lines saved = 10, value = 10 x $10.40 = $104.00
    net    $104.00 - $573.40 = -$469.40  -> reject

LEVER C (the candidate): re-promise 61 same-day single-SKU lines to 1 day
    net    +$113.24  -> adopt, conditional on the power test in 5.5
```

Lever A fails by a factor of 3.7 and is worth stating in exactly those terms, because "express
shipping fixes lateness" is the most frequently proposed and least cost-effective remedy in the
domain. Its break-even requires 2,016.00 / 10.40 = 193.8 late lines removed, and the store only
produces 52. Lever B fails by a smaller margin, 5.5 times the value it can create, and it has a
further defect: consolidation often increases the promise date because a single node rarely holds the
whole order, so the $6.10 is a floor and the promise may lengthen.

Lever B deserves one more calculation because it is the one a reader will challenge. Its value
depends entirely on the 19 late split lines falling to 9, which is the independence assumption again.
With the observed residual correlation of C-03, the true post-consolidation rate would land closer to
12%, i.e. 11 late lines rather than 9, and the saving falls to 8 lines worth $83.20. The rejection
survives the correction, which is the right way to use a bound: check that the conclusion is
insensitive to the bound's error before relying on it.

### 5.7 Independent arithmetic re-check

Every headline number in this section is re-derived by a second route, so that an error in one route
does not propagate silently.

```text
1. OTIF by chain: (952/1,000)(902/952)(840/902) = 0.952 x 0.947479 x 0.931264
   = 0.902000 x 0.931264 = 0.839999 -> 84.00%   agrees with 840/1,000
2. Cohort blend: 118(0.779661) + 402(0.905473) + 268(0.929104) + 164(0.951220)
   = 92.000 + 364.000 + 248.998 + 156.000 = 861.0 -> 861/952 = 90.44%   agrees
3. Split penalty: 52/532 = 0.097744; 1-(1-p)^2 = 0.185927 vs 19/94 = 0.202128
   ratio 0.202128/0.097744 = 2.068 -> the split rate is 2.07x, stated as 2.07x
4. Lever C saving: 19 - 61(0.094527) = 19 - 5.766 = 13.234 -> 13 lines
   value 135.20; net 135.20 - 21.96 = 113.24   agrees
5. Lever A break-even: 2,016.00/10.40 = 193.85 lines needed vs 52 available
6. Lever B saving: 19 - 94(0.097744) = 19 - 9.188 = 9.812 -> 10 lines
```

Check 3 is the one that most often fails in practice and it is worth spelling out why. The ratio is
not 19/94 divided by 71/626; the blended rate of 11.34% is contaminated by the split lines
themselves, and using it gives 20.21/11.34 = 1.78, which understates the penalty. The correct
denominator population is single-parcel lines only, because that is the population where fan-out is
one and the comparison isolates the effect of the second parcel.

### 5.8 Sensitivity note

Four assumptions carry the recommendation, and each is varied by a plausible amount to see whether
the conclusion survives.

```text
assumption                 base      stressed        effect on net $113.24
late-line unit cost        10.40     14.00           +46.80  (net 160.04, still adopt)
abandonment share          3.0%      8.0%            -37.00  (net  76.24, still adopt)
projection rate            9.45%     12.28%          -36.40  (net  76.84, still adopt)
handling cost per contact   2.40      2.40 + 120 fixed -120.00 (net  -6.76, reject)
```

The first three stresses leave the sign intact; the fourth reverses it. That asymmetry is the actual
finding: the policy's viability is governed by contact volume, not by late-line cost, and the
measurement plan should therefore count calls rather than count late parcels. A sensitivity analysis
that varies every input by the same percentage would have shown four similar-sized swings and hidden
the concentration of risk in one variable.

Takeaway for the section: a lever worth $113 per quarter is worth adopting only after its contact
cost is counted, and a 1.30-point effect is worth claiming only at 13,000 lines per arm.

## 6. Visual / Geometric Intuition

### 6.1 OTIF as a rectangle with two cuts

Draw the 1,000 issued lines as a rectangle 1,000 units wide. Cut it vertically at 952: that is the
fulfilment gate, and the 48 units to the right are cancellations, 31 of them customer-initiated. Now
cut the remaining 952 horizontally at 902: the 50 units above the cut are incomplete lines. Finally
cut the 902 column at 840 vertically: the 62 units to the right are complete-but-late.

The area of the small bottom-left rectangle is the OTIF count, 840, and it is 84.00% of the original
rectangle's area. The three cut widths are the three losses: 4.80 points for the first cut, 5.00 for
the second, 6.20 for the third. Because they are sequential cuts of a shrinking shape, their widths
add exactly to 16.00, which is why the chain identity in F-03 is additive in points of the issued
base and why the three losses can be plotted as a stacked bar with no residual.

The rectangle makes one thing visceral that a table does not: **the third cut operates on the widest
remaining column**. Lateness is applied to 90.2% of the base while cancellation is applied to 100% and
incompleteness to 95.2%, so a one-point change in the lateness rate moves OTIF by 0.90 points while a
one-point change in the cancellation rate moves it by 1.00. Lateness dominates here purely because
its base rate is higher (6.87% versus 4.80%), not because its lever is stronger.

### 6.2 The fan-out tree

For a single-parcel line the outcome is one coin flip at p = 9.77% lateness. For a split line it is
two flips, and the line is on time only if both land heads:

```text
                 +-- parcel 1 late (9.77%) --> line late
   split line ---|
                 +-- parcel 1 on time (90.23%)
                        |
                        +-- parcel 2 late (9.77%)  --> line late
                        +-- parcel 2 on time      --> line on time

   P(on time) = 0.9023 x 0.9023 = 0.8141   =>  18.59% late
```

The tree is deliberately drawn unbalanced in width to show that the second parcel's failure matters
only on the 90.23% branch where the first succeeded. That is why the penalty per extra parcel shrinks
as k grows in absolute terms while growing in relative terms: the third parcel adds 0.9023 cubed
minus 0.9023 squared, i.e. 73.45% - 81.41% = a further 7.96 points of lateness, less than the second
parcel's 8.82 points.

Plotting on-time reliability against fan-out gives a decay curve: 90.23% at k=1, 81.41% at k=2,
73.46% at k=3. A reader who sees that curve should immediately ask what share of lines sit at k>=2,
because the blended rate is the area under the curve weighted by that distribution. In D-1000 the
weights are 532 at k=1 and 94 at k=2, and weighting the TWO OBSERVED rates gives
0.8498 x 90.23% + 0.1502 x 79.79% = 76.68% + 11.98% = 88.66%, exactly the observed 555/626.
Substituting the independence prediction of 81.41% for the split cell would instead give 88.91%
and overstate parcel on-time performance by a quarter of a point. That reproduction is the strongest available
evidence that the mechanism is understood rather than merely described.

### 6.3 The promise frontier as a curve, not a point

A promise length is a point on a curve, not a policy in itself. Holding the operation fixed and
varying buffer from 0 to 2 days produces a constructed frontier:

```text
buffer (days)   late rate   late lines per 952   marginal lines saved
     0.0          11.25%          107.1                 --
     0.5           7.00%           66.6               40.5
     1.0           4.40%           41.9               24.7
     1.5           2.80%           26.7               15.2
     2.0           1.90%           18.1                8.6
```

The curve is convex, which is the geometric statement of diminishing returns: the first half-day of
buffer removes 40.5 late lines, the last half-day removes 8.6. The marginal line saved costs whatever
the buffer costs in conversion, so the optimal buffer sits where the marginal conversion loss per
line saved equals the marginal late-line saving. With $10.40 per late line, the last half-day is worth
8.6 x 10.40 = $89.44 per window, so any buffer extension costing more than that in lost margin is
over-buffered. The curve is constructed, not measured, and the numbers in the table are the
ILLUSTRATIVE part of this subsection; the shape is the transferable claim.

### 6.4 The capacity wall

Plotting promised demand against committed capacity over the 13 weeks produces a flat line for most
of the window and one spike. The spike is the interesting object: a peak day with 1,180 promised
lines against 340 committed picks is a utilisation of 347%, and the area between the demand curve and
the capacity line is the backlog. Integrated over the spike it is 3.47 days of work. Nothing
downstream of the pick can repair that area, and this is why the capacity check belongs *before* the
promise is quoted rather than in the post-mortem.

The wall is also why a promise metric can be simultaneously accurate and useless. If the promise
engine already quotes from capacity-constrained dates, its OTIF will look excellent and its late
volume will be low, but the dates will be long and the conversion cost invisible in any fulfilment
report. That is the failure mode a fulfilment dashboard structurally cannot see, and it needs the
commercial side of the ledger.

## 7. Comparison / Selection Table

### 7.1 Choosing the promise metric

| Metric | Question it answers | Fails when | Selection rule |
|---|---|---|---|
| line fill F-01 | did the customer get everything | partial lines rare so gap is hidden | use alongside unit fill |
| unit fill F-01 | how much volume was short | single-unit baskets dominate | prefer when baskets are large |
| OTP F-02 | did parcels land by the date | split lines mixed with single | always stratify by fan-out |
| OTIF F-03 | did we keep the whole promise | denominator undisclosed | quote with the base named |
| fan-out F-05 | how much structural risk | independence assumed | use as a bound, not a forecast |
| pickup readiness F-10 | was the store ready | notification treated as ready | timestamp the ready scan |
| late cost F-04 | what a breach costs | overhead allocated in | marginal costs only |
| censored rate F-11 | rate at snapshot edge | open pool large | report mature plus open count |

Read the "fails when" column as a list of questions to ask whoever produced the number. A line fill
quoted without unit fill, an OTP quoted without a fan-out split, and an OTIF quoted without its base
are the three most common defective statements in retail fulfilment reporting, and each corresponds
to one row above.

### 7.2 Choosing the comparator

| Comparator | Legitimate use | Illegitimate use |
|---|---|---|
| prior period, same length | steady-state trend | a period with a promotion or a promise change |
| previous promise policy | policy change effect | mixed with a carrier change in the same week |
| parallel holdout | causal claim | holdout contaminated by shared carrier capacity |
| node-to-node | operational diagnosis | league table without standardisation |
| industry benchmark | context only | target setting, since definitions differ |

The benchmark row deserves emphasis. Published fulfilment benchmarks are computed on unknown grains
with unknown exclusion rules; using one as a target imports another company's definitional choices.
It is legitimate to say "our 84% is low relative to a published 92% and we should investigate"; it is
not legitimate to write 92% into a target without re-deriving it on your own grain.

### 7.3 Choosing the recovery lever

| Lever | Cost driver | Precondition | Break-even in D-1000 |
|---|---|---|---|
| proactive notification | contact cost per flagged line | flag available before promise date | 4.2 notifications per breach saved |
| express upgrade | fixed uplift per parcel | carrier has capacity | 193.8 late lines; only 52 exist |
| split consolidation | premium per consolidated order | single node holds the order | 55.1 late lines; only 19 exist |
| promise re-scoping | conversion loss | cohort distinguishable | 13 lines saved, net positive |
| inventory accuracy | count labour | phantom stock is the cause | 4.5 phantom units per late line |

The table's value is in the last column: it converts each lever into the number of prevented
failures it must achieve to pay for itself, so the comparison becomes arithmetic rather than
argument. Notice that every lever except re-scoping fails, and all three failures fail by similar
orders of magnitude (3.7x to 5.5x short). That pattern is itself a finding: the cost of prevention in
this dataset exceeds the cost of failure by a wide margin, which is why a rational retailer invests in
measurement and prevention of the largest, most concentrated failure rather than in universal
insurance against every breach.

### 7.4 Choosing the grain for a report

| Grain | Use | Distortion introduced |
|---|---|---|
| unit | inventory and replenishment | hides orders that arrived partially |
| line | operational promise tracking | weights small orders equally with large |
| order | customer-experience reporting | a single late item marks a whole order late |
| parcel | carrier and cost management | a split order appears as two successes or failures |
| customer | loyalty and retention analysis | repeated failures on one customer are averaged away |

The customer grain is the one most often omitted and the one most consequential commercially. A
customer who experiences three late orders in a quarter is not three times a late line; they are a
different behavioural object, and the loyalty consequence is non-linear. Section 10.4 states the
bound that makes the point precisely: an 8% line-level failure rate produces roughly a 22% chance
that a five-order customer sees at least one failure, and that is before any correlation.

Takeaway for the section: pick the metric by the decision, the comparator by the confounder, the
lever by the break-even, and the grain by the consequence. Four independent choices, and getting
three right does not compensate for the fourth.
