## 5. Worked Numerical Example

The example is one constructed week at one constructed node, carried through five checks. Every
input is ILLUSTRATIVE; every arithmetic step is recomputed and shown so that a reader can find the
first place where their own numbers diverge.

### 5.1 The setup

A node quotes same-day pickup and next-day delivery. Over a two-week comparison window it takes
18,300 orders in week 41 and 18,400 in week 42 — nearly identical volume, which is deliberate,
because it removes volume as a candidate cause before the analysis starts. The stage pass rates are
those used in Section 4.2. The question the operations lead asks is the wrong one: "did the pickers
get worse?" The right question is "which stage moved, and how much of the headline does each stage
explain?"

### 5.2 Step 1 — recompute the customer-visible rate from counts, not from the model

```text
week 41: R1 = 0.940275 ; orders = 18,300 ; promise-kept = round(18300 x 0.940275) = 17,207
         observed rate = 17,207 / 18,300 = 0.940273
week 42: R2 = 0.896054 ; orders = 18,400 ; promise-kept = round(18400 x 0.896054) = 16,487
         observed rate = 16,487 / 18,400 = 0.896033
headline change = 0.940273 - 0.896033 = 0.044240 = 4.42 pp of the order population
count change    = 17,207 - 16,487 = 720 orders
```

The last line is the one to put in front of an operations lead, because it is denominated in
orders rather than in points. A 4.42-point decline on a stable base is 720 additional failed
promises. Note also that recomputing from counts gives 0.940273 rather than 0.940275 — a 0.0002 pp
difference caused by rounding the count. That is acceptable here; it would not be if the same
rounding were applied to a segment of 300 orders, where one order is 0.33 pp.

### 5.3 Step 2 — separate the volume effect from the rate effect

The base grew by 100 orders, so a naive "fewer on-time orders" count comparison mixes two effects.
Expected promise-kept at week-41 rates on week-42 volume:

```text
expected = 18,400 x 0.940273 = 17,301.0
actual   = 16,487
shortfall attributable to rate = 17,301 - 16,487 = 814 orders
volume contribution = 17,301 - 17,207 = 94 orders (a base effect, not a performance effect)
check: 814 + 94 = 908 ... and 17,207 - 16,487 = 720
```

The check line exposes a trap. The count difference between the two weeks is 720, but the rate
shortfall is 814 and the volume contribution is +94 (the larger base *helped* the count). The
identity is `720 = 814 - 94`, not `814 + 94`. Decompositions that add contributions with the wrong
sign are the most common arithmetic defect in a fulfilment review, and the way to catch it is
always to force the identity to hold numerically before writing the sentence.

### 5.4 Step 3 — attribute the rate effect across stages

Using P-02, the log shares are capacity 50.95%, stock 40.56%, pick 6.40%, notification 2.09%. Two
sentences follow, and they must be kept apart:

- **Arithmetic:** half of the log decline is attributable to the capacity stage and 41% to stock.
- **Action:** restoring stock alone recovers 1.77 pp of the 4.42 pp gap; restoring capacity alone
  recovers 2.23 pp; restoring both recovers 4.04 pp, still 0.38 pp short of week 41 because pick
  and notification also degraded.

The residual 0.38 pp is not noise. It is the part of the gap that no single-factor plan touches, and
it is exactly the size at which a programme announces success while the customer-visible rate stays
below baseline. Declaring the target as "recover 4.4 points" and delivering 4.04 is a 92% delivery
that a customer experiences as still-broken.

### 5.5 Step 4 — check whether the attribution is stable under segmentation

Aggregate attribution can be an artefact of a changing mix. Recompute the shares for the two
largest store clusters separately.

| Segment | wk41 R | wk42 R | capacity share | stock share | pick share | notify share |
|---|---|---|---|---|---|---|
| urban cluster (11 nodes) | 0.9381 | 0.8917 | 55.2% | 36.9% | 5.8% | 2.1% |
| suburban cluster (23 nodes) | 0.9429 | 0.9008 | 46.1% | 44.5% | 7.0% | 2.4% |
| all nodes | 0.9403 | 0.8960 | 51.0% | 40.6% | 6.4% | 2.1% |

The ordering is stable — capacity first in both clusters — but the magnitude is not: capacity is
55% of the urban decline and 46% of the suburban one. A single number presented to a meeting will be
argued about by both cluster leads; two numbers with their populations attached will not. Note the
segments are ILLUSTRATIVE and are included to demonstrate the check, not to assert a real pattern.

### 5.6 Step 5 — sensitivity, and the value at which the recommendation flips

The remediation choice between "fix stock accuracy" and "raise slot capacity" does not depend on
the attribution shares alone; it depends on cost per recovered point and on the confidence interval
around each stage rate. Two sensitivities:

```text
(a) measurement error: each stage rate carries +/- 0.3 pp sampling error at n ~ 18,000.
    The capacity share (50.95%) and stock share (40.56%) differ by 10.4 pp of share, which is
    roughly 3x the propagated error on a share of this size -> the ordering is robust.
(b) cost: if stock-accuracy work costs 1.0 unit and recovers 1.77 pp, while capacity work costs
    2.4 units and recovers 2.23 pp, then cost per recovered point is 0.565 vs 1.076 units.
    The cheaper programme is stock, even though it explains less of the decline.
INTERPRETATION: attribution answers "where did the gap come from"; cost per recovered point
answers "what to do". They are different questions and the answers can point at different stages.
```

This is the single most transferable result in the file: **the stage that explains the most of a
regression is often not the stage that is cheapest to fix, and a remediation plan justified only by
attribution shares will systematically over-invest in the largest factor.** The correct decision rule
ranks by recovered points per unit of cost, subject to a floor on each stage so that no factor is
left in a degraded state.

### 5.7 Independent re-check of the whole example

A second path that shares no intermediate value with the first: multiply the four week-42
reliabilities in a different order and in fractional form.

```text
0.966 x 0.992 = 0.958272
0.971 x 0.963 = 0.935073
0.958272 x 0.935073 = 0.896054   (matches P-01 to 6 dp)
16,487 / 18,400 = 0.896033       (count-based, 0.000021 lower from rounding the count)
```

Both paths agree to five significant figures. The residual disagreement is the rounding of
`16,487.4` to `16,487`, and it is stated rather than smoothed away — a review that claims exact
agreement between a model product and a rounded count is hiding an arithmetic step.

## 6. Visual / Geometric Intuition

### 6.1 The reliability surface

Picture the four stage rates as axes, each in `[0,1]`. Because `R` is their product, the level sets
`R = c` are curved surfaces bowing toward the origin: near `(1,1,1,1)` a small move along any axis
costs little; near `(0.7,0.7,0.7,0.7)` the same move costs much more. Concretely, dropping one stage
from 0.99 to 0.98 costs 1.0% of `R` when the rest are 0.99, but costs 1.4% of `R` when the rest are
0.70 — because the loss is now measured against a smaller base. This curvature is why a node that is
already degraded needs a smaller absolute improvement per stage to gain the same points, and why
improvement programmes at good nodes report small numbers while reporting large ones at bad nodes.

### 6.2 The queue as a bathtub

Queue depth is water level, arrivals are the tap, service rate is the drain. The important geometric
fact is that the *level* responds to the *difference* between tap and drain, integrated over time —
so a queue that has been growing for three hours cannot be judged by its current depth. A depth of
400 with `mu - lambda = 50/hour` is a worse situation than a depth of 600 with `mu - lambda = 200/hour`,
because the first will take 8 hours to clear and the second 3. Two numbers, always: level and
slope. A dashboard that shows only depth is a dashboard that will always be read as "it is big"
rather than "it is closing" or "it is opening".

### 6.3 The identity block graph

Records are vertices; candidate pairs within a block are edges. A block of `m` records has
`m(m-1)/2` edges, which grows quadratically. At `m=40` there are 780 pairs; at `m=120` there are
7,140 — 9.15 times as many edges for 3 times as many records. This is the geometry behind
SCALE_FAILURE in identity resolution: the cost per run is governed by the largest block, not by the
number of records, so a single popular name or a shared household address can dominate the run time
and the error budget simultaneously.

```text
m     pairs     pairs/m     note
40      780      19.5      typical block, cheap
80    3,160      39.5      double the records, four times the pairs
120   7,140      59.5      one household or one common-name cluster
200  19,900      99.5      a single block that can exceed a run's time budget
```

### 6.4 The three-clock diagram

Draw three horizontal timelines, one per clock, and mark the same fact on each. The defect region is
the vertical strip between the decision instant and the processing instant, restricted to facts whose
event instant is before the decision. Visually, the strip is narrow for fast data and wide for batch
data — which is why the same policy of "use what you have" is safe for order events and dangerous for
inventory snapshots that arrive daily.

### 6.5 Where the visual intuition fails

All four pictures assume the subsystem is stationary over the window. In retail that is the exception:
promotions, weather, and slot boundaries make the rates non-stationary at a scale of hours. Use the
pictures to choose a test, not to estimate a value; the estimate must come from a frozen extract with
a stated window.

## 7. Comparison / Selection Table

### 7.1 Which diagnostic to run first

| Symptom as reported | First check | Why not the obvious alternative | Expected signal if this is the cause |
|---|---|---|---|
| promise rate fell, stage rates all slightly down | log-share decomposition (P-02) at matched grain | comparing stage means hides which factor dominates | one factor holds >40% of the log change |
| promise rate fell, one stage flat | segment that stage by node and shift | an aggregate flat rate can hide two offsetting tails | worst decile far below the mean |
| promise rate fell after a definition change | recompute both weeks under one definition | the change may be pure definition, no behaviour | the gap closes under the old definition |
| slots report full but picks are idle | compare released demand with P-06 capacity | idle pickers with full slots is a release defect | released > feasible by a stable ratio |
| queue deep but not growing | compute `mu - lambda` and drain time (P-07) | depth alone cannot distinguish clearing from opening | `mu > lambda` and the ETA is credible |
| customers say "no notification" but logs show sends | join notification attempts to delivery outcomes | "sent" is not "delivered" and neither is "read" | a delivery-status gap by carrier or channel |
| points balance disagrees with visible transactions | recompute from the transaction ledger only | the balance may include expired or reversed entries | the ledger reconciles, the balance does not |
| two dashboards disagree on one headline | compare denominators, not numerators | numerators usually match; populations differ | the eligible sets differ by a stated rule |

### 7.2 Containment action selection

| Situation | Containment | Reversal condition | Never do |
|---|---|---|---|
| inventory feed late by hours | switch the promise rule to the conservative variant | feed lag under the SLA for two consecutive windows | silently restate yesterday's promises |
| slot oversold | cap releases at `C_slot` and offer the next slot | queue drains below the cap for four hours | accept the oversell and hope picks catch up |
| identity merges suspected | freeze automated merges, route to manual review | false-merge rate under the review threshold on a replay | unmerge automatically without an audit trail |
| points posting wrong | pause posting, keep accrual recorded | ledger replay matches the balance for a closed cohort | adjust balances by a plug figure |
| notification failure | add a fallback channel for the affected cohort | delivery rate restored on both channels | re-send to everyone and double the contact |
| consent state unclear | suppress the cohort for the channel | state confirmed with a timestamp per customer | contact and assume the record is right |

### 7.3 Choosing between correction strategies

| Strategy | When it is right | Cost shape | Its characteristic failure |
|---|---|---|---|
| fix the comparator | boundary or rounding errors | one-time code change | the fix shifts the error to a new boundary |
| version the contract | source meaning or availability changed | coordination across teams | the old version lingers in one consumer |
| add capacity | genuinely volume-driven and stationary | recurring operating cost | capacity is added to a release defect |
| change the definition | two teams computed different quantities | documentation and fixtures | the old number stays in a deck and gets compared |
| replay and restate | a metric was wrong and no decision depended on it | compute plus review | a decision did depend on it and nobody checked |
| do nothing but annotate | the effect is inside noise and no action changes | near zero | the annotation is forgotten and the number is reused |

### 7.4 The selection rule in one line

Choose the cheapest action that separates the two live hypotheses; commit to a permanent correction
only after a replay has shown the hypothesis is the mechanism, and choose between permanent
corrections by recovered points per unit cost, not by share of the regression.
