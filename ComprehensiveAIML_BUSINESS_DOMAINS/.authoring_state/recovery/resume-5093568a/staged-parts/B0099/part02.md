## 4. Core Formulas with Full Derivation

This section establishes the five quantities the catalog diagnoses: the compound promise rate and
its attribution, the points liability and its drift, the identity false-merge rate and its precision
consequence, slot feasibility, and the queue drain time. Each derivation uses the same five-part
anatomy: the formula, the domain of the symbols, the constraints that must hold for the formula to
mean anything, a numerical example with recomputed arithmetic, and the interpretation — including
what the number does not license you to conclude.

### 4.1 Base case: compound promise reliability

**Formula.** Start from the definition rather than the formula. An order is promise-kept only if
every stage it traverses succeeds. If the stages were independent, the probability that all succeed
is the product of the stage pass rates. The derivation is one line of probability: for independent
events, `P(A ∩ B) = P(A)·P(B)`, extended by induction to `k` stages.

```text
FORMULA [P-01]: R = product over i of s_i ,  i = 1..k
WHERE:
  R   [fraction, 0..1]  compound probability that one order is promise-kept
  s_i [fraction, 0..1]  pass rate of stage i, measured on units entering stage i in the same window
  k   [integer, >=1]    number of stages the order must traverse
CONSTRAINT: stages conditionally independent given the cohort; each s_i measured at the same grain
            (order, node, week) and over the same population; no stage counted twice.
NUMERICAL EXAMPLE:
  s = (0.963, 0.971, 0.992, 0.966)
  R = 0.963 x 0.971 x 0.992 x 0.966
    = 0.963 x 0.971 = 0.935073
    = 0.935073 x 0.992 = 0.927592
    = 0.927592 x 0.966 = 0.896054
INTERPRETATION: about 89.6% of orders survive all four stages. The worst single stage is 96.3%,
so no individual stage looks alarming while four of them together lose one order in ten.
```

Two things about this number are worth stating plainly. First, it is much lower than any stage, and
that gap is the whole reason a dashboard of green stage metrics can coexist with a red customer
metric. Second, the independence assumption is the fragile part. Real stages are positively
correlated — a node that is short-staffed picks slowly and also misses notifications — so the
product understates the true rate of multi-stage failures and overstates the rate of exactly-one-
stage failures. The product is a good central estimate and a bad tail estimate.

**Where the base case stops being informative.** When a single stage's pass rate is below roughly
0.90, the product is dominated by that stage and the decomposition adds nothing you could not see
from the stage itself. When `k` grows — say a ship-from-store path with five or six hops — the
product becomes so sensitive that a 0.1 percentage-point measurement error in one stage moves the
headline by the same 0.1 points, and comparisons across weeks need confidence intervals rather than
point estimates.

### 4.2 General case: multiplicative attribution of a change

The diagnostic question is never "what is R"; it is "which factor moved". Because the product is
multiplicative, the log is additive, and additive shares are the correct way to attribute a change
in R across factors. Using absolute percentage-point differences instead is the most common
arithmetic error in a fulfilment post-mortem.

```text
FORMULA [P-02]: ln(R2/R1) = sum over i of ln(s_i2 / s_i1)
                share_i   = ln(s_i2/s_i1) / ln(R2/R1)
WHERE:
  R1, R2      compound rate in window 1 and window 2
  s_i1, s_i2  stage i pass rate in the same two windows
  share_i     [fraction, can be negative] additive contribution of stage i to the log change
CONSTRAINT: identical stage set, grain, and eligibility in both windows; if a stage is added or
            removed, the comparison is between different systems and P-02 does not apply.
NUMERICAL EXAMPLE (ILLUSTRATIVE):
  window 1 s = (0.982, 0.974, 0.993, 0.990)   R1 = 0.940275
  window 2 s = (0.963, 0.971, 0.992, 0.966)   R2 = 0.896054
  total log change = ln(0.896054 / 0.940275) = ln(0.952970) = -0.048171
  stock     ln(0.963/0.982) = ln(0.980652) = -0.019538  -> 40.56%
  pick      ln(0.971/0.974) = ln(0.996920) = -0.003085  ->  6.40%
  notify    ln(0.992/0.993) = ln(0.998993) = -0.001008  ->  2.09%
  capacity  ln(0.966/0.990) = ln(0.975758) = -0.024541  -> 50.95%
  check: -0.019538 - 0.003085 - 0.001008 - 0.024541 = -0.048172  (rounding, 1e-6)
INTERPRETATION: the capacity stage carries 50.95% of the log decline and stock carries 40.56%;
pick and notification together carry 8.5%. A remediation that fixes only stock moves R from
0.896 to 0.914; only capacity, to 0.918.
```

```text
FORMULA [P-03]: R_counterfactual = R2 x (s*_i / s_i2)
WHERE: s*_i is the restored pass rate for factor i, all other factors held at window-2 values.
CONSTRAINT: one factor at a time; combined restorations are not additive in R and must be recomputed.
NUMERICAL EXAMPLE (continuing the case above):
  restore stock only:     R2 x (0.982/0.963) = 0.896054 x 1.019730 = 0.913733  (+1.768 pp)
  restore capacity only:  R2 x (0.990/0.966) = 0.896054 x 1.024845 = 0.918316  (+2.226 pp)
  restore both:           0.896054 x 1.019730 x 1.024845      = 0.936435  (+4.038 pp)
  sum of the two single-factor gains = 1.768 + 2.226 = 3.994 pp, 0.044 pp less than combined
INTERPRETATION: the interaction term is small here (0.044 pp) because both restorations are
modest. It is not small in general: two factors each restored from 0.60 to 0.90 produce a
combined gain of 125% in R, not the 100% implied by adding the two single-factor gains.
```

The last line matters because remediation plans are usually written as a list of single-factor
gains and then budgeted as if they summed. At moderate degradations the error is a rounding
artefact; at severe degradations it is large, and it is always in the optimistic direction —
the plan promises more than the sum of its parts delivers.

**Boundary and counterexample for P-02.** Suppose the decline in R came with a *rise* in one stage:
stock fell from 0.982 to 0.963 while notification rose from 0.993 to 0.999. Then the notification
share is negative, and saying "notification contributed −2% of the decline" is arithmetically true
and rhetorically useless. The correct statement is that notification improved and partially offset
the decline; report offsets separately with a sign, and never clip them to zero. Clipping makes the
shares sum above 100% and destroys the audit trail that the log decomposition exists to provide.

### 4.3 Variants: rate pooling, rounding, and slot feasibility

**Variant A — pooling rates across populations of unequal size.** Averages of rates are only
meaningful when denominators match. The failure is common enough to state as a formula with a
warning attached.

```text
FORMULA [P-04]: r_pooled = sum_j (n_j . r_j) / sum_j n_j
WHERE: n_j [count] units in stratum j; r_j [fraction] rate in stratum j.
CONSTRAINT: strata must partition the population; if they overlap, the denominator double-counts.
NUMERICAL EXAMPLE (ILLUSTRATIVE):
  stratum A: n=200,  r=0.90   -> 180 favourable
  stratum B: n=5000, r=0.80   -> 4000 favourable
  pooled    = (180 + 4000) / 5200 = 4180 / 5200 = 0.803846
  unweighted mean of the two rates = (0.90 + 0.80)/2 = 0.850000
  difference = 4.62 pp, entirely an artefact of which formula was used
INTERPRETATION: reporting 85.0% when the true pooled rate is 80.4% overstates performance by
4.6 points. The unweighted mean answers "how do the average store's numbers look", which is a
different question and must be labelled as such.
```

**Variant B — rounding a sum versus summing rounded parts.** Loyalty balances are frequently
displayed in whole points while accrual is fractional. The two operations commute only when every
part is an integer.

```text
FORMULA [P-05]: round(sum_j x_j) - sum_j round(x_j)  = rounding discrepancy, per basket
WHERE: x_j [points, real] accrual for line j; round() is floor for display truncation.
NUMERICAL EXAMPLE (ILLUSTRATIVE), five lines accruing 4.35, 2.99, 12.40, 0.89, 7.55 points:
  sum = 28.18  -> floor 28
  sum of floors = 4 + 2 + 12 + 0 + 7 = 25
  discrepancy = 3 points per basket, always in the customer's favour here, never necessarily so
  at 6.1 million baskets per month the discrepancy is up to 18.3 million points of display drift
INTERPRETATION: the fix is a single rule — compute in the smallest unit (points x 100), round once
at the boundary — not a reconciliation job. Two rounding sites in one flow guarantee a permanent
unexplained difference that grows with basket size.
```

**Variant C — slot feasibility under a labour constraint.** A slot's capacity is not a number
someone chose; it is what the node's pickers can actually complete in the slot, discounted for
variability.

```text
FORMULA [P-06]: C_slot = n_p . u . h_slot . eta
WHERE:
  C_slot [orders]        feasible orders in the slot
  n_p    [pickers]       pickers rostered for the slot
  u      [orders/picker-hour] sustained completed-orders rate per picker
  h_slot [hours]         slot duration
  eta    [fraction]      utilisation discount for breaks, replenishment, and variability
CONSTRAINT: u measured on the same assortment mix and node layout, and converted from line rate by
            dividing by mean lines per order; eta is an operating decision that must be stated,
            never a hidden fudge factor.
NUMERICAL EXAMPLE (ILLUSTRATIVE):
  n_p=12, u=22 orders/picker-hour, h_slot=4.0, eta=0.85
  C_slot = 12 x 22 x 4.0 x 0.85 = 1056 x 0.85 = 897.6 -> 897 orders feasible
  cross-check in line terms: 22 orders/h x 3.4 lines/order = 74.8 lines/picker-hour; over 40.8
  effective picker-hours that is 3,051 lines = 897 orders at the same 3.4 lines per order.
  demand released into the slot = 1120 orders
  oversell = 1120 - 897 = 223 orders, 19.9% of the slot
INTERPRETATION: the slot is oversold by design or by accident, and no amount of picker effort
recovers it. Whether the cause is a capacity error or a release error is settled by checking
which of the two numbers changed.
```

**Variant D — queue drain time.** The number that decides whether containment is working.

```text
FORMULA [P-07]: T_drain = Q / (mu - lambda)   for mu > lambda
WHERE:
  Q      [orders]      queue depth at the observation instant
  mu     [orders/hour] net service rate of the constrained resource
  lambda [orders/hour] arrival rate during the drain, held constant
  T_drain[hours]       time to empty, ignored if mu <= lambda (the queue diverges)
NUMERICAL EXAMPLE (ILLUSTRATIVE):
  Q=500 orders, mu=300/hour, lambda=250/hour
  T_drain = 500 / (300 - 250) = 500/50 = 10.0 hours
  at lambda=264 (a 36/hour backlog) T_drain = 500/36 = 13.9 hours
INTERPRETATION: a 14-order-per-hour arrival increase stretches the drain from 10 to 13.9 hours.
The queue depth tells you the size of the problem; the difference mu - lambda tells you whether
the problem is closing.
```

### 4.4 Limits and counterexamples

**Limit 1 — independent-stage product with correlated stages.** If two stages share a cause (one
node, one roster, one carrier), `P(A∩B) > P(A)P(B)`. The product then understates joint success and
overstates joint failure. Discriminating check: compute the observed joint pass rate from order-level
data and compare with the product. A material positive gap means the stages share a driver, and the
attribution in P-02 must be run at the shared-driver level instead of the stage level.

**Limit 2 — log attribution with a factor near zero or one.** Shares are unstable when a factor's
relative change is tiny (division by a near-zero log) or when the total log change is near zero (the
denominator vanishes while numerator terms do not). In both cases report absolute percentage-point
changes and state that shares are undefined, rather than printing a 4,000% share that nobody can act
on.

**Limit 3 — feasibility formula treated as a measurement.** P-06 produces a capacity, not an
observed throughput. If observed throughput is systematically below `C_slot`, the cause is in `u`,
`eta`, or the assortment mix — never "the formula is wrong". Recalibrate the inputs and record the
calibration date; a capacity number without a calibration date is an opinion.

**Limit 4 — drain time with a non-stationary arrival rate.** P-07 assumes `lambda` constant. In
retail it is not: arrivals spike at slot boundaries and after notifications. Use P-07 for a
directional decision only, and re-evaluate at each hour boundary. A queue predicted to clear in
14 hours that is still growing at hour six has had an arrival-rate change, not a service-rate
change, and the fix belongs upstream.

**Limit 5 — the counterexample that breaks all of the above.** Suppose every stage pass rate is
99% and `k = 6`, giving `R = 0.941`. The compound metric says 5.9% of orders fail, which looks
acceptable until you learn that all failures are concentrated in one node on one shift with one
broken scanner, where the per-node rate is 40%. Aggregation over a heterogeneous population
produces a number that is true of the population and false of every member of it. The defence is
not a better formula; it is a segmentation requirement — every compound rate in this file must be
accompanied by its worst-decile segment, or it cannot be used to decide where to spend.
