## 4. Core Formulas with Full Derivation

Section 4 is deliberately arithmetic-first. Each result is presented in five parts — FORMULA, WHERE, CONSTRAINT, NUMERICAL EXAMPLE, INTERPRETATION — and closes with an explicit validity condition. If you cannot state the validity condition, you do not yet own the result.

### 4.1 Base case

**Result 1 — posterior precision from a confusion matrix (Bayes).**

```text
FORMULA
  PPV = (pi * TPR) / (pi * TPR + (1 - pi) * FPR)

WHERE
  pi  = prevalence of fraud among eligible events, dimensionless, 0 < pi < 1
  TPR = P(alert | fraud), dimensionless, 0..1        (recall, sensitivity)
  FPR = P(alert | legitimate), dimensionless, 0..1   (1 - specificity)
  PPV = P(fraud | alert), dimensionless, 0..1        (precision)

CONSTRAINT
  The three inputs must be measured on the same eligible cohort and the same
  observation window. Mixing a 90-day TPR with a 120-day FPR breaks the identity.

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  pi = 0.004, TPR = 0.55, FPR = 0.01
  numerator   = 0.004 * 0.55          = 0.002200
  denominator = 0.002200 + 0.996*0.010 = 0.002200 + 0.009960 = 0.012160
  PPV         = 0.002200 / 0.012160    = 0.18095
  lift        = 0.18095 / 0.004        = 45.2

INTERPRETATION
  A model that catches 55% of fraud while flagging 1% of legitimate traffic still
  produces a queue that is 82% false. The false positives number 9,960 per million
  legitimate events against 2,200 per million frauds; the ratio, not the rate, sets
  precision. Precision is a property of prevalence as much as of the model.
```

Validity condition: the identity holds exactly for any cohort with a well-defined TPR and FPR, but it is only *useful* when FPR is estimated on at least a few hundred false alerts, because FPR's own sampling error propagates multiplicatively into PPV.

**Result 2 — expected value of a single alert.**

```text
FORMULA
  EV(p) = p * rho * L - c_a - (1 - p) * c_f

WHERE
  p    = calibrated P(fraud | alert), dimensionless
  rho  = recovery rate given timely action, 0..1
  L    = loss given undetected fraud, currency per fraud event
  c_a  = review cost per alert, currency
  c_f  = friction cost per false alert, currency
  EV   = expected net value per alert, currency

CONSTRAINT
  p must be a probability, not a rank or margin. If p is monotone-but-uncalibrated,
  EV(p) is a monotone transform of nothing economic; only its ordering survives.

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  p = 0.24, rho = 0.62, L = 320, c_a = 14.00, c_f = 2.10
  benefit  = 0.24 * 0.62 * 320 = 47.616
  review   = 14.000
  friction = 0.76 * 2.10       =  1.596
  EV       = 47.616 - 14.000 - 1.596 = 32.02 per alert

INTERPRETATION
  At 24% precision an alert is worth 32 currency units in expectation. Multiply by
  8,800 alerts and you get 281,776 per week — the same figure the whole file builds
  toward. Note the friction term is only 5% of the benefit here; in low-value
  segments it dominates and flips the sign (see Section 8, failure 3).
```

Validity condition: EV is linear in p, which requires the benefit and cost terms to be independent of p. If high-risk alerts also take longer to review — true in practice — the linear form overstates value at the top of the queue.

**Result 3 — break-even probability.**

```text
FORMULA
  p* = (c_a + c_f) / (rho * L + c_f)

DERIVATION
  Set EV(p) = 0 and solve:
    p*rho*L - c_a - (1 - p)c_f = 0
    p*(rho*L + c_f) = c_a + c_f
    p* = (c_a + c_f) / (rho*L + c_f)

WHERE
  all symbols as in Result 2; p* in 0..1, currency units cancel

CONSTRAINT
  Requires rho*L + c_f > 0, always true for positive economics. Requires the
  friction to be paid on false alerts only; if friction also accrues to true
  positives (customer contacted, card replaced) the denominator changes.

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  c_a = 14.00, c_f = 2.10, rho = 0.62, L = 320
  rho*L + c_f = 198.40 + 2.10 = 200.50
  p* = 16.10 / 200.50 = 0.08030

  Check by substitution at p = 0.0803:
    0.0803 * 198.40 = 15.931
    15.931 - 14.00 - 0.9197*2.10 = 15.931 - 14.00 - 1.931 = 0.000  (rounding)

INTERPRETATION
  Only alerts with a calibrated probability above 8.03% are worth investigating.
  Anything below should be closed without review — not because the model is wrong,
  but because the review costs more than the expected recovery. The threshold is a
  business number that happens to be expressed in probability units.
```

Validity condition: p* assumes a binary review decision with a fixed review cost. With tiered actions (Result 7) the single threshold is replaced by a set of thresholds, one per action.

**Result 4 — net value of a capacity-limited policy.**

```text
FORMULA
  V(a) = TP(a) * rho * L - a * c_a - (a - TP(a)) * c_f
  TP(a) = a * precision(a) = a * PPV at the a-th ranked alert

WHERE
  a      = alerts worked per period (capacity used), integer >= 0
  TP(a)  = true positives among them
  others as above

CONSTRAINT
  V(a) is defined only along the model's own ranking; comparing V across models
  requires each model to be evaluated at its own best a, not at a common a.

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  a = 8,800, TP = 2,112, rho = 0.62, L = 320
  benefit  = 2,112 * 198.40        = 419,020.8
  review   = 8,800 * 14.00         = 123,200.0
  friction = (8,800 - 2,112) * 2.10 = 14,044.8
  V        = 419,020.8 - 123,200.0 - 14,044.8 = 281,776.0

INTERPRETATION
  V is the objective. Precision and recall are inputs to it. A model with better
  precision at the same a always improves V by the difference in TP times 200.50,
  minus the friction saved at 2.10 per displaced false alert.
```

Validity condition: V assumes independent alerts. When 6% of alerts are duplicates of the same underlying case (Section 8, failure 4), the benefit is overcounted because one fraud is credited multiple times while its review cost is paid multiple times.

### 4.2 General case

The general case replaces "one threshold" with "one capacity plus one ranking", which is what operations actually constrains.

**Result 5 — marginal band value and the optimal capacity.**

```text
FORMULA
  For a band of alerts between cumulative counts a1 and a2 with true positives
  TP1 and TP2, define the band marginal precision:
      mp = (TP2 - TP1) / (a2 - a1)
  The band contributes:
      V_band = (a2 - a1) * (mp * rho * L - c_m - (1 - mp) * c_f)
  Optimal capacity a* is the largest a such that V_band > 0 for the marginal band.

WHERE
  c_m = marginal review cost (variable portion), currency per alert
  mp  in 0..1; bands are contiguous segments of the ranked alert list

CONSTRAINT
  Requires c_m <= c_a. Using average cost in the marginal formula systematically
  under-expands capacity, because it charges fixed supervision to each extra alert.

NUMERICAL EXAMPLE  (ILLUSTRATIVE, cumulative frontier)
  a=8,800  TP=2,112  precision 0.2400   V=281,776
  a=11,000 TP=2,450  precision 0.2227   V=314,125
  a=14,000 TP=2,800  precision 0.2000   V=336,000
  a=17,600 TP=3,200  precision 0.1818   V=358,240
  a=21,000 TP=3,360  precision 0.1600   V=335,580

  Band 8,800 -> 11,000: n=2,200, dTP=338, mp=0.15364
     0.15364*198.40 = 30.482
     - 11.07 - 0.84636*2.10 = -11.07 - 1.777 = 30.482 - 12.847 = 17.63 > 0
     band value = 2,200 * 17.63 = 38,786  (gross of the fixed cost per alert)
     cumulative difference 314,125 - 281,776 = 32,349 is net of that fixed
     cost: 2,200 * 2.93 = 6,446, and 38,786 - 6,446 = 32,340 (9 units of
     rounding in the displayed precision). The two differ by a known amount,
     not by an error.

  Band 17,600 -> 21,000: n=3,400, dTP=160, mp=0.04706
     0.04706*198.40 = 9.337
     - 11.07 - 0.95294*2.10 = 9.337 - 13.071 = -3.73 < 0

INTERPRETATION
  Cumulative value peaks at 17,600 alerts and declines after. The band arithmetic
  explains why: beyond 17,600, each extra alert brings 4.7% precision, worth 9.34,
  against 13.07 of marginal cost. Expanding from 8,800 to 11,000 is worth about
  17.63 per alert; expanding from 21,000 to 26,000 destroys about 1.94 each.
  A programme that reports only cumulative precision would expand past the peak.
```

Reconciliation note: the band sum (2,200 × 17.63 = 38,786) exceeds the cumulative difference (314,125 − 281,776 = 32,349) by 6,437 because the two use different cost bases. The cumulative figure charges the full average cost of 14.00 to each alert; the band formula charges the marginal 11.07 and excludes the fixed 2.93 that is already paid. The gap is exactly the fixed component: 2,200 × 2.93 = 6,446, with the residual 9 units of rounding in the displayed precision. Both numbers are correct answers to different questions — "what did this tranche cost to add?" versus "what does the total queue cost?" — and mixing them in one table is the error.

Validity condition: the band test assumes bands are processed in rank order and that capacity is divisible. With hiring in whole FTEs (195.65 alerts per FTE-week) the practical a* is the largest multiple of the staffing quantum below the true optimum.

**Result 6 — Lagrangian form: one ranking, many action costs.**

```text
FORMULA
  Choose the cut tau to maximise  sum_i [ p_i * rho * L - c_a - (1-p_i)*c_f ] * 1[alert_i]
  subject to                     sum_i 1[alert_i] <= K
  Lagrangian:  alert i iff  p_i * (rho*L + c_f) > c_a + c_f + lambda
  giving the capacity-adjusted threshold
      p*(lambda) = (c_a + c_f + lambda) / (rho*L + c_f)

WHERE
  K        = capacity in alerts per period
  lambda   = shadow price of capacity, currency per alert, >= 0
  p*(lambda) rises monotonically in lambda

CONSTRAINT
  Requires the constraint to bind. If K is generous relative to the score
  distribution, lambda = 0 and the unconstrained threshold p* applies.

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  Unconstrained p* = 0.0803. At a = 8,800 the observed cut is 0.081, so the
  implied lambda is small:
      lambda = 0.081 * 200.50 - 16.10 = 16.24 - 16.10 = 0.14 per alert
  If capacity were halved to 4,400, the cut would move up to roughly 0.145 on
  this ranking, and lambda would rise to 0.145*200.50 - 16.10 = 12.97.

INTERPRETATION
  The shadow price is a useful management number: it says what one more unit of
  reviewer capacity is worth at the current margin. Here 0.14 per alert means the
  queue is almost exactly right-sized — capacity is marginally binding, not
  painfully so. A lambda of 13 would justify hiring.
```

Validity condition: the Lagrangian is exact for a single linear constraint and separable objective; it is approximate when review cost depends on alert complexity (see Result 7), because then the objective is no longer separable in the same way.

**Result 7 — tiered actions.**

```text
FORMULA
  For actions k = 1..K with cost c_k and avoided-loss fraction rho_k, choose for
  alert i the action maximising
      argmax_k  [ p_i * rho_k * L - c_k - (1 - p_i) * c_f,k ]
  and choose "no action" (k=0, all terms zero) if every k gives a negative value.

WHERE
  rho_k is non-decreasing in k, c_k and c_f,k are non-decreasing in k
  boundaries between actions occur where two value lines cross:
      p_k = (c_{k+1} - c_k) / (L*(rho_{k+1} - rho_k))

CONSTRAINT
  Requires the action set to be ordered by intensity. Unordered actions
  (block vs. contact) can have crossing lines twice and need explicit enumeration.

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  Action A: auto-decline, c=0.30, rho=0.20, c_f=1.10
  Action B: analyst review, c=14.00, rho=0.62, c_f=2.10
  Action C: review + customer contact, c=26.50, rho=0.78, c_f=6.40

  A vs B boundary: (14.00-0.30) / (320*(0.62-0.20)) = 13.70 / 134.40 = 0.1019
  B vs C boundary: (26.50-14.00) / (320*(0.78-0.62)) = 12.50 / 51.20 = 0.2441

  So: p < 0.0215 -> no action; 0.0215..0.1019 -> auto-decline;
      0.1019..0.2441 -> analyst review; p > 0.2441 -> review plus contact.
  (0.0215 is p* for action A alone: (0.30+1.10)/(0.20*320+1.10) = 1.40/65.10.)

INTERPRETATION
  Tiering raises value in both tails: the very low band is handled by automation
  at 0.30 instead of 14.00, and the very high band buys the extra recovery that
  only customer contact achieves. The queue is then sized on the middle band only,
  which is a fraction of total alerts — the capacity problem shrinks without
  losing coverage.
```

Validity condition: tier boundaries are computed from mean L. With heavy-tailed L the crossing points shift per alert, and a segment-level threshold is the practical compromise.

### 4.3 Variants

**Variant A — average versus marginal cost in the threshold.**
With average cost 14.00 the threshold is 0.0803; with marginal cost 11.07 it is 0.0657.

```text
  p*_avg = (14.00 + 2.10) / 200.50 = 0.08030
  p*_mrg = (11.07 + 2.10) / 200.50 = 0.06569
  logit difference = ln(0.0803/0.9197) - ln(0.06569/0.93431)
                   = -2.4387 - (-2.6552) = +0.2165
```

A logit shift of 0.2165 is small in probability terms at the bottom of the scale and large at the top: at p = 0.05 it moves the score by about 0.010 in probability, at p = 0.60 by about 0.043.
The practical rule: use p*_avg when deciding whether the *function* should exist, p*_mrg when deciding whether to work *one more* alert this week.

**Variant B — expected-loss minimisation instead of value maximisation.**
Minimising expected loss per event gives the same threshold only under a symmetric-accounting assumption: loss if missed = rho L, loss if falsely flagged = c_f plus the review cost allocated to that alert. The two formulations coincide when review cost is charged per alert regardless of outcome. When review cost is charged per *hour* and fraud alerts take longer, the value form is the correct one and the loss form silently under-weights high-risk alerts.

**Variant C — precision at a fixed alert rate instead of a fixed threshold.**
If the queue is capped, express everything as a function of alert rate r: precision(r), recall(r), V(r). The threshold then floats with score distribution drift, which is usually the desired behaviour — the queue stays constant and the model quality shows up as varying precision. The failure mode is that a drifting threshold hides model degradation: precision stays flat because the threshold silently loosens. Always publish both the threshold and the alert rate.

**Variant D — cost-weighted recall.**
For regulatory coverage targets the objective is often "catch at least X% of value", not "maximise net value". Value-weighted recall uses L_i per event rather than a count:

```text
  R_value(a) = sum_{i in caught, ranked <= a} L_i / sum_{all frauds} L_i
```

With a heavy tail, R_value can be much lower than count recall at the same a — a programme reporting 26.4% count recall may be catching 18% of value if the largest cases sit below the cut. Section 8, failure 5 develops this.

**Variant E — threshold on the logit for model portability.**
Thresholds on probability are not portable across prevalence shifts; thresholds on the logit are portable *if* the shift is a constant offset. If a new geography has half the prevalence, the logit shifts by ln(0.002/0.998) - ln(0.004/0.996) = -6.215 - (-5.517) = -0.698. Recomputing the operating point means shifting the logit threshold by +0.698, not halving the probability.

```text
  p1 = 0.004  -> logit = -5.5175
  p2 = 0.002  -> logit = -6.2146
  offset = -0.6971
```

Applying a naive "half the threshold" rule (0.0803 -> 0.0402) instead of the logit shift (which gives 0.0403 at the same relative economics) happens to be close here because both p are small; at p above 0.2 the two rules diverge sharply and the naive rule is wrong.

### 4.4 Limits and counterexamples

**Limit 1 — heavy-tailed loss breaks the mean.**
If L is lognormal with median 90 and 1% of cases above 3,000, the mean may be 320 while the median case is worth far less than the threshold implies. Then p* computed from mean L under-alerts on the tail and over-alerts on the body: a case at median loss has true p* of (16.10)/(0.62*90+2.10) = 16.10/57.90 = 0.278, not 0.0803.

```text
  Counterexample: two alerts, both p = 0.12
    Alert X: L = 90    EV = 0.12*0.62*90  - 14.00 - 0.88*2.10 = -9.15
    Alert Y: L = 3,000 EV = 0.12*0.62*3000 - 14.00 - 0.88*2.10 = 207.34
  Mean-L arithmetic says both are worth 0.12*198.40 - 16.10 = +7.71, which is
  wrong for X by 16.86 and wrong for Y by 199.63.
```

The correction is to segment by expected loss band, or to use a per-alert EV and rank by EV rather than by p. Ranking by p when L varies is a known and common error.

**Limit 2 — prevalence non-transportability.**
PPV is not a property of the model. Using TPR = 0.264 and FPR = 0.003357 (the illustrative operating point) on populations with different prevalence:

```text
  pi = 0.0005 -> PPV 0.0378, EV  -8.51 per alert
  pi = 0.0010 -> PPV 0.0730, EV  -1.47
  pi = 0.0020 -> PPV 0.1361, EV +11.19
  pi = 0.0040 -> PPV 0.2400, EV +32.02
  pi = 0.00667-> PPV 0.3455, EV +53.18
  pi = 0.0200 -> PPV 0.6161, EV +107.42
```

At pi = 0.001 the same model, same threshold, same reviewers produce a *negative*-value queue. A model validated on one channel and deployed on another without re-deriving p* is not a validated model.

**Limit 3 — calibration is not monotone-invariant, discrimination is.**
Any strictly increasing transform of the score leaves AUC and PR-AUC unchanged while changing every probability-based threshold. Therefore: an AUC regression test cannot catch a calibration break, and a calibration test cannot catch a ranking break. A monitoring plan with only one of the two is blind to half the failure surface.

**Limit 4 — the break-even threshold is undefined when friction is paid on true positives.**
If every alert triggers a customer contact costing 2.10 regardless of outcome, then c_f applies to all alerts and cancels from both sides:

```text
  EV = p*rho*L - c_a - c_f
  p* = (c_a + c_f) / (rho*L) = 16.10 / 198.40 = 0.0811
```

versus 0.0803 in the base case — a 1% shift. Small, but the *structure* changes: the threshold no longer depends on c_f at all, and the sensitivity analysis must be redone. The general lesson: write down which costs attach to which outcome before differentiating.

**Limit 5 — capacity constraints make the objective non-separable across segments.**
If a large case takes 3 hours and a small one 5 minutes, a single alert-count capacity is wrong. Replace it with minutes: capacity 1,349.3 minutes per week becomes a knapsack, and the greedy value-per-minute ordering (not value-per-alert) is optimal only under the fractional relaxation. The integer version is NP-hard and in practice solved by value-per-minute sorting with a correction for the last few large cases.

```text
  value per minute:  EV_i / minutes_i
  small case: 32.02 / 5.0  = 6.40 per minute
  large case: 240.00 / 180 = 1.33 per minute
  A thousand small alerts beat sixty large ones on the same clock, even though
  the large ones carry more value apiece. Capacity in alerts hides this.
```

The counterexample that breaks value-per-minute: when a large case's loss is recoverable only if worked *today*, its option value exceeds the static EV, and strict value-per-minute ordering liquidates the tail. Treat time-critical cases as a separate queue with reserved capacity.
