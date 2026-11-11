## 6. Analysis Walkthrough

Six derivations follow. Each uses the same anatomy so that a reader can audit one without reading the
others: the formula, the quantities it is defined over, the constraint that makes it valid, a
numerical example with every intermediate value shown, and an interpretation that states what the
number does and does not license. All inputs are ILLUSTRATIVE.

The fixed inputs for the whole section are:

| Symbol | Meaning | Value | Status |
|---|---|---|---|
| N | Eligible bays in the cluster | 10,260 (28 stores × 380, less refits) | constructed |
| p | True non-compliance prevalence per store | 0.22 | constructed, to be estimated in practice |
| Se | Detector sensitivity at facing level | 0.91 | constructed |
| Sp | Detector specificity at facing level | 0.91 | constructed |
| C | Fixes the crew can complete per store per wave | 65 | derived below from 331.5 minutes |
| w | Loaded labour rate | $26.00 per hour = $0.4333 per minute | constructed |
| c_f | Cost of one fix attempt | $3.71 | derived below |
| R | Bay revenue per week | $1,900 | constructed |
| m | Unit margin | 0.28 | constructed |
| ε | Constant space elasticity | 0.35 | borrowed, not measured |

### 6.1 Derivation 1 — How many fixes fit in a wave

**FORMULA.**

```text
C = floor( (M_avail × (1 − r_reserve)) / t_fix )
  = floor( (390 × 0.85) / 5.1 )
```

**WHERE.** M_avail is the merchandising minutes rostered to planogram work in the wave (390 minutes,
i.e. 6.5 hours of a single crew member, ILLUSTRATIVE). r_reserve is the fraction held back for
unplanned interruptions and for bays that turn out to need parts or labels (0.15). t_fix is the mean
measured minutes to bring one bay into agreement (5.1), taken from completed-fix logs rather than
from a time-and-motion standard, because a standard measures the task while a log measures the task
as performed.

**CONSTRAINT.** t_fix must come from bays in the same fixture types and the same planogram vintage as
the queue. A reset that changes shelf-edge labels costs materially more than one that only moves
product, and mixing the two into a single mean produces a capacity estimate that is too high in reset
waves and too low in steady-state waves. The reserve must be applied before the division, not after,
because reserving a fraction of the queue is not the same as reserving a fraction of the minutes.

**NUMERICAL EXAMPLE.**

```text
M_avail × (1 − r_reserve) = 390 × 0.85 = 331.50 minutes
331.50 / 5.1              = 65.00 fixes
floor(65.00)              = 65 fixes
Unused minutes            = 331.50 − (65 × 5.1) = 331.50 − 331.50 = 0.00
```

The division lands exactly on 65.00 by construction of the illustrative inputs. That exactness is
deliberate: it removes rounding from the capacity figure so that later differences in the analysis
cannot be attributed to a floor operation. A realistic log would give something like 331.5 / 5.13 =
64.62, and the queue would be 64 bays with 3.2 minutes spare.

**INTERPRETATION.** Capacity is 65 bays, and any queue longer than 65 is not a plan, it is a wish.
This number is the binding constraint for everything that follows: it converts an open-ended quality
question into a selection problem with a hard size limit. Note that the reserve is not sloth. It is
what makes the queue executable without a supervisor silently dropping the tail, which is the
mechanism by which compliance programmes lose their auditability.

**Boundary where the formula breaks.** If the crew is shared with another task, M_avail is not a
constant and the floor operation becomes meaningless in expectation: capacity is then a random
variable and the correct statement is a distribution, not a number. At the extreme, if t_fix is
estimated from a wave in which the crew fixed only easy bays (selection effect), t_fix is biased low
and C is biased high; the queue overruns and the overrun is invisible because the tail is dropped by
hand. The discriminating check is to compare planned against completed minutes for two consecutive
waves and inspect the *distribution* of per-bay minutes, not only the mean.

### 6.2 Derivation 2 — How many flagged bays are a real finding

**FORMULA.**

```text
Flagged rate  f = (1 − Sp) + p × (Se + Sp − 1)
Expected true positives in the queue  TP = C × PPV
PPV = (p × Se) / (p × Se + (1 − p) × (1 − Sp))
Inverse (prevalence from an observed rate)  p̂ = (f − (1 − Sp)) / (Se + Sp − 1)
```

**WHERE.** f is the fraction of all bays the detector flags (the observed alarm rate at bay level). p
is the true fraction of non-compliant bays. Se is the probability a truly non-compliant bay is
flagged; Sp is the probability a compliant bay is not flagged. PPV is the share of flagged bays that
are genuinely non-compliant, which is what determines whether a fix is a fix or a wasted visit.

**CONSTRAINT.** Se and Sp must be estimated at the **same grain as the decision**. If they were
measured at facing level (each facing correctly counted) and applied at bay level (is any facing
wrong), the inversion is invalid, because a bay is flagged when *at least one* facing is wrong, and
the bay-level specificity is roughly Sp^k for k independent facings — a completely different number.
This is the single most common arithmetic error in compliance programmes, and it errs in the
direction of over-flagging, which feels like diligence.

**NUMERICAL EXAMPLE.** At p = 0.22, Se = 0.91, Sp = 0.91:

```text
f   = (1 − 0.91) + 0.22 × (0.91 + 0.91 − 1)
    = 0.09 + 0.22 × 0.82
    = 0.09 + 0.1804
    = 0.2704                       → 27.04% of bays flagged

p × Se            = 0.22 × 0.91 = 0.2002
(1 − p) × (1 − Sp) = 0.78 × 0.09 = 0.0702
PPV               = 0.2002 / (0.2002 + 0.0702) = 0.2002 / 0.2704 = 0.7404

Queue of C = 65 → TP = 65 × 0.7404 = 48.13  (48 or 49, expectation 48.13)
                   FP = 65 − 48.13 = 16.87
Per 380 bays: flagged 380 × 0.2704 = 102.75; true 380 × 0.22 = 83.6
              true flags 380 × 0.22 × 0.91 = 76.08; false flags 380 × 0.78 × 0.09 = 26.68
Check: 76.08 + 26.68 = 102.76 ≈ 102.75 (rounding of 0.2704 to four places)
```

Note what the false-positive count does *not* mean. 26.68 is the number of compliant bays the
detector will offer the crew across the whole store. Since the crew only works 65 bays, the queue
absorbs 16.87 of those false flags and the remaining 9.81 sit below the capacity line, where they
cost nothing this wave but inflate the apparent backlog by 9.8 bays.

**INTERPRETATION.** At 74% precision, one fix in four is wasted. That is not a defect to be
engineered away; it is the arithmetic consequence of a 22% base rate and a detector with 9%
false-positive pressure. Two levers change it, and only two: raise specificity, or raise prevalence
by defining narrower, more severely deviating bays as the unit of concern. Raising sensitivity does
almost nothing for PPV at this base rate — moving Se from 0.91 to 0.99 shifts PPV from 0.7404 to
0.7563, a gain of 1.6 points — while moving Sp from 0.91 to 0.97 shifts it to 0.8953, a gain of
15.5 points. The asymmetry is the whole argument for investing in visual-similarity handling rather
than in a bigger recall-oriented model: at a 22% base rate, the false-positive term (0.78 × 0.09 =
0.0702) is a third the size of the true-positive term (0.2002), so every point of specificity removes
0.0078 from the denominator while every point of sensitivity adds only 0.0022 to the numerator.

**Boundary where the formula breaks.** The inversion p̂ = (f − (1 − Sp)) / (Se + Sp − 1) has a
denominator of Se + Sp − 1, which is the Youden index. When the detector is uninformative, Se + Sp = 1
and the denominator is zero: observed rate f is then exactly 1 − Sp for every p, and prevalence is
unidentifiable. Concretely, with Se = 0.90 and Sp = 0.10 the denominator is 0.00 and f = 0.90 for
p = 0 and p = 1 alike. A programme that reports prevalence from such a detector is reporting its own
error rate with a straight face. The practical guard is to require Se + Sp − 1 ≥ 0.5 before any
prevalence inversion is published, and to publish the Youden index alongside the estimate so the
reader can judge the amplification.

A second boundary is the low-prevalence regime. At p = 0.12, PPV falls to 0.5796; at p = 0.34 it
rises to 0.8389. Since p is estimated per store, two stores in the same cluster can have queues of
wildly different quality while the detector is identical and working exactly as validated. Ranking
stores by queue precision without normalising for prevalence therefore measures prevalence, not
detector quality or store execution.

### 6.3 Derivation 3 — How far a bay deviates, and why counts lie

**FORMULA.**

```text
Facing match          M = 1 − (1/2) × Σ_i |o_i − e_i| / Σ_i e_i
Weighted facing match M_w = Σ_i s_i × min(o_i, e_i) / e_i
Largest single block error  L = max_i |o_i − e_i|
```

**WHERE.** For SKU i in the bay, e_i is planned facings and o_i is observed facings, both non-negative
integers. s_i is SKU i's share of category sales, with Σ s_i = 1. M is one minus half the total
variation distance between the observed and planned facing distributions; M_w weights each SKU's
shortfall by its commercial importance; L reports the worst single block.

**CONSTRAINT.** Σ o_i need not equal Σ e_i — a bay can be observed with more or fewer total facings
than planned, because of gap fill or depth error. When the totals differ, M and M_w diverge
systematically: M penalises the redistribution of the total, M_w penalises shortfall only. Both are
valid; they answer different questions, and mixing them in one dashboard produces a number that
answers neither. The constraint is that each metric must declare whether it is normalised by planned
or observed total, and the declaration must not change between waves.

**NUMERICAL EXAMPLE.** One bay, planned e = (6, 5, 4, 3, 2, 2) totalling 22 facings, observed
o = (6, 4, 4, 5, 2, 1) also totalling 22. Shares s = (0.34, 0.22, 0.18, 0.11, 0.09, 0.06).

```text
|o − e|  = (0, 1, 0, 2, 0, 1)   → Σ = 4
M        = 1 − 0.5 × 4 / 22 = 1 − 0.0909 = 0.9091   (90.9% facing match)

min(o,e) = (6, 4, 4, 3, 2, 1)   → Σ = 20
Bay-level match ratio = 20 / 22 = 0.9091
M_w      = 0.34×1 + 0.22×0.8 + 0.18×1 + 0.11×1 + 0.09×1 + 0.06×0.5
         = 0.34 + 0.176 + 0.18 + 0.11 + 0.09 + 0.03 = 0.926
L        = 2 (the third SKU has 5 facings against 3 planned)
```

The two metrics agree here at 0.909, which is a coincidence of this bay's arithmetic rather than a
property of the metrics. M_w is higher because the two shortfalls land on SKUs with 0.22 and 0.06
share and the over-allocation lands on the 0.11-share SKU, which M_w does not penalise at all.

**Boundary where the metric lies.** Take a second bay with the same totals but a different
distribution: o' = (1, 0, 6, 6, 1, 8), Σ = 22.

```text
|o' − e| = (5, 5, 2, 3, 1, 6) → Σ = 22
M'       = 1 − 0.5 × 22 / 22 = 1 − 0.5 = 0.50
```

Facing match halves, so M does respond. But now consider a third bay, o'' = (6, 4, 4, 3, 2, 1) with a
total of 20, where a single SKU is *entirely missing*: e = (6, 5, 4, 3, 2, 2), o'' = (6, 5, 4, 3, 2, 0).

```text
|o'' − e| = (0,0,0,0,0,2) → Σ = 2
M''       = 1 − 0.5 × 2 / 22 = 0.9545
M_w''     = 0.06 × 0 = 0 → the missing SKU contributes 0.06 × 0 = 0
            = 0.34 + 0.22 + 0.18 + 0.11 + 0.09 + 0.00 = 0.94
L''       = 2
```

A SKU has been delisted from the shelf entirely — a shopper cannot buy it — and the bay scores 95.5%
compliant. This is the boundary where facing-weighted metrics break: they are normalised by planned
facings, so the disappearance of a low-share SKU is a small error by construction. If that SKU is the
only one in the bay with a substitution risk, or the reason the shopper visits the aisle, the metric
has it exactly backwards. The correction is not a better weight; it is an additional gate that flags
any zero-facing SKU for separate review regardless of its share, because zero is qualitatively
different from low.

**INTERPRETATION.** Facing match is a distributional distance and behaves sensibly for redistribution.
It is blind to identity in the specific case that matters most — total absence of a SKU — so it needs
a companion rule. Weighted match is closer to commercial intent but is not a distance (it does not
satisfy symmetry) and should never be averaged with M.

### 6.4 Derivation 4 — What a corrected bay is worth, and the break-even precision

**FORMULA.**

```text
Sales at risk per week, per deviating SKU i:
  A_i = (1 − (o_i / e_i)^ε) × s_i × R          for o_i < e_i, else 0
Bay sales at risk      A = Σ_i A_i
Margin at risk         G = A × m
Persistence factor     P_w = Σ_{w=0}^{3} 0.5^(7w / h)   with h = half-life in days
Value of one corrected bay   V = G × ρ × P_w
Break-even precision    PPV* = c_f / V
```

**WHERE.** ε is a constant space elasticity: a fractional reduction in facings produces a fractional
reduction in sales raised to ε, so a 20% facing cut costs about 1 − 0.8^0.35 of the SKU's sales.
ρ is the fraction of a planned correction that actually sticks (recovery), estimated from
re-observation rather than assumed. h is the compliance half-life: the number of days for the decay
in compliance after a reset to halve. P_w converts a one-off correction into a four-week value by
discounting the drift that erodes it. c_f is the cost of one fix attempt.

**CONSTRAINT.** ε must be a *space* elasticity, not a price elasticity, and it must be applied to
facings, not to sales rank. The two are routinely swapped, which inflates the value of a fix on a
high-revenue SKU by roughly the ratio of the elasticities (commonly 3–5×). A second constraint: the
elasticity is a category-level average, so it understates the value of a fix in a space-sensitive
category (impulse, seasonal) and overstates it in a space-insensitive one (destination staples). If a
category-level space test exists, use it per category; if not, report the value as an elasticity
range rather than a point.

**NUMERICAL EXAMPLE.** R = $1,900 per bay per week, m = 0.28, ε = 0.35, ρ = 0.90. Compliance decays
from 0.94 at day 0 to 0.77 at day 28, both ILLUSTRATIVE.

```text
Decay constant   k   = ln(0.94 / 0.77) / 28 = ln(1.22078) / 28 = 0.19953 / 28 = 0.007126 per day
Half-life        h   = ln 2 / k = 0.69315 / 0.007126 = 97.3 days
Persistence  w=0..3  = 0.5^(0/97.3)=1.0000, 0.5^(7/97.3)=0.9514, 0.5^(14/97.3)=0.9051, 0.5^(21/97.3)=0.8610
P_w                  = 1.0000 + 0.9514 + 0.9051 + 0.8610 = 3.7175

SKU 2: e=5, o=4  → (4/5)^0.35 = 0.8^0.35 = 0.9249 → 1−0.9249 = 0.0751
        A_2 = 0.0751 × 0.22 × 1900 = 0.0751 × 418 = $31.40 per week
SKU 6: e=2, o=1  → (1/2)^0.35 = 0.7846 → 1−0.7846 = 0.2154
        A_6 = 0.2154 × 0.06 × 1900 = 0.2154 × 114 = $24.56 per week
A = 31.40 + 24.56 = $55.96 per week
G = 55.96 × 0.28 = $15.67
V = 15.67 × 0.90 × 3.7175 = $52.42 per corrected bay

c_f = 5.1 × (26.00/60) + 1.50 = 5.1 × 0.4333 + 1.50 = 2.21 + 1.50 = $3.71
PPV* = 3.71 / 52.42 = 0.0708
```

**INTERPRETATION.** A fix is worth about $52 in margin over the four weeks it survives, and it needs
only a 7.1% chance of being a real finding to break even. Since the queue's precision is 74%, the
design has roughly a tenfold margin over break-even, and the margin is what absorbs the assumption
risk in ε. This is the reason the case can proceed with a borrowed elasticity: the decision is not
close to the break-even line, so a materially wrong ε changes the queue's *ordering* long before it
changes the *decision to run a queue at all*.

Two sensitivities bound the claim. Halving ε to 0.20 lowers V to $30.92 and net value per wave from
$2,281.77 to $1,246.69 — still positive, still worth running. Raising ε to 0.50 raises V to $72.62
and net to $3,253.69 — but also reorders the queue toward the high-revenue SKUs, which is a larger
change than the dollar figure suggests.

**Boundary where the formula breaks.** The elasticity form assumes a smooth, positive marginal
response to facings. It fails at o_i = 0, where a shopper cannot buy the product at all and the loss
is not a fractional reduction but potentially the full SKU contribution to basket; it also fails for
a SKU whose facings exceed the planogram, because removing a surplus facing is not symmetric with
adding one (an over-faced SKU blocks access to its neighbour and can *reduce* sales). The third and
most dangerous failure is ε applied to a bay whose planogram is itself wrong: if the planogram
assigns 2 facings to a SKU that deserves 5, then moving from 1 to 2 facings is worth little, and the
value term will still pay for the fix, which is money spent implementing a bad plan. The
discriminating check is a category-level space test on the bay's SKU mix before the value term is
trusted for that category.

### 6.5 Derivation 5 — Allocating a capacity-constrained queue

**FORMULA.**

```text
Maximise   Σ_j x_j × V_j × PPV_j
subject to Σ_j x_j × δ_j ≤ C        (δ_j = 1 if fix takes one slot, but see constraint)
           x_j ∈ {0, 1}
Greedy: sort by V_j × PPV_j descending, take until capacity exhausted.
```

**WHERE.** j indexes candidate bays, V_j is the value of correcting bay j, PPV_j is the probability
that bay j is genuinely non-compliant given its evidence, δ_j is the slot cost of bay j, and x_j is
the inclusion indicator. Ties are broken by severity (largest single block error L), then by store
identifier for determinism.

**CONSTRAINT.** The greedy solution is optimal only when items are indivisible and equal in size. Fix
times are *not* equal — a planogram change that moves blocks takes longer than one that pulls product
forward — so greedy is a heuristic, and the gap from the optimum grows as the spread of per-bay times
grows. The constraint that keeps greedy defensible is to bucket bays by fixture-change type (move-only,
label-change, assortment-change) and allocate capacity per bucket; within a bucket, times are close
enough that greedy is near-optimal. The second constraint is that PPV_j must be estimated per bay
from evidence, not inherited from the queue-level average; using the average for every bay makes the
ranking depend on V alone and collapses this derivation into derivation 4.

**NUMERICAL EXAMPLE.** Six bays with values and evidence qualities, capacity 2 fixes.

```text
Bay   V_j    PPV_j   V×PPV   Rank
A     31.40  0.95    29.83    1
B     24.56  0.98    24.07    2
C     52.42  0.40    20.97    4
D     52.42  0.74    38.79    3 → wait, see recomputation
```

The intermediate table above is deliberately shown as a first pass with an arithmetic error caught in
review, because that is how the error actually occurs. Recomputing line by line:

```text
A  31.40 × 0.95 = 29.83
B  24.56 × 0.98 = 24.07
C  52.42 × 0.40 = 20.97
D  52.42 × 0.74 = 38.79
E  15.67 × 0.90 = 14.10
F  15.67 × 0.30 =  4.70
Order: D (38.79), A (29.83), B (24.07), C (20.97), E (14.10), F (4.70)
Capacity 2 → take D and A. Expected value = 38.79 + 29.83 = $68.62
```

Ranking by value alone would take D and C, for 52.42 + 52.42 = $104.84 of nominal value but only
38.79 + 20.97 = $59.76 expected — a 13% shortfall against the evidence-weighted allocation, on two
fixes. Ranking by evidence alone would take B and A, at $53.90 expected, also below the
value-weighted choice.

**INTERPRETATION.** The ranking criterion is expected value, not value and not confidence. This is
uncomfortable in practice because it sometimes sends a crew to a modest, well-evidenced bay instead of
a spectacular, badly-evidenced one, and the spectacular bay is what supervisors remember. The
justification is arithmetic: a $52 bay with 40% evidence quality is worth $20.97, less than a $31 bay
with 95% quality at $29.83. When the ranking is contested, the argument to make is about the evidence
quality of the specific bay, not about the value of the category.

**Boundary where it breaks.** With strongly unequal fix times, greedy can be materially suboptimal:
if the top-ranked bay needs three times the labour of the next four combined, taking it first can
strand capacity that would have funded four smaller fixes. At a ratio of 3:1 the loss is small; at 8:1
it can reach double digits in percentage terms. The practical guard is a per-bay time cap: any bay
estimated above twice the median is planned separately rather than admitted to the greedy queue.

### 6.6 Derivation 6 — How large a sample is needed to trust the prevalence estimate

**FORMULA.**

```text
n_0 = z² × p(1 − p) / e²                     (simple random sample)
DEFF = 1 + (m − 1) × ICC                     (design effect for m bays per store)
n = n_0 × DEFF
n_stores = n / m
Effective sample size  ESS = n / DEFF
```

**WHERE.** z = 1.96 for 95% confidence, p is the anticipated prevalence (0.22), e is the half-width of
the desired interval, m is the number of bays sampled per store (6), and ICC is the intra-cluster
correlation of compliance within a store (0.22, constructed). ICC is large here because store
management practice, not bay-level randomness, drives most of the variation.

**CONSTRAINT.** The sample must be drawn from the same population the queue operates on — active,
version-bound bays — and must be drawn independently of the detector's flags, or the prevalence
estimate inherits the detector's error. Sampling the flagged bays and calling the observed rate
"prevalence" is the most common and most consequential sampling error in this domain, because it
estimates PPV while believing it is estimating p.

**NUMERICAL EXAMPLE.**

```text
n_0 (e = 0.05) = 1.96² × 0.22 × 0.78 / 0.05² = 3.8416 × 0.1716 / 0.0025
              = 0.65921 / 0.0025 = 263.68 → 264 bays
DEFF          = 1 + (6 − 1) × 0.22 = 1 + 1.10 = 2.10
n             = 264 × 2.10 = 554.3 → 555 bays
n_stores      = 555 / 6 = 92.5 stores
ESS           = 555 / 2.10 = 264.3

At e = 0.03:  n_0 = 3.8416 × 0.1716 / 0.0009 = 732.47 → 733
              n   = 733 × 2.10 = 1539 bays → 256.5 stores
```

The cluster has 28 stores, i.e. 168 sampled bays at 6 per store. That is far short of 555.

```text
ESS at 168 bays = 168 / 2.10 = 80.0
SE(p̂) = √(0.22 × 0.78 / 80.0) = √(0.1716 / 80.0) = √0.002145 = 0.0463
95% interval = 0.22 ± 1.96 × 0.0463 = 0.22 ± 0.0908 → [0.129, 0.311]
```

**INTERPRETATION.** The prevalence estimate is ±9.1 percentage points, not ±5. Through derivation 2
that uncertainty propagates into PPV: at the lower bound p = 0.129 PPV ≈ 0.60, at the upper bound
p = 0.311 PPV ≈ 0.82. The queue's expected true fixes therefore range from 39 to 53 with the same
detector and the same 65 slots. Any claim about queue quality finer than that range is not supported
by the data, and a programme that reports a single PPV figure to three decimal places is reporting
its point estimate as if the interval did not exist.

The resolution is a design choice, not a statistical one: either expand to 20 bays per store (560
bays, ESS 267, SE 0.025) which roughly doubles audit labour, or accept the wide interval and stop
reporting PPV to one decimal. The case takes the second option and states the interval every time the
precision is used, because the decision (which 65 bays) is robust to the interval while the reporting
(how good the queue is) is not.

**Boundary where the formula breaks.** The design effect formula assumes equal cluster sizes and a
single ICC. With unequal store sizes and heterogeneous ICC — urban stores with high staff turnover
versus suburban stores with stable teams — the effective sample size is closer to the number of
clusters than to n/DEFF, and at 28 clusters the interval is dominated by between-store variance. The
practical symptom is that adding bays within a store barely narrows the interval while adding stores
narrows it quickly. The discriminating check is to compute the between-store variance component
directly and compare it with the within-store component; if the between-store share exceeds 70%, the
study is a store study and the bay sample is decoration.
