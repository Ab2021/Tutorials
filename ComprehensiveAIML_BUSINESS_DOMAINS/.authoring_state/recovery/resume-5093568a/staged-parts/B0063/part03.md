## 5. Worked Numerical Example

### 5.1 The decision and the specification

A subscription retailer runs a **14-day personalized-offer test** against its standard email policy. ILLUSTRATIVE throughout: the customer counts, purchase counts, cost per contact, and contribution margin are constructed for teaching and were recomputed by hand below. No company result is represented.

| Specification item | Value in this example |
|---|---|
| Decision | Adopt the personalized-offer policy for the next quarter, or keep standard email |
| Eligible population | Active subscribers, tenure ≥ 30 days, purchasable in the US, consent on file |
| Unit | Customer (household IDs collapsed first, so one row per household) |
| Assignment | Hash of `household_id` salted with `test_id`, 50/50 within three value strata |
| Comparator | Standard email policy, same send day, same suppression list |
| Outcome | Any purchase with `event_at` in `[assigned_at, assigned_at + 14 days)` |
| Maturity cut-off | `assigned_at + 21 days`, allowing seven days of ingest lag |
| Primary estimand | ATE over eligible population, stratified weights |
| Secondary estimand | CACE at observed delivery (85% treatment, 0% control) |
| Economic inputs | Contact cost \\$0.25 per assigned customer; contribution margin \\$18 per purchase |

The specification is written before the data is touched because three of its rows are the ones analysts silently change when the result disappoints: the comparator (swapping to "no email at all" inflates the effect), the maturity cut-off (extending the window for the treated arm only), and the estimand (switching to CACE when ITT is null).

### 5.2 The counts

Assignment ran over 12,000 households. Realized allocation was 6,180 treatment and 5,820 control, which is a 51.5/48.5 split rather than the designed 50/50.

```text
FORMULA      chi2_SRM = sum_arm (observed - expected)^2 / expected,  df = 1
WHERE        expected = 6,000 per arm under a 50/50 design
NUMERICAL    (6180-6000)^2/6000 + (5820-6000)^2/6000
             = 32400/6000 + 32400/6000
             = 5.40 + 5.40 = 10.80
             p = P(chi2_1 > 10.80) = 2*(1 - Phi(sqrt(10.80)))
               = 2*(1 - Phi(3.2863)) = 0.00102
INTERPRETATION  A global SRM test rejects at 0.1%, but this design is NOT 50/50
             globally: stratum probabilities are 0.60 / 0.50 / 0.40.
```

The global test is the wrong test, and this is the most common SRM false alarm in marketing. Expected counts must be computed within stratum:

```text
stratum       N_s    pi_s   E[n1] = N_s*pi_s   observed n1   chi2 contribution
High value    4800   0.60   2880               2880          (0)^2/2880  = 0
Mid value     4200   0.50   2100               2100          (0)^2/2100  = 0
Low value     3000   0.40   1200               1200          (0)^2/1200  = 0

total chi2 = 0.000, df = 3, p = 1.000
```

The assignment service is behaving exactly as designed. The 180-unit global imbalance is the *expected consequence* of unequal stratum probabilities combined with real stratum sizes — a mathematical artifact, not a defect. Had the analyst stopped at the global test, the correct test would have been abandoned or, worse, the data re-randomized.

### 5.3 The unadjusted estimate

```text
n1 = 2880 + 2100 + 1200 = 6180      y1 = 187 + 99 + 43 = 329
n0 = 1920 + 2100 + 1800 = 5820      y0 = 108 + 84 + 58 = 250

p1 = 329/6180 = 0.0532362
p0 = 250/5820 = 0.0429553
ITT_naive = 0.0102809                       -> 1.0281 pp

Var = 0.0532362*0.9467638/6180 + 0.0429553*0.9570447/5820
    = 0.0000081557 + 0.0000070636 = 0.0000152193
SE  = 0.0039012                             -> 0.3901 pp
CI  = [0.0026347, 0.0179271]                -> [0.26 pp, 1.79 pp]
z   = 2.6353        two-sided p = 0.0084
```

An unstratified readout therefore reports a statistically significant 1.03-point lift and the team starts drafting the rollout memo. Section 5.4 is why that memo would have been wrong.

### 5.4 The stratified estimate

```text
w_s      = (0.40, 0.35, 0.25)
d_s      = (0.0086806, 0.0071429, 0.0036111)          [p1_s - p0_s]
ITT_strat= 0.40*0.0086806 + 0.35*0.0071429 + 0.25*0.0036111
         = 0.00347222 + 0.00250000 + 0.00090278
         = 0.00687500                                  -> 0.6875 pp

v_s (stratum variance of the difference):
  s=1: 0.00004873038
  s=2: 0.00003967638
  s=3: 0.00004611551
Var  = 0.16*0.00004873038 + 0.1225*0.00003967638 + 0.0625*0.00004611551
     = 0.00001553944
SE   = 0.00394201                                      -> 0.3942 pp
CI   = [-0.000851, 0.014601]                           -> [-0.09 pp, 1.46 pp]
z    = 1.7438        two-sided p = 0.0812
```

Two properties deserve attention. First, the standard error barely changed (0.3942 versus 0.3901) while the point estimate fell by a third — the correction is about bias, not precision. Second, the sign of the decision flipped: the interval now includes zero and the lower bound is slightly negative.

The three estimands that a real analysis must choose between, computed from the same table:

```text
ATE  (population weights)      = 0.0068750   (0.69 pp)
ATT  (treated-arm weights)     = 0.0065579   (0.66 pp)
  recompute: p1_std = sum_s (n0_s/n0)*p1_s
           = 0.329897*p1_1 + 0.360825*p1_2 + 0.309278*p1_3
           = 0.0214204 + 0.0170103 + 0.0110825 = 0.0495132  [4.95132%]
           ATT = 0.0495132 - 0.0429553 = 0.0065579
ATE via IPW (check)           = 0.0068750
  per stratum: w_s*(p1_s - p0_s) with weights on arm-specific means
  equals the weighted sum above because stratum probabilities are common
  within stratum across arms; it differs from ATT because the weights differ
```

The ATE-versus-ATT distinction is not academic here. The rollout decision applies to the eligible population (ATE), but a retention team that only ever touches high-value customers would be making an ATT decision and should use the 0.66-point figure. Section 8 F-11 catalogues the case where an analyst computes ATE weights but describes the result as if it applied to the treated, which overstates the benefit to the customers most likely to be targeted.

### 5.5 Design-quality corrections and what they do to power

The interval above assumes independent Bernoulli outcomes. Two realistic violations apply: household-level correlation among cohabiting customers, and the availability of a pre-period purchase count.

```text
cluster effect:  m = 50 cohabiting households per block, rho = 0.02
  DEFF  = 1 + 49*0.02 = 1.98
  SE_c  = 0.00394201 * sqrt(1.98) = 0.0055469        -> 0.55 pp
  CI_c  = [0.0068750 - 1.959964*0.0055469, ... ]
        = [-0.003997, 0.017747]                     -> [-0.40 pp, 1.77 pp]
  n_eff = 12000/1.98 = 6060.6   (half the enrolments do the work of 6,061)

CUPED with pre-period purchase count, rho_YX = 0.5
  Var factor = 1 - 0.25 = 0.75,  SE factor = 0.86603
  SE_cc = 0.0055469 * 0.86603 = 0.0048038           -> 0.48 pp
  CI_cc = [-0.002539, 0.016290]                     -> [-0.25 pp, 1.63 pp]
```

CUPED recovers roughly three quarters of the variance that clustering cost, and the combined interval is still wider than the naive one. This is the normal outcome: a properly designed and analysed test usually has a *wider* interval than the naive readout, because the naive readout omitted real sources of variance.

### 5.6 Economic break-even and the decision

```text
cost per assigned customer      c = $0.25
contribution per purchase       v = $18.00
break-even ITT                  t* = c/v = 0.25/18 = 0.0138889   -> 1.389 pp

projected incremental purchases at N = 12,000 (holds population fixed):
  point: N*0.0068750 = 82.5 purchases
         revenue 82.5*18 = $1,485 ; spend 12,000*0.25 = $3,000 ; net = -$1,515
  lower: N*(-0.000851) = -10.2 purchases
         revenue -$183.86 ; spend $3,000 ; net = -$3,183.86
  upper: N*(0.014601) = 175.2 purchases
         revenue $3,153.86 ; spend $3,000 ; net = +$153.86
```

The point estimate is unprofitable: the campaign would spend \$3,000 to earn \$1,485 in contribution, a loss of \$1,515. The 95% interval spans a \$3,337 range of outcomes and its upper end clears break-even by only \$154. A probability statement makes the decision sharper than a significance test:

```text
P(true effect > 0)        = 1 - Phi((0 - 0.006875)/0.00394201)
                          = 1 - Phi(-1.7440) = 0.9594
P(true effect > 1.389 pp) = 1 - Phi((0.0138889 - 0.006875)/0.00394201)
                          = 1 - Phi(1.7790) = 0.0376
```

The campaign is probably positive and almost certainly not profitable at these unit economics. Under the cluster-plus-CUPED standard error the second probability moves to `1 - Phi(1.4601) = 0.0721`, and the interval's upper bound no longer clears break-even. Any decision rule that ignores the design effect reports more confidence in profitability than the design supports.

### 5.7 Sensitivity to the economic inputs

Because the decision is a comparison of an estimate to a threshold, the threshold deserves as much scrutiny as the estimate. Recomputing net contribution at the point estimate and at the interval endpoints across five input pairs:

| `c` (\\$/contact) | `v` (\\$/purchase) | break-even (pp) | net at point | net at lower | net at upper | power at break-even (iid) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 18 | 0.556 | +\\$285 | -\\$1,384 | +\\$1,954 | 0.291 |
| 0.25 | 18 | 1.389 | -\\$1,515 | -\\$3,184 | +\\$154 | 0.941 |
| 0.40 | 18 | 2.222 | -\\$3,315 | -\\$4,984 | -\\$1,646 | 1.000 |
| 0.25 | 9 | 2.778 | -\\$2,258 | -\\$3,092 | -\\$1,423 | 1.000 |
| 0.25 | 36 | 0.694 | -\\$30 | -\\$3,368 | +\\$3,308 | 0.421 |

Two rows are decisive. At `c = \\$0.10` the campaign is profitable at the point estimate and break-even sits *below* the estimate; the same data supports adoption. At `v = \\$9` — a margin half the base case, plausible for a discounted first order — no point in the interval is profitable and the answer is a clean no. The estimate did not change; the economics did, and the economics carry more decision weight than the p-value.

The "power at break-even" column is a diagnostic for test design, not for this result: it shows the probability that a future test of the same size would correctly reject a null of no effect if the true effect were exactly break-even. Below about 0.8, a null result cannot distinguish "no effect" from "break-even effect", which is the worst outcome a decision maker can be handed.

### 5.8 Independent arithmetic re-check

A second path that avoids the rate subtraction entirely:

```text
purchases: treatment 329, control 250, total assigned 12,000
count difference = 329 - 250 = 79
79 / 12,000 = 0.0065833                       -> 0.658 pp

This differs from the stratified 0.6875 pp by 0.029 pp, because the
pooled denominator weights the arms by their realized sizes rather than
by stratum population shares. It is the right cross-check that the
stratified figure was NOT produced by dividing counts by a single
denominator: if an analyst's stratified result equals 0.6583 pp exactly,
they recomputed the naive number and mislabelled it.
```

Three additional re-checks, each targeting a specific arithmetic slip:

```text
1. Stratum counts must sum to arm counts:
   2880+2100+1200 = 6180 = n1   and   1920+2100+1800 = 5820 = n0   OK
   187+99+43 = 329 = y1   and   108+84+58 = 250 = y0                OK

2. Weights must sum to one: 0.40+0.35+0.25 = 1.00                   OK

3. Variance must be positive and SE must be smaller than the largest
   plausible binomial SE at these counts:
   worst-case binomial SE at p=0.5, n=6180: sqrt(0.25/6180) = 0.00636
   both SEs (0.00390, 0.00394) are below it                        OK
```

A fourth check that catches unit errors: the effect in percentage points must equal the effect in proportion multiplied by 100, and the *counts* of incremental purchases must equal the effect times the eligible population under the population estimand. `0.0068750 * 12000 = 82.5`, consistent with the table in 5.6. If a deck reports "82 incremental purchases" against a 1.03-point effect, `0.0102809*12000 = 123.4`, not 82.5 — the mismatch means the deck mixed the naive point estimate with the stratified projection, which is the most common internal inconsistency in experiment write-ups.

### 5.9 What the estimate does not license

- It does not license "the offer lifts conversion 24%" without the denominator: `0.0102809/0.0429553 = 23.93%` relative, which is the same data as 1.03 points and sounds three times larger.
- It does not license an exposure effect. Delivery was 85%, so `CACE = 0.020/0.85 = 2.35 pp` under monotonicity — but that number requires an exclusion restriction nobody verified.
- It does not license a rollout to the full 120,000-household list at the same margin. The enrolled population was filtered on tenure ≥30 days and prior purchasability; the marginal household beyond the filter has a lower baseline rate, and the same absolute effect would produce fewer incremental purchases.
- It does not license a repeated quarterly read as if each quarter were independent. The same customers appear in successive experiments; assignment histories correlate with outcome histories, so the third quarter's test is not a clean replication.

**Takeaway:** Compute the population-weighted estimate, the design-corrected standard error, and the break-even threshold; the decision is where those three meet, and all three can be computed before the test launches.

## 6. Visual / Geometric Intuition

### 6.1 The assignment wall

```text
   PRE-ASSIGNMENT (covariates)   |   POST-ASSIGNMENT (consequences)
   ----------------------------- | ------------------------------------
   tenure, prior spend, region   |   delivery, open, click, session
   consent state, list source    |   cart, purchase, return, churn
   eligibility flags as of T0    |   support ticket, unsubscribe
                                 |
   may be adjusted for           |   never a covariate in the primary model
   may define strata             |   may define a secondary, assumption-heavy estimand
```

Draw this wall once per experiment. Any variable to the right of it that appears in a model is a decision, not a default, and the decision must be written in the analysis plan with the estimand it produces. The wall is also the reason "control for engagement" fails: engagement is the first thing assignment changes.

### 6.2 The collider, drawn and quantified

```text
                    latent intent U
                    /            \
             (a)   /              \  (b)
                  v                v
   assignment Z --------> click S <-------- nothing else
                  \                /
             (c)   \              /
                    v            v
                     purchase Y
```

Paths: `Z -> S` is the treatment effect on clicking; `U -> S` and `U -> Y` are latent intent acting on both. Conditioning on `S` (restricting to clickers) opens the path `Z -> S <- U -> Y`, a collider path. The result is a comparison between treated clickers and control clickers who are not comparable on `U`.

A numeric demonstration where the **true ITT is exactly zero**:

```text
ILLUSTRATIVE model
  P(U=1) = 0.5, independent of Z                      (randomization holds)
  P(S=1 | Z=1, U=1) = 0.80    P(S=1 | Z=1, U=0) = 0.40
  P(S=1 | Z=0, U=1) = 0.30    P(S=1 | Z=0, U=0) = 0.10
  P(Y=1 | U=1) = 0.20         P(Y=1 | U=0) = 0.02     (no Z term: effect = 0)

all-assigned means:
  E[Y|Z=1] = 0.5*0.20 + 0.5*0.02 = 0.110
  E[Y|Z=0] = 0.5*0.20 + 0.5*0.02 = 0.110
  ITT = 0.000                                        <- truth

click-through rates:
  E[S|Z=1] = 0.5*0.80 + 0.5*0.40 = 0.600
  E[S|Z=0] = 0.5*0.30 + 0.5*0.10 = 0.200

composition of clickers:
  treated:   P(U=1|S=1,Z=1) = (0.5*0.80)/0.600 = 0.6667
  control:   P(U=1|S=1,Z=0) = (0.5*0.30)/0.200 = 0.7500

conversion among clickers:
  E[Y|S=1,Z=1] = 0.6667*0.20 + 0.3333*0.02 = 0.14000
  E[Y|S=1,Z=0] = 0.7500*0.20 + 0.2500*0.02 = 0.15500
  "clicker lift" = 0.14000 - 0.15500 = -0.01500     -> -1.50 pp
```

The true effect is zero, the treated arm clicks three times as often, and the clicker-restricted readout reports a *negative* 1.5-point effect. The sign reversed because treatment pulled in lower-intent clickers (0.40 click rate among low-intent treated versus 0.30 among low-intent control), diluting the clicker pool. An analyst who reasoned "treated clickers are worse, so the campaign is bad" would have concluded the opposite of the truth, and an analyst in a different parameterization would have concluded the opposite error — the point is that the direction of collider bias is not signable in advance.

The same geometry explains why "viewability-weighted lift" is untrustworthy: viewability is a post-assignment selection variable, and the treated/control compositions of viewable impressions differ for exactly the reason above.

### 6.3 Intervals as ranges of decisions, not error bars

```text
net contribution at N = 12,000, c = $0.25, v = $18

 -$3,184        -$1,515          $0          +$154
    |--------------|--------------|--------------|
    lower         point        break-even      upper
    bound        (0.69 pp)     (1.39 pp)     bound
    |<----------- loss ----------->|<-- tiny gain -->|
```

The picture makes the decision readable without any statistics vocabulary: most of the interval is loss, the profitable region is a thin sliver at the top, and the point estimate sits on the loss side. A bar chart of arm conversion rates cannot show this, because it omits the threshold. Any readout that shows two bars and no threshold is asking the audience to supply their own economics, which they will do differently from one another.

### 6.4 Sequential looks and the invisible multiplicity

```text
nominal 95% bands as a function of analysis time, unadjusted:

z
4 |                                    ....
3 |                          ....
2 |--+--+--+--+--+--+--+--+--          <- fixed 1.96 boundary
1 |     *     *
0 |  *
  +----------------------------------> fraction of planned sample
    .25   .50   .75   1.00

under H0 with 4 equally spaced unadjusted looks, the probability that
at least one look crosses the boundary is 0.127, not 0.05
```

The boundary line is what a fixed-horizon test implies; the wandering path is what the data does. Stopping when the path first crosses is what inflates the error rate — Section 9.4 gives the alpha-spending boundaries that repair it, with the actual cumulative alpha values.

**Takeaway:** Three pictures carry the whole method — the assignment wall, the collider graph, and the interval against a break-even threshold; each prevents a specific class of wrong conclusion.
