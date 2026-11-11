## 4. Core Formulas with Full Derivation

All derivations below use the same five-part anatomy: **FORMULA** (the equation and its symbol definitions), **WHERE** (the data that must exist for it to be computable), **CONSTRAINT** (the assumption that makes it valid), **NUMERICAL EXAMPLE** (arithmetic shown at each step), **INTERPRETATION** (what a practitioner may and may not say). Each subsection closes with an explicit validity condition. Numbers marked ILLUSTRATIVE are constructed.

### 4.1 Base case — the two-arm ITT difference and its standard error

**FORMULA**

```text
Let  n1, n0     = assigned units in treatment and control arms   [count, 1..N]
     y1, y0     = units with the outcome in each arm             [count, 0..n]
     p1 = y1/n1, p0 = y0/n0                                      [proportion, 0..1]

ITT   = p1 - p0                                                 [proportion-difference]
Var   = p1(1-p1)/n1 + p0(1-p0)/n0                               [variance of difference]
SE    = sqrt(Var)
CI95  = ITT +/- 1.959964 * SE                                   [Wald interval]
```

`p1` and `p0` are arm-specific outcome rates over *assigned* units. The variance expression uses the normal approximation to the binomial separately inside each arm and adds them because the arms are independent samples. The coefficient 1.959964 is the two-sided 95% normal quantile.

**WHERE** The assignment ledger with one row per unit and columns `unit_id`, `stratum`, `arm`, `assigned_at`; the outcome table with one row per unit and `outcome_flag`, `event_at`. The join is a left join from the ledger so that zeros are counted, not dropped. Both counts must come from the same maturity cut-off timestamp.

**CONSTRAINT** Independence between arms (no shared budget, no household spillover), independent outcomes within arm (no clustering), and a fixed analysis horizon declared before data access. If outcomes are clustered, the variance formula understates `SE` by `sqrt(DEFF)` with `DEFF = 1 + (m-1)rho` — Section 4.3.

**NUMERICAL EXAMPLE** ILLUSTRATIVE. 12,000 eligible customers, 6,180 assigned to the personalized offer and 5,820 to standard email; 329 and 250 purchasers respectively within 14 days.

```text
p1 = 329 / 6180 = 0.0532362
p0 = 250 / 5820 = 0.0429553
ITT = 0.0532362 - 0.0429553 = 0.0102809   -> 1.028 percentage points

Var = 0.0532362*0.9467638/6180 + 0.0429553*0.9570447/5820
    = 0.0504021/6180 + 0.0411102/5820
    = 0.0000081557 + 0.0000070636
    = 0.0000152193
SE  = sqrt(0.0000152193) = 0.0039012   -> 0.3901 pp

CI95 = 0.0102809 +/- 1.959964*0.0039012
     = 0.0102809 +/- 0.0076462
     = [0.0026347, 0.0179271]   -> [0.26 pp, 1.79 pp]

z    = 0.0102809 / 0.0039012 = 2.6353  (two-sided p = 0.0084)
```

Every intermediate is shown because rounding at the wrong step changes the third digit: `0.3901` rounded to `0.39` gives a CI upper bound of `1.793` instead of `1.793`, but rounding `SE` to `0.4` gives `1.81`, a visible difference at the two-decimal reporting precision that executives read.

**INTERPRETATION** The assignment policy raised the 14-day purchase rate by 1.03 points, with a 95% interval excluding zero. That sentence is licensed *only* on assigned units. It is not licensed to say the offer "works for people who see it", and it is not licensed to say "2.4% lift" without stating that the relative figure uses `p0` as the denominator: `0.0102809/0.0429553 = 23.9%` relative, which sounds far larger than 1.03 points and is the same estimate.

**Validity condition:** valid iff each arm's outcomes are independent Bernoulli draws with no cross-arm interference, the two counts cover the same window, and the analyst declared this contrast and this horizon before seeing outcomes. Any deviation requires a different variance estimator, not a caveat added after the fact.

### 4.2 General case — stratified assignment with unequal probabilities

Marketing randomization is rarely simple. Targeting sends high-value segments to treatment with probability 0.6, mid-value with 0.5, and low-value with 0.4. In that design the unweighted difference is biased for the population effect because the arm composition differs by stratum.

**FORMULA**

```text
Let  s = 1..S            strata defined on pre-assignment covariates
     N_s                 eligible units in stratum s        [count]
     N = sum_s N_s       total eligible                       [count]
     w_s = N_s / N       population weight                    [-]
     p1_s, p0_s          arm rates within stratum s
     n1_s, n0_s          arm counts within stratum s

ITT_strat = sum_s w_s (p1_s - p0_s)                          [weighted average]
Var_strat = sum_s w_s^2 [ p1_s(1-p1_s)/n1_s + p0_s(1-p0_s)/n0_s ]
```

The estimand being computed is the **ATE over the eligible population**, so the weights are population shares, not arm shares. Using treated-arm shares computes an **ATT** instead; using control shares computes yet another quantity. All three are legitimate and all three give different numbers — see the numeric example.

**WHERE** The ledger needs a `stratum` column assigned before randomization, plus the prespecified stratum probabilities `pi_s`. The outcome join is identical to the base case. Strata must be defined on fields that exist at `T0`; a stratum built from the customer's outcome-adjacent behaviour is a post-treatment variable.

**CONSTRAINT** Within-stratum randomization with known `pi_s`, and enough units per stratum that the normal approximation holds (rule of thumb: at least 5 expected outcomes and 5 expected non-outcomes per arm per stratum). Strata with sparse outcomes must be collapsed before the weights are applied, or the variance explodes.

**NUMERICAL EXAMPLE** ILLUSTRATIVE. Three strata, `N = 4800 + 4200 + 3000 = 12000`.

| Stratum `s` | `N_s` | `pi_s` | `n1_s` | `y1_s` | `p1_s` | `n0_s` | `y0_s` | `p0_s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| High value | 4800 | 0.60 | 2880 | 187 | 0.0649306 | 1920 | 108 | 0.0562500 |
| Mid value | 4200 | 0.50 | 2100 | 99 | 0.0471429 | 2100 | 84 | 0.0400000 |
| Low value | 3000 | 0.40 | 1200 | 43 | 0.0358333 | 1800 | 58 | 0.0322222 |

```text
naive pooled:  n1 = 6180, y1 = 329 -> p1 = 0.0532362
               n0 = 5820, y0 = 250 -> p0 = 0.0429553
               naive ITT = 0.0102809  (1.0281 pp)
               naive Var = 0.0000152193, SE = 0.0039012

weights: w = (0.40, 0.35, 0.25)                         [4800/12000 etc.]
stratum effects: d_s = (0.0086806, 0.0071429, 0.0036111)
weighted ITT = 0.40*0.0086806 + 0.35*0.0071429 + 0.25*0.0036111
             = 0.0034722 + 0.0025000 + 0.0009028
             = 0.0068750                          -> 0.6875 pp

stratum variances: v_s = p1(1-p1)/n1 + p0(1-p0)/n0
   s=1: 0.0649306*0.9350694/2880 + 0.05625*0.94375/1920
      = 0.00002108145 + 0.00002764893 = 0.00004873038
   s=2: 0.0471429*0.9528571/2100 + 0.0400000*0.9600000/2100
      = 0.00002139067 + 0.00001828571 = 0.00003967638
   s=3: 0.0358333*0.9641667/1200 + 0.0322222*0.9677778/1800
      = 0.00002879109 + 0.00001732442 = 0.00004611551
Var_strat = 0.16*0.00004873038 + 0.1225*0.00003967638 + 0.0625*0.00004611551
          = 0.00000779686 + 0.00000486036 + 0.00000288222
          = 0.00001553944
SE_strat  = 0.00394201     -> 0.3942 pp
CI95      = 0.0068750 +/- 1.959964*0.00394201 = [-0.000851, 0.014601]
```

The naive estimate is 1.028 points and significant (`z = 2.635`); the correctly weighted estimate is 0.688 points and not significant at 95% (`z = 1.744`, two-sided `p = 0.081`). The gap is not noise — it is Simpson-type composition: the high-value stratum received treatment more often *and* has the highest baseline rate, so the pooled comparison credits part of the segment mix to the offer. Recovering the treated-arm weighting instead gives `0.006558` (0.66 pp), and reweighting control to treated shares gives `0.007174` (0.72 pp); three defensible estimands, three different decisions under a 0.7-point break-even threshold.

**INTERPRETATION** The number an executive hears must be paired with its weighting. "Lift of 0.69 points for the eligible population" and "lift of 1.03 points among the assigned mix" are both true statements about the same data and support different rollout decisions. The reporting template must print the weights.

**Validity condition:** valid iff strata are pre-assignment, probabilities are known rather than estimated from realized counts, and no stratum is so sparse that its arm rate is 0 or 1. If a stratum is degenerate, collapse it and renormalize, documenting the merge.

### 4.3 Variants — four corrections that change the decision

#### 4.3a Clustered assignment (design effect)

When assignment happens at store, household, or geography level, the effective sample size is not the unit count.

```text
FORMULA      DEFF = 1 + (m - 1) * rho
             SE_cluster = SE_iid * sqrt(DEFF)
             n_eff = n_total / DEFF
WHERE        m   = units per cluster [count]
             rho = intra-cluster correlation of the outcome [-]
CONSTRAINT   equal cluster sizes; unequal sizes require the
             more general 1 + (CV_m^2 + 1)*m_bar*rho - 1 form
```

ILLUSTRATIVE with `m = 50` households per block and `rho = 0.02`: `DEFF = 1 + 49*0.02 = 1.98`. Using the 12,000 units from Section 4.1, `n_eff = 12000/1.98 = 6061` and `SE` becomes `0.0039422*sqrt(1.98) = 0.005547` (0.55 pp). The 95% interval on a 0.69-point effect then runs from `-0.40` to `+1.78` points. Raising `rho` to `0.05` with the same block size gives `DEFF = 3.45` and `SE = 0.007322` (0.73 pp), at which point a 0.69-point effect is indistinguishable from zero regardless of how many customers are enrolled. The lever is block size, not budget: halving `m` to 25 at `rho = 0.05` gives `DEFF = 2.20`.

#### 4.3b Pre-period covariate adjustment (CUPED)

Subtracting a pre-period proxy scaled by its regression coefficient removes variance without touching the estimand, provided the proxy is measured before assignment.

```text
FORMULA      Y* = Y - theta * (X - E[X]),   theta = Cov(Y,X)/Var(X)
             Var(Y*) = Var(Y) * (1 - rho^2)
             rho = Corr(Y, X)
CONSTRAINT   X must be pre-assignment and unaffected by Z; theta estimated
             out-of-sample or with the appropriate degrees-of-freedom correction
```

ILLUSTRATIVE: a 90-day pre-period purchase count correlates at `rho = 0.5` with the 14-day outcome. `Var` falls by factor `1 - 0.25 = 0.75`, so `SE` falls by `sqrt(0.75) = 0.8660`. Applied to the clustered `SE` of 0.005547, the combined `SE` is `0.005547*0.8660 = 0.004804` (0.48 pp). At `rho = 0.7` the factor is `0.7141` and the combined `SE` is 0.003961. The trap: if `X` is measured after assignment — "pre-period" defined as the 90 days before the *outcome* rather than before *assignment* — CUPED absorbs part of the treatment effect and biases the estimate toward zero.

#### 4.3c Difference-in-differences for non-randomized rollouts

When a region gets the campaign and a comparable region does not, the estimator uses two differences so that a common time shock cancels.

```text
FORMULA      DiD = (p_T,post - p_T,pre) - (p_C,post - p_C,pre)
             Var = sum over the four cells of p(1-p)/n
CONSTRAINT   parallel trends: absent treatment, the treated region's change
             would have equalled the control region's change
```

ILLUSTRATIVE with 1,000 units per cell: treated `10.0% -> 13.0%`, control `10.0% -> 11.0%`.

```text
DiD = (0.13 - 0.10) - (0.11 - 0.10) = 0.03 - 0.01 = 0.02  (2.00 pp)
Var = 0.13*0.87/1000 + 0.10*0.90/1000 + 0.11*0.89/1000 + 0.10*0.90/1000
    = 0.00011310 + 0.00009000 + 0.00009790 + 0.00009000
    = 0.00039100
SE  = 0.019774   -> 1.977 pp
CI95 = 0.020 +/- 1.959964*0.019774 = [-0.01876, +0.05876]
```

The point estimate of 2.00 points looks identical to a randomized 2-point effect but its interval is five times wider at the same cell size, because DiD variance comes from four noisy cells rather than two. More importantly: if the treated region was already trending at `+0.5` pp per period for reasons unrelated to the campaign, the true campaign effect is `2.0 - 0.5 = 1.5` pp and the estimator is biased by the entire trend violation. Parallel trends is not testable with two periods; it is only falsifiable with pre-period data showing the trends tracked before treatment.

#### 4.3d Compliance rescaling (CACE) and the dilution from leakage

```text
FORMULA      CACE = ITT / (P(D=1|Z=1) - P(D=1|Z=0))
CONSTRAINT   monotonicity (no defiers) and exclusion (Z affects Y only via D)
```

ILLUSTRATIVE: ITT of 2.0 points, delivery 85% in treatment and 0% in control, gives `CACE = 0.020/0.85 = 0.02353` (2.35 pp). If 5% of controls also receive the offer through a suppression failure, the denominator becomes `0.85 - 0.05 = 0.80` and `CACE = 0.0250` (2.50 pp). With 40% delivery the rescaling yields 5.00 points — but at that point the exclusion restriction deserves scrutiny, because assignment to a mostly-undelivered campaign may itself change behaviour (inventory holds, pricing changes) through channels other than delivery.

**Validity condition for all of 4.3:** each correction changes the variance, the estimand, or both. State which. A design effect changes variance only; CUPED changes variance only if `X` is pre-assignment; DiD changes the identifying assumption; CACE changes the estimand from assignment effect to complier effect. Never present a corrected number without naming which of these happened.
