## 9. Complexity & Cost Analysis

### 9.1 What "cost" means here

Four costs matter and they are not interchangeable: statistical cost (sample size and time needed to reach a decision), engineering cost (assignment service, logging, measurement joins), opportunity cost (revenue withheld from control), and decision cost (the expected loss from acting on an estimate that is wrong in either direction). Analysts habitually optimize the first and ignore the fourth, which is the one the business feels.

### 9.2 Cost of precision

The sample size required for a two-arm test at power `1 - beta` and two-sided alpha:

```text
FORMULA      n_per_arm = (z_{1-alpha/2} + z_{1-beta})^2 * (p1(1-p1) + p0(1-p0)) / (p1-p0)^2
             with alpha = 0.05, power = 0.80: (1.959964 + 0.841621)^2 = 7.84888

WHERE        p0 = baseline rate, p1 = p0 + delta, delta = MDE
CONSTRAINT   normal approximation; each arm needs >= 5 expected events
```

ILLUSTRATIVE evaluations at `p0 = 0.05` (a realistic purchase rate for a broad campaign):

| MDE (pp) | `p1` | numerator term `p1(1-p1)+p0(1-p0)` | `n` per arm | total assigned | weeks at 6,000/week |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.0550 | 0.099475 | 31,231 | 62,461 | 10.4 |
| 1.00 | 0.0600 | 0.103900 | 8,155 | 16,310 | 2.7 |
| 1.39 (break-even in §5.6) | 0.0639 | 0.107307 | 4,366 | 8,732 | 1.5 |
| 2.00 | 0.0700 | 0.112600 | 2,209 | 4,419 | 0.7 |
| 5.00 | 0.1000 | 0.137500 | 432 | 863 | 0.1 |

Two things are visible. First, sample size scales as `1/delta^2`: halving the target effect multiplies the enrolment by roughly 3.7-4, so detecting 1 point instead of 2 costs 8,155 versus 2,209 per arm. Second, the baseline rate enters only through `p(1-p)`, so the same absolute MDE is *harder* to detect at a 50% baseline than at 5%: at `p0 = 0.50` a 2-point effect needs `7.84888*(0.52*0.48 + 0.50*0.50)/0.0004 = 9,803` per arm, more than four times the 2,209 at `p0 = 0.05`. Teams that port a sample-size calculator from a click-through test to a purchase test routinely under-enrol by an order of magnitude because they carried the baseline forward without recomputing.

The cluster correction multiplies the requirement by `DEFF` and CUPED divides it by `(1 - rho^2)`:

```text
DEFF = 1.98, CUPED factor 0.75:
  total required = 8,732 * 1.98 * 0.75 = 12,967 assigned units at the
  break-even MDE, versus 8,732 for the iid design.
```

So a design that is "adequately powered" on paper becomes underpowered once the real correlation structure is acknowledged — and this is the single most common reason a well-run test returns an interval that cannot support a decision.

### 9.3 Cost of the four design families

ILLUSTRATIVE cost model: assignment and logging engineering measured in engineer-weeks, opportunity cost measured in withheld contribution, and calendar time measured at 6,000 eligible units per week.

| Design | Engineering | Opportunity cost | Calendar time | Effective n | Realistic use |
|---|---|---|---|---|---|
| Individual holdout, 50/50 | 2–4 eng-weeks (ledger, sticky assignment, join) | 50% of campaign contribution withheld | 1.5 weeks at break-even MDE | 8,732 | Default |
| Individual holdout, 90/10 | 2–4 eng-weeks | 10% withheld | 4x longer for same precision | ~same total needed | When withholding is expensive |
| Cluster (geo) | 4–8 eng-weeks (geo matching, cluster-robust analysis) | 50% of clusters withheld; larger blast radius | Similar | 8,732 × 1.98 | Spillover or platform-level changes |
| Switchback | 6–10 eng-weeks (scheduling, washout enforcement) | None withheld, but every period is degraded vs the better policy | 12+ weeks of calendar | ~600 (see below) | One shared auction with no geo split |
| DiD / synthetic control | 3–6 eng-weeks | None | Needs long pre-period (8–12 weeks) | 4× cells for same SE | Retrospective, or rollout already decided |
| Propensity / IV | 3–6 eng-weeks | None | Retrospective | Collapses with poor overlap | When randomization is impossible |

The switchback effective sample is the number that surprises people:

```text
12 weekly periods, 20,000 units per period, rho = 0.02
  DEFF  = 1 + (20000-1)*0.02 = 401.0
  n_eff = 12*20000/401.0 = 598.5
  SE    = sqrt(2*0.10*0.90/598.5) = 0.01734        -> 1.73 pp
```

A switchback over a quarter with 240,000 unit-periods delivers the precision of a 599-unit individual test. Increasing periods helps linearly but slowly: at 6 periods per arm the SE is 2.45 pp, at 8 it is 2.12 pp, at 4 it is 3.00 pp. The lever that matters is the within-period correlation `rho` — reducing it from 0.02 to 0.005 would cut `DEFF` roughly fourfold, which is a measurement design question (shorter periods, cleaner washout), not a budget question.

### 9.4 Cost of sequential monitoring

Fixed-horizon tests cost one number: the planned sample. Sequential designs cost a boundary and buy the right to stop early. The trade is real and quantifiable.

```text
O'Brien-Fleming spending, alpha = 0.05 two-sided, equally spaced looks:
  t = 0.25  cumulative alpha = 0.000089   -> |z| > 3.9198
  t = 0.50  cumulative alpha = 0.005575   -> |z| > 2.7718
  t = 0.75  cumulative alpha = 0.023625   -> |z| > 2.2632
  t = 1.00  cumulative alpha = 0.050000   -> |z| > 1.9600

Pocock-style constant boundary, 4 looks (simulated, 200,000 paths):
  |z| > 2.361 at every look gives one-sided+two-sided FWER 0.0500
  |z| > 2.500 gives 0.0350
  unadjusted |z| > 1.960 at every look gives 0.1269
```

The last line is the price of ignoring the problem: 2.5x the nominal error rate. The OBF boundary is nearly impossible to cross early (3.92 standard errors at the first look), which is appropriate when early evidence should be treated with suspicion; the Pocock boundary is roughly flat and easier to cross early, at the cost of a higher final threshold (2.36 versus 1.96). The choice is a policy decision about how much the organization trusts early signals.

Time cost, expressed in required effect size: an OBF design that requires `|z| > 2.7718` at the halfway point needs an effect about 41% larger to stop early than the fixed-horizon test would need at the same sample size, which is why early stopping mostly happens for unexpectedly large effects and those estimates are then biased upward.

### 9.5 Decision cost and the value of a better estimate

The decision-theoretic frame: if you roll out when the true effect is below break-even you lose `N*(c - v*effect)`; if you withhold when it is above break-even you lose `N*(v*effect - c)`. Using the Section 5 numbers with `N = 120,000` (the quarterly eligible base, ten times the test):

```text
roll out at true effect = 0 (the pessimistic case):
  loss = 120,000 * ($0.25 - $18*0) = $30,000
withhold at true effect = 0.0139 (exactly break-even): loss = $0
withhold at true effect = 0.0200: loss = 120,000*($18*0.020 - $0.25) = $13,200
roll out at true effect = 0.0069 (the point estimate):
  loss = 120,000*($0.25 - $18*0.0069) = $15,096
```

The test itself costs `12,000 * $0.25 = $3,000` in contact cost plus engineering. So a test that reduces the probability of a \$30,000 misallocation by a meaningful margin pays for itself several times over — and the relevant question is not "was the test significant" but "how much did it change the probability of the wrong action". A test with an interval that spans break-even has not resolved the decision even though it produced a number.

The counter-case matters too: at `c = $0.10` and `v = $18` the break-even is 0.56 pp, and the point estimate of 0.69 pp is above it, so the decision is "roll out" and further testing has low value. Recognizing when a test is decision-complete prevents the pathological pattern of re-running the same underpowered test every quarter.

### 9.6 Cost of the analysis pipeline itself

The arithmetic is cheap; the joins are not. A production-grade causal pipeline needs, at minimum: an immutable assignment ledger with sticky assignment and rule versioning; an outcome extract keyed to the assignment grain with event-time and ingest-time; a delivery/exposure log; a covariate snapshot at `T0`; and a readout service that recomputes SRM, balance, and maturity checks on every refresh. The recurring engineering cost is in keeping those five consistent under schema change, which is where most experiment infrastructure decay originates.

A useful sizing heuristic: the analysis code is a few hundred lines and the pipeline is a few thousand. Teams that spend the effort on the estimator and none on the ledger end up with a statistically sophisticated readout over a denominator nobody can defend.

**Takeaway:** Buy precision with sample size, buy validity with design, and buy decision quality with the break-even threshold; only the last one changes what you should do.

## 10. Key Theorems / Results Reference

### 10.1 Identification results

**Randomization identifies the ATE.** If `Z` is independent of `(Y(1), Y(0))`, then `E[Y|Z=1] - E[Y|Z=0] = E[Y(1) - Y(0)]`. Proof: by independence, `E[Y|Z=1] = E[Y(1)|Z=1] = E[Y(1)]` and similarly for control; subtract. No modelling assumption is involved, which is why randomization survives criticism that regressions do not.

**Conditional randomization identifies the stratum ATE.** If `Z` is independent of potential outcomes within stratum `s`, the within-stratum difference identifies `E[Y(1)-Y(0)|s]`, and averaging with population weights `w_s` identifies the population ATE. This is the result behind Section 4.2, and it is why stratification is a design fix rather than an analysis trick.

**Instrumental variables identify the LATE, not the ATE.** With relevance (`E[D|Z=1] > E[D|Z=0]`), independence, exclusion, and monotonicity, the Wald ratio `(E[Y|Z=1]-E[Y|Z=0])/(E[D|Z=1]-E[D|Z=0])` identifies the average effect among compliers. It says nothing about always-takers or never-takers. In marketing, compliers are exactly the units whose exposure responds to the nudge, and they are usually not representative of the population.

**Conditional exchangeability identifies the ATE under overlap.** If `(Y(1),Y(0))` is independent of `Z` given `X` and `0 < e(X) < 1`, then `E[ZY/e(X) - (1-Z)Y/(1-e(X))] = E[Y(1)-Y(0)]`. The overlap condition is not decorative: without it the ratio is undefined for some units and the estimator's variance is infinite.

**DiD identifies the ATT under parallel trends.** `E[Y(1)-Y(0)|treated] = (E[Y|treated,post]-E[Y|treated,pre]) - (E[Y|control,post]-E[Y|control,pre])` provided the untreated change in the treated group equals the observed change in controls. The assumption is about a counterfactual trend, so it can be supported but never proven by pre-period data.

### 10.2 Variance results

**Two-sample variance.** `Var(p1-p0) = p1(1-p1)/n1 + p0(1-p0)/n0` under independent Bernoulli sampling. Two refinements are worth knowing:

```text
pooled (used for testing H0: p1 = p0)
  Var_pool = p_bar(1-p_bar)(1/n1 + 1/n0),  p_bar = (y1+y0)/(n1+n0)
unpooled (used for the interval)
  Var_unpooled = p1(1-p1)/n1 + p0(1-p0)/n0

Neyman's third term, which the unpooled form omits:
  Var = S1^2/n1 + S0^2/n0 - S_tau^2/N
  with S_tau^2 the variance of the individual effect
  at n1=n0=6000 and tau = 0.01028 (effects in {0,1}):
    S_tau^2 = 0.01028*0.98972 = 0.010175
    term   = 0.010175/12000 = 8.479e-7
    Var_unpooled = 1.52193e-5   Var_Neyman = 1.43714e-5
    ratio 0.9443  ->  SE ratio 0.9717
```

The third term makes the conservative unpooled interval about 3% wider than necessary. It is almost never applied in marketing practice, and mentioning it is the difference between knowing the formula and knowing its conservatism.

**Cluster design effect.** `Var(cluster mean) = Var(unit mean) * (1 + (m-1)rho)` for equal cluster sizes `m`. For unequal sizes, replace `m` with `m_bar * (1 + CV_m^2)` where `CV_m` is the coefficient of variation of cluster size; a design with highly variable geo sizes is much less efficient than its average size suggests.

**CUPED variance.** `Var(Y - theta(X - E[X])) = Var(Y)(1 - rho^2)` at the optimal `theta`. The result is exact for the linear adjustment and asymptotically valid when `theta` is estimated; with small samples the degrees-of-freedom correction matters.

**Delta-method for ratios.** Rescaling ITT to CACE divides two random quantities. The variance is approximately `Var(ITT)/C^2` where `C` is the compliance difference, so a small compliance difference inflates variance quadratically: at `C = 0.85` the variance multiplier is 1.38; at `C = 0.40` it is 6.25.

### 10.3 Sequential and multiplicity results

**Repeated unadjusted testing inflates type I.** For `k` independent looks at the same nominal level, the family-wise error rate approaches `1 - (1-alpha)^k`; for `k = 4` and `alpha = 0.05` that is 0.185 under independence, and 0.127 with the positive correlation induced by overlapping data (simulated above). The correlation reduces but does not eliminate the inflation.

**O'Brien-Fleming spending.** Cumulative alpha at information fraction `t` is `2 - 2*Phi(z_{1-alpha/2}/sqrt(t))`, giving the boundaries tabulated in Section 9.4. The boundary is conservative early by construction and approaches the fixed-horizon critical value at `t = 1`.

**Bonferroni is conservative under positive dependence.** For `m` tests at level `alpha/m` the family-wise error is at most `alpha`; when tests are positively correlated the actual rate is well below `alpha`, so Bonferroni costs power. Holm's step-down procedure dominates it without extra assumptions.

**Selection on the maximum.** With `m` independent standard normal statistics, `E[max z] ≈ sqrt(2 ln m)`: 15 segments give about 2.33, and 100 give about 3.03. This is the number to quote against any "best segment" claim, because it is the effect size that pure noise produces.

### 10.4 Sensitivity results

**E-value for a risk ratio.** The minimum strength of association, on the risk-ratio scale, that an unmeasured confounder would need with both treatment and outcome to explain away the observed association is `RR + sqrt(RR*(RR-1))`. ILLUSTRATIVE at the Section 5 observed rates:

```text
RR = p1/p0 = 0.0532362/0.0429553 = 1.2393
E  = 1.2393 + sqrt(1.2393*0.2393) = 1.2393 + 0.5445 = 1.7838
```

An unmeasured confounder associated with both assignment and purchase by a risk ratio of 1.78 each would suffice to move the estimate to the null. In a randomized test that is a strange hypothesis (randomization should make such an association impossible); in an observational comparison it is entirely plausible, which is exactly what the E-value is for. Its role here is as a calibration of what "this observational result is fragile" means numerically.

**Bias factor for a binary confounder.** For a confounder with prevalence difference `gamma` between arms and effect `delta` on the outcome, the bias in the risk difference is approximately `gamma * delta`. ILLUSTRATIVE: if 30% more of the treated arm are high-intent (`gamma = 0.30`) and high intent raises purchase probability by 3 points (`delta = 0.03`), the bias is `0.30*0.03 = 0.9` points — larger than the entire stratified estimate of 0.69 points. This arithmetic is the fastest way to kill a retargeting "lift" claim.

**Takeaway:** Five results carry the practice — randomization identifies the ATE, stratification identifies strata, IV identifies compliers only, clustering inflates variance by `DEFF`, and repeated looks inflate error by roughly the square root of the number of looks.
