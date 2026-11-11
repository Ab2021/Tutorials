### 4.3 Variants: switchback, stepped wedge, and compliance adjustment

**Switchback.** When only a few stores exist, or when store-level differences are enormous relative to the effect, assignment can alternate inside a store across periods. Each store is its own control, so `sigma_b^2` drops out of the comparison entirely.

```text
FORMULA [F-5]: Delta_switch = mean(Y | treated weeks) - mean(Y | control weeks), within store
               Var(Delta_switch) = (sigma_w^2 / (n * m_t)) + (sigma_w^2 / (n * m_c))
WHERE: m_t, m_c [weeks] treated and control weeks per store; sigma_w^2 within-store variance.
CONSTRAINT: no carry-over from one period into the next; balanced period length;
            the switching schedule must not align with day-of-week or month effects.
BOUNDARY: if carry-over exists, the second period is contaminated and the estimate is biased
          toward zero by an amount proportional to the carry-over fraction.
```

Compare the two designs on the same numbers: switchback with `sigma_w^2 = 0.00091875` (i.e. `rho = 0.25` applied to `sigma^2 = 0.001225`), `n = 12` stores, `m_t = m_c = 2` gives `Var = 2*0.00091875/24 = 0.00007656` and `SE = 0.008750`. That is 7.4% tighter than the parallel design's `0.009451`, with half the stores' weeks exposed. The gain is real but modest, and it is bought with the carry-over assumption.

The carry-over cost is easy to bound. If the first week of each control period retains 30% of the treatment effect, the contrast is diluted by `1 - 0.30/2 = 0.85` when periods are two weeks, since only one of two control weeks is contaminated. A 2.6 pp true effect would read as 2.21 pp. The design must either drop the first week after each switch or state the dilution.

**Stepped wedge.** Stores switch from control to treatment in staggered waves. The analysis compares each store's treated weeks with its own not-yet-treated weeks, plus a time trend.

```text
FORMULA [F-6]: Y_ij = mu + a_i + g_j + tau * T_ij + e_ij
WHERE: a_i [outcome units] store intercept; g_j [outcome units] period effect common to all
       stores; T_ij in {0,1} treatment indicator; tau [outcome units] the effect of interest.
CONSTRAINT: the period effect g_j must be identifiable, which requires at least one store still
            untreated in every period for the whole exposure window.
BOUNDARY: once every store is treated, later periods contribute only the trend, not the contrast.
```

The stepped wedge's advantage is logistical: no store is asked to run the old process for the whole trial, which is often the only politically feasible arrangement. Its cost is that the effect is estimated against a time trend, so any shock that coincides with the rollout (a price change, a competitor opening) enters `tau` directly and cannot be separated.

**Compliance adjustment.** When `pi_c < 0.95`, the ITT estimate understates the effect on store-weeks that actually used the process.

```text
FORMULA [F-7]: Delta_CACE = Delta_ITT / (pi_t - pi_c)
WHERE: pi_t, pi_c [fraction] treatment share in assigned-treated and assigned-control arms;
       denominator must exceed 0.10 for the ratio to be interpretable.
CONSTRAINT (exclusion restriction): assignment affects the outcome only through the receipt
       of the process. Stores that would never comply must not differ in outcome-relevant ways.
CONSTRAINT (monotonicity): no store is deterred from the process by being assigned to it.
NUMERICAL EXAMPLE: Delta_ITT = 0.0260, pi_t = 0.85, pi_c = 0.10
                   denominator = 0.75;  Delta_CACE = 0.0260 / 0.75 = 0.03467 -> 3.47 pp
INTERPRETATION: among store-weeks that switch because they were assigned, the effect is 3.5 pp.
```

The exclusion restriction is the fragile part, and it fails in a specific and common way in retail: assignment to the new process changes supervisor attention, so control stores get *less* coaching than they otherwise would have. That is a channel from assignment to outcome that does not pass through the process, and it inflates `Delta_ITT` and therefore `Delta_CACE`. Check it by asking whether the control arm received a documented amount of business-as-usual support comparable to previous periods.

### 4.4 Limits and counterexamples

**Limit 1: running longer stops helping.** From F-1, the per-store variance floor is `sigma_b^2 = rho * sigma^2`. So `Var(mean_i) -> rho * sigma^2` as `m -> infinity`, and the arm-level variance floor is `2 * rho * sigma^2 / n`. With `rho = 0.25`, `sigma = 0.0350`, `n = 12`, the floor SE is `sqrt(2*0.25*0.001225/12) = sqrt(0.0000510417) = 0.0071443`. No amount of extra weeks beats that number. Buying weeks beyond the point where `DEFF` saturates spends exposure without buying resolution.

The saturation point is where `DEFF` stops growing appreciably: with `rho = 0.25`, `DEFF` is 1.75 at `m = 4`, 2.50 at `m = 7`, and 3.25 at `m = 10`. The marginal information in week 10 versus week 9 is roughly `1/(3.25*10) - 1/(3.25*9)` scaled — about 3% of the total. Practically: `m` beyond about `1/rho` weeks adds little.

**Limit 2: `rho` is a property of the metric, not of the stores.** Availability measured at a single audit visits has a high `rho` because a store's shelf condition persists. Sales per square metre has a lower `rho` but a large `sigma`. Choosing the metric with the smaller `rho * sigma^2` product is a legitimate design move and is usually more effective than adding stores.

**Counterexample: the test that cannot fail.** Suppose a design has `n = 2` stores per arm and the analyst plans to read it weekly. The per-arm variance is `DEFF * sigma^2 / (2m)`; at `n = 2, m = 4, DEFF = 1.75`, `Var = 1.75*0.001225/8 = 0.00026797`, `SE_diff = sqrt(2*0.00026797) = 0.023148`. With `df = 2`, the critical value is `4.3027`, so the half-width is `0.0996` — about 10 pp. No plausible 2-3 pp effect can be distinguished from zero, yet with weekly reads over eight weeks there are eight chances to see a difference at the 5% level, giving roughly `1 - 0.95^8 = 33.7%` chance of at least one spurious "significant" week even with no true effect.

That is the arithmetic behind the most expensive class of field-test error: a small test with a frequent read-out produces a confident-looking, wrong answer. The counterexample's structure — too few clusters, too many looks, no pre-declared window — is exactly what Section 7's gates are designed to catch before the money is spent.

**Counterexample: the design effect computed at the wrong level.** If stores are randomised but the analysis regresses individual audit records on treatment with a store fixed effect, the clustering is absorbed for the store mean but the standard error is still computed on the wrong count. The tempting fix — cluster-robust standard errors — is correct only when the number of clusters is adequate for the asymptotics; with 24 clusters the small-sample correction matters, and with fewer than about 40 clusters an interval from a wild cluster bootstrap is the defensible choice.

## 5. Worked Numerical Example

**All inputs in this section are `ILLUSTRATIVE`; all arithmetic is `DERIVED` and re-checked below.**

A grocery format runs a shelf-recovery procedure in 12 stores against 12 control stores for 4 weeks. The primary outcome is shelf availability — the share of audited shelf positions holding the correct, in-date item — measured at two visits per store-week. The guardrails are checkout queue minutes per transaction, overtime hours per store-week, and shrink as a percentage of sales.

### 5.1 Step 1 — assemble the arm means at the correct grain

```text
ARM SIZES AND OUTCOMES (store-week grain, 4 weeks per store)
treated:  12 stores x 4 weeks = 48 store-weeks; mean availability 0.8913
control:  12 stores x 4 weeks = 48 store-weeks; mean availability 0.8653
Delta = 0.8913 - 0.8653 = 0.0260   -> 2.60 pp
```

The arm means are unweighted means of store-week values, so each store contributes 4 observations. Weighting by audit count would give more weight to stores with more audits, which is a different estimand — appropriate if the question is "what is the total availability across audited positions", inappropriate if the question is "what is the typical store-week". State which one you want.

### 5.2 Step 2 — compute the interval with clustering

Using F-2 with `rho = 0.250` (estimated from the control arm's between-store share of variance, `sigma_b^2 = 0.00030625`, `sigma^2 = 0.001225`, so `rho = 0.25`):

```text
DEFF    = 1 + (4-1)*0.250     = 1.7500
Var     = 1.7500*0.001225*2/48 = 0.0000446615      (per-arm, both arms)
SE_diff = sqrt(0.0000893229)   = 0.0094510
df      = 2*12 - 2 = 22;  t_crit = 2.0739
half    = 0.0196006
CI_95   = [0.0063994, 0.0456006]   -> [0.64 pp, 4.56 pp]
```

Interpretation: the interval excludes zero, so the procedure is associated with a positive availability change. But the lower bound is 0.64 pp, and Section 5.5 shows the break-even effect is 2.05 pp. So the interval straddles `D*` and the correct decision under DR-1 is **iterate**, not ship — despite what a headline of "significant at 5%" would suggest.

### 5.3 Step 3 — compliance and fidelity

```text
compliance: treated arm store-weeks with procedure log completed = 41 of 48  -> pi_t = 0.8542
control  arm store-weeks with any procedure use               =  5 of 48  -> pi_c = 0.1042
denominator = 0.8542 - 0.1042 = 0.7500
Delta_CACE  = 0.0260 / 0.7500 = 0.034667  -> 3.47 pp
```

Two checks on the adjustment. First, the denominator 0.75 exceeds the interpretability floor of 0.10 comfortably, so the ratio is meaningful. Second, the control arm's 5 of 48 is not zero, which is expected in a live estate where staff rotate; if `pi_c` had been 0 the adjustment would have been unnecessary, and if `pi_c` had exceeded `pi_t` the monotonicity assumption would have been visibly violated.

Fidelity is separate: 7 of the 41 compliant treated store-weeks used the procedure but skipped the mandatory 20-minute end-of-day review. Fidelity is therefore `34/41 = 0.8293` of compliant weeks fully delivered. The test measured a diluted version of the procedure, which is the honest thing to report and is also the strongest argument for a retest rather than a stop.

### 5.4 Step 4 — guardrails

```text
queue minutes per transaction:   treated 2.41 vs control 2.29  -> +0.12 min (+5.2%)
overtime hours per store-week:   treated 4.9  vs control 4.4   -> +0.5 h (+11.4%)
shrink as share of sales:        treated 1.18% vs control 1.21% -> -0.03 pp
```

The queue guardrail threshold was pre-declared at `+0.20` minutes per transaction; observed `+0.12` is inside it, so `pass`. The overtime threshold was pre-declared at `+1.0` hour per store-week; observed `+0.5` is inside it, so `pass`. Shrink moved in the favourable direction, but the interval on that difference was not computed and the metric is noisy at this sample size, so it is recorded as `not-established` rather than as a benefit.

The overtime figure is the one that changes the economics, and Section 5.5 prices it. A test that ignores the labour consequence of its own procedure will produce a payback that is wrong in the direction that flatters the proposal.

### 5.5 Step 5 — break-even and payback

The break-even effect is the availability gain whose gross margin equals the incremental cost of running the procedure.

```text
FORMULA [F-8]: D* = (c_labour + c_system) / (S * g)
WHERE: c_labour, c_system [currency per store-week] incremental cost;
       S [currency per store-week] sales per store-week; g [fraction] gross margin rate;
       D* [fraction] break-even availability gain, expressed as a share of sales.
CONSTRAINT: the availability gain must convert to sales at rate 1:1, i.e. every recovered
            unit of availability becomes a sale rather than displacing one.
```

With `c_labour = 38.00` per store-week (0.5 h at 76.00 per hour fully loaded), `c_system = 9.00` per store-week (amortised handheld and software), `S = 62000.00` per store-week, `g = 0.28`:

```text
numerator = 38.00 + 9.00            = 47.00
denominator = 62000.00 * 0.28       = 17360.00
D* = 47.00 / 17360.00               = 0.002708  -> 0.271 pp of sales
```

Now convert: an availability gain of `x` pp is assumed to convert into sales at a rate `kappa` — the share of recovered availability that becomes incremental sales rather than shifting demand between items or weeks. With `kappa` ILLUSTRATIVE at 0.35:

```text
required availability gain = D* / kappa = 0.002708 / 0.35 = 0.007737  -> 0.774 pp
```

Compare with the interval: `[0.64 pp, 4.56 pp]`. The lower bound 0.64 pp is *below* 0.774 pp, so the pessimistic end does not clear break-even, and DR-1 gives **iterate**. The point estimate 2.60 pp clears comfortably, which is precisely why a decision note written from the point estimate would have been wrong.

### 5.6 Independent re-check of the arithmetic

Recompute each quantity by a different route, to catch a slip in the first pass.

```text
CHECK 1  SE via sums of squares.  per-arm Var = DEFF*sigma^2/(n*m)
         1.7500*0.001225 = 0.00214375; /48 = 0.0000446615  matches 5.2
CHECK 2  t-statistic via ratio.  0.0260/0.0094510 = 2.7510; squared = 7.5680
         F = t^2 = 7.5680 compared with F(1,22) crit 4.3009  -> same conclusion
CHECK 3  Break-even via unit economics.  0.7735 pp on 62000.00 sales = 479.59 per store-week
         479.59 * 0.28 = 134.29 gross margin; costs 47.00.  Margin/cost = 2.8571
         So break-even availability = 0.7735/2.8571 = 0.2708 pp   matches D* from F-8
CHECK 4  CACE sanity.  0.0260/0.75 = 0.034667; and 0.75*0.034667 = 0.0260  round-trips
CHECK 5  Design floor.  floor SE = sqrt(2*0.25*0.001225/12) = 0.0071443 < 0.0094510,
         so the observed SE is consistent with finite m and not with an error that dropped DEFF
```

Check 5 is the most useful of the five: it is a cheap test that the design effect was actually applied. A reported SE below the floor is arithmetically impossible given the stated `rho` and `n`, and the floor is easy to compute in a spreadsheet before trusting any output.

### 5.7 Sensitivity: which input moves the decision

```text
rho 0.10 -> DEFF 1.300 -> SE 0.0081453 -> CI [0.0091, 0.0429]; lower bound 0.91 pp > 0.774 pp -> ship
rho 0.25 -> DEFF 1.750 -> SE 0.0094510 -> CI [0.0064, 0.0456]; lower bound 0.64 pp < 0.774 pp -> iterate
rho 0.40 -> DEFF 2.200 -> SE 0.0105970 -> CI [0.0040, 0.0480]; lower bound 0.40 pp -> iterate
kappa 0.50 -> required gain 0.542 pp; lower bound 0.64 pp > 0.542 pp -> ship
kappa 0.20 -> required gain 1.354 pp; lower bound 0.64 pp -> iterate
cost 70.00/wk -> D* = 70/17360 = 0.403 pp; required 1.152 pp -> iterate
```

The decision flips on `rho` at the plausible end and on `kappa` across its plausible range. That is the substantive finding: this test's conclusions are driven less by the observed effect than by two inputs that were assumed rather than measured. The correct response is to measure `rho` from the control arm's variance decomposition (cheap, already available) and to bound `kappa` by a small elasticity study or by the retailer's own historical markdown recovery data.

A test whose decision flips on an unmeasured input is not a test result; it is an argument. Section 9 prices what it costs to measure the input instead.
