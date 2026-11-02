> Module 04 — Statistics and Causal Inference
> Topic 09 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 04.06, 04.07, 04.08

# Worked Example: Price Elasticity and Endogeneity

---

## 1. Why this topic exists

Every pricing, promotion, and assortment decision needs one number: how much volume moves when
price moves. The number is almost always produced the same way. Take store-weeks, regress the log
of units on the log of price, read the coefficient, call it the elasticity. The software returns a
coefficient, a standard error, and a p-value. All three are computed correctly.

The regression does not estimate the demand curve. Price and quantity in that table were not
produced by one curve held still while the other moved. They were produced by two curves moving at
once, meeting at an equilibrium in each week. The scatter you fit is the set of intersections, and
its slope is a mixture of the demand slope and the supply slope, weighted by how much of the price
variation came from each side. When demand shocks dominate, the mixture can be positive.

A positive price elasticity is not a small error. It says that raising the price raises the volume
sold. A commercial team that believes it will raise prices and forecast growth, and the forecast
will be wrong in the direction that costs the most money.

The problem is a century old. Moore fitted demand curves to American commodity data in 1914 and got
upward-sloping ones, notably for pig iron; Working explained why in 1927 — the data trace neither
curve cleanly, and Moore had estimated something closer to a shifting supply curve. Wright gave the
instrumental-variable solution in 1928, Haavelmo the formal system in 1944. It remains the most
common defect in commercial elasticity work.

**Why this sits here.** Topic 04.06 gave you the directed acyclic graph (DAG) and the back-door
criterion, which closes a back door by conditioning on a variable. Here the back door runs through
the error term of the demand equation, and the variable that would close it — the demand shock —
is unobserved by construction. No covariate list fixes it. Topic 04.07 gave you randomisation,
which does fix it, and explained why firms rarely randomise price at scale. Topic 04.08 gave you
difference-in-differences, which needs a clean untreated comparison group. This topic is the case
where none of the three is available, and the analyst must find an instrument, exploit panel
structure, or state plainly that the number is not identified.

**Three shapes of the failure**, all routine:

1. **Simultaneity.** The retailer sets price partly in response to expected demand. A competitor
   opens, the manager cuts price to defend volume, and volume falls anyway. Price and quantity move
   together for a reason that has nothing to do with the slope of demand.
2. **Omitted variables that move both.** Competitor pricing, own promotion and display, on-shelf
   availability, season, and upstream cost shocks all shift price and volume at once. Several are
   consequences of price itself, which makes them bad controls in the sense of Topic 04.06.
3. **The wrong counterfactual.** Even a correctly identified elasticity answers a narrow question.
   The response to a cost-driven price change in stores that pass cost through is not the response
   to a strategic price change in a store that holds price fixed. Confusing the two is a local
   average treatment effect (LATE) error, and it survives after the statistics are fixed.

The commercial stakes are concrete. In the worked example in section 6, a naive log-log regression
returns **+0.64**; the true elasticity in the constructed data is **−1.60**. A 5% price increase that
the naive model says raises revenue by 8.4% in fact lowers it by 3.4%. Gross profit still rises — the
margin is 35% and break-even is 2.86 — but the model overstates the gain by about EUR 233,000 a year
across 40 stores. One wrong number that looked fine in the output.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** price elasticity as a ratio of percentage changes, dimensionless, distinct from a
   slope, a semi-elasticity, and a revenue elasticity.
2. **Explain** why regressing log quantity on log price does not recover the demand curve, and
   **state** the two limiting cases in which it recovers one of the curves.
3. **Derive** the simultaneity bias of the ordinary least squares (OLS) slope in a two-equation
   linear-in-logs system and **compute** its size from the ratio of shock variances.
4. **Read** a log-log coefficient as an elasticity, **state** its units and approximation error, and
   **convert** it into a volume, revenue, and gross profit change.
5. **State** the two instrument conditions — relevance and exclusion — and **explain** which can be
   tested and which cannot.
6. **Interpret** a first-stage F, including what 10 does and does not mean, and **explain** how weak
   instruments bias two-stage least squares (2SLS) toward OLS.
7. **Explain** why an instrument identifies a LATE for compliers rather than a population average,
   and **name** that subgroup in a specific supply chain setting.
8. **Quantify** what panel fixed effects buy and cost, and **state** when they do not solve
   simultaneity at all.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture in your head: a price–quantity plane, a downward-sloping line, an upward-sloping
line, and a cloud of dots. The two lines are demand and supply. The dots are the only thing you
observe. Each dot is where the lines crossed in one week in one store.

```text
   price
   EUR/unit
      ^
 3.10 |                                o
      |                           o
 2.95 |                      o             S  (supply)
      |                 o           .
 2.80 |            o          .           /
      |        o        .               /
 2.65 |    o       .                 /        o  observed equilibria
      |        .       o           /           (fitted slope about
 2.50 |  .        o           .   /             +0.6 in log points)
      |       o        .       /
 2.35 |   o       .        .  /
      |        .      o   . /
 2.20 |  .      o    .  . /
      |   .  o    .  . /
      |  . /  .  . /
      +----------------------------------------> quantity, units/store/week
        700   850  1000  1150  1300  1450  1600

            \   D  (demand, slope -1.6 in log points)
             \
              \
```

The dots lie on neither line. That is the whole topic. Everything else is machinery for saying
which line the dots are closest to, and why.

### 3.1 Why the dots lie where they do

Write the two curves in logs, with lowercase letters for logarithms:

```text
   demand:   q = a - e*p + u        e > 0 is the elasticity you want
   supply:   q = b + h*p + v        h > 0 is the supply slope
```

`u` is a demand shock: a local event, a competitor's move, weather, a payday. `v` is a supply shock:
a cost change, a freight surcharge, a harvest. Both are unobserved. Both move the equilibrium.
Equating and solving gives the reduced form:

```text
   p = (a - b + u - v) / (e + h)
   q = (h*a + e*b + h*u + e*v) / (e + h)
```

Regressing `q` on `p`, the probability limit of the OLS slope is the covariance ratio:

```text
   b_OLS = (h*var(u) - e*var(v)) / (var(u) + var(v))
         = -e + (e + h) * var(u) / (var(u) + var(v))
```

Read the second line twice. The OLS slope equals the negative of the true elasticity, **plus a
simultaneity premium**: the sum of the two slopes multiplied by the share of price variation that
comes from demand shocks. It is not a small-sample problem or a functional-form problem. It is
present in the population, with infinite data.

| Share of price variance from demand shocks | What OLS estimates | What the dots trace |
|---:|---|---|
| 0% (only cost moves) | `-e` — the demand curve | The demand curve |
| 100% (only demand moves) | `+h` — the supply curve | The supply curve |
| In between | A weighted mixture of `-e` and `+h` | Neither curve |

The middle row is the normal case. Cost moves and demand moves in the same weeks, in unknown
proportions, and the regression returns a number that is a property of the data-generating process
rather than of consumer behaviour.

> **Note:** The formula assumes a linear-in-logs system with additive, independent shocks and no
> dynamics; under those assumptions the bias is exact. In a real category with cross-item
> substitution, reference-price effects and inventory carry-over it is no longer additive, but the
> qualitative result holds. Do not quote the arithmetic as a bias correction for a real estimate.

### 3.2 The two limiting cases

```text
   CASE A — supply shifts, demand fixed     CASE B — demand shifts, supply fixed
   P ^                                      P ^
     |   S1  S2  S3                           |         / S
     |  /   /   /            D               |        /     D2  D1  D0
     | /   /   /           /                |       /       \   |   /
     |/   /   /          /                  |      /         \  |  /
     +--*- --*---*------ > Q               |     /           \ | /
        dots on D                           |    /             \|/
                                            |   /               *  dots on S
                                            +------------------ > Q
```

Case A happens when supply is volatile and demand is stable: a commodity whose cost swings weekly
while preference is steady. Case B produces wrong signs — a promoted consumer good where cost
changes slowly but local demand swings violently. Classifying a category into A or B does most of
the work. If it is Case B, the naive regression does not estimate demand slightly badly; it
estimates something else entirely.

### 3.3 Four properties worth memorising

- **More data does not help.** The bias is in the probability limit; a larger sample only tightens
  the interval around a wrong number.
- **The sign of the bias is not determined by theory.** It depends on the relative variance of demand
  and supply shocks, so two categories with identical true elasticities can give `−2.1` and `+0.6`.
- **Adding covariates helps only if they are the shocks.** A control correlated with price but not
  with demand is an instrument entered as a covariate — a bad control in the sense of Topic 04.06.
- **The elasticity you want depends on the counterfactual.** A cost-driven price increase and a
  strategic one are different treatments, so there may be no single "the elasticity" to report.

### 3.4 The identification taxonomy

| Design | Source of price variation | What it identifies | Main threat |
|---|---|---|---|
| Naive OLS | All observed price variation | A mixture of demand and supply slopes | Simultaneity; omitted variables |
| Panel fixed effects | Within-unit changes over time | Response to within-unit price changes | Time-varying shocks remain; between variation discarded |
| Instrumental variables | The part of price moved by the instrument | Response for units the instrument moved | Exclusion restriction; weak instrument |
| Experiment | Randomly assigned price | Response in the experimental population | Cost, ethics, interference, scale |

None is uniformly better. The experiment is cleanest and least representative of scale operations.
The instrument is most portable and most fragile, because its validity rests on an argument the
data cannot check. Panel fixed effects are the most available and the most over-credited. The naive
regression is the most common and the only one that is not a design at all.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Price elasticity of demand | Percentage change in quantity demanded divided by the percentage change in price that caused it | Called "the slope of the demand curve" | Dimensionless |
| Own-price elasticity | Elasticity with respect to the product's own price | Reported without saying which price (shelf, invoice, net) | Dimensionless |
| Cross-price elasticity | Percentage change in quantity of A per 1% change in the price of B | Read as an own-price elasticity in a multi-item model | Dimensionless |
| Arc elasticity | Elasticity between two points computed on midpoint averages | Reported as a point elasticity | Dimensionless |
| Point elasticity | `(dQ/dP) x (P/Q)` evaluated at one price | Applied far outside the price range where it was computed | Dimensionless |
| Log-log specification | Regression of `ln Q` on `ln P` and controls; the price coefficient equals the elasticity | Read as a semi-elasticity | Natural logs of the raw variables |
| Semi-elasticity | Percentage change in quantity per one-unit change in a level regressor | Called an elasticity | Percent per unit of the regressor |
| Log point | A difference in natural logarithms, approximately a proportional change for small values | Treated as an exact percentage | `100 x Δln` |
| Structural equation | An equation derived from behaviour — demand, supply, cost | Any regression labelled "structural" | Coefficients in units of the variables |
| Reduced form | An endogenous variable expressed as a function of exogenous variables only | Confused with the structural equation it came from | Stated units |
| Endogeneity | Correlation between a regressor and the error term in the equation of interest | Used loosely to mean "biased" | None |
| Simultaneity | Two or more variables determined jointly in equilibrium | Treated as a synonym for omitted-variable bias | None |
| Omitted-variable bias | Bias from an unobserved common cause of regressor and outcome | Used for every kind of misspecification | None |
| Identification | Whether a parameter is uniquely recoverable from the distribution of observables | Used to mean "we managed to estimate it" | None |
| Instrument | A variable that shifts the endogenous regressor and affects the outcome only through it | Any variable correlated with price | The instrument's own unit |
| Relevance | The instrument is correlated with the endogenous regressor, conditional on controls | Asserted from theory without a first stage | First-stage F statistic (unitless) |
| Exclusion restriction | The instrument affects the outcome only through the treatment | Tested as though it were an empirical hypothesis | None; not testable in the just-identified case |
| Independence | The instrument is as-good-as-randomly assigned with respect to the outcome | Confused with relevance | None; not testable |
| Monotonicity | The instrument moves treatment in the same direction for every unit | Left unstated in almost all applied work | None |
| First stage | The regression of the endogenous regressor on the instruments and controls | Reported without its specification or controls | Coefficient in regressor units per instrument unit |
| First-stage F | The F statistic on the excluded instruments in the first stage | Compared with 10 as if it were a pass mark | Unitless |
| Weak instrument | An instrument with too little first-stage signal to identify the parameter reliably | Defined by whether the first stage is significant at 5% | F statistic, or an effective F under clustering |
| Two-stage least squares (2SLS) | The estimator that replaces the endogenous regressor with its first-stage fitted value | Called "IV" without naming the estimator | Coefficient units |
| Local average treatment effect (LATE) | The average effect for the units whose treatment status the instrument changed | Reported as the average treatment effect | Outcome units per treatment unit |
| Complier | A unit whose treatment follows the instrument | Treated as an observable subgroup | Indicator |
| Always-taker, never-taker | Units treated, or untreated, whether or not the instrument fires | Ignored in interpretation | Indicator |
| Defier | A unit whose treatment moves opposite to the instrument | Assumed absent without argument | Indicator |
| Within estimator | Regression on unit-demeaned data; the fixed-effects estimator | Described as "controlling for the store" | Coefficient units |
| Fixed effects | Unit or time intercepts that absorb level differences | Assumed to solve endogeneity as such | Indicator variables |
| Between and within variation | Variation across units versus variation within units over time | Reported as one variance figure | Variance components in squared log points |
| Pass-through | The fraction of a cost change that reaches the retail price | Confused with the elasticity of demand | Proportion, or elasticity of price with respect to cost |
| Cost shifter | A variable that moves supplier cost and therefore price | Assumed not to move demand | Currency per unit, or an index |
| Scanner data | Transaction records from retail point of sale, aggregated to item, store, and week | Treated as clean experimental data | Item x store x week |
| Gross margin | `(Price - unit cost) / Price` | Confused with markup on cost | Proportion or percent |
| Break-even elasticity | The absolute elasticity at which a price change leaves gross profit unchanged; equals `1 / margin` | Not computed before a pricing decision | Dimensionless |

**Relevance versus exclusion.** Relevance can be checked: regress the endogenous variable on the
instrument and read the F statistic. Exclusion cannot be checked in the just-identified case, and
with surplus instruments the overidentification test only rejects if the instruments disagree with
each other — a set that is invalid in the same direction passes. That asymmetry is why the
exclusion argument in a pricing study is prose, not a result. If you cannot write three sentences a
sceptical category manager would accept, you do not have an instrument.

**LATE versus average treatment effect.** LATE is not a technicality. The number belongs to the
units the instrument moved. In a retail panel the compliers are the store-weeks where cost changes
reached the shelf price. Stores on a fixed national price contribute nothing; stores that absorb
cost into margin contribute nothing. The estimate describes the pass-through segment, and the
decision may concern a different one.

## 5. The mechanics, step by step

Six phases. Each ends with an artifact a reviewer can inspect. They are ordered so that a failure
early invalidates the later ones, which is why skipping to estimation is expensive.

### Phase A — Fix the estimand and the decision

**Step 1. Name the decision the elasticity will change.** A list price change, a promotional depth,
a zone price architecture, a trade term, a markdown cadence — and whose behaviour is measured: the
retailer's shoppers, a distributor's orders, or a manufacturer's shipments. These have different
elasticities and are routinely confused. Artifact: "the decision is X, taken by role Y, at cadence
Z, and the elasticity enters as the response of W to V."

**Step 2. State the counterfactual.** A cost-driven increase, a strategic increase, and a
promotional cut are three treatments and may have three responses. If the decision is strategic and
identification comes from cost variation, say so now. Artifact: the treatment in one line, with its
time window.

**Step 3. Choose the outcome and its unit.** Units, revenue net of trade spend, or gross profit. An
elasticity of units plus a margin assumption produces a profit answer; an elasticity of revenue
does not, except under constant elasticity.

### Phase B — Write the system before you write the regression

**Step 4. Write both equations.** Demand: quantity as a function of price, demand shifters, and an
error. Pricing: price as a function of cost, competitive pressure, and an error. Retail pricing is
a markup rule rather than a competitive supply curve, but it plays the same role — it makes price
endogenous.

**Step 5. Check identification by the order condition.** Count the excluded exogenous variables.
With one endogenous regressor you need at least one variable in the pricing equation and not in the
demand equation. If the count is zero, the elasticity is not identified from these data by any
estimator. Stop and say so.

**Step 6. Write down what the errors contain.** Demand error: local taste shocks, competitor
actions not in the data, weather, income shocks. Pricing error: cost changes the index misses,
managerial discretion, competitive response. That list is the candidate set for instruments and for
the sensitivity analysis of Topic 04.10.

### Phase C — Choose the design

**Step 7. Rank designs by credibility, not convenience.** Randomised price test; natural experiment
or policy change; instrument; panel fixed effects as a partial fix; naive regression only with an
explicit statement that it is descriptive.

**Step 8. Decide what the panel can and cannot remove.** Store fixed effects remove time-invariant
store differences: format, region, catchment income, baseline preference. Week fixed effects
remove common shocks. The store-week shock — what simultaneity is made of — survives both. That is
why fixed effects often shrink the naive estimate without fixing the sign.

**Step 9. Write the exclusion argument in three sentences.** What moves the instrument. Why that
movement reaches quantity only through price. The most plausible way the argument fails, and the
direction of the resulting bias. If you cannot write the third, you have not understood your own
instrument.

### Phase D — Diagnose the instrument

**Step 10. Estimate the first stage in full.** It is a regression, not a formality: coefficient,
standard error, controls, fixed effects, clustering. Check the sign. A cost increase should raise
the shelf price; a negative coefficient on a cost index in raw price levels is a data problem.

**Step 11. Compute the F correctly.** With one instrument the F equals the square of its t
statistic. Under clustering, use a cluster-robust or effective F; the classical F overstates
strength when errors correlate within clusters. The number 10 is a heuristic from Staiger and Stock
(1997), roughly corresponding to 2SLS bias of 10% of the OLS bias in the simplest case. It is not a
test, and it is too lax with many instruments.

**Step 12. Decide what to do if the instrument is weak.** Do not report 2SLS with conventional
standard errors and move on. Report weak-instrument-robust intervals (Anderson–Rubin), apply the
t-ratio correction of Lee, McCrary, Moreira and Porter (2022), use LIML, or find a stronger
instrument. Report OLS alongside: a weak instrument pulls 2SLS toward OLS, and the comparison is
diagnostic.

### Phase E — Estimate and interpret

**Step 13. Read the coefficient.** In `ln Q = a + b ln P + gX + e`, `b` is the elasticity: a 1% price
increase changes quantity by about `b` per cent. The approximation is good to about 10% price
changes; beyond that compute `exp(b ln(1 + Δ))`. With multiple items, `b` is a category average
unless the model allows item-specific slopes.

**Step 14. Convert to decision units.** Volume `Q1 = Q0 x (P1/P0)^b`; revenue `R1 = P1 x Q1`; gross
profit `GP1 = (P1 - c) x Q1`. Compare with the break-even elasticity `1/margin`: above it, a price
increase destroys gross profit even when it raises revenue per unit.

**Step 15. State the LATE.** Name the compliers in business language — which stores, which weeks,
which price changes — and say whether the decision concerns those units or others.

### Phase F — Stress-test and report

**Step 16. Run the three comparisons.** OLS against 2SLS gives the endogeneity bias; 2SLS against
LIML or a weak-IV interval gives instrument strength; 2SLS with and without fixed effects shows what
the panel was buying. Report the exclusion argument and its leading threat in the memo body, not an
appendix.

**Step 17. Report the decision under the interval, not the point.** An elasticity interval
translated into a revenue or profit interval is what the decision-maker needs; a point estimate
invites confidence in exactly the dimension where the design is weakest.

### 5.1 Why the log-log form, and what it costs

| Reason | Benefit | Cost |
|---|---|---|
| Coefficient is the elasticity directly | No conversion; comparable across items | Forces constant elasticity over the observed range |
| Reduces skew in price and quantity | Closer to homoskedastic errors | Drops zero-quantity rows, which are informative |
| Multiplicative interpretation | A 10% change has the same proportional effect at any baseline | Cannot represent saturation or a stock-out ceiling |

The constant-elasticity restriction matters more than most analysts admit. If demand is more
elastic at high prices — common in categories with reference prices — a single coefficient is an
average over the observed range, and extrapolating beyond it overstates the volume loss.

### 5.2 What panel fixed effects actually remove

The within transformation subtracts each unit's mean:

```text
   y_it = a_i + b*x_it + e_it
   y_it - mean_i(y) = b*(x_it - mean_i(x)) + (e_it - mean_i(e))
```

The unit effect `a_i` vanishes whether or not it is observed, and whether or not it is correlated
with `x`. That is the appeal. The cost is threefold: time-invariant regressors vanish with it, so
format, region, and supplier country become inestimable; between variation is discarded, so if most
price variation is cross-sectional the estimator uses a small and possibly unrepresentative slice
and standard errors rise; and time-varying confounders remain, because consistency needs the
residual error to be uncorrelated with the residual regressor, which a store-week demand shock
violates.

> **Note:** Two-way fixed effects with staggered treatment timing is a separate and well-documented
> problem in the difference-in-differences literature. This topic uses fixed effects only as a
> control for time-invariant confounders, never as identification for a treatment rolling out over
> time.

## 6. Worked example

A grocery retailer sells one stock-keeping unit (SKU): a 500 g pack of roasted coffee, in 40 stores
over 52 weeks — 2,080 store-weeks. Mean shelf price EUR 2.50, mean volume 1,000 units per
store-week. The commercial team is considering a 5% list price increase.

All figures are constructed from a stated structural system so every step can be checked by hand.
They are not a claim about any real retailer, and the parameters were chosen to make the arithmetic
exact rather than to represent a category.

Throughout, `p` and `q` are log points multiplied by 100, as deviations from the sample mean.
`p = +9` means a price 9 log points above the mean: EUR 2.50 x exp(0.09) = EUR 2.735. `q = −3.36`
means 1,000 x exp(−0.0336) = 967 units. The elasticity is invariant to this rescaling of logs, so
coefficients remain elasticities.

### 6.1 The naive estimate, computed by hand

| Store-week | `p` | `q` | `p x q` | `p²` |
|---|---:|---:|---:|---:|
| S01 W03 | −9 | −3.36 | 30.24 | 81 |
| S04 W11 | −7 | −3.68 | 25.76 | 49 |
| S07 W18 | −5 | −3.60 | 18.00 | 25 |
| S11 W22 | −3 | −3.12 | 9.36 | 9 |
| S14 W27 | −1 | −2.24 | 2.24 | 1 |
| S19 W31 | 1 | −0.96 | −0.96 | 1 |
| S23 W36 | 3 | 0.72 | 2.16 | 9 |
| S28 W41 | 5 | 2.80 | 14.00 | 25 |
| S33 W46 | 7 | 5.28 | 36.96 | 49 |
| S38 W50 | 9 | 8.16 | 73.44 | 81 |
| **Sum** | **0** | **0** | **211.20** | **330** |

No intercept is needed because both variables are centred. The slope is the ratio of the sums:

```text
   b_OLS = sum(p*q) / sum(p²) = 211.20 / 330 = 0.64
```

The full panel gives the same answer: `Cov(p,q) = 26.4`, `Var(p) = 41.2`, and `26.4 / 41.2 = 0.6408`.
An elasticity of **+0.64** states that a 1% price increase raises volume by 0.64% — a Giffen good,
which coffee is not. Two responses are available. Report it, because the regression did what it was
asked. Or ask what process generated it. The rest of this section is the second response.

### 6.2 The system that produced the data

```text
   demand:   q = 8.00 - 1.60*p + u        true elasticity e = 1.60
   pricing:  q = 6.00 + 1.20*p + v        pass-through slope h = 1.20
```

Errors are mean-zero and independent with standard deviations 16.1 and 8.0 log points, so
`Var(u) = 258.3` and `Var(v) = 64.6`. The second equation is a pricing rule, not a cost curve:
retailers set price as a markup over cost, and the rule behaves like a supply curve in the system.

```text
   p = (8.00 - 6.00 + u - v) / (1.60 + 1.20) = (2.00 + u - v) / 2.80
   q = (1.20*8.00 + 1.60*6.00 + 1.20*u + 1.60*v) / 2.80
     = (19.20 + 1.20*u + 1.60*v) / 2.80
```

Three consequences, each exact in this system. Price variance decomposes additively:
`(258.3 + 64.6) / 7.84 = 41.2`, matching the observed variance, of which 258.3/322.9 = 80% comes
from demand shocks and 20% from cost. The OLS slope is the variance-weighted mixture:
`(1.20 x 258.3 − 1.60 x 64.6) / 322.9 = 0.640`. And the sign is set by which shock dominates: demand
variance is four times cost variance, so the supply-like term wins and the coefficient turns
positive.

> **Note:** The 80/20 split is the whole story. The naive regression is not estimating demand badly;
> it is estimating an 80/20 blend of the supply and demand slopes. Change the split to 20/80 and the
> same regression returns −1.31 with consumer behaviour unchanged: the estimate is a property of the
> pricing process, not of the shopper.

### 6.3 The simultaneity bias, decomposed

```text
   b_OLS = -e + (e + h) * var(u)/(var(u)+var(v))
         = -1.60 + (1.60 + 1.20) * (258.3 / 322.9)
         = -1.60 + 2.80 * 0.800
         = -1.60 + 2.24
         = +0.64
```

| Component | Value | Meaning |
|---|---:|---|
| True elasticity `-e` | −1.60 | What the business wants |
| Sum of slopes `e + h` | 2.80 | How far apart the curves are |
| Demand share of price variance | 0.800 | How much price movement is demand-driven |
| **Simultaneity premium** | **+2.24** | `2.80 x 0.800` |
| **Naive estimate** | **+0.64** | `−1.60 + 2.24` |

The premium is 2.24 elasticity points, 1.4 times the parameter being estimated. It is bounded: it
cannot exceed `e + h` and is zero when the demand share is zero. That bound is useful when auditing
a suspicious estimate — if a naive elasticity is implausibly far from any plausible true value,
something other than simultaneity is also wrong.

### 6.4 The instrument, and the estimate from moments

The obvious instrument is the wholesale cost index for green coffee, measured at store-week level
and centred. Relevance holds by construction. Exclusion holds if the cost movement reaches shoppers
only through the shelf price — no simultaneous advertising campaign, no packaging redesign, no
availability effect tied to the cost change.

With one instrument the estimand is a ratio of covariances, `b_IV = cov(c,q) / cov(c,p)`:

| Moment | Value | Note |
|---|---:|---|
| `Var(p)` | 41.2 | Total price variation |
| `Cov(p,q)` | 26.4 | Produces the naive slope 0.64 |
| `Var(c)` | 38.0 | Instrument variation |
| `Cov(c,p)` | −14.8 | Cost up, price up; opposite signs because `p` enters the reduced form negatively |
| `Cov(c,q)` | 23.7 | Cost up, quantity up along the supply-like relation |

```text
   b_IV = 23.7 / (-14.8) = -1.601
```

The instrument recovers −1.60 to three decimals. This is not a miracle of econometrics; it is the
definition of the reduced form. Because `Cov(c,q)/Cov(c,p)` equals `−e` identically in this system,
any variable correlated with `v` and uncorrelated with `u` returns the same number. That identity
is what identification means.

### 6.5 The first stage and what the F means

```text
   pi_1 = Cov(c,p) / Var(c) = -14.8 / 38.0 = -0.390

   R²_1 = Cov(c,p)² / (Var(c)*Var(p)) = 14.8² / (38.0*41.2)
        = 219.04 / 1565.6 = 0.140

   F = [R²_1/(1-R²_1)] * (n-2) = (0.140/0.860) * 2078 = 338.3
```

An F of 338 is far above any threshold, so the instrument is strong on the classical test. Two
qualifications. First, with store-clustered standard errors the effective F is 41.6, because price
and cost are correlated within stores across weeks, so 2,080 observations are not 2,080 independent
pieces of information. This is normal in panel data, and the cluster-robust or effective F should
be the reported number. Second, strength is not validity. A strong instrument with a violated
exclusion restriction returns a precise wrong answer, and the F says nothing about that.

The weak case is worth computing. Take a distant futures index lagged two quarters, diluted by
contract lags and hedging. Its first-stage R² is 0.0026, giving F = 5.4. Under the standard
approximation, 2SLS retains roughly `1/F` of the OLS bias:

```text
   retained bias = (1/5.4) * 2.24 = 0.41
   plausible 2SLS estimate = -1.60 + 0.41 = -1.19
```

Still negative, but a third of the way back toward the naive value, with a confidence interval that
understates the uncertainty. At F = 1.5 the same arithmetic gives −0.11, indistinguishable from
zero. A weak instrument does not fail loudly; it drifts toward the number you were escaping.

### 6.6 What the instrument identifies, and for whom

The instrument is cost variation, so the compliers are store-weeks where a cost change reached the
shelf price.

| Group | Definition in this setting | Share |
|---|---|---:|
| Compliers | Cost change passed into shelf price within the quarter | 78% |
| Never-takers | Fixed national price for the window; price does not respond to cost | 14% |
| Always-takers | Price at a local ceiling or floor; cannot move either way | 6% |
| Defiers | Cut price while cost rose, typically matching a local competitor | 2% |

The 2% of defiers is the problem. Monotonicity is violated, so the estimand is not a clean weighted
average. Identify them, report the share, and check whether excluding them moves the estimate.

Even with monotonicity, the estimate belongs to the 78%. It answers: for store-weeks where a cost
change moved the shelf price, how did volume respond. It does not answer what happens if the
category team raises price 5% next quarter for strategic reasons, in stores that may not be
compliers, at a moment that may not resemble a cost shock. If demand is more elastic at high
prices, and cost-driven increases happen when prices are already high, the IV estimate overstates
the volume loss from an increase taken from the current level.

### 6.7 The panel alternative, and its cost

| Component | Squared log points | Share of total |
|---|---:|---:|
| Total price variance | 41.2 | 100% |
| Between-store | 25.5 | 62% |
| Within-store | 15.7 | 38% |
| Within-store, after week effects | 11.4 | 28% |

Two-way fixed effects, store and week dummies, give **+0.18**: toward zero from +0.64 but not a sign
change. This is the honest result and the one most often suppressed. Fixed effects removed the
store and week level differences, but the simultaneity lives in the store-week demand shock, which
survives both dummies. The estimator is consistent only if the remaining error is uncorrelated with
the remaining price variation, and here it is not. Using 28% of the variance instead of 100% also
inflates the standard error by `sqrt(41.2/11.4) = 1.90`, so the estimate is both biased and
imprecise.

> **Note:** Fixed effects do not necessarily shrink a simultaneity bias. Whether they do depends on
> whether demeaning removes more demand-shock variance than cost-shock variance, an empirical
> property of the panel. Here the bias falls from 2.24 to about 1.78; in another panel it could
> rise. Never report "we added fixed effects, so the estimate is now causal".

### 6.8 The decision, under each estimate

Baseline per store-week: price EUR 2.50, volume 1,000, unit cost EUR 1.625, gross margin 35%.
Revenue EUR 2,500, gross profit EUR 875. The proposal is +5% to EUR 2.625.

| Quantity | Naive (`b = +0.64`) | IV (`b = −1.60`) | Difference |
|---|---:|---:|---:|
| Volume change | +3.2% | −8.0% | 11.2 points |
| Units | 1,032 | 920 | −112 |
| Revenue | EUR 2,709 | EUR 2,415 | −EUR 294 |
| Cost of goods | EUR 1,677 | EUR 1,495 | −EUR 182 |
| Gross profit | EUR 1,032 | EUR 920 | −EUR 112 |
| Revenue vs baseline | +8.4% | −3.4% | 11.8 points |
| Gross profit vs baseline | +17.9% | +5.1% | 12.8 points |

Across 40 stores and 52 weeks the overstatement of the annual profit gain is
`112 x 40 x 52 = EUR 232,960`, on a baseline gross profit of `875 x 2,080 = EUR 1,820,000` — 12.8%
of the profit base.

Two readings matter. **Revenue and profit disagree, and that is normal.** At a 35% margin the
break-even elasticity is `1/0.35 = 2.86`: any increase is gross-profit-accretive while the true
elasticity is below 2.86 and revenue-dilutive above 1.0. Judged on revenue the increase is
rejected; judged on gross profit it is accepted. **The sign error is invisible in the fit.** The
naive regression has R-squared 0.20 and t = 22.8. Every summary statistic is healthy. Only the
structural reasoning flags it.

### 6.9 What an analyst should have asked

Is the price the shelf price, the invoice price, or the net price after trade spend? They differ by
6–11% here. Are stock-out weeks in the sample? Dropping them selects on availability, which price
causes through sell-through. Is the cost index measured before or after supplier hedging, and does
it vary within store over time? Is the counterfactual cost-driven pass-through or strategic
repricing? Each is answerable before estimation, each changes the estimate, and none requires a
better model.

## 7. Code, query, or calculation

One block, deliberately short: three estimates from the same columns, with the within transformation
written out so the fixed-effects step is not a black box.

```python
# df columns: store, week, ln_q, ln_p, ln_cost  (logs, already centred)
import numpy as np
from linearmodels.iv import IV2SLS

ols = IV2SLS(df.ln_q, None, df[["ln_p"]], None).fit(cov_type="clustered", clusters=df.store)
iv  = IV2SLS(df.ln_q, None, df[["ln_p"]], df[["ln_cost"]]).fit(
          cov_type="clustered", clusters=df.store)
fs  = IV2SLS(df.ln_p, None, df[["ln_cost"]], None).fit(
          cov_type="clustered", clusters=df.store)

t_fs = fs.params["ln_cost"] / fs.std_errors["ln_cost"]
print("OLS", round(ols.params["ln_p"], 3), "| 2SLS", round(iv.params["ln_p"], 3),
      "| first-stage t", round(t_fs, 2), "F =", round(t_fs**2, 1))

# two-way within transformation: subtract store and week means, add back the grand mean
d = df.copy()
for c in ["ln_q", "ln_p"]:
    d[c] = (d[c] - d.groupby("store")[c].transform("mean")
                 - d.groupby("week")[c].transform("mean") + d[c].mean())
print("TWFE", round(np.polyfit(d.ln_p, d.ln_q, 1)[0], 3))
```

Three checks before trusting the output. A negative first-stage coefficient on cost usually means a
coding error or a price already net of cost, and a 2SLS standard error below the OLS one means the
instrument is not doing what you think. The squared t equals the F only with one instrument and no
other excluded regressors; otherwise use the joint test.

## 8. Data, standards, and regulatory context

An elasticity is only as good as its price variable, and the price variable is almost never a
single number.

### 8.1 What the dataset must contain

| Field | Why it is needed | Failure if absent | Typical source |
|---|---|---|---|
| Item identifier | Defines the unit whose demand is estimated | Aggregating across substitutes biases the elasticity toward zero | GTIN in the item master |
| Store identifier | Defines the panel unit and clustering level | Standard errors understated; fixed effects impossible | GLN or internal store number |
| Week identifier | Defines the time unit and fixed-effect level | Seasonality absorbed into the price coefficient | Fiscal calendar, stated |
| Shelf price | The price the shopper responds to | Invoice or list price gives the wrong treatment | Point-of-sale (POS) price file |
| Net price after trade spend | The price the retailer actually receives | Margin arithmetic is wrong | Trade promotion management system |
| Units sold | The outcome | Revenue as outcome changes the estimand | POS scan data |
| Promotion flag and display | Demand shifters correlated with price | Omitted-variable bias, usually toward zero | Promotion calendar |
| On-shelf availability | Separates demand from rationing | Stock-out weeks look like low demand | Inventory or planogram audit |
| Unit cost | Converts volume to profit; supplies the instrument | No break-even; no identification | Supplier invoices, cost index |
| Competitor price | The most important omitted variable in most categories | Bias of unknown sign and size | Syndicated audit or web scrape |

The last row is the one most often missing and most often fatal: a competitor's price cut in the same
week produces a price–quantity pair that looks like a strong demand response and is nothing of the
kind.

### 8.2 Identity and price-display standards

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN | Trade item identity | That the SKU in the price file is the SKU in the sales file | B |
| GS1 GPC | Product classification | Comparability across categories and markets | B |
| GS1 GLN | Location identity | Store and legal-entity joins across systems | B |
| EPCIS 2.0 | Event data | Whether a price change or stock-out is timestamped at event level | B |
| ISO 8601 | Date and time representation | Whether "week" means the same thing in two systems | B |

A GTIN change from a packaging refresh splits one item's history into two, and a log-log regression on
the split history estimates a different elasticity from the same consumer behaviour. Such item-master
changes are invisible in the regression output.

### 8.3 Price-display regulation

| Requirement | Jurisdiction | What it constrains | Grade |
|---|---|---|---|
| Price Indication Directive 98/6/EC, as amended by Directive (EU) 2019/2161 | European Union | An announced price reduction must state the prior price, defined as the lowest price applied in at least the 30 days before the reduction | B |
| Unit price display rules | European Union, member-state variants | Price per unit of measure for pre-packaged goods | B |
| FTC Guides Against Deceptive Pricing, 16 CFR Part 233 | United States | Comparative advertising may not reference a price that was not genuinely offered | B |
| National unfair-commercial-practices regimes | Member states, United Kingdom | Enforcement and penalties for misleading reference prices | B |

The "regular price" in the transaction file may not be the reference price the promotion is
legally measured against, and the "was" price is a regulated construct rather than a free
variable. An elasticity estimated on a reference price the retailer is not permitted to advertise
is an elasticity for a price the shopper never saw.

> **Note:** Regulatory summaries here are educational and may lag amendment. Confirm against the
> current consolidated text before relying on any definition in a compliance context. This is not
> legal advice.

### 8.4 Accounting definitions that change the number

| Definition | Where it bites | Why the elasticity changes |
|---|---|---|
| Gross versus net revenue | Trade allowances, slotting fees, rebates | An elasticity on gross price overstates the margin lever |
| Gross margin versus markup | `(P-c)/P` versus `(P-c)/c` | Break-even is `1/margin`, not `1/markup` |
| Variable consideration under IFRS 15 / ASC 606 | Promotional allowances reduce the transaction price | The recorded price is already net of some promotion |
| Standard cost versus marginal cost | Standard cost includes absorption and variance accounts | The instrument must proxy marginal cost, not an accounting number |
| Shrink and waste | Perishable and high-theft categories | Units sold understate units demanded |

The fourth row is the quiet one. A standard cost revised annually moves for accounting reasons
rather than market reasons, and its variation is often disconnected from the price the retailer
sets week to week.

### 8.5 Privacy and data governance

Loyalty and scanner data identify households, and in the European Union the General Data Protection
Regulation (GDPR, Regulation (EU) 2016/679) constrains their use. Elasticity work usually uses
aggregated store-week data, which avoids most of the problem, but small-cell suppression can bias
availability measures in low-volume stores, and syndicated licences often prohibit onward sharing
even after aggregation. Neither changes the econometrics; both change what you may publish.

## 9. Industry practice and case evidence

### 9.1 The identification problem, documented

- Moore (1914) fitted demand curves to American commodity data and obtained upward-sloping
  estimates, including for pig iron. The original anomaly.
- Working (1927), "What Do Statistical 'Demand Curves' Show?", *Quarterly Journal of Economics* 41(1),
  212–235, showed the data are intersections of shifting curves and that Moore had estimated
  something closer to a supply curve. First presented in 1926, hence the alternative citation date.
  Grade A.
- Wright (1928), Appendix B, gave the first instrumental-variable derivation, documented in Stock
  and Trebbi's history of the idea; Haavelmo (1944) formalised simultaneous-equations estimation.
  Grade A.

The estimator was invented after the problem was understood, not the reverse. An analyst who
reaches for a software routine before drawing the two curves repeats Moore's mistake with better
tools.

### 9.2 Scanner-data evidence

| Study | Data and design | Reported finding | Grade |
|---|---|---|---|
| Dhar, Chavas and Gould (2002) | IRI Infoscan, 46 cities, 9 carbonated-soft-drink brands; Almost Ideal Demand System with instruments | Rejected exogeneity of price and expenditure; own-price elasticities roughly 90% larger once endogeneity was addressed | A |
| Rosenthal-Kay, Traina and Tran (2021) | Nielsen RMS scanner data, about 14,000 products; log-linear demand with brand-market-quarter-year fixed effects; Hausman-style shift-share instrument using the chain's average log price in other markets | 7.5 million elasticities; documents a practical instrument at production scale | C |
| Chernozhukov, Hausman and Newey (NBER 26424) | Nielsen Homescan, Houston, 2004–2006; control functions and debiased machine learning | Panel elasticities substantially smaller than cross-sectional ones; prices correlate with preferences | C |

The pattern is the one this topic predicts: naive estimates are biased, the direction is usually
toward zero in absolute value for promoted categories, and the correction is large enough to change
decisions. The figures are as reported in the sources retrieved; they are not re-derived here.

### 9.3 The problem in automated pricing systems

The modern version is not a spreadsheet regression. It is a forecasting model that treats price as
one more feature, trained on data generated by a policy that responded to demand. The model learns
the policy, not the response.

| System type | How it handles price | Failure mode | Grade |
|---|---|---|---|
| Gradient-boosted demand forecast with price as a feature | Price enters as a covariate beside weather and promotion | Learns the historical pricing rule; fails off-policy | C |
| Causal machine learning with orthogonalisation | Price effect estimated after removing confounding in a learned representation | Requires a valid nuisance model and identifying variation | D |
| Structural demand model with an instrument | Explicit supply and demand equations | Requires the instrument and functional form to be right | C |
| Randomised price test | Price assigned | Costly, limited duration, interference across stores | B |

Zalando's "Causal Forecasting for Pricing" preprint (arXiv:2312.15282, 2023) states the first
failure mode from an operator's perspective: standard forecasters match the state of the art on
historical pricing but lose on counterfactual pricing, and the authors attribute the gap to
regularisation bias in the price coefficient. The CausalDemand benchmark reaches a similar
conclusion from the other direction: forecast accuracy and causal sign recovery are different
objectives, and models that win on the first lose on the second. Both are preprints or benchmark
reports, graded D.

The peer-reviewed version is Nambiar, Simchi-Levi and Wang in *Management Science*, which shows
that misspecification creates correlation between price and forecast error, proposes a
random-price-shock algorithm, and reports revenue gains of 8–20% over methods that ignore price
endogeneity on fashion retail data. Grade A for the method; the percentage is as reported by the
authors and not independently verified here.

### 9.4 What firms do, and what the design requires

| Requirement of a defensible elasticity | Common practice | Gap and why it persists |
|---|---|---|
| Price variable matched to the shopper's price | Shelf or invoice price used, inconsistently | Price files are owned by different teams |
| Promotion and availability controlled | Sometimes promotion, rarely availability | Availability data lives in another system |
| Instrument with an argued exclusion restriction | Rare; cost indices used as controls instead | The distinction is not taught in most analytics curricula |
| First-stage strength reported with clustering | F reported without clustering, or not at all | Software defaults |
| LATE stated | Almost never | The concept is unfamiliar outside economics |
| Interval translated into a decision | Point estimate used in the business case | Planning templates ask for a number |

A first-stage F computed without clustering, in a panel of 40 stores and 52 weeks, overstates
instrument strength by roughly the square root of the intraclass correlation in the instrument. In
the worked example the classical F is 338 and the clustered effective F is 41.6: both strong, but the
ratio shows how far the conventions diverge, and in weaker settings it is the difference between
"strong" and "weak".

### 9.5 The evidence gap

No public dataset lets you verify a firm's elasticity claim, and almost no published elasticity
arrives with its instrument and first stage attached. Benchmarking an internal estimate against a
published one is therefore nearly impossible: the published number may be a different estimand, on a
different price variable, with a different design, and no way to tell. Treat cross-firm comparisons
as descriptive at best.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Positive elasticity reported as a finding | Coefficient has the wrong sign for the category | Demand-shock variance dominates; the regression traces the pricing rule | Price increases forecast to raise volume | Decompose shock variances; find an instrument |
| More data offered as the fix | "We will rerun with three more years" | Bias is in the probability limit | Larger sample, tighter wrong interval | State that consistency is not the problem |
| Cost index entered as a control | Cost sits beside price in the demand equation | Cost is a supply shifter, not a demand determinant | Absorbs identifying variation; coefficient uninterpretable | Move it to the instrument |
| First-stage F reported without clustering | F in the hundreds in a 40-cluster panel | Clustering reduces effective information | Weak instrument presented as strong | Report the cluster-robust or effective F |
| F above 10 treated as proof of validity | "F = 24, instrument is valid" | F tests relevance only | Exclusion violation undetected | Write the exclusion argument in prose |
| Fixed effects credited with causality | "We control for store and week" | Store-week shocks remain | Sign may still be wrong | Compare with and without; report both |
| Elasticity applied outside the observed price range | Proposed change 20% when data span 8% | Constant-elasticity extrapolation | Overstated volume loss or gain | State the observed range; test nonlinearity |
| Revenue elasticity confused with profit elasticity | Business case built on revenue | `1 + b` versus `(P-c)/P` arithmetic | Right answer, wrong objective | Compute break-even elasticity |
| Stock-out weeks dropped | Sample excludes zero-sales rows | Selection on availability, which price causes | Elasticity biased toward zero | Model availability; do not drop silently |
| LATE reported as an average | "The elasticity of demand is −1.6" | Compliers are a selected subgroup | Wrong number for a different decision | Name the compliers explicitly |
| Reference and shelf price mixed | Promotion file and price file disagree | Different definitions, different legal status | Elasticity for a price never shown | Reconcile definitions before estimation |

Two deserve expansion. **A positive elasticity reported as a finding** is usually a data problem
quietly dropped; when it survives it is sometimes written up as a Veblen good or quality signal, which
is occasionally true. Usually it is a demand-shock-dominated panel, and the tell is heavy promotion
plus volatile local demand — Case B of section 3.2. **Fixed effects credited with causality** appears
in almost every retail deck and never carries the argument it implies: fixed effects remove level
differences, while simultaneity is a within-week, within-store phenomenon.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Price and quantity measured at the same aggregation | The elasticity is defined on a matched pair | Chain-level price joined to store-level units | Check join cardinality; look for repeated prices across stores |
| The price variable is the price the shopper faced | Response is to the shelf price | Loyalty discounts applied at the till | Compare POS price to shelf audit on sampled weeks |
| Demand and cost shocks are separable | The instrument must move one curve only | A cost shock that also changes quality or availability | Argue it; test correlation with observables |
| The instrument is uncorrelated with the demand error | Exclusion restriction | Cost changes correlated with local demand, such as a harvest shock that also raises income | Not testable when just identified; argue and stress-test |
| Monotonicity holds | LATE interpretation | Defier stores that cut price when cost rises | Estimate the first stage by store; look for negative pass-through |
| Elasticity is constant over the observed range | The log-log coefficient is a single number | Reference-price effects, thresholds, competitor reaction | Split by price level; add a quadratic in logs |
| The category boundary is stable | Cross-item substitution is excluded | New entrant, delisting, packaging change with a new GTIN | Check item-master history for breaks |
| Cost pass-through is stable over the window | The first stage depends on it | A price war, a contractual price freeze | Estimate the first stage by sub-period |

Three limits are worth stating plainly. **The estimate is local by construction**: every design here
identifies a response to a particular kind of price variation in a particular set of units, and none
recovers "the" elasticity for all purposes. **The exclusion restriction is an argument, not a test**:
overidentification can reject a set of instruments but cannot confirm one, and a set invalid in the
same direction will pass. **The structural system is a model**: real categories have dynamics,
reference prices, inventory carry-over and cross-item substitution, so the decomposition is exact
within the model and approximate outside it.

## 12. Exercises

### Exercise 1 — Elasticity arithmetic, four ways (calculation, 45 minutes)

| Price (EUR) | 4.00 | 4.20 | 4.40 | 4.60 | 4.80 | 5.00 | 5.20 | 5.40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Units | 1,240 | 1,180 | 1,150 | 1,090 | 1,060 | 1,010 | 990 | 940 |

Compute the log-log elasticity, the arc elasticity from first to last observation, the linear
slope, and the point elasticity at the means. Then compute revenue at each price and the gross
profit change from EUR 4.00 to EUR 5.40 at a 40% gross margin.

**Deliverable:** one page, four elasticities stated, revenue and profit tables.
**Acceptance criteria:** log-log correct to two decimals; arc and point agreeing within 0.02;
break-even elasticity computed and compared; one sentence on why the four differ.

### Exercise 2 — Draw the identification problem (application, 40 minutes)

For a category you know, draw the price–quantity plane with demand, supply, and a plausible
scatter. Mark which shocks dominate and therefore which curve the scatter approximates.

**Deliverable:** one hand-drawn diagram plus a 150-word note.
**Acceptance criteria:** the diagram distinguishes both curves from the scatter; the note names two
demand shocks and two cost shocks specific to the category; it states whether the naive regression
is biased toward zero, away from zero, or toward a positive sign.

### Exercise 3 — Derive the simultaneity bias (calculation, 60 minutes)

A category has demand slope `e = 1.2`, pricing slope `h = 0.9`, demand-shock variance 180, and
cost-shock variance 45, both in squared log points.

(a) Compute the OLS elasticity and the bias. (b) Compute the cost-shock variance at which the
elasticity changes sign. (c) State what happens to the bias if the retailer moves to a fixed
national pricing schedule.

**Deliverable:** the algebra and the three answers, with the threshold derived rather than guessed.
**Acceptance criteria:** the decomposition shown as `-e + (e+h) x share`; the threshold expressed as
a variance ratio; part (c) reasoned from the formula.

### Exercise 4 — Instrument audit (critique, 50 minutes)

A memo proposes instrumenting shelf price with (i) the supplier's list price, (ii) the competitor's
shelf price, (iii) the store's own footfall, and (iv) a national fuel price index. Audit each.

**Deliverable:** a four-row table with relevance, exclusion, and the direction of bias if exclusion
fails.
**Acceptance criteria:** at least one candidate rejected for relevance and one for exclusion; footfall
identified as a bad control rather than an instrument; bias direction stated where determinable.

### Exercise 5 — First-stage F and the weak-instrument correction (calculation, 45 minutes)

A panel of 520 store-weeks gives an OLS elasticity of +0.85 and a 2SLS estimate of −1.40 with one
instrument whose first-stage R-squared is 0.018.

(a) Compute the first-stage F. (b) Apply the `1/F` approximation and state a bias-corrected point
estimate. (c) State what the tF correction of Lee, McCrary, Moreira and Porter does to the interval,
and why.

**Deliverable:** the three computations and a two-sentence recommendation.
**Acceptance criteria:** F with correct degrees of freedom; the approximation labelled as
approximate; the recommendation states whether the estimate can carry a pricing decision.

### Exercise 6 — Design an instrument (design, 60 minutes)

Choose a setting where price is set by the firm — an airline route, a contract manufacturer, a
distributor's transfer price, a quick-commerce delivery fee. Propose one instrument and defend it.

**Deliverable:** the instrument, the expected first stage, the exclusion argument in three sentences,
the leading threat, and the complier group.
**Acceptance criteria:** the instrument varies within the unit of analysis; the exclusion argument
names a specific confound it does not resolve; compliers described in business language; a fallback
design named if the instrument proves weak.

### Exercise 7 — Panel fixed-effects cost accounting (calculation, 50 minutes)

Total log-price variance is 0.062, of which 71% is between stores and 29% within. After week fixed
effects, within variation falls to 0.011.

(a) Compute the standard-error inflation from two-way fixed effects. (b) State what share of price
variation the estimator uses. (c) Explain why the estimate may still be biased.

**Deliverable:** variance table, inflation factor, 200-word assessment.
**Acceptance criteria:** inflation computed as a square root of a variance ratio; the answer to (c)
names a specific time-varying confounder, not "omitted variables" in general.

### Exercise 8 — The LATE memo (critique, 45 minutes)

An analyst reports "the price elasticity of demand is −1.6, instrumented by wholesale cost" on a
slide with no further detail. Write the paragraph that should replace it.

**Deliverable:** a 200-word replacement paragraph.
**Acceptance criteria:** names the identifying variation; names the complier group; states one
decision the estimate supports and one it does not; reports first-stage strength with the
clustering convention.

### Exercise 9 — Reproduce and critique a published elasticity (analysis, 90 minutes)

Find a published price elasticity for a consumer category — journal article, working paper, or
regulator's market study. Extract the specification, the price variable, the identification
strategy, and the sample.

**Deliverable:** a two-page critique with a reproduction plan.
**Acceptance criteria:** the price variable classified as shelf, invoice, or net; the identification
strategy classified using section 3.4; at least one claim identified as unsupported by the reported
evidence; the plan names the data required and is honest about what cannot be obtained.

## 13. Self-check answers

### 13.1 Exercise 1 arithmetic, in full

**Step 1 — take logs.**

| Price | ln P | Units | ln Q | `x` | `y` |
|---:|---:|---:|---:|---:|---:|
| 4.00 | 1.386294 | 1,240 | 7.122866 | −0.156475 | 0.139753 |
| 4.20 | 1.435085 | 1,180 | 7.073270 | −0.107684 | 0.090157 |
| 4.40 | 1.481605 | 1,150 | 7.047517 | −0.061164 | 0.064404 |
| 4.60 | 1.526056 | 1,090 | 6.993933 | −0.016713 | 0.010820 |
| 4.80 | 1.568616 | 1,060 | 6.966024 | 0.025847 | −0.017089 |
| 5.00 | 1.609438 | 1,010 | 6.917706 | 0.066669 | −0.065407 |
| 5.20 | 1.648659 | 990 | 6.897705 | 0.105890 | −0.085408 |
| 5.40 | 1.686399 | 940 | 6.845880 | 0.143630 | −0.137233 |
| **Mean** | **1.542769** | | **6.983113** | **0.000000** | **0.000000** |

**Step 2 — sum the squares and cross-products.** Squaring the `x` column and multiplying `x` by
`y` row by row, then summing:

```text
   sum(x²)  = 0.077056
   sum(x*y) = -0.069255
   b = sum(x*y) / sum(x²) = -0.069255 / 0.077056 = -0.8988
```

**Step 4 — arc elasticity, first to last.**

```text
   %ΔQ = (940-1240) / ((940+1240)/2) = -240/1090 = -0.22018
   %ΔP = (5.40-4.00) / ((5.40+4.00)/2) = 1.40/4.70 = 0.29787
   arc = -0.22018 / 0.29787 = -0.739
```

**Step 5 — linear slope and point elasticity at the means.**

```text
   slope = -240 / 1.40 = -171.43 units per EUR
   point = -171.43 x (4.70 / 1082.5) = -171.43 x 0.004342 = -0.744
```

The arc and point estimates agree to three decimals, which is the expected check: at the midpoint
the two formulas coincide for a linear curve. The log-log value is larger in absolute value because
it imposes a single constant elasticity over a range where the response is not exactly log-linear.

**Step 6 — revenue and gross profit.**

| Price | Units | Revenue | Gross profit at 40% margin |
|---:|---:|---:|---:|
| 4.00 | 1,240 | 4,960 | 1,984 |
| 4.20 | 1,180 | 4,956 | 2,124 |
| 4.40 | 1,150 | 5,060 | 2,300 |
| 4.60 | 1,090 | 5,014 | 2,398 |
| 4.80 | 1,060 | 5,088 | 2,544 |
| 5.00 | 1,010 | 5,050 | 2,626 |
| 5.20 | 990 | 5,148 | 2,772 |
| 5.40 | 940 | 5,076 | 2,820 |

Unit cost is `4.00 x 0.60 = 2.40`, so gross profit is `(P - 2.40) x Q`: EUR 1,984 at 4.00 and
EUR 2,820 at 5.40, a rise of 42.1%, while revenue moves non-monotonically and ends only 2.3% higher.
The break-even elasticity is `1/0.40 = 2.50`, far above the estimated 0.90, so every increase here is
gross-profit-accretive — the argument for judging a pricing decision on profit, not on revenue.

### 13.2 Exercise 3 — the simultaneity bias, derived in full

**Given:** `e = 1.2`, `h = 0.9`, `Var(u) = 180`, `Var(v) = 45`.

**Step 1 — mixture form.**

```text
   b_OLS = [h*var(u) - e*var(v)] / [var(u)+var(v)]
         = [0.9*180 - 1.2*45] / 225
         = [162 - 54] / 225 = 108/225 = 0.48
```

**Step 2 — decompose.**

```text
   b_OLS = -e + (e+h) * var(u)/(var(u)+var(v))
         = -1.2 + 2.1 * (180/225)
         = -1.2 + 2.1 * 0.800
         = -1.2 + 1.680 = +0.48
```

The true elasticity is **−1.2**; the naive estimate is **+0.48**; the bias is **+1.68**, 1.4 times
the magnitude of the parameter.

**Step 3 — sign-flip threshold.** Set the numerator to zero:

```text
   h*var(u) = e*var(v)  =>  var(v)/var(u) = h/e = 0.9/1.2 = 0.75
   with var(u) = 180:  var(v)* = 0.75 x 180 = 135
   check: share = 180/315 = 0.571;  -1.2 + 2.1*0.571 = 0
```

The naive estimate is positive whenever the cost-shock variance exceeds `h/e` times the demand-
shock variance. For most consumer categories `h/e` lies between 0.5 and 1.5, so a positive estimate
arises whenever cost is at least as volatile as local demand. That is not a rare condition.

**Step 4 — fixed national pricing.** If the shelf price is identical in every store in a week, the
cost shock no longer reaches store-level price: `h` collapses toward zero, and the remaining
variation is mostly week-to-week national changes driven by cost. The demand share of price
variance falls, so the bias falls, and the naive regression moves toward Case A — it is close to
tracing the demand curve. The fixed-schedule regime is, in effect, an instrument supplied by the
pricing policy itself. That is one reason national price files sometimes give reasonable
elasticities while store-level files do not.

> **Note:** This is the mechanism behind a common empirical observation, not a recommendation. A
> fixed national price also destroys the within-store variation panel fixed effects need, and
> confounds the price effect with the national marketing calendar.

### 13.3 Exercise 5 — first-stage F, interpreted in full

**Step 1 — compute.**

```text
   F = [R²_1/(1-R²_1)] x (n-k)
     = [0.018/0.982] x (520-2)
     = 0.018330 x 518
     = 9.49
```

**Step 2 — what 10 means.** Staiger and Stock (1997) proposed 10 for a single endogenous regressor,
corresponding roughly to 2SLS bias of about 10% of the OLS bias. It is a heuristic, not a test, and
it was calibrated for homoskedastic errors and independent observations. Both conditions fail in a
store-week panel with clustered errors. The number survives because it is easy to remember, not
because it is correct here.

**Step 3 — bias correction.**

```text
   OLS bias = 0.85 - (-1.40) = 2.25
   retained fraction ≈ 1/F = 1/9.49 = 0.105
   retained bias = 0.105 x 2.25 = 0.237
   bias-corrected ≈ -1.40 + 0.237 = -1.16
```

The 2SLS point estimate is itself probably too close to zero by about 0.24 elasticity points. The
corrected −1.16 is closer to the truth, but it has no sampling distribution attached and should not
be reported as though it did.

**Step 4 — what the tF correction does.** Lee, McCrary, Moreira and Porter (2022) construct a
standard-error adjustment that is a smooth function of the first-stage F, producing tests with correct
size: the interval widens as F falls. From 61 *American Economic Review* papers they report that for
one quarter of specifications the corrected standard errors are at least 49% larger at the 5% level.
At F = 9.5 the inflation is real but usually modest; at F = 3 it can put zero inside the interval.

**Step 5 — the recommendation.** The estimate of −1.40 is directionally credible and statistically
fragile. It supports a decision robust to a 0.3 elasticity-point error, such as choosing between two
price points that differ materially in margin, but not one that turns on whether the elasticity is 1.2
or 1.6. Report the corrected interval and state which decision the data can carry.

### 13.4 Design rubrics

| Exercise | Strong answer | Weak answer |
|---|---|---|
| 4 — instrument audit | Each candidate tested against both conditions separately; competitor price rejected for exclusion, footfall rejected as a confounder belonging in the regression, fuel index rejected on relevance | Instruments ranked by correlation with price, the most correlated called best |
| 6 — instrument design | Something varying within the unit of analysis, with the exclusion threat written as a specific competing story, compliers named, fallback design stated | A national index absorbed by fixed effects, or "demand" offered as an instrument |
| 7 — fixed-effects accounting | `sqrt(0.062/0.011) = 2.37`, the estimator uses 18% of price variation, a specific time-varying confounder named | Inflation reported without interpretation, or a claim that the estimate is now causal |
| 8 — LATE memo | Names the identifying variation and the complier group, reports the clustered first-stage F, states one decision supported and one not | Caveats added without naming compliers, leaving the reader unable to tell what the number supports |

## 14. Interview questions

**1. "What is price elasticity of demand, and what are its units?"**
**Strong:** the ratio of percentage changes, dimensionless, invariant to the currency of price but
not to whether quantity is units or revenue. **Weak:** "the slope of the demand curve", which
conflates a dimensionless ratio with a units-bearing slope.

**2. "You regress log quantity on log price and get +0.4. What do you do?"**
**Strong:** the sign is implausible for a normal good; diagnose simultaneity first and decompose
price variation into demand-driven and cost-driven components; more data will not help. **Weak:**
"add controls" or "the data must be wrong", neither of which identifies a mechanism.

**3. "Why doesn't the observed price–quantity relationship trace the demand curve?"**
**Strong:** draws both curves, explains that observations are intersections, and states that the
scatter's slope is a variance-weighted mixture of the two slopes, with the limiting cases named.
**Weak:** "because of confounding", without being able to say which variable confounds or why it is
unobservable.

**4. "What makes a valid instrument?"**
**Strong:** relevance, exclusion, independence; relevance is testable, exclusion is not, and the
overidentification test only rejects rather than confirms; the argument is prose. **Weak:** lists
the conditions and says the F statistic tests validity, confusing strength with validity.

**5. "Is a first-stage F of 12 good?"**
**Strong:** it depends on clustering, the number of instruments, and the estimator; under store
clustering with many instruments 12 may be weak; mentions the effective F and the tF correction.
**Weak:** "above 10, so yes", which treats a 1997 heuristic as a threshold test.

**6. "What does an instrument estimate, and for whom?"**
**Strong:** a LATE — the average response for compliers, the units whose treatment the instrument
moved; complier status is not observable individual by individual; monotonicity is required; gives
a concrete complier group. **Weak:** "the causal effect", which is the overclaim LATE exists to
prevent.

**7. "When do fixed effects solve endogeneity?"**
**Strong:** only when the endogeneity is a time-invariant unit characteristic or a common time
shock; a store-week demand shock survives both dummies, and simultaneity is exactly that. Adds the
cost: between variation discarded, standard errors rise. **Weak:** "fixed effects control for
unobserved heterogeneity", full stop.

**8. "How would you estimate elasticity for a new SKU with no history?"**
**Strong:** refuses the premise; borrows from a comparable item or the category, or runs a designed
price test in a subset of stores; states the transfer assumption and how to test it. **Weak:** "run
a regression when the data arrives", which postpones the decision without informing it.

**9. "What is the difference between revenue elasticity and price elasticity?"**
**Strong:** revenue elasticity is `1 + price elasticity` under constant elasticity, and the
profit-relevant threshold is `1/margin`, not 1; a price increase can raise profit while lowering
revenue. **Weak:** treats revenue maximisation as the objective.

**10. "Your instrument is a cost index. What is the leading threat?"**
**Strong:** names a specific competing channel — a harvest shock that also raises local income, or
a cost increase accompanied by a quality or availability change — states the bias direction, and
proposes a robustness check. **Weak:** "the exclusion restriction might be violated", with no
mechanism.

**11. "When would you report the naive elasticity anyway?"**
**Strong:** when the decision is robust across the plausible range, when the report is explicitly
descriptive, and when the bias direction is known and does not flip the recommendation — with the
label visible. **Weak:** "never", which is impractical, or "always", which is the failure this topic
exists to prevent.

## 15. Key terms

- **Elasticity** — percentage change in quantity per percentage change in price; dimensionless.
- **Log-log specification** — regression of log quantity on log price, whose slope is the elasticity.
- **Log point** — a difference in natural logs, approximately a proportional change when small.
- **Semi-elasticity** — percentage change in quantity per one-unit change in a level regressor.
- **Structural equation** — a behavioural equation, such as demand or pricing, written from theory.
- **Reduced form** — an endogenous variable expressed as a function of exogenous variables only.
- **Simultaneity** — price and quantity determined jointly, so neither is exogenous to the other.
- **Simultaneity bias** — the gap between the OLS slope and the demand slope; the sum of the slopes times the demand-shock variance share.
- **Identification** — whether a parameter is uniquely recoverable from the observable distribution.
- **Order condition** — the counting rule requiring at least as many excluded exogenous variables as endogenous regressors.
- **Instrument** — a variable that shifts the endogenous regressor and reaches the outcome only through it.
- **Relevance** — the instrument moves the endogenous regressor; testable via the first stage.
- **Exclusion restriction** — no direct path from instrument to outcome; not testable when just identified.
- **Monotonicity** — the instrument moves treatment in one direction for all units.
- **First stage** — the regression of the endogenous regressor on the instruments and controls.
- **First-stage F** — the strength statistic for excluded instruments; 10 is a heuristic, not a test.
- **Weak instrument** — too little first-stage signal; 2SLS inherits roughly `1/F` of the OLS bias.
- **Two-stage least squares (2SLS)** — the estimator using first-stage fitted values in the second stage.
- **LIML** — limited-information maximum likelihood; more robust to weak instruments than 2SLS.
- **LATE** — the average effect for units whose treatment the instrument changed.
- **Complier** — a unit whose treatment follows the instrument; **defier** — one that moves opposite, violating monotonicity.
- **Within estimator** — the fixed-effects estimator, computed on unit-demeaned data.
- **Between variation** — variation across units; discarded by fixed effects.
- **Pass-through** — the fraction of a cost change that reaches the retail price.
- **Cost shifter** — a variable that moves supplier cost and therefore price; the natural instrument for price.
- **Break-even elasticity** — `1/margin`; the absolute elasticity at which a price change leaves gross profit unchanged.

## 16. Sources

**Peer-reviewed and monographs (Grade A)**

- Working, E. J. (1927). "What Do Statistical 'Demand Curves' Show?" *Quarterly Journal of
  Economics*, 41(1), 212–235. First presented in 1926, which explains the alternative citation
  date. The founding statement of the identification problem.
- Wright, P. G. (1928). *The Tariff on Animal and Vegetable Oils*, Appendix B. Macmillan. First
  derivation of instrumental-variable estimation as a solution to the identification problem.
- Haavelmo, T. (1944). "The Probability Approach in Econometrics." *Econometrica*, 12, supplement.
- Staiger, D., and Stock, J. H. (1997). "Instrumental Variables Regression with Weak Instruments."
  *Econometrica*, 65(3), 557–586. DOI: 10.2307/2171753. Source of the F = 10 rule of thumb.
- Angrist, J. D., Imbens, G. W., and Rubin, D. B. (1996). "Identification of Causal Effects Using
  Instrumental Variables." *Journal of the American Statistical Association*, 91(434), 444–455.
  DOI: 10.1080/01621459.1996.10476902. The LATE theorem and complier typology.
- Imbens, G. W., and Angrist, J. D. (1994). "Identification and Estimation of Local Average
  Treatment Effects." *Econometrica*, 62(2), 467–475.
- Stock, J. H., and Yogo, M. (2005). "Testing for Weak Instruments in Linear IV Regression." In
  *Identification and Inference for Econometric Models*, Cambridge University Press. Critical
  values depend on the number of instruments.
- Andrews, I., Stock, J. H., and Sun, L. (2019). "Weak Instruments in Instrumental Variables
  Regression: Theory and Practice." *Annual Review of Economics*, 11. Recommends effective F.
- Lee, D. S., McCrary, J., Moreira, M. J., and Porter, J. (2022). "Valid t-Ratio Inference for IV."
  *American Economic Review*, 112(10), 3260–3290. DOI: 10.1257/aer.20211063.
- Dhar, T., Chavas, J.-P., and Gould, B. W. (2002). "An Empirical Assessment of Endogeneity Issues
  in Demand Analysis for Differentiated Products." AgEcon Search record, DOI:
  10.22004/ag.econ.201561.
- Nambiar, A., Simchi-Levi, D., and Wang, H. "Dynamic Learning and Pricing with Model
  Misspecification." *Management Science*. DOI: 10.1287/mnsc.2018.3194.
- Kropko, J., and Kubinec, R. (2020). "Interpretation and Identification of Within-Unit and
  Cross-Sectional Variation in Panel Data Models." *PLoS ONE*.
- Plümper, T., and Troeger, V. E. (2019). "Not so Harmless After All: The Fixed-Effects Model."
  *Political Analysis*.

**Working papers, preprints, and benchmarks (Grade C or D, as marked)**

- Chernozhukov, V., Hausman, J., and Newey, W. (2019). "Demand Analysis with Many Prices." NBER
  Working Paper 26424. Grade C.
- Rosenthal-Kay, J., Traina, A., and Tran, K. (2021). "Seven Million Demand Elasticities." Working
  paper. Grade C.
- Zalando (2023). "Causal Forecasting for Pricing." arXiv:2312.15282. Grade D.
- CausalDemand benchmark (2025–2026). Public repository. Grade D.

**Official and regulatory (Grade B)**

- Directive 98/6/EC on consumer protection in the indication of prices, as amended by Directive
  (EU) 2019/2161.
- Regulation (EU) 2016/679 (General Data Protection Regulation).
- 16 CFR Part 233, Guides Against Deceptive Pricing, United States Federal Trade Commission.
- IFRS 15 *Revenue from Contracts with Customers*; ASC 606 (FASB).
- GS1 General Specifications, including GTIN, GLN, GPC, and EPCIS 2.0.

**Practitioner evidence (Grade C)**

- The practice-gap table in section 9.4 is a practitioner diagnostic assembled from internal
  pricing documentation, not a survey. It must not be quoted as a base rate.

**Press and anecdote (Grade E)**

- Not used. No press summary supports a quantitative claim in this file.

**Note on citations.** Details are as returned by the searches. Working's paper is dated 1927 and is
often cited as 1926 because of earlier circulation. The Dhar, Chavas and Gould citation carries the
AgEcon Search DOI because the journal attribution was not confirmed. The 8–20% figure from Nambiar,
Simchi-Levi and Wang and the 49% figure from Lee et al. are as reported by those authors and were not
recomputed.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Prerequisite, written — DAGs and bad controls: [`06_dags.md`](06_dags.md)
- Prerequisite, written — experiments and quasi-experiments: [`07_experiments-quasi.md`](07_experiments-quasi.md)
- Prerequisite, planned — worked difference-in-differences: `08_difference-in-differences.md` (planned)
- Next topic, planned — sensitivity and omitted-variable bias: `10_sensitivity-omitted-bias.md` (planned)
- Written sibling — estimands: [`01_estimands-question-types.md`](01_estimands-question-types.md)
- Written sibling — robust errors and clustering: [`05_autocorrelation-robust-errors.md`](05_autocorrelation-robust-errors.md)
- Survey file: [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
- Cost and margin definitions used in section 6.8: [`../01_supply_chain_foundations/06_cost-and-working-capital.md`](../01_supply_chain_foundations/06_cost-and-working-capital.md)
- Return to the foundations map: [`../_index.md`](../_index.md)
