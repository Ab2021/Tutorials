# Module 04 — Statistics and Causal Inference

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 32 hours (12 lecture, 10 lab, 10 project)
> **Track fit:** Common core
> **Prerequisites:** Module 01A (units, time, measurement, assumptions); introductory probability; linear regression; Python
> **Tools:** Python 3.12, pandas, statsmodels, linearmodels, DuckDB; R (`fixest`, `did`, `Synth`) optional; DoWhy or EconML for refutation checks
> **Industry question:** "Did the intervention change the outcome, by how much, for whom — and what would have happened anyway?"

---

## 1. Why industry cares

Every supply chain organisation already produces causal claims. A category manager says the promotion lifted sales 18%. A logistics director says the new carrier cut cost per drop 9%. A vendor says its engine reduced stockouts 30%. A buyer says the preferred supplier is 6 percentage points worse on on-time delivery (OTD). Each sentence has the same shape: X caused Y. Most are wrong, and the errors move money.

| Error | What happens | Typical cost |
|---|---|---|
| False positive | Budget, capacity, or inventory committed to an effect that will not repeat | Recurring spend on a null intervention; write-off of the build |
| False negative | A working change is killed because the comparison was contaminated | Lost margin and service improvement; competitive drift |
| Wrong magnitude | Direction right, size wrong; the plan is sized off a biased estimate | Inventory, overtime, expedite, working capital |

The third is the most common and the least discussed. A promotion genuinely lifts volume, but the estimate is inflated because the promoted weeks were also pre-holiday weeks. The direction survives, the number does not, and the safety stock built on it is wrong for a year.

The 2026 context raises the stakes. Data volume has outrun design literacy — it is easy to compute a confident answer from a comparison no statistician would accept. AI recommendations arrive as point estimates with no counterfactual, and planners are expected to act on them. Regulation now touches decision governance: the EU AI Act (Regulation 2024/1689) imposes risk-management, data-governance, and human-oversight duties on high-risk systems, making a documented counterfactual an audit artifact. And finance teams have learned to ask for incrementality rather than correlation, because trade spend and working capital are both under scrutiny.

This module teaches the minimum toolkit for defending a number in a planning meeting: what quantity you are estimating, what would have happened otherwise, what would make the estimate wrong, and how to say all three in one sentence a planner can use.

## 2. Learning outcomes

By the end of this module, you can:

- Write an estimand — population, treatment, outcome, horizon, summary — before touching data, and distinguish it from a model specification.
- Separate prediction from causation and explain why a coefficient can be valid for one and invalid for the other.
- Construct bootstrap and cluster-robust intervals and state what the interval covers.
- Apply Bonferroni or Benjamini-Hochberg control and explain FWER versus FDR in decision terms.
- Select linear, logistic, Poisson, quantile, or mixed-effects models from the outcome distribution and dependence structure.
- Model seasonality with Fourier terms and holiday indicators, and correct for autocorrelation and heteroskedasticity.
- Draw a DAG, name confounders, colliders, mediators, and bad controls, and defend the adjustment set.
- Estimate a difference-in-differences (DiD) effect with a parallel-trends audit and convert it to currency with a break-even.
- Estimate elasticity with and without an instrument and explain the direction of endogeneity bias.
- Diagnose selection bias, missing-data mechanisms, and stockout censoring, and choose a defensible correction.
- Run sensitivity analyses and negative controls, and communicate uncertainty as a decision range.

## 3. Mental model

### 3.1 Three question types

```
Question type   What you want            What identifies it          What you report
---------------------------------------------------------------------------------------
Descriptive     What happened?           No identification           Counts, rates, distributions
Predictive      What will happen?        Association + stability     Prediction + error distribution
Causal          What if we did X?        Counterfactual or design    Effect + CI + assumptions
```

A demand forecast is predictive. "Will this promotion pay back?" is causal. A model can be excellent at one and useless at the other.

### 3.2 The counterfactual is the whole problem

```
 Treatment group   Y(1) observed | Y(0) missing   ← the missing cell
 Control group     Y(0) observed | Y(1) irrelevant
```

The causal effect compares two states of the same unit at the same time. One is always missing. Every method here is an explicit assumption about reconstructing it: DiD assumes the control trend would have applied; synthetic control builds a weighted donor pool; instrumental variables (IV) use exogenous treatment variation; regression discontinuity (RDD) uses units either side of a threshold; randomisation makes groups exchangeable by construction.

No causal estimate is assumption-free. The professional output is therefore always three things: **the estimate, the interval, and the assumption that would break it.**

### 3.3 Design selection

Randomise if feasible and ethical (Module 05). If not, look for a threshold (RDD), a timing shock (DiD or event study), an exogenous price or cost shifter (IV), or a single treated unit with a donor pool (synthetic control). If none exists, model selection credibly with matching or weighting plus a sensitivity analysis — or report an association and label it as such. "We cannot identify this" is cheaper than defending a number that will not replicate.

## 4. Core concepts

### 4.1 Estimands

An estimand is the precise quantity you are trying to learn, fixed before estimation. The same data supports several estimands with different values.

| Estimand | Question | Supply chain example |
|---|---|---|
| ATE (average treatment effect) | Mean effect across the whole population | New replenishment rule across all 620 stores |
| ATT (effect on the treated) | Effect on the units that received it | The 40 vendor-managed inventory (VMI) pilot stores |
| CATE (conditional ATE) | How the effect varies with characteristics | Effect by store volume decile or lane distance band |
| LATE (local ATE) | Effect for units the instrument moved | Stores whose order pattern the cost shifter actually changed |
| Quantile effect | Effect at a chosen quantile | Effect on the worst 10% of delivery delays |

A complete statement has five parts: **population, treatment (with version), outcome (units, currency, window), summary (mean, proportion, quantile, total), and time horizon.** Omitting the horizon is the most common defect: a promotion's week-1 and 13-week effects can have opposite signs because of pantry loading.

> **Industry interpretation.** "Lift was 18%" is not an estimand. "Mean incremental units per store per week, across the 40 pilot stores, over the 8 promotion weeks plus 4 post weeks, against the counterfactual in §4.10" is.

### 4.2 Prediction versus causation

| Reading | Question | Coefficient means | Requirement |
|---|---|---|---|
| Predictive | Given X, what is the best guess of Y? | Conditional association | Out-of-sample stability; no causal claim |
| Causal | If we intervene on X, how does Y move? | ∂Y/∂X under intervention | No unblocked back-door paths; correct functional form |

Regress units on whether an item was on display and the coefficient is large and stable — displays sell product. Use it to plan the effect of buying more displays and you overstate it, because merchandisers display products they already expect to sell. The predictive model is fine; the decision is not. Two rules follow: **a model optimised for prediction is not automatically valid for a decision**, and **a causal estimate is usually worse at prediction**, because it discards the confounded variation that carries predictive signal.

### 4.3 Probability, sampling, bootstrap, and intervals

A confidence interval (CI) is a statement about the procedure: if you repeated sampling and estimation many times, 95% of intervals so constructed would contain the true value. It is not a 95% probability that the parameter lies in this range — although that is how it is read, and how it is usually communicated.

Bootstrap: resample with replacement B times (1,000–10,000), recompute the statistic, take the 2.5th and 97.5th percentiles. It needs no distributional assumption and works for ratios, quantiles, and composite metrics where textbook formulas are awkward. For time series, independent resampling destroys autocorrelation and yields intervals that are far too narrow; use a moving-block bootstrap with block length ℓ roughly equal to the autocorrelation length.

| Situation | Resampling unit | Watch out for |
|---|---|---|
| Independent store means | Store | Few stores → unstable intervals |
| Order data clustered by store | Store (cluster bootstrap) | Ignoring clustering understates the CI |
| Weekly series with seasonality | Blocks of one seasonal period | Block length changes the answer |
| Ratio metric (cost per drop) | Cluster bootstrap, recompute ratio | Delta method fragile with small denominators |
| P95 lead time | Cluster bootstrap | Small tail samples give unstable tails |

Worked check: 40 stores, weekly units, between-store standard deviation 210 units. Naive SE of the mean = 210/√40 = 33.2 units. If each store contributes 52 weeks treated as independent rows, the naive SE becomes 210/√(40×52) = 4.6 units — a sevenfold understatement. Cluster at the store.

### 4.4 Hypothesis tests and multiple comparisons

A p-value is the probability of data at least as extreme as observed, given the null. It is not the probability the null is true, and it says nothing about effect size.

Test 20 independent hypotheses at α = 0.05 under a global null: the expected number of false positives is 1, and the probability of at least one is 1 − 0.95²⁰ = 0.642. Supply chain analytics generates many-test situations routinely — 12 regions, 8 categories, 6 carriers, 40 stores.

| Method | Controls | Rule | When to use |
|---|---|---|---|
| Bonferroni | Family-wise error rate (FWER) | Test each at α/m | Few tests, strong consequences (safety, contract) |
| Holm | FWER | Step-down Bonferroni | Same, uniformly more powerful |
| Benjamini-Hochberg (BH) | False discovery rate (FDR) | Rank p-values, compare p₍ᵢ₎ ≤ iα/m | Many exploratory tests, screening |
| Pre-registration | Error rate by construction | Fix the primary metric before looking | Any confirmatory test |

BH is the right default for exploratory screening: a false discovery among 200 store tests is cheaper than missing a real one. Bonferroni is right when a single wrong "this supplier is failing" claim triggers a contract action. The deeper defence is a **pre-specified primary metric** with everything else labelled secondary.

### 4.5 Model families

| Model | Outcome | Link | Coefficient reading | Supply chain use |
|---|---|---|---|---|
| Linear (OLS) | Continuous | Identity | Change in units per unit of X | Volume, cost per drop, lead-time days |
| Logistic | Binary | Logit | Log-odds; exponentiate for odds ratio | Late delivery, stockout, damage, churn |
| Poisson | Count | Log | Log rate; exponentiate for rate ratio | Orders per store-day, returns, incidents |
| Negative binomial | Overdispersed count | Log | Rate ratio plus variance parameter | Damage claims, high-variance counts |
| Quantile | Continuous, any quantile | None | Change in the τ-quantile | P95 lead time, worst-case replenishment |
| Mixed effects | Clustered or repeated | Varies | Fixed effects plus variance components | Store-week and lane-month panels |

**Quantile regression matters because the mean lead time is not the planning number.** Safety stock depends on the upper tail. A supplier can improve the mean and worsen the 95th percentile at the same time, and a mean regression will call that an improvement.

**Mixed effects separate "this store is busy" from "this store responded".** A random intercept for store handles the first; a random slope for treatment handles the second. When unit effects are nuisance rather than interest, fixed effects are usually preferable — they absorb time-invariant heterogeneity without a distributional assumption.

### 4.6 Time-series regression, seasonality, Fourier terms, holidays

Ignoring the calendar is the single most common source of spurious lift. Fourier terms represent smooth seasonality with few parameters:

```
s(t) = Σ_{k=1..K} [ a_k · sin(2πkt / m) + b_k · cos(2πkt / m) ]
```

where m is the seasonal period (52 for weekly annual seasonality) and K is small, typically 3–6. With weekly data and m = 52, K = 4 gives 8 parameters instead of 51 dummies.

**Holidays are not seasonality.** Easter moves, Thanksgiving moves relative to a 4-5-4 fiscal calendar, Chinese New Year moves and idles factories for a variable number of days. Model them as explicit event indicators at the correct date with lead and lag windows, not as a fixed week number. In code, generate the sine and cosine pairs for k = 1 through K at periods m = 52 and m = 26, concatenate them to the design matrix, and fit with `statsmodels` `OLS` using `cov_type="HAC"`.

> **Industry interpretation.** A promotion in weeks 46–48 looks enormously effective if the model has no Christmas terms. The lift is real only to the extent it exceeds the holiday baseline, not the annual average.

### 4.7 Autocorrelation, heteroskedasticity, and robust errors

| Problem | Symptom | Consequence | Fix |
|---|---|---|---|
| Serial correlation | Slow-decaying residual ACF; Durbin-Watson far from 2 | Standard errors too small | Newey-West (HAC); cluster by unit |
| Heteroskedasticity | Residual variance grows with fitted value or store size | Errors wrong, usually too small | HC1 or HC3 errors |
| Cross-sectional dependence | Shocks common to a region | Understated SE in panels | Two-way clustering (store and week) |
| Clustering | Repeated observations per store or lane | Understated SE | Cluster at the treatment assignment level |

**Cluster at the level of treatment assignment.** If treatment is assigned to a store, cluster at the store even when the analysis is at order level. Clustering below the assignment level is the most common error in supply chain evaluations and can understate the standard error by a factor of two or more. Newey-West lag length: a common rule is L = floor(4(T/100)^(2/9)); for T = 156 weeks this gives L = 4. Report the choice, and check that results are insensitive to ±1 lag.

### 4.8 DAGs: confounding, colliders, mediators, bad controls

| Structure | Shape | Action | Supply chain example |
|---|---|---|---|
| Confounder | X ← C → Y, X → Y | Adjust for C | Seasonality drives both promotion timing and volume |
| Mediator | X → M → Y | Do **not** adjust for the total effect | Price → display → sales |
| Collider | X → C ← Y | Do **not** adjust; adjusting creates bias | Conditioning on "item was replenished" |
| Descendant of a collider | C → D | Do not adjust for D | Store sales rank, derived from demand and allocation |
| Instrument | Z → X → Y, Z ⊥ Y | Use Z; do not adjust for X | Cost shifter for price |

The collider case is the one practitioners miss. Analyse service level only among SKUs in stock at the start of the week, and stock status is caused by demand (which depletes stock) and supply (which fills it). Conditioning on it induces a spurious relationship between demand and availability. The analysis concludes "high demand causes stockouts" — true — but the magnitude is inflated by selection, and the proposed fix looks more effective than it is.

| Bad control | Why it is bad | Better |
|---|---|---|
| Post-treatment store traffic | Mediator or collider | Pre-period traffic |
| Realised service level | Descendant of treatment and outcome | Planned allocation |
| SKU margin | Correlated with unobserved demand | Category fixed effects |
| Whether the order shipped complete | Collider when demand and supply both affect it | Fix the sample definition, or model jointly |
| Future-period price | Leaks the outcome window | Contemporaneous or lagged only |

Test each covariate with one question: could the treatment have changed this variable? If yes, it is post-treatment and needs a justification, not a default inclusion.

### 4.9 Experiments versus quasi-experiments

| Design | Identification assumption | Strength | Weakness | Typical use |
|---|---|---|---|---|
| Randomised experiment | Exchangeability by randomisation | Strongest, transparent | Cost, ethics, spillover | Store pilot, A/B test |
| Difference-in-differences | Parallel trends absent treatment | Intuitive, uses panels | Fails if trends diverge | System rollout, regional launch |
| Synthetic control | Treated unit tracked by weighted donors | Handles one large unit | Donor choice; placebo inference | National policy, network change |
| Instrumental variables | Instrument affects outcome only via treatment | Handles unobserved confounding | Weak or invalid instruments | Elasticity with cost shifters |
| Regression discontinuity | Units near a threshold are comparable | Credible locally | Local only; manipulation risk | Allocation rules, service cutoffs |
| Event study | No anticipation, clean event date | Shows dynamics | Confounded events | Recall, port closure, tariff date |

### 4.10 Worked example: difference-in-differences

A distributor pilots VMI in 40 of 100 stores for 26 weeks; 60 continue the existing process. Outcome: weekly units sold per store. Decision: expand to all 100 stores at $310,000 per year in software and planner time.

**Step 1 — the 2×2.**

| Group | Stores | Pre mean (units/store/week) | Post mean | Raw change |
|---|---:|---:|---:|---:|
| Treatment (VMI) | 40 | 1,850 | 2,010 | +160 |
| Control (existing) | 60 | 1,900 | 1,960 | +60 |
| Difference | | −50 | +50 | **+100** |

The naive post-period comparison is +50 units. The DiD estimate is +100, because the control group also grew by 60 units for reasons unrelated to VMI (market growth and a competitor closure in the region).

**Step 2 — regression form.** `Y_it = α + β₁·Treat_i + β₂·Post_t + β₃·(Treat_i × Post_t) + ε_it`, where β₃ is the DiD estimate. With store and week fixed effects the same number is recovered and store size and common weekly shocks are absorbed without a linearity assumption.

**Step 3 — precision.** Store-clustered standard errors give SE(β₃) = 37.2 units.

```
t = 100 / 37.2 = 2.69
95% CI = 100 ± 1.96 × 37.2 = 100 ± 72.9 = [27.1, 172.9] units/store/week
```

**Step 4 — parallel trends.** Over 12 pre-period weeks, treatment stores grew 2.1 units/week and control stores 1.9 units/week: a slope difference of 0.2 units/week, or 2.4 units over 12 weeks, against an estimated effect of 100. An event study with 12 leads shows no lead coefficient above 8 units in absolute value, all intervals spanning zero. This is supportive, not proof: a clean pre-period cannot rule out a shock beginning exactly at week 0 that affects only treatment stores.

**Step 5 — money.**

```
Annual incremental units = 40 × 100 × 52          = 208,000 units
Gross profit at $4.20/unit                        = $873,600
Net of $310,000 program cost                      = $563,600 per year
Break-even effect = 310,000 / (40 × 52 × 4.20)    = 35.5 units/store/week
```

| Case | Effect (units/store/week) | Annual units | Gross profit (USD) | Net (USD) |
|---|---:|---:|---:|---:|
| Lower 95% | 27.1 | 56,368 | 236,746 | −73,254 |
| Point estimate | 100.0 | 208,000 | 873,600 | 563,600 |
| Upper 95% | 172.9 | 359,632 | 1,510,454 | 1,200,454 |

**Step 6 — the decision.** The point estimate supports expansion; the lower bound does not. The honest recommendation is conditional: expand a second wave of 30 stores while instrumenting a measurement plan that can detect below 100 units, or renegotiate the software cost so break-even falls below the lower bound. The break-even of 35.5 units sits inside the interval, which is exactly the risk finance needs to see.

### 4.11 Worked example: price elasticity and endogeneity

A 12-pack sells at $4.00 with unit cost $2.00, so unit margin is $2.00 (50%) on volume of 1,000 units per store-week. A log-log regression across three years of store-weeks gives `ln(Q) = 9.12 − 1.25·ln(P) + controls`. In log-log form the coefficient is the elasticity directly.

**Naive calculation.** Raise price 8% to $4.32.

```
Volume  = 1,000 × (1 − 1.25 × 0.08) = 1,000 × 0.90 = 900 units
Revenue = 900 × $4.32 = $3,888        (was $4,000, −2.8%)
Cost    = 900 × $2.00 = $1,800
Margin  = $3,888 − $1,800 = $2,088    (was $2,000, +4.4%)
```

Revenue falls while margin rises. Judging a price decision on revenue would reject a margin-accretive move — a routine and expensive error.

**Why the OLS number is probably too elastic.** Price is not assigned.

| Mechanism | Bias direction | Reason |
|---|---|---|
| Promoted weeks have lower price plus display and feature | Toward zero (too elastic) | The price coefficient absorbs the merchandising effect |
| Supplier cost rises, retailer raises price, demand is simultaneously high | Away from zero (too inelastic) | Price responds to demand shocks |
| Shelf price measured with error | Toward zero | Classical measurement error in the regressor |
| Competitor price omitted | Either direction | Omitted variable correlated with own price |

**Instrumental variables.** Use a cost shifter: the manufacturer list price, which moves retail price but has no independent path to consumer demand. Two-stage least squares (2SLS) with store and week fixed effects gives a first stage `π₁ = 0.71`, F = 24.3 — above the conventional threshold of 10, itself a heuristic rather than a test — and a second-stage elasticity of **−0.82**, 95% CI [−1.04, −0.60].

```
Volume  = 1,000 × (1 − 0.82 × 0.08) = 1,000 × 0.9344 = 934.4 units
Revenue = 934.4 × $4.32 = $4,036.61
Cost    = 934.4 × $2.00 = $1,868.80
Margin  = $2,167.81    (+8.4% vs $2,000)
```

The direction survives; the magnitude nearly doubles. The naive model understated profit by about $80 per store-week — roughly $4,150 per store-year — enough to change a rollout decision at scale.

**What to report.** Both estimates, the instrument, the first-stage strength, the exclusion argument, and the sensitivity of the conclusion across the interval. A defensible sentence: "At our central IV estimate a 10% price increase raises unit margin about 10%; at the elastic end of the interval about 8%; at the inelastic end about 13%. The decision is robust in sign but not in size."

### 4.12 Selection bias in supplier and on-time delivery data

Scorecards are computed on data selected by the process being measured.

| Mechanism | What it does | Example |
|---|---|---|
| Only received orders scored | Late or cancelled orders may leave the denominator | A supplier that never ships has no OTD record |
| Late shipments generate more events | Missingness correlates with the outcome | Advance ship notice (ASN) arrives after receipt for late loads |
| Difficult lanes assigned to specific suppliers | Confounding by assignment | The "worst" supplier serves the longest lanes |
| Volume-weighted versus order-weighted | Different populations | Many small on-time orders beat few large late ones |
| Expedites excluded | Removes the worst cases | Expedited orders reclassified as non-standard |
| Scorecard gaming | Behaviour fits the metric | Shipping early and incomplete to hit OTD |

Two corrections matter most: **define the denominator at order creation, not receipt**, reconciling cancellations explicitly; and **adjust for lane and order characteristics** with a mixed-effects model — supplier random effects, with lane distance, order size, and requested lead time as fixed effects — so that supplier performance is separated from the difficulty of the work assigned. Compute every scorecard both raw and difficulty-adjusted and show both. Where they disagree, the disagreement is the finding.

### 4.13 Missing data mechanisms

| Mechanism | Definition | Consequence of naive deletion | Remedy |
|---|---|---|---|
| MCAR (missing completely at random) | Missingness independent of everything | Unbiased but inefficient | Complete-case acceptable |
| MAR (missing at random) | Depends on observed variables | Biased if predictors unused | Multiple imputation, inverse probability weighting |
| MNAR (missing not at random) | Depends on the unobserved value | Biased, often severely | Selection model, bounds, sensitivity analysis |

Multiple imputation by chained equations: impute each variable from the others, cycle to stability, repeat m times (m = 20 is a reasonable default for moderate fractions of missing information), analyse each completed dataset, and pool with Rubin's rules. Report the fraction of missing information; if it is high, m = 20 is insufficient.

Two disciplines are non-negotiable: **never impute the outcome**, and **never use future information** — imputing a missing lead time from a shipment that has not arrived is leakage, not imputation. MNAR is the normal case in supply chain: a missing delivery timestamp is usually missing *because* the delivery was late. No imputation model fixes that. The fix is process — obtain the timestamp from an independent source such as a carrier scan, gate record, or EPCIS event — or report bounds.

### 4.14 Censored demand: stockouts hide true demand

Sales are not demand. Sales are `min(demand, available)`. When an item is out of stock, unmet demand disappears from the data and the SKU looks less popular than it is.

| Store | Days in stock | Days out | Observed sales (units) | Observed per day | Understatement |
|---|---:|---:|---:|---:|---:|
| A | 30 | 0 | 600 | 20.0 | 0% |
| B | 21 | 9 | 420 | 14.0 | 30% |

Both stores have true demand of 20 units/day over 30 days, or 600 units. A forecast trained on observed sales predicts 14/day for store B and allocates accordingly; store B stocks out again. The system has learned the stockout, not the demand. The loop is self-reinforcing and hard to see in aggregate error metrics.

Corrections, ascending in ambition: (1) flag stockout days and exclude them from the target; (2) estimate lost sales from in-stock-day demand and scale to the full period; (3) censored regression (Tobit), where an in-stock day contributes a density and a stockout day contributes the probability that demand exceeded availability; (4) choice-based estimation of primary demand and substitution from transaction data; (5) causal demand forecasting that separates baseline demand from the availability constraint. Option 5 is architecturally correct: the forecast is of demand, and the plan applies the constraint separately. It also changes what accuracy means, since a demand forecast cannot be scored against sales in stockout weeks.

### 4.15 Causal demand forecasting and counterfactual baselines

A planning forecast answers: what would we sell if we could supply freely, at this price, promotion, and assortment? That is a causal question delivered as a number.

```
Observed sales = baseline demand (seasonality, trend, price, promotion, distribution, assortment)
               × availability constraint (in-stock probability, substitution)
               + noise
```

Estimate the baseline on periods where availability was not binding, or with an explicit censoring correction. The four outputs must not be conflated: the **demand forecast** is unconstrained expected demand (demand planning); the **plan** is the feasible supply response (supply planning); the **target** is the commercially desired outcome (commercial and finance); the **commitment** is what was promised to a customer or supplier (sales and procurement).

A counterfactual baseline applies the same logic to an intervention: build it from non-intervention periods with seasonality, trend, and known drivers, and **freeze it before the intervention** so it cannot be tuned to the result.

### 4.16 Sensitivity analysis and negative controls

| Tool | Question | Implementation |
|---|---|---|
| Placebo treatment | Would the method find an effect where none can exist? | Fake treatment dates in the pre-period; estimate should be ≈ 0 |
| Negative control outcome | Does it find effects on outcomes the treatment cannot touch? | A category not stocked in pilot stores |
| Unobserved confounding | How strong must a confounder be to overturn the result? | E-value, or Rosenbaum bounds |
| Specification curve | Does the result survive reasonable analytic choices? | Distribution of estimates across specifications |
| In-space placebo | Are donors producing effects this size by chance? | Synthetic control placebo rank |

**E-value.** For a risk ratio, the minimum confounder association with both treatment and outcome needed to explain the effect away. For RR = 1.5, E = 1.5 + √(1.5 × 0.5) = 2.37. For RR = 2.0, E = 2 + √2 = 3.41. Reporting an E-value turns "we cannot rule out confounding" into a quantity a decision-maker can weigh against domain knowledge.

### 4.17 Communicating uncertainty to planners

Planners act on ranges, triggers, and consequences, not p-values.

| Do not say | Say instead |
|---|---|
| "Significant at p = 0.03." | "Most likely effect is 100 units/store/week; the plausible range is 27 to 173." |
| "The model is 92% accurate." | "On 1,000 comparable shipments, expect 80 to 120 misclassified." |
| "We reject the null." | "If the program did nothing, a result this large would appear about 1 time in 140." |
| "Elasticity is −0.82." | "A 10% price rise costs about 7% of volume; margin still rises about 8%." |
| "The interval is wide." | "At the low end this loses $73,000 a year; at the high end it makes $1.2m." |

Three rules survive planning meetings: **give the decision-relevant range** converted into units, currency, and service; **name the break-even** ("we need 35.5 units/store/week to cover cost"); and **state what would change the answer**, so uncertainty becomes a governance rule rather than a caveat.

## 5. Data, standards, and workflow

| Object | Grain | Source system | Known defect | Causal use |
|---|---|---|---|---|
| Sales transactions | Line, timestamp | POS / OMS | Censored by stockouts; returns netted inconsistently | Outcome, with censoring treatment |
| Shipments | Shipment | TMS / ERP | Late events missing or backfilled | OTD outcome, exposure |
| ASN (EDI 856) | Shipment | EDI | Arrives after receipt when late | Timestamp source; selection risk |
| Receipts | Receipt line | WMS | Discrepancy adjustments applied late | Lead-time outcome |
| Inventory snapshot | SKU × location × day | WMS / ERP | Snapshot time varies by site | Availability, censoring flag |
| Price and promotion | SKU × store × week | Pricing / TPM | Average price ≠ paid price | Treatment, elasticity regressor |
| Master data | Item, location | PIM / ERP | Hierarchy drift, duplicate GTINs | Entity keys, fixed effects |
| External | Region × day | Weather, macro | Coverage gaps, revision | Controls, instruments |

| Standard | What it fixes | What it does not fix |
|---|---|---|
| ASCM SCOR Digital Standard (SCOR DS) | Metric definitions for perfect order, OTD, fill rate | Your denominator choice, which remains yours to defend |
| GS1 General Specifications (GTIN, GLN, SSCC) | Entity and location identity | Whether two records are the same physical case |
| GS1 EPCIS 2.0 | Event semantics: what, where, when, why | Partner data completeness |
| ISO 3534-1 | Statistical vocabulary and symbols | Analysis design |
| ANSI X12 856 / UN/EDIFACT DESADV | Advance ship notice content | Timeliness of the message |

Metric definitions are causal infrastructure. Two teams can both report "OTD of 94%" over different populations — one against the original request date, one against the confirmed date after a change. Both are defensible, they are not comparable, and a before/after comparison across a definition change is not an effect.

Workflow: (1) write the estimand and the decision it serves; (2) draw the DAG and list the adjustment set plus prohibited controls; (3) inspect how records are created, which are missing and why, and what changed in the systems; (4) check the design — pre-period balance, parallel trends, instrument strength, threshold density, donor plausibility; (5) estimate with errors clustered at the assignment level; (6) run the robustness suite; (7) convert to decision units and compute break-even; (8) write the assumption ledger; (9) freeze the counterfactual before launch and register the analysis plan; (10) archive the data snapshot, code, seed, and environment.

## 6. Industry case studies

### 6.1 Card and Krueger, New Jersey minimum wage (1994)

New Jersey raised its minimum wage from $4.25 to $5.05 per hour on 1 April 1992; eastern Pennsylvania stayed at $4.25 and served as the comparison. The study compared fast-food employment before and after across the two states and reported no measurable reduction in New Jersey relative to Pennsylvania, with some specifications showing a small increase.

What transferred is the design, not the conclusion: a policy change with a clean date and a neighbouring comparison area is a strong natural experiment. What did not transfer is certainty. The result was contested on the employment measure, on the control region, and in later reanalyses using administrative payroll data. The enduring lesson is that a DiD result is only as good as its comparison group and its outcome measure, and both must be defended explicitly. Cite the paper as the canonical demonstration of the design and its fragility, not as settled fact about minimum wages.

### 6.2 Abadie, Diamond, and Hainmueller, California tobacco control (2010)

California's Proposition 99 (1988) raised tobacco taxes and funded a large anti-smoking programme. A synthetic control was built as a weighted combination of other states matching California's pre-1988 sales, price, income, and demographics. The synthetic series tracked California closely before 1988 and diverged afterwards, with the gap read as the programme's effect on per-capita sales.

What transferred is the discipline: fit donor weights on pre-period data only, report the pre-period fit as a diagnostic, and infer by in-space placebos — apply the method to each untreated state and ask where California ranks. What did not transfer is general applicability: the method needs a credible donor pool and a long, stable pre-period, it does not address simultaneous state-level shocks, and its "significance" comes from placebo rank rather than a conventional standard error.

### 6.3 Angrist, Vietnam draft lottery and earnings (1990)

Military service is not randomly assigned; volunteers differ from non-volunteers in ways that affect earnings. The draft lottery assigned random sequence numbers by date of birth, so low numbers raised the probability of service without an independent path to civilian earnings. Lottery-based estimates implied a negative effect of service on earnings for white men, in contrast with the positive ordinary least squares (OLS) association, which reflected selection.

What transferred is the structure of IV in supply chain work: find something that moves the treatment (price, allocation, mode) but has no independent path to the outcome, and report the reduced form alongside the second stage. What did not transfer is scope: the estimate is local to men whose service the lottery induced. The supply chain analogue is direct — an instrument identifies the effect for the units it moved, which may not be the units the business cares about.

### 6.4 Lee, regression discontinuity in U.S. House elections (2008)

Do incumbents have a causal advantage, or are they simply better candidates? Because the candidate with the higher vote share wins, the winner is quasi-random just either side of 50%, so a comparison near the threshold isolates incumbency. The study found a large advantage near the threshold, with estimates sensitive to bandwidth and to the running-variable polynomial.

What transferred is the threshold logic, which recurs in supply chain rules: volume tiers, credit limits, service cutoffs, expedite thresholds, minimum order quantities (MOQ). Wherever a rule creates a jump, units near the cutoff are comparable. What did not transfer is extrapolation: RDD is local, and it fails if units can manipulate the running variable. Always plot the density of the running variable before trusting the design.

### 6.5 Vulcano, van Ryzin, and Ratliff, primary demand from sales data (2012)

Retailers and airlines observe bookings, not demand; unavailable products hide unmet demand, so naive estimation understates popularity and willingness to pay. The authors proposed a choice-based model estimating primary demand and substitution jointly from sales transaction data with availability treated explicitly.

What transferred is the censoring logic of §4.14 and the principle that demand and availability must be modelled jointly rather than sequentially. What did not transfer is universal applicability: the method assumes a substitution structure and needs assortment variation to identify it. A retailer with a fixed, never-changing assortment has little information about substitution, and the model's identifying power is weak.

## 7. Labs

### Lab 4.1 — Bootstrap and cluster-robust intervals

**Objective.** Quantify how much a naive interval understates uncertainty in clustered supply chain data.

**Data.** 104 weeks of store-week units for 40 stores from Rossmann or the M5 store aggregates; the course synthetic panel as fallback.

**Steps.** (1) Compute mean weekly units and the naive SE treating all store-weeks as independent. (2) Compute a cluster bootstrap resampling stores, 2,000 draws, with seed recorded. (3) Compute a moving-block bootstrap within store, block length 8 weeks. (4) Report width ratios and explain the difference.

**Deliverable.** Notebook plus a half-page note with three intervals, width ratios to two decimals, and a recommendation for which interval belongs in a planning document.

**Acceptance criteria.** Three intervals with stated seeds; ratio correct; explanation attributes the difference to within-store correlation rather than sample size.

**Stretch.** Repeat with a ratio metric (units per labour hour) and show the delta method and bootstrap disagree; determine which is wrong by simulation.

### Lab 4.2 — Difference-in-differences with a parallel-trends audit

**Objective.** Estimate a DiD effect and defend or reject the identifying assumption.

**Data.** Course VMI panel: 100 stores × 78 weeks (52 pre, 26 post), with documented assignment and one known confounder — a regional competitor closure in post-week 3.

**Steps.** (1) Build the 2×2 and the raw DiD. (2) Estimate two-way fixed effects with store-clustered errors. (3) Run an event study with 12 leads and 26 lags and plot coefficients with intervals. (4) Run a placebo date 26 weeks before the real start. (5) Run a negative-control outcome in a category not stocked in pilot stores. (6) Report estimate, interval, break-even, and a recommendation.

**Deliverable.** Four-page memo with the 2×2 table, event-study plot, placebo results, and a recommendation with a stop condition.

**Acceptance criteria.** Correct DiD arithmetic; store-level clustering; leads included; both placebo and negative control reported; memo states that parallel trends is untestable and names what would break it.

**Stretch.** Add a synthetic-control estimate using the 60 control stores as donors, then explain the divergence.

### Lab 4.3 — Elasticity with and without an instrument

**Objective.** Show the direction and size of endogeneity bias in a price elasticity estimate.

**Data.** Course promotion panel: three years of store-week data for 200 SKUs with shelf price, paid price, feature and display flags, competitor price, and a manufacturer list-price series.

**Steps.** (1) Estimate OLS log-log elasticity with store and week fixed effects and clustered errors. (2) Re-estimate with feature and display included. (3) Estimate 2SLS on the list-price series; report the first stage, F statistic, and second-stage elasticity. (4) Compute revenue and margin effects of a 10% price rise under each estimate. (5) State the exclusion argument in three sentences plus one threat to it.

**Deliverable.** Notebook and one-page memo with the elasticity table (OLS, OLS with merchandising controls, 2SLS), margin arithmetic, and exclusion argument.

**Acceptance criteria.** Correct log-log interpretation; first-stage F reported; margin arithmetic correct to the cent; exclusion argument names an economic mechanism and a specific threat.

**Stretch.** Add a weak-instrument-robust Anderson-Rubin interval and show how the conclusion moves when the instrument is deliberately weakened.

### Lab 4.4 — Censored demand correction

**Objective.** Estimate demand rather than sales when stockouts censor the outcome.

**Data.** Store-SKU-day sales and inventory snapshots for 500 SKUs over two years with an in-stock flag; a synthetic version with known true demand is provided for validation.

**Steps.** (1) Identify stockout days. (2) Fit a baseline forecast on observed sales with no correction. (3) Fit a Tobit model treating stockout days as censored. (4) Fit a heuristic scaling in-stock-day sales by the in-stock fraction. (5) Compare all three by velocity decile against true demand.

**Deliverable.** Comparison table (method × velocity decile × WAPE against true demand) plus a recommendation with a stated limitation.

**Acceptance criteria.** Censored observations correctly identified; all methods scored on the same rows; WAPE computed against true demand only on the synthetic set; the recommendation names the decile where the method breaks.

**Stretch.** Add substitution — demand transfers to a substitute when the primary is out — and show how much the naive method overstates the substitute.

## 8. Assignment — Causal demand protocol

Produce a **causal demand protocol** for one real or realistic intervention. It is a decision document written before analysis, graded on whether another analyst could execute it without asking a question.

| Section | Content | Length |
|---|---|---|
| 1. Decision | Decision, owner, cadence, cost of a wrong answer | 1 paragraph |
| 2. Estimand | Population, treatment, outcome, summary, horizon | 5 lines |
| 3. DAG | Nodes, edges, adjustment set, prohibited controls | 1 figure + 10 lines |
| 4. Design | Chosen design and why alternatives were rejected | 1 page |
| 5. Identification | Assumption, test, falsification condition | 10 lines |
| 6. Power | Minimum detectable effect and available sample | 1 table |
| 7. Robustness | Placebo, negative control, sensitivity analysis | 10 lines |
| 8. Decision rule | Thresholds for scale, stop, redesign | 5 lines |
| 9. Communication | One sentence for planners and the range they will see | 3 lines |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Estimand precision | 20% | Names a metric, not a quantity; no horizon | All five parts present but summary or horizon vague | Decision-relevant, versioned, and matches the analysis exactly |
| Design and identification | 25% | Method chosen by familiarity | Design fits the question; assumption stated | Alternatives compared; assumption tested; falsification condition named |
| Statistical execution | 20% | Naive standard errors; no clustering | Clustering correct; diagnostics run | Clustering at assignment level; HAC where needed; bootstrap for non-standard metrics |
| Robustness | 15% | None | One placebo or negative control | Placebo, negative control, specification curve, and confounding sensitivity |
| Communication | 20% | Reports p-values and coefficients | Converts to units and currency | Gives decision range, break-even, and a governance rule a planner can act on |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| EU AI Act (Regulation 2024/1689) | High-risk systems require risk management, data governance, documentation, logging, and human oversight | Keep a decision log linking each recommendation to its counterfactual and its approver |
| GDPR / CCPA-CPRA | Minimisation, lawful basis, purpose limitation | Aggregate to store or lane level where individual data is unnecessary |
| Statistical disclosure control | Small-cell counts can re-identify a supplier or store | Suppress cells below a threshold; report distributions |
| Antitrust | Estimating competitor response is lawful; sharing competitor non-public data is not | Restrict inputs to public or licensed sources; legal review before competitor-signal pricing |
| Supplier fairness | A biased scorecard can trigger penalties or delisting | Publish metric definition, denominator, and adjustment; allow dispute |
| Commercial | A causal claim in a business case becomes a commitment | Separate point estimate, range, and break-even in every case |

Two ethical points deserve emphasis. Selection bias in supplier scorecards is a fairness issue, not only a statistical one: suppliers assigned difficult lanes are penalised for the assignment. And uncertainty hidden from planners is a form of misrepresentation — a point forecast presented without its range invites overcommitment, and the person bearing the consequence is rarely the person who built the model.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Correlation sold as causation | "Impact" appears with no counterfactual | No comparison group or pre-period | Budget committed to a non-effect | Require the counterfactual sentence before any number is shown |
| Naive pre/post comparison | "Before we sold X, after we sold Y" | Seasonality attributed to treatment | Effect overstated or sign-flipped | Add a control group or seasonal baseline |
| Clustering ignored | Standard errors shrink as rows are added | Repeated observations treated as independent | False significance | Cluster at assignment level |
| Multiple comparisons uncorrected | "We tested 40 stores, 6 were significant" | FWER inflation | Two or three false winners scaled | Pre-specify primary metric; BH for exploratory |
| Bad control included | Covariate could have been changed by treatment | Mediator or collider adjustment | Bias in an unpredictable direction | Post-treatment variable audit |
| Collider conditioning | Analysis restricted to "valid" records | Selection induced by the outcome | Spurious, plausible-looking association | Define the sample before seeing outcomes |
| Stockouts ignored | Zero-sales days treated as zero demand | Censoring | Low forecasts for popular items; recurring stockouts | Stockout flag plus censored model |
| Missingness ignored | Complete-case analysis at 30% missing | MAR or MNAR | Bias toward the missing mechanism | Multiple imputation for MAR; bounds for MNAR |
| Weak instrument | First-stage F below 10 | Instrument barely moves treatment | Unstable IV, biased toward OLS | Report F; use weak-instrument-robust intervals |
| Parallel trends assumed | No pre-period plot | Divergent trends | Contaminated DiD | Event study with leads; placebo date |
| Elasticity from observational price | Coefficient moves with promotion intensity | Simultaneity and omitted merchandising | Wrong price recommendation | Cost-shifter instrument; feature and display controls |
| Uncertainty omitted | Single number in the business case | Point estimate only | Overcommitment; surprise at the low end | Report range and break-even |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | What is an estimand, and why does it precede the model? | Names population, treatment, outcome, summary, horizon. Fixing it prevents specification shopping. |
| 2 | Give a case where a great predictive model is a bad decision model. | Display coefficient; stockout prediction used for safety stock without a causal availability model. Strong answers name the confounder. |
| 3 | Your pilot shows +12% at p = 0.04. What are your next three questions? | What was the comparison? Was treatment assigned at the unit analysed? How many tests were run? |
| 4 | When is a 95% interval misleading? | Clustering ignored; block structure ignored; ratio with a noisy denominator; the parameter is a LATE for a different population. |
| 5 | Draw a DAG for "does expediting reduce late delivery?" and name a bad control. | Confounders: lane distance, order priority, carrier. Bad control: whether the order was expedited after a delay was detected — a collider. |
| 6 | A supplier ranks last on your scorecard. How could that be an artefact? | Lane mix, order-size mix, denominator excludes cancellations, missingness correlated with lateness, volume versus order weighting. |
| 7 | Why is sales not demand, and what does it change? | Censoring by availability; changes the target, the metric, the allocation; creates a stockout feedback loop. |
| 8 | Compare DiD and synthetic control for a national rollout. | DiD needs credible parallel trends; synthetic control suits one treated unit with a long pre-period and donor pool; inference differs. |
| 9 | What makes an instrument valid, and what can you test? | Relevance (first-stage F), exclusion (economic argument, not directly testable), independence. Test relevance; overidentify if possible. |
| 10 | How do you explain a wide interval without losing the decision? | Convert to units and currency, name break-even, state what narrows it, set a stop condition. |
| 11 | A vendor claims 30% fewer stockouts. What evidence do you require? | Baseline definition, comparison period, concurrent changes, stockout definition, demand shift, holdout or stepped rollout, cost. |

## 13. Key terms

| Term | Definition |
|---|---|
| Estimand | Precise quantity to estimate: population, treatment, outcome, summary, horizon |
| ATE / ATT / LATE | Average effect across the population / on the treated / for units the instrument moved |
| Counterfactual | The outcome without the treatment; always unobserved |
| Confounder | Variable causing both treatment and outcome |
| Collider | Variable caused by both treatment and outcome; conditioning on it creates bias |
| Mediator | Variable on the causal path from treatment to outcome |
| Bad control | Covariate whose inclusion biases the estimate, usually because it is post-treatment |
| DAG | Directed acyclic graph encoding assumed causal relationships |
| DiD | Difference-in-differences; compares changes across treated and control groups |
| Parallel trends | Absent treatment, treated and control would have moved in parallel |
| Synthetic control | Weighted donor pool reproducing a treated unit's pre-period trajectory |
| IV / 2SLS | Instrumental variables / two-stage least squares |
| RDD | Regression discontinuity design exploiting a threshold rule |
| Exclusion restriction | The instrument affects the outcome only through the treatment |
| HAC | Heteroskedasticity- and autocorrelation-consistent standard errors (Newey-West) |
| Cluster-robust SE | Standard errors accounting for within-cluster correlation |
| Block bootstrap | Resampling contiguous blocks to preserve autocorrelation |
| FWER / FDR | Family-wise error rate / false discovery rate |
| MCAR / MAR / MNAR | Missing completely at random / at random / not at random |
| Censoring | Observed value is a bound, not the true value; stockout sales censor demand |
| E-value | Minimum confounder association strength needed to explain away an effect |
| Elasticity | Percentage change in quantity per percentage change in price |
| Break-even effect | Smallest effect at which the intervention covers its cost |

## 14. Sources and further reading

**Standards and regulation**

- Regulation (EU) 2024/1689 (Artificial Intelligence Act), consolidated text, EUR-Lex. Accessed 27 October 2026.
- ASCM, Supply Chain Operations Reference Digital Standard (SCOR DS), metric definitions. Accessed 27 October 2026.
- GS1, General Specifications (GTIN, GLN, SSCC) and EPCIS 2.0. Accessed 27 October 2026.
- ISO 3534-1, *Statistics — Vocabulary and symbols — Part 1*.
- ISO 8601, *Date and time — Representations for information interchange*.

**Peer-reviewed and methodological**

- Angrist, J. D. "Lifetime Earnings and the Vietnam Era Draft Lottery." *American Economic Review*, 1990.
- Angrist, J. D., and Pischke, J.-S. *Mostly Harmless Econometrics*. Princeton University Press, 2009.
- Abadie, A., Diamond, A., and Hainmueller, J. "Synthetic Control Methods for Comparative Case Studies." *Journal of the American Statistical Association*, 2010.
- Card, D., and Krueger, A. B. "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania." *American Economic Review*, 1994.
- Hernán, M. A., and Robins, J. M. *Causal Inference: What If*. Chapman and Hall/CRC, 2020. Open access.
- Imbens, G. W., and Rubin, D. B. *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press, 2015.
- Lee, D. S. "Randomized Experiments from Non-random Selection in U.S. House Elections." *Journal of Econometrics*, 2008.
- Pearl, J. *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge University Press, 2009.
- Vulcano, G., van Ryzin, G., and Ratliff, R. "Estimating Primary Demand for Substitutable Products from Sales Transaction Data." *Marketing Science*, 2012.
- Efron, B., and Tibshirani, R. J. *An Introduction to the Bootstrap*. Chapman and Hall, 1993.
- Cameron, A. C., and Miller, D. L. "A Practitioner's Guide to Cluster-Robust Inference." *Journal of Human Resources*, 2015.
- Benjamini, Y., and Hochberg, Y. "Controlling the False Discovery Rate." *Journal of the Royal Statistical Society B*, 1995.
- VanderWeele, T. J., and Ding, P. "Sensitivity Analysis in Observational Research: Introducing the E-Value." *Annals of Internal Medicine*, 2017.
- Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021. Open access.

**Industry and practice**

- Gilliland, M. *The Business Forecasting Deal*. Wiley, 2010. Source for forecast value added.
- Bijmolt, T. H. A., van Heerde, H. J., and Pieters, R. G. M. "New Empirical Generalizations on the Determinants of Price Elasticity." *Journal of Marketing Research*, 2005.
- Hendricks, K. B., and Singhal, V. R. "The Effect of Supply Chain Glitches on Shareholder Wealth." *Journal of Operations Management*, 2003.

**Datasets and tools**

- M5 (Walmart) and Rossmann store-level data, Kaggle. Accessed 27 October 2026.
- `statsmodels` documentation — `OLS`, `Logit`, `Poisson`, `QuantReg`, `MixedLM`, `cov_type`. https://www.statsmodels.org — accessed 27 October 2026.
- `linearmodels` documentation for panel and IV estimators. https://bashtage.github.io/linearmodels/ — accessed 27 October 2026.
- DoWhy and EconML documentation for refutation and heterogeneous effects. Accessed 27 October 2026.

> **Note:** M5 and Rossmann are competition datasets. They are suitable for method practice but are not representative of any specific company's operations, and their error levels must never be quoted as achievable production performance.

---

> **Next:** [Module 05](05_experimentation_pilots_evidence.md) — how to design the experiment or pilot that produces the counterfactual this module assumes.
