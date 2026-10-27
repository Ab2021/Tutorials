# Module 08 — Hierarchical, Intermittent, and Probabilistic Forecasting

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 34 hours (12 lecture, 12 lab, 10 project)
> **Track fit:** Common core; required for Retail & E-commerce and Manufacturing & Industrial; recommended for Logistics & Distribution
> **Prerequisites:** Module 07 (benchmarks, backtesting, diagnostics); Module 06 (calibration, leakage); Module 01A (units, time buckets, distributions)
> **Tools:** Python 3.12, pandas, NumPy, SciPy, statsforecast, hierarchicalforecast, matplotlib, DuckDB
> **Industry question:** "The numbers you publish do not add up, half your SKUs sell on one day in five, and the planner needs a 90th percentile rather than a mean — which of those three problems is actually costing you service?"

---

## 1. Why industry cares

Module 07 ends with a point forecast at one grain. No planning system consumes that. A planning system consumes a **coherent set** of numbers across a product hierarchy, a location hierarchy, and a time hierarchy; it consumes them for a portfolio in which most items are slow-moving; and it consumes them to set buffers, which requires the tail of the distribution rather than the middle.

Three failures recur in every demand-planning review, and they have different causes and different fixes:

1. **Incoherence.** The category forecast is 10,400 cases, the five brand forecasts sum to 9,870, and the SKU forecasts sum to 10,120. The planner notices. From that moment the forecast is treated as an opinion rather than a measurement, and overrides rise. Incoherence is not a rounding problem; it is a structural problem that appears whenever each level is forecast independently, which is the default in most planning stacks.
2. **Intermittent demand.** In a typical aftermarket or grocery portfolio, 40–65% of SKUs have demand in fewer than three-quarters of periods, and those SKUs often carry 30–50% of the inventory value even when they carry a small share of revenue. Methods tuned on fast movers over-forecast them systematically, and the standard accuracy metrics cannot even be computed on them.
3. **Point forecasts feeding buffer decisions.** Safety stock is a quantile of lead-time demand. If the forecast is a mean and the planner adds a fixed "10% cushion", the resulting service level is an accident, not a decision. Two items with the same mean and different variance get the same buffer, which is exactly backwards.

The 2026 context sharpens all three. Planning suites now advertise automatic reconciliation and probabilistic forecasting as table stakes, and the analyst's job has shifted from computing a reconciled number to **defending which covariance structure, which service level, and which distribution** were used. Gartner's September 2025 prediction that 70% of large organizations will adopt AI-based supply chain forecasting or touchless demand planning by 2030 sits alongside vendor surveys in 2026 showing only about 10% reporting AI embedded in planning decisions — a gap that is mostly about trust in the numbers, and trust is built by coherence and calibration, not by model class.

The cost is concrete. An incoherent hierarchy forces planners to hold buffer at every level "just in case", which in a three-level network can add 15–25% to total safety stock for the same service. A biased intermittent-demand method on 4,000 slow movers produces excess stock that ages into markdown. A mean-based buffer at a 95% target delivers something between 80% and 97% depending on the item's variance — a range wide enough to lose a retail account.

## 2. Learning outcomes

By the end of this module, you can:

- Map a real planning hierarchy across product, location, channel, customer, and time, and write the summing matrix that defines coherence.
- Produce bottom-up, top-down, middle-out, and MinT-reconciled forecasts, and explain which one wins at which level and why.
- Derive the MinT estimator for OLS, WLS, and shrinkage covariance, and implement it on a small hierarchy by hand.
- Reconcile a temporal hierarchy (monthly to quarterly to annual) and explain when temporal reconciliation helps more than cross-sectional reconciliation.
- Classify a portfolio with ADI and CV², and select Croston, SBA, TSB, or aggregate-disaggregate per class with a stated reason.
- Specify a compound Poisson or hurdle model for a slow-moving item and compute its lead-time demand distribution.
- Construct split-conformal and adaptive-conformal intervals for time series, and state precisely what the coverage guarantee does and does not cover.
- Score probabilistic forecasts with pinball loss and CRPS, and diagnose calibration with PIT and coverage-by-horizon.
- Convert a forecast distribution into a newsvendor quantile, a safety-stock level, and a plan range, and defend the choice to finance.
- Audit a hierarchy for coherence, bias propagation, and level-specific degradation before publishing.

## 3. Mental model

### 3.1 Three axes, one number

```
   TIME             PRODUCT          LOCATION
   annual  ─┐       total            national
   quarter  ├──►    family    ──►    region
   month    │       SKU              store
   week    ─┘       all summed by the same S matrix
```

A forecast set is coherent when every aggregate equals the sum of its parts, on every axis, at every time bucket. Reconciliation is the projection of an incoherent set onto the coherent subspace — the nearest coherent set under a chosen metric.

### 3.2 The projection picture

```
   base forecasts ŷ (incoherent)  ──►  ŷ_rec = S (S' W⁻¹ S)⁻¹ S' W⁻¹ ŷ
                                            │
                                            ├──► point plan at the decision grain
                                            └──► quantiles ──► safety stock / service
```

`W` encodes what you believe about the relative reliability of each level: OLS says every series is equally trustworthy, WLS says the aggregate is more trustworthy than the detail, and shrinkage pulls `W` toward a stable diagonal.

### 3.3 Which level do you forecast, which do you reconcile?

| Situation | Forecast at | Reconcile to | Why |
|---|---|---|---|
| Dense history at every level | All levels, then MinT | All levels | Uses all information; usually lowest total error |
| Sparse bottom level, stable aggregates | Top level | Disaggregate by stable shares | Bottom level is noise |
| Local promotions and store effects | Bottom level | Aggregate up | Local signal is real and invisible at the top |
| One level is the planning grain (DC-week) | Middle level | Both directions | Matches the decision; both directions are constrained |
| Very large hierarchy, tight runtime | Bottom level, structural WLS | All levels | OLS/shrinkage may be computationally heavy at 10⁶ series |

## 4. Core concepts

### 4.1 Hierarchy structures

A **strictly hierarchical** structure has one parent per node: total → division → category → subcategory → SKU, or national → region → district → store. A **grouped** structure has nodes with multiple parents: the same SKU belongs to a brand group and a pack-size group, and those groupings cross. Grouped structures matter because most real product hierarchies are not trees — a "promotional pack" is both a pack-size member and a channel member.

| Axis | Typical levels | Summing relation | Trap |
|---|---|---|---|
| Product | Total, division, category, subcategory, brand, SKU | Quantity or value | Unit-of-measure mixing (cases versus eaches) |
| Location | National, region, DC, store, customer | Quantity | Cross-docking means a unit passes through two nodes |
| Channel | Total, channel, account, store | Quantity or value | Channel reclassification mid-year |
| Customer | Total, banner, account, ship-to | Quantity or value | Account merges and acquisitions |
| Time | Year, quarter, month, week, day | Quantity | 4-5-4 versus ISO weeks; 53-week years |

Write the summing matrix explicitly before modelling. For a two-level tree with total `T` and children `A`, `B`:

```
        [ T ]   [1 1]        [ a ]
   ŷ =  [ A ] = [1 0]  β  =  [ b ]
        [ B ]   [0 1]
```

`S` is 3×2. Every reconciliation method in this module is a rule for choosing `β` given a base forecast vector `ŷ`.

### 4.2 Aggregation consistency and the four classical methods

| Method | Rule | Strength | Weakness |
|---|---|---|---|
| Bottom-up | Forecast leaves, sum upward | Always coherent; local detail preserved | Leaf noise propagates; low-volume leaves are the worst forecasts in the system |
| Top-down | Forecast total, split by historical or forecast proportions | Stable, smooth, immune to leaf noise | Assumes proportions are stationary; suppresses real local variation |
| Middle-out | Forecast the planning level, aggregate up, disaggregate down | Matches the decision grain; balances noise | Needs a good middle level; two separate disaggregation assumptions |
| MinT | Forecast all levels, project onto coherence with weight `W` | Usually lowest aggregate error; uses all information | Needs `W`; heavier computation; can move a level in a counter-intuitive direction |

**Worked reconciliation example.** One product family, two SKUs, one week. Base forecasts from independent models are incoherent: total 210, SKU A 120, SKU B 95, whose sum is 215. Historical share of A is 55%.

| Method | Total | SKU A | SKU B | Note |
|---|---:|---:|---:|---|
| Base (incoherent) | 210.00 | 120.00 | 95.00 | Sums to 215 |
| Bottom-up | 215.00 | 120.00 | 95.00 | Total moves to the sum of parts |
| Top-down (55/45) | 210.00 | 115.50 | 94.50 | Parts move to the total |
| MinT-OLS | 211.67 | 118.33 | 93.33 | Squared-error projection |
| MinT-WLS (4:1:1) | 210.56 | 117.78 | 92.78 | Trusts the aggregate more |

The OLS arithmetic, spelled out. With `S` above, `S'S = [[2,1],[1,2]]`, `(S'S)⁻¹ = (1/3)[[2,−1],[−1,2]]`, and `S'ŷ = [210+120, 210+95] = [330, 305]`:

```
β̂ = (1/3) [ 2·330 − 305 ]  = (1/3)[355] = 118.33
          [ −330 + 2·305 ]         [280]     93.33
ŷ_rec = S β̂ = [211.67, 118.33, 93.33]
```

The WLS version weights the total four times as heavily as each SKU, because the total's forecast variance was estimated at 25 against 100 for each SKU. Solving the weighted normal equations `5a + 4b = 960` and `4a + 5b = 935` gives `a = 117.78`, `b = 92.78`, total 210.56. The aggregate barely moves from its base, and the SKUs absorb the correction — which is what you want when the aggregate is genuinely more reliable.

**Industry interpretation.** Bottom-up is the safe default when leaves are dense. Top-down is the safe default when leaves are sparse. In practice the choice should be made **per segment**, not per hierarchy: use bottom-up for the 20% of SKUs that carry 80% of volume, and MinT-WLS for the long tail where leaf forecasts are noise. Publishing one method for the whole hierarchy is a policy choice that costs accuracy somewhere.

### 4.3 MinT in detail

Wickramasuriya, Athanasopoulos, and Hyndman (2019) show that the reconciliation problem is a generalized least-squares projection:

```
ŷ_rec = S β̂,   β̂ = (S' W⁻¹ S)⁻¹ S' W⁻¹ ŷ
```

where `W = E[e e']` is the covariance of base-forecast errors. Three practical choices:

| Estimator | `W` | When it wins | When it fails |
|---|---|---|---|
| OLS | Identity | Small hierarchies, unknown error structure, first pass | Ignores that aggregates are usually more accurate |
| WLS-structural | `diag(S·1)` — weight by the number of leaves aggregated | No residual history; new hierarchy; grouped structure | Assumes error scales with aggregation count |
| WLS-variance | `diag(Ŵ)` from in-sample residuals | Enough history per series; variance differs sharply by level | Ignores error correlation across levels |
| MinT-shrinkage | `λ diag(Ŵ) + (1−λ) Ŵ` | Default in production; stabilises a noisy `Ŵ` | Needs a tuning window for `λ` |

Shrinkage matters because `Ŵ` estimated from 20–50 forecast errors per series is nearly singular, and inverting it amplifies noise. The shrinkage estimator pulls the off-diagonal correlations toward zero with weight `(1−λ)`, keeping the useful correlation structure while guaranteeing a well-conditioned matrix. Reported empirical work typically finds shrinkage at or near the top of the accuracy ranking, and rarely far behind the best method on any given dataset; `λ` between 0.5 and 1.0 is a reasonable starting grid.

Two implementation warnings. First, `W` must be estimated on **forecast errors**, not on the series themselves — using the series covariance is a common and silent mistake. Second, the residuals must come from the same forecast method and horizon as the base forecasts; mixing one-step residuals with a 12-step forecast understates the error and over-trusts the model.

### 4.4 Temporal and cross-temporal hierarchies

A temporal hierarchy stacks the same series at several frequencies: day → week → month → quarter → year. Each frequency has its own forecast, and those forecasts are usually incoherent — twelve monthly forecasts do not sum to the annual forecast.

| Approach | How | Benefit | Cost |
|---|---|---|---|
| Temporal reconciliation (Athanasopoulos et al., 2017) | Forecast each frequency, reconcile across frequencies | Uses the stability of the annual view and the detail of the daily view | More models; needs a temporal summing matrix |
| Cross-temporal reconciliation | Reconcile across both time and cross-section simultaneously | Best measured accuracy in tourism and energy studies | Large `S`; computationally expensive; harder to explain |

**Rule of thumb.** Temporal reconciliation earns its keep when the series is noisy at the planning frequency and the annual or quarterly aggregate is stable — typical of promotional weekly data and of highly seasonal categories. It earns much less when the series is already smooth at the planning frequency. For a weekly SKU-store forecast feeding a weekly replenishment policy, temporal reconciliation is usually not worth the operational complexity; for an annual financial plan built from monthly demand, it often is.

### 4.5 Intermittent demand: classification and methods

Definitions, computed on the demand history only (never on a calendar that includes non-selling days):

```
ADI  = number of periods / number of periods with nonzero demand
CV²  = (standard deviation of nonzero demand sizes / mean of nonzero demand sizes)²
```

| ADI | CV² | Class | Share in a typical aftermarket | Starting method |
|---|---|---|---|---|
| < 1.32 | < 0.49 | Smooth | 15–25% | ETS, ARIMA |
| ≥ 1.32 | < 0.49 | Intermittent | 25–35% | SBA, TSB |
| < 1.32 | ≥ 0.49 | Erratic | 15–25% | Robust ETS, TSB |
| ≥ 1.32 | ≥ 0.49 | Lumpy | 20–35% | Aggregate-disaggregate, compound Poisson |

The 1.32 and 0.49 cut-offs are conventions from the Syntetos–Boylan classification, not statistical laws. They are a triage tool. A series just over the line does not change behaviour because of the line.

**Croston (1972).** Maintain two exponentially smoothed series: the interval between demands `p_t` and the size of demand `z_t`. Forecast the rate:

```
p_t = α (t − t_last) + (1−α) p_{t−1}      (updated only when demand occurs)
z_t = α y_t        + (1−α) z_{t−1}        (updated only when demand occurs)
ŷ   = z_t / p_t
```

**SBA (Syntetos–Boylan, 2001).** Croston is biased upward by roughly `(1 − α/2)`; multiply by 0.95 for the commonly used α:

```
ŷ_SBA = 0.95 · z_t / p_t
```

**TSB (Teunter–Syntetos–Babai, 2011).** Updates the **probability** of occurrence every period, including zero periods, which makes it respond to obsolescence:

```
p_t = α d_t + (1−α) p_{t−1}        d_t = 1 if demand occurred, else 0
z_t = β y_t + (1−β) z_{t−1}        updated only when demand occurs
ŷ   = p_t · z_t
```

| Method | Responds to obsolescence | Bias | Best class | Failure mode |
|---|---|---|---|---|
| Croston | No | Upward | Intermittent | Slow to reach zero when an item dies |
| SBA | No | Corrected | Intermittent | Still lags a decline |
| TSB | Yes | Near-neutral | Intermittent, erratic, end-of-life | More parameters; noisy on very short histories |
| Aggregate-disaggregate | Inherited from group | Depends on share stability | Lumpy | Share drift when assortment changes |

**Worked TSB example.** An item sold in 3 of the last 12 weeks, sizes 8, 6, and 10, with `α = 0.2`, `β = 0.3`, starting `p = 0.25`, `z = 8`.

| Week | Demand | `p_t` | `z_t` | Forecast `p·z` |
|---|---:|---:|---:|---:|
| 1 | 0 | 0.200 | 8.00 | 1.60 |
| 2 | 8 | 0.360 | 8.00 | 2.88 |
| 3 | 0 | 0.288 | 8.00 | 2.30 |
| 4 | 0 | 0.230 | 8.00 | 1.84 |
| 5 | 0 | 0.184 | 8.00 | 1.47 |
| 6 | 0 | 0.147 | 8.00 | 1.18 |

The forecast decays toward zero without any explicit obsolescence rule. That is the property that makes TSB the default for end-of-life and slow-moving portfolios, and the property that makes it dangerous for an item with a genuine seasonal gap — it will keep decaying through the off-season.

### 4.6 Compound Poisson and hurdle models

For very slow movers, a rate-and-size model is more honest than smoothing.

```
Arrivals:  N_L ~ Poisson(λL)                    over lead time L
Sizes:     Z_i iid with mean μ_z and variance σ_z²
Demand:    D_L = Σ_{i=1}^{N_L} Z_i

E[D_L]   = λL μ_z
Var[D_L] = λL (μ_z² + σ_z²)
```

The variance is larger than the mean whenever sizes vary — the reason a Poisson demand assumption understates buffer requirements for lumpy items. The distribution of `D_L` is computed by convolution (or by simulation), and the quantile is then read directly.

| Model | Assumption | Use when | Watch |
|---|---|---|---|
| Poisson | Unit sizes, constant rate | High-volume, low-value eaches | Understates variance when order sizes vary |
| Compound Poisson | Poisson arrivals, arbitrary sizes | Spare parts, wholesale orders | Needs a size distribution |
| Negative binomial | Over-dispersed counts | Count data with variance > mean | Fitting on short histories is unstable |
| Hurdle / zero-inflated | Two processes: occurrence, then size | Many zeros with a separate buying process | Easy to overfit; difficult to explain to planners |

### 4.7 Probabilistic forecasts: what a distribution is for

A probabilistic forecast is a predictive distribution `F_{t+h|t}`. It answers "how uncertain", not just "how much". Three ways to produce one:

| Approach | How | Strength | Weakness |
|---|---|---|---|
| Parametric | Assume a family (normal, lognormal, gamma, negative binomial), estimate parameters | Compact, smooth, easy to explain; quantiles are closed-form | Wrong family gives confidently wrong tails |
| Empirical / bootstrap | Resample residuals or simulate paths | No family assumption; captures skew | Needs history; quantiles are noisy beyond the sample |
| Quantile regression / direct quantile model | Fit each quantile separately | No distributional assumption; flexible covariates | Quantiles can cross; needs pinball training |
| Conformal | Calibrate any point or quantile model with held-out residuals | Finite-sample marginal coverage | Only marginal coverage; assumes exchangeability |

**Distribution choice by demand type.**

| Demand type | Reasonable family | Why | Avoid |
|---|---|---|---|
| High-volume, stable | Normal | Symmetric, well-behaved | At low means, where normal gives negative quantiles |
| High-volume, multiplicative error | Lognormal or gamma | Positive, right-skewed | Normal on the raw scale |
| Counts, low mean | Poisson or negative binomial | Discrete, non-negative | Normal |
| Lumpy | Compound Poisson, empirical | Two-part structure | Any single symmetric family |
| Multi-modal (promo versus base) | Mixture or empirical | Two regimes | Single unimodal family |

Check the quantile for non-negativity and monotonicity before it reaches a planning system. A negative 5th-percentile forecast is not a small problem; it will break a replenishment calculation or be silently floored at zero, which biases the whole distribution.

### 4.8 Conformal prediction for time series

Split conformal is simple and useful. Fit the model on a training block, compute absolute residuals on a calibration block, take the `⌈(n+1)(1−α)⌉`-th smallest residual as the half-width `q`, and publish `[ŷ ± q]`.

```python
import numpy as np

def split_conformal(residuals, y_hat, alpha=0.1):
    """Absolute residuals from a held-out calibration block. Coverage is
    marginal under exchangeability, which time series violate -- so report
    realised coverage by horizon, not the nominal level."""
    k = min(int(np.ceil((len(residuals) + 1) * (1 - alpha))), len(residuals))
    q = np.sort(residuals)[k - 1]
    return y_hat - q, y_hat + q, q
```

Three corrections matter in practice:

1. **Exchangeability fails.** Consecutive time-series errors are correlated, so the nominal 90% interval may cover 85% or 94% depending on the regime. Report realised coverage by horizon and by segment.
2. **Heteroskedasticity.** A constant half-width is too wide in calm periods and too narrow in volatile ones. Normalise residuals by a scale estimate (a "normalised conformal" variant), or use adaptive conformal inference (Gibbs and Candès, 2021), which updates `α_t` online to track the realised miscoverage rate.
3. **Horizon.** Fit the calibration on residuals at the **same horizon** as the forecast. A one-step residual distribution is far too narrow for a 13-week forecast.

EnbPI (Xu and Xie, 2021) and conformal time-series forecasting (Stankeviciute et al., 2021) are the two most cited time-series-specific constructions; both are worth knowing by name and both are implemented in open-source libraries. The practical message is simpler than the literature: conformal gives you a defensible interval when you cannot justify a parametric family, at the cost of assuming a stable error distribution over the calibration window.

### 4.9 Scoring probabilistic forecasts

**Pinball loss** for quantile level `q`:

```
L_q(y, ŷ_q) = q (y − ŷ_q)          if y ≥ ŷ_q
            = (1 − q)(ŷ_q − y)     if y < ŷ_q
```

Averaged over a grid of quantiles (typically 0.01 to 0.99 in steps of 0.01 or 0.05), it approximates the **continuous ranked probability score**:

```
CRPS(F, y) = ∫₀¹ 2 L_q(y, F⁻¹(q)) dq
```

For a normal predictive distribution there is a closed form, useful for checking an implementation:

```
CRPS(N(μ,σ), y) = σ [ z(2Φ(z) − 1) + 2φ(z) − 1/√π ],   z = (y − μ)/σ
```

**Worked CRPS.** `μ = 100`, `σ = 20`.

| Actual `y` | `z` | CRPS | Interpretation |
|---:|---:|---:|---|
| 100 | 0.00 | 4.67 units | Forecast at the median; CRPS ≈ 0.23σ |
| 120 | 1.00 | 12.05 units | One sigma high; penalised but not catastrophically |

**Calibration diagnostics.** Three checks, all cheap:

| Check | What it shows | Failure signature |
|---|---|---|
| PIT histogram | Distribution of `F(y)`; should be uniform | U-shape: intervals too narrow; hump: too wide |
| Coverage by horizon | Realised share inside the interval, per `h` | Coverage falls with `h`: uncertainty understated at long horizons |
| Interval width | Sharpness subject to calibration | Widths identical across SKUs: variance ignored |

Report calibration **and** sharpness together. An interval of `[0, ∞]` is perfectly calibrated and useless; a sharp interval that covers 70% when it claims 90% is worse than useless because it will be trusted.

### 4.10 Decision use of quantiles

This is the section that turns a forecast into money.

**Newsvendor quantile.** With underage cost `C_u` (margin lost per unit short) and overage cost `C_o` (cost per unit left over), the optimal order quantity is the `q*` quantile of demand:

```
q* = C_u / (C_u + C_o)
```

| Situation | `C_u` | `C_o` | `q*` | Order at |
|---|---:|---:|---:|---|
| Full-price retail, 40% margin, salvage at cost | 10 | 0 | 1.00 | Maximum plausible demand — usually wrong because `C_o = 0` is unrealistic |
| Fashion, 50% margin, 20% markdown | 10 | 8 | 0.56 | Just above the median |
| Critical spare part, downtime cost dominates | 500 | 20 | 0.96 | Far tail |

**Mapping a service target to a quantile.** Cycle service level is the probability of no stockout in a replenishment cycle, which is exactly the CDF of lead-time demand evaluated at the reorder point. A 95% CSL is the 0.95 quantile of lead-time demand, not the 0.95 quantile of the one-step forecast. Module 10 builds the policy; the point here is that **the forecast distribution must be over the lead-time horizon**, not over the planning bucket.

**Three different intervals, often confused:**

| Interval | Question it answers | Built from | Used for |
|---|---|---|---|
| Forecast interval | How uncertain is demand? | Predictive distribution | Safety stock, service target |
| Plan range | What range of outcomes can the plan absorb? | Demand distribution plus supply constraints | Capacity, sourcing, S&OP |
| Commitment | What have we promised? | Contract, not a model | Customer and supplier agreements |

Publishing a forecast interval and calling it a plan range is a common and expensive error: the plan range must widen for supply uncertainty, and narrow for committed supply.

### 4.11 Reference implementation

Two blocks. The first reconciles a hierarchy with the Nixtla stack; the second implements MinT directly so the arithmetic is auditable.

```python
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, CrostonSBA, TSB
from hierarchicalforecast.utils import aggregate
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, TopDown, MinTrace


def reconcile_hierarchy(df: pd.DataFrame, spec: list, h: int = 12,
                        freq: str = "MS", season_length: int = 12):
    """df columns: [unique_id, ds, y] at the leaf level only. spec is the
    level definition, leaf last: [['Total'], ['Total','State'], ['Total','State','Store']].
    Pin the hierarchicalforecast version; aggregate() and reconcile() have changed."""
    Y_df, S_df, tags = aggregate(df=df, spec=spec)

    sf = StatsForecast(models=[AutoETS(season_length=season_length)],
                       freq=freq, n_jobs=-1)
    Y_hat_df = sf.forecast(df=Y_df, h=h, fitted=True)

    hrec = HierarchicalReconciliation(reconcilers=[
        BottomUp(),
        TopDown(method="forecast_proportions"),
        MinTrace(method="ols"),
        MinTrace(method="wls_struct"),
        MinTrace(method="mint_shrink"),
    ])
    return hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_df, S=S_df, tags=tags), S_df, tags


def mint(base: np.ndarray, S: np.ndarray, W: np.ndarray | None = None) -> np.ndarray:
    """MinT reconciliation. base: (n_series,), S: (n_series, n_bottom).

    W=None gives OLS. Pass diag variances for WLS, or a full covariance
    from forecast residuals for full MinT. The projection is the same
    operation in all three cases; only W changes.
    """
    n = S.shape[0]
    W = np.eye(n) if W is None else np.asarray(W, dtype=float)
    Winv = np.linalg.inv(W)
    A = S.T @ Winv @ S
    beta = np.linalg.solve(A, S.T @ Winv @ base)
    return S @ beta


if __name__ == "__main__":
    S = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
    base = np.array([210.0, 120.0, 95.0])
    print("OLS :", np.round(mint(base, S), 2))
    print("WLS :", np.round(mint(base, S, np.diag([25.0, 100.0, 100.0])), 2))
```

Expected output, matching the worked example in §4.2:

```
OLS : [211.67 118.33  93.33]
WLS : [210.56 117.78  92.78]
```

If your output differs by more than rounding, check the orientation of `S` first — a transposed summing matrix is the most common bug in reconciliation code, and it fails silently because the result is still coherent.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| Demand history | SKU × location × period | ERP / POS | Zeros explicit; stockouts flagged; returns netted consistently |
| Hierarchy master | SKU × location × as-of date | PIM / ERP | Versioned; effective-dated; historic forecasts reconciled on the historic hierarchy |
| Summing matrix | Built from master | Derived | Must be regenerated whenever the hierarchy changes |
| Calendar | Period | Planning system | One calendar convention, documented; 53-week handling |
| Forecast errors | Series × horizon × vintage | Forecast archive | Required for `W` estimation; most systems do not retain this |
| Service and cost inputs | SKU × location | Finance / operations | Underage and overage costs for the newsvendor quantile |

**Standards.** GS1 GTIN for item identity and GLN for location identity; without stable identifiers the hierarchy cannot be versioned, and reconciliation across a master-data change becomes guesswork. ASCM SCOR DS for the service metrics that the quantiles ultimately drive. ISO 8601 for dates and week numbering, with the retail 4-5-4 calendar chosen once and documented.

**Workflow checks before publishing any reconciled set:**

1. `S @ β` equals the published vector at every level, to numerical tolerance. Assert it in code, not by eye.
2. The hierarchy version used for reconciliation matches the version in the planning system.
3. Bottom-level forecasts are non-negative and quantiles are monotone in `q`.
4. Coverage is reported by level as well as overall — a reconciled hierarchy can be well calibrated at the total and badly calibrated at the leaf.
5. Intermittent series were reconciled with a method that does not produce fractional units where the item ships in cases; apply the case pack after reconciliation, and accept that rounding breaks exact coherence (document the tolerance).

## 6. Industry case studies

### 6.1 M5 and the hierarchy metric (Walmart data, 2020–2021)

**Context.** The M5 Accuracy competition used Walmart daily sales: 42,840 hierarchical series across 3,049 products and 10 stores in three states, 1,913 training days, a 28-day horizon, 5,507 teams.

**Intervention.** The scoring rule, WRMSSE, forced competitors to produce forecasts that were coherent across 12 aggregation levels and weighted by dollar sales, scaled by each series' in-sample seasonal-naive error. The winning entry was an equal-weighted combination of 220 LightGBM models with a Tweedie objective, organised per store, per store-category, and per store-department, reaching WRMSSE 0.520 — 22.4% better than the exponential-smoothing bottom-up benchmark.

**Evidence.** The published results paper and the competition repository report the ranking and the metric definition. This is competition evidence, not a production trial.

**Outcome.** All top-50 methods were machine learning and beat the statistical benchmark by more than 14%. Cross-learning, price and promotion features, combination, and time-based validation were the reported drivers.

**What transferred.** The metric design: a hierarchy-aware, revenue-weighted, scaled error forces coherence and stops a model from winning on the easy series. Any organisation publishing a hierarchy should adopt an equivalent weighted metric rather than an unweighted mean WAPE.

**What did not.** The winning ensemble. 220 models is a research artifact; the operational lesson is that combination and cross-learning help, not that a specific count of models is a design. And WRMSSE's revenue weighting is wrong for a spare-parts portfolio, where the tail carries the inventory.

### 6.2 Australian tourism and optimal reconciliation (2019)

**Context.** Australian domestic tourism visitor nights form a deep geographic hierarchy — national, state, tourism region — with strongly seasonal series and very different volumes at each level. It is the reference dataset for the `hts` and later reconciliation work.

**Intervention.** Wickramasuriya, Athanasopoulos, and Hyndman (2019) introduced MinT: estimate the error covariance `W`, project the base forecasts onto the coherent subspace, and compare against bottom-up, top-down, and OLS.

**Evidence.** The JASA paper reports that MinT with a shrinkage covariance estimator outperforms the classical methods across the hierarchy in the tourism application, and that the gains are largest where the base forecasts at different levels disagree most.

**Outcome.** Optimal reconciliation became the default in the R `hts` package and, later, in commercial planning tools. The deeper result is that the choice of `W` matters less than the decision to reconcile at all.

**What transferred.** The estimator and the discipline of estimating `W` from forecast errors. **What did not.** The exact covariance estimator — tourism series are long, smooth, and annual, with hundreds of observations per series; a 12-week-old SKU with 40 observations cannot estimate a covariance matrix and must fall back to structural WLS.

### 6.3 GEFCom2014 and the probabilistic track (2016)

**Context.** The Global Energy Forecasting Competition 2014 ran five tracks — load, price, wind, solar, and probabilistic load — with the probabilistic track requiring a full quantile forecast at 99 levels for each horizon.

**Intervention.** Competitors submitted quantiles rather than point forecasts, scored by pinball loss averaged across quantiles. This forced explicit treatment of the tails and made calibration a first-class objective.

**Evidence.** Hong, Pinson, Fan, Zareipour, Troccoli, and Hyndman (2016) report the design, the scoring, and the comparative results across methods.

**Outcome.** The competition established quantile scoring and the pinball/CRPS family as the standard for probabilistic forecasting in energy, and the approach transferred into load-forecasting practice at utilities.

**What transferred.** Scoring on quantiles rather than on a point plus an ad hoc interval; the practice of publishing a full predictive distribution to the operational user. **What did not.** The energy setting's smooth, high-frequency, strongly periodic demand. Retail SKU demand is far noisier, with intermittency and stockouts, and the same quantile grid is far harder to estimate.

### 6.4 DeepAR and global probabilistic models (2020)

**Context.** Salinas, Flunkert, Gasthaus, and Januschowski (2020) described DeepAR, an autoregressive recurrent network trained on many related series at once to produce a predictive distribution per series.

**Intervention.** A single global model learns shared seasonality and dynamics across a large panel, then emits distribution parameters (for example, negative binomial or normal) per series and step.

**Evidence.** The IJF paper reports evaluations on several public and proprietary datasets, including retail demand, showing that the global model is competitive with or better than per-series classical methods, particularly for series with short histories.

**Outcome.** Global probabilistic models became a standard option in forecasting libraries, and the "one model for thousands of series" pattern is now common in retail.

**What transferred.** The idea that a distribution can be learned jointly across series, which is exactly what a sparse portfolio needs. **What did not.** The claim that a neural model is required: the same joint structure can be captured by conformalised LightGBM quantile models at a fraction of the operational cost, and the paper's own comparisons are dataset-specific.

## 7. Labs

### Lab 8.1 — Reconcile a three-level hierarchy and score it

**Objective.** Produce coherent forecasts at every level and measure whether reconciliation helps or hurts at the decision grain.

**Data.** M5 or Favorita, aggregated to a state → store → department hierarchy (or a synthetic hierarchy if runtime is a constraint); at least 100 bottom-level series and 2 years of weekly history.

**Steps.** (1) Build `S` from the master data and assert its shape. (2) Generate base forecasts at all levels with AutoETS. (3) Reconcile with bottom-up, top-down, OLS, WLS-structural, and shrinkage MinT. (4) Score each method with WAPE at the leaf, at the middle, and at the total. (5) Compute a revenue-weighted scaled error across levels. (6) Report the level at which each method wins.

**Deliverable.** A reconciliation report with a method × level table, a coherence assertion log, and a recommendation per level.

**Acceptance criteria.** Every reconciled set passes an exact coherence check; `W` is estimated from forecast residuals and the estimation window is stated; results are reported at three levels, not one; the recommendation names a method per level and gives the cost of the extra complexity.

**Stretch.** Estimate `W` with three different shrinkage intensities and plot leaf WAPE against `λ`.

### Lab 8.2 — Intermittent-demand benchmark by class

**Objective.** Show where each intermittent method fails, by demand class rather than on average.

**Data.** 2,000 SKUs × 104 weeks from UCI Online Retail II, or a synthetic generator calibrated to a stated ADI/CV² mix.

**Steps.** (1) Compute ADI and CV² on nonzero demand. (2) Assign the four classes and report the revenue-weighted share of each. (3) Backtest SES, Croston, SBA, TSB, and aggregate-disaggregate with MASE at a 4-week horizon. (4) Report accuracy by class and by revenue decile. (5) Identify the class where each method is worst and explain the mechanism. (6) Recommend a per-class method policy.

**Deliverable.** A classification table, an accuracy-by-class table, and a one-page policy recommendation.

**Acceptance criteria.** Diagnostics computed on nonzero demand only; MASE denominator stated; revenue weighting included; the recommendation is per class, not a single global method; the end-of-life behaviour of TSB is demonstrated on at least one series.

**Stretch.** Add a compound Poisson model and compare on the lumpy class at the 0.9 quantile rather than at the mean.

### Lab 8.3 — Probabilistic scoring and calibration

**Objective.** Produce quantile forecasts and evaluate them honestly.

**Data.** The Lab 8.1 panel plus a synthetic panel with known heteroskedasticity.

**Steps.** (1) Produce quantiles 0.05 to 0.95 from a parametric model, a quantile-regression model, and a split-conformal wrapper. (2) Score with pinball loss at each quantile and with CRPS. (3) Plot PIT histograms by segment. (4) Compute realised coverage by horizon. (5) Inject a variance regime change and show which method degrades first. (6) Report calibration and sharpness side by side.

**Deliverable.** A calibration dossier with PIT plots, coverage tables, and a scoring table.

**Acceptance criteria.** Pinball computed correctly (spot-check three observations by hand); coverage reported per horizon; calibration and sharpness both reported; the conformal coverage claim is stated as marginal and approximate, with the exchangeability caveat written down.

**Stretch.** Implement adaptive conformal inference and show coverage recovery after the regime change.

### Lab 8.4 — From quantile to safety stock

**Objective.** Connect the forecast distribution to an inventory decision and quantify the cost of getting the quantile wrong.

**Data.** Lab 8.1 panel plus a cost table (unit cost, price, salvage, stockout penalty).

**Steps.** (1) Compute the critical fractile per SKU from the cost table. (2) Read the corresponding quantile from each predictive distribution. (3) Convert to a lead-time demand quantile using the lead time and its variability. (4) Compare the buffer from the mean-plus-flat-percentage heuristic against the quantile-based buffer. (5) Estimate the service and cost consequences with a simple newsvendor calculation. (6) Identify the SKUs where the two approaches diverge most.

**Deliverable.** A short memo with a per-SKU comparison table and an estimate of the aggregate buffer difference.

**Acceptance criteria.** Critical fractile derived, not assumed; lead-time horizon used rather than the planning bucket; the flat-percentage baseline computed explicitly; divergence explained by variance, not by assertion.

**Stretch.** Add lead-time variability and show how the buffer changes when supplier lead time is uncertain.

## 8. Assignment — Probabilistic forecast dossier

Produce a **dossier** for one hierarchy that a planning director could sign off. It must answer: which reconciliation method, which intermittent policy, which distribution, and what service level the numbers imply.

| Section | Content |
|---|---|
| 1. Decision and grain | The planning decision, the grain it requires, and the horizon |
| 2. Hierarchy | Levels, summing matrix, version, and the coherence definition |
| 3. Base forecasts | Method per level, backtest design, and base accuracy |
| 4. Reconciliation | Methods compared, `W` estimation, results by level |
| 5. Intermittent policy | Classification, method per class, evidence |
| 6. Distribution | Family or empirical method per segment, with justification |
| 7. Scoring | Pinball, CRPS, coverage by horizon, PIT diagnostics |
| 8. Decision translation | Critical fractiles, buffer comparison, service implication |
| 9. Risks | Where the distribution is least trustworthy and what would change the answer |
| 10. Reproduction | Lockfile, data hash, one-command rerun, seed |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Hierarchy modelling | 15% | Levels listed, no summing matrix | `S` built and coherence asserted | Grouped structure handled; hierarchy versioning addressed; `S` regenerated on master-data change |
| Reconciliation method | 20% | One method, no comparison | Three methods compared at one level | `W` estimated from residuals; shrinkage tuned; method chosen per level with a cost argument |
| Intermittent demand | 20% | One method for all SKUs | ADI/CV² classification and per-class methods | Revenue-weighted class analysis; end-of-life behaviour demonstrated; compound Poisson where warranted |
| Probabilistic method | 20% | Point forecast plus fixed percentage | Parametric interval with stated family | Parametric, empirical, and conformal compared; exchangeability and horizon caveats explicit |
| Scoring and calibration | 15% | Interval coverage asserted | Pinball and coverage reported | CRPS, PIT, coverage by horizon, sharpness, and a segment-level diagnosis |
| Decision translation and communication | 10% | Quantile handed over without interpretation | Critical fractile computed | Buffer and service consequences quantified; memo readable by a finance reviewer |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| Forecast versus commitment | A reconciled forecast shared with a supplier can be read as a commitment | Label the artifact; state the accuracy range and the non-binding status |
| Allocation fairness | Quantile-based buffers concentrate inventory in high-margin or high-variance locations | Report service outcomes by region and store format; set a service floor for essential items |
| Essential goods | A 90% CSL is a commercial choice; for medicines and staple foods it is a public-health choice | Separate commercial and essential-item policies; document the service floor and who approved it |
| EU AI Act (Regulation 2024/1689) | Automated allocation affecting access to goods may fall within high-risk obligations depending on use | Document method, validation, human oversight, and the appeal path for affected parties |
| Vendor claims | Reconciliation and probabilistic modules are marketed with accuracy uplift numbers | Require a backtest on your data, at your grain, with your intermittency mix |
| Computational cost | Full MinT on 10⁶ series can dominate the planning batch window | Structural WLS or a blocked covariance; measure runtime, not just accuracy |

The commercial argument for this module is working capital. The same service target, implemented with a correctly specified lead-time demand distribution instead of a flat percentage, typically moves total buffer by 10–25% — in either direction. Presenting that as a range with its assumptions is more credible than presenting a single saving, and it survives the first review by finance.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Incoherent hierarchy | Planner notices totals do not sum | Independent forecasts per level | Forecast loses credibility; overrides rise | Reconcile and assert coherence in code |
| Transposed summing matrix | Reconciled output still coherent but wrong | `S` orientation error | Silent, systematic distortion | Unit test `S` against known aggregates |
| `W` estimated from series, not errors | Suspiciously good reconciliation | Wrong covariance | Over-trusts one level | Estimate from forecast residuals, same horizon |
| Shrinkage ignored on short history | Unstable reconciled values week to week | Near-singular covariance inversion | Forecast noise amplified | Shrinkage or structural WLS |
| Reconciliation at the wrong grain | Leaf accuracy falls after reconciliation | Optimising the total | Decision grain degraded | Score at the decision grain, not the total |
| Croston on a dying item | Forecast stays positive for months after demand stops | No occurrence update | Obsolete stock | TSB, or an explicit end-of-life rule |
| Mean-based safety stock | Buffer identical for high- and low-variance SKUs | Point forecast used as quantile | Service varies unpredictably | Critical fractile per SKU |
| One-step residuals for a 13-week interval | Intervals too narrow at long horizons | Horizon mismatch | Chronic stockouts in volatile items | Calibrate at the decision horizon |
| Constant-width conformal | Same interval width across all SKUs | Heteroskedasticity ignored | Over-buffer calm items, under-buffer volatile ones | Normalised or adaptive conformal |
| Non-monotone quantiles | 90th percentile below the 80th | Independent quantile models | Broken buffer calculation | Sort and repair; use a joint model |
| Negative lower quantile | Negative demand in the plan | Normal family at low mean | Floored at zero, biased distribution | Lognormal, gamma, or empirical |
| Calibration never checked | No PIT, no coverage report | Intervals assumed correct | False confidence in service | Coverage by horizon in every review |
| Reconciliation cost unmeasured | Batch window overruns | Full covariance at scale | Planning cycle delayed | Benchmark runtime against the cadence |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | Why is a hierarchy incoherent by default? | Each level is forecast by a separate model with separate features and errors; nothing enforces the summing relation. |
| 2 | When would you choose bottom-up over MinT? | Dense leaves with real local signal, simple operational ownership, no reliable error covariance; MinT wins on average but not always at the leaf. |
| 3 | What does `W` represent in MinT, and how do you estimate it? | Covariance of base-forecast errors at the forecast horizon; from archived forecast errors, with shrinkage for short histories. |
| 4 | A colleague says "reconciliation always improves accuracy". Is that right? | No. It usually improves the aggregate and can degrade a specific level; score at the decision grain. |
| 5 | How do you decide between Croston, SBA, and TSB? | Classify with ADI/CV²; SBA corrects Croston's bias; TSB when obsolescence or end-of-life matters. |
| 6 | What is wrong with a Poisson demand assumption for spare parts? | It assumes unit sizes and equal variance to the mean; lumpy demand needs compound Poisson or empirical. |
| 7 | Explain the coverage guarantee of split conformal in one sentence, and its main caveat. | Marginal coverage over calibration and test under exchangeability; time series violate exchangeability, so check realised coverage by horizon. |
| 8 | Pinball versus CRPS — when do you use each? | Pinball for a single quantile or a quantile grid; CRPS when a full distribution is available and you want one number. |
| 9 | How does the critical fractile change with margin? | Higher margin raises the underage cost and the optimal quantile; a 50% margin with cheap salvage lands near the median. |
| 10 | Why is a 95% CSL not the 95th percentile of next week's demand? | CSL is the CDF of demand over the protection interval (lead time plus review), not over one planning bucket. |

## 13. Key terms

| Term | Definition |
|---|---|
| Hierarchical forecast | A set of forecasts across nested aggregation levels |
| Summing matrix `S` | The matrix mapping bottom-level forecasts to every level |
| Coherence | The property that aggregates equal the sum of their parts |
| Bottom-up | Forecast leaves and sum upward |
| Top-down | Forecast the total and split by proportions |
| Middle-out | Forecast a middle level, aggregate up, disaggregate down |
| MinT | Minimum-trace reconciliation; a generalized least-squares projection |
| Shrinkage estimator | A covariance estimate pulled toward a stable diagonal |
| ADI | Average demand interval; periods per demand occurrence |
| CV² | Squared coefficient of variation of nonzero demand sizes |
| Croston | Separate smoothing of demand interval and size |
| SBA | Syntetos–Boylan approximation; bias-corrected Croston |
| TSB | Teunter–Syntetos–Babai; updates occurrence probability every period |
| Compound Poisson | Poisson arrivals with a random size distribution |
| Predictive distribution | The distribution of a future value given information to date |
| Quantile forecast | An estimate of a specific point of the predictive distribution |
| Pinball loss | Asymmetric loss minimised by the true quantile |
| CRPS | Continuous ranked probability score; integral of pinball over all quantiles |
| PIT | Probability integral transform; a calibration diagnostic |
| Conformal prediction | Distribution-free interval construction from held-out residuals |
| Coverage | The realised share of outcomes inside a stated interval |
| Critical fractile | The service level that minimises expected newsvendor cost |

## 14. Sources and further reading

**Standards and regulation**

- GS1, *General Specifications*, current issue; GTIN and GLN identifier rules. https://www.gs1.org/standards — accessed 27 October 2026.
- ASCM, Supply Chain Operations Reference Digital Standard (SCOR DS). Accessed 27 October 2026.
- ISO 8601, *Date and time — Representations for information interchange*.
- Regulation (EU) 2024/1689 (Artificial Intelligence Act), EUR-Lex. https://eur-lex.europa.eu/eli/reg/2024/1689/oj — accessed 27 October 2026.

**Peer-reviewed and methodological**

- Hyndman, R. J., Ahmed, R. A., Athanasopoulos, G., and Shang, H. L. "Optimal combination forecasts for hierarchical time series." *Computational Statistics & Data Analysis*, 2011.
- Wickramasuriya, S. L., Athanasopoulos, G., and Hyndman, R. J. "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization." *Journal of the American Statistical Association*, 114(526), 2019.
- Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., and Petropoulos, F. "Forecasting with temporal hierarchies." *European Journal of Operational Research*, 262(1), 2017.
- Kourentzes, N., and Athanasopoulos, G. "Cross-temporal coherent forecasts for Australian tourism." *Annals of Tourism Research*, 2019.
- Croston, J. D. "Forecasting and stock control for intermittent demands." *Operational Research Quarterly*, 1972.
- Syntetos, A. A., and Boylan, J. E. "On the bias of intermittent demand estimates." *International Journal of Production Economics*, 2001.
- Syntetos, A. A., and Boylan, J. E. "The accuracy of intermittent demand estimates." *International Journal of Forecasting*, 2005.
- Teunter, R. H., Syntetos, A. A., and Babai, M. Z. "Intermittent demand: Linking forecasting to inventory obsolescence." *European Journal of Operational Research*, 2011.
- Gneiting, T., and Raftery, A. E. "Strictly proper scoring rules, prediction, and estimation." *Journal of the American Statistical Association*, 2007.
- Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A., and Hyndman, R. J. "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond." *International Journal of Forecasting*, 32(3), 2016.
- Salinas, D., Flunkert, V., Gasthaus, J., and Januschowski, T. "DeepAR: Probabilistic forecasting with autoregressive recurrent networks." *International Journal of Forecasting*, 36(3), 2020.
- Xu, C., and Xie, Y. "Conformal prediction interval for dynamic time-series." *Proceedings of ICML*, 2021.
- Stankeviciute, K., Alaa, A. M., and van der Schaar, M. "Conformal time-series forecasting." *Advances in Neural Information Processing Systems*, 2021.
- Gibbs, I., and Candès, E. J. "Adaptive conformal inference under distribution shift." *Proceedings of ICLR*, 2021.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. "M5 accuracy competition: Results, findings, and conclusions." *International Journal of Forecasting*, 2022.
- Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021. Open access.

**Industry and analyst**

- Gartner, "Gartner Predicts 70% of Large Organizations Will Adopt AI-Based Supply Chain Forecasting to Predict Future Demand by 2030," press release, 16 September 2025. https://www.gartner.com/en/newsroom/press-releases/2025-09-16-gartner-predicts-70-percent-of-large-orgs-will-adopt-ai-based-supply-chain-forecasting-to-predict-future-demand-by-2030 — accessed 27 October 2026. Analyst prediction, not an outcome.
- M5 competition repository and data. https://github.com/Mcompetitions/M5-methods — accessed 27 October 2026.

**Datasets and tools**

- `hierarchicalforecast` documentation: `aggregate`, `HierarchicalReconciliation`, `BottomUp`, `TopDown`, `MinTrace`. https://nixtlaverse.nixtla.io/hierarchicalforecast — accessed 27 October 2026.
- `statsforecast` documentation: `AutoETS`, `CrostonSBA`, `TSB`, `cross_validation`. https://nixtlaverse.nixtla.io/statsforecast — accessed 27 October 2026.
- Hyndman's `hts` package for R, including the Australian tourism example data. Accessed 27 October 2026.
- UCI Machine Learning Repository, Online Retail II dataset. Accessed 27 October 2026.

> **Note:** Coverage is a property of a procedure over repeated samples, not of one interval. Never report a single interval's hit rate as proof of calibration.

---

> **Next:** [Module 09](09_demand_sensing_promotions_npi.md) — shortening the horizon with POS signals, decomposing a promotion into baseline and lift, and forecasting a product that has never sold.
