# Module 09 — Demand Sensing, Promotions, and New Product Forecasting

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 30 hours (10 lecture, 12 lab, 8 project)
> **Track fit:** Required for Retail & E-commerce and Manufacturing & Industrial; recommended for Logistics & Distribution
> **Prerequisites:** Module 07 (benchmarks, backtesting); Module 08 (hierarchies, distributions); Module 04 (causal inference, elasticity endogeneity); Module 05 (experiments, uplift)
> **Tools:** Python 3.12, pandas, NumPy, statsmodels, scikit-learn, LightGBM, DuckDB; POS and trade-promotion-management extracts
> **Industry question:** "Your promotion returned a 45% lift and a negative profit — which of those two numbers does finance believe, and what would you have had to measure to know before you funded it?"

---

## 1. Why industry cares

Statistical forecasting extrapolates a history. Demand sensing tries to answer a different question: **what is happening now that the history cannot yet see?** A retailer that sees a competitor's store close on Tuesday, a heat wave forecast for the weekend, or a shipment surge into a distribution centre has information that a 52-week seasonal model will not incorporate until the effect is already in the sales series — by which time the replenishment order has shipped.

The commercial stakes sit in three places.

**Promotions.** Trade spend is one of the largest controllable line items in consumer goods, commonly reported in the range of 15–25% of gross sales depending on category and market; the exact figure is company-specific and should be taken from the firm's own P&L, not from an industry average. A large share of promotions do not pay back. The analytical failure is rarely the lift estimate; it is that the lift is measured against a baseline that already contains the promotion's own pull-forward, or that the incremental margin ignores the discount given on units that would have sold anyway.

**New products.** A new SKU has no history, and the forecast for it drives the first buy, the first allocation, and the shelf space decision. Launch forecasts are usually made under a deadline with partial information, and the cost of being wrong is asymmetric: too low means lost sales and a delayed ramp; too high means markdown, write-off, and a damaged relationship with the buyer.

The 2026 context: retail media, loyalty-card transaction data, and store-level inventory feeds have made daily signals available at a granularity that did not exist a decade ago, while privacy regulation and retailer data-sharing terms have made access more contested. Meanwhile the analytical standard has risen. A vendor claiming "AI demand sensing" is making a claim about signal latency and decision integration, not about a model class, and it should be evaluated on that basis.

## 2. Learning outcomes

By the end of this module, you can:

- Distinguish demand sensing from demand forecasting by decision latency, horizon, and signal, and decide which a given planning process needs.
- Map the signal ladder from POS to DC shipment to store inventory, and quantify the lead each signal provides.
- Build a promotion calendar dataset with the fields a model requires, including mechanic, display, feature, price, and funding.
- Decompose observed sales into baseline, incremental lift, cannibalization, halo, and pull-forward, with a stated identification strategy.
- Estimate own- and cross-price effects with a defensible causal design, and explain why naive elasticity is biased.
- Choose between a response model and an uplift model for a promotion targeting decision, and evaluate with Qini or an equivalent.
- Forecast a new product with analogy, attribute-based, and Bass-diffusion methods, and state the uncertainty range honestly.
- Model distribution and assortment effects (ACV, numeric distribution) that dominate launch performance.
- Design a post-launch learning process that updates the forecast as real sales arrive.
- Compute promotion ROI on an incremental-margin basis and reconcile it with the total profit change that finance will report.

## 3. Mental model

### 3.1 The signal ladder

```
  statistical forecast  ──►  POS / orders  ──►  store inventory, weather
  (weeks–months)             (days–weeks)         (hours–days)
  plan and buy               replenish            execute
```

Each step to the right shortens the horizon and raises the resolution. The mistake is to use a long-horizon model for a short-horizon decision, or to build a sensing capability that no decision consumes.

### 3.2 Sales decomposition

```
Observed sales  =  baseline
                 + promotional lift
                 + halo (complementary items)
                 − cannibalization (own portfolio)
                 − pull-forward from future periods
                 + noise, stockout censoring, distribution change
```

Every term except the baseline has to be measured, not assumed. The baseline is itself a model, and the choice of baseline determines the lift — a point that is worth repeating in every review.

### 3.3 Which method for which question

| Question | Method | Evaluation |
|---|---|---|
| How much did this promotion sell? | Baseline plus lift decomposition | Backtest against non-promoted weeks; DiD where a control exists |
| What will this new SKU sell in week 1? | Analogy or attribute-based | Post-launch error at weeks 4, 8, 13 |
| Is demand shifting this week? | Nowcast with leading indicators | Signal-to-noise, false-alarm rate, decision value |

## 4. Core concepts

### 4.1 Demand sensing versus demand forecasting

| Dimension | Statistical forecasting | Demand sensing |
|---|---|---|
| Horizon | 4–52 weeks | 1–14 days |
| Input | History, calendar, plan | POS, orders, inventory, weather, web signals |
| Refresh | Weekly or monthly | Daily or intraday |
| Decision served | Buy, produce, allocate | Replenish, reposition, expedite |
| Failure mode | Blind to a regime change | Noisy, over-reacts, no owner |
| Success metric | WAPE at the planning horizon | Decision latency and service/cost outcome |

Sensing is not a better forecast; it is a shorter loop. A sensing model with 4% WAPE at a 3-day horizon can be worse for a 13-week buy than a seasonal model with 18% WAPE, because the decision horizons differ. The right question is always "which decision, at what latency".

**Nowcasting** is the formal version: estimate the current state from partial, high-frequency indicators before the official series is complete. It is standard in macroeconomics and increasingly used for store-level demand, where the current week's POS is available before the weekly sales extract closes.

### 4.2 Signals, lead times, and failure modes

| Signal | Refresh | Typically leads the sales series by | Failure mode |
|---|---|---|---|
| Store POS | Daily | 0 days (it is demand) | Stockouts censor it; returns not netted |
| E-commerce orders | Hourly | 0 days | Bot traffic; cancelled orders |
| DC shipments to store | Daily | 1–7 days | Order-up-to policy hides demand; forward buying |
| Store on-hand inventory | Daily | Negative lead (lags demand) | Scan errors; cycle-count drift |
| Weather forecast | Daily | 1–14 days | Forecast error; nonlinear response |
| Search and web traffic | Hourly | 1–7 days | Media spend confounds; seasonality |
| Loyalty transactions | Daily–weekly | 0 days, with panel bias | Coverage of total sales; panel churn |
| Competitor openings and closures | Event-driven | Days–weeks | Incomplete coverage; legal limits on sharing |

**The sell-in versus sell-out distinction is the single most common sensing error.** Shipments to a store reflect the store's ordering policy, not consumer demand. When a promotion causes forward buying, shipments spike before the promotion and collapse after it, while POS shows the true pattern. Sensing on shipments without a POS anchor will amplify the bullwhip rather than dampen it.

### 4.3 Promotion mechanics and calendar data

| Mechanic | What it is | Data fields required | Modelling note |
|---|---|---|---|
| Temporary price reduction (TPR) | Shelf price cut, usually funded by the manufacturer | Regular price, promo price, funding rate, start and end | Distinguish from permanent price change |
| Feature | Item in the retailer's circular or app | Feature flag, page, position | Often co-occurs with display; collinear |
| Display | Secondary placement (end-cap, aisle, pallet) | Display type, location, duration | Strong effect; frequently missing from data |
| Coupon | Price reduction at redemption | Face value, redemption rate, distribution | Redemption lag; cost is realised, not planned |
| Digital offer / loyalty | Targeted price or points | Segment, offer value, redemption | Selection bias in who receives it |

The promotion calendar is a **forecast input**, and it must be known at forecast time. Three data-quality rules apply:

1. **Version and date-stamp the calendar.** A promotion added after the forecast was cut must not be back-filled into the model's features; it belongs in a separate "unplanned event" adjustment.
2. **Record the mechanic even when it is inferred.** If the sales data shows a price drop with no matching TPM record, flag it — unrecorded promotions are common and they bias the baseline upward.
3. **Align calendars across systems.** POS, TPM, and finance often use different week definitions. Validate that a known promotion lands in the same week in all three before modelling.

### 4.4 Baseline and incrementality

Two definitions matter, and they are not the same:

```
Incremental units   = actual units − counterfactual baseline units
Lift (%)            = incremental units / baseline units
Incremental revenue = incremental units × realised promo price
Incremental margin  = incremental units × (promo price − unit cost)
```

The **baseline** is the counterfactual: what would have sold without the promotion. It is never observed. Four ways to estimate it, in descending order of credibility:

| Method | How | Strength | Weakness |
|---|---|---|---|
| Randomised control | Hold out stores or customers | Unbiased by construction | Expensive; retailer consent; spillover |
| Difference-in-differences | Compare promoted versus matched control stores, before and after | Controls for common shocks | Parallel trends assumption; control contamination |
| Synthetic control | Weighted combination of untreated units | Handles one treated unit well | Needs a good donor pool |
| Model-based counterfactual | Fit on non-promoted periods, predict the promoted period | Always available | Model misspecification; promotions cluster in seasonal peaks |

Model-based baselines are the default in practice and the weakest. They fail in a specific and predictable way: promotions are scheduled when demand is already high (holidays, paydays, weather), so a model fitted on non-promoted weeks systematically under-predicts the counterfactual and overstates lift. Chevalier, Kashyap, and Rossi's finding that retail prices tend to fall exactly when demand peaks is the demand-side mirror of this selection problem.

### 4.5 Cannibalization, halo, and pull-forward

| Effect | Definition | Measurement | Sign |
|---|---|---|---|
| Cannibalization | Promoted item steals from sibling items in the same portfolio | Cross-item lift within the promoted store-week | Negative on siblings |
| Halo | Promoted item raises sales of complements | Basket affinity or cross-lift on complements | Positive on complements |
| Brand switching | Consumers switch brand within the category | Brand share before, during, after | Negative for competitors |
| Pull-forward | Consumers buy early and buy less later | Post-promotion dip versus baseline | Negative after the promotion |
| Category expansion | The promotion grows total category demand | Category total versus baseline | Positive, usually small |

The correct incrementality measure for a manufacturer is **net of cannibalization and halo, and evaluated over a window long enough to include the post-promotion dip**. A 4-week window that ends the week the promotion ends will always look better than the truth.

**Post-promotion dip.** After a promotion, sales often fall below baseline for two to six weeks as pantry loading unwinds. If the dip is not included, the measured lift is inflated by roughly the depth of the dip times its duration. A practical rule: measure over the promotion plus a dip window equal to the observed stock-up horizon, typically 2–6 weeks for packaged goods and shorter for perishables.

### 4.6 Price and display effects

A workhorse multiplicative model, estimated in logs:

```
ln(Q_it) = α_i + β ln(P_it) + γ Feature_it + δ Display_it
           + Σ_k θ_k ln(P_kt)  (cross-price terms)
           + seasonality + ε_it
```

`β` is the own-price elasticity; `θ_k` the cross-price elasticity with item `k`. Typical published own-price elasticities for grocery categories cluster between −1.5 and −3.0 depending on category, aggregation, and whether the price change is temporary; treat any specific number as category-specific.

| Problem | Mechanism | Fix |
|---|---|---|
| Price endogeneity | Retailers set prices in response to expected demand | Instrumental variables; control function; randomised price tests |
| Collinearity | Feature, display, and TPR co-occur | Separate cells by mechanic; regularise; report the confound |

Display and feature usually dominate price for impulse categories; price dominates for staples. Estimating one coefficient without the others attributes the co-occurring display effect to price and overstates elasticity.

### 4.7 Causal machine learning and uplift

Two different prediction problems are often confused:

| Problem | Question | Target | Output |
|---|---|---|---|
| Response model | Who will buy? | Conditional outcome `E[Y given X]` | Rank by propensity |
| Uplift model | Who will buy **because of** the treatment? | `E[Y given T=1, X] − E[Y given T=0, X]` | Rank by incremental effect |

Ranking by response targets the customers who would have bought anyway. Ranking by uplift targets the persuadables. For a promotion with a fixed budget, the uplift ranking is the one that maximises incremental margin.

| Estimator | How | When |
|---|---|---|
| T-learner | Separate models for treated and control | Enough data in both arms |
| S-learner | One model with treatment as a feature | Treatment effect may be smooth |
| X-learner | Impute individual effects, then combine | Imbalanced arms |
| Causal forest | Honest sample splitting, local moment conditions | Many covariates, moderate sample |

Evaluation uses the **Qini curve**: sort by predicted uplift, plot cumulative incremental conversions against population share, and compare the area under the Qini curve (AUUC) with random targeting. A model with high AUC for response can have a Qini curve no better than random.

### 4.8 New product forecasting

**Analogy.** Find comparable products launched into similar conditions, match on attributes and distribution, and scale their curve. The quality of the answer depends almost entirely on the quality of the comparables, and the most common failure is selecting comparables that succeeded.

**Attribute-based.** Regress launch performance on item attributes (price tier, pack size, flavour, category, brand strength) across a historical launch panel. Useful when the portfolio has many launches and consistent attribute data; weak when the new item is genuinely novel.

**Bass diffusion.** With market potential `m`, innovation coefficient `p`, and imitation coefficient `q`:

```
n(t)   = p·m + (q − p)·N(t−1) − (q/m)·N(t−1)²
N(t)   = m · [1 − e^{−(p+q)t}] / [1 + (q/p)·e^{−(p+q)t}]
t*     = ln(q/p) / (p + q)          (time of peak adoption rate)
```

**Worked Bass example.** `m = 1,000,000` units, `p = 0.02`, `q = 0.40`.

| Year | Cumulative `N(t)` | Share of potential |
|---:|---:|---:|
| 1 | 24,300 | 2.4% |
| 3 | 107,300 | 10.7% |
| 5 | 254,400 | 25.4% |
| 7.1 (peak) | 475,000 | 47.5% |

Peak adoption occurs at `ln(20)/0.42 = 7.1` years, and cumulative adoption is near `m/2` at that point. The parameters are usually fitted on analogous products, not on the new product, which is why the confidence interval on `m` matters more than the point estimate. For consumer durables the curve is measured in years; for a fashion SKU or a snack flavour it is measured in weeks, and the "diffusion" is really a launch-and-decay curve.

| NPI method | Data needed | Best for | Failure |
|---|---|---|---|
| Analogy | Historic launches with attributes | Fast-moving consumer goods | Survivorship bias in comparables |
| Attribute regression | Launch panel plus item master | Large portfolios, incremental items | Poor on true innovation |
| Bass diffusion | Analogous category adoption curves | Durables, subscriptions, technologies | Parameter instability; `m` dominates |
| Pre-launch intent | Survey or pre-order data | High-consideration products | Intent–behaviour gap; small samples |

### 4.9 Assortment and distribution effects

A launch forecast is meaningless without a distribution assumption. Two metrics control it:

```
Numeric distribution = stores carrying the item / total stores
ACV distribution     = revenue of carrying stores / total revenue
Velocity             = units per carrying store per week
Total volume         = ACV-weighted store count × velocity
```

A product with 60% numeric distribution but 85% ACV distribution sells far more than the numeric figure suggests, because it is in the big stores. Launch forecasts that assume national distribution on day one over-forecast by the ramp period; forecasts that ignore the ACV weighting under-forecast the mature state. Model the distribution ramp explicitly as a curve, not a step.

### 4.10 Post-launch learning

The first four weeks of a launch contain more information than any pre-launch model. A disciplined process:

1. Fix the launch forecast and record it, with its date and assumptions.
2. At week 2, compare actual to forecast and to the plan's distribution ramp; separate velocity error from distribution error.
3. Update a Bayesian posterior over launch velocity using the observed weeks, with a prior from the analogy set.
4. At week 8, re-forecast the annual curve and re-plan the buy without overwriting the original; at week 26, classify the launch and add it — including failures — to the analogy library.

Separating **velocity error** from **distribution error** is the key diagnostic: a launch that missed because only 40% of planned stores took the item needs a selling fix, not a demand fix.

### 4.11 Worked example: baseline, lift, margin, and ROI

A packaged-goods SKU, one four-week promotion in 300 stores.

| Input | Value |
|---|---|
| Baseline units (counterfactual, from a matched-control DiD) | 1,000 |
| Actual units during promotion | 1,450 |
| Regular price / promo price | $12.00 / $10.50 |
| Unit cost | $5.00 |
| Trade funds: off-invoice allowance | $1.50 per unit sold |
| Fixed spend: display and circular | $600 |

**Step 1 — lift and incremental margin at the realised price.**

```
Incremental units = 1,450 − 1,000 = 450
Lift              = 450 / 1,000 = 45.0%
Incremental rev.  = 450 × $10.50 = $4,725
Incremental COGS  = 450 × $5.00  = $2,250
Incremental margin                 = $2,475
```

**Step 2 — trade spend.**

```
Variable allowance = 1,450 × $1.50 = $2,175
Fixed spend                        =   $600
Total trade spend                  = $2,775
```

**Step 3 — incremental ROI and the total profit change.**

```
Incremental ROI      = ($2,475 − $2,775) / $2,775 = −10.8%
Margin given away on baseline units = 1,000 × ($12.00 − $10.50) = $1,500
Total profit change  = −$300 − $1,500 = −$1,800
```

The promotion generated a 45% lift and lost $1,800. The gap between "45% lift" and "negative ROI" is the discount on the 1,000 units that would have sold anyway. The break-even increment is:

```
Break-even units = $2,775 / ($10.50 − $5.00) = 504.5 units
Break-even lift  = 504.5 / 1,000 = 50.5%
```

The promotion needed 50.5% lift to pay back and delivered 45%. Run the same promotion as a feature-only event with no price cut, and the arithmetic changes: incremental margin becomes `450 × $7.00 = $3,150`, ROI becomes `+13.5%`, and the total profit change is `+$375`. Same lift, different mechanic, opposite sign.

| Scenario | Lift | Incremental margin | Trade spend | ROI | Total profit change |
|---|---:|---:|---:|---:|---:|
| TPR at 12.5% off plus display | 45.0% | $2,475 | $2,775 | −10.8% | −$1,800 |
| Feature only, no price cut | 45.0% | $3,150 | $2,775 | +13.5% | +$375 |

This table is the whole module in one view: the lift is identical, and only the mechanic and the baseline erosion distinguish the outcomes.

### 4.12 Reference implementation

```python
import numpy as np

def decompose_promo(df, unit_cost, fixed_spend, allowance_per_unit,
                    promo_weeks, dip_weeks=2):
    """df: one row per week with columns [week, units, baseline, promo].

    baseline must come from a counterfactual design (control stores, DiD, or a
    model fitted only on non-promoted weeks). Returns promotion-level metrics
    including the post-promotion dip, which is otherwise silently dropped."""
    promo = df[df["week"].isin(promo_weeks)]
    post = df[(df["week"] > max(promo_weeks)) &
              (df["week"] <= max(promo_weeks) + dip_weeks)]

    inc_units = float((promo["units"] - promo["baseline"]).sum())
    dip_units = float((post["units"] - post["baseline"]).sum())   # negative
    net_units = inc_units + dip_units

    base_units = float(promo["baseline"].sum())
    realised_price = float((promo["revenue"].sum() / promo["units"].sum()))
    inc_margin = net_units * (realised_price - unit_cost)
    spend = allowance_per_unit * float(promo["units"].sum()) + fixed_spend
    erosion = base_units * (float(promo["regular_price"].iloc[0]) - realised_price)
    return {
        "lift_pct": inc_units / base_units,
        "dip_units": dip_units,
        "net_incremental_units": net_units,
        "incremental_margin": inc_margin,
        "trade_spend": spend,
        "incremental_roi": (inc_margin - spend) / spend,
        "baseline_margin_erosion": erosion,
        "total_profit_change": inc_margin - spend - erosion,
        "breakeven_units": spend / (realised_price - unit_cost),
    }

def bass_cumulative(t, m, p, q):
    """Cumulative Bass adoption at time t."""
    e = np.exp(-(p + q) * t)
    return m * (1 - e) / (1 + (q / p) * e)
```

On the worked example (`net_units` equals 450 before the dip), the function returns `lift_pct = 0.45`, `incremental_roi = -0.108`, and `total_profit_change = -1800.0`. If the post-promotion dip is two weeks at −60 units per week, `net_units` falls to 330, incremental margin falls to $1,815, ROI falls to −34.6%, and the total profit change becomes −$2,460.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| POS sales | SKU × store × day | Retailer POS / syndicated data | Units, revenue, regular and promo price; stockout flag |
| Shipments | SKU × DC × week | ERP / EDI 852 | Sell-in, not demand; forward-buy flag |
| Promotion calendar | SKU × store × week | TPM / retailer portal | Mechanic, funding, display, feature, dates |
| Item master | SKU | PIM | Attributes for analogy: size, price tier, category, brand |
| Launch history | SKU × week | Internal | Successes and failures, with attributes |

**Standards.** GS1 GTIN for item identity, GLN for store identity, and EPCIS 2.0 where item-level events are used for launch tracking. GS1 GPC or the retailer's own category taxonomy for category mapping; UNSPSC is a procurement taxonomy and a poor fit for consumer categories. ISO 8601 weeks, with the retail 4-5-4 calendar documented where the retailer uses it. Workflow checks: Confirm that the promotion calendar is complete for the test period (unrecorded promotions are the most common source of overstated lift); confirm that POS price and TPM-funded price agree; confirm that stockouts are flagged so that censored demand is not read as low demand; confirm that the control stores used for the counterfactual were not themselves promoted.

## 6. Industry case studies

### 6.1 Dominick's Finer Foods and countercyclical markups (2003)

**Context.** Dominick's, a Chicago supermarket chain, provided store-level scanner data covering 1989–1997, a period long enough to observe price and promotion behaviour across many categories and stores.

**Intervention.** Chevalier, Kashyap, and Rossi estimated the relationship between retail prices and demand, testing whether markups rise or fall when demand is high.

**Evidence.** The *American Economic Review* paper reports that prices fall during periods of peak demand — markups are countercyclical — consistent with demand elasticity rising in peak periods and with retail competition.

**Outcome.** The result reframed promotion timing: promotions cluster in high-demand periods not only because that is when traffic is available, but because that is when price sensitivity is highest.

**What transferred.** The analytical discipline of treating the promotion calendar as endogenous to demand, and of requiring a counterfactual that accounts for the seasonal peak. **What did not.** The specific elasticities, which are category- and chain-specific, and the assumption that a 1990s supermarket price-response structure still holds in a market with digital shelf pricing and retail media.

### 6.2 Decomposing the promotion bump (2004)

**Context.** Store-level scanner data for a frequently purchased category, with enough weeks before, during, and after promotions to separate the components of the sales bump.

**Intervention.** Van Heerde, Leeflang, and Wittink decomposed the bump into cross-brand switching, cross-store switching, stockpiling, and category expansion.

**Evidence.** The *Marketing Science* paper reports that brand switching and stockpiling account for most of the promotional bump, with primary-demand expansion a comparatively small component in the categories studied.

**Outcome.** The decomposition became the standard frame for promotion evaluation and made explicit that a manufacturer's "incremental" volume is often another brand's lost volume plus a timing shift.

**What transferred.** The four-way decomposition and the insistence on a post-promotion window. **What did not.** The relative magnitudes, which vary by category; and the store-level design, which cannot see household-level switching without a panel.

### 6.3 Who captures the value (2004)

**Context.** Trade promotions involve two parties with different margins, different costs, and different pass-through behaviour.

**Intervention.** Srinivasan, Pauwels, Hanssens, and Dekimpe modelled promotion profitability separately for the manufacturer and the retailer, allowing pass-through and category characteristics to differ.

**Evidence.** The *Management Science* paper shows that the profitability split depends on pass-through, baseline margin, and whether the promotion expands the category, and that a promotion profitable at one tier can be unprofitable at the other.

**Outcome.** Promotion ROI computed from a single firm's perspective is systematically incomplete; the same event can be a transfer rather than a value creation.

**What transferred.** The requirement to state whose P&L the ROI is computed on. **What did not.** Any universal conclusion about who wins; the paper's contribution is the conditional result, not a rule.

### 6.4 Bass diffusion and the durable-goods curve (1969)

**Context.** New consumer durables in the post-war United States — room air conditioners, televisions, refrigerators — showed S-shaped adoption that linear extrapolation could not capture.

**Intervention.** Bass proposed a mixed-influence model in which adoption is driven by an innovation term (independent of prior adopters) and an imitation term (proportional to prior adopters), yielding the formula in §4.8.

**Evidence.** The *Management Science* paper reports fittings to the first-purchase sales of several durables, with the model reproducing the peak and the subsequent decline.

**Outcome.** The Bass model became the reference diffusion model and the basis for analogy-based launch forecasting.

**What transferred.** The structural insight that adoption has a peak and that the peak time depends on the ratio `q/p`. **What did not.** The assumption of a stable market potential `m` for short-lifecycle products, and the use of annual durables data as an analogue for a 12-week fashion or snack launch.

## 7. Labs

### Lab 9.1 — Estimate promotion lift with a defensible counterfactual

**Objective.** Produce a lift estimate and show how much it moves when the counterfactual changes.

**Data.** Favorita or a retailer extract with promotion flags, price, and store metadata; at least 200 stores and 2 years of weekly data.

**Steps.** (1) Build a clean promotion calendar and verify it against price changes in the data. (2) Estimate the baseline four ways: pre-period mean, seasonal naive, regression on non-promoted weeks, and DiD with matched control stores. (3) Compute lift under each. (4) Compare the spread across methods. (5) Add a two-week post-promotion dip window and recompute. (6) Report which method you would defend and why.

**Deliverable.** A lift comparison table with four baselines, a dip-adjusted estimate, and a one-page defence of the chosen counterfactual.

**Acceptance criteria.** At least one design-based counterfactual (DiD or control stores); dip window included; the spread across methods quantified, not just the preferred number; stockout censoring addressed.

**Stretch.** Add a synthetic control using donor stores and compare with the matched-pair DiD.

### Lab 9.2 — Cannibalization and halo

**Objective.** Measure net incrementality rather than gross lift.

**Data.** Basket-level transactions (Instacart or Dunnhumby) or store-level category data with sibling items.

**Steps.** (1) Define the sibling set and the complement set from basket affinity. (2) Estimate cross-item effects during promotions with store and week fixed effects. (3) Compute cannibalized units and halo units. (4) Recompute net incremental margin. (5) Report the ratio of net to gross for the category. (6) Identify the promotions whose sign flips after adjustment.

**Deliverable.** A net-incrementality table and a short memo on which promotions survive the adjustment.

**Acceptance criteria.** Sibling and complement sets defined before estimation; fixed effects used; net-to-gross ratio reported; at least one sign-flip case discussed.

**Stretch.** Estimate the effect separately for loyal versus switching households using panel data.

### Lab 9.3 — Forecast a new SKU

**Objective.** Build three NPI forecasts and compare them after the fact.

**Data.** A launch panel of 30–100 SKUs with at least 26 weeks of post-launch sales, attributes, and distribution.

**Steps.** (1) Hold out the five most recent launches. (2) Build an analogy forecast using attribute-matched comparables. (3) Build an attribute regression on the remaining launches. (4) Fit a Bass curve to the category analogue and calibrate `m` to the distribution assumption. (5) Compare forecasts at weeks 4, 8, 13, and 26. (6) Decompose the error into velocity and distribution components.

**Deliverable.** An NPI forecast comparison with error by horizon and a decomposition of the errors.

**Acceptance criteria.** Holdout launches never used in fitting; distribution ramp modelled explicitly, not assumed flat; velocity and distribution errors separated; uncertainty stated as a range.

**Stretch.** Add a Bayesian update at week 4 and show how much the 13-week error improves.

### Lab 9.4 — Uplift targeting for a promotion budget

**Objective.** Show that targeting by response and targeting by uplift give different answers.

**Data.** A randomised or quasi-randomised promotion with treatment and control groups and customer or store covariates.

**Steps.** (1) Fit a response model and rank by propensity. (2) Fit a T-learner uplift model and rank by predicted effect. (3) Plot both Qini curves against random. (4) Compute incremental margin for the top 20% under each ranking. (5) Test the ranking on a held-out fold. (6) Recommend a targeting rule and state its assumptions.

**Deliverable.** Qini curves, an incremental-margin comparison, and a targeting recommendation.

**Acceptance criteria.** Treatment and control separated correctly; Qini computed on held-out data; comparison against random included; the recommendation states the cost of mis-targeting.

**Stretch.** Add an X-learner and compare under class imbalance.

## 8. Assignment — Demand-sensing evaluation

Produce an **evaluation** of one sensing or promotion intervention for a real or realistic category, answering whether it improved a decision and at what cost.

| Section | Content |
|---|---|
| 1. Decision and latency | The decision, its cadence, and how many hours or days earlier a signal would need to arrive |
| 2. Signals | Available signals, refresh rate, coverage, and known defects |
| 3. Baseline | Counterfactual method, control construction, and validation |
| 4. Lift and decomposition | Gross lift, cannibalization, halo, dip, net incrementality |
| 5. Economics | Incremental margin, trade spend, baseline erosion, ROI, total profit change |
| 6. Sensing value | Whether the signal changed a decision, and the cost or service consequence |
| 7. Risks | Confounds, data gaps, and the conditions under which the conclusion fails |
| 8. Recommendation | Fund, fix, or stop, with the evidence that would change the answer |
| 9. Reproduction | Data snapshot, code, seeds, and the promotion calendar version |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Counterfactual design | 25% | Pre-period average used as baseline | Model-based baseline with validation | Design-based counterfactual with parallel-trends or randomisation evidence |
| Decomposition | 20% | Gross lift only | Cannibalization or dip considered | Full decomposition with net incrementality and sign-flip analysis |
| Economics | 20% | Revenue lift reported | Incremental margin computed | Baseline erosion, total profit change, and break-even reconciled to finance |
| NPI method | 15% | Single analogy | Two methods compared | Three methods, distribution ramp, and calibrated uncertainty |
| Measurement and evaluation | 10% | No holdout | Holdout used | Qini or equivalent, held-out evaluation, and decision-value framing |
| Communication | 10% | Numbers without a decision | Recommendation given | Recommendation with the conditions under which it reverses |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| Reference pricing (US) | FTC Guides Against Deceptive Pricing, 16 CFR Part 233, govern "was/now" and comparable-value claims | Verify the prior price was genuinely offered; retain the price history |
| Reference pricing (EU) | Directive (EU) 2019/2161 amended the Unfair Commercial Practices Directive to require the prior price in reduction announcements | Check the look-back window required in each member state before publishing a "was" price |
| Trade funds and discrimination | US Robinson–Patman Act governs discriminatory promotional allowances between competing resellers | Legal review of account-specific funding before execution |
| Antitrust | Sharing forward promotion plans with competitors can facilitate coordination | No competitor data in promotion models; review data-sharing agreements |
| Consumer targeting | Personalised offers can proxy for protected characteristics | Fairness testing across segments; document the targeting logic |
| EU AI Act (Regulation 2024/1689) | Automated pricing or targeting may trigger transparency or high-risk obligations depending on use | Maintain model documentation and human review of pricing actions |
| Data access | Retailer POS and panel data are licensed, not owned | Honour usage terms; do not redistribute store-level data to third parties |
| Measurement ethics | Reporting lift without the dip or the baseline erosion misleads finance | Publish the full decomposition, including the negative terms |

The commercial point is blunt: a promotion analytics function that reports gross lift will keep funding promotions that lose money, and the loss will be attributed to "the market" rather than to the measurement. Reporting net incrementality is unpopular precisely because it is decision-relevant.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Baseline includes the promotion | Lift looks implausibly high | Model fitted on promoted weeks | Overstated lift, over-funded promotions | Fit baseline on non-promoted periods; use controls |
| Selection into peak periods | Lift concentrated in holiday weeks | Promotions scheduled when demand is high | Counterfactual under-predicted | Seasonal controls; DiD with matched stores |
| Overlapping promotions | Two mechanics in the same store-week | Attribution split arbitrarily between events | Spend misallocated to the wrong mechanic | Flag overlaps; use a holdout or a factorial design |
| Dip excluded | Measurement window ends with the promotion | Pull-forward unaccounted | Inflated lift by the dip depth | Extend the window 2–6 weeks |
| Gross lift reported as incremental | No sibling analysis | Cannibalization ignored | Portfolio volume flat while "lift" is celebrated | Net-of-cannibalization reporting |
| Shipments used as demand | Sell-in spikes before the promotion | Forward buying | Wrong inventory and wrong lift | POS anchor; forward-buy flag |
| Elasticity without instruments | Price coefficient unstable across specs | Price endogeneity | Wrong price recommendation | IV, control function, or price test |
| Response model used for targeting | High AUC, flat Qini | Ranking by propensity, not effect | Budget spent on sure buyers | Uplift model and Qini evaluation |
| Analogy set of winners only | All comparables succeeded | Survivorship bias | Launch forecast too high | Include failures in the library |
| Distribution assumed at launch | Forecast assumes day-one national distribution | Ramp ignored | First buy too large | Model the distribution ramp |
| Bass fitted on the new product | Stable parameters from four weeks of data | Overfitting a diffusion curve | False precision | Fit on analogues; wide interval on `m` |
| Post-launch forecast overwritten | No record of the original | Silent revision | No learning, no accountability | Version and archive every forecast |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | What is the difference between demand sensing and demand forecasting? | Horizon, refresh cadence, signal source, and decision served; sensing is a shorter loop, not a better model. |
| 2 | Why is a 45% lift compatible with a negative promotion ROI? | The discount applies to baseline units too; incremental margin can be smaller than the margin given away plus trade spend. |
| 3 | How do you build a counterfactual when no control stores exist? | Model-based baseline on non-promoted weeks, with seasonal controls, validated against known non-promoted periods; state the assumption. |
| 4 | Why are promotions scheduled in high-demand periods, and why does it matter for measurement? | Traffic and price sensitivity peak together; a naive baseline under-predicts the counterfactual and overstates lift. |
| 5 | What is the post-promotion dip and why include it? | Pantry loading unwinds after the event; excluding it inflates measured lift. |
| 6 | Response model versus uplift model — when does each win? | Response ranks likely buyers; uplift ranks incremental buyers; with a fixed budget, uplift maximises incremental margin. |
| 7 | How would you forecast a product with no history? | Analogy with attribute matching, attribute regression, Bass on a category analogue, and an explicit distribution ramp; report a range. |
| 8 | What does the Bass `q/p` ratio control? | The speed of imitation relative to innovation, and therefore the timing of the peak. |
| 9 | How do you measure halo? | Basket affinity or cross-item lift on complements during the promotion, with store and week controls. |
| 10 | What would make you distrust a vendor's demand-sensing claim? | No stated latency, no decision integration, no control comparison, and accuracy quoted at the wrong horizon. |

## 13. Key terms

| Term | Definition |
|---|---|
| Demand sensing | Short-horizon demand estimation from high-frequency signals |
| Sell-out | Sales from the store to the consumer |
| Baseline | The counterfactual demand without the intervention |
| Cannibalization | Volume stolen from sibling items in the same portfolio |
| Halo | Incremental volume on complementary items |
| Pull-forward | Buying early at the expense of later purchases |
| Post-promotion dip | Below-baseline sales after a promotion ends |
| Trade spend | Funds a manufacturer pays to support a promotion |
| Pass-through | Share of a trade allowance reflected in the retail price |
| TPR | Temporary price reduction |
| ACV distribution | Revenue-weighted share of stores carrying an item |
| Numeric distribution | Unweighted share of stores carrying an item |
| Uplift model | Model of the incremental effect of a treatment |
| Bass diffusion | Innovation-and-imitation model of new product adoption |
| Analogy forecasting | Forecasting a new item from comparable previous launches |

## 14. Sources and further reading

**Standards and regulation**

- U.S. Federal Trade Commission, Guides Against Deceptive Pricing, 16 CFR Part 233. https://www.ftc.gov/legal-library/browse/rules/guides-against-deceptive-pricing — accessed 27 October 2026.
- Directive (EU) 2019/2161 (Omnibus Directive), amending Directive 2005/29/EC on unfair commercial practices, including price-reduction announcement rules. EUR-Lex. Accessed 27 October 2026.
- GS1, *General Specifications*; GTIN and GLN. https://www.gs1.org/standards — accessed 27 October 2026.
- Regulation (EU) 2024/1689 (Artificial Intelligence Act). EUR-Lex. https://eur-lex.europa.eu/eli/reg/2024/1689/oj — accessed 27 October 2026.

**Peer-reviewed**

- Chevalier, J. A., Kashyap, A. K., and Rossi, P. E. "Why Don't Prices Rise During Periods of Peak Demand? Evidence from Scanner Data." *American Economic Review*, 93(1), 2003.
- Van Heerde, H. J., Leeflang, P. S. H., and Wittink, D. R. "Decomposing the Sales Promotion Bump with Store Data." *Marketing Science*, 23(3), 2004.
- Van Heerde, H. J., Gupta, S., and Wittink, D. R. "Is 75% of the Sales Increase Due to Promotion? Not Really." *Journal of Marketing Research*, 40(4), 2003.
- Srinivasan, S., Pauwels, K., Hanssens, D. M., and Dekimpe, M. G. "Do Promotions Benefit Manufacturers, Retailers, or Both?" *Management Science*, 50(5), 2004.
- Bijmolt, T. H. A., van Heerde, H. J., and Pieters, R. G. M. "New Empirical Generalizations on the Determinants of Price Elasticity." *Journal of Marketing Research*, 42(2), 2005.
- Bass, F. M. "A New Product Growth for Model Consumer Durables." *Management Science*, 15(5), 1969.
- Fildes, R., and Goodwin, P. "Against Your Better Judgment? How Organizations Can Improve Their Use of Management Judgment in Forecasting." *Interfaces*, 37(6), 2007.

**Industry and analyst**

- Gartner, "Gartner Predicts 70% of Large Organizations Will Adopt AI-Based Supply Chain Forecasting to Predict Future Demand by 2030," press release, 16 September 2025. Analyst prediction. https://www.gartner.com/en/newsroom/press-releases/2025-09-16-gartner-predicts-70-percent-of-large-orgs-will-adopt-ai-based-supply-chain-forecasting-to-predict-future-demand-by-2030 — accessed 27 October 2026.
- DP World, *Global Trade Observatory 2026*, 3,500+ executives, 2026. Reported: 94% expect 2026 trade growth to match or exceed 2025. Survey evidence, not a forecast. [VERIFY: final published sample and wording.]

**Datasets and tools**

- Corporación Favorita Grocery Sales Forecasting, Kaggle, with promotion flags and store metadata. Accessed 27 October 2026.
- M5 (Walmart) competition data, including price and promotion calendar features. https://github.com/Mcompetitions/M5-methods — accessed 27 October 2026.
- Dunnhumby, *Complete Journey* household panel; Instacart Market Basket dataset. Accessed 27 October 2026.

> **Note:** A lift number without its counterfactual is an opinion. A ROI without baseline erosion is a marketing claim.

---

> **Next:** [Module 10](10_inventory_optimization_multi_echelon.md) — turning a distribution into a policy: newsvendor fractiles, safety stock, multi-echelon buffers, and the working capital they consume.
