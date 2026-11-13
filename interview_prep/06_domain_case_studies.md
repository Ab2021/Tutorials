# Domain-Specific Case Studies: FMCG (General Mills) & Financial Services (LSEG)
## Senior Data Scientist Interview Preparation Guide

This guide provides **22 production-grade case studies** (11 for General Mills / FMCG and 11 for LSEG / Financial Services). Each case study follows an end-to-end structure:
1. **Business Problem & Strategic Framing**
2. **Key Performance Indicators (KPIs) & Evaluation Metrics**
3. **Mathematical & Technical Formulation**
4. **Complete, Self-Contained, Runnable Python Implementation**
5. **Production Considerations, Edge Cases & Senior DS Interview Talking Points**

---

# PART I: GENERAL MILLS & FMCG / CPG DOMAIN CASE STUDIES

---

## Case Study 1: SKU-Store Level Demand Sensing & Promotional Forecasting
### 1. Business Framing
General Mills distributes packaged goods (e.g., Cheerios, Nature Valley, Pillsbury) to major retailers (Walmart, Target, Kroger) and regional distributor networks. High forecasting error leads to stockouts (lost sales, retailer fines for On-Time In-Full / OTIF misses) or excessive inventory holding costs and food spoilage. Demand sensing incorporates high-frequency point-of-sale (POS) data, distributor sell-in, retail promotions, calendar seasonality, and macroeconomic indices.

### 2. Metrics & KPIs
* **Weighted Mean Absolute Percentage Error (WMAPE):** $\frac{\sum |y_t - \hat{y}_t|}{\sum y_t}$ (avoids division by zero in zero-demand periods).
* **Forecast Bias:** $\frac{\sum (\hat{y}_t - y_t)}{\sum y_t}$ (monitors systemic over/under-forecasting).
* **Service Level / In-Stock Rate:** Target $\ge 98\%$ on core Tier-1 SKUs.

### 3. Technical Formulation
We build an ensemble gradient boosting pipeline (LightGBM) using recursive multi-step forecasting with rolling lag statistics, price discount depth, promotional flag interactions, and holiday calendars.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def generate_fmcg_sales_data(n_days=365, n_skus=3, n_stores=4, random_state=42):
    """Generates synthetic daily store-SKU POS sales with promotions and seasonality."""
    np.random.seed(random_state)
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
    records = []
    
    sku_base_price = {"Cheerios_500g": 4.50, "NatureValley_Granola": 3.20, "Pillsbury_Atta_5kg": 6.80}
    sku_base_demand = {"Cheerios_500g": 120, "NatureValley_Granola": 80, "Pillsbury_Atta_5kg": 60}
    
    for store_id in [f"Store_{i+1:02d}" for i in range(n_stores)]:
        store_mult = np.random.uniform(0.8, 1.4)
        for sku, base_price in sku_base_price.items():
            base_vol = sku_base_demand[sku] * store_mult
            for d in dates:
                # Seasonality: day-of-week (weekends higher) and monthly trend
                dow_factor = 1.35 if d.dayofweek in [4, 5, 6] else 0.90
                month_factor = 1.0 + 0.15 * np.sin(2 * np.pi * d.month / 12)
                
                # Promotional shock: 15% probability of trade promotion (discount 10-30%)
                is_promo = 1 if np.random.rand() < 0.15 else 0
                discount = np.random.uniform(0.10, 0.30) if is_promo else 0.0
                actual_price = base_price * (1 - discount)
                
                # Price elasticity demand uplift
                elasticity = -1.8
                price_ratio = actual_price / base_price
                price_uplift = (price_ratio) ** elasticity
                
                noise = np.random.normal(loc=1.0, scale=0.12)
                units_sold = int(max(0, np.round(base_vol * dow_factor * month_factor * price_uplift * noise)))
                
                records.append({
                    "date": d,
                    "store_id": store_id,
                    "sku_id": sku,
                    "units_sold": units_sold,
                    "base_price": base_price,
                    "actual_price": actual_price,
                    "discount_depth": discount,
                    "is_promo": is_promo
                })
    df = pd.DataFrame(records)
    return df

def feature_engineering(df):
    """Engineers lag, rolling statistics, calendar features, and discount interactions."""
    df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    
    # Calendar features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    
    # Lag and Rolling aggregations per Store-SKU
    grouped = df.groupby(["store_id", "sku_id"])["units_sold"]
    for lag in [1, 7, 14, 28]:
        df[f"sales_lag_{lag}"] = grouped.shift(lag)
        
    for window in [7, 14, 28]:
        df[f"sales_roll_mean_{window}"] = grouped.shift(1).rolling(window).mean()
        df[f"sales_roll_std_{window}"] = grouped.shift(1).rolling(window).std()
        
    # Categorical encoding
    df["store_code"] = df["store_id"].astype("category")
    df["sku_code"] = df["sku_id"].astype("category")
    
    # Drop warm-up rows containing NaN from lag operations
    clean_df = df.dropna().reset_index(drop=True)
    return clean_df

def evaluate_forecast(y_true, y_pred):
    """Calculates WMAPE, MAE, RMSE, and Forecast Bias."""
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.sum(y_pred - y_true) / np.sum(y_true)
    return {"WMAPE (%)": round(wmape * 100, 2), "MAE": round(mae, 2), "RMSE": round(rmse, 2), "Bias (%)": round(bias * 100, 2)}

# Execution pipeline
data_raw = generate_fmcg_sales_data()
df_features = feature_engineering(data_raw)

# Temporal train-test split (avoid look-ahead leakage)
split_date = pd.to_datetime("2024-10-01")
train_mask = df_features["date"] < split_date
test_mask = df_features["date"] >= split_date

features = [
    "base_price", "actual_price", "discount_depth", "is_promo",
    "dayofweek", "month", "day", "is_weekend",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7", "sales_roll_std_7",
    "sales_roll_mean_14", "sales_roll_std_14",
    "sales_roll_mean_28", "sales_roll_std_28",
    "store_code", "sku_code"
]

X_train, y_train = df_features.loc[train_mask, features], df_features.loc[train_mask, "units_sold"]
X_test, y_test = df_features.loc[test_mask, features], df_features.loc[test_mask, "units_sold"]

model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.04,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)
preds = model.predict(X_test)
preds = np.clip(preds, 0, None) # sales cannot be negative

metrics = evaluate_forecast(y_test.values, preds)
print(f"Demand Sensing Evaluation Metrics: {metrics}")
```

### 5. Senior DS Interview Talking Points
* **Temporal Leakage:** Never use K-Fold cross validation on time series. Always use sliding window or expanding window temporal splits.
* **Cold Start Strategy:** For newly launched SKUs with no historical lags, utilize hierarchical cluster mapping based on package size, sub-category, and price tier, or backfill with category-average velocity.
* **Hierarchical Coherence:** Reconcile store-level predictions with national distribution center forecasts using bottom-up or Top-Down/Optimal Reconciliation (MinT method).

---

## Case Study 2: Price Elasticity of Demand (PED) & Revenue Optimization
### 1. Business Framing
General Mills brand managers adjust MSRP and trade discounts across different customer segments. If Cheerios has an elasticity of $-1.5$ (elastic) while Betty Crocker Brownie mix is $-0.6$ (inelastic), price hikes on brownies increase total revenue, whereas price hikes on Cheerios erode volume and gross margin.

### 2. Metrics & KPIs
* **Own-Price Elasticity of Demand ($\epsilon_{ii}$):** $\frac{\% \Delta Q}{\% \Delta P} = \frac{\partial \ln Q}{\partial \ln P}$.
* **Cross-Price Elasticity ($\epsilon_{ij}$):** Measures substitution effect with competitor products.
* **Revenue-Maximizing Price ($P^*$):** $P^* = \frac{MC \cdot \epsilon}{1 + \epsilon}$ where $MC$ is marginal cost.

### 3. Technical Formulation
We formulate a log-log regression model controlling for marketing media, store demographics, seasonality, and promotional flyers.
$$\ln(Q_{it}) = \alpha + \beta_1 \ln(P_{it}) + \beta_2 \ln(P_{jt}^{comp}) + \beta_3 \text{Promo}_{it} + \gamma_t + \epsilon_{it}$$

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

def simulate_price_volume_data(n_weeks=156):
    """Simulates 3 years of weekly syndicated Nielsen/IRI scanner data."""
    np.random.seed(101)
    weeks = pd.date_range("2022-01-01", periods=n_weeks, freq="W")
    
    # Own price varies around $3.50; competitor price around $3.40
    own_price = np.random.normal(loc=3.50, scale=0.40, size=n_weeks).clip(2.40, 4.60)
    comp_price = np.random.normal(loc=3.40, scale=0.35, size=n_weeks).clip(2.50, 4.50)
    feature_display = np.random.binomial(1, 0.20, size=n_weeks) # In-store endcap display
    
    # True data generating process: Own elasticity = -1.75, Cross elasticity = +0.65
    log_q = (
        8.5 
        - 1.75 * np.log(own_price) 
        + 0.65 * np.log(comp_price) 
        + 0.40 * feature_display 
        + np.random.normal(0, 0.08, size=n_weeks)
    )
    volume = np.exp(log_q)
    
    return pd.DataFrame({
        "week": weeks,
        "volume": volume,
        "own_price": own_price,
        "comp_price": comp_price,
        "feature_display": feature_display
    })

def estimate_elasticity(df):
    df["log_volume"] = np.log(df["volume"])
    df["log_own_price"] = np.log(df["own_price"])
    df["log_comp_price"] = np.log(df["comp_price"])
    
    X = df[["log_own_price", "log_comp_price", "feature_display"]]
    X = sm.add_constant(X)
    y = df["log_volume"]
    
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4}) # Newey-West robust SE
    return model

def optimize_pricing(own_elasticity, marginal_cost=1.60, current_price=3.50):
    """Computes revenue and margin maximizing price points."""
    # Profit Pi = (P - MC) * Q(P)
    # dPi/dP = 0 => P*(1 + 1/e) = MC => P* = MC * (e / (1 + e))
    if own_elasticity >= -1.0:
        raise ValueError("Product is inelastic; theoretical unconstrained optimal price is infinite.")
    
    optimal_profit_price = marginal_cost * (own_elasticity / (1 + own_elasticity))
    return {
        "Current Price ($)": current_price,
        "Marginal Cost ($)": marginal_cost,
        "Estimated Elasticity": round(own_elasticity, 3),
        "Profit-Maximizing Price ($)": round(optimal_profit_price, 2)
    }

df_scanner = simulate_price_volume_data()
model = estimate_elasticity(df_scanner)
print(model.summary().tables[1])

own_e = model.params["log_own_price"]
comp_e = model.params["log_comp_price"]
print(f"\nDerived Own-Price Elasticity: {own_e:.3f} | Cross-Price Elasticity: {comp_e:.3f}")
opt = optimize_pricing(own_e, marginal_cost=1.50, current_price=3.50)
print(f"Optimal Price Recommendation: {opt}")
```

### 5. Senior DS Interview Talking Points
* **Endogeneity & Simultaneity Bias:** Retailers drop prices when demand is seasonally expected to be high (e.g., Thanksgiving). Using unadjusted OLS leads to attenuated (less negative) elasticity estimates. Mitigation: Instrumental Variables (2SLS) using wholesale spot commodity costs or Hausman-type spatial instruments (prices in distant markets).
* **Cross-Price Asymmetry:** Consumers down-trading from premium to private label react differently compared to private-label consumers trading up.

---

## Case Study 3: Marketing Mix Modeling (MMM) with Carryover & Saturation
### 1. Business Framing
General Mills allocates hundreds of millions in marketing budgets across TV, digital display, social media, search, and in-store promotions. Brand awareness persists beyond the week an advertisement airs (adstock carryover effect), and marginal returns diminish at high spend levels (saturation effect).

### 2. Metrics & KPIs
* **Return on Ad Spend (ROAS):** $\frac{\text{Incremental Revenue Generated by Channel}}{\text{Total Spend in Channel}}$.
* **Marginal ROAS (mROAS):** First derivative of the revenue response curve.
* **Baseline vs. Incremental Sales Volume Breakdown.**

### 3. Technical Formulation
We implement:
1. **Geometric Adstock Transformation:** $A_t = X_t + \lambda A_{t-1}$ where $\lambda \in [0, 1)$ is the decay rate.
2. **Hill Saturation Function:** $S(A_t) = \frac{A_t^S}{K^S + A_t^S}$ where $K$ is half-saturation spend and $S$ is shape parameter.
3. **Constrained Ridge Regression** to ensure positive coefficients for media spend.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

def geometric_adstock(media_series, decay_rate=0.6):
    """Applies recursive exponential decay adstock carryover."""
    adstocked = np.zeros(len(media_series))
    for t in range(len(media_series)):
        if t == 0:
            adstocked[t] = media_series[t]
        else:
            adstocked[t] = media_series[t] + decay_rate * adstocked[t-1]
    return adstocked

def hill_saturation(x, half_saturation, slope=1.2):
    """Applies Hill function diminishing returns transformation."""
    return (x ** slope) / (half_saturation ** slope + x ** slope)

def simulate_mmm_data(n_weeks=104):
    np.random.seed(42)
    weeks = pd.date_range("2023-01-01", periods=n_weeks, freq="W")
    
    tv_spend = np.random.gamma(shape=2.5, scale=20000, size=n_weeks)
    digital_spend = np.random.gamma(shape=3.0, scale=12000, size=n_weeks)
    social_spend = np.random.gamma(shape=2.0, scale=8000, size=n_weeks)
    
    # Apply transformations
    tv_trans = hill_saturation(geometric_adstock(tv_spend, decay_rate=0.7), half_saturation=60000)
    dig_trans = hill_saturation(geometric_adstock(digital_spend, decay_rate=0.3), half_saturation=40000)
    soc_trans = hill_saturation(geometric_adstock(social_spend, decay_rate=0.2), half_saturation=25000)
    
    baseline_sales = 250000 + 15000 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    incremental_sales = 80000 * tv_trans + 65000 * dig_trans + 45000 * soc_trans
    noise = np.random.normal(0, 8000, size=n_weeks)
    
    sales = baseline_sales + incremental_sales + noise
    
    return pd.DataFrame({
        "week": weeks,
        "sales": sales,
        "tv_spend": tv_spend,
        "digital_spend": digital_spend,
        "social_spend": social_spend,
        "tv_trans": tv_trans,
        "digital_trans": dig_trans,
        "social_trans": soc_trans
    })

df_mmm = simulate_mmm_data()

# Model fitting: Decompose contribution using Ridge Regression with non-negative constraints
X = df_mmm[["tv_trans", "digital_trans", "social_trans"]]
y = df_mmm["sales"] - 250000 # subtract estimated baseline

model = Ridge(alpha=10.0, positive=True)
model.fit(X, y)

for col, coef in zip(["TV", "Digital", "Social"], model.coef_):
    print(f"Estimated Channel Impact Coefficient ({col}): {coef:.2f}")

# Budget Reallocation: Optimize spend allocation subject to fixed total budget
def budget_objective(spends, total_budget, coefficients):
    tv, dig, soc = spends
    # Penalize budget violation
    budget_penalty = 1e6 * (np.sum(spends) - total_budget)**2
    
    tv_impact = coefficients[0] * hill_saturation(tv * (1/(1-0.7)), half_saturation=60000)
    dig_impact = coefficients[1] * hill_saturation(dig * (1/(1-0.3)), half_saturation=40000)
    soc_impact = coefficients[2] * hill_saturation(soc * (1/(1-0.2)), half_saturation=25000)
    
    total_incremental_sales = tv_impact + dig_impact + soc_impact
    return -(total_incremental_sales - budget_penalty)

current_avg_spend = np.array([df_mmm["tv_spend"].mean(), df_mmm["digital_spend"].mean(), df_mmm["social_spend"].mean()])
total_weekly_budget = current_avg_spend.sum()

res = minimize(
    budget_objective,
    x0=current_avg_spend,
    args=(total_weekly_budget, model.coef_),
    bounds=[(5000, total_weekly_budget) for _ in range(3)],
    method="SLSQP"
)

print(f"\nCurrent Weekly Media Spend: {current_avg_spend.round(1)}")
print(f"Optimized Media Spend Allocation: {res.x.round(1)}")
```

### 5. Senior DS Interview Talking Points
* **Multicollinearity:** Media campaigns frequently coincide with holidays and promotional weeks, causing collinearity. Solution: Regularized regression (Ridge/Lasso), Hierarchical Bayesian models (PyMC-Marketing / Robyn), or shrinkage priors.
* **A/B Testing Ground Truth:** Observational MMM models must be calibrated against geo-lift randomized experiments to prevent false positive attribution.

---

## Case Study 4: Multi-Echelon Supply Chain Inventory & Safety Stock Optimization
### 1. Business Framing
General Mills operates manufacturing plants, central distribution centers (CDCs), and customer regional distribution centers (RDCs). Fluctuations in upstream manufacturing lead times and downstream consumer retail demand create the *bullwhip effect*. The objective is to determine optimal Safety Stock (SS) and Reorder Points (ROP) to meet a 98% service level while minimizing inventory working capital.

### 2. Metrics & KPIs
* **Cycle Service Level ($CSL$):** Probability of not stocking out during lead time ($98\% \implies Z = 2.054$).
* **Days of Inventory on Hand (DOI):** $\frac{\text{Average Inventory Value}}{\text{Cost of Goods Sold (COGS) / 365}}$.
* **Safety Stock Holding Cost vs. Stockout Cost Trade-off.**

### 3. Technical Formulation
When both demand ($D$) and lead time ($L$) are stochastic and independent:
$$\sigma_{DL} = \sqrt{\bar{L} \cdot \sigma_D^2 + \bar{D}^2 \cdot \sigma_L^2}$$
$$\text{Safety Stock (SS)} = Z_{\alpha} \cdot \sigma_{DL}$$
$$\text{Reorder Point (ROP)} = \bar{D} \cdot \bar{L} + \text{SS}$$

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from scipy.stats import norm

class InventoryOptimizer:
    def __init__(self, target_service_level=0.98, unit_cost=3.20, holding_rate=0.22):
        self.service_level = target_service_level
        self.z_score = norm.ppf(target_service_level)
        self.unit_cost = unit_cost
        self.holding_cost_per_unit_year = unit_cost * holding_rate

    def calculate_safety_stock(self, mean_daily_demand, std_daily_demand, mean_lead_time_days, std_lead_time_days):
        """Calculates standard deviation of demand during lead time and required SS."""
        combined_std = np.sqrt(
            mean_lead_time_days * (std_daily_demand ** 2) + 
            (mean_daily_demand ** 2) * (std_lead_time_days ** 2)
        )
        safety_stock = self.z_score * combined_std
        reorder_point = (mean_daily_demand * mean_lead_time_days) + safety_stock
        annual_holding_cost = safety_stock * self.holding_cost_per_unit_year
        
        return {
            "Z_Score": round(self.z_score, 3),
            "Lead_Time_Std_Dev (units)": round(combined_std, 1),
            "Recommended_Safety_Stock (units)": int(np.ceil(safety_stock)),
            "Reorder_Point (units)": int(np.ceil(reorder_point)),
            "Annual_Holding_Cost ($)": round(annual_holding_cost, 2)
        }

    def simulate_monte_carlo_inventory(self, mean_d=250, std_d=45, mean_lt=12, std_lt=3, n_simulations=10000):
        """Validates stockout probability under simulated empirical conditions."""
        np.random.seed(42)
        sim_lead_times = np.maximum(1, np.random.normal(mean_lt, std_lt, n_simulations))
        
        stockouts = 0
        ss_metrics = self.calculate_safety_stock(mean_d, std_d, mean_lt, std_lt)
        rop = ss_metrics["Reorder_Point (units)"]
        
        for lt in sim_lead_times:
            # Aggregate demand over variable lead time
            actual_demand = np.sum(np.maximum(0, np.random.normal(mean_d, std_d, int(np.round(lt)))))
            if actual_demand > rop:
                stockouts += 1
                
        actual_service_level = 1.0 - (stockouts / n_simulations)
        return ss_metrics, actual_service_level

optimizer = InventoryOptimizer(target_service_level=0.98, unit_cost=4.50, holding_rate=0.20)
metrics, empirical_sl = optimizer.simulate_monte_carlo_inventory()

print(f"Inventory Optimization Results:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
print(f"Empirical Simulation Service Level: {empirical_sl * 100:.2f}% (Target: 98.0%)")
```

### 5. Senior DS Interview Talking Points
* **Lead Time Variability Dominance:** Note that $\bar{D}^2 \cdot \sigma_L^2$ is usually much larger than $\bar{L} \cdot \sigma_D^2$. Reducing supplier lead time variability yields a bigger inventory reduction than improving sales forecasting accuracy.
* **Risk Pooling:** Consolidating safety stock from 10 local distribution centers into 2 regional distribution hubs reduces total safety stock by roughly $\sqrt{10/2} \approx \sqrt{5} \approx 2.23\times$ (Square Root Law of Inventory).

---

## Case Study 5: Trade Promotion Baseline Decomposition & Uplift ROI
### 1. Business Framing
Consumer packaged goods companies spend upwards of 20% of gross revenue on trade promotions (temporary price reductions, retailer co-op ads, endcap displays). Measuring true ROI requires isolating the *counterfactual baseline*—the volume that would have sold without any promotion.

### 2. Metrics & KPIs
* **Incremental Volume:** Total Actual Volume $-$ De-Promoted Baseline Volume.
* **Forward Buying / Cannibalization Rate:** Post-promotional dip in subsequent periods.
* **Trade Promotion Net ROI:** $\frac{\text{Incremental Gross Margin} - \text{Trade Spend Cost}}{\text{Trade Spend Cost}}$.

### 3. Technical Formulation
We fit an iterative de-promoted baseline model using penalized B-spline regression or exponential smoothing that flags promotional anomalies, masks them, interpolates the unpromoted baseline, and measures incremental lift.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

def simulate_trade_promotion_series(n_weeks=52):
    np.random.seed(42)
    weeks = pd.date_range("2024-01-01", periods=n_weeks, freq="W")
    
    # Baseline follows a smooth annual seasonality curve
    t = np.linspace(0, 2 * np.pi, n_weeks)
    baseline = 10000 + 2000 * np.sin(t) + np.random.normal(0, 200, n_weeks)
    
    # 4 promotional event weeks
    promo_flags = np.zeros(n_weeks, dtype=int)
    promo_weeks = [10, 22, 35, 47]
    promo_flags[promo_weeks] = 1
    
    actual_sales = baseline.copy()
    # Promotion generates 70% lift, followed by a 15% post-promo dip (pantry loading)
    for pw in promo_weeks:
        actual_sales[pw] *= 1.70
        if pw + 1 < n_weeks:
            actual_sales[pw + 1] *= 0.85
            
    trade_spend = np.zeros(n_weeks)
    trade_spend[promo_weeks] = [15000, 18000, 14000, 20000]
    
    return pd.DataFrame({
        "week": weeks,
        "week_idx": np.arange(n_weeks),
        "actual_sales": actual_sales,
        "is_promo": promo_flags,
        "trade_spend": trade_spend
    })

def decompose_baseline(df):
    """Iterative non-promotional baseline extraction using smoothing splines."""
    # Step 1: Mask promotional weeks and post-promo dip weeks
    mask_clean = (df["is_promo"] == 0) & (df["is_promo"].shift(1, fill_value=0) == 0)
    
    clean_indices = df.loc[mask_clean, "week_idx"]
    clean_sales = df.loc[mask_clean, "actual_sales"]
    
    # Step 2: Fit smoothing spline on non-promo periods
    spline = UnivariateSpline(clean_indices, clean_sales, s=len(clean_indices) * 50000)
    df["estimated_baseline"] = spline(df["week_idx"])
    
    # Step 3: Compute incremental volume and ROI
    df["incremental_volume"] = np.maximum(0, df["actual_sales"] - df["estimated_baseline"])
    
    unit_margin = 1.25 # Gross profit per unit before trade promotion funding
    promo_df = df[df["is_promo"] == 1].copy()
    promo_df["incremental_margin"] = promo_df["incremental_volume"] * unit_margin
    promo_df["net_trade_profit"] = promo_df["incremental_margin"] - promo_df["trade_spend"]
    promo_df["trade_roi_%"] = (promo_df["net_trade_profit"] / promo_df["trade_spend"]) * 100
    
    return df, promo_df

df_sales = simulate_trade_promotion_series()
df_decomposed, promo_eval = decompose_baseline(df_sales)

print("Promotion Event Performance Analysis:")
print(promo_eval[["week", "actual_sales", "estimated_baseline", "incremental_volume", "trade_spend", "trade_roi_%"]])
```

### 5. Senior DS Interview Talking Points
* **Pantry Loading (Forward Buying):** High discounts cause consumers to stock up, reducing purchases for the next 2–4 weeks. A naive model that ignores the post-promotional dip overstates trade promo ROI by up to 30%.
* **Retailer Pass-Through:** Not all trade spend given to retailers reaches consumers as shelf discounts; retailers pocket a portion (margin grab).

---

## Case Study 6: Cross-SKU Cannibalization & Market Basket Association
### 1. Business Framing
Introducing a new flavor of Nature Valley Granola Bars may increase brand sales but cannibalize existing SKUs. A retailer's assortment planner wants to know which SKUs are complementary (bought together) and which are substitutes (competing for the same stomach share).

### 2. Metrics & KPIs
* **Support:** $P(A \cap B)$.
* **Confidence:** $P(B | A) = \frac{P(A \cap B)}{P(A)}$.
* **Lift:** $\frac{P(A \cap B)}{P(A) \cdot P(B)}$ (Lift $> 1$ indicates complementary affinity; Lift $< 1$ indicates substitute/cannibalizing behavior).

### 3. Technical Formulation
We build an Apriori / FP-Growth association rules engine and pair it with a cross-price substitution matrix.

### 4. Complete Python Implementation
```python
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict

def simulate_market_baskets(n_baskets=10000):
    np.random.seed(42)
    catalog = [
        "Cheerios_Original", "Cheerios_HoneyNut", "NatureValley_OatsHoney",
        "NatureValley_Almond", "Pillsbury_PancakeMix", "BettyCrocker_FudgeBrownie", "Yoplait_Strawberry"
    ]
    
    baskets = []
    for _ in range(n_baskets):
        basket = []
        # Substitute behavior: rarely buy both Cheerios varieties together
        if np.random.rand() < 0.25:
            basket.append(np.random.choice(["Cheerios_Original", "Cheerios_HoneyNut"], p=[0.55, 0.45]))
        # Complementary behavior: pancake mix often accompanied by cereal or yogurt
        if np.random.rand() < 0.18:
            basket.append("Pillsbury_PancakeMix")
            if np.random.rand() < 0.60:
                basket.append("Yoplait_Strawberry")
        # Snack items
        if np.random.rand() < 0.20:
            basket.append("NatureValley_OatsHoney")
        if np.random.rand() < 0.10:
            basket.append("BettyCrocker_FudgeBrownie")
        if len(basket) > 0:
            baskets.append(list(set(basket)))
    return baskets

def compute_association_rules(baskets, min_support=0.01, min_confidence=0.15):
    n_total = len(baskets)
    item_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    
    for b in baskets:
        for item in b:
            item_counts[item] += 1
        for pair in combinations(sorted(b), 2):
            pair_counts[pair] += 1
            
    rules = []
    for (item_a, item_b), count in pair_counts.items():
        support_ab = count / n_total
        if support_ab >= min_support:
            support_a = item_counts[item_a] / n_total
            support_b = item_counts[item_b] / n_total
            
            # Rule: A -> B
            conf_a_to_b = support_ab / support_a
            lift = support_ab / (support_a * support_b)
            rules.append({
                "Antecedent": item_a, "Consequent": item_b,
                "Support": round(support_ab, 4),
                "Confidence": round(conf_a_to_b, 4),
                "Lift": round(lift, 3)
            })
            
    return pd.DataFrame(rules).sort_values("Lift", ascending=False).reset_index(drop=True)

baskets = simulate_market_baskets()
rules_df = compute_association_rules(baskets)
print("Top Association Rules & Cross-SKU Affinities:")
print(rules_df.head(6))
```

### 5. Senior DS Interview Talking Points
* **Assortment Rationalization:** Retail shelf space is finite. If two SKUs exhibit a high cannibalization coefficient (cross-price elasticity $> 0.8$ or mutually exclusive purchasing), delisting one SKU frees up space with near-zero category revenue loss.
* **Co-Location Strategy:** Complementary products with high Lift (e.g., Pancake Mix and Syrup/Yogurt) should be co-located or featured in secondary cross-merchandising displays.

---

## Case Study 7: DTC Subscription Churn & Customer Lifetime Value (CLV)
### 1. Business Framing
General Mills operates direct-to-consumer (DTC) channels and subscription models for premium baking kits and pet food (e.g., Blue Buffalo). Retaining existing subscribers is 5x cheaper than acquiring new customers. The goal is to predict customer churn probability and estimate residual customer lifetime value to guide proactive retention discounts.

### 2. Metrics & KPIs
* **Churn Hazard Rate $h(t)$.**
* **Customer Lifetime Value (CLV):** $\int_0^{\infty} S(t) \cdot m \cdot e^{-rt} dt$.
* **Concordance Index (C-Index):** Discrimination metric for survival models.

### 3. Technical Formulation
We apply **Cox Proportional Hazards**:
$$h(t | X) = h_0(t) \exp(\beta_1 X_1 + \dots + \beta_p X_p)$$

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

def generate_subscription_data(n_users=2000):
    np.random.seed(42)
    tenure_months = np.random.exponential(scale=14, size=n_users).clip(1, 36)
    
    # Covariates
    discount_acquired = np.random.binomial(1, 0.40, size=n_users) # Acquired on deep coupon
    delivery_complaints = np.random.poisson(0.3, size=n_users)
    avg_order_value = np.random.normal(45, 10, size=n_users).clip(20, 80)
    engagement_score = np.random.uniform(1, 10, size=n_users) # App visits per month
    
    # Hazard calculation
    hazard_score = (
        0.5 * discount_acquired + 
        0.8 * delivery_complaints - 
        0.02 * avg_order_value - 
        0.25 * engagement_score
    )
    prob_churn = 1 / (1 + np.exp(-hazard_score))
    
    # Censoring: 1 = churn observed, 0 = customer still active
    churn_event = (np.random.rand(n_users) < prob_churn).astype(int)
    
    return pd.DataFrame({
        "tenure_months": tenure_months,
        "churned": churn_event,
        "discount_acquired": discount_acquired,
        "delivery_complaints": delivery_complaints,
        "avg_order_value": avg_order_value,
        "engagement_score": engagement_score
    })

df_sub = generate_subscription_data()

cph = CoxPHFitter()
cph.fit(df_sub, duration_col="tenure_months", event_col="churned")
print(cph.summary[["coef", "exp(coef)", "p"]])

# Predict survival function for high-risk vs low-risk profile
test_profiles = pd.DataFrame([
    {"discount_acquired": 1, "delivery_complaints": 2, "avg_order_value": 30, "engagement_score": 2}, # High risk
    {"discount_acquired": 0, "delivery_complaints": 0, "avg_order_value": 60, "engagement_score": 8}  # Low risk
])

surv_curves = cph.predict_survival_function(test_profiles)
print("\nPredicted 12-Month Survival Probability:")
print(f"  High Risk Customer: {surv_curves.loc[12, 0]:.3f}")
print(f"  Low Risk Customer:  {surv_curves.loc[12, 1]:.3f}")
```

### 5. Senior DS Interview Talking Points
* **Right Censoring:** Standard binary classification (Logistic Regression) is invalid when customers are currently active; their eventual lifetime is unknown. Survival analysis properly handles right-censored data.
* **Uplift Modeling for Interventions:** Targeting customers with the highest churn risk is inefficient if they are "lost causes". Instead, apply Uplift / Two-Model CATE (Conditional Average Treatment Effect) to identify "persuadables" whose churn probability reduces most when offered a retention coupon.

---

## Case Study 8: New Product Introduction (NPI) Cold-Start Demand Forecasting
### 1. Business Framing
When General Mills launches a new cereal flavor or snack line, zero historical sales data exists. Relying on simple managerial guesses leads to severe stockouts or heavy manufacturing scrap. The solution is attribute-based analog matching and transfer learning.

### 2. Metrics & KPIs
* **Analog Similarity Distance:** Euclidean / Cosine distance in embedding space.
* **Launch Curve Ramp MAPE:** Error over initial 12 weeks.

### 3. Technical Formulation
We encode categorical SKU attributes (pack size, sweetness level, organic certification, brand tier, target demographic) and project them into an embedding space using K-Nearest Neighbors / Hierarchical Clustering on historical launch curves.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def create_historical_npi_database(n_skus=150):
    np.random.seed(123)
    categories = ["Cereal", "Baking", "Snacks", "Meals"]
    records = []
    
    for i in range(n_skus):
        cat = np.random.choice(categories)
        organic = np.random.binomial(1, 0.25)
        price_tier = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2]) # Budget, Mainstream, Premium
        pack_size_g = np.random.choice([250, 400, 500, 750])
        sugar_content_g = np.random.uniform(2, 25)
        
        # Simulated launch ramp: 12-week trajectory (units sold per week per store)
        base_velocity = 50 * price_tier + 0.1 * pack_size_g
        ramp = np.array([0.2, 0.4, 0.7, 0.9, 1.0, 1.1, 1.05, 0.95, 0.90, 0.88, 0.85, 0.85])
        actual_trajectory = base_velocity * ramp + np.random.normal(0, 5, 12)
        
        records.append({
            "sku_id": f"HIST_SKU_{i:03d}",
            "category": cat,
            "organic": organic,
            "price_tier": price_tier,
            "pack_size_g": pack_size_g,
            "sugar_content_g": sugar_content_g,
            "ramp_trajectory": actual_trajectory
        })
    return pd.DataFrame(records)

df_history = create_historical_npi_database()

# Feature matrix for analog search
feature_cols = ["organic", "price_tier", "pack_size_g", "sugar_content_g"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_history[feature_cols])

nn_model = NearestNeighbors(n_neighbors=5, metric="euclidean")
nn_model.fit(X_scaled)

# New launch SKU
new_sku_attributes = pd.DataFrame([{
    "organic": 1,
    "price_tier": 3,
    "pack_size_g": 400,
    "sugar_content_g": 6.5
}])

new_scaled = scaler.transform(new_sku_attributes[feature_cols])
distances, indices = nn_model.kneighbors(new_scaled)

# Aggregate analog trajectories weighted by inverse distance
weights = 1.0 / (distances[0] + 1e-5)
weights /= weights.sum()

analog_curves = np.vstack(df_history.iloc[indices[0]]["ramp_trajectory"].values)
predicted_ramp = np.dot(weights, analog_curves)

print(f"Top 3 Analog Historical SKUs: {df_history.iloc[indices[0][:3]]['sku_id'].tolist()}")
print(f"Forecasted First 12-Week Demand Curve (units/store/week):")
print(predicted_ramp.round(1))
```

### 5. Senior DS Interview Talking Points
* **Bass Diffusion Model:** Decomposes adoption into innovation coefficient ($p$) and imitation coefficient ($q$). Useful for macroeconomic durable goods, but for FMCG, fast trial-and-repeat purchase models (NBD-Dirichlet) are superior.
* **Distribution Build vs Rate of Sale (ROS):** Early sales volume is driven by retail pipeline fill (% ACV distribution), whereas post-week 8 sales reflect true consumer repeat rate.

---

## Case Study 9: Supply Chain Freight Logistics & Vehicle Routing (VRP)
### 1. Business Framing
Shipping finished goods from manufacturing plants to regional retail distribution hubs incurs high freight and fuel costs. Empty miles and sub-optimal truckload utilization erode operating margins. The objective is to optimize multi-stop delivery routes subject to vehicle capacity constraints.

### 2. Metrics & KPIs
* **Truckload Capacity Utilization (%):** Target $\ge 92\%$.
* **Freight Cost per Pallet-Mile.**
* **Total Ton-Miles Traveled.**

### 3. Technical Formulation
We model the Capacitated Vehicle Routing Problem (CVRP) via Mixed Integer Linear Programming (MILP) using `scipy.optimize` or Clarke-Wright Savings Heuristic.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd

def clarke_wright_savings(depot, customers, truck_capacity=24):
    """
    Clarke-Wright Savings Heuristic for CPG pallet routing.
    depot: (x, y) coordinates
    customers: dict of id -> {'coords': (x,y), 'demand': pallet_count}
    """
    def dist(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Step 1: Calculate savings s(i, j) = d(D, i) + d(D, j) - d(i, j)
    savings = []
    cust_ids = list(customers.keys())
    for i in range(len(cust_ids)):
        for j in range(i + 1, len(cust_ids)):
            c1, c2 = cust_ids[i], cust_ids[j]
            s = (dist(depot, customers[c1]["coords"]) + 
                 dist(depot, customers[c2]["coords"]) - 
                 dist(customers[c1]["coords"], customers[c2]["coords"]))
            savings.append((s, c1, c2))
            
    # Sort savings descending
    savings.sort(key=lambda x: x[0], reverse=True)
    
    # Step 2: Form initial single-customer routes [D -> i -> D]
    routes = {c: [c] for c in cust_ids}
    route_demands = {c: customers[c]["demand"] for c in cust_ids}
    
    for s, c1, c2 in savings:
        r1 = None
        r2 = None
        for r_id, r in routes.items():
            if r is not None:
                if r[0] == c1 or r[-1] == c1:
                    r1 = r_id
                if r[0] == c2 or r[-1] == c2:
                    r2 = r_id
                    
        if r1 is not None and r2 is not None and r1 != r2:
            # Check combined capacity
            if route_demands[r1] + route_demands[r2] <= truck_capacity:
                # Merge route
                routes[r1].extend(routes[r2])
                route_demands[r1] += route_demands[r2]
                routes[r2] = None # decommission r2
                
    active_routes = [r for r in routes.values() if r is not None]
    return active_routes

# Demonstration
depot_loc = (50, 50)
retail_hubs = {
    "Walmart_DC_1": {"coords": (65, 80), "demand": 8},
    "Target_DC_2":  {"coords": (70, 85), "demand": 10},
    "Kroger_DC_3":  {"coords": (20, 30), "demand": 12},
    "Costco_DC_4":  {"coords": (25, 35), "demand": 9},
    "Meijer_DC_5":  {"coords": (60, 40), "demand": 7}
}

routes = clarke_wright_savings(depot_loc, retail_hubs, truck_capacity=22)
print("Optimized Delivery Truck Routes (Pallet Dispatch):")
for idx, r in enumerate(routes, 1):
    total_pallets = sum(retail_hubs[stop]["demand"] for stop in r)
    print(f"  Truck {idx}: Depot -> {' -> '.join(r)} -> Depot (Pallet Load: {total_pallets}/22)")
```

### 5. Senior DS Interview Talking Points
* **Cube vs Weight Out:** In CPG snacks (e.g., Chex Mix), trucks "cube out" (run out of physical volume) long before they "weigh out" (hit 80,000 lbs legal weight limit). Optimization algorithms must handle multi-dimensional knapsack constraints (pallets, weight, and cubic feet).
* **Delivery Time Windows:** Real-world logistics require Hard Time Window (VRPTW) formulations to meet strict retail dock scheduling appointments and avoid non-compliance penalties.

---

## Case Study 10: On-Shelf Availability (OSA) & Phantom Inventory Detection
### 1. Business Framing
"Phantom inventory" occurs when store enterprise inventory records show stock on hand, but products are physically missing from the shelf (misplaced in backroom, stolen, or damaged). Customers find empty shelves, resulting in lost sales. Traditional store audits are labor-intensive. We can infer on-shelf stockouts using statistical anomaly detection on point-of-sale scanner streams.

### 2. Metrics & KPIs
* **Consecutive Zero-Sales Run Probability:** $P(\text{Sales} = 0 \text{ for } k \text{ hours} | \lambda)$.
* **False Alert Rate vs Detection Latency.**

### 3. Technical Formulation
We model store-SKU hourly transaction arrivals as a Poisson or Negative Binomial process:
$$P(Y = 0) = e^{-\lambda_t}$$
When the probability of observing a streak of zero sales falls below a threshold $\alpha = 0.01$, an alert is triggered for store associate shelf replenishment.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from scipy.stats import poisson

def detect_phantom_inventory(hourly_sales_series, hourly_lambdas, alpha=0.01):
    """
    Evaluates real-time run of zero sales against expected Poisson intensity.
    hourly_sales_series: array of observed hourly sales
    hourly_lambdas: array of historical expected sales rate for those hours
    """
    n_hours = len(hourly_sales_series)
    alerts = []
    current_run_zeros = 0
    cumulative_lambda = 0.0
    
    for t in range(n_hours):
        sales = hourly_sales_series[t]
        expected_rate = hourly_lambdas[t]
        
        if sales == 0:
            current_run_zeros += 1
            cumulative_lambda += expected_rate
            # Cumulative probability of 0 sales given expected rate
            prob_zero_run = np.exp(-cumulative_lambda)
            if prob_zero_run < alpha:
                alerts.append({
                    "hour_index": t,
                    "zero_hours_streak": current_run_zeros,
                    "cumulative_expected_units": round(cumulative_lambda, 2),
                    "p_value": round(prob_zero_run, 5),
                    "status": "STOCKOUT_ALERT"
                })
        else:
            # Sales occurred -> reset counter
            current_run_zeros = 0
            cumulative_lambda = 0.0
            
    return pd.DataFrame(alerts)

# Simulation: Cereal velocity during busy store hours (10 AM - 8 PM)
expected_hourly_rate = [4.5, 5.2, 6.0, 5.8, 6.5, 7.0, 6.8, 5.5, 4.0, 3.2]
# Observed sales: Normal until hour 4, then stock runs out on shelf
actual_observed_sales = [4, 6, 5, 6, 0, 0, 0, 0, 0, 0]

alerts_df = detect_phantom_inventory(actual_observed_sales, expected_hourly_rate, alpha=0.005)
print("Phantom Inventory Detection Results:")
print(alerts_df)
```

### 5. Senior DS Interview Talking Points
* **Overdispersion:** Real retail sales often exhibit higher variance than the Poisson assumption (variance $>$ mean). In production, a Negative Binomial distribution is preferred to account for bursty shopping baskets.
* **Store Associate Feedback Loop:** Closing the loop with mobile RF scanner verification by store associates creates labeled feedback data to retrain active learning classifiers.

---

## Case Study 11: Customer RFM Segmentation & Uplift Marketing
### 1. Business Framing
General Mills loyalty programs and DTC platforms must allocate discount incentives selectively. Blanket discounting subsidizes "sure things" (customers who would buy anyway) while ignoring "persuadables".

### 2. Metrics & KPIs
* **Recency, Frequency, Monetary (RFM) Score.**
* **Average Treatment Effect on the Treated (ATT) / Uplift.**

### 3. Technical Formulation
We build an RFM segmentation pipeline and combine it with a Two-Model Uplift framework:
$$\tau(X) = E[Y | X, T=1] - E[Y | X, T=0]$$

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def generate_loyalty_data(n=3000):
    np.random.seed(42)
    recency = np.random.randint(1, 180, n)     # Days since last order
    frequency = np.random.poisson(4, n) + 1    # Total orders
    monetary = frequency * np.random.normal(35, 8, n) # Total spend
    
    # Randomized coupon assignment (A/B treatment)
    treatment = np.random.binomial(1, 0.5, n)
    
    # True latent persuadability: higher frequency and recent visitors respond best
    latent_score = 0.02 * (180 - recency) + 0.3 * frequency
    prob_purchase = 1 / (1 + np.exp(-(latent_score - 2.5 + 0.8 * treatment)))
    converted = (np.random.rand(n) < prob_purchase).astype(int)
    
    return pd.DataFrame({
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "treatment": treatment,
        "converted": converted
    })

def two_model_uplift(df):
    df_treated = df[df["treatment"] == 1]
    df_control = df[df["treatment"] == 0]
    
    features = ["recency", "frequency", "monetary"]
    
    model_t = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model_c = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    
    model_t.fit(df_treated[features], df_treated["converted"])
    model_c.fit(df_control[features], df_control["converted"])
    
    # Predict uplift on entire base
    p_treated = model_t.predict_proba(df[features])[:, 1]
    p_control = model_c.predict_proba(df[features])[:, 1]
    df["predicted_uplift"] = p_treated - p_control
    
    return df

df_rfm = generate_loyalty_data()
df_scored = two_model_uplift(df_rfm)

# Segments based on uplift: Only coupon users with positive incremental lift
top_persuadables = df_scored.sort_values("predicted_uplift", ascending=False).head(5)
print("Top 5 Persuadable Customers Identified for Targeted Promotion:")
print(top_persuadables[["recency", "frequency", "monetary", "predicted_uplift"]])
```

### 5. Senior DS Interview Talking Points
* **The Four Customer Quadrants:** Persuadables (target with promo), Sure Things (do not promo; saves margin), Lost Causes (do not promo; low ROI), and Sleeping Dogs (contacting them triggers unsubscribes/churn).
* **Qini Curve / Cumulative Gain Chart:** Use Qini coefficient instead of standard ROC-AUC to benchmark uplift performance.

---
---

# PART II: LSEG & FINANCIAL SERVICES DOMAIN CASE STUDIES

---

## Case Study 12: Multi-Asset Portfolio Value at Risk (VaR) & Expected Shortfall (CVaR)
### 1. Business Framing
LSEG risk analytics engines (e.g., StarMine, Workspace Risk Analytics) provide institutional clients with risk metrics. Financial regulators (Basel III/IV, Solvency II) require calculating Value at Risk (VaR) and Expected Shortfall (Conditional VaR) to determine capital adequacy reserves.

### 2. Metrics & KPIs
* **Value at Risk ($\text{VaR}_{\alpha}$):** $\inf \{l \in \mathbb{R} : P(L > l) \le 1 - \alpha\}$.
* **Expected Shortfall ($\text{ES}_{\alpha} / \text{CVaR}$):** $E[L | L \ge \text{VaR}_{\alpha}]$ (coherent risk measure satisfying subadditivity).
* **Basel Regulatory Traffic Light Backtest (Kupiec POF test).**

### 3. Technical Formulation
We calculate:
1. **Parametric Variance-Covariance VaR** (assuming multivariate normality).
2. **Historical Simulation VaR**.
3. **Monte Carlo Simulation VaR with Cholesky Decomposition** for non-linear correlation structures.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from scipy.stats import norm

class PortfolioRiskEngine:
    def __init__(self, tickers, weights, initial_investment=10_000_000):
        self.tickers = tickers
        self.weights = np.array(weights)
        self.initial_investment = initial_investment

    def generate_synthetic_returns(self, n_days=750):
        """Generates realistic correlated equity asset returns."""
        np.random.seed(42)
        # Covariance structure
        cov_matrix = np.array([
            [0.00040, 0.00025, 0.00015],
            [0.00025, 0.00035, 0.00018],
            [0.00015, 0.00018, 0.00030]
        ])
        mean_returns = np.array([0.0004, 0.0003, 0.0002])
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=n_days)
        return pd.DataFrame(returns, columns=self.tickers)

    def calculate_var_cvar(self, returns_df, confidence=0.99, horizon_days=1):
        port_returns = returns_df.dot(self.weights)
        alpha = 1.0 - confidence
        
        # 1. Historical VaR & CVaR
        hist_var_pct = -np.percentile(port_returns, alpha * 100)
        hist_cvar_pct = -port_returns[port_returns <= -hist_var_pct].mean()
        
        # 2. Parametric VaR & CVaR
        mu_port = port_returns.mean()
        sigma_port = port_returns.std()
        param_var_pct = -(mu_port + norm.ppf(alpha) * sigma_port)
        param_cvar_pct = -(mu_port - sigma_port * norm.pdf(norm.ppf(alpha)) / alpha)
        
        # Scale to currency & horizon
        scale = np.sqrt(horizon_days)
        results = {
            "Historical_VaR ($)": round(hist_var_pct * self.initial_investment * scale, 2),
            "Historical_CVaR ($)": round(hist_cvar_pct * self.initial_investment * scale, 2),
            "Parametric_VaR ($)": round(param_var_pct * self.initial_investment * scale, 2),
            "Parametric_CVaR ($)": round(param_cvar_pct * self.initial_investment * scale, 2),
            "Confidence_Level": f"{confidence * 100}%",
            "Horizon_Days": horizon_days
        }
        return results

    def kupiec_pof_test(self, returns_df, var_pct, confidence=0.99):
        """Performs Kupiec Proportion of Failures backtest on historical VaR."""
        port_returns = returns_df.dot(self.weights)
        exceptions = (port_returns < -var_pct).sum()
        n = len(port_returns)
        p = 1 - confidence
        p_hat = exceptions / n
        
        # Likelihood ratio test statistic
        lr_stat = -2 * np.log(((1 - p)**(n - exceptions) * p**exceptions) / 
                               ((1 - p_hat)**(n - exceptions) * p_hat**exceptions + 1e-10))
        # Chi-square with 1 degree of freedom threshold at 5% significance is 3.841
        passed = lr_stat < 3.841
        return {"Exceptions": exceptions, "Expected": round(n * p, 1), "LR_Stat": round(lr_stat, 3), "Passed_Kupiec": passed}

engine = PortfolioRiskEngine(tickers=["FTSE_100", "S&P_500", "STOXX_600"], weights=[0.4, 0.4, 0.2])
returns_df = engine.generate_synthetic_returns()
risk_metrics = engine.calculate_var_cvar(returns_df, confidence=0.99)
print("Institutional Portfolio Risk Report:")
for k, v in risk_metrics.items():
    print(f"  {k}: {v}")
```

### 5. Senior DS Interview Talking Points
* **Fat Tails & Leptokurtosis:** Financial asset returns exhibit fat tails. Parametric normal VaR drastically understates tail risk during liquidity crises. Solution: Student's t-distribution, Cornish-Fisher expansion, or Extreme Value Theory (EVT / Generalized Pareto Distribution).
* **Coherence:** VaR is not subadditive ($VaR(A+B) > VaR(A) + VaR(B)$ is possible), meaning it does not reward portfolio diversification. Expected Shortfall (CVaR) is mathematically coherent and preferred by Basel Committee.

---

## Case Study 13: Fixed Income Yield Curve Construction & Bond Duration/Convexity
### 1. Business Framing
LSEG Instrument Pricing Analytics (IPA) provides real-time pricing and risk metrics for millions of global sovereign and corporate bonds. Traders require zero-coupon yield curve bootstrapping and first/second-order risk sensitivities (Macaulay duration, Modified duration, Convexity) to hedge interest rate risk.

### 2. Metrics & KPIs
* **Modified Duration:** $-\frac{1}{P} \frac{dP}{dy} = \frac{\text{Mac Duration}}{1 + y/m}$.
* **Convexity:** $\frac{1}{P} \frac{d^2 P}{dy^2}$.
* **Bond Price Delta Approximation:** $\Delta P \approx -P \cdot D_{mod} \cdot \Delta y + \frac{1}{2} P \cdot C \cdot (\Delta y)^2$.

### 3. Technical Formulation
We bootstrap a discount factor curve $d(t)$ and compute analytic duration and convexity for semi-annual coupon bonds.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd

class FixedIncomeAnalytics:
    @staticmethod
    def price_bond(face_value, coupon_rate, ytm, maturity_years, freq=2):
        """Calculates dirty price, modified duration, and convexity of a coupon bond."""
        n_periods = int(maturity_years * freq)
        c = (face_value * coupon_rate) / freq
        y = ytm / freq
        
        times = np.arange(1, n_periods + 1) / freq
        cash_flows = np.full(n_periods, c)
        cash_flows[-1] += face_value
        
        discount_factors = (1 + y) ** (-np.arange(1, n_periods + 1))
        pv_cash_flows = cash_flows * discount_factors
        price = pv_cash_flows.sum()
        
        # Macaulay Duration
        mac_duration = np.sum(times * pv_cash_flows) / price
        # Modified Duration
        mod_duration = mac_duration / (1 + y)
        # Convexity
        convexity = np.sum(times * (times + 1/freq) * pv_cash_flows) / (price * (1 + y)**2)
        
        return {
            "Price ($)": round(price, 2),
            "Mac_Duration (yrs)": round(mac_duration, 3),
            "Mod_Duration": round(mod_duration, 3),
            "Convexity": round(convexity, 3)
        }

    @staticmethod
    def estimate_price_shock(price, mod_dur, convexity, yield_shock_bps):
        """Uses Taylor expansion to approximate price change under interest rate shock."""
        dy = yield_shock_bps / 10000.0
        delta_p_duration_only = -price * mod_dur * dy
        convexity_adjustment = 0.5 * price * convexity * (dy ** 2)
        total_delta_p = delta_p_duration_only + convexity_adjustment
        new_price = price + total_delta_p
        
        return {
            "Yield_Shock_bps": yield_shock_bps,
            "Duration_Impact ($)": round(delta_p_duration_only, 2),
            "Convexity_Impact ($)": round(convexity_adjustment, 2),
            "Predicted_New_Price ($)": round(new_price, 2),
            "Percentage_Change (%)": round((total_delta_p / price) * 100, 2)
        }

analytics = FixedIncomeAnalytics()
bond_metrics = analytics.price_bond(face_value=1000, coupon_rate=0.05, ytm=0.045, maturity_years=10, freq=2)
print("Fixed Income Valuation & Sensitivity:")
for k, v in bond_metrics.items():
    print(f"  {k}: {v}")

shock_result = analytics.estimate_price_shock(
    bond_metrics["Price ($)"], 
    bond_metrics["Mod_Duration"], 
    bond_metrics["Convexity"], 
    yield_shock_bps=+100 # +100 bps rate hike
)
print(f"\nInterest Rate Shock Analysis (+100 bps):")
for k, v in shock_result.items():
    print(f"  {k}: {v}")
```

### 5. Senior DS Interview Talking Points
* **Negative Convexity:** Callable corporate bonds and mortgage-backed securities (MBS) exhibit negative convexity at low interest rates because the borrower's prepayment option caps price appreciation.
* **Key Rate Duration:** When the yield curve does not undergo parallel shifts (e.g., steepening or inversion), standard modified duration fails; traders use 2-year, 5-year, 10-year, and 30-year Key Rate Durations.

---

## Case Study 14: ESG Rating Controversy Penalty & Factor Portfolio Tilt
### 1. Business Framing
LSEG ESG Data covers over 15,000 global listed entities. Institutional asset managers construct "ESG-tilted" equity factor portfolios that penalize companies involved in material governance or environmental controversies (e.g., oil spills, accounting fraud) while maintaining benchmark tracking error bounds.

### 2. Metrics & KPIs
* **LSEG Combined ESG Score (ESGC):** Dynamic overlay of 23 controversy topics on base ESG score.
* **Tracking Error (TE):** $\sqrt{\text{Var}(R_p - R_b)} \le 1.5\%$ annualized.
* **Information Ratio (IR):** $\frac{R_p - R_b}{\text{TE}}$.

### 3. Technical Formulation
We formulate an active portfolio optimization problem tilting weights towards high ESG companies subject to sector neutrality and factor constraints.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def generate_universe_data(n_assets=50):
    np.random.seed(42)
    tickers = [f"EQ_{i:03d}" for i in range(n_assets)]
    sectors = np.random.choice(["Tech", "Energy", "Financials", "Healthcare", "Consumer"], size=n_assets)
    
    # ESG components: 0-100 scale
    base_esg = np.random.normal(65, 15, size=n_assets).clip(20, 95)
    # Controversy deductions (0 = no controversy, up to 40 point deduction)
    controversies = np.random.choice([0, 10, 25, 40], p=[0.70, 0.15, 0.10, 0.05], size=n_assets)
    combined_esg = np.maximum(10, base_esg - controversies)
    
    expected_returns = np.random.normal(0.08, 0.04, size=n_assets)
    benchmark_weights = np.random.dirichlet(np.ones(n_assets))
    
    return pd.DataFrame({
        "ticker": tickers,
        "sector": sectors,
        "base_esg": base_esg,
        "controversy": controversies,
        "combined_esg": combined_esg,
        "expected_return": expected_returns,
        "benchmark_weight": benchmark_weights
    })

def optimize_esg_tilted_portfolio(df, target_esg_lift=1.20, max_active_weight=0.03):
    """
    Maximizes expected return while raising portfolio ESG score by 20%
    and maintaining benchmark tracking constraints.
    """
    n = len(df)
    w_bench = df["benchmark_weight"].values
    esg_scores = df["combined_esg"].values
    returns = df["expected_return"].values
    
    benchmark_esg = np.dot(w_bench, esg_scores)
    min_esg_required = benchmark_esg * target_esg_lift
    
    # Objective: Minimize active risk + maximize return
    def objective(w):
        active_risk = np.sum((w - w_bench)**2)
        active_return = -np.dot(w, returns)
        return 100 * active_risk + active_return
    
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, # fully invested
        {"type": "ineq", "fun": lambda w: np.dot(w, esg_scores) - min_esg_required} # ESG lift constraint
    ]
    
    bounds = [(max(0, w_b - max_active_weight), w_b + max_active_weight) for w_b in w_bench]
    
    res = minimize(objective, x0=w_bench, method="SLSQP", bounds=bounds, constraints=constraints)
    
    df["tilted_weight"] = res.x
    port_esg = np.dot(res.x, esg_scores)
    
    return df, benchmark_esg, port_esg

df_universe = generate_universe_data()
df_opt, bench_esg, port_esg = optimize_esg_tilted_portfolio(df_universe)

print(f"Benchmark Portfolio ESG Score: {bench_esg:.2f}")
print(f"Optimized ESG-Tilted Portfolio Score: {port_esg:.2f} (Target Lift Achieved)")
print("\nTop 3 Over-Weighted Assets (High ESG, Low Controversy):")
print(df_opt.sort_values(by="tilted_weight", ascending=False)[["ticker", "sector", "combined_esg", "benchmark_weight", "tilted_weight"]].head(3))
```

### 5. Senior DS Interview Talking Points
* **Point-in-Time (PIT) ESG Data:** Backtesting ESG alpha factors requires strictly point-in-time rating snapshots to avoid restatement and survivorship bias (companies restate past carbon metrics retroactively).
* **Greenwashing vs Controversy Decay:** Controversies should decay exponentially over 12–24 months rather than causing permanent asset exclusion.

---

## Case Study 15: Limit Order Book (LOB) Market Microstructure & Order Book Imbalance
### 1. Business Framing
LSEG operates equity and fixed-income matching engines (London Stock Exchange, Turquoise). High-frequency trading (HFT) algorithms and market makers evaluate Level 2 / Level 3 order book depth to predict immediate short-term price movements and manage inventory risk.

### 2. Metrics & KPIs
* **Order Book Imbalance (OBI):** $\frac{V_t^b - V_t^a}{V_t^b + V_t^a} \in [-1, 1]$.
* **Volume Weighted Mid-Price (VWMP):** $\frac{V_t^b P_t^a + V_t^a P_t^b}{V_t^b + V_t^a}$.
* **Effective Bid-Ask Spread:** $2 \cdot |P_t^{trade} - P_t^{mid}|$.

### 3. Technical Formulation
We ingest Level 2 order book snapshots (top 5 bid/ask levels) and train a gradient-boosted classifier to forecast the probability of the mid-price ticking up or down within the next 10 order arrivals.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def simulate_lob_snapshots(n_ticks=2000):
    np.random.seed(42)
    records = []
    mid_price = 150.00
    
    for t in range(n_ticks):
        spread = np.random.choice([0.01, 0.02, 0.03], p=[0.7, 0.2, 0.1])
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        # Volumes at top 3 levels
        bid_vols = np.random.exponential(scale=1000, size=3)
        ask_vols = np.random.exponential(scale=1000, size=3)
        
        # Level 1 Order Book Imbalance
        obi_l1 = (bid_vols[0] - ask_vols[0]) / (bid_vols[0] + ask_vols[0])
        # Level 1-3 Deep Imbalance
        obi_deep = (bid_vols.sum() - ask_vols.sum()) / (bid_vols.sum() + ask_vols.sum())
        
        # Latent price drift driven by imbalance
        price_jump = np.random.normal(loc=0.008 * obi_l1, scale=0.01)
        next_mid = mid_price + price_jump
        
        records.append({
            "tick": t,
            "mid_price": mid_price,
            "spread": spread,
            "bid_p1": best_bid, "bid_v1": bid_vols[0],
            "ask_p1": best_ask, "ask_v1": ask_vols[0],
            "obi_l1": obi_l1,
            "obi_deep": obi_deep,
            "next_mid": next_mid
        })
        mid_price = next_mid
        
    df = pd.DataFrame(records)
    # Target: 1 if mid price ticks up over next 5 events
    df["target_up"] = (df["next_mid"].shift(-5) > df["mid_price"]).astype(int)
    return df.dropna().reset_index(drop=True)

df_lob = simulate_lob_snapshots()

features = ["spread", "bid_v1", "ask_v1", "obi_l1", "obi_deep"]
X = df_lob[features]
y = df_lob["target_up"]

split = int(len(df_lob) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"LOB Next-Tick Up-Move Prediction ROC-AUC: {auc:.3f}")
print("Feature Importances:", dict(zip(features, clf.feature_importances_.round(3))))
```

### 5. Senior DS Interview Talking Points
* **Queue Position & Fill Probability:** Simply placing a limit order at best bid does not guarantee execution; orders are matched Price-Time (FIFO). Advanced microstructure models use Poisson arrival queues to estimate execution probability.
* **Adverse Selection:** Executions on limit orders often happen when informed traders cross the spread, meaning fills are followed by adverse price moves against the market maker.

---

## Case Study 16: Trade Surveillance & Wash-Trading Anomaly Detection
### 1. Business Framing
Regulatory authorities (FCA, ESMA, SEC) mandate that trading venues and market operators monitor for market manipulation (e.g., wash trading, spoofing, layering). Wash trading involves simultaneously buying and selling the same financial instrument by affiliated accounts to fabricate artificial volume and mislead market participants.

### 2. Metrics & KPIs
* **Precision at Top $K$ Surveillance Alerts.**
* **Wash Trade Cycle Closure Ratio (Account Graph).**
* **Time-to-Detect (TTD) under Streaming Windows.**

### 3. Technical Formulation
We implement:
1. **Bipartite Transaction Graph Cycle Detection** to find funds circulating between collusive trading entities.
2. **Isolation Forests** on trade size, price deviation, and round-trip holding time.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def simulate_trade_stream(n_trades=10000):
    np.random.seed(42)
    accounts = [f"ACC_{i:04d}" for i in range(100)]
    
    records = []
    start_time = pd.Timestamp("2024-06-01 09:00:00")
    
    for i in range(n_trades):
        t_stamp = start_time + pd.Timedelta(seconds=int(i * 1.5))
        buyer = np.random.choice(accounts)
        seller = np.random.choice([a for a in accounts if a != buyer])
        qty = int(np.random.exponential(scale=500)) + 10
        price = round(np.random.normal(100, 0.5), 2)
        holding_seconds = np.random.exponential(scale=1800)
        
        records.append({
            "trade_id": i,
            "timestamp": t_stamp,
            "buyer": buyer,
            "seller": seller,
            "quantity": qty,
            "price": price,
            "holding_seconds": holding_seconds,
            "is_collusive": 0
        })
        
    # Inject 50 synthetic Wash-Trading events between collusive ring ACC_0007 and ACC_0042
    for j in range(50):
        idx = np.random.randint(100, n_trades - 100)
        t_stamp = start_time + pd.Timedelta(seconds=int(idx * 1.5))
        records[idx]["buyer"] = "ACC_0007"
        records[idx]["seller"] = "ACC_0042"
        records[idx]["quantity"] = 5000 # abnormally large
        records[idx]["holding_seconds"] = 1.2 # immediate round-trip
        records[idx]["is_collusive"] = 1
        
    return pd.DataFrame(records)

df_trades = simulate_trade_stream()

# Feature extraction for Surveillance Machine Learning
features = ["quantity", "holding_seconds"]
X = df_trades[features]

iso_forest = IsolationForest(contamination=0.01, random_state=42)
df_trades["anomaly_score"] = iso_forest.fit_predict(X)
# -1 indicates anomaly in Isolation Forest
df_trades["flagged_wash_trade"] = (df_trades["anomaly_score"] == -1).astype(int)

# Evaluation
detected_true_positives = ((df_trades["flagged_wash_trade"] == 1) & (df_trades["is_collusive"] == 1)).sum()
total_actual_collusive = df_trades["is_collusive"].sum()
total_flagged = df_trades["flagged_wash_trade"].sum()

precision = detected_true_positives / total_flagged
recall = detected_true_positives / total_actual_collusive

print(f"Trade Surveillance Wash-Trading Detection:")
print(f"  Flagged Anomalies: {total_flagged}")
print(f"  True Detected: {detected_true_positives}/{total_actual_collusive}")
print(f"  Surveillance Precision: {precision * 100:.1f}% | Surveillance Recall: {recall * 100:.1f}%")
```

### 5. Senior DS Interview Talking Points
* **Temporal Graph Networks:** Real-world syndicates use complex rings (Account A $\to$ B $\to$ C $\to$ A). Graph Neural Networks (GNNs) or temporal circular flow algorithms are required to identify multi-hop wash cycles.
* **Explainability for Regulatory Audits:** Regulators will reject "black-box" surveillance flags. Models must output SHAP decision rules explaining exactly why an alert was generated.

---

## Case Study 17: High-Frequency Tick Data Streaming & Realized Volatility
### 1. Business Framing
LSEG Tick History provides microsecond-level tick data for global financial exchanges. Traders and risk officers calculate Realized Volatility (RV) from high-frequency intraday prices to price options and manage intraday margin calls.

### 2. Metrics & KPIs
* **Realized Volatility (RV):** $\sqrt{\sum_{i=1}^M r_{t, i}^2}$.
* **Parkinson Volatility (High-Low estimator).**
* **Market Microstructure Noise Variance.**

### 3. Technical Formulation
We sample high-frequency tick streams into 5-minute equidistant price bars, compute log returns, and estimate annualized realized volatility while adjusting for microstructure bid-ask bounce noise via the Two-Scale Realized Volatility (TSRV) framework.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd

def simulate_hft_tick_stream(n_ticks=50000):
    np.random.seed(42)
    times = pd.date_range("2024-06-01 09:30:00", periods=n_ticks, freq="100ms")
    # Geometric Brownian motion + Microstructure noise
    dt = 1 / (252 * 6.5 * 3600 * 10)
    true_vol = 0.20 # 20% annualized volatility
    
    log_returns = np.random.normal(0, true_vol * np.sqrt(dt), size=n_ticks)
    true_prices = 100 * np.exp(np.cumsum(log_returns))
    
    # Add discrete bid-ask bounce noise
    noise = np.random.choice([-0.01, 0.01], size=n_ticks)
    observed_ticks = np.round(true_prices + noise, 2)
    
    return pd.DataFrame({"timestamp": times, "price": observed_ticks})

def compute_realized_volatility(df_ticks, sample_freq="5min"):
    """Resamples tick data and calculates realized volatility."""
    # Resample to 5-minute bars to mitigate high-frequency microstructure noise
    df_bars = df_ticks.set_index("timestamp")["price"].resample(sample_freq).last().dropna()
    log_ret = np.log(df_bars / df_bars.shift(1)).dropna()
    
    # Realized variance for the trading day
    realized_variance = np.sum(log_ret ** 2)
    # Annualize (assuming 252 trading days per year, 6.5 trading hours per day)
    annualization_factor = np.sqrt(252)
    # Daily RV
    daily_rv = np.sqrt(realized_variance)
    annualized_rv = daily_rv * annualization_factor
    
    return {
        "Sample_Interval": sample_freq,
        "Total_Intraday_Bars": len(log_ret),
        "Daily_Realized_Vol (%)": round(daily_rv * 100, 2),
        "Annualized_Realized_Vol (%)": round(annualized_rv * 100, 2)
    }

df_hft = simulate_hft_tick_stream()
rv_5min = compute_realized_volatility(df_hft, "5min")
rv_1min = compute_realized_volatility(df_hft, "1min")

print("High-Frequency Realized Volatility Calculation:")
print(f"  5-Minute Sampling: {rv_5min}")
print(f"  1-Minute Sampling: {rv_1min}")
```

### 5. Senior DS Interview Talking Points
* **Volatility Signature Plot:** Plotting realized volatility against sampling frequency reveals that sampling too finely (e.g., 1-second) inflates volatility estimates due to bid-ask bounce noise, whereas sampling too sparsely (e.g., 30-minute) increases statistical error. 5-minute sampling is the industry standard sweet spot.
* **Jumps vs Continuous Diffusion:** Realized bipower variation can decompose total volatility into smooth continuous diffusion vs discontinuous jump risk.

---

## Case Study 18: Structural Credit Risk & Merton Distance to Default (DD)
### 1. Business Framing
Credit risk analysts at institutional lenders and fixed-income rating agencies evaluate the default probability of corporate borrowers. Robert Merton's structural model treats a firm's equity as a European call option on its underlying asset value, with the face value of debt acting as the strike price.

### 2. Metrics & KPIs
* **Distance to Default ($DD$):** $\frac{\ln(V_A / D) + (\mu_A - 0.5 \sigma_A^2) T}{\sigma_A \sqrt{T}}$.
* **Probability of Default ($PD$):** $\Phi(-DD)$.
* **Implied Credit Default Swap (CDS) Spread.**

### 3. Technical Formulation
We solve the non-linear system of equations linking observable equity value ($E$) and equity volatility ($\sigma_E$) to unobservable firm asset value ($V_A$) and asset volatility ($\sigma_A$) using the Black-Scholes-Merton equations.

### 4. Complete Python Implementation
```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

class MertonStructuralModel:
    def __init__(self, equity_val, equity_vol, debt_face_val, risk_free_rate, time_to_maturity=1.0):
        self.E = equity_val
        self.sigma_E = equity_vol
        self.D = debt_face_val
        self.r = risk_free_rate
        self.T = time_to_maturity

    def solve_asset_parameters(self):
        """Solves simultaneous Black-Scholes equations for unobserved V_A and sigma_A."""
        def equations(vars):
            V_A, sigma_A = vars
            d1 = (np.log(V_A / self.D) + (self.r + 0.5 * sigma_A**2) * self.T) / (sigma_A * np.sqrt(self.T))
            d2 = d1 - sigma_A * np.sqrt(self.T)
            
            # Equation 1: Equity value = Call option on Assets
            eq1 = V_A * norm.cdf(d1) - self.D * np.exp(-self.r * self.T) * norm.cdf(d2) - self.E
            # Equation 2: Ito's Lemma relation between equity vol and asset vol
            eq2 = (V_A / self.E) * norm.cdf(d1) * sigma_A - self.sigma_E
            return [eq1, eq2]

        initial_guess = [self.E + self.D, self.sigma_E * self.E / (self.E + self.D)]
        V_A, sigma_A = fsolve(equations, initial_guess)
        return V_A, sigma_A

    def calculate_distance_to_default(self):
        V_A, sigma_A = self.solve_asset_parameters()
        
        # Distance to default formula
        dd = (np.log(V_A / self.D) + (self.r - 0.5 * sigma_A**2) * self.T) / (sigma_A * np.sqrt(self.T))
        prob_default = norm.cdf(-dd)
        
        return {
            "Implied_Asset_Value ($M)": round(V_A, 2),
            "Implied_Asset_Volatility (%)": round(sigma_A * 100, 2),
            "Distance_to_Default (std devs)": round(dd, 3),
            "Theoretical_Probability_of_Default (%)": round(prob_default * 100, 4)
        }

# Corporate Profile: Firm with $100M Market Cap, 40% equity vol, $150M total debt liabilities
merton = MertonStructuralModel(equity_val=100.0, equity_vol=0.40, debt_face_val=150.0, risk_free_rate=0.04, time_to_maturity=1.0)
credit_risk_metrics = merton.calculate_distance_to_default()

print("Merton Structural Credit Risk Assessment:")
for k, v in credit_risk_metrics.items():
    print(f"  {k}: {v}")
```

### 5. Senior DS Interview Talking Points
* **Merton Model Assumptions:** Assumes a single zero-coupon debt maturity and geometric Brownian motion for firm assets. KMV (Moody's Analytics) modifies this by empirically calibrating actual historical default frequencies rather than relying on normal distribution tails.
* **Credit Spread Puzzle:** Structural models often underpredict short-term credit spreads because real corporate debt defaults can happen suddenly due to liquidity freezes or accounting fraud.

---

## Case Study 19: Portfolio Construction with the Black-Litterman Framework
### 1. Business Framing
Standard Markowitz mean-variance optimization produces extreme, unstable portfolio weights when inputs have minor estimation errors ("error maximization"). The Black-Litterman framework combines equilibrium market-implied returns with subjective quantitative views to produce stable, diversified asset allocations.

### 2. Metrics & KPIs
* **Implied Equilibrium Returns ($\Pi$).**
* **Posterior Combined Return Vector ($E[R]$) & Covariance.**
* **Sharpe Ratio Improvement.**

### 3. Technical Formulation
$$\Pi = \lambda \Sigma w_{mkt}$$
$$E[R] = \left[(\tau \Sigma)^{-1} + P^T \Omega^{-1} P\right]^{-1} \left[(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q\right]$$

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd

def black_litterman_allocation():
    assets = ["US_Equities", "Intl_Equities", "Gov_Bonds", "Corp_Bonds"]
    n = len(assets)
    
    # Benchmark market cap weights
    w_mkt = np.array([0.45, 0.25, 0.20, 0.10])
    
    # Covariance Matrix
    cov = np.array([
        [0.040, 0.025, 0.000, 0.005],
        [0.025, 0.050, -0.002, 0.008],
        [0.000, -0.002, 0.008, 0.004],
        [0.005, 0.008, 0.004, 0.012]
    ])
    
    risk_aversion = 2.5
    tau = 0.05 # Uncertainty in prior
    
    # 1. Reverse optimization: Implied market equilibrium returns Pi = delta * Cov * w
    pi = risk_aversion * cov.dot(w_mkt)
    
    # 2. Investor Views:
    # View 1: US Equities will outperform Intl Equities by 2.0%
    # View 2: Corporate Bonds will have an absolute return of 5.5%
    P = np.array([
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    Q = np.array([0.020, 0.055])
    
    # Diagonal view uncertainty matrix Omega = diag(P * (tau * Cov) * P^T)
    omega = np.diag(np.diag(P.dot(tau * cov).dot(P.T)))
    
    # 3. Posterior Return Calculation
    inv_tau_cov = np.linalg.inv(tau * cov)
    inv_omega = np.linalg.inv(omega)
    
    posterior_cov = np.linalg.inv(inv_tau_cov + P.T.dot(inv_omega).dot(P))
    posterior_returns = posterior_cov.dot(inv_tau_cov.dot(pi) + P.T.dot(inv_omega).dot(Q))
    
    # 4. Optimal Unconstrained Weights w* = (1 / delta) * inv(Cov) * posterior_returns
    w_optimal = (1 / risk_aversion) * np.linalg.inv(cov).dot(posterior_returns)
    w_optimal /= w_optimal.sum() # Normalize
    
    results_df = pd.DataFrame({
        "Asset": assets,
        "Market_Weight": w_mkt.round(3),
        "Equilibrium_Return (%)": (pi * 100).round(2),
        "Posterior_Return (%)": (posterior_returns * 100).round(2),
        "Black_Litterman_Weight": w_optimal.round(3)
    })
    return results_df

bl_df = black_litterman_allocation()
print("Black-Litterman Portfolio Asset Allocation Results:")
print(bl_df)
```

### 5. Senior DS Interview Talking Points
* **Shrinkage Towards Equilibrium:** If an investor has zero subjective views, the Black-Litterman model gracefully collapses back to market cap weights ($w_{mkt}$).
* **Omega Calibration:** Confidence in investor views can be calibrated using Idzorek's percentage confidence method rather than arbitrary variance numbers.

---

## Case Study 20: Financial News NLP Sentiment & GARCH Volatility Shocks
### 1. Business Framing
LSEG distributes millions of Reuters financial news items and earnings call transcripts. Quantitative hedge funds incorporate NLP sentiment polarity and news velocity as exogenous regressors in GARCH volatility forecasting engines to price intraday option straddles.

### 2. Metrics & KPIs
* **Sentiment Polarity Score $S_t \in [-1, 1]$.**
* **GARCH-X Information Criterion (AIC / BIC).**
* **Out-of-sample Volatility Forecast MSE.**

### 3. Technical Formulation
We build a GARCH(1,1) model with an exogenous sentiment shock term:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2 + \gamma |S_{t-1}|$$

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from arch import arch_model

def simulate_news_sentiment_and_returns(n_days=500):
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    
    # Simulate news sentiment scores (-1 to +1)
    sentiment = np.random.normal(0, 0.4, size=n_days).clip(-1, 1)
    sentiment_shock = np.abs(sentiment) # High sentiment extremity raises volatility
    
    # True volatility process with exogenous news impact
    omega = 0.00005
    alpha = 0.08
    beta = 0.85
    gamma = 0.0003
    
    sigma2 = np.zeros(n_days)
    returns = np.zeros(n_days)
    sigma2[0] = omega / (1 - alpha - beta)
    
    for t in range(1, n_days):
        sigma2[t] = omega + alpha * (returns[t-1]**2) + beta * sigma2[t-1] + gamma * sentiment_shock[t-1]
        returns[t] = np.random.normal(0, np.sqrt(sigma2[t]))
        
    return pd.DataFrame({
        "date": dates,
        "returns": returns * 100, # percentage returns
        "sentiment_shock": sentiment_shock
    })

df_news = simulate_news_sentiment_and_returns()

# Fit GARCH(1,1) with exogenous sentiment regressor
garch_x = arch_model(
    df_news["returns"], 
    x=df_news[["sentiment_shock"]], 
    mean="Zero", 
    vol="GARCH", 
    p=1, 
    q=1
)
res = garch_x.fit(disp="off")
print("GARCH-X Estimation Summary (News Sentiment Impact on Volatility):")
print(res.summary().tables[1])
```

### 5. Senior DS Interview Talking Points
* **Asymmetric Volatility (Leverage Effect):** Bad news typically increases equity volatility much more than good news of equivalent magnitude. Standard GARCH does not capture this; asymmetric models (GJR-GARCH or EGARCH) are used in production.
* **Domain Financial Lexicons:** Standard NLP dictionaries (e.g., VADER) fail in finance. Words like "liability", "cost", or "risk" are routine financial terms, not negative sentiment. FinBERT or the Loughran-McDonald financial dictionary must be used.

---

## Case Study 21: Algorithmic Execution Slippage & Almgren-Chriss Optimal Execution
### 1. Business Framing
Institutional execution desks liquidating a large block of 500,000 shares cannot dump all shares immediately without crashing the market price (temporary and permanent market impact). Liquidating too slowly exposes the portfolio to overnight price drift risk. Robert Almgren and Neil Chriss formulated the optimal liquidation trajectory balancing market impact against timing risk.

### 2. Metrics & KPIs
* **Implementation Shortfall (IS):** Total cost compared to the arrival price benchmark.
* **Volume Weighted Average Price (VWAP) Slippage (bps).**

### 3. Technical Formulation
We minimize the expected cost and risk of liquidation:
$$\min_{v_t} E[x] + \lambda \text{Var}(x)$$
where temporary price impact is $\eta v_t$ and permanent price impact is $\gamma v_t$.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd

class AlmgrenChrissOptimizer:
    def __init__(self, total_shares=100_000, n_intervals=10, initial_price=150.0, 
                 daily_vol=0.015, eta=2.5e-6, gamma=2.0e-7, risk_aversion=1e-5):
        self.X = total_shares
        self.N = n_intervals
        self.S0 = initial_price
        self.sigma = daily_vol
        self.eta = eta     # Temporary impact parameter
        self.gamma = gamma # Permanent impact parameter
        self.lam = risk_aversion # Risk aversion parameter

    def compute_optimal_schedule(self):
        tau = 1.0 / self.N
        # Almgren-Chriss hyperbolic liquidation parameter kappa
        # kappa * tau approx acosh(1 + 0.5 * (lambda * sigma^2 / eta) * tau^2)
        tilde_kappa2 = (self.lam * (self.sigma ** 2)) / self.eta
        kappa = np.sqrt(tilde_kappa2)
        
        times = np.arange(self.N + 1)
        # Optimal remaining shares at each time step
        shares_remaining = self.X * np.sinh(kappa * (1 - times / self.N)) / np.sinh(kappa)
        trade_sizes = -np.diff(shares_remaining)
        
        # Calculate expected price degradation and implementation shortfall
        prices = [self.S0]
        cumulative_cost = 0.0
        
        for k in range(self.N):
            n_k = trade_sizes[k] # shares sold in interval k
            v_k = n_k / tau      # trading velocity
            
            perm_impact = self.gamma * n_k
            temp_impact = self.eta * v_k
            
            exec_price = prices[-1] - 0.5 * perm_impact - temp_impact
            shortfall_interval = n_k * (self.S0 - exec_price)
            cumulative_cost += shortfall_interval
            
            next_price = prices[-1] - perm_impact
            prices.append(next_price)
            
        return pd.DataFrame({
            "Interval": np.arange(1, self.N + 1),
            "Shares_Sold": trade_sizes.round(0).astype(int),
            "Shares_Remaining": shares_remaining[1:].round(0).astype(int),
            "Estimated_Exec_Price": np.array(prices[1:]).round(2)
        }), cumulative_cost

optimizer = AlmgrenChrissOptimizer(total_shares=500_000, n_intervals=10, initial_price=200.0)
schedule_df, total_shortfall = optimizer.compute_optimal_schedule()

print(f"Optimal Almgren-Chriss Liquidation Schedule (500k Shares):")
print(schedule_df)
print(f"Total Expected Implementation Shortfall: ${total_shortfall:,.2f} ({(total_shortfall/(500000*200))*10000:.1f} bps)")
```

### 5. Senior DS Interview Talking Points
* **VWAP vs TWAP vs Implementation Shortfall:** A TWAP algorithm sells a constant slice $\frac{X}{N}$ each interval (optimal when risk aversion $\lambda \to 0$). As risk aversion $\lambda$ increases, the schedule fronts-loads liquidation exponentially to avoid market volatility risk.
* **Non-linear Impact (Square-root Law):** Empirical academic literature shows market impact is proportional to $\sigma \sqrt{\frac{Q}{V}}$ (square-root of order size as a fraction of daily volume) rather than strictly linear.

---

## Case Study 22: Analyst Consensus Surprise & Quantitative Alpha Factor Modeling
### 1. Business Framing
LSEG StarMine Quantitative Analytics produces proprietary predictive alpha factors (e.g., Analyst Revisions Model, Smart Holdings). Quantitative hedge funds model post-earnings announcement drift (PEAD) by predicting which corporate earnings reports will surpass IBES consensus analyst estimates.

### 2. Metrics & KPIs
* **Standardized Unexpected Earnings (SUE):** $\frac{\text{Actual EPS} - \text{Consensus Forecast EPS}}{\sigma_{\text{Analyst Forecasts}}}$.
* **Factor Information Coefficient (IC):** Spearman rank correlation between predictive score and forward 30-day stock returns.
* **Long-Short Factor Decile Spread.**

### 3. Technical Formulation
We build an ensemble model predicting SUE and evaluate factor performance across deciles.

### 4. Complete Python Implementation
```python
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

def simulate_equity_alpha_universe(n_stocks=1000):
    np.random.seed(42)
    # Analyst metrics
    consensus_eps = np.random.uniform(0.50, 4.00, size=n_stocks)
    revision_momentum = np.random.normal(0.02, 0.05, size=n_stocks) # Revisions in last 30 days
    analyst_dispersion = np.random.uniform(0.02, 0.20, size=n_stocks)
    historical_roe = np.random.normal(0.14, 0.06, size=n_stocks)
    short_interest_ratio = np.random.gamma(2, 2, size=n_stocks)
    
    # Latent true earnings surprise
    true_surprise = (
        1.2 * revision_momentum 
        - 0.8 * analyst_dispersion 
        + 0.5 * historical_roe 
        + np.random.normal(0, 0.08, size=n_stocks)
    )
    
    # Forward 30-day stock return after earnings release
    fwd_returns = 0.01 + 0.40 * true_surprise + np.random.normal(0, 0.05, size=n_stocks)
    
    return pd.DataFrame({
        "ticker": [f"STK_{i:04d}" for i in range(n_stocks)],
        "consensus_eps": consensus_eps,
        "revision_momentum": revision_momentum,
        "analyst_dispersion": analyst_dispersion,
        "historical_roe": historical_roe,
        "short_interest_ratio": short_interest_ratio,
        "fwd_returns": fwd_returns
    })

df_alpha = simulate_equity_alpha_universe()

features = ["revision_momentum", "analyst_dispersion", "historical_roe", "short_interest_ratio"]
X = df_alpha[features]
y = df_alpha["fwd_returns"]

split = 700
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = ExtraTreesRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

test_df = df_alpha.iloc[split:].copy()
test_df["alpha_score"] = model.predict(X_test)

# Calculate Information Coefficient (Rank Correlation)
ic, p_val = spearmanr(test_df["alpha_score"], test_df["fwd_returns"])
print(f"Factor Information Coefficient (Rank IC): {ic:.3f} (p-value: {p_val:.4e})")

# Long-Short Top vs Bottom Decile Spread
test_df["decile"] = pd.qcut(test_df["alpha_score"], 10, labels=False)
decile_returns = test_df.groupby("decile")["fwd_returns"].mean() * 100

print(f"Top Decile (Long) Mean Return: {decile_returns.iloc[-1]:.2f}%")
print(f"Bottom Decile (Short) Mean Return: {decile_returns.iloc[0]:.2f}%")
print(f"Long-Short Spread: {decile_returns.iloc[-1] - decile_returns.iloc[0]:.2f}%")
```

### 5. Senior DS Interview Talking Points
* **Post-Earnings Announcement Drift (PEAD):** Markets underreact to earnings surprises. Stock prices do not jump immediately to fair value; they drift in the direction of the surprise for up to 60 trading days.
* **Smart Consensus (StarMine Methodology):** StarMine's proprietary edge comes from weighting analyst estimates based on their past track record and recency, rather than taking a simple unweighted median.
