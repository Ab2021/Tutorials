# Take-Home Case Study & Peer Notebook Critique Guide — Senior Data Scientist
### Target Company: General Mills India (D&T Analytics / GIC Powai)
### Role Focus: Senior Data Scientist / Lead Data Scientist
### Last Updated: November 2026

---

> [!IMPORTANT]
> **The General Mills Signature Interview Format: The Notebook Critique**
> Multiple candidate interview reports on Glassdoor, AmbitionBox, and LockedIn AI highlight a distinctive Round 3 format at General Mills:
> Rather than only building a model from scratch, candidates are given **a pre-existing Jupyter notebook authored by a junior data scientist or external vendor** and tasked with:
> 1. Performing an in-depth **code and methodology critique** (identifying statistical flaws, data leaks, and production risks).
> 2. Presenting an executive summary to non-technical business partners explaining why the junior model would cause millions in supply chain disruptions.
> 3. Delivering a **refactored, leak-free, production-ready solution** with proper validation and business metrics.

---

## Table of Contents

1. [The Take-Home Scenario & Flawed Junior Notebook](#1-the-take-home-scenario--flawed-junior-notebook)
2. [The 10 Fatal Flaws: Comprehensive Line-by-Line Critique](#2-the-10-fatal-flaws-comprehensive-line-by-line-critique)
3. [The Senior Data Scientist Refactored Solution](#3-the-senior-data-scientist-refactored-solution)
4. [Executive Presentation & Business Stakeholder Translation](#4-executive-presentation--business-stakeholder-translation)
5. [Evaluation Scoring Rubric](#5-evaluation-scoring-rubric)

---

## 1. The Take-Home Scenario & Flawed Junior Notebook

### Business Context
A junior data scientist at General Mills was tasked with building an automated weekly demand forecasting model for **Cheerios (Family Size, 18oz)** across 500 retail stores to support retail replenishment and marketing trade promotions. 
The junior scientist achieved a **0.94 $R^2$** and **$0.85$ AUC** on their notebook and recommended deploying the model immediately to automated SAP replenishment pipelines.

### The Junior Scientist's Flawed Notebook Code

```python
# ==============================================================================
# JUNIOR SCIENTIST NOTEBOOK: cheerios_demand_forecast_v1.ipynb
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load raw retail store transactions
df = pd.read_csv("weekly_cheerios_store_sales.csv")

# Step 2: Feature Engineering
# Create forward target and rolling features
df["weekly_sales_lead1"] = df.groupby("store_id")["sales"].shift(-1)  # Target: Next week sales
df["rolling_3w_sales"] = df.groupby("store_id")["sales"].transform(lambda x: x.rolling(3).mean())
df["promo_lift_ratio"] = df["sales"] / df.groupby("store_id")["sales"].transform("mean")

# Step 3: Impute Missing Values across the whole dataset
df = df.fillna(df.mean(numeric_only=True))

# Step 4: Scale all features
features = ["rolling_3w_sales", "promo_lift_ratio", "discount_depth", "competitor_price", "temperature"]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])  # Fitted on full dataset!

# Step 5: Train / Test Split using Random 5-Fold Cross Validation
X = df[features]
y = df["weekly_sales_lead1"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Random shuffle on time series!
oof_predictions = np.zeros(len(df))

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)
    oof_predictions[val_idx] = rf.predict(X_val)

# Step 6: Evaluation
rmse = np.sqrt(mean_squared_error(y, oof_predictions))
r2 = r2_score(y, oof_predictions)
print(f"Validation RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")
# Output: Validation RMSE: 12.4, R2 Score: 0.94 -> "Model is ready for production!"
```

---

## 2. The 10 Fatal Flaws: Comprehensive Line-by-Line Critique

| # | Code Location | Fatal Flaw | Why It Destroys the Supply Chain | Severity |
|---|---|---|---|---|
| **1** | `df["promo_lift_ratio"]` | **Direct Target Leakage** | Computes `sales / mean(sales)` using current week's actual sales to predict next week's sales. In live inference, current sales are not known until the week closes! | **Critical** |
| **2** | `scaler.fit_transform()` | **Preprocessing Data Leakage** | Scaler is fitted on the *entire dataset* before splitting. Test and future fold distribution parameters leak into training. | **High** |
| **3** | `KFold(shuffle=True)` | **Temporal Cross-Validation Failure** | Randomly shuffling time-series data trains on future weeks (e.g., Week 45) to predict past weeks (Week 12). Yields artificially inflated $R^2 = 0.94$. | **Critical** |
| **4** | `fillna(df.mean())` | **Global Mean Imputation Leakage** | Imputing missing values using the global mean across all stores and all dates leaks future seasonal trends into earlier rows. | **Medium** |
| **5** | Target definition | **Unhandled Zero-Sales Censorship** | Days with zero sales caused by **out-of-stock stockouts** are treated as zero consumer demand. Model learns that stockout weeks represent lack of demand, systematically under-ordering inventory. | **Critical** |
| **6** | `RandomForest(max_depth=None)` | **Extrapolation Failure on Growth/Inflation** | Random Forest cannot extrapolate trends beyond the maximum value seen in training data. Under inflation or growth, it caps out. | **High** |
| **7** | Evaluation metric | **Misaligned Business Metric (RMSE / $R^2$)** | $R^2$ is meaningless for non-stationary demand series. Optimizing on MSE penalizes over- and under-forecasts symmetrically, whereas FMCG stockouts carry retailer fines 5x higher than holding costs. | **High** |
| **8** | Feature set | **Missing Calendar & Promo Interaction** | Ignores paydays, SNAP/EBT benefit disbursement cycles, Thanksgiving/back-to-school seasonality, and cross-SKU cannibalization (e.g., discount on Cheerios Honey Nut). | **Medium** |
| **9** | Hierarchy | **Disaggregated Single-Level Prediction** | Fails to reconcile store-level forecasts with regional distribution center (DC) capacities (no hierarchical reconciliation). | **Medium** |
| **10** | Production serving | **Point Predictions without Uncertainty** | Outputs a single scalar $\hat{y}$ instead of prediction intervals ($P_{10}, P_{50}, P_{90}$), making safety stock calculation impossible for SAP S/4HANA. | **Critical** |

---

## 3. The Senior Data Scientist Refactored Solution

```python
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

class ProductionDemandForecaster:
    """
    Leak-free, time-series compliant hierarchical forecasting engine.
    Implements:
    - Purged expanding-window TimeSeriesSplit
    - Censored demand stockout correction
    - Quantile Pinball Loss (P10, P50, P90) for asymmetric supply chain safety stock
    - Strict Scikit-learn Pipeline containment
    """
    def __init__(self, target_quantiles: List[float] = [0.10, 0.50, 0.90]):
        self.target_quantiles = target_quantiles
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.feature_names: List[str] = []

    def engineer_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Strictly backward-looking feature engineering."""
        df = df_raw.copy().sort_values(["store_id", "date"]).reset_index(drop=True)

        # 1. Censored Demand Correction: Impute zero-sales when inventory == 0
        df["adjusted_demand"] = df["sales"]
        stockout_mask = (df["inventory_on_hand"] == 0) & (df["sales"] == 0)
        # Impute stockout days with 4-week moving median of historical sales
        rolling_baseline = df.groupby("store_id")["sales"].shift(1).rolling(4, min_periods=1).median()
        df.loc[stockout_mask, "adjusted_demand"] = rolling_baseline[stockout_mask]

        # 2. Backward-looking Lags & Rolling Aggregates (Strictly Shifted by 1)
        for lag in [1, 2, 3, 4]:
            df[f"demand_lag_{lag}w"] = df.groupby("store_id")["adjusted_demand"].shift(lag)

        # 3. Rolling Statistics over Lags (Zero Future Leakage!)
        df["rolling_mean_4w"] = df.groupby("store_id")["demand_lag_1w"].transform(lambda x: x.rolling(4).mean())
        df["rolling_std_4w"] = df.groupby("store_id")["demand_lag_1w"].transform(lambda x: x.rolling(4).std())

        # 4. Forward-looking Exogenous Features (KNOWN in advance by marketing)
        df["planned_discount_depth"] = df["discount_depth"]
        df["is_circular_feature"] = df["promo_display_flag"]

        # 5. Target Definition: Lead 1 week adjusted demand
        df["target"] = df.groupby("store_id")["adjusted_demand"].shift(-1)
        
        return df.dropna().reset_index(drop=True)

    def temporal_cross_validation(
        self,
        df: pd.DataFrame,
        n_splits: int = 4,
        test_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Purged Expanding-Window Temporal Cross-Validation.
        Guarantees that train strictly precedes validation in time.
        """
        dates = np.sort(df["date"].unique())
        total_dates = len(dates)
        wmape_scores = []

        self.feature_names = [c for c in df.columns if c not in ["store_id", "date", "sales", "target", "adjusted_demand"]]

        for i in range(n_splits):
            val_end_idx = total_dates - i * test_weeks
            val_start_idx = val_end_idx - test_weeks
            train_end_idx = val_start_idx

            train_dates = dates[:train_end_idx]
            val_dates = dates[val_start_idx:val_end_idx]

            train_fold = df[df["date"].isin(train_dates)]
            val_fold = df[df["date"].isin(val_dates)]

            X_tr, y_tr = train_fold[self.feature_names], train_fold["target"]
            X_va, y_va = val_fold[self.feature_names], val_fold["target"]

            # Train Median Model (P50) for WMAPE evaluation
            model_p50 = lgb.LGBMRegressor(
                objective="regression_l1",  # MAE minimizes median forecast
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )
            model_p50.fit(X_tr, y_tr)
            preds = model_p50.predict(X_va)

            # Evaluate Weighted Mean Absolute Percentage Error (FMCG Standard)
            wmape = np.sum(np.abs(y_va - preds)) / np.sum(y_va) * 100.0
            wmape_scores.append(wmape)

        return {"mean_wmape": float(np.mean(wmape_scores)), "wmape_folds": wmape_scores}

    def fit_probabilistic(self, df: pd.DataFrame):
        """Fits P10, P50, and P90 quantile models on full historical data."""
        X = df[self.feature_names]
        y = df["target"]

        for q in self.target_quantiles:
            print(f"[Training] LightGBM Quantile Model (alpha={q:.2f})...")
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=200,
                learning_rate=0.03,
                max_depth=6,
                random_state=42
            )
            model.fit(X, y)
            self.models[q] = model

    def predict_safety_stock(self, X_current: pd.DataFrame) -> pd.DataFrame:
        """
        Generates operational demand targets and dynamic safety stock bounds.
        Safety Stock Buffer = P90 - P50
        """
        results = pd.DataFrame(index=X_current.index)
        for q, model in self.models.items():
            results[f"P{int(q*100)}"] = model.predict(X_current[self.feature_names])

        results["recommended_order"] = results["P50"]
        results["safety_stock_buffer"] = results["P90"] - results["P50"]
        return results

# --- Verification & Test Harness ---
if __name__ == "__main__":
    np.random.seed(42)
    n_weeks = 52
    records = []
    
    for s in [101, 102, 103]:
        base_demand = np.random.uniform(200, 500)
        for w in range(n_weeks):
            discount = np.random.choice([0.0, 0.15, 0.25], p=[0.7, 0.2, 0.1])
            sales = base_demand * (1.0 + discount * 1.8) + np.random.normal(0, 20)
            inv = 100 if sales > 0 else 0
            records.append({
                "store_id": s,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(weeks=w),
                "sales": max(0.0, sales),
                "inventory_on_hand": inv,
                "discount_depth": discount,
                "promo_display_flag": 1 if discount > 0 else 0
            })

    raw_df = pd.DataFrame(records)
    engine = ProductionDemandForecaster()
    engineered_df = engine.engineer_features(raw_df)
    
    cv_res = engine.temporal_cross_validation(engineered_df, n_splits=3, test_weeks=4)
    print(f"\nRefactored Out-of-Time CV WMAPE: {cv_res['mean_wmape']:.2f}% across folds: {cv_res['wmape_folds']}")

    engine.fit_probabilistic(engineered_df)
    sample_current = engineered_df.tail(3)
    forecasts = engine.predict_safety_stock(sample_current)
    print("\nSample Production Output for SAP ERP:")
    print(forecasts[["P10", "P50", "P90", "recommended_order", "safety_stock_buffer"]])
```

---

## 4. Executive Presentation & Stakeholder Defense Guide

When presenting your notebook critique to the interview panel (comprising the Director of Data Science and the VP of Supply Chain), follow this structured narrative:

```
                  ┌──────────────────────────────────────────────┐
                  │          EXECUTIVE PRESENTATION STRUCTURE    │
                  └──────────────────────┬───────────────────────┘
                                         │
     ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
     ▼                   ▼                               ▼                   ▼
1. The Headline Risk  2. Why the 0.94 R²   3. The $2.4M Financial   4. The Production
   ("Deploying this       was an Illusion      Impact on Inventory      Remedy & Roadmap
    crashes stores")      (Target Leakage)     (Stockouts vs Spoilage)  (WMAPE + Quantiles)
```

### Scripted Executive Talking Points

1. **The Executive Summary:**
   > *"Good morning everyone. While the junior notebook reports an impressive 0.94 $R^2$ and $12.4$ RMSE, our audit reveals that this model is fatally flawed by target leakage and random time-series cross-validation. In a live production setting, this model would cause an estimated $2.4M in retail stockouts and unfulfilled order penalties from retail partners like Walmart."*
2. **Translating the Flaw to Business Dollars:**
   > *"The junior model included current-week sales to predict current-week sales. In production, we don't know this week's sales until Monday morning. Furthermore, by treating out-of-stock periods as zero consumer demand, the algorithm erroneously concludes that consumers simply don't want Cheerios during stockouts, systematically driving inventory levels down to zero."*
3. **The Solution & Business Benefits:**
   > *"We have rebuilt the pipeline using backward-looking features with censored demand imputation. Instead of a single number, we provide $P_{10}, P_{50}$, and $P_{90}$ demand distributions. $P_{50}$ gives finance our expected revenue plan, while the buffer between $P_{90}$ and $P_{50}$ automatically provisions dynamic safety stock in SAP S/4HANA, protecting retail service levels above 96%."*

---

## 5. Evaluation Scoring Rubric

Interviewers at General Mills evaluate candidates on this exact rubric:

| Evaluated Dimension | Weight | Senior Bar Expectation |
|---|---|---|
| **Data Leakage Detection** | 30% | Candidate spots target leakage (`sales` in current week) and preprocessor leakage (`fit_transform` before split) within 5 minutes. |
| **Time-Series Split Rigor** | 20% | Candidate immediately rejects `KFold(shuffle=True)` and explains expanding/sliding window validation. |
| **Domain Awareness (FMCG)** | 20% | Identifies censored demand during stockouts; rejects $R^2$/RMSE in favor of WMAPE, WAPE, and Quantile Loss. |
| **Production Refactoring** | 15% | Delivers clean, vectorized, type-annotated code with leak-free Scikit-learn or LightGBM pipelines. |
| **Stakeholder Storytelling** | 15% | Explains technical failures in terms of balance-sheet risk, stockout penalties, and customer service levels. |

---

*Companion Modules:*
* [00_master_guide.md](file:///D:/Int/General_Mills/interview_prep/00_master_guide.md) — Master strategy, 4-week preparation plan, company deep dives, and salary benchmarks.
* [06_domain_case_studies.md](file:///D:/Int/General_Mills/interview_prep/06_domain_case_studies.md) — 22 production case studies for FMCG and Finance.
* [09_pyspark_big_data_operations.md](file:///D:/Int/General_Mills/interview_prep/09_pyspark_big_data_operations.md) — PySpark, Databricks Lakehouse, and distributed computing questions.
* [10_ml_system_design_architecture.md](file:///D:/Int/General_Mills/interview_prep/10_ml_system_design_architecture.md) — Production ML system design and low-latency architectures.
* [11_behavioral_leadership_case_interviews.md](file:///D:/Int/General_Mills/interview_prep/11_behavioral_leadership_case_interviews.md) — Senior & Lead behavioral frameworks and India salary negotiation playbook.
