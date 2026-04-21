# 💻 HO GRIND — Complete Live Coding Challenges & EDA Playbook
### Round 3: Hands-On | Full implementations ready to write live

---

## ═══════════════════════════════════════
## SECTION 1: COMPLETE EDA PLAYBOOK
## ═══════════════════════════════════════

### The Live EDA Protocol — What to Do in the First 20 Minutes

```python
"""
LIVE INTERVIEW EDA PROTOCOL
Always run in this EXACT order. Narrate every step out loud.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ─── STEP 0: Load and Sanity Check (2 min) ───
df = pd.read_csv('fraud_data.csv')  # or whatever they give you

# First things first — ALWAYS
print("=== BASIC INFO ===")
print(f"Shape: {df.shape}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"\nDTypes:\n{df.dtypes.value_counts()}")
print(f"\nFirst 5 rows:")
display(df.head())

# ─── STEP 1: Missing Values (ALWAYS do this before anything else) ───
print("\n=== MISSING VALUES ===")
missing = pd.DataFrame({
    'count': df.isnull().sum(),
    'pct': (df.isnull().sum() / len(df) * 100).round(2),
    'dtype': df.dtypes
}).sort_values('pct', ascending=False)
missing = missing[missing['count'] > 0]

if len(missing) > 0:
    print(missing)
    # Key question to ask out loud: "Is this missing at random?"
    # Check: is missingness correlated with target?
    TARGET = 'is_fraud'  # Whatever target column is
    for col in missing[missing['pct'] > 5].index:
        print(f"\n{col}: Fraud rate where missing = "
              f"{df[df[col].isnull()][TARGET].mean():.3f} "
              f"vs where present = {df[df[col].notna()][TARGET].mean():.3f}")
        # Key insight: if fraud rate differs → missingness is INFORMATIVE
else:
    print("No missing values — lucky!")

# ─── STEP 2: Target Distribution (Critical for imbalance) ───
print("\n=== TARGET DISTRIBUTION ===")
TARGET = 'is_fraud'  # change as appropriate
value_counts = df[TARGET].value_counts()
print(value_counts)
print(f"\nFraud rate: {df[TARGET].mean():.4f} ({df[TARGET].mean()*100:.2f}%)")
print(f"Imbalance ratio: {value_counts.max()/value_counts.min():.0f}:1")

if df[TARGET].mean() < 0.1:
    print("⚠️  SEVERE IMBALANCE: Will need scale_pos_weight or PR-AUC as metric")
    print(f"   Recommended scale_pos_weight = {value_counts[0]/value_counts[1]:.0f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df[TARGET].value_counts().plot(kind='bar', ax=axes[0], color=['green','red'])
axes[0].set_title('Class Distribution')
df[TARGET].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
axes[1].set_title('Fraud Rate')
plt.tight_layout()
plt.show()

# ─── STEP 3: Feature Type Separation ───
num_cols = df.select_dtypes(include=['number']).columns.drop(TARGET, errors='ignore').tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
dt_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]

print(f"\n=== FEATURE TYPES ===")
print(f"Numeric: {len(num_cols)} features")
print(f"Categorical: {len(cat_cols)} features")
print(f"Datetime: {len(dt_cols)} features")

# ─── STEP 4: Numeric Features Analysis ───
print("\n=== NUMERIC FEATURES ===")
stats_df = df[num_cols].describe().T
stats_df['skewness'] = df[num_cols].skew()
stats_df['kurtosis'] = df[num_cols].kurtosis()
stats_df['pct_zero'] = (df[num_cols] == 0).mean()
stats_df['n_unique'] = df[num_cols].nunique()
print(stats_df[['mean', 'std', 'min', 'max', 'skewness', 'kurtosis', 'pct_zero', 'n_unique']].round(3))

# Correlation with target
print("\nCorrelation with target (absolute value, top 15):")
corrs = df[num_cols].corrwith(df[TARGET]).abs().sort_values(ascending=False).head(15)
print(corrs)

# Outlier detection
print("\nPotential outliers (IQR method):")
for col in num_cols[:10]:  # Check first 10
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    if outliers > 0:
        print(f"  {col}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")

# ─── STEP 5: Categorical Features Analysis ───
print("\n=== CATEGORICAL FEATURES ===")
for col in cat_cols:
    n_unique = df[col].nunique()
    top_val = df[col].value_counts().index[0] if n_unique > 0 else 'N/A'
    top_pct = df[col].value_counts().iloc[0] / len(df) * 100 if n_unique > 0 else 0
    fraud_by_cat = df.groupby(col)[TARGET].mean()
    print(f"\n{col}: {n_unique} unique | Top: '{top_val}' ({top_pct:.1f}%)")
    print(f"  Fraud rate range: [{fraud_by_cat.min():.3f}, {fraud_by_cat.max():.3f}]")
    if fraud_by_cat.max() / (fraud_by_cat.min() + 1e-8) > 2:
        print(f"  ✓ High fraud rate variability → STRONG predictor")

# ─── STEP 6: DateTime Features ───
if dt_cols:
    print("\n=== TEMPORAL FEATURES ===")
    for col in dt_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"{col}: {df[col].min()} to {df[col].max()}")
        print(f"  Temporal split needed? YES — never random split time series")

# ─── STEP 7: Key Bivariate Insights (Fraud-focused) ───
print("\n=== KEY BIVARIATE INSIGHTS ===")

# Most powerful viz for fraud: Distribution of key features by class
top_features = corrs.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(top_features):
    ax = axes[i//3][i%3]
    df[df[TARGET]==0][col].hist(alpha=0.5, bins=50, label='Legitimate', ax=ax, density=True)
    df[df[TARGET]==1][col].hist(alpha=0.5, bins=50, label='Fraud', ax=ax, density=True, color='red')
    ax.set_title(f'{col}\n(corr={corrs[col]:.3f})')
    ax.legend()
plt.suptitle('Feature Distribution: Fraud vs. Legitimate')
plt.tight_layout()
plt.show()

print("\n=== EDA SUMMARY ===")
print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} features")
print(f"Fraud rate: {df[TARGET].mean()*100:.2f}% — {'SEVERE' if df[TARGET].mean() < 0.05 else 'MODERATE'} imbalance")
print(f"Missing data: {'Yes — need imputation strategy' if len(missing) > 0 else 'Clean'}")
print(f"Top predictors: {', '.join(corrs.head(5).index.tolist())}")
print(f"Split strategy: {'TEMPORAL' if dt_cols else 'STRATIFIED K-FOLD'}")
```

---

## ═══════════════════════════════════════
## SECTION 2: FEATURE ENGINEERING — LIVE PATTERNS
## ═══════════════════════════════════════

```python
"""
COMPLETE FEATURE ENGINEERING PIPELINE
Narrate your thinking: "I'm creating velocity features because 
fraud often involves sudden spikes in activity..."
"""

class FraudFeatureEngineer:
    """Production-grade fraud feature engineering."""
    
    def __init__(self, user_col='user_id', time_col='timestamp', 
                 amount_col='amount'):
        self.user_col = user_col
        self.time_col = time_col
        self.amount_col = amount_col
        self.fitted_stats = {}
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and transform."""
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df = df.sort_values([self.user_col, self.time_col])
        
        df = self._temporal_features(df)
        df = self._velocity_features(df)
        df = self._behavioral_anomaly_features(df)
        df = self._recency_features(df)
        df = self._ratio_features(df)
        
        # Store user stats for transform (avoid leakage)
        self.fitted_stats['user_stats'] = self._compute_user_stats(df)
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform test/serving data using fitted stats."""
        df = df.copy()
        # Join pre-computed user stats (no leakage)
        df = df.merge(self.fitted_stats['user_stats'], 
                     on=self.user_col, how='left')
        df = self._temporal_features(df)
        return df
    
    def _temporal_features(self, df):
        """Time-of-day and seasonal features."""
        df['hour_of_day'] = df[self.time_col].dt.hour
        df['day_of_week'] = df[self.time_col].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_nighttime'] = (~df['hour_of_day'].between(8, 22)).astype(int)
        df['is_business_hours'] = df['hour_of_day'].between(9, 17).astype(int)
        # Cyclical encoding (preserves circular nature of hour/day)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        return df
    
    def _velocity_features(self, df):
        """Transaction velocity — most predictive for fraud."""
        # Rolling count in time windows (IMPORTANT: use expanding for first transactions)
        for col_name, desc in [('txn_seq_num', 'sequential number')]:
            df[col_name] = df.groupby(self.user_col).cumcount() + 1
        
        # Simplified velocity (in real code, use proper time-indexed rolling)
        df['txn_count_7d_approx'] = (
            df.groupby(self.user_col)[self.amount_col]
            .transform(lambda x: x.expanding().count())
        )
        df['amount_sum_7d_approx'] = (
            df.groupby(self.user_col)[self.amount_col]
            .transform(lambda x: x.expanding().sum())
        )
        df['amount_avg_7d_approx'] = (
            df.groupby(self.user_col)[self.amount_col]
            .transform(lambda x: x.expanding().mean())
        )
        return df
    
    def _behavioral_anomaly_features(self, df):
        """Z-score from user's own baseline — catches behavioral anomalies."""
        # User-level statistics (computed on training data to avoid leakage)
        user_stats = df.groupby(self.user_col)[self.amount_col].agg(
            user_mean='mean', user_std='std', user_median='median',
            user_max='max', user_min='min'
        ).reset_index()
        
        df = df.merge(user_stats, on=self.user_col, how='left')
        
        # Z-score: how unusual is this transaction for THIS user?
        df['amount_zscore_vs_user'] = (
            (df[self.amount_col] - df['user_mean']) / 
            (df['user_std'] + 1e-8)
        )
        df['amount_ratio_to_user_max'] = df[self.amount_col] / (df['user_max'] + 1e-8)
        df['is_new_high_for_user'] = (df[self.amount_col] > df['user_max']).astype(int)
        df['amount_vs_user_median_ratio'] = df[self.amount_col] / (df['user_median'] + 1e-8)
        
        return df
    
    def _recency_features(self, df):
        """How long since first transaction? Time between transactions."""
        df['time_since_last_txn'] = (
            df.groupby(self.user_col)[self.time_col]
            .transform(lambda x: x.diff().dt.total_seconds() / 3600)  # hours
        )
        df['is_rapid_succession'] = (df['time_since_last_txn'] < 0.5).astype(int)  # <30 min
        df['days_as_customer'] = (
            df.groupby(self.user_col)[self.time_col]
            .transform(lambda x: (x - x.min()).dt.days)
        )
        df['is_new_customer'] = (df['days_as_customer'] < 30).astype(int)
        return df
    
    def _ratio_features(self, df):
        """Business-relevant ratio features."""
        # Transaction amount as % of user's 7-day average
        df['amount_vs_7d_avg'] = df[self.amount_col] / (df['amount_avg_7d_approx'] + 1e-8)
        # High-value flag
        df['is_high_value'] = (df[self.amount_col] > df[self.amount_col].quantile(0.95)).astype(int)
        return df
    
    def _compute_user_stats(self, df):
        return df.groupby(self.user_col).agg(
            user_global_mean=(self.amount_col, 'mean'),
            user_global_std=(self.amount_col, 'std'),
            user_total_txns=(self.amount_col, 'count'),
        ).reset_index()


# Categorical Encoding — Smart Choices
def encode_categoricals(df_train, df_test, cat_cols, target_col='is_fraud'):
    """
    Strategy matrix:
    < 10 unique values: one-hot
    10-50 unique values: target encoding (with regularization)
    > 50 unique values: frequency encoding + target encoding
    """
    from sklearn.preprocessing import LabelEncoder
    
    for col in cat_cols:
        n_unique = df_train[col].nunique()
        
        if n_unique <= 10:
            # One-hot encoding
            dummies = pd.get_dummies(df_train[col], prefix=col)
            df_train = pd.concat([df_train, dummies], axis=1)
            # Apply same categories to test (handle new categories)
            dummies_test = pd.get_dummies(df_test[col], prefix=col)
            df_test = df_test.reindex(
                columns=df_train.columns, fill_value=0
            )
            print(f"{col}: One-hot ({n_unique} unique)")
            
        elif n_unique <= 50:
            # Target encoding with smoothing (avoids overfitting for rare categories)
            smoothing = 10  # Weight for global mean
            global_mean = df_train[target_col].mean()
            stats = df_train.groupby(col)[target_col].agg(['mean', 'count'])
            smoothed = (stats['mean'] * stats['count'] + global_mean * smoothing) / \
                       (stats['count'] + smoothing)
            df_train[f'{col}_te'] = df_train[col].map(smoothed).fillna(global_mean)
            df_test[f'{col}_te'] = df_test[col].map(smoothed).fillna(global_mean)
            print(f"{col}: Target encoding with smoothing ({n_unique} unique)")
            
        else:
            # Frequency encoding (less overfitting than target encoding for high-cardinality)
            freq_map = df_train[col].value_counts(normalize=True)
            df_train[f'{col}_freq'] = df_train[col].map(freq_map).fillna(0)
            df_test[f'{col}_freq'] = df_test[col].map(freq_map).fillna(0)
            print(f"{col}: Frequency encoding ({n_unique} unique — HIGH CARDINALITY)")
    
    return df_train, df_test
```

---

## ═══════════════════════════════════════
## SECTION 3: COMPLETE MODEL TRAINING PIPELINE
## ═══════════════════════════════════════

```python
"""
PRODUCTION-QUALITY FRAUD MODEL TRAINING
Includes: class imbalance, temporal CV, calibration, SHAP, threshold tuning
"""

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             precision_recall_curve, f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import shap
import optuna

def full_fraud_training_pipeline(
    df: pd.DataFrame, 
    feature_cols: list,
    target_col: str = 'is_fraud',
    time_col: str = 'timestamp',
    n_cv_folds: int = 5,
    investigation_capacity: int = 500
) -> dict:
    """
    Full pipeline: temporal CV + tuning + calibration + SHAP + threshold.
    """
    
    # ─── 1. TEMPORAL SPLIT ───
    df = df.sort_values(time_col)
    n = len(df)
    train_df = df.iloc[:int(n*0.8)]
    test_df = df.iloc[int(n*0.8):]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"Train fraud rate: {y_train.mean():.4f} | Test: {y_test.mean():.4f}")
    
    # ─── 2. HANDLE CLASS IMBALANCE ───
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.1f}")
    
    # ─── 3. HYPERPARAMETER OPTIMIZATION (Optuna) ───
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'average_precision',
            'scale_pos_weight': scale_pos_weight,
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1
        }
        
        # Temporal CV
        tscv = TimeSeriesSplit(n_splits=n_cv_folds)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_cv_train, y_cv_train,
                     eval_set=[(X_cv_val, y_cv_val)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
            
            y_proba = model.predict_proba(X_cv_val)[:, 1]
            cv_scores.append(average_precision_score(y_cv_val, y_proba))
        
        return np.mean(cv_scores)
    
    print("Optimizing hyperparameters...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    best_params = study.best_params
    print(f"Best PR-AUC (CV): {study.best_value:.4f}")
    print(f"Best params: {best_params}")
    
    # ─── 4. FINAL MODEL TRAINING ───
    final_params = {**best_params, 
                   'objective': 'binary', 'metric': 'average_precision',
                   'scale_pos_weight': scale_pos_weight, 'n_estimators': 2000,
                   'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    
    model = lgb.LGBMClassifier(**final_params)
    model.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             callbacks=[lgb.early_stopping(100, verbose=False)])
    
    # ─── 5. PROBABILITY CALIBRATION ───
    # Split train into pre-calibration train + calibration set
    cal_size = int(len(X_train) * 0.2)
    X_cal, y_cal = X_train.iloc[-cal_size:], y_train.iloc[-cal_size:]
    
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_cal, y_cal)
    
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    # ─── 6. EVALUATION ───
    print("\n=== EVALUATION RESULTS ===")
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)
    print(f"AUC-ROC:  {auc_roc:.4f}")
    print(f"PR-AUC:   {auc_pr:.4f}")
    
    # KS Statistic
    fraud_scores = y_proba[y_test == 1]
    good_scores = y_proba[y_test == 0]
    ks_stat = max(abs(
        np.array([(fraud_scores <= t).mean() - (good_scores <= t).mean() 
                  for t in sorted(y_proba)])
    ))
    print(f"KS Stat:  {ks_stat:.4f} ({'Good' if ks_stat > 0.4 else 'Check'})")
    print(f"Gini:     {2*auc_roc-1:.4f}")
    
    # Calibration check
    fraction_of_positives, mean_predicted = calibration_curve(
        y_test, y_proba, n_bins=10
    )
    cal_error = np.mean(np.abs(fraction_of_positives - mean_predicted))
    print(f"Calibration ECE: {cal_error:.4f} ({'Good' if cal_error < 0.05 else 'Needs work'})")
    
    # ─── 7. THRESHOLD TUNING ───
    print("\n=== THRESHOLD ANALYSIS (Investigation Capacity: {}) ===".format(investigation_capacity))
    
    # Method 1: Fixed capacity threshold
    n_test = len(y_test)
    capacity_threshold = np.percentile(y_proba, (1 - investigation_capacity/n_test) * 100)
    y_pred_capacity = (y_proba >= capacity_threshold).astype(int)
    
    tp = ((y_pred_capacity == 1) & (y_test == 1)).sum()
    fp = ((y_pred_capacity == 1) & (y_test == 0)).sum()
    fn = ((y_pred_capacity == 0) & (y_test == 1)).sum()
    
    print(f"At capacity {investigation_capacity}/day:")
    print(f"  Precision: {tp/(tp+fp+1e-8):.3f}")
    print(f"  Recall:    {tp/(tp+fn+1e-8):.3f}")
    print(f"  F1:        {2*tp/(2*tp+fp+fn+1e-8):.3f}")
    
    # Method 2: PR curve threshold sweep
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    print("\nPR curve operating points:")
    for min_precision in [0.5, 0.6, 0.7, 0.8]:
        mask = precisions[:-1] >= min_precision
        if mask.any():
            best_recall = recalls[:-1][mask].max()
            best_thresh = thresholds[mask][recalls[:-1][mask].argmax()]
            flags = (y_proba >= best_thresh).sum()
            print(f"  Precision≥{min_precision}: "
                  f"Recall={best_recall:.3f}, Threshold={best_thresh:.3f}, "
                  f"Flags={flags} ({flags/n_test*100:.1f}%)")
    
    # ─── 8. SHAP INTERPRETABILITY ───
    print("\n=== FEATURE IMPORTANCE (SHAP) ===")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # Positive class for binary
    else:
        shap_vals = shap_values
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'shap_mean_abs': np.abs(shap_vals).mean(axis=0)
    }).sort_values('shap_mean_abs', ascending=False)
    
    print("Top 10 features by SHAP:")
    print(feature_importance.head(10).to_string())
    
    # SHAP summary plot
    shap.summary_plot(shap_vals, X_test, feature_names=feature_cols, 
                     max_display=15, show=False)
    plt.tight_layout()
    plt.show()
    
    # ─── 9. ERROR ANALYSIS ───
    print("\n=== ERROR ANALYSIS ===")
    
    # False Negatives (missed fraud) — business-critical
    fn_mask = (y_pred_capacity == 0) & (y_test == 1)
    fn_df = test_df[fn_mask.values]
    print(f"False Negatives (missed fraud): {fn_mask.sum()}")
    if len(fn_df) > 0:
        print("FN characteristics (vs. all fraud):")
        fraud_df = test_df[y_test == 1]
        for col in ['amount', 'days_as_customer', 'txn_seq_num'][:3]:
            if col in fn_df.columns:
                print(f"  {col}: FN mean={fn_df[col].mean():.2f} "
                      f"vs All Fraud mean={fraud_df[col].mean():.2f}")
    
    return {
        'model': calibrated_model,
        'base_model': model,
        'explainer': explainer,
        'shap_values': shap_vals,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'ks_stat': ks_stat,
        'best_params': best_params,
        'y_proba': y_proba,
        'threshold': capacity_threshold,
        'feature_importance': feature_importance
    }
```

---

## ═══════════════════════════════════════
## SECTION 4: LIVE CODING CHALLENGE PROBLEMS
## ═══════════════════════════════════════

### Challenge 1: Return Fraud Velocity Feature (Common Flipkart Problem)

**Problem:** Given a log of return events, compute for each return event:
- Number of returns by this user in last 30 days
- Total return value in last 30 days
- Return rate (returns/orders) for user in last 90 days

```python
import pandas as pd
from typing import Optional

def compute_return_velocity_features(
    returns_df: pd.DataFrame,    # user_id, return_date, return_amount
    orders_df: pd.DataFrame,     # user_id, order_date, order_amount
    window_days: int = 30
) -> pd.DataFrame:
    """
    Compute return velocity features for each return event.
    KEY: Point-in-time correct — only use information available AT the time of return.
    
    Naive (WRONG) approach: compute over entire history including future returns.
    Correct approach: for each return at time t, only count returns before time t.
    """
    
    returns_df = returns_df.copy()
    returns_df['return_date'] = pd.to_datetime(returns_df['return_date'])
    returns_df = returns_df.sort_values(['user_id', 'return_date'])
    
    # ─── Self-join approach (brute force, correct) ───
    # For each return event, look back window_days
    result_rows = []
    
    for idx, row in returns_df.iterrows():
        user_id = row['user_id']
        event_date = row['return_date']
        window_start = event_date - pd.Timedelta(days=window_days)
        
        # Prior returns (EXCLUDING current event)
        prior_returns = returns_df[
            (returns_df['user_id'] == user_id) &
            (returns_df['return_date'] < event_date) &  # Strictly before
            (returns_df['return_date'] >= window_start)
        ]
        
        # Prior orders for return rate calculation (90-day window)
        prior_orders = orders_df[
            (orders_df['user_id'] == user_id) &
            (orders_df['order_date'] < event_date) &
            (orders_df['order_date'] >= event_date - pd.Timedelta(days=90))
        ]
        
        result_rows.append({
            'user_id': user_id,
            'return_date': event_date,
            'return_amount': row['return_amount'],
            f'return_count_{window_days}d': len(prior_returns),
            f'return_value_{window_days}d': prior_returns['return_amount'].sum(),
            'order_count_90d': len(prior_orders),
            'return_rate_90d': (
                len(prior_returns) / max(len(prior_orders), 1)
            ),
        })
    
    return pd.DataFrame(result_rows)

# OPTIMIZED VERSION (using merge_asof for production scale)
def compute_return_velocity_fast(
    returns_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    window_days: int = 30
) -> pd.DataFrame:
    """
    Vectorized approach using merge_asof + cumulative sums.
    O(n log n) vs O(n²) for naive approach.
    """
    returns_df = returns_df.sort_values(['user_id', 'return_date'])
    
    # Add cumulative sum per user (expanding window)
    returns_df['cumsum_count'] = returns_df.groupby('user_id').cumcount()
    returns_df['cumsum_value'] = (returns_df.groupby('user_id')['return_amount']
                                             .cumsum())
    
    # For the window calculation, use the fact that:
    # count in window = cumsum(now) - cumsum(window_start)
    # This requires an efficient lookup of "cumsum at window_start"
    # → Use merge_asof for this
    
    # Create a "window start" lookup table
    returns_left = returns_df[['user_id', 'return_date', 'cumsum_count', 'cumsum_value']].copy()
    returns_right = returns_df[['user_id', 'return_date', 'cumsum_count', 'cumsum_value']].copy()
    returns_right.columns = ['user_id', 'window_start_date', 'past_count', 'past_value']
    
    returns_left['window_start'] = returns_left['return_date'] - pd.Timedelta(days=window_days)
    
    merged = pd.merge_asof(
        returns_left.sort_values('return_date'),
        returns_right.sort_values('window_start_date'),
        left_by='user_id', right_by='user_id',
        left_on='window_start', right_on='window_start_date',
        direction='backward'
    )
    
    merged[f'return_count_{window_days}d'] = (
        merged['cumsum_count'] - merged['past_count'].fillna(0)
    )
    merged[f'return_value_{window_days}d'] = (
        merged['cumsum_value'] - merged['past_value'].fillna(0)
    )
    
    return merged
```

---

### Challenge 2: Implement KS Statistic from Scratch

```python
def compute_ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    Compute KS Statistic for binary classifier.
    Industry standard for credit risk model validation.
    
    Returns: KS stat, optimal threshold, and the KS table.
    """
    # Sort by score descending
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_score_sorted = y_score[sorted_indices]
    
    n_total = len(y_true)
    n_bad = y_true.sum()
    n_good = n_total - n_bad
    
    ks_table = []
    cum_bad = 0
    cum_good = 0
    
    for i in range(n_total):
        if y_true_sorted[i] == 1:
            cum_bad += 1
        else:
            cum_good += 1
        
        cum_bad_rate = cum_bad / n_bad
        cum_good_rate = cum_good / n_good
        ks = abs(cum_bad_rate - cum_good_rate)
        
        ks_table.append({
            'score_threshold': y_score_sorted[i],
            'cum_bad_rate': cum_bad_rate,
            'cum_good_rate': cum_good_rate,
            'ks': ks
        })
    
    ks_df = pd.DataFrame(ks_table)
    max_ks_idx = ks_df['ks'].idxmax()
    
    return {
        'ks_statistic': ks_df['ks'].max(),
        'optimal_threshold': ks_df.loc[max_ks_idx, 'score_threshold'],
        'gini': 2 * roc_auc_score(y_true, y_score) - 1,
        'ks_table': ks_df
    }

# Test it
np.random.seed(42)
y_true = np.random.binomial(1, 0.05, 10000)  # 5% fraud rate
y_score = np.random.beta(0.3, 2, 10000) * y_true + np.random.beta(2, 5, 10000) * (1-y_true)

result = compute_ks_statistic(y_true, y_score)
print(f"KS Stat: {result['ks_statistic']:.4f}")
print(f"Optimal threshold: {result['optimal_threshold']:.4f}")
print(f"Gini: {result['gini']:.4f}")
```

---

### Challenge 3: Decile Lift Chart (Standard Interview Question)

```python
def create_decile_lift_table(
    y_true: np.ndarray, 
    y_score: np.ndarray,
    n_deciles: int = 10
) -> pd.DataFrame:
    """
    Create decile lift table — standard business reporting metric.
    Shows: if we investigate top X%, what % of fraud do we capture?
    """
    df = pd.DataFrame({'score': y_score, 'label': y_true})
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    df['decile'] = pd.qcut(df.index, n_deciles, labels=range(1, n_deciles+1))
    
    total_fraud = y_true.sum()
    total_n = len(y_true)
    overall_fraud_rate = y_true.mean()
    
    results = []
    cum_fraud = 0
    cum_n = 0
    
    for decile in range(1, n_deciles + 1):
        decile_df = df[df['decile'] == decile]
        n = len(decile_df)
        n_fraud = decile_df['label'].sum()
        fraud_rate = n_fraud / n
        
        cum_fraud += n_fraud
        cum_n += n
        
        results.append({
            'decile': decile,
            'n': n,
            'n_fraud': int(n_fraud),
            'fraud_rate': round(fraud_rate, 4),
            'lift': round(fraud_rate / overall_fraud_rate, 2),  # vs. random
            'cum_fraud_pct': round(cum_fraud / total_fraud * 100, 1),
            'cum_population_pct': round(cum_n / total_n * 100, 1),
        })
    
    lift_df = pd.DataFrame(results)
    print(lift_df.to_string())
    print(f"\nTop decile captures {lift_df.iloc[0]['cum_fraud_pct']}% of all fraud")
    print(f"Top decile lift: {lift_df.iloc[0]['lift']}x vs. random")
    
    return lift_df
```

---

### Challenge 4: Implement NDCG@k from Scratch

```python
def compute_ndcg_at_k(
    y_true_relevance: list,  # relevance scores for each item
    y_ranked: list,          # predicted ranking (item indices, most relevant first)
    k: int
) -> float:
    """
    Compute NDCG@k for recommendation evaluation.
    
    Args:
        y_true_relevance: dict or list of true relevance scores per item
        y_ranked: list of item indices in predicted order
    """
    def dcg_at_k(ranked_items, relevance, k):
        dcg = 0
        for i, item in enumerate(ranked_items[:k]):
            rel = relevance.get(item, 0) if isinstance(relevance, dict) else relevance[item]
            dcg += (2**rel - 1) / np.log2(i + 2)  # +2 because i is 0-indexed
        return dcg
    
    # Actual DCG
    actual_dcg = dcg_at_k(y_ranked, y_true_relevance, k)
    
    # Ideal DCG (items ranked in perfect order)
    if isinstance(y_true_relevance, dict):
        ideal_order = sorted(y_true_relevance.keys(), 
                            key=lambda x: y_true_relevance[x], reverse=True)
        ideal_dcg = dcg_at_k(ideal_order, y_true_relevance, k)
    else:
        ideal_order = np.argsort(y_true_relevance)[::-1]
        ideal_dcg = dcg_at_k(ideal_order, y_true_relevance, k)
    
    if ideal_dcg == 0:
        return 0
    
    return actual_dcg / ideal_dcg

# Example with graded relevance (0=not relevant, 1=relevant, 2=highly relevant)
relevant_items = {
    'item_1': 2,  # highly relevant
    'item_2': 1,  # relevant
    'item_3': 0,  # not relevant
    'item_4': 2,  # highly relevant
    'item_5': 1,  # relevant
}
predicted_order = ['item_3', 'item_1', 'item_4', 'item_2', 'item_5']

ndcg = compute_ndcg_at_k(relevant_items, predicted_order, k=5)
print(f"NDCG@5: {ndcg:.4f}")
```

---

### Challenge 5: Graph-Based Fraud Ring Detection (BFS/Union-Find)

```python
from collections import defaultdict, deque

class FraudRingDetector:
    """
    Detect fraud rings using Union-Find with fraud propagation.
    Time complexity: O(α(n)) per union/find ≈ O(1) amortized.
    """
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.fraud_in_component = {}  # Is there any known fraudster in component?
    
    def add_user(self, user_id: str, is_fraud: bool = False):
        if user_id not in self.parent:
            self.parent[user_id] = user_id
            self.rank[user_id] = 0
            self.fraud_in_component[user_id] = is_fraud
    
    def find(self, x: str) -> str:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: str, y: str):
        """Union by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        # Propagate fraud signal
        self.fraud_in_component[px] = (
            self.fraud_in_component[px] or self.fraud_in_component[py]
        )
    
    def is_in_fraud_ring(self, user_id: str) -> bool:
        """Check if user belongs to a component with any known fraudster."""
        root = self.find(user_id)
        return self.fraud_in_component[root]
    
    def get_all_fraud_rings(self) -> dict:
        """Get all components containing at least one fraudster."""
        components = defaultdict(list)
        for user_id in self.parent:
            root = self.find(user_id)
            components[root].append(user_id)
        
        fraud_rings = {
            root: members 
            for root, members in components.items()
            if self.fraud_in_component[root] and len(members) > 1
        }
        return fraud_rings

# Usage
detector = FraudRingDetector()

# Add users
users = {'u1': True, 'u2': False, 'u3': False, 'u4': True, 'u5': False}
for uid, is_fraud in users.items():
    detector.add_user(uid, is_fraud)

# Connect users who share devices/addresses
connections = [('u1', 'u2'), ('u2', 'u3'), ('u4', 'u5')]
for u, v in connections:
    detector.union(u, v)

# Check which users are in fraud rings
for uid in users:
    print(f"{uid}: In fraud ring = {detector.is_in_fraud_ring(uid)}")

print("\nFraud rings found:")
for ring_id, members in detector.get_all_fraud_rings().items():
    print(f"  Ring {ring_id}: {members}")
```

---

## ═══════════════════════════════════════
## SECTION 5: ADVANCED SQL PATTERNS
## ═══════════════════════════════════════

```sql
-- ─── PATTERN 1: USER RETURN FRAUD VELOCITY ───
-- For each return event: count returns in last 30 days
WITH return_events AS (
    SELECT 
        user_id,
        return_date,
        return_amount,
        order_id,
        return_reason
    FROM returns
    WHERE return_date >= DATE_SUB('2025-01-01', INTERVAL 1 YEAR)
),
return_velocity AS (
    SELECT 
        r1.user_id,
        r1.return_date,
        r1.return_amount,
        -- Count prior returns in window (NOT including current)
        COUNT(r2.order_id) AS return_count_30d,
        SUM(r2.return_amount) AS return_value_30d,
        MAX(r2.return_date) AS last_return_date
    FROM return_events r1
    LEFT JOIN return_events r2 
        ON r1.user_id = r2.user_id
        AND r2.return_date < r1.return_date           -- Strictly before
        AND r2.return_date >= DATE_SUB(r1.return_date, INTERVAL 30 DAY)
    GROUP BY r1.user_id, r1.return_date, r1.return_amount
)
SELECT * FROM return_velocity ORDER BY return_count_30d DESC;

-- ─── PATTERN 2: FIRST ORDER COHORT ANALYSIS ───
WITH first_orders AS (
    SELECT 
        user_id,
        MIN(order_date) AS first_order_date,
        DATE_FORMAT(MIN(order_date), '%Y-%m') AS cohort_month
    FROM orders
    GROUP BY user_id
),
cohort_activity AS (
    SELECT 
        f.user_id,
        f.cohort_month,
        TIMESTAMPDIFF(MONTH, f.first_order_date, o.order_date) AS months_after_first,
        o.gmv
    FROM first_orders f
    JOIN orders o ON f.user_id = o.user_id
)
SELECT 
    cohort_month,
    COUNT(DISTINCT CASE WHEN months_after_first = 0 THEN user_id END) AS m0_users,
    COUNT(DISTINCT CASE WHEN months_after_first = 1 THEN user_id END) AS m1_users,
    COUNT(DISTINCT CASE WHEN months_after_first = 3 THEN user_id END) AS m3_users,
    ROUND(
        100.0 * COUNT(DISTINCT CASE WHEN months_after_first = 1 THEN user_id END) /
        NULLIF(COUNT(DISTINCT CASE WHEN months_after_first = 0 THEN user_id END), 0), 1
    ) AS m1_retention_pct
FROM cohort_activity
GROUP BY cohort_month
ORDER BY cohort_month;

-- ─── PATTERN 3: SELLER FRAUD RING DETECTION VIA SQL ───
WITH seller_shared_signals AS (
    -- Sellers sharing ANY of: device, bank account, or address
    SELECT 
        a.seller_id AS seller_a,
        b.seller_id AS seller_b,
        COUNT(DISTINCT CASE WHEN a.device_id = b.device_id THEN a.device_id END) AS shared_devices,
        COUNT(DISTINCT CASE WHEN a.bank_account = b.bank_account THEN a.bank_account END) AS shared_banks,
        COUNT(DISTINCT CASE WHEN a.address_hash = b.address_hash THEN a.address_hash END) AS shared_addresses
    FROM seller_signals a
    JOIN seller_signals b ON a.seller_id < b.seller_id  -- Avoid duplicates
    GROUP BY a.seller_id, b.seller_id
    HAVING shared_devices + shared_banks + shared_addresses >= 2  -- Require ≥2 signals
),
ring_summary AS (
    SELECT 
        s.*,
        f1.fraud_status AS seller_a_status,
        f2.fraud_status AS seller_b_status,
        f1.gmv_last_90d AS seller_a_gmv,
        f2.gmv_last_90d AS seller_b_gmv
    FROM seller_shared_signals s
    LEFT JOIN seller_info f1 ON s.seller_a = f1.seller_id
    LEFT JOIN seller_info f2 ON s.seller_b = f2.seller_id
)
SELECT *,
       CASE 
           WHEN seller_a_status = 'FLAGGED' OR seller_b_status = 'FLAGGED' 
           THEN 'HIGH_RISK_RING'
           WHEN shared_devices >= 2 AND shared_banks >= 1 
           THEN 'MEDIUM_RISK_RING'
           ELSE 'MONITOR'
       END AS risk_level
FROM ring_summary
ORDER BY (shared_devices + shared_banks + shared_addresses) DESC;

-- ─── PATTERN 4: RUNNING PERCENTILE FOR FRAUD SCORING ───
WITH user_daily_spend AS (
    SELECT 
        user_id,
        DATE(txn_timestamp) AS txn_date,
        SUM(amount) AS daily_spend
    FROM transactions
    GROUP BY user_id, DATE(txn_timestamp)
),
user_spend_percentile AS (
    SELECT 
        user_id,
        txn_date,
        daily_spend,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY daily_spend) 
            OVER (PARTITION BY user_id ROWS BETWEEN 90 PRECEDING AND 1 PRECEDING) AS p50_spend_90d,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY daily_spend) 
            OVER (PARTITION BY user_id ROWS BETWEEN 90 PRECEDING AND 1 PRECEDING) AS p95_spend_90d
    FROM user_daily_spend
)
SELECT 
    *,
    CASE 
        WHEN daily_spend > p95_spend_90d * 2 THEN 'EXTREME_ANOMALY'
        WHEN daily_spend > p95_spend_90d THEN 'HIGH_ANOMALY'
        ELSE 'NORMAL'
    END AS anomaly_level
FROM user_spend_percentile
ORDER BY daily_spend / NULLIF(p50_spend_90d, 0) DESC;
```

---

*This document contains everything needed for the Hands-On round: EDA protocol, feature engineering, model training, coding challenges, and SQL patterns.*
