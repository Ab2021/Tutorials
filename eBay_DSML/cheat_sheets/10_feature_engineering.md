# eBay DS/ML — Feature Engineering & Data Pipeline Patterns

> Feature engineering is often the highest-ROI activity in DS.
> eBay interviewers care about HOW you engineer features, not just which model you pick.

---

## 🛠️ Part 1: Feature Engineering Techniques

### 1.1 Numerical Features

| Technique | When to Use | Code/Example |
|---|---|---|
| **StandardScaler** | SVM, KNN, neural nets, PCA | `(x - mean) / std` |
| **MinMaxScaler** | Fixed range [0,1] needed | `(x - min) / (max - min)` |
| **RobustScaler** | Many outliers | `(x - median) / IQR` |
| **Log transform** | Right-skewed (prices, income) | `np.log1p(x)` |
| **Box-Cox** | Generalized power transform | `scipy.stats.boxcox(x)` |
| **Binning** | Capture non-linear relationships | `pd.cut(age, bins=[0,18,35,55,100])` |
| **Polynomial** | Feature interactions | `price * quantity`, `price²` |
| **Clipping** | Outlier handling | `np.clip(x, q01, q99)` |

### 1.2 Categorical Features

| Technique | Best For | Pitfall |
|---|---|---|
| **One-Hot** | Low cardinality (<20 categories) | Dimensionality explosion |
| **Label/Ordinal** | Ordinal data (size: S<M<L) | Implies false ordering for nominal |
| **Target Encoding** | High cardinality (zip code, seller_id) | Data leakage without cross-val |
| **Frequency Encoding** | When category frequency matters | Loses category identity |
| **Hash Encoding** | Very high cardinality | Collisions, loses interpretability |
| **Embedding** | Deep learning, NLP | Needs sufficient training data |
| **Binary Encoding** | Medium cardinality | Good compromise one-hot vs label |

### 1.3 Text Features

| Technique | Output | Best For |
|---|---|---|
| **Bag of Words** | Sparse count matrix | Simple classification |
| **TF-IDF** | Weighted sparse matrix | Search relevance, document similarity |
| **Word2Vec/GloVe** | Dense vectors (static) | Feature input to ML models |
| **FastText** | Dense vectors (subword) | Handles typos, rare words |
| **BERT/eBERT** | Contextual embeddings | Semantic search, listing understanding |
| **Sentence Transformers** | Sentence-level embeddings | Cross-listing similarity |

### 1.4 Time/Date Features

```python
# From a single timestamp, create rich features:
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
df['month'] = df['timestamp'].dt.month
df['is_holiday'] = df['date'].isin(holiday_list).astype(int)
df['days_since_signup'] = (df['event_date'] - df['signup_date']).dt.days
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)  # cyclical encoding
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

### 1.5 Interaction & Domain-Specific Features

**For eBay marketplace:**
```
# User behavior features
user_avg_spend_30d = transactions.groupby('buyer_id').rolling(30d).mean()
user_category_diversity = nunique(categories) / total_orders
user_session_depth = page_views_per_session
user_time_since_last_purchase = now - last_purchase_date
user_price_sensitivity = avg(purchased_price) / avg(viewed_price)

# Item features
item_price_vs_category_median = item_price / category_median_price
item_days_since_listed = now - listing_date
item_seller_rating = seller_avg_rating
item_title_length = len(title.split())
item_has_image = 1 if image_url else 0
item_price_history_trend = price_7d_ago - current_price

# Cross features (user × item)
user_category_affinity = user's purchase count in this category
user_price_match = |item_price - user_avg_purchase_price| / user_avg_purchase_price
user_seller_history = has user bought from this seller before?
```

---

## 🔄 Part 2: Data Pipeline Anti-Patterns

### Anti-Pattern 1: Data Leakage

**WRONG (leak test info into training):**
```python
# ❌ Fit scaler on ALL data before splitting
scaler.fit_transform(X)
X_train, X_test = train_test_split(X)

# ❌ Use future data as features
df['next_month_purchases'] = df.groupby('user')['purchases'].shift(-1)

# ❌ Target encoding without cross-validation
df['category_target_mean'] = df.groupby('category')['target'].transform('mean')
```

**CORRECT:**
```python
# ✅ Split FIRST, then fit on train only
X_train, X_test = train_test_split(X)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Only use past data for features
df['prev_month_purchases'] = df.groupby('user')['purchases'].shift(1)

# ✅ Target encoding with cross-validation
from category_encoders import TargetEncoder
encoder = TargetEncoder(cols=['category'])
# fit only on training folds
```

### Anti-Pattern 2: Not Handling Missing Values Properly

```python
# ❌ Drop all rows with any NaN (lose data)
df.dropna()

# ❌ Fill with global mean (leaks test info)
df['price'].fillna(df['price'].mean())

# ✅ For numeric: median (robust to outliers), fit on train
train_median = X_train['price'].median()
X_train['price'].fillna(train_median)
X_test['price'].fillna(train_median)

# ✅ For categorical: mode or 'Unknown' category
X_train['category'].fillna('Unknown')

# ✅ Add missingness indicator
X_train['price_is_missing'] = X_train['price'].isnull().astype(int)
```

### Anti-Pattern 3: Treating All Features Equally

```python
# ❌ Throwing everything into the model without analysis

# ✅ Feature selection pipeline:
# 1. Remove zero-variance features
# 2. Remove highly correlated features (>0.95)
# 3. Use tree-based feature importance for initial filtering
# 4. Use SHAP for final feature set + interpretability
# 5. Monitor feature drift in production
```

---

## 📐 Part 3: Feature Selection Methods

| Method | Type | How It Works | When to Use |
|---|---|---|---|
| **Correlation filter** | Filter | Remove features correlated >0.95 with another | Quick cleanup |
| **Variance threshold** | Filter | Remove features with near-zero variance | Preprocessing |
| **Chi-squared test** | Filter | Statistical test for categorical features | Classification |
| **Mutual Information** | Filter | Non-linear dependency measure | Any |
| **RFE (Recursive Feature Elimination)** | Wrapper | Train model, remove least important, repeat | When accuracy is critical |
| **L1 regularization (Lasso)** | Embedded | Zero out unimportant weights | Linear/logistic regression |
| **Tree-based importance** | Embedded | Split importance from GBDT/RF | Tabular data |
| **SHAP** | Model-agnostic | Game-theory-based feature attribution | Explainability + selection |
| **Permutation importance** | Model-agnostic | Shuffle feature, measure accuracy drop | Reliable but slow |
| **Boruta** | Wrapper | Compare feature importance to shadow features | Comprehensive selection |

---

## 🏗️ Part 4: Production Data Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│ Data Sources │    │  Processing  │    │   Storage    │
│              │    │              │    │              │
│ eBay Events  │───►│ Apache Spark │───►│ Data Lake    │
│ User Actions │    │ (batch ETL)  │    │ (Parquet/    │
│ Transactions │    │              │    │  Delta Lake) │
│ Listings     │    │ Apache Flink │───►│              │
│              │    │ (streaming)  │    │ Feature Store│
└─────────────┘    └──────────────┘    │ (Redis/NuKV) │
                                       └──────┬───────┘
                                              │
                           ┌──────────────────┴──────────────────┐
                           │                                     │
                    ┌──────▼───────┐                   ┌─────────▼────────┐
                    │   Training   │                   │   Serving        │
                    │ (Krylov GPU  │                   │ (Real-time API)  │
                    │  cluster)    │                   │                  │
                    │ PyTorch /    │                   │ Feature lookup + │
                    │ XGBoost     │                   │ Model inference  │
                    └──────────────┘                   └──────────────────┘
```

### Key Principles:
1. **Same feature code** for training and serving (via Feature Store)
2. **Point-in-time correctness** — no future data leakage in training
3. **Versioned data** — track which data trained which model
4. **Monitoring** — alert on feature distribution drift
5. **Idempotent pipelines** — safe to re-run without side effects

---

## ❓ Part 5: 10 Interview Questions

1. *"How would you engineer features for a search ranking model?"*
2. *"What is target encoding and how do you prevent leakage?"*
3. *"When would you use log transformation? Give an eBay example."*
4. *"How do you handle a categorical feature with 10,000 unique values?"*
5. *"Explain the difference between StandardScaler and RobustScaler."*
6. *"What time-based features would you create for purchase prediction?"*
7. *"How do you detect and handle data leakage in a pipeline?"*
8. *"What is a Feature Store? Why does eBay need one?"*
9. *"How would you create user-item interaction features for recommendations?"*
10. *"Walk me through your approach to feature selection for a new model."*
