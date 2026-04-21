# 💻 Round 3: Hands-On (HO) — Master Guide
### Flipkart Senior Data Scientist | Coding & Implementation Round

> **Round Format:** 60-90 min | Live coding or take-home Jupyter Notebook
> **What They Test:** Real implementation skills, EDA quality, feature engineering creativity, model building, code quality, communication of thought process
> **Your Winning Strategy:** Narrate your thinking aloud | Write clean modular code | Document your WHY not just WHAT | Always connect to business context

---

## 🎯 WHAT THIS ROUND IS REALLY ABOUT

The HO round provides a **dataset + business problem** and expects you to work through it live. You will be evaluated on:
1. **EDA quality** — do you find the right insights, not just run `.describe()`?
2. **Feature engineering creativity** — do you engineer domain-relevant features?
3. **Model selection judgment** — can you justify your choice?
4. **Code quality** — modular, readable, production-forward
5. **Communication** — can you narrate your thought process coherently?
6. **Business orientation** — do you always connect technical choices to business outcomes?

---

## 🗺️ THE END-TO-END HO EXECUTION PLAYBOOK

```
STEP 1: PROBLEM DECOMPOSITION (5 min)
├── Read problem statement carefully — what exactly is being predicted?
├── Identify: Classification vs. Regression vs. Ranking vs. Clustering?
├── What is the success metric? (Ask if not given)
└── What constraints? (Latency, interpretability, fairness)

STEP 2: DATA LOADING & INITIAL INSPECTION (10 min)
├── Shape, dtypes, memory usage
├── Missing values (absolute count + percentage)
├── Target variable distribution (class imbalance?)
├── Sample rows — spot obvious data quality issues
└── Temporal structure? (Don't randomly split time series!)

STEP 3: EXPLORATORY DATA ANALYSIS (15 min)
├── Univariate: distributions of key features
├── Bivariate: feature vs. target (KDE plots, box plots)
├── Correlation matrix (tabular) or embedding visualization (text)
├── Outlier detection (IQR, z-score)
└── Key business insights — what patterns tell the story?

STEP 4: FEATURE ENGINEERING (15 min)
├── Domain-specific features (velocity, ratios, temporal aggregations)
├── Interaction features (if tree-based: less important, but still valuable)
├── Encoding: target encoding for high-cardinality, ordinal for ordered
├── Missing value imputation: strategy and why
└── Feature importance sanity check (correlation with target)

STEP 5: MODELING (15 min)
├── Baseline first (logistic regression / simple model)
├── Main model (XGBoost / LGB with cross-validation)
├── Hyperparameter search (RandomizedSearch budget)
├── Class imbalance handling (scale_pos_weight or SMOTE)
└── Final evaluation on holdout set

STEP 6: EVALUATION & INTERPRETATION (10 min)
├── Primary metric + all relevant evaluation metrics
├── Confusion matrix + threshold analysis
├── SHAP / feature importance — which features matter?
├── Error analysis: where does the model fail?
└── Business impact estimation

STEP 7: CONCLUSION & NEXT STEPS (5 min)
├── Key findings summary
├── Model limitations and risks
├── Production deployment considerations
└── What you'd do with more time/data
```

---

## 🔥 CORE CODE PATTERNS TO MASTER

### Pattern 1: Production-Grade EDA Template

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def comprehensive_eda(df: pd.DataFrame, target_col: str) -> dict:
    """
    Production-grade EDA function that returns a structured summary.
    Always start with this — shows systematic thinking.
    """
    report = {}
    
    # 1. Shape and memory
    print(f"Shape: {df.shape} | Memory: {df.memory_usage().sum() / 1024**2:.1f} MB")
    
    # 2. Missing value analysis (CRITICAL — always do this first)
    missing = pd.DataFrame({
        'count': df.isnull().sum(),
        'pct': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('pct', ascending=False)
    missing = missing[missing['count'] > 0]
    print("\nMissing Values:\n", missing)
    report['missing'] = missing
    
    # 3. Target variable distribution
    if df[target_col].dtype in ['int64', 'float64'] and df[target_col].nunique() <= 10:
        # Classification
        print(f"\nTarget Distribution:\n{df[target_col].value_counts(normalize=True)}")
        imbalance_ratio = df[target_col].value_counts().max() / df[target_col].value_counts().min()
        print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
        report['imbalance_ratio'] = imbalance_ratio
    else:
        # Regression
        print(f"\nTarget Stats:\n{df[target_col].describe()}")
        print(f"Skewness: {stats.skew(df[target_col].dropna()):.3f}")
    
    # 4. Numeric feature stats
    num_cols = df.select_dtypes(include=['number']).columns.drop(target_col, errors='ignore')
    print(f"\nNumeric features ({len(num_cols)}):")
    print(df[num_cols].describe().T.round(3))
    
    # 5. Categorical feature cardinality
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        n_unique = df[col].nunique()
        top_val = df[col].value_counts().index[0]
        print(f"{col}: {n_unique} unique | top: '{top_val}'")
    
    # 6. Correlation with target (numeric only)
    if df[target_col].dtype in ['int64', 'float64']:
        corr_with_target = df[num_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        print(f"\nTop correlations with target:\n{corr_with_target.head(10)}")
        report['top_correlations'] = corr_with_target
    
    return report
```

---

### Pattern 2: Fraud-Specific Feature Engineering

```python
def engineer_fraud_features(df: pd.DataFrame, 
                             user_col: str = 'user_id',
                             time_col: str = 'timestamp',
                             amount_col: str = 'amount') -> pd.DataFrame:
    """
    Domain-specific feature engineering for fraud detection.
    These features are what separate a mediocre model from a great one.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([user_col, time_col])
    
    # === VELOCITY FEATURES (fraud often spikes in short windows) ===
    for window in ['1H', '24H', '7D']:
        df[f'txn_count_{window}'] = (
            df.groupby(user_col)[time_col]
            .transform(lambda x: x.expanding().count())  # simplified; use rolling in prod
        )
        df[f'amount_sum_{window}'] = (
            df.groupby(user_col)[amount_col]
            .transform(lambda x: x.rolling(window, on=time_col).sum())
        )
    
    # === BEHAVIORAL ANOMALY FEATURES ===
    user_stats = df.groupby(user_col)[amount_col].agg(['mean', 'std']).rename(
        columns={'mean': 'user_avg_amount', 'std': 'user_std_amount'}
    )
    df = df.merge(user_stats, on=user_col, how='left')
    df['amount_zscore'] = (df[amount_col] - df['user_avg_amount']) / (df['user_std_amount'] + 1e-8)
    df['is_amount_outlier'] = (df['amount_zscore'].abs() > 3).astype(int)
    
    # === TEMPORAL FEATURES ===
    df['hour_of_day'] = df[time_col].dt.hour
    df['is_weekend'] = df[time_col].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_nighttime'] = df['hour_of_day'].between(22, 6).astype(int)  # 10pm - 6am
    
    # === RECENCY FEATURES ===
    df['days_since_first_txn'] = (
        df.groupby(user_col)[time_col]
        .transform(lambda x: (x - x.min()).dt.days)
    )
    df['txn_seq_number'] = df.groupby(user_col).cumcount() + 1
    
    # === RATIO FEATURES ===
    df['return_rate'] = df.get('returns', 0) / (df['txn_seq_number'] + 1)
    df['high_value_txn_ratio'] = (df[amount_col] > df['user_avg_amount'] * 2).astype(int)
    
    # === TIME BETWEEN TRANSACTIONS ===
    df['time_since_last_txn_hours'] = (
        df.groupby(user_col)[time_col]
        .transform(lambda x: x.diff().dt.total_seconds() / 3600)
    )
    df['is_rapid_succession'] = (df['time_since_last_txn_hours'] < 0.5).astype(int)
    
    return df
```

---

### Pattern 3: Production-Grade Model Training with Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import shap

def train_fraud_model(X_train, y_train, X_val, y_val, 
                      optimize: bool = True) -> dict:
    """
    Production-grade fraud model training pipeline.
    Includes: class imbalance handling, calibration, SHAP interpretability.
    """
    
    # Class imbalance — CRITICAL for fraud
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Class ratio: {scale_pos_weight:.1f}:1 | Applying scale_pos_weight")
    
    # Base model
    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',          # PR-AUC for imbalanced data!
        'scale_pos_weight': scale_pos_weight,
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,               # L1 for feature selection
        'reg_lambda': 1.0,              # L2 for weight regularization
        'random_state': 42,
        'n_jobs': -1
    }
    
    if optimize:
        # Hyperparameter search
        param_dist = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [200, 500, 1000],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            xgb.XGBClassifier(**base_params), 
            param_distributions=param_dist,
            n_iter=30,                   # Budget-conscious
            cv=cv, 
            scoring='average_precision',  # PR-AUC
            verbose=1,
            random_state=42
        )
        search.fit(X_train, y_train, 
                   eval_set=[(X_val, y_val)], 
                   early_stopping_rounds=50,
                   verbose=False)
        best_model = search.best_estimator_
        print(f"Best params: {search.best_params_}")
    else:
        best_model = xgb.XGBClassifier(**base_params)
        best_model.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=50, 
                      verbose=100)
    
    # Probability calibration — CRITICAL for fraud risk scores
    # Uncalibrated XGBoost probabilities are often overconfident
    calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_val, y_val)
    
    # Evaluation
    y_pred_proba = calibrated_model.predict_proba(X_val)[:, 1]
    auc_roc = roc_auc_score(y_val, y_pred_proba)
    auc_pr = average_precision_score(y_val, y_pred_proba)
    
    print(f"\n=== Evaluation Results ===")
    print(f"ROC-AUC:  {auc_roc:.4f}")
    print(f"PR-AUC:   {auc_pr:.4f}")
    
    # Business threshold analysis
    print("\n=== Threshold Analysis ===")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = (y_pred[y_val==1] == 1).mean() if y_pred.sum() > 0 else 0
        recall = (y_pred[y_val==1] == 1).mean()
        flags_per_100 = y_pred.mean() * 100
        print(f"Threshold {threshold:.1f}: Precision={precision:.2f}, Recall={recall:.2f}, Flags/100={flags_per_100:.1f}")
    
    # SHAP interpretability
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_val)
    
    return {
        'model': calibrated_model,
        'base_model': best_model,
        'explainer': explainer,
        'shap_values': shap_values,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'y_pred_proba': y_pred_proba
    }
```

---

### Pattern 4: RAG System Quick Implementation

```python
# Quick RAG implementation for the hands-on round
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def build_fraud_rag_system(documents: list[str], 
                           embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
                          ) -> RetrievalQA:
    """
    Minimal RAG system for fraud pattern retrieval.
    Production version would use: ChromaDB/Pinecone + domain-fine-tuned embeddings.
    """
    # Text splitting — chunk size matters for retrieval quality
    # Smaller chunks → higher precision, lower recall
    # Larger chunks → higher recall, context pollution risk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,      # ~128 tokens — balance for claims documents
        chunk_overlap=50,    # Overlap prevents context loss at chunk boundaries
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.create_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Embedding — domain-tuned is better, but MiniLM is fast for POC
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # For cosine similarity
    )
    
    # Vector store — FAISS for local, Pinecone for production scale
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Retriever with MMR (Maximal Marginal Relevance) — reduces redundancy
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    
    # QA chain with structured output prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),  # temperature=0 for determinism
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain
```

---

### Pattern 5: Time-Series Safe Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

def temporal_cross_validate(model, X: pd.DataFrame, y: pd.Series, 
                             n_splits: int = 5) -> dict:
    """
    CRITICAL: For fraud data with temporal structure, NEVER use random K-fold.
    Always use time-based splits to prevent data leakage.
    
    Why: Random split means you train on future data to predict past — impossible in production.
    """
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = {'auc_roc': [], 'auc_pr': [], 'precision_at_top1pct': []}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Evaluate
        auc = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)
        
        # Precision at top 1% (business-relevant: investigation capacity)
        threshold_1pct = np.percentile(y_proba, 99)
        top_1pct_mask = y_proba >= threshold_1pct
        prec_1pct = y_val[top_1pct_mask].mean() if top_1pct_mask.sum() > 0 else 0
        
        scores['auc_roc'].append(auc)
        scores['auc_pr'].append(pr_auc)
        scores['precision_at_top1pct'].append(prec_1pct)
        
        print(f"Fold {fold+1}: AUC-ROC={auc:.4f} | PR-AUC={pr_auc:.4f} | "
              f"Prec@Top1%={prec_1pct:.4f} | "
              f"Train size={len(X_train):,} | Val size={len(X_val):,}")
    
    print(f"\nCV Summary:")
    for metric, vals in scores.items():
        print(f"{metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    return scores
```

---

### Pattern 6: Advanced SQL Patterns for DS Interviews

```sql
-- === FRAUD VELOCITY DETECTION ===
-- Find users with sudden transaction spike (>3 sigma above their baseline)
WITH user_daily_stats AS (
    SELECT 
        user_id,
        DATE(txn_timestamp) AS txn_date,
        COUNT(*)            AS daily_txn_count,
        SUM(amount)         AS daily_amount,
        COUNT(DISTINCT merchant_id) AS distinct_merchants
    FROM transactions
    WHERE txn_timestamp >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
    GROUP BY user_id, DATE(txn_timestamp)
),
user_baseline AS (
    SELECT 
        user_id,
        AVG(daily_txn_count)     AS avg_daily_txns,
        STDDEV(daily_txn_count)  AS std_daily_txns,
        AVG(daily_amount)        AS avg_daily_amount,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY daily_txn_count) AS p95_txns
    FROM user_daily_stats
    WHERE txn_date < DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)  -- Use historical baseline
    GROUP BY user_id
)
SELECT 
    d.user_id,
    d.txn_date,
    d.daily_txn_count,
    b.avg_daily_txns,
    b.std_daily_txns,
    ROUND((d.daily_txn_count - b.avg_daily_txns) / NULLIF(b.std_daily_txns, 0), 2) AS zscore,
    CASE 
        WHEN d.daily_txn_count > b.avg_daily_txns + 3 * b.std_daily_txns THEN 'HIGH_ALERT'
        WHEN d.daily_txn_count > b.avg_daily_txns + 2 * b.std_daily_txns THEN 'MEDIUM_ALERT'
        ELSE 'NORMAL'
    END AS alert_level
FROM user_daily_stats d
JOIN user_baseline b USING (user_id)
WHERE d.txn_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
  AND d.daily_txn_count > b.avg_daily_txns + 2 * b.std_daily_txns
ORDER BY zscore DESC;

-- === COHORT RETENTION ANALYSIS (Common Flipkart question) ===
WITH first_order AS (
    SELECT user_id, MIN(DATE(order_timestamp)) AS cohort_date
    FROM orders
    GROUP BY user_id
),
monthly_activity AS (
    SELECT 
        f.user_id,
        f.cohort_date,
        DATE_DIFF(DATE(o.order_timestamp), f.cohort_date, MONTH) AS months_since_first
    FROM first_order f
    JOIN orders o USING (user_id)
)
SELECT
    cohort_date,
    COUNT(DISTINCT CASE WHEN months_since_first = 0 THEN user_id END)  AS month_0,
    COUNT(DISTINCT CASE WHEN months_since_first = 1 THEN user_id END)  AS month_1,
    COUNT(DISTINCT CASE WHEN months_since_first = 2 THEN user_id END)  AS month_2,
    COUNT(DISTINCT CASE WHEN months_since_first = 3 THEN user_id END)  AS month_3,
    -- Retention rates
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN months_since_first = 1 THEN user_id END) /
          NULLIF(COUNT(DISTINCT CASE WHEN months_since_first = 0 THEN user_id END), 0), 1) AS m1_retention_pct
FROM monthly_activity
GROUP BY cohort_date
ORDER BY cohort_date;

-- === SELLER FRAUD RING DETECTION (Graph-based in SQL) ===
-- Find sellers sharing device IDs — proxy for linked accounts
WITH seller_device_map AS (
    SELECT DISTINCT seller_id, device_fingerprint
    FROM seller_logins
    WHERE login_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
),
shared_devices AS (
    SELECT 
        a.seller_id AS seller_a,
        b.seller_id AS seller_b,
        a.device_fingerprint,
        COUNT(*) AS shared_device_count
    FROM seller_device_map a
    JOIN seller_device_map b 
        ON a.device_fingerprint = b.device_fingerprint 
        AND a.seller_id < b.seller_id  -- Avoid duplicates
    GROUP BY a.seller_id, b.seller_id, a.device_fingerprint
)
SELECT 
    seller_a,
    seller_b,
    device_fingerprint,
    shared_device_count,
    -- Join to fraud labels
    f1.is_flagged AS seller_a_flagged,
    f2.is_flagged AS seller_b_flagged
FROM shared_devices
LEFT JOIN seller_fraud_flags f1 ON seller_a = f1.seller_id
LEFT JOIN seller_fraud_flags f2 ON seller_b = f2.seller_id
WHERE shared_device_count >= 2
ORDER BY shared_device_count DESC;
```

---

### Pattern 7: LeetCode-Style Python Problems (Fraud/DS Flavor)

```python
# ===== PROBLEM 1: Sliding Window Max Fraud Velocity =====
# Given a list of (timestamp, amount, is_fraud) sorted by time,
# find max fraud count in any 60-minute window.
# O(n) solution using deque

from collections import deque

def max_fraud_in_window(events: list[tuple], window_minutes: int = 60) -> int:
    """
    Sliding window for velocity-based fraud detection.
    Classic fraud interview question at Flipkart/Paytm.
    """
    queue = deque()  # Stores indices where is_fraud=True
    max_fraud = 0
    left = 0
    
    for right, (timestamp, amount, is_fraud) in enumerate(events):
        # Remove events outside window
        while queue and events[queue[0]][0] < timestamp - window_minutes * 60:
            queue.popleft()
        
        if is_fraud:
            queue.append(right)
        
        max_fraud = max(max_fraud, len(queue))
    
    return max_fraud

# ===== PROBLEM 2: Connected Components for Fraud Ring Detection =====
# Given a graph of user connections (same device/address), find all fraud rings.
# Union-Find (Disjoint Set) O(α(n)) ≈ O(1) per operation

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def find_fraud_rings(connections: list[tuple], fraud_labels: dict) -> list[set]:
    """
    Find connected components that contain at least one known fraudster.
    
    Args:
        connections: list of (user_a, user_b) pairs sharing device/address
        fraud_labels: dict of {user_id: is_fraud}
    
    Returns: list of fraud rings (sets of connected user_ids)
    """
    all_users = set()
    for a, b in connections:
        all_users.update([a, b])
    
    user_to_idx = {u: i for i, u in enumerate(sorted(all_users))}
    uf = UnionFind(len(all_users))
    
    for a, b in connections:
        uf.union(user_to_idx[a], user_to_idx[b])
    
    # Group users by component
    from collections import defaultdict
    components = defaultdict(set)
    for user in all_users:
        root = uf.find(user_to_idx[user])
        components[root].add(user)
    
    # Return components containing at least one fraudster
    fraud_rings = [
        comp for comp in components.values()
        if any(fraud_labels.get(u, False) for u in comp)
    ]
    
    return fraud_rings

# ===== PROBLEM 3: Top-K Sellers by Revenue with Heap =====
import heapq

def top_k_sellers_by_revenue(orders: list[dict], k: int) -> list[tuple]:
    """
    Find top-k sellers by total revenue. O(n log k) using min-heap.
    Better than sorting all: O(n log n).
    """
    seller_revenue = {}
    for order in orders:
        sid = order['seller_id']
        seller_revenue[sid] = seller_revenue.get(sid, 0) + order['amount']
    
    # Min-heap of size k
    heap = []
    for seller_id, revenue in seller_revenue.items():
        heapq.heappush(heap, (revenue, seller_id))
        if len(heap) > k:
            heapq.heappop(heap)  # Remove smallest
    
    return sorted(heap, reverse=True)  # Return in descending order
```

---

## 🔥 EXPECTED HANDS-ON PROBLEM TYPES AT FLIPKART

### Type 1: Fraud Classification Dataset
- Dataset: ~100K transactions, ~1% fraud rate, 20-30 features
- Expected: Full EDA → Feature Engineering → XGBoost → Threshold Analysis → Business Recommendation
- Differentiator: Velocity features from timestamp data, decile lift chart, calibration curve

### Type 2: Credit Risk Scoring (EMI/Pay Later)
- Dataset: Applicant features + bureau data + behavioral data, binary default label
- Expected: Scorecard-style model, KS statistic, Gini coefficient, vintage analysis simulation
- Differentiator: Interpret scorecard weights as business "points", monotonicity constraints

### Type 3: Recommendation/Ranking Problem
- Dataset: User-item interaction matrix (implicit feedback: clicks, purchases)
- Expected: Collaborative filtering baseline → Matrix factorization → Evaluation (NDCG@k)
- Differentiator: Cold-start handling, temporal split for evaluation, business metric translation

### Type 4: Text Classification / NLP Task
- Dataset: Claims text or seller descriptions, binary classification
- Expected: TF-IDF baseline → BERT embeddings → classification → error analysis
- Differentiator: BERT fine-tuning vs. feature extraction decision, calibration, NER for key entities

---

## 🚦 HO ROUND MISTAKES TO AVOID

| ❌ Mistake | ✅ What to Do Instead |
|---|---|
| Random split for time-series data | Always use temporal split — mention this explicitly |
| Default 0.5 threshold | Tune threshold on PR curve, justify business constraint |
| Only reporting accuracy | Report: AUC-ROC, PR-AUC, Precision/Recall at operating threshold |
| Not checking target distribution | Always plot class distribution FIRST |
| Skipping calibration | Add Platt/isotonic calibration, show reliability diagram |
| Not narrating your thinking | Talk through EVERY step — the interview is about your thought process |
| Not discussing feature leakage | Explicitly check: "Can this feature be computed at inference time?" |
| Ignoring missing values | Always show missing value analysis + imputation strategy |

---

*See companion files: 01_EDA_Patterns.md, 02_Feature_Engineering_Code.md, 04_RAG_Implementation.md, 06_SQL_Advanced_Patterns.md*
