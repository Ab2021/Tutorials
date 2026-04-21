# 💻 HO GRIND: PySpark, NLP & Advanced Code Patterns
### Round 3 Hands-On Supplement — Production Code They Will Ask About

> Your resume explicitly lists PySpark, Spark MLlib, GCP Dataproc, Airflow, Neo4j, lifelines. These become live-coding targets.

---

## ═══════════════════════════════════════
## SECTION 1: PYSPARK PRODUCTION PATTERNS
## ═══════════════════════════════════════

### Pattern 1: Window Functions for Rolling Features (CLV / Fraud)

```python
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

def build_rolling_features(df, user_col="user_id", time_col="event_date", 
                            amount_col="purchase_amount"):
    """
    Production-grade rolling feature computation using Spark Window functions.
    NEVER use UDFs for this — Window functions are 10-100x faster.
    """
    # === DEFINE WINDOWS ===
    # Rows-based: last N transactions (ordered by time)
    w_last_10 = (Window.partitionBy(user_col)
                       .orderBy(F.col(time_col).cast("long"))
                       .rowsBetween(-9, 0))  # current + 9 previous rows
    
    # Range-based: last N days (ordered by timestamp in seconds)
    w_30d = (Window.partitionBy(user_col)
                   .orderBy(F.col(time_col).cast("long"))
                   .rangeBetween(-30 * 86400, 0))  # 30 days in seconds
    
    w_7d = (Window.partitionBy(user_col)
                  .orderBy(F.col(time_col).cast("long"))
                  .rangeBetween(-7 * 86400, 0))
    
    # Running total (unbounded preceding)
    w_cumulative = (Window.partitionBy(user_col)
                          .orderBy(F.col(time_col).cast("long"))
                          .rowsBetween(Window.unboundedPreceding, 0))
    
    # === COMPUTE FEATURES ===
    df = df.withColumn("txn_count_7d", 
                       F.count("*").over(w_7d))
    
    df = df.withColumn("amount_sum_30d",
                       F.sum(amount_col).over(w_30d))
    
    df = df.withColumn("amount_avg_30d",
                       F.avg(amount_col).over(w_30d))
    
    df = df.withColumn("amount_max_30d",
                       F.max(amount_col).over(w_30d))
    
    df = df.withColumn("txn_count_total",
                       F.count("*").over(w_cumulative))
    
    # Amount z-score (current txn vs user's 30d average)
    df = df.withColumn("amount_zscore",
                       (F.col(amount_col) - F.col("amount_avg_30d")) / 
                       (F.stddev(amount_col).over(w_30d) + 1e-8))
    
    # Days since last transaction
    w_prev = (Window.partitionBy(user_col)
                    .orderBy(F.col(time_col).cast("long")))
    
    df = df.withColumn("prev_txn_time",
                       F.lag(F.col(time_col).cast("long"), 1).over(w_prev))
    
    df = df.withColumn("hours_since_last_txn",
                       (F.col(time_col).cast("long") - F.col("prev_txn_time")) / 3600)
    
    # Sequential rank within user (useful for "is this their first purchase?")
    df = df.withColumn("txn_rank_asc",
                       F.rank().over(w_prev))
    
    df = df.withColumn("is_first_purchase",
                       (F.col("txn_rank_asc") == 1).cast("int"))
    
    return df.drop("prev_txn_time")
```

---

### Pattern 2: Salting for Skewed GroupBy (The VIP Data Problem)

```python
def groupby_with_salting(df, key_col, agg_cols, n_salts=10):
    """
    Handle data skew in groupBy using salting technique.
    Use when: one key has disproportionate row count (VIP customers, 
    popular products, hub warehouses).
    
    Theory: Distributes skewed key's records across n_salts reducers.
    Speedup ≈ min(n_salts, skew_ratio).
    """
    # Step 1: Add random salt to distribute skewed keys
    df_salted = df.withColumn(
        "salt", (F.rand() * n_salts).cast("int")
    ).withColumn(
        "salted_key", 
        F.concat(F.col(key_col).cast("string"), F.lit("_"), F.col("salt"))
    )
    
    # Step 2: Partial aggregation on salted key (parallelized)
    agg_exprs_partial = {col: "sum" for col in agg_cols}  # Adjust per metric type
    
    partial_result = df_salted.groupBy("salted_key", key_col).agg(
        *[F.sum(col).alias(f"partial_{col}") for col in agg_cols],
        F.count("*").alias("partial_count")
    )
    
    # Step 3: Final aggregation on original key (now balanced)
    final_result = partial_result.groupBy(key_col).agg(
        *[F.sum(f"partial_{col}").alias(col) for col in agg_cols],
        F.sum("partial_count").alias("total_count")
    )
    
    return final_result

# Usage
result = groupby_with_salting(
    transactions_df, 
    key_col="customer_id",
    agg_cols=["purchase_amount", "discount_amount"],
    n_salts=10
)
```

---

### Pattern 3: Point-in-Time Correct Feature Join (Training Data Safety)

```python
def point_in_time_correct_join(events_df, features_df, 
                                entity_col, event_time_col, 
                                feature_time_col):
    """
    CRITICAL for ML training pipelines: join features as they existed 
    at the time of each event, NOT as they exist today.
    
    Prevents: training-serving skew, data leakage.
    
    Args:
        events_df: df with events to label (e.g., transactions)
        features_df: df with feature snapshots (daily feature store)
        entity_col: join key (e.g., "user_id")
        event_time_col: timestamp of the event
        feature_time_col: snapshot date of the feature
    """
    from pyspark.sql.functions import col, datediff
    
    # Join all feature snapshots to all events (cross-temporal)
    joined = events_df.alias("e").join(
        features_df.alias("f"),
        on=col(f"e.{entity_col}") == col(f"f.{entity_col}"),
        how="left"
    )
    
    # Keep only feature snapshots that are:
    # 1. Before or on the event date (no future leakage)
    # 2. The MOST RECENT snapshot before the event
    joined_filtered = joined.filter(
        col(f"f.{feature_time_col}") <= col(f"e.{event_time_col}")
    )
    
    # Rank feature snapshots per event: most recent first
    w = Window.partitionBy(
        f"e.{entity_col}", event_time_col
    ).orderBy(col(f"f.{feature_time_col}").desc())
    
    joined_ranked = joined_filtered.withColumn("rank", F.rank().over(w))
    
    # Keep only the most recent feature snapshot per event
    result = joined_ranked.filter(col("rank") == 1).drop("rank")
    
    return result
```

---

### Pattern 4: Efficient Large-Scale Scoring (Batch Inference)

```python
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd
import pickle
import numpy as np

def create_scoring_udf(model_path: str):
    """
    Create a broadcast-efficient Pandas UDF for model scoring.
    Pandas UDFs are vectorized (Apache Arrow) — 10-100x faster than row UDFs.
    
    Key: Load the model ONCE per executor, not per row.
    """
    # Broadcast the model path — actual model loaded lazily on executor
    model_broadcast = None
    
    @pandas_udf(DoubleType())
    def score_batch(feature_series: pd.Series) -> pd.Series:
        global model_broadcast
        
        # Lazy loading: model loaded once per executor process
        if model_broadcast is None:
            with open(model_path, 'rb') as f:
                model_broadcast = pickle.load(f)
        
        # Vectorized prediction over the entire partition
        features = np.stack(feature_series.values)
        predictions = model_broadcast.predict_proba(features)[:, 1]
        return pd.Series(predictions)
    
    return score_batch

# Usage in pipeline:
# score_udf = create_scoring_udf("gs://models/fraud/v2.3/model.pkl")
# scored_df = features_df.withColumn("fraud_score", 
#     score_udf(F.struct(*feature_cols)))
```

---

### Pattern 5: Spark MLlib - Entity Matching with MinHashLSH

```python
from pyspark.ml.feature import (HashingTF, IDF, Tokenizer, 
                                  NGram, MinHashLSH, CountVectorizer)
from pyspark.ml import Pipeline

def build_entity_matching_pipeline(df_a, df_b, 
                                    text_col="company_name",
                                    id_col="id",
                                    jaccard_threshold=0.3):
    """
    Scalable entity matching using MinHashLSH.
    Reduces O(n²) comparisons to O(n log n) via locality-sensitive hashing.
    
    From EXL Plan Sponsor Entity Matching project.
    """
    # Step 1: Text normalization
    def normalize_text(df, col_name):
        return (df
            .withColumn(col_name, F.lower(F.col(col_name)))
            .withColumn(col_name, F.regexp_replace(F.col(col_name), 
                                                    r'\b(inc|llc|corp|ltd|co)\b\.?', ''))
            .withColumn(col_name, F.regexp_replace(F.col(col_name), 
                                                    r'[^a-z0-9\s]', ' '))
            .withColumn(col_name, F.trim(F.regexp_replace(F.col(col_name), 
                                                           r'\s+', ' ')))
        )
    
    df_a_norm = normalize_text(df_a, text_col)
    df_b_norm = normalize_text(df_b, text_col)
    
    # Step 2: Character n-gram tokenization (3-grams for typo robustness)
    def add_char_ngrams(df, col_name, n=3):
        # Split into characters first
        df = df.withColumn("chars", 
                           F.split(F.col(col_name), ""))
        # Create n-grams from character array
        # Simpler: use NGram on character tokens
        df = df.withColumn("char_tokens",
                           F.array_distinct(
                               F.split(
                                   F.regexp_replace(F.col(col_name), 
                                                    r'(?<=.)', ' '), ' '
                               )
                           ))
        return df
    
    # Step 3: Vectorize with CountVectorizer (character n-gram bag)
    tokenizer = Tokenizer(inputCol=text_col, outputCol="words")
    ngram = NGram(n=3, inputCol="words", outputCol="ngrams")
    cv = CountVectorizer(inputCol="ngrams", outputCol="features", 
                         minDF=2.0, vocabSize=50000)
    
    pipeline_model = Pipeline(stages=[tokenizer, ngram, cv]).fit(df_a_norm)
    
    features_a = pipeline_model.transform(df_a_norm)
    features_b = pipeline_model.transform(df_b_norm)
    
    # Step 4: MinHashLSH for approximate nearest neighbor
    mh = MinHashLSH(inputCol="features", outputCol="hashes",
                    numHashTables=20,  # More tables = higher recall, more compute
                    seed=42)
    
    mh_model = mh.fit(features_a)
    
    # Step 5: Approximate join — only compare pairs that share hash buckets
    similar_pairs = mh_model.approxSimilarityJoin(
        features_a.select(id_col, "features"),
        features_b.select(id_col, "features"),
        threshold=jaccard_threshold,
        distCol="jaccard_distance"
    )
    
    # Step 6: Exact similarity for candidate pairs (within LSH buckets only)
    result = similar_pairs.select(
        F.col(f"datasetA.{id_col}").alias("id_a"),
        F.col(f"datasetB.{id_col}").alias("id_b"),
        (1 - F.col("jaccard_distance")).alias("jaccard_similarity")
    ).filter(F.col("jaccard_similarity") >= jaccard_threshold)
    
    return result
```

---

## ═══════════════════════════════════════
## SECTION 2: NLP & TEXT PATTERNS
## ═══════════════════════════════════════

### Pattern 6: BERT Feature Extraction for Tabular + Text Fusion

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    """Dataset for batched BERT inference. Use DataLoader for efficiency."""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,          # CRITICAL: clinical notes can be very long
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def extract_bert_embeddings(texts: list[str], 
                             model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                             batch_size: int = 32,
                             device: str = "cuda" if torch.cuda.is_available() else "cpu"
                             ) -> np.ndarray:
    """
    Extract mean-pooled CLS token embeddings from ClinicalBERT.
    
    Choices justified:
    - ClinicalBERT over standard BERT: trained on MIMIC-III clinical notes
    - Mean pooling over CLS only: more robust for longer texts
    - batch_size=32: balance GPU memory vs. throughput
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()  # Disable dropout during inference
    
    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_embeddings = []
    
    with torch.no_grad():  # No gradient computation needed for inference
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Mean pooling: average over non-padding tokens only
            # Better than CLS for longer texts with complex information
            token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
            
            # Expand mask for broadcasting: (batch, seq_len) -> (batch, seq_len, hidden_size)
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Sum embeddings where mask=1, divide by count of non-padding tokens
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(mean_embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def fuse_bert_with_tabular(tabular_df, text_col: str, bert_embeddings: np.ndarray,
                            embedding_dim: int = 768):
    """
    Concatenate BERT embeddings with tabular features.
    Returns a feature matrix ready for XGBoost.
    """
    import pandas as pd
    
    # Create embedding DataFrame
    embedding_cols = [f"bert_dim_{i}" for i in range(embedding_dim)]
    embedding_df = pd.DataFrame(bert_embeddings, columns=embedding_cols,
                                 index=tabular_df.index)
    
    # Drop original text column
    tabular_no_text = tabular_df.drop(columns=[text_col])
    
    # Concatenate
    fused = pd.concat([tabular_no_text, embedding_df], axis=1)
    
    print(f"Fused feature matrix: {fused.shape}")
    print(f"  Tabular features: {tabular_no_text.shape[1]}")
    print(f"  BERT embedding features: {embedding_dim}")
    
    return fused
```

---

### Pattern 7: Survival Analysis with lifelines (Cox PH + KM)

```python
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

def run_survival_analysis(df, duration_col, event_col, 
                           covariate_cols, strata_col=None):
    """
    Complete survival analysis pipeline from your EXL patient readmission work.
    
    Args:
        duration_col: time until event or censoring (e.g., "days_to_readmission")
        event_col: 1 if event occurred, 0 if censored (e.g., "readmitted")
        covariate_cols: feature columns for Cox model
        strata_col: column to stratify on (if PH assumption violated)
    """
    
    # === STEP 1: KAPLAN-MEIER CURVES ===
    kmf = KaplanMeierFitter()
    
    plt.figure(figsize=(12, 5))
    
    # Overall survival
    plt.subplot(1, 2, 1)
    kmf.fit(df[duration_col], event_observed=df[event_col], label="All Patients")
    kmf.plot_survival_function(ci_show=True)
    plt.title("Kaplan-Meier Survival Curve")
    plt.xlabel("Days")
    plt.ylabel("P(No Readmission)")
    
    # Stratified KM (e.g., by age group) — log-rank test
    if strata_col and df[strata_col].nunique() == 2:
        plt.subplot(1, 2, 2)
        for val, group in df.groupby(strata_col):
            kmf.fit(group[duration_col], event_observed=group[event_col], 
                    label=f"{strata_col}={val}")
            kmf.plot_survival_function(ci_show=False)
        
        # Log-rank test
        groups = df[strata_col].unique()
        g1 = df[df[strata_col] == groups[0]]
        g2 = df[df[strata_col] == groups[1]]
        
        result = logrank_test(
            g1[duration_col], g2[duration_col],
            event_observed_A=g1[event_col],
            event_observed_B=g2[event_col]
        )
        plt.title(f"Stratified KM | Log-rank p={result.p_value:.4f}")
    
    plt.tight_layout()
    plt.savefig("survival_curves.png", dpi=150, bbox_inches='tight')
    
    # === STEP 2: COX PROPORTIONAL HAZARDS MODEL ===
    cox_df = df[[duration_col, event_col] + covariate_cols].dropna()
    
    if strata_col:
        # Stratified Cox: different baseline hazard per stratum, shared coefficients
        cph = CoxPHFitter(strata=strata_col)
    else:
        cph = CoxPHFitter(penalizer=0.1)  # L2 regularization
    
    cph.fit(cox_df, duration_col=duration_col, event_col=event_col,
            show_progress=False)
    
    print("=== Cox PH Summary ===")
    cph.print_summary(decimals=4)
    
    # === STEP 3: CHECK PROPORTIONAL HAZARDS ASSUMPTION ===
    print("\n=== Schoenfeld Residuals Test (PH Assumption) ===")
    cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=True)
    # If p < 0.05 for a covariate → PH assumption violated for that covariate
    # Fix: add strata or time-varying coefficient
    
    # === STEP 4: DISCRIMINATION METRICS ===
    # Concordance index (C-index): equivalent to AUC for survival models
    # C-index = P(model predicts higher risk for patient who fails first)
    c_index = cph.concordance_index_
    print(f"\nC-index (discrimination): {c_index:.4f}")
    # C-index > 0.7: good | > 0.8: excellent
    
    # === STEP 5: RISK SCORES FOR HIGH-RISK PATIENTS ===
    cox_df["risk_score"] = cph.predict_log_partial_hazard(cox_df)
    cox_df["risk_percentile"] = cox_df["risk_score"].rank(pct=True)
    
    high_risk = cox_df[cox_df["risk_percentile"] >= 0.8]  # Top 20%
    print(f"\nHigh-risk patients (top 20%): {len(high_risk):,}")
    print(f"Event rate in high-risk group: {high_risk[event_col].mean():.1%}")
    print(f"Event rate in low-risk group: "
          f"{cox_df[cox_df['risk_percentile'] < 0.8][event_col].mean():.1%}")
    
    return cph, cox_df
```

---

## ═══════════════════════════════════════
## SECTION 3: ADVANCED SQL PATTERNS
## ═══════════════════════════════════════

### Pattern 8: Recursive CTE for Fraud Ring Traversal

```sql
-- Find all accounts connected (directly or indirectly) to a known fraudster
-- using recursive CTE graph traversal

WITH RECURSIVE fraud_network AS (
    -- Base case: start from known fraudster
    SELECT 
        account_id,
        account_id AS root_fraudster,
        0 AS depth
    FROM accounts
    WHERE is_confirmed_fraud = TRUE
    
    UNION ALL
    
    -- Recursive case: find connected accounts (via shared device/address)
    SELECT 
        c.account_id_b AS account_id,
        fn.root_fraudster,
        fn.depth + 1
    FROM fraud_network fn
    JOIN account_connections c ON fn.account_id = c.account_id_a
    WHERE fn.depth < 3  -- Limit traversal depth to prevent infinite loops
      AND c.connection_type IN ('shared_device', 'shared_address', 'shared_phone')
)
SELECT 
    fn.account_id,
    fn.root_fraudster,
    fn.depth AS hops_from_fraudster,
    a.account_name,
    a.total_gmv,
    a.is_confirmed_fraud
FROM fraud_network fn
JOIN accounts a ON fn.account_id = a.account_id
WHERE fn.account_id != fn.root_fraudster  -- Exclude the fraudster itself
ORDER BY fn.depth, a.total_gmv DESC;
```

---

### Pattern 9: Advanced Window Functions for Cohort Analysis

```sql
-- Full cohort retention analysis with rolling windows
WITH user_cohorts AS (
    SELECT 
        user_id,
        DATE_TRUNC('month', MIN(order_date)) AS cohort_month
    FROM orders
    GROUP BY user_id
),
user_monthly_activity AS (
    SELECT 
        o.user_id,
        uc.cohort_month,
        DATE_TRUNC('month', o.order_date) AS activity_month,
        -- Months since cohort start
        DATE_DIFF(
            DATE_TRUNC('month', o.order_date),
            uc.cohort_month,
            MONTH
        ) AS months_since_cohort,
        SUM(o.order_amount) AS monthly_revenue,
        COUNT(o.order_id) AS order_count
    FROM orders o
    JOIN user_cohorts uc USING (user_id)
    GROUP BY 1, 2, 3, 4
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(DISTINCT user_id) AS cohort_size
    FROM user_cohorts
    GROUP BY cohort_month
),
retention_matrix AS (
    SELECT 
        uma.cohort_month,
        uma.months_since_cohort,
        COUNT(DISTINCT uma.user_id) AS active_users,
        SUM(uma.monthly_revenue) AS cohort_revenue,
        -- Calculate retention rate using window function
        1.0 * COUNT(DISTINCT uma.user_id) / 
            FIRST_VALUE(COUNT(DISTINCT uma.user_id)) OVER (
                PARTITION BY uma.cohort_month 
                ORDER BY uma.months_since_cohort
            ) AS retention_rate
    FROM user_monthly_activity uma
    GROUP BY 1, 2
)
SELECT 
    rm.cohort_month,
    rm.months_since_cohort,
    cs.cohort_size,
    rm.active_users,
    ROUND(rm.retention_rate * 100, 1) AS retention_pct,
    ROUND(rm.cohort_revenue, 0) AS revenue,
    -- LTV accumulation using running sum
    SUM(rm.cohort_revenue) OVER (
        PARTITION BY rm.cohort_month 
        ORDER BY rm.months_since_cohort
    ) / cs.cohort_size AS cumulative_avg_ltv
FROM retention_matrix rm
JOIN cohort_sizes cs USING (cohort_month)
ORDER BY cohort_month, months_since_cohort;
```

---

### Pattern 10: Uplift Modeling (Propensity vs. Uplift)

```python
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd

class TwoModelUpliftEstimator(BaseEstimator, RegressorMixin):
    """
    Two-model approach for uplift (CATE) estimation.
    
    From EXL propensity modeling work: instead of targeting high-propensity
    users (who convert anyway), target high-UPLIFT users (who convert 
    BECAUSE of the treatment).
    
    CATE(x) = E[Y(1) - Y(0) | X=x]  — treated vs. untreated expected outcome
    
    Customer segments:
    - Persuadables: high uplift → TARGET THESE
    - Sure Things: convert anyway, low marginal value of treatment
    - Lost Causes: won't convert regardless
    - Sleeping Dogs: treat negatively (don't treat)
    """
    
    def __init__(self, base_learner=None):
        self.base_learner = base_learner or GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42
        )
        self.model_treated = None
        self.model_control = None
    
    def fit(self, X, y, treatment):
        """
        X: features, y: outcome, treatment: binary treatment indicator
        """
        treated_mask = treatment == 1
        control_mask = treatment == 0
        
        # Model 1: P(Y=1 | X, T=1) — outcome model for treated group
        from sklearn.base import clone
        self.model_treated = clone(self.base_learner)
        self.model_treated.fit(X[treated_mask], y[treated_mask])
        
        # Model 2: P(Y=1 | X, T=0) — outcome model for control group
        self.model_control = clone(self.base_learner)
        self.model_control.fit(X[control_mask], y[control_mask])
        
        return self
    
    def predict_uplift(self, X):
        """
        Estimate CATE: P(Y=1|X,T=1) - P(Y=1|X,T=0)
        Positive → treatment helps
        Negative → treatment hurts (sleeping dogs)
        """
        p_treated = self.model_treated.predict_proba(X)[:, 1]
        p_control = self.model_control.predict_proba(X)[:, 1]
        
        uplift = p_treated - p_control
        return uplift
    
    def score_and_segment(self, X, top_k_pct=0.2):
        """Segment users into Persuadables, Sure Things, etc."""
        uplift = self.predict_uplift(X)
        p_control = self.model_control.predict_proba(X)[:, 1]
        
        results = pd.DataFrame({
            'uplift_score': uplift,
            'p_no_treatment': p_control,
            'segment': 'Other'
        })
        
        threshold = np.percentile(uplift, (1 - top_k_pct) * 100)
        
        results.loc[results['uplift_score'] >= threshold, 'segment'] = 'Persuadable'
        results.loc[(results['p_no_treatment'] > 0.7) & 
                    (results['uplift_score'] < 0.05), 'segment'] = 'Sure Thing'
        results.loc[(results['p_no_treatment'] < 0.1) & 
                    (results['uplift_score'] < 0.0), 'segment'] = 'Lost Cause'
        results.loc[results['uplift_score'] < -0.1, 'segment'] = 'Sleeping Dog'
        
        print("Segment distribution:")
        print(results['segment'].value_counts())
        print(f"\nPersuadables: {(results['segment']=='Persuadable').sum():,} users")
        print(f"Avg uplift in persuadable group: "
              f"{results.loc[results['segment']=='Persuadable','uplift_score'].mean():.3f}")
        
        return results


# Usage:
# uplift_model = TwoModelUpliftEstimator()
# uplift_model.fit(X_train, y_train, treatment=treatment_train)
# scores = uplift_model.score_and_segment(X_test, top_k_pct=0.20)
# Target only the 'Persuadable' segment for campaigns
```

---

## ═══════════════════════════════════════
## SECTION 4: QUICK-FIRE CODING CHALLENGES
## ═══════════════════════════════════════

### Challenge 1: Implement PSI (Population Stability Index) from scratch

```python
def compute_psi(expected: np.ndarray, actual: np.ndarray, 
                n_bins: int = 10) -> float:
    """
    Population Stability Index — production model monitoring metric.
    
    PSI < 0.1:  No significant shift
    PSI 0.1-0.25: Moderate shift, investigate  
    PSI > 0.25: Significant shift, retrain!
    
    Usage: Run weekly on model score distributions to detect drift.
    """
    # Create bin edges based on expected distribution quantiles
    bin_edges = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf   # Handle values below min
    bin_edges[-1] = np.inf   # Handle values above max
    
    # Compute frequencies
    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]
    
    # Convert to proportions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Avoid log(0) — clip to small epsilon
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    
    # PSI formula: sum over bins of (actual% - expected%) * ln(actual%/expected%)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return psi
```

### Challenge 2: Implement Jaro-Winkler from scratch

```python
def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """
    Jaro-Winkler similarity. Range [0, 1]. Higher = more similar.
    Better than Levenshtein for name matching — rewards common prefixes.
    """
    if s1 == s2:
        return 1.0
    
    len_s1, len_s2 = len(s1), len(s2)
    match_dist = max(len_s1, len_s2) // 2 - 1
    
    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2
    
    matches = 0
    transpositions = 0
    
    # Find matches
    for i in range(len_s1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len_s2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # Count transpositions
    k = 0
    for i in range(len_s1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    jaro = (matches / len_s1 + matches / len_s2 + 
            (matches - transpositions / 2) / matches) / 3
    
    # Winkler prefix bonus
    prefix = 0
    for i in range(min(4, min(len_s1, len_s2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    
    return jaro + prefix * p * (1 - jaro)
```

---

*This file supplements: 00_HO_Master_Guide.md, 01_Complete_Implementation_Playbook.md, 02_MLOps_Docker_K8s_Airflow_Grind.md*
