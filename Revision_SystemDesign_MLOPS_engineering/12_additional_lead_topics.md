# ADDITIONAL LEAD-LEVEL TOPICS — Python, SQL, Time Series, RecSys, Causal, Data Engineering
> Covers remaining interview gaps that appear at lead-level AI/ML/Data Science and engineering screenings.

---

## SECTION 1: PYTHON FOR DATA SCIENCE AND ML INTERVIEWS

### Decorators

A decorator is a function that takes another function as input, adds behavior, and returns a wrapped function. In ML, decorators are used for logging, timing, caching, retry logic, and authentication.

**Interview one-liner:**
> "I use decorators for cross-cutting concerns like logging execution time of training functions, caching repeated computations, or adding retry logic around API calls."

### Generators

Generators use `yield` to produce values lazily. They are useful when processing large datasets that do not fit in memory, such as streaming rows from a file or iterating over batches of training data.

**Interview one-liner:**
> "I use generators when processing large files or data streams because they yield one item at a time instead of loading everything into memory."

### List vs Generator Comprehension

- List comprehensions build the full list in memory
- Generator comprehensions produce values lazily
- Use generator comprehensions for large sequences

### Global Interpreter Lock (GIL)

The GIL allows only one thread to execute Python bytecode at a time. This means threads are not truly parallel for CPU-bound work. For CPU-bound tasks, use multiprocessing. For I/O-bound tasks, threading or asyncio is fine.

**Interview one-liner:**
> "For CPU-bound ML work like feature engineering on large datasets, I use multiprocessing to bypass the GIL. For I/O-bound work like API calls, I use threading or asyncio."

### Multiprocessing vs Threading vs Asyncio

| Use Case | Approach |
|---|---|
| CPU-bound (training, feature engineering) | Multiprocessing |
| I/O-bound (network, disk) | Threading or asyncio |
| Many concurrent connections | Asyncio |

### Pandas Optimization Patterns

- Use vectorized operations instead of row-wise loops
- Use `category` dtype for low-cardinality strings
- Avoid chained indexing; use `.loc` and `.iloc`
- Use `read_csv` with chunking for large files
- Use `eval` or `query` for complex filters on large frames

**Interview one-liner:**
> "I optimize pandas by using vectorized operations, categorical dtypes for repeated strings, and chunking for files too large for memory. I avoid Python loops over DataFrames."

### When to Use NumPy, Pandas, Polars

| Tool | Best For |
|---|---|
| NumPy | Numerical arrays and matrix operations |
| Pandas | Tabular data exploration and manipulation |
| Polars | Large-scale DataFrame operations, faster than pandas |
| PySpark | Distributed data processing |

### Python Memory Management

- Python uses reference counting and garbage collection
- Circular references are handled by the generational garbage collector
- Large objects in ML can cause memory pressure; use generators, chunking, and del unused variables

### Common Python Interview Scenarios

**"How would you process a 10GB CSV file?"**
> "I would use chunking with pandas `read_csv(chunksize=...)`, process each chunk, and aggregate results. If the computation is complex, I would move to Dask or PySpark for distributed processing."

**"How do you avoid data leakage in a pandas preprocessing pipeline?"**
> "I fit scalers, encoders, and imputers only on the training set, then transform validation and test sets. I never fit on the full dataset before splitting."

---

## SECTION 2: SQL FOR DATA SCIENCE AND ML INTERVIEWS

### Window Functions

Window functions compute values across a set of rows related to the current row without collapsing rows.

Common window functions:
- `ROW_NUMBER()`: unique rank, no ties
- `RANK()`: rank with gaps for ties
- `DENSE_RANK()`: rank without gaps
- `LEAD()` / `LAG()`: access previous or next row
- `SUM() OVER`, `AVG() OVER`, `COUNT() OVER`: running aggregates
- `NTILE()`: divide rows into buckets (deciles)

**Example use case:**
> "I use LAG to compute time between a claimant's consecutive claims, and NTILE to create risk deciles for model evaluation."

### Common Table Expressions (CTEs)

CTEs make complex queries readable and reusable within a single query. They are useful for multi-step feature extraction.

**Interview one-liner:**
> "I use CTEs to break complex feature engineering queries into logical steps. Each CTE computes one transformation, and the final SELECT combines them."

### Self-Joins

Self-joins connect a table to itself. In ML, they are used to find prior records for the same entity, such as a claimant's previous claims.

### SQL Optimization for ML

- Index columns used in JOIN, WHERE, and GROUP BY
- Avoid SELECT *; fetch only needed columns
- Use appropriate partitioning for large tables
- Use EXPLAIN to understand query plans
- Materialize frequently used intermediate tables

### Fraud-Specific SQL Patterns

**Running count of claims per claimant:**
> "I use a window function `COUNT(*) OVER (PARTITION BY claimant_id ORDER BY claim_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` to compute the claim sequence number without collapsing rows."

**Finding claimants with sudden change in claim amount:**
> "I use LAG to get the previous claim amount, then compute the ratio of current to previous amount. A ratio above a threshold flags a sudden spike."

### SQL Interview Scenarios

**"Find the median claim amount per policy type."**
> "I would use `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY claim_amount) OVER (PARTITION BY policy_type)` on databases that support it, such as PostgreSQL or Snowflake. For systems without it, I would use a window function to rank rows and pick the middle value."

**"Find users who made a purchase within 7 days of signing up."**
> "I would join the users table with the purchases table on user_id, filter where purchase_date is between signup_date and signup_date + 7 days, and use DISTINCT to get qualifying users."

---

## SECTION 3: TIME SERIES FORECASTING

### Components of a Time Series

- **Trend:** long-term direction
- **Seasonality:** repeating patterns at fixed intervals
- **Cycle:** longer-term oscillations not tied to calendar
- **Residual:** random noise

### Stationarity

A stationary time series has constant mean, variance, and autocorrelation over time. Most forecasting models assume stationarity.

**Tests:**
- Augmented Dickey-Fuller (ADF)
- KPSS

**Transformations:**
- Differencing
- Log transform for variance stabilization
- Box-Cox / Yeo-Johnson

### Common Time Series Models

| Model | Use Case |
|---|---|
| Moving average | Smoothing, simple forecasting |
| Exponential smoothing | Short-term forecasts with trend/seasonality |
| ARIMA | Univariate forecasting with autocorrelation |
| SARIMA | ARIMA with seasonality |
| Prophet | Business forecasting with holidays and changepoints |
| XGBoost/LightGBM with lags | Multi-variate forecasting with many features |
| Deep learning (LSTM, Transformer) | Large-scale sequence forecasting |

### Feature Engineering for Time Series

- Lag features
- Rolling mean, std, min, max
- Date features: day of week, month, quarter, holidays
- Fourier terms for seasonality
- Difference features

### Cross-Validation for Time Series

Never use random k-fold for time series; it leaks future information into training. Use walk-forward validation or expanding window validation.

**Walk-forward:**
- Train on [1..t], validate on [t+1..t+k]
- Move window forward and repeat

### Interview One-Liner

> "For time series I decompose into trend, seasonality, and residual; check stationarity with ADF; and use walk-forward validation instead of random cross-validation. I start with ARIMA or exponential smoothing and move to gradient boosting with lag features if I need many external regressors."

---

## SECTION 4: RECOMMENDATION SYSTEM ALGORITHMS

### Collaborative Filtering

Recommends items based on user-item interaction patterns.

**User-user collaborative filtering:** find similar users and recommend what they liked.
**Item-item collaborative filtering:** find items similar to what the user already interacted with.

### Matrix Factorization

Decomposes the user-item interaction matrix into lower-dimensional user and item latent factor matrices.

**Methods:** SVD, ALS, NMF

**Pros:**
- Handles sparse data
- Learns latent representations
- Scales to large catalogs

**Cons:**
- Cold-start problem for new users/items
- Struggles with non-linear patterns

### Two-Tower Neural Networks

A deep learning approach with separate neural networks for users and items. Their outputs are embeddings, and a dot product or cosine similarity produces the score.

**Use case:** large-scale retrieval in recommender systems.

### Content-Based Filtering

Recommends items similar to what the user liked before, based on item attributes.

**Use case:** new items with rich metadata but few interactions.

### Hybrid Recommenders

Combine collaborative, content-based, and business rules to get the best of each.

### Interview One-Liner

> "For recommendations, I start with collaborative filtering or matrix factorization for established users and items. For cold-start, I use content-based filtering. At scale, I use two-tower embeddings for candidate generation and a learned ranker for scoring."

---

## SECTION 5: MODEL COMPRESSION AND OPTIMIZATION

### ONNX

ONNX (Open Neural Network Exchange) is a format that allows models to be converted and run with optimized inference engines.

**Benefits:**
- Faster inference
- Framework independence
- Production deployment optimization

**Use case:** Convert XGBoost or LightGBM models to ONNX for low-latency serving.

### Quantization

Quantization reduces model precision, typically from float32 to int8, to reduce size and improve inference speed.

**Types:**
- Post-training quantization: quantize an already trained model
- Quantization-aware training: train with quantization constraints

**Trade-off:** slight accuracy loss for significant speed and memory gains.

### Pruning

Pruning removes weights or neurons that contribute little to predictions.

**Types:**
- Weight pruning: set small weights to zero
- Structured pruning: remove entire channels or neurons

### Knowledge Distillation

Train a smaller "student" model to mimic a larger "teacher" model. The student learns from the teacher's soft probabilities, capturing richer information than hard labels.

### When to Use Each

| Technique | When |
|---|---|
| ONNX | Need faster inference with same model |
| Quantization | Need smaller model and faster inference, can tolerate small accuracy loss |
| Pruning | Need smaller model, especially for edge deployment |
| Distillation | Need a small model that behaves like a large model |

### Interview One-Liner

> "When latency is critical, I convert models to ONNX, apply quantization, and use pruning or distillation if the model is still too large. I always measure the accuracy-latency trade-off on a validation set."

---

## SECTION 6: TRANSFER LEARNING

### What Is Transfer Learning

Transfer learning uses a model trained on one task as the starting point for another related task. The idea is that representations learned from a large dataset are useful for a smaller, related dataset.

### Examples

- Use BERT pre-trained on general text, fine-tune on fraud note classification
- Use ImageNet-pre-trained CNN, fine-tune on medical imaging
- Use embeddings from a large recommender, adapt to a new product category

### Fine-Tuning Strategies

- **Feature extraction:** freeze base layers, train only the head
- **Fine-tuning:** unfreeze some or all layers and train with a low learning rate
- **Progressive unfreezing:** unfreeze layers gradually

### When Transfer Learning Helps

- Small labeled dataset
- Related source task with abundant data
- Need to reduce training time and improve generalization

### Interview One-Liner

> "I use transfer learning when I have limited labeled data but a related pre-trained model exists. For fraud NLP, I might fine-tune BERT on claim notes. For images, I would use an ImageNet-pre-trained model as a backbone."

---

## SECTION 7: CAUSAL INFERENCE BASICS

### Correlation vs Causation

Machine learning models often learn correlation. Causal inference tries to estimate the effect of an intervention.

### Propensity Score Matching

Matches treated and control units with similar propensity scores to reduce selection bias. Used when randomization is not possible.

### Difference-in-Differences

Compares the change in outcomes over time between a treatment group and a control group. Useful for evaluating policy or marketing interventions.

### Uplift Modeling

Predicts which users will respond positively to a treatment. Instead of predicting outcome, it predicts the incremental effect of treatment.

### A/B Testing as Causal Inference

Randomized A/B tests are the gold standard for causal inference because randomization balances confounders across groups.

### Interview One-Liner

> "ML predicts correlation; causal inference estimates the effect of an action. When I cannot run an A/B test, I use propensity score matching or difference-in-differences. For targeting treatments, I use uplift modeling to find users most likely to benefit."

---

## SECTION 8: DATA ENGINEERING BASICS FOR ML LEADS

### Data Warehouse vs Data Lake vs Lakehouse

| System | Strengths |
|---|---|
| Data warehouse | Structured data, SQL analytics, strong governance |
| Data lake | Cheap storage for raw structured, semi-structured, unstructured data |
| Lakehouse | Combines data lake storage with warehouse-like transactions and performance (Delta Lake, Iceberg) |

### ETL vs ELT

- **ETL:** Extract, Transform, Load — transform before loading to warehouse
- **ELT:** Extract, Load, Transform — load raw data first, transform in warehouse

**When to use ETL:** data must be cleansed before storage due to compliance or quality.
**When to use ELT:** warehouse is powerful enough and raw data has analytical value.

### Streaming vs Batch

- **Batch:** process large volumes of data at scheduled intervals
- **Streaming:** process events as they arrive

**Streaming concepts:**
- Event time vs processing time
- Windowing: tumbling, sliding, session windows
- Watermarking: handle late-arriving events
- Exactly-once vs at-least-once semantics

### Tools

| Use Case | Tools |
|---|---|
| Batch orchestration | Airflow, Dagster, Prefect |
| Stream processing | Kafka, Flink, Spark Structured Streaming, ksqlDB |
| Data quality | Great Expectations, dbt tests, Deequ |
| Schema registry | Confluent Schema Registry, AWS Glue |

### Interview One-Liner

> "I use a lakehouse architecture with Delta Lake for structured ML data and a data lake for raw files. For feature freshness, I use Spark Structured Streaming or Flink. I enforce data quality with Great Expectations and schema registries."

---

## SECTION 9: CLUSTERING AND DIMENSIONALITY REDUCTION

### Clustering Algorithms

| Algorithm | Use Case |
|---|---|
| K-means | Simple, spherical clusters |
| Hierarchical clustering | When number of clusters is unknown |
| DBSCAN | Arbitrary shapes, noise detection |
| Gaussian Mixture Models | Soft cluster assignment |

**Choosing K:** elbow method, silhouette score, domain knowledge.

### Dimensionality Reduction

| Technique | Use Case |
|---|---|
| PCA | Linear dimensionality reduction, remove collinearity |
| t-SNE | Visualization of high-dimensional data |
| UMAP | Faster visualization with better global structure |
| Autoencoder | Non-linear dimensionality reduction |

### Interview One-Liner

> "For customer segmentation I use K-means or DBSCAN depending on cluster shape. For visualization, I use UMAP or t-SNE. For removing redundancy in features, I use PCA before modeling if multicollinearity is a concern."

---

## SECTION 10: OUTLIER DETECTION

### Methods

| Method | Use Case |
|---|---|
| Z-score / IQR | Univariate, roughly normal data |
| Isolation Forest | Multivariate outliers |
| One-class SVM | Learn boundary of normal data |
| Local Outlier Factor (LOF) | Density-based anomaly detection |
| Autoencoder | High-dimensional reconstruction error |

### Production Use

- Outliers may be fraud cases or data quality issues
- Treat outliers based on cause; do not always remove them
- Monitor outlier rate over time

### Interview One-Liner

> "I detect outliers using statistical methods for simple cases and isolation forest or autoencoder for multivariate patterns. In fraud, outliers are often the signal, so I keep them and model them rather than removing them."

---

## SECTION 11: FEATURE SELECTION METHODS

### Filter Methods

- Correlation
- Mutual information
- Chi-square
- Information value (IV)

### Wrapper Methods

- Forward selection
- Backward elimination
- Recursive feature elimination (RFE)

### Embedded Methods

- Lasso (L1 regularization)
- Tree-based feature importance
- SHAP-based selection

### Interview One-Liner

> "I start with filter methods like correlation and IV to remove obviously weak features, then use embedded methods like Lasso or tree importance, and finally validate with SHAP. I avoid wrapper methods on large datasets because they are computationally expensive."

---

## SECTION 12: CROSS-VALIDATION AND LEAKAGE PREVENTION

### Cross-Validation Strategies

| Strategy | Use When |
|---|---|
| K-fold | General tabular data, no temporal component |
| Stratified k-fold | Imbalanced classification |
| Group k-fold | Same group appears in multiple rows; prevent group leakage |
| Time series split | Temporal data |
| Walk-forward validation | Sequential forecasting |

### Common Leakage Sources

- Target leakage: using features not available at prediction time
- Train-test contamination: preprocessing on full data before split
- Group leakage: same entity in train and test
- Temporal leakage: future information in training

### Prevention

- Split before any preprocessing that learns from data
- Use group-aware or time-aware splits
- Validate feature availability timestamps
- Keep a final holdout set untouched until the end

### Interview One-Liner

> "I choose cross-validation based on data structure: stratified k-fold for imbalanced data, group k-fold when rows share entities, and walk-forward validation for time series. I always split before fitting scalers or encoders and validate feature timestamps to prevent leakage."

---

## SECTION 13: MISSING DATA STRATEGIES

### Missing Data Mechanisms

- **MCAR:** Missing completely at random — unrelated to any observed or unobserved value
- **MAR:** Missing at random — depends on observed values
- **MNAR:** Missing not at random — depends on the missing value itself

### Strategies

| Strategy | Use Case |
|---|---|
| Deletion | MCAR, small proportion missing |
| Mean/median imputation | Quick baseline, numeric features |
| Mode imputation | Categorical features |
| Model-based imputation | KNN imputer, iterative imputer |
| Missing indicator | MNAR — the fact that data is missing is informative |
| Domain-specific imputation | Use business rules or external data |

### Production Concern

Training-time imputation must match serving-time imputation. Use the same imputation values stored from training or use a model that handles missing values natively, like XGBoost.

### Interview One-Liner

> "I choose imputation based on the missing mechanism. For MCAR, simple imputation works. For MNAR, I add a missing indicator as a feature. In production, I ensure training and serving use identical imputation logic, or I use models like XGBoost that handle missing values natively."

---

## SECTION 14: EXPERIMENTAL DESIGN AND POWER ANALYSIS

### Power Analysis

Power analysis determines the sample size needed to detect an effect of a given size with a given probability.

**Inputs:**
- Minimum detectable effect
- Desired statistical power (usually 80%)
- Significance level alpha (usually 5%)
- Baseline metric value and variance

### Why It Matters in ML A/B Tests

Fraud and healthcare labels are rare. Without power analysis, you may run an experiment too short to detect a real improvement.

### Minimal Detectable Effect

The smallest effect size that is both statistically significant and business meaningful. Do not run experiments to detect effects smaller than the business cares about.

### Interview One-Liner

> "Before running an ML A/B test, I do a power analysis to determine how long we need to wait for enough labeled outcomes. I define the minimum detectable effect based on business impact, not just statistical significance."

---

## SECTION 15: ADDITIONAL INTERVIEW SCENARIOS

### "What is the difference between a data warehouse and a data lake?"

> "A data warehouse stores structured, processed data optimized for SQL analytics. A data lake stores raw data in any format cheaply. A lakehouse combines both: cheap storage plus ACID transactions and performance."

### "How do you handle seasonality in a forecasting model?"

> "I add calendar features like month, day-of-week, and holidays. I also use Fourier terms or seasonal decomposition. For ARIMA, I use SARIMA. For gradient boosting, lag and rolling features capture seasonal effects."

### "When would you use PCA?"

> "I use PCA when features are highly correlated and I want to reduce dimensionality or remove multicollinearity before a model like linear regression. I avoid PCA when interpretability of individual features is important."

### "How do you choose the number of clusters?"

> "I use the elbow method, silhouette score, and domain knowledge. I also inspect cluster stability and business interpretability. The number of clusters should make sense for the problem, not just optimize a metric."

### "Explain transfer learning in simple terms."

> "Transfer learning is using knowledge from a large, related problem to solve a smaller problem faster and with less data. For example, a model trained on millions of general text sentences can be fine-tuned on a few thousand fraud claim notes to classify them."
