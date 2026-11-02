# General Mills — Senior Data Scientist / AI Interview Preparation Guide

> **Sources validated from:** Glassdoor, InterviewQuery, LockedIn AI, PracHub, General Mills Careers page (job ID 30610 + Sr. D&T Data Scientist JD), Weekday.works GM job listing, and candidate-shared interview reviews (2024–2026). Questions are grounded in actual GM interview reports and official JD requirements.

---

## 📋 Table of Contents

1. [About the Role & What GM Looks For](#1-about-the-role--what-gm-looks-for)
2. [Interview Process Overview](#2-interview-process-overview)
3. [Technical Basics You Must Know](#3-technical-basics-you-must-know)
4. [30+ Interview Questions with Answers](#4-30-interview-questions-with-answers)
   - [Machine Learning & Modeling](#a-machine-learning--modeling)
   - [Statistics & Probability](#b-statistics--probability)
   - [SQL & Data Engineering](#c-sql--data-engineering)
   - [Time-Series Forecasting & Supply Chain](#d-time-series-forecasting--supply-chain)
   - [Generative AI / LLM / RAG / Agentic AI](#e-generative-ai--llm--rag--agentic-ai)
   - [Python Coding](#f-python-coding)
   - [Behavioral & Stakeholder Management](#g-behavioral--stakeholder-management)
5. [Tips to Ace the GM Interview](#5-tips-to-ace-the-gm-interview)

---

## 1. About the Role & What GM Looks For

General Mills' **Sr. Data Scientist** roles sit inside the **Digital & Technology (D&T)** group and the **Global Innovation Center (GIC)** in Mumbai/Minneapolis. The official JD (March 2026) lists these as the core pillars:

| Domain | Key Skills |
| --- | --- |
| Core ML | Supervised/unsupervised learning, ensemble models, neural networks, regression, classification |
| Time-Series & Forecasting | ARIMA/SARIMA, Prophet, Exponential Smoothing, hierarchical forecasting |
| Operations Research | Linear programming, MIP, constraint optimization (Google OR-Tools, Gurobi, CPLEX) |
| Generative AI / LLM | RAG pipelines, Agentic AI, NLP (NER, classification), fine-tuning, prompt engineering |
| Cloud & MLOps | GCP (VertexAI, BigQuery), containerization, data lineage, model monitoring |
| Domain | CPG/FMCG, supply chain, demand planning, price optimization, market mix modeling |
| Communication | Storytelling to non-technical stakeholders, Agile/sprint-based delivery |

**Experience range:** 10–12 years total analytics; 5+ years in supply chain analytics. Bachelor's/Master's in CS, Statistics, Applied Math, OR, or Industrial Engineering.

---

## 2. Interview Process Overview

Based on multiple candidate reports (Glassdoor, LockedIn AI — 2024–2026):

```
Round 1: Recruiter Screen (30 min)
   → Background, motivation, salary expectations, basic SQL/Python check

Round 2: Technical Screen – Coding (60 min)
   → Shared coding pad (Codility or VS Code Live Share)
   → Python + SQL questions (window functions, time-series wrangling)
   → Evaluation metrics discussion (MAPE, WAPE, RMSE)

Round 3: Take-Home Case Study (1–3 days)
   → Jupyter notebook: EDA → model → business recommendation
   → OR: Review a pre-existing DS notebook and critique it

Round 4: On-site / Virtual Panel (Half-day)
   → Presentation of take-home to stakeholders (Marketing + Supply Chain)
   → Deep technical/modeling interview with a senior/principal DS
   → Cross-functional interview with Supply Chain planner
   → Behavioral interview with hiring manager (STAR format)
   → Culture/values interview (peer DS or team manager)

Optional: Hogan Personality Assessment (for leadership track roles)
```

---

## 3. Technical Basics You Must Know

### 🔹 Machine Learning Fundamentals

- **Bias-Variance Tradeoff** — underfitting vs. overfitting, regularization (L1/L2)
- **Cross-Validation** — k-fold, stratified k-fold, time-based splits for time-series
- **Evaluation Metrics** — accuracy, precision, recall, F1, AUC-ROC, MAPE, WAPE, RMSE, MAE
- **Ensemble Methods** — Bagging (Random Forest), Boosting (XGBoost, LightGBM), Stacking
- **Hyperparameter Tuning** — Grid Search, Bayesian Optimization (Optuna)

### 🔹 Statistics Essentials

- Hypothesis testing (t-test, chi-square, ANOVA), p-values, confidence intervals
- Central Limit Theorem, Law of Large Numbers
- A/B testing design: power, sample size, significance, multiple testing correction
- Bayesian inference vs. frequentist approach
- Random effects vs. fixed effects models
- Outlier detection: IQR method, Z-score, isolation forests

### 🔹 SQL Must-Haves

- Window functions: `ROW_NUMBER()`, `RANK()`, `LAG()`, `LEAD()`, `SUM() OVER`, `AVG() OVER`
- CTEs, subqueries, self-joins
- Aggregations, GROUP BY, HAVING
- Query optimization: indexes, execution plans, partitioning
- Deduplication strategies

### 🔹 Time-Series Fundamentals

- Stationarity (ADF test), differencing
- ARIMA/SARIMA components (p, d, q), seasonal decomposition (STL)
- Prophet: trend, seasonality, holiday effects
- Exponential smoothing (Holt-Winters)
- Hierarchical forecasting: top-down, bottom-up, middle-out
- Train/test split for time-series (no shuffling — always anchor in time)
- Metrics: MAPE, sMAPE, WAPE, RMSE

### 🔹 Generative AI / LLM Basics (critical for 2026 GM roles)

- Transformer architecture: attention, positional encoding, encoder/decoder
- RAG pipeline: chunking → embeddings → vector store → retrieval → generation
- Prompt engineering: zero-shot, few-shot, chain-of-thought
- Fine-tuning vs. RAG trade-offs
- Hallucination mitigation strategies
- Vector databases: Pinecone, FAISS, ChromaDB, Weaviate
- Agentic AI: tool-calling, ReAct pattern, multi-agent orchestration (LangGraph, CrewAI)
- Evaluation: RAGAS, LLM-as-judge, groundedness, faithfulness scores

### 🔹 Operations Research (for optimization-heavy roles)

- Linear Programming (LP), Mixed Integer Programming (MIP)
- Constraint programming
- Network flow, knapsack, cutting stock problems
- Python tools: Google OR-Tools, PuLP, Gurobi/CPLEX APIs
- Heuristics: greedy, simulated annealing, genetic algorithms

### 🔹 MLOps / Cloud

- GCP services: Vertex AI (Training, Pipelines, Feature Store), BigQuery, Dataflow
- Model versioning and experiment tracking (MLflow, Vertex AI Experiments)
- Containerization: Docker, Kubernetes basics
- CI/CD for ML pipelines
- Model drift monitoring and retraining triggers
- Data lineage concepts

---

## 4. 30+ Interview Questions with Answers

---

### A. Machine Learning & Modeling

---

**Q1. Can you explain the ROC AUC curve and its significance?** *(Reported by multiple GM DS candidates — InterviewQuery)*

**Answer:** The ROC curve plots True Positive Rate (Sensitivity) vs. False Positive Rate (1 – Specificity) at different classification thresholds. AUC summarizes this into a single number: 1.0 = perfect classifier, 0.5 = random. It is threshold-agnostic, making it useful for comparing models. In CPG contexts (e.g., predicting which customers will churn or whether a promotion drives purchase), AUC helps you pick the model independent of the operating threshold, which you tune separately based on business cost of false positives vs. false negatives.

---

**Q2. How do you determine when to use precision vs. recall?** *(InterviewQuery — GM DS guide)*

**Answer:** It depends on the cost of each error type.

- **High precision priority** → false positives are expensive (e.g., sending costly promotions to non-buyers).
- **High recall priority** → false negatives are expensive (e.g., missing a demand spike that causes stockouts). At GM, a demand anomaly detector would prioritize recall (don't miss a real spike), while a targeted marketing model would prioritize precision (don't waste spend).

---

**Q3. Explain logistic regression and how it works. What are its assumptions?** *(Glassdoor — GM DS aptitude test)*

**Answer:** Logistic regression models the log-odds of a binary outcome as a linear combination of features, squashed through a sigmoid function to produce probabilities. Key assumptions: (1) binary or ordinal outcome, (2) linear relationship between log-odds and predictors, (3) no multicollinearity among features, (4) independence of observations, (5) large enough sample size. In contrast to linear regression, it does not require normally distributed residuals or homoscedasticity.

---

**Q4. What is the relationship between PCA and K-Means clustering?** *(Listed explicitly on InterviewQuery as a GM question)*

**Answer:** PCA reduces dimensionality by projecting data onto orthogonal principal components that maximize variance. Running K-Means after PCA can improve clustering because: (1) it reduces noise and collinearity in features, (2) Euclidean distance in PCA space is more meaningful, (3) it speeds up computation. The mathematical link is deep — K-Means in the original space is equivalent to K-Means in the space spanned by the top PCA components, assuming data is centered. Drawback: PCA components may be hard to interpret, making cluster labels less actionable for business stakeholders.

---

**Q5. Walk me through building a demand forecasting model.** *(Reported as a GM technical question — CleverPrep, LockedIn AI)*

**Answer (structured response):**

1. **Problem framing** — Define the forecasting horizon (daily/weekly/monthly), granularity (SKU × DC), and KPI (MAPE, WAPE).
2. **Data sources** — ERP/WMS for historical sales, promotional calendars, pricing data, external signals (weather, holidays, macro indicators).
3. **EDA** — Trend analysis, seasonality decomposition (STL), identify outlier promotions/events.
4. **Feature engineering** — Days-of-supply, lag features, rolling averages, promotion flags, price elasticity signals.
5. **Modeling** — Baseline (Naïve, Holt-Winters), then statistical (SARIMA, Prophet), then ML (LightGBM with lag features), then deep learning (N-BEATS, TFT) if data is sufficient.
6. **Validation** — Time-based train/test split (never shuffle). Evaluate MAPE, WAPE, bias.
7. **Deployment** — Batch inference pipeline (Vertex AI Pipelines), monitoring for concept drift, automated retraining trigger.
8. **Stakeholder output** — Aggregated forecast vs. actuals dashboard; confidence intervals.

---

**Q6. What do you do when the distribution of train and test data is different?** *(Glassdoor — GM Senior DS Codility final round)*

**Answer:** This is covariate shift or concept drift. Approaches:

- **Detect** — Compare feature distributions using KS-test, PSI (Population Stability Index), or MMD.
- **Reweight** — Apply importance weighting (e.g., density ratio estimation) to training samples to match test distribution.
- **Retrain** — Use a sliding window or expanding window strategy, so the model always trains on the most recent data.
- **Domain adaptation** — Techniques like CORAL or adversarial domain adaptation for severe shift. For time-series specifically, always use time-ordered splits; never random shuffling, as it artificially inflates performance metrics.

---

**Q7. Explain feature importance analysis and its methods.** *(InterviewQuery — GM question)*

**Answer:** Feature importance quantifies each predictor's contribution to model performance.

- **Tree-based (impurity/Gini)** — Fast but biased toward high-cardinality features.
- **Permutation importance** — Shuffle each feature and measure performance drop; model-agnostic, more reliable.
- **SHAP (SHapley Additive exPlanations)** — Theoretically grounded, shows directional impact per sample; preferred for stakeholder explainability.
- **Coefficient magnitude (linear models)** — After standardizing features. At GM, SHAP is valuable for explaining why a model predicts a specific SKU will have a demand spike to a supply chain planner who needs to act on it.

---

**Q8. Describe how you would build a customer/product churn or recommendation model.** *(InterviewQuery — GM ML question)*

**Answer:** For a recommendation system (relevant to GM's e-commerce and retail analytics):

1. **Content-based filtering** — Feature vectors from product attributes (ingredients, category, price) + cosine similarity.
2. **Collaborative filtering** — Matrix factorization (ALS, SVD) on purchase/click history.
3. **Hybrid** — Combine both; handle cold-start with content-based for new SKUs.
4. **Evaluation** — Offline: Precision@K, Recall@K, NDCG. Online: A/B test click-through rate, add-to-cart lift. Churn prediction: logistic regression / gradient boosting on engagement features; threshold tuned by business cost matrix.

---

**Q9. What is the difference between bagging and boosting?**

**Answer:** Both are ensemble methods that combine weak learners.

- **Bagging (Bootstrap Aggregating)** — Trains independent models on random subsamples (with replacement) and averages predictions. Reduces variance. Example: Random Forest.
- **Boosting** — Trains models sequentially; each corrects the errors of the prior one via re-weighting. Reduces bias and variance. Examples: XGBoost, LightGBM, AdaBoost. In practice at GM, XGBoost/LightGBM tends to outperform Random Forest on tabular CPG data with many engineered features; the trade-off is higher risk of overfitting and more hyperparameter tuning.

---

**Q10. How do you handle class imbalance in a classification problem?**

**Answer:** Strategies:

- **Resampling** — Oversample minority class (SMOTE, ADASYN) or undersample majority.
- **Class weights** — Pass `class_weight='balanced'` to sklearn models; XGBoost `scale_pos_weight`.
- **Threshold moving** — Train on raw probabilities, tune the classification threshold using the precision-recall curve.
- **Evaluation** — Use F1, AUC-PR, MCC instead of accuracy which is misleading on imbalanced data.
- **Algorithm choice** — Tree-based methods generally handle imbalance better than logistic regression.

---

### B. Statistics & Probability

---

**Q11. What are the assumptions of linear regression?** *(Glassdoor + InterviewQuery — confirmed GM aptitude test question)*

**Answer (LINE acronym):**

- **L**inearity — relationship between predictors and outcome is linear.
- **I**ndependence — observations are independent (critical for time-series; use Durbin-Watson test).
- **N**ormality — residuals are normally distributed (check Q-Q plot).
- **E**qual variance (Homoscedasticity) — residuals have constant variance (check residuals vs. fitted plot).
- **Bonus:** No multicollinearity among predictors (check VIF > 10 = concern).

---

**Q12. Explain hypothesis testing and A/B test design.** *(InterviewQuery — GM Statistics question)*

**Answer:** A/B testing involves:

1. **Null hypothesis (H₀)** — No difference between control and treatment.
2. **Significance level (α)** — Typically 0.05; probability of Type I error (false positive).
3. **Power (1 – β)** — Typically 0.80; probability of detecting a true effect.
4. **Sample size calculation** — Depends on effect size, α, power, and variance.
5. **Test statistic** — Two-sample t-test (continuous), z-test (proportions), chi-square (categorical).
6. **Multiple testing** — Bonferroni or Benjamini-Hochberg correction when testing multiple variants/metrics. At GM, A/B tests might evaluate whether a new promotional strategy increases sell-through rate or whether a UI change on a D2C site lifts conversion.

---

**Q13. What is the difference between random effects and fixed effects in modeling?** *(InterviewQuery — GM DS guide)*

**Answer:**

- **Fixed effects** — Assume the group-level variable has a fixed, non-random influence (e.g., region is always the same set of regions). Accounts for all between-group differences. Used in panel data econometrics.
- **Mixed effects (random effects)** — Assume group-level effects are drawn from a distribution; generalizes to unseen groups. More appropriate when groups are a sample from a larger population (e.g., stores in a market). At GM, a mixed-effects model for price elasticity would treat individual stores as random effects (generalize to new stores) while treating brand or category as fixed effects.

---

**Q14. Explain Bayesian vs. frequentist probability. When would you use each?** *(InterviewQuery — GM Bayesian probability question)*

**Answer:**

- **Frequentist** — Probability is the long-run frequency of an event; parameters are fixed; uses p-values and confidence intervals.
- **Bayesian** — Probability quantifies belief; prior knowledge + data → posterior distribution; produces credible intervals. **Use Bayesian when:** you have strong prior domain knowledge (e.g., historical promotion lift), small sample sizes, or need uncertainty quantification per-region/SKU. Use frequentist for large-scale A/B tests where p-values are the organizational standard.

---

**Q15. How do you handle outliers in production data?** *(Glassdoor — GM hiring manager round question)*

**Answer:** Three-step approach:

1. **Detect** — IQR method (>1.5×IQR), Z-score (>3σ), isolation forests (multivariate), DBSCAN-based detection.
2. **Investigate** — Is it a data quality issue (typo, system error) or a genuine extreme event (promotion, COVID disruption)?
3. **Treat** — Remove if error; cap/floor (Winsorize) if extreme but valid; flag as a feature (binary indicator) for tree-based models; use robust estimators (Huber regression) for linear models. In supply chain contexts, promotional outliers should be modeled explicitly, not removed.

---

### C. SQL & Data Engineering

---

**Q16. Write a SQL query to get the current salary for each employee (ETL deduplication).** *(Confirmed GM SQL question — InterviewQuery)*

```sql
SELECT first_name, last_name, salary
FROM (
    SELECT 
        first_name,
        last_name,
        salary,
        ROW_NUMBER() OVER (
            PARTITION BY first_name, last_name 
            ORDER BY id DESC
        ) AS rn
    FROM employees
) ranked
WHERE rn = 1;
```

*Key insight: Use `ROW_NUMBER()` with `ORDER BY id DESC` since ETL inserted new salary rows rather than updating, so the highest `id` = most recent.*

---

**Q17. Calculate a moving average in SQL.** *(InterviewQuery — confirmed GM SQL Easy question)*

```sql
SELECT
    product_id,
    sale_date,
    quantity,
    AVG(quantity) OVER (
        PARTITION BY product_id
        ORDER BY sale_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM daily_sales;
```

*Interviewers check: do you know the default frame changes when you add ORDER BY inside OVER? Always be explicit with `ROWS BETWEEN`.*

---

**Q18. Write a query for average monthly sales per product.** *(DataLemur — confirmed GM SQL question)*

```sql
SELECT
    product_id,
    DATE_TRUNC('month', sale_date) AS sale_month,
    AVG(quantity) AS avg_monthly_quantity
FROM sales
GROUP BY product_id, DATE_TRUNC('month', sale_date)
ORDER BY product_id, sale_month;
```

---

**Q19. How would you optimize a slow SQL query?** *(InterviewQuery — GM SQL Hard question)*

**Answer:** Systematic approach:

1. Run `EXPLAIN ANALYZE` / query execution plan to identify bottlenecks.
2. Add indexes on filter columns (WHERE) and join keys.
3. Avoid `SELECT *`; fetch only needed columns.
4. Push filters as early as possible (filter before JOIN).
5. Replace correlated subqueries with window functions or CTEs.
6. Use partitioning on large tables (e.g., partition by date in BigQuery).
7. Materialize intermediate results with temp tables when a CTE is called multiple times.
8. Use approximate aggregates (`APPROX_COUNT_DISTINCT`) when exact values aren't needed.

---

**Q20. What is a cross-join and when would you use it?** *(DataLemur — confirmed GM SQL question)*

**Answer:** A CROSS JOIN returns the Cartesian product of two tables — every row in Table A paired with every row in Table B. Use cases in DS: generating all customer × product combinations for a recommendation baseline, creating a feature store scaffold for time-series (all SKU × date combinations), or building a scoring table to run a trained model across all possible input combinations. Warning: combinatorial explosion — use `LIMIT` or filter downstream.

---

### D. Time-Series Forecasting & Supply Chain

---

**Q21. What are ARIMA/SARIMA? How do you select p, d, q parameters?** *(Confirmed in GM demand planning interviews — Glassdoor India Dec 2023, May 2026)*

**Answer:**

- **ARIMA(p, d, q):** p = autoregressive lags, d = differencing order (for stationarity), q = moving average terms.
- **SARIMA(p, d, q)(P, D, Q)\[s\]:** adds seasonal components where s = seasonality period (e.g., 52 for weekly data). **Parameter selection:**

1. Check stationarity with ADF/KPSS test → determine `d`.
2. Plot ACF → guides `q` (cuts off after lag q).
3. Plot PACF → guides `p` (cuts off after lag p).
4. Use AIC/BIC for model comparison; `auto_arima` (pmdarima) for automated search. **Modern alternatives:** Prophet (handles holidays natively), LightGBM with lag features (often outperforms ARIMA on CPG data), N-BEATS, TFT (Temporal Fusion Transformer).

---

**Q22. What is WAPE, and why is it preferred over MAPE in supply chain?** *(Confirmed in GM Supply Chain DS interview — LockedIn AI May 2026)*

**Answer:**

- **MAPE** = Mean Absolute Percentage Error = `mean(|actual – forecast| / actual) × 100`. Problem: undefined or explodes when actual = 0 (very common for new SKUs or off-shelf periods in CPG).
- **WAPE** = Weighted Absolute Percentage Error = `sum(|actual – forecast|) / sum(actual) × 100`. It weights errors by volume, so high-volume SKUs drive the metric. It is never undefined (as long as total actuals > 0) and aligns better with business impact (a 10% error on a 1M unit SKU matters more than on a 10-unit SKU).

---

**Q23. How do you handle seasonality and promotional effects in demand forecasting?** *(Multiple GM DS interviews — supply chain track)*

**Answer:**

- **Seasonality** — STL decomposition to isolate it; include seasonal dummies or Fourier terms (for multiple seasonalities) in ML models.
- **Promotional effects** — Create binary `is_promo` flag, `promo_depth` (% discount), `promo_type` (BOGO, display, TPR) as features. Segment promo vs. non-promo periods in model evaluation. Use causal inference (DiD, synthetic control) to estimate true lift.
- **Holiday effects** — Prophet's holiday component; country-specific calendars.
- **New SKU / cold start** — Cluster similar SKUs and use their history; apply cross-series learning (hierarchical models, N-HiTS).

---

**Q24. Explain hierarchical forecasting. What are the trade-offs between top-down, bottom-up, and middle-out approaches?**

**Answer:**

- **Bottom-up** — Forecast at lowest granularity (SKU × store), aggregate up. Most granular, captures local patterns, but noisiest.
- **Top-down** — Forecast at aggregate level, disaggregate using historical share. Stable but ignores local variation.
- **Middle-out** — Forecast at an intermediate level (brand × region), disaggregate down and aggregate up. Balances stability and granularity. Often used in CPG for brand-level planning.
- **Optimal reconciliation (MinT)** — Simultaneously reconcile all levels to ensure coherence; theoretically optimal but computationally heavy. At GM, middle-out at brand × division level with reconciliation is common for integrated business planning (IBP).

---

**Q25. What is random train-test split inappropriate for time-series data?** *(Confirmed in GM Supply Chain DS technical screen — LockedIn AI May 2026)*

**Answer:** Randomly splitting time-series data causes **data leakage** — future data ends up in the training set, and the model implicitly learns patterns it couldn't know at prediction time. This inflates performance metrics. The correct approach: always use a **temporal split** (train on older data, test on more recent data). For cross-validation, use **TimeSeriesSplit** (rolling/expanding windows) where each fold's test set is always after its training set.

---

### E. Generative AI / LLM / RAG / Agentic AI

*(Validated against GM Sr. D&T Data Scientist JD, March 2026 — this is explicitly listed as an expert-level requirement)*

---

**Q26. What is RAG (Retrieval-Augmented Generation)? Walk through the architecture.** *(Directly listed as required expertise in GM Sr. D&T DS JD — March 2026)*

**Answer:** RAG grounds an LLM's responses in external, up-to-date knowledge. Pipeline:

1. **Ingestion** — Load documents (PDFs, wikis, databases), chunk into segments (fixed-size, semantic, or recursive splitting).
2. **Embedding** — Convert chunks to dense vectors using an embedding model (OpenAI `text-embedding-3-large`, BGE, E5).
3. **Vector indexing** — Store vectors in a vector database (Pinecone, FAISS, ChromaDB, Weaviate, AlloyDB for GCP).
4. **Retrieval** — On query, embed the question, retrieve top-k most similar chunks (cosine similarity or ANN search); optionally use hybrid search (dense + BM25 sparse).
5. **Generation** — Inject retrieved chunks into the LLM prompt as context; LLM generates a grounded answer.
6. **Evaluation** — Use RAGAS framework: faithfulness, answer relevancy, context precision, context recall.

**GM application:** Product specification Q&A bot, procurement contract intelligence, supply chain policy search.

---

**Q27. What is the difference between fine-tuning and RAG? When do you choose each?**

**Answer:**

|  | Fine-tuning | RAG |
| --- | --- | --- |
| **Updates model weights?** | Yes | No |
| **Best for** | Teaching new reasoning patterns, style, domain-specific format | Injecting specific, up-to-date facts |
| **Data requirement** | Labeled instruction-response pairs (hundreds–thousands) | Document corpus |
| **Hallucination risk** | Can bake in errors from training data | Lower; grounded in retrieved source |
| **Freshness** | Stale after training cutoff | Always current if index is updated |
| **Cost** | High (GPU compute) | Low (inference + retrieval) |

**Use RAG** when the knowledge base changes frequently (product catalogs, regulatory docs). **Use fine-tuning** when you need the model to follow a specific output format or reasoning style (e.g., always produce JSON in a specific schema for a downstream pipeline).

---

**Q28. What is an Agentic AI system? How does it differ from a standard LLM call?** *(Explicitly required in GM Sr. D&T Data Scientist JD — March 2026)*

**Answer:** An agent augments an LLM with **tools** (functions it can call: SQL queries, APIs, calculators, web search) and **memory** (conversation history, vector store, structured state). Instead of a single-turn prompt → response, the agent operates in a loop:

1. Observe (input/state)
2. Reason (LLM plans what to do next)
3. Act (call a tool)
4. Observe result → repeat until goal satisfied.

**Patterns:**

- **ReAct** — Reason + Act interleaved.
- **Plan-and-Execute** — Separate planner and executor LLMs.
- **Multi-agent orchestration** — Supervisor routes tasks to specialist sub-agents (LangGraph, CrewAI, AutoGen).

**GM application:** An agentic procurement assistant that retrieves supplier contracts, checks compliance rules, runs pricing queries against BigQuery, and drafts a PO recommendation.

---

**Q29. What are hallucinations in LLMs, and how do you mitigate them in production?**

**Answer:** Hallucinations occur when an LLM generates factually incorrect but plausible-sounding outputs. Mitigation strategies:

- **RAG with source citation** — Ground outputs in retrieved documents; require the model to cite which chunk supports its claim.
- **Constrained output schemas** — Use structured generation (JSON mode, function calling) to limit output space.
- **Temperature tuning** — Lower temperature (0.0–0.2) reduces creativity but also hallucination rate for fact-based tasks.
- **Self-consistency sampling** — Generate multiple outputs and take the majority answer.
- **LLM-as-judge evaluation** — Use a separate LLM to score groundedness and faithfulness.
- **Human-in-the-loop** — Flag low-confidence outputs for human review (especially in high-stakes supply chain decisions).

---

**Q30. How would you design an end-to-end RAG pipeline for GM's supply chain knowledge base?** *(System design question — aligns with GM JD requirement for "architecting high-quality Agentic AI and RAG workflows")*

**Answer (structured system design):**

```
1. DATA INGESTION
   Sources: SAP/ERP docs, supplier contracts, SOP wikis, internal reports
   Format: PDF, Word, structured tables
   Pipeline: GCS → Document AI (OCR for scans) → text extraction

2. CHUNKING STRATEGY
   - Recursive character splitter (500–800 tokens, 10% overlap)
   - Semantic chunking for narrative docs
   - Table-aware chunking (preserve table structure)

3. EMBEDDING & INDEXING
   Model: text-embedding-3-large or GCP's Vertex AI Embeddings
   Store: AlloyDB (GCP-native) or Pinecone
   Metadata: document type, source, date, category, brand

4. RETRIEVAL
   - Dense retrieval (cosine similarity, top-k=5)
   - Hybrid search: dense + BM25 sparse (for exact keyword matching of SKU codes)
   - Reranking: Cohere Rerank or cross-encoder for precision

5. GENERATION
   Model: Gemini 1.5 Pro (GCP VertexAI) or GPT-4o
   Prompt: System prompt with business ontology + retrieved chunks + query
   Output: Structured answer + cited source chunks

6. EVALUATION (RAGAS)
   Faithfulness, Answer Relevance, Context Precision, Context Recall

7. MONITORING
   Latency, retrieval quality (click-through on sources), user thumbs up/down
   Automated drift detection on query distribution

8. MLOPS
   Vertex AI Pipelines for re-indexing on doc updates
   Feature store for pre-computed embeddings of frequently accessed docs
```

---

**Q31. What are embeddings? How are they used in NLP and recommender systems?**

**Answer:** Embeddings are dense, low-dimensional vector representations of objects (words, sentences, products, users) where semantic similarity is reflected in geometric proximity (high cosine similarity = similar meaning). In NLP: Word2Vec, GloVe, BERT contextual embeddings. In recommender systems: collaborative filtering learns user and item embeddings jointly from interaction data (matrix factorization). At GM, product embeddings on ingredient profiles + consumer purchase embeddings enable a hybrid recommendation engine that works even for new products (content-based cold start).

---

### F. Python Coding

---

**Q32. Implement Mutual Information probability distribution in Python.** *(Confirmed GM Codility question — Glassdoor Senior DS interview)*

```python
import numpy as np
from collections import Counter

def mutual_information(x, y):
    """Compute mutual information between two discrete variables."""
    n = len(x)
    assert n == len(y), "x and y must have same length"
    
    # Joint distribution P(X, Y)
    joint = Counter(zip(x, y))
    # Marginal distributions
    px = Counter(x)
    py = Counter(y)
    
    mi = 0.0
    for (xi, yi), cnt in joint.items():
        p_xy = cnt / n
        p_x = px[xi] / n
        p_y = py[yi] / n
        mi += p_xy * np.log(p_xy / (p_x * p_y))
    
    return mi

# Example
x = [1, 0, 1, 1, 0, 0, 1, 0]
y = [1, 0, 1, 0, 0, 1, 1, 0]
print(f"MI: {mutual_information(x, y):.4f}")
```

---

**Q33. Write Python code to handle missing values in a production pipeline.** *(InterviewQuery — GM Python Easy question; Glassdoor — GM hiring manager round)*

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

def handle_missing(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    Handles missing values:
    - Numeric cols: median imputation (robust to outliers)
    - Categorical cols: mode imputation
    - High missingness (>40%): flag column dropped
    """
    df = df.copy()
    
    # Drop cols with >40% missing (configurable threshold)
    high_miss_cols = [c for c in df.columns if df[c].isna().mean() > 0.4]
    df.drop(columns=high_miss_cols, inplace=True)
    print(f"Dropped high-missingness columns: {high_miss_cols}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Numeric: median
    if numeric_cols:
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Categorical: most_frequent
    if cat_cols:
        imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imputer.fit_transform(df[cat_cols])
    
    return df
```

---

**Q34. Given a time-series dataframe, create lag features and a rolling-mean feature.** *(Supply chain DS technical screen — LockedIn AI May 2026)*

```python
import pandas as pd

def create_ts_features(df: pd.DataFrame, 
                        target_col: str, 
                        date_col: str,
                        group_col: str,
                        lags: list = [1, 2, 4, 8],
                        windows: list = [4, 8]) -> pd.DataFrame:
    """
    Creates lag and rolling features for time-series.
    Groups by SKU/product to avoid leaking across entities.
    """
    df = df.sort_values([group_col, date_col]).copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = (
            df.groupby(group_col)[target_col].shift(lag)
        )
    
    for window in windows:
        df[f'{target_col}_roll_mean_{window}'] = (
            df.groupby(group_col)[target_col]
              .transform(lambda x: x.shift(1).rolling(window).mean())
        )
    
    # Drop rows with NaN from lag/rolling
    df.dropna(inplace=True)
    return df
```

---

### G. Behavioral & Stakeholder Management

---

**Q35. Describe a time when you had to make a decision with incomplete information.** *(#1 reported GM DS interview question — Glassdoor July 2026)*

**STAR framework answer:**

- **Situation:** During a demand planning cycle, the ERP system migration left 3 months of historical sales data missing for 200 SKUs, right before the annual operating plan was due.
- **Task:** Build demand forecasts anyway, with acceptable confidence for the finance team.
- **Action:** Applied a combination of (1) similar-SKU proxy (cluster SKUs by attribute similarity, borrow history from closest cluster), (2) Bayesian priors from category-level trend, (3) flagged high-uncertainty SKUs with wider confidence intervals. Presented the approach transparently to the finance team with explicit uncertainty ranges.
- **Result:** Forecasts were accepted. The approach was later standardized as the "cold-start" protocol for new SKU launches.

---

**Q36. How do you handle bad/dirty data in a production model?** *(Glassdoor — GM Senior DS hiring manager round question)*

**Answer:** Multi-layer defense:

1. **Schema validation** — Enforce data types, range checks, required fields at ingestion (Great Expectations, Deequ).
2. **Statistical checks** — Monitor distribution shifts (PSI), Z-score for sudden spikes/drops.
3. **Business rules** — Domain-specific constraints (e.g., units sold cannot be negative; lead time cannot be zero).
4. **Fallback logic** — If a data quality flag fires, fall back to previous forecast or a simpler baseline rather than producing a corrupted prediction.
5. **Alerting** — PagerDuty / Slack alerts for data quality SLAs.
6. **Root-cause investigation** — Work with data engineering to fix upstream source.

---

**Q37. Describe a project where you employed optimization (OR) techniques to solve a business problem.** *(Glassdoor — GM Lead Data Scientist – Optimization, September 2025)*

**STAR example:**

- **Situation:** Distribution network had 15 warehouses serving 500+ stores; routing was manual, resulting in 22% underutilization and high freight cost.
- **Task:** Build an automated supply allocation and routing optimization.
- **Action:** Formulated as a Mixed Integer Program (MIP): minimize total freight cost subject to capacity, service-level, and shelf-life constraints. Solved using Google OR-Tools with a branch-and-bound solver for exact solution up to \~200 nodes; switched to a greedy + simulated annealing heuristic for larger instances.
- **Result:** 14% reduction in freight cost; solver ran nightly as a batch job on Vertex AI.

---

**Q38. How do you prioritize projects when working with multiple stakeholders?** *(InterviewQuery — GM business acumen question)*

**Answer:** Framework:

1. **Impact × Feasibility matrix** — Estimate business value (revenue impact, cost savings, risk reduction) × implementation effort.
2. **Stakeholder alignment** — Use a RACI matrix; align with the senior sponsor on priorities quarterly.
3. **Agile sprint planning** — Break projects into 2-week sprints; ensure quick-win deliverables visible every sprint to maintain stakeholder trust.
4. **Transparent communication** — Maintain a living roadmap in Confluence/JIRA; flag blockers early.
5. **Say no strategically** — Push back politely on low-value requests that dilute bandwidth from high-impact work; quantify the trade-off.

---

**Q39. How do you communicate a complex model to a non-technical stakeholder?** *(Multiple GM DS behavioral questions)*

**Answer:** Key principles:

- **Lead with the business outcome**, not the algorithm. "This model predicts which promotions will drive more than 15% incremental volume" — not "I trained an XGBoost classifier with 200 estimators."
- **Visualize, don't tabulate.** SHAP waterfall plots explain why the model made a specific prediction in terms the stakeholder owns (e.g., "this SKU's forecast is higher because it's a summer product and we're in June").
- **Acknowledge uncertainty.** Give confidence intervals; say "this forecast has ±12% error at the SKU level" rather than presenting a point estimate as truth.
- **Practice with analogies.** "The model is like an experienced demand planner who has read the purchase history of 10,000 SKUs — it spots seasonal patterns automatically."

---

**Q40. Walk through a pre-existing data science notebook and assess it.** *(Glassdoor — confirmed GM Senior DS on-site question)*

**What interviewers look for:**

1. **Data integrity checks** — Are NaNs handled? Are there data leakage issues?
2. **EDA quality** — Are distributions examined? Correlations checked? Class balance verified?
3. **Feature engineering** — Are features engineered thoughtfully or just raw columns?
4. **Modeling rigor** — Is train/test split done correctly? Is cross-validation used?
5. **Evaluation completeness** — Is the right metric for the business problem used? Are baselines compared?
6. **Reproducibility** — Is random seed set? Are library versions pinned?
7. **Code quality** — Is the code modular, documented, and DRY?
8. **Business interpretation** — Does the notebook conclude with actionable insights, not just metrics?

**Red flags to call out:** Data leakage (scaling before split), accuracy on imbalanced data, no baseline comparison, magic numbers hardcoded, no discussion of model limitations.

---

## 5. Tips to Ace the GM Interview

### ✅ DO's

- **Know the GM business context** — Be ready to map your DS experience to CPG use cases: demand forecasting, price optimization, market mix modeling, supply chain planning, consumer insights.
- **Use STAR format** — All behavioral questions expect Situation → Task → Action → Result with quantified outcomes.
- **Demonstrate stakeholder fluency** — GM explicitly values ability to "educate stakeholders on GCP analytic practices" and "develop trust with business leaders." Show you can translate.
- **Prepare 2–3 deep-dive projects** — They will ask you to go deep on your resume projects; prepare for "what went wrong?" and "what would you do differently?"
- **Know GCP** — VertexAI, BigQuery, Cloud Composer (Airflow), Dataflow. Even if you've used AWS/Azure, be able to map equivalences.
- **Read the JD carefully** — The March 2026 JD explicitly lists GenAI/Agentic AI as an expert-level requirement. If you have this experience, lead with it.
- **Practice Codility-style questions** — GM uses Codility for Python/SQL coding rounds. Practice window functions, time-series feature engineering, and algorithm questions (mutual information, probability distributions).

### ❌ DON'Ts

- Don't use only accuracy as a metric — always acknowledge class imbalance and pick the right metric for the problem.
- Don't skip baselines — always compare your model against a naïve/simple baseline.
- Don't conflate correlation and causation — especially in A/B testing and promotion effectiveness discussions.
- Don't oversell model precision — GM stakeholders act on forecasts; acknowledge uncertainty.
- Don't ignore MLOps — model deployment and monitoring are as important as model building in a senior role.

---

## 📚 Quick Reference: Key Frameworks & Tools

| Category | Tools / Frameworks |
| --- | --- |
| ML | scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch |
| Time-Series | statsmodels (ARIMA), Prophet, Darts, NeuralForecast, pmdarima |
| GenAI / LLM | LangChain, LangGraph, LlamaIndex, Vertex AI (Gemini), OpenAI API |
| Vector Stores | FAISS, Pinecone, ChromaDB, AlloyDB (GCP) |
| OR / Optimization | Google OR-Tools, Gurobi, PuLP, AIMMS |
| Cloud | GCP: VertexAI, BigQuery, GCS, Cloud Composer, Dataflow |
| MLOps | MLflow, Vertex AI Experiments, Docker, Kubernetes, Airflow |
| SQL | BigQuery SQL, PostgreSQL, window functions, CTEs |
| Evaluation | RAGAS (RAG), SHAP (ML explainability), Great Expectations (data quality) |
| Experiment Tracking | MLflow, W&B (Weights & Biases), Vertex AI Experiments |

---

*Last updated: November 2026 | Compiled from Glassdoor, InterviewQuery, LockedIn AI, General Mills Careers JDs (2025–2026), and candidate interview reports.*