# General Mills & LSEG Data Scientist Interview Preparation Suite

### Branch: `int_mill`
### Target Companies: General Mills India (Powai GIC) & London Stock Exchange Group (LSEG India)
### Target Roles: Senior Data Scientist / Lead Data Scientist / Big Data ML Architect

---

## Overview

This repository contains a comprehensive, production-grade interview preparation curriculum designed for Senior and Lead Data Scientist technical rounds, Codility/take-home challenges, and system design interviews in India.

All 12 modules include mathematically rigorous derivations, complete and runnable code solutions, and real-world case studies reflecting actual interview rounds at General Mills (CPG & Supply Chain Analytics) and LSEG (Financial Markets & Quantitative Infrastructure).

---

## 📂 Curriculum Index

| # | Module | Core Topics & Focus |
|---|---|---|
| **00** | [Master Interview Guide](interview_prep/00_master_guide.md) | Company profiles, Azure/Databricks vs AWS/Snowflake stacks, 5-stage interview loops, India CTC benchmarks (₹30L–₹65L+), 4-week preparation plan. |
| **01** | [Pandas & NumPy Operations](interview_prep/01_pandas_numpy_operations.md) | 22 live-coding problems: complex `groupby`, `merge_asof`, reshaping/pivots, missing value imputation, vectorized broadcasting, and gotchas. |
| **02** | [SQL & Database Operations](interview_prep/02_sql_database_operations.md) | 26 advanced SQL scenarios: window functions (`ROW_NUMBER`, `DENSE_RANK`, `LEAD`/`LAG`), recursive CTEs, FMCG DOI, LSEG wash-trading, and SQLAlchemy chunking. |
| **03** | [ML Algorithms from Scratch](interview_prep/03_ml_algorithms_implementation.md) | 24 algorithms implemented from scratch in NumPy: Ridge/Logistic regression, CART Decision Trees, K-Means++, PCA (SVD), Adam, ROC/PR-AUC, SMOTE, and WoE/IV scorecards. |
| **04** | [Statistics, Probability & Econometrics](interview_prep/04_statistics_probability.md) | 25 questions: hypothesis testing, A/B testing pipeline (CUPED, SRM, multiple testing corrections), price elasticity log-log OLS, stationarity (ADF/KPSS), VaR/CVaR, and GARCH(1,1). |
| **05** | [Python DSA & Engineering Patterns](interview_prep/05_python_coding_dsa.md) | 21 questions: generators for 50GB CSV streaming, custom retry decorators, OOP ModelRegistry, downcasting memory optimization, monotonic deques, and Colab GPU patterns. |
| **06** | [Domain Case Studies (FMCG & Finance)](interview_prep/06_domain_case_studies.md) | 22 end-to-end case studies with complete Python code:<br>• **11 General Mills:** Hierarchical SKU demand forecasting, price elasticity, MMM with Adstock/Hill, phantom inventory.<br>• **11 LSEG:** Multi-asset VaR/CVaR, bond duration/convexity, ESG controversy tilt, order book imbalance (OBI). |
| **07** | [Deep Learning, NLP & MLOps](interview_prep/07_deep_learning_nlp_advanced.md) | 24 questions: PyTorch backprop from scratch, Multi-Head Attention, FinBERT earnings call sentiment, multivariate LSTM demand forecasting, planogram CV, and PSI/KS drift detection. |
| **08** | [Pythonic Programming Fundamentals](interview_prep/08_pythonic_programming_fundamentals.md) | 23 advanced questions: dunder data model (`__getitem__`, `__len__`, `__hash__`), custom descriptors, metaclasses, `typing.Protocol`, memory internals, GIL benchmarks, and `asyncio.Semaphore`. |
| **09** | [PySpark & Big Data Operations](interview_prep/09_pyspark_big_data_operations.md) | 20 distributed data operations: Catalyst/Tungsten internals, salting data skew joins, Broadcast Hash Join, window aggregates, `pandas_udf` with Apache Arrow, Delta Lake ACID merges, and OOM debugging. |
| **10** | [ML System Design & Architecture](interview_prep/10_ml_system_design_architecture.md) | 6 enterprise system designs with Mermaid blueprints: Demand Sensing, Retail Media RGM, Sub-millisecond Trade Surveillance, StarMine Factor Platform, Financial RAG, and Feature Stores. |
| **11** | [Behavioral & Leadership Playbook](interview_prep/11_behavioral_leadership_case_interviews.md) | Executive STAR framework, cross-functional conflict resolution, pushing back on deadlines, 4-tier code review rubric, and India compensation negotiation strategy. |

---

## 🛠️ Tech Stack Covered

* **Languages & Core:** Python 3.10+, SQL (PostgreSQL / Spark SQL), PySpark.
* **Data & Compute:** Pandas, NumPy, SciPy, Statsmodels, Delta Lake, Databricks.
* **Machine Learning & DL:** Scikit-Learn, LightGBM, PyTorch, Hugging Face Transformers.
* **MLOps & Streaming:** MLflow, Feast, Apache Flink, Kafka, Triton Inference Server, FastAPI, ONNX Runtime.

---

*Authored for Senior & Lead Data Science recruitment loops.*
