# Phase 3 – Machine Learning for Insurance (Days 71–120)

**Theme:** The Data Science Revolution – Beyond Linearity

---

### Days 71–72 – ML Foundations & Insurance Nuances (SOA Exam PA)
*   **Topic:** The ML Workflow
*   **Content:**
    *   **Supervised vs. Unsupervised:** Labels (Fraud/No Fraud) vs. Patterns (Segmentation).
    *   **Splitting:** Train/Test vs. Time-based splitting (Crucial for insurance).
    *   **Metrics:** RMSE/MAE (Regression), AUC/LogLoss (Classification), Gini/Lift (Ranking).
*   **Interview Pointers:**
    *   "Why is random splitting bad for claims data?" (Data leakage – claims from the same policy/event might end up in both sets).
*   **Challenges:** Imbalanced data (Fraud is <1%).

### Days 73–75 – Tree-based Models for Frequency & Severity
*   **Topic:** Non-linear Pricing
*   **Content:**
    *   **Decision Trees:** Visual, interpretable, but prone to overfitting.
    *   **Random Forests:** Bagging to reduce variance.
    *   **Gradient Boosting (XGBoost/LightGBM/CatBoost):** Boosting to reduce bias. The gold standard for tabular insurance data.
    *   **Objective Functions:** Poisson Loss, Gamma Loss, Tweedie Loss.
*   **Interview Pointers:**
    *   "How do you handle exposure in an XGBoost frequency model?" (Use `base_margin` or `ln(exposure)` as an offset).
*   **Tricky Parts:** Monotonicity constraints (Rate must increase with points/accidents).

### Days 76–78 – Neural Nets & Interpretable Deep Pricing
*   **Topic:** Deep Learning in Actuarial Science
*   **Content:**
    *   **Embeddings:** Handling high-cardinality categoricals (Zip Code, Vehicle Model).
    *   **Combined Models:** Multi-output networks (Frequency & Severity heads).
    *   **Interpretability:** CANN (Combined Actuarial Neural Networks) – GLM skip connection + NN residual.
*   **Interview Pointers:**
    *   "Why are Neural Networks hard to get approved by regulators?" (Black box nature).
*   **Data Requirements:** Large datasets (>100k rows).

### Days 79–81 – Fraud Detection (Classification)
*   **Topic:** Finding the Needle
*   **Content:**
    *   **Types:** Hard Fraud (Staged accidents) vs. Soft Fraud (Padding claims).
    *   **Techniques:**
        *   Supervised: XGBoost on past confirmed fraud.
        *   Unsupervised: Isolation Forests, Autoencoders (Reconstruction error).
    *   **Social Network Analysis (SNA):** Graph features (Cycles, cliques).
*   **Interview Pointers:**
    *   "How do you evaluate a fraud model if you don't have many labeled fraud cases?" (Precision at Top K, feedback loops).
*   **Challenges:** Adversarial behavior (Fraudsters adapt).

### Days 82–84 – Litigation & High-Severity Claim Modelling
*   **Topic:** Predicting the Boom
*   **Content:**
    *   **Propensity Models:** Binary classification (Will this claim go to court?).
    *   **NLP Features:** Mining adjuster notes ("Attorney", "Representation", "Angry").
    *   **Quantile Regression:** Predicting the 95th percentile of severity.
*   **Interview Pointers:**
    *   "How does early litigation detection save money?" (Early settlement, assigning senior adjusters).
*   **Data Requirements:** Unstructured text data (Notes).

### Days 85–87 – Claim Lifecycle & Severity Development
*   **Topic:** Granular Reserving
*   **Content:**
    *   **Hierarchical Models:** Predicting individual claim transactions.
    *   **Survival Analysis for Claims:** Time to Settlement, Time to Reopen.
    *   **Machine Learning Reserving:** Using GBMs to predict IBNR at the claim level.
*   **Interview Pointers:**
    *   "What is the advantage of individual claim reserving over triangles?" (Captures mix changes, e.g., more litigated claims).

### Days 88–90 – Customer Churn & Renewal Modelling
*   **Topic:** Retention Analytics
*   **Content:**
    *   **Churn Prediction:** Logistic Regression / GBM.
    *   **Features:** Rate change (%), Tenure, Claims history, Competitor rates.
    *   **Survival Analysis:** Cox Proportional Hazards for time-to-churn.
*   **Interview Pointers:**
    *   "Is a price decrease always the best way to save a customer?" (No, some are price inelastic).
*   **Tricky Parts:** Distinguishing "Death" (Lapse) from "Censoring" (Still active).

### Days 91–93 – Customer Lifetime Value (LTV/CLV)
*   **Topic:** Long-term Value
*   **Content:**
    *   **Formula:** $LTV = \sum \frac{Premium_t - Claims_t - Expense_t}{(1+i)^t} \times P(\text{Active}_t)$.
    *   **Predictive CLV:** ML models to predict future margins and retention.
    *   **Use Cases:** Acquisition cost caps (CAC), tiered service.
*   **Interview Pointers:**
    *   "How do you use LTV in marketing?" (Bid more for high-LTV segments).
*   **Challenges:** Predicting claims far into the future.

### Days 94–96 – Customer Segmentation (Clustering)
*   **Topic:** Unsupervised Grouping
*   **Content:**
    *   **Algorithms:** K-Means, DBSCAN, GMM.
    *   **Features:** Demographics, Behavior, Risk profile.
    *   **Applications:** Marketing personas, risk pools.
*   **Interview Pointers:**
    *   "How do you determine the optimal number of clusters?" (Elbow method, Silhouette score, Business utility).

### Days 97–99 – Recommendation Systems & Personalization
*   **Topic:** Cross-sell / Up-sell
*   **Content:**
    *   **Collaborative Filtering:** "Users like you bought..."
    *   **Content-based:** "You have a house, you need flood insurance."
    *   **Next Best Action (NBA):** Propensity models for every product.
*   **Interview Pointers:**
    *   "How do you handle the 'Cold Start' problem?" (Use demographic proxies).
*   **Data Requirements:** Product holding matrix.

### Days 100–102 – Marketing Mix Modelling (MMM) & Attribution
*   **Topic:** Optimizing Spend
*   **Content:**
    *   **MMM:** Time-series regression (Sales ~ TV + Digital + Seasonality).
    *   **Attribution:** Multi-touch (First click, Last click, Shapley values).
    *   **Adstock:** Decay effect of advertising.
*   **Interview Pointers:**
    *   "Difference between MMM and Attribution?" (MMM = Macro/Strategic, Attribution = Micro/Tactical).

### Days 103–105 – Uplift Modelling
*   **Topic:** Causal Inference
*   **Content:**
    *   **Concept:** Predicting the *change* in behavior due to treatment (e.g., discount).
    *   **Quadrants:** Persuadables, Sure Things, Lost Causes, Sleeping Dogs.
    *   **Models:** T-Learner, S-Learner, X-Learner (CausalML).
*   **Interview Pointers:**
    *   "Why not just target people with high churn probability?" (You might wake sleeping dogs – people who would stay but leave if reminded/annoyed).

### Days 106–108 – Advanced Loss & Tail Modelling (EVT + ML)
*   **Topic:** Catastrophes
*   **Content:**
    *   **EVT:** Peaks Over Threshold (POT), Generalized Pareto Distribution (GPD).
    *   **Deep EVT:** Neural networks estimating GPD parameters.
*   **Interview Pointers:**
    *   "How do you model 1-in-100 year events with 10 years of data?" (EVT theory extrapolates the tail).

### Days 109–111 – Integrated Pricing + Reserving + Capital
*   **Topic:** Holistic Modeling
*   **Content:**
    *   **One Model to Rule Them All:** Using the same granular simulations for pricing, reserving, and capital.
    *   **Feedback Loops:** Reserving errors feeding back into pricing indications.
*   **Interview Pointers:**
    *   "Why are pricing and reserving often siloed?" (Different timelines, different data aggregations).

### Days 112–114 – Data Engineering & Pipelines
*   **Topic:** MLOps
*   **Content:**
    *   **Feature Stores:** Consistent features for training and serving (Feast).
    *   **Pipelines:** Airflow/Dagster for ETL.
    *   **Data Quality:** Great Expectations tests.
*   **Interview Pointers:**
    *   "How do you prevent training-serving skew?" (Use a Feature Store).

### Days 115–117 – Model Governance & Monitoring
*   **Topic:** Keeping it Safe
*   **Content:**
    *   **Drift:** Data Drift (Input distribution changes) vs. Concept Drift (Relationship changes).
    *   **Monitoring:** PSI (Population Stability Index), CSI.
    *   **Explainability:** SHAP, LIME.
*   **Interview Pointers:**
    *   "What do you do if your model's PSI > 0.2?" (Investigate, likely retrain).

### Days 118–120 – Phase 3 Capstone Planning
*   **Project:** ML System Design
*   **Task:** Select a track (Fraud, Pricing, or LTV) and design the full system.
    *   **Data:** Sources, features, labels.
    *   **Model:** Algorithm, objective, metrics.
    *   **Deployment:** API vs. Batch, monitoring.
*   **Deliverable:** Design Document (1-pager).

