# 🔍 Resume Project Drill-Down: EXL / CVS Health (2016 - 2022)
### Extreme Deep Dive for Huge Solutions Architect ML/AI Interview

---

> [!IMPORTANT]
> The EXL experience is your **Scale and Statistics Foundation**. While Chubb is your GenAI showcase, EXL proves you can handle massive datasets (2M+ customers, PySpark, Dataproc) and rigorous biostatistics (Survival Analysis). Huge needs architects who understand data engineering just as well as prompt engineering.

---

## ⚕️ EXL PROJECT 1: Customer Lifetime Value (CLV) Prediction at Scale

**Resume Line:** *"Developed distributed Random Forest model using PySpark to predict CLV for 2M+ prospects... Utilized GCP Dataproc for orchestrating scalable model training and scoring, reducing processing time by 70%."*

### 🔴 1. Architecture: Distributed ML on GCP Dataproc
**The Interview Question:** "Walk me through the Dataproc architecture. Why did you choose PySpark over standard Python libraries, and how exactly did you achieve a 70% reduction in processing time?"

**Deep Dive Architecture:**
*   **The Bottleneck (Before):** The legacy CLV model was a Pandas/Scikit-learn script running on a massive vertically-scaled on-premise server. For 2 million customers, calculating a 3-year historical feature matrix (e.g., rolling averages of medical claims, pharmacy refill rates) caused constant Out-Of-Memory (OOM) crashes. It took 3 days to score the base.
*   **The Cloud-Native Solution (After):**
    1.  **Storage:** Raw claims data and member demographics resided in GCP Cloud Storage (GCS) and BigQuery.
    2.  **Compute Infrastructure:** We orchestrated an ephemeral **GCP Dataproc** cluster using Airflow. We provisioned 1 Master Node and dynamically scaled Preemptible (Spot) Worker Nodes to keep costs low.
    3.  **Distributed Feature Engineering (Spark SQL):** We rewrote the feature pipelines using PySpark DataFrames. This allowed transformations (like calculating the standard deviation of a member's copays over 24 months) to be computed in parallel across the worker nodes using Resilient Distributed Datasets (RDDs) under the hood, completely bypassing single-machine memory limits.
    4.  **Distributed Modeling (Spark MLlib):** We replaced the Scikit-learn Random Forest with `pyspark.ml.classification.RandomForestClassifier`.
    5.  **Output:** Predictions were written directly back to a BigQuery partitioned table for downstream marketing use. The cluster was immediately spun down.
*   **The Result:** By parallelizing the workload and utilizing cloud compute elasticity, we reduced the end-to-end execution time from 72 hours down to ~20 hours (a 70% reduction), allowing us to score the entire US prospect base weekly instead of monthly.

### 🔴 2. Code Implementation Mental Model
**The Interview Question:** "Show me the PySpark logic for training the distributed model."

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark Session (managed by Dataproc)
spark = SparkSession.builder.appName("CVS_CLV_Prediction").getOrCreate()

# 1. Load Data from BigQuery
df = spark.read.format("bigquery").load("project.dataset.clv_features_vw")

# 2. Feature Engineering Assembly
# Assuming categorical features are already StringIndexed
feature_cols = ['age', 'chronic_condition_count', 'pharmacy_spend_12m', 'er_visits_12m', 'risk_score_tier']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 3. Model Definition
# PySpark distributes the training of the 200 trees across the Dataproc worker nodes
rf = RandomForestRegressor(featuresCol="features", labelCol="actual_clv_3yr", numTrees=200, maxDepth=10)

# 4. Pipeline Execution
pipeline = Pipeline(stages=[assembler, rf])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_df)

# 5. Evaluation
predictions = model.transform(test_df)
evaluator = RegressionEvaluator(labelCol="actual_clv_3yr", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")
```

### 🔴 3. Cross-Examination & Trap Questions
*   **Trap Q: "Why Random Forest instead of XGBoost for CLV? XGBoost almost always wins Kaggle competitions."**
    *   *Defense:* Two reasons: Infrastructure maturity and Interpretability. At the time this was built, native distributed XGBoost on PySpark was highly unstable compared to Spark MLlib's battle-tested Random Forest. Secondly, health insurance CLV requires extreme explainability for actuaries. While SHAP exists for both, the variance reduction of Bagging (Random Forest) provided more stable feature importance over time compared to Boosting (XGBoost), which was highly sensitive to data drifts in the claims data.
*   **Q: "How do you define CLV for health insurance? It's not like e-commerce where they buy a shirt."**
    *   *Defense:* Exactly. In retail, CLV = (Average Order Value) x (Purchase Frequency) x (Lifespan). In health insurance (Aetna), the revenue is mostly fixed (the monthly premium). Therefore, CLV is an inverse optimization problem: **CLV = (Fixed Premiums over Lifespan) - (Predicted Medical Loss Ratio / Claims Cost)**. The model was fundamentally predicting future medical utilization (cost) and churn probability (lifespan).

---

## ⚕️ EXL PROJECT 2: Patient Readmission Risk Prediction (Deep Learning & Clinical NLP)

**Resume Line:** *"Built production ML pipeline for readmission risk prediction incorporating clinical notes; leveraged PyTorch with BERT for rich text feature extraction... improving model AUC from 0.82 to 0.89."*

### 🔴 1. Architecture: Multimodal Fusion
**The Interview Question:** "How did you combine unstructured clinical text with structured tabular claims data to predict readmission?"

**Deep Dive Answer:**
We used an early "Multimodal Fusion" architecture (Late Fusion).
1.  **Tabular Pipeline:** Structured claims data (Age, Gender, ICD-10 diagnosis codes, CPT procedure codes, Length of Stay) was pushed through a standard Multi-Layer Perceptron (MLP) to generate a dense representation.
2.  **NLP Pipeline:** Discharge summaries are messy. We used a domain-adapted BERT model (similar to ClinicalBERT) implemented in PyTorch. We fed the discharge notes into BERT, and extracted the `[CLS]` token embedding from the final hidden layer. This 768-dimensional vector mathematically represented the clinical narrative (capturing concepts like "patient lacks transportation" or "medication non-adherence" which don't exist in billing codes).
3.  **Fusion Layer:** We concatenated the tabular dense representation and the BERT `[CLS]` embedding. This combined vector was passed through a final set of fully connected layers with a Sigmoid activation to predict the probability of 30-day readmission.
4.  **Impact:** This addition of unstructured data drove the massive AUC improvement from 0.82 to 0.89.

### 🔴 2. Trap Questions
*   **Trap Q: "Did you fine-tune the BERT model end-to-end with the tabular data, or use it as a frozen feature extractor?"**
    *   *Defense:* We used it as a **frozen feature extractor** initially. End-to-end fine-tuning of BERT alongside an MLP requires massive GPU memory and often causes the MLP to overfit before BERT converges. We extracted the embeddings offline, stored them, and trained the lightweight fusion network rapidly. We later explored parameter-efficient tuning (similar to my Chubb work), but the frozen embeddings gave us the 0.89 AUC baseline.

---

## ⚕️ EXL PROJECT 3: Survival Analysis & Predictive Analytics

**Resume Line:** *"Implemented Kaplan-Meier estimator and Cox proportional hazards model for patient outcomes and customer attrition analysis... developed custom feature importance method combining SHAP values and permutation importance."*

### 🔴 1. The Mathematics of Survival Analysis
**The Interview Question:** "Why use Cox Proportional Hazards for customer churn instead of a standard logistic regression predicting churn Yes/No?"

**Deep Dive Answer:**
Standard logistic regression only answers "Will the customer churn?" within a fixed window. It treats a customer who churns on Day 2 exactly the same as a customer who churns on Day 364. It also cannot handle **Right-Censored Data** (customers who haven't churned *yet*, but might tomorrow).

**Survival Analysis** answers a better question: "When will the customer churn?"

1.  **Kaplan-Meier (Non-Parametric):** We used this to plot the baseline survival curve. It estimates the survival probability $S(t)$ over time. We used the Log-Rank test to compare curves (e.g., "Do members on the Silver plan survive longer than those on the Bronze plan?").
2.  **Cox Proportional Hazards (Semi-Parametric):** We used this to understand the *drivers* of churn. The model evaluates the Hazard Rate $h(t)$—the risk of churning at time $t$, given survival up to time $t$.
    $$ h(t|X) = h_0(t) \times \exp(\beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p) $$
    Where $h_0(t)$ is the baseline hazard (left unspecified, hence semi-parametric) and the exponential term represents the covariates (features like premium increases, customer service calls).

### 🔴 2. Combining SHAP with Cox PH
**The Interview Question:** "How did you use SHAP with a Cox model? What did you show the business?"

**Deep Dive Answer:**
While Cox models provide $\beta$ coefficients (Hazard Ratios), they are global and hard to interpret for a specific patient. I applied SHAP (specifically `shap.Explainer` for linear models) to the log-hazard outputs.
*   Instead of predicting a single probability, we used SHAP to show a customer service rep exactly *why* John Doe's hazard rate was spiking *today* (e.g., "SHAP value +0.4 due to a denied claim 3 days ago").
*   I built a **Streamlit** dashboard that allowed operations managers to input patient factors and instantly visualize how the Kaplan-Meier survival curve shifted in real-time, driving a 10% improvement in intervention outcomes.

---
*End of EXL Deep Dive.*
