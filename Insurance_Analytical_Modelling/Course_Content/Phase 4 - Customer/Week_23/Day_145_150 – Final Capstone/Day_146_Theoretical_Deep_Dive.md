# Final Capstone: Data Engineering & Pipelines (Part 2) - Feature Stores & Quality - Theoretical Deep Dive

## Overview
"Garbage In, Garbage Out." (The First Law of Data Science).
In the Capstone, you don't just "load a CSV". You build a **Pipeline**.
This day focuses on the "Plumbing" of ML: Ingestion, Validation, and Feature Serving.

---

## 1. Conceptual Foundation

### 1.1 The Modern Data Stack for Insurance

1.  **Ingestion:** Airflow (Orchestration).
2.  **Transformation:** dbt (Data Build Tool).
3.  **Validation:** Great Expectations (Data Quality).
4.  **Serving:** Feast (Feature Store).

### 1.2 The "Training-Serving Skew" Problem

*   **Scenario:**
    *   *Training:* You calculate `avg_claims_last_30_days` using SQL batch processing.
    *   *Serving:* You calculate `avg_claims_last_30_days` inside the Python API code.
*   **Risk:** The logic drifts. The SQL definition $\neq$ Python definition.
*   **Solution:** **Feature Store (Feast)**. Define the logic *once*. Serve it to both Training (Offline) and API (Online).

---

## 2. Mathematical Framework

### 2.1 Data Quality Metrics

*   **Completeness:** % of non-null values.
    $$ C = \frac{N_{\text{valid}}}{N_{\text{total}}} $$
*   **Uniqueness:** % of unique Primary Keys (Policy ID).
*   **Validity:** % of values within range (e.g., Age 18-100).
*   **Timeliness:** Lag between Event Time and Ingestion Time.

### 2.2 Feature Freshness

*   **Batch Features:** Updated Daily (e.g., Credit Score).
*   **Real-Time Features:** Updated Milliseconds (e.g., "User just clicked 'Quote'").
*   **Architecture:** Lambda Architecture (Batch Layer + Speed Layer).

---

## 3. Theoretical Properties

### 3.1 Idempotency

*   **Definition:** Running the same pipeline twice produces the same result.
*   **Why it matters:** If your pipeline fails halfway, you need to be able to "Retry" without creating duplicate records.
*   **Implementation:** `INSERT OVERWRITE` instead of `INSERT INTO`.

### 3.2 Data Lineage

*   **Question:** "Why is the Churn Model predicting 100% risk?"
*   **Answer:** Trace the lineage back.
    *   *Model* <- *Feature Table* <- *dbt Model* <- *Raw Table*.
    *   *Root Cause:* The Raw Table was empty today.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Great Expectations (Validation)

```python
import great_expectations as ge

# 1. Load Data
df = ge.read_csv("claims_data.csv")

# 2. Define Expectations
# Policy ID must be unique
df.expect_column_values_to_be_unique("policy_id")

# Claim Amount must be positive
df.expect_column_values_to_be_between("claim_amount", min_value=0, max_value=1000000)

# State must be valid US State
df.expect_column_values_to_be_in_set("state", ["NY", "CA", "TX", "FL"])

# 3. Validate
results = df.validate()
if not results["success"]:
    raise ValueError("Data Quality Check Failed!")
```

### 4.2 Feast Feature Store (Definition)

```python
# features.py
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# 1. Define Entity (Primary Key)
driver = Entity(name="driver", join_keys=["driver_id"])

# 2. Define Source (Parquet/SQL)
driver_stats_source = FileSource(
    path="driver_stats.parquet",
    timestamp_field="event_timestamp"
)

# 3. Define Feature View
driver_stats_view = FeatureView(
    name="driver_stats",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
    ],
    source=driver_stats_source,
)
```

---

## 5. Evaluation & Validation

### 5.1 The "Null" Test

*   **Scenario:** A new column `telematics_score` is added, but it's NULL for 50% of users.
*   **Impact:** XGBoost handles NULLs, but Neural Networks might crash (NaNs).
*   **Fix:** Imputation Strategy (Mean, Median, or -1 flag) defined in the Pipeline.

### 5.2 Drift Detection (Data Drift)

*   **Tool:** Evidently AI / Alibi Detect.
*   **Check:** Compare the distribution of `Age` in Training vs. Production.
*   **Alert:** If KL Divergence > 0.1, trigger a retraining pipeline.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Time Travel" Leakage**
    *   *Mistake:* Using `next_payment_date` to predict `churn`.
    *   *Why:* In the future, you know the payment date. In the past (at prediction time), you didn't.
    *   *Fix:* Point-in-Time Correctness (Feast handles this automatically).

2.  **Trap: Over-Engineering**
    *   *Mistake:* Setting up Kafka + Spark Streaming for a dataset with 100 rows.
    *   *Reality:* For Capstone, a Cron Job + Python Script is fine.

---

## 7. Advanced Topics & Extensions

### 7.1 dbt (Data Build Tool)

*   **Concept:** "Analytics Engineering".
*   **Workflow:** Write SQL `SELECT` statements. dbt compiles them into Tables/Views.
*   **Testing:** `dbt test` runs SQL assertions (Unique, Not Null) automatically.

### 7.2 Airflow DAGs

*   **Structure:** Directed Acyclic Graph.
*   **Tasks:**
    1.  `sensor_check`: Is the file in S3?
    2.  `ingest_data`: Copy to Postgres.
    3.  `run_dbt`: Transform.
    4.  `train_model`: Run Scikit-Learn.

---

## 8. Regulatory & Governance Considerations

### 8.1 Data Catalog

*   **Requirement:** You must know where PII lives.
*   **Tool:** Amundsen / DataHub.
*   **Tagging:** Tag columns as `PII`, `Sensitive`, `Public`.

---

## 9. Practical Example

### 9.1 Worked Example: Building the "Fraud Pipeline"

**Goal:** Ingest Claims CSV -> Clean -> Feature Store.
**Steps:**
1.  **Ingest:** Python script reads `daily_claims.csv` from S3.
2.  **Validate:** Great Expectations checks if `claim_amount` < 0. (Fails job if true).
3.  **Transform:** Pandas calculates `days_since_last_claim`.
4.  **Store:** Write to `offline_store.parquet` (for Training) and Redis (for Online Serving).
5.  **Outcome:** The model always has clean, fresh features.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Pipelines** > Notebooks.
2.  **Validation** prevents silent failures.
3.  **Feature Stores** solve skew.

### 10.2 When to Use This Knowledge
*   **Data Engineer Interview:** "How do you handle schema changes?"
*   **Capstone Defense:** "How do you ensure your data isn't garbage?"

### 10.3 Critical Success Factors
1.  **Reproducibility:** Can I delete the database and rebuild it from scratch with one command?
2.  **Observability:** Do I get a Slack alert when the pipeline fails?

### 10.4 Further Reading
*   **Chip Huyen:** "Designing Machine Learning Systems" (Chapter on Data Engineering).

---

## Appendix

### A. Glossary
*   **ETL:** Extract, Transform, Load.
*   **DAG:** Directed Acyclic Graph.
*   **TTL:** Time To Live (Feature expiry).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Completeness** | $N_{\text{valid}} / N_{\text{total}}$ | Quality Metric |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
