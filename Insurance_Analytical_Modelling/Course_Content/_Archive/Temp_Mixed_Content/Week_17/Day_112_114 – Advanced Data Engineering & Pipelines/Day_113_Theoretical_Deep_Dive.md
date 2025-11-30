# Advanced Data Engineering & Pipelines (Part 2) - Feature Stores & MLOps - Theoretical Deep Dive

## Overview
"The model works in the notebook but fails in production."
Why? **Training-Serving Skew**.
In the notebook, you calculated "Average Claims" using SQL. In production, the app calculates it using Java. They don't match.
**Feature Stores (Feast)** solve this by defining the feature *once* and serving it *everywhere*.

---

## 1. Conceptual Foundation

### 1.1 The Two Worlds of ML

1.  **Training (Offline):** High throughput, high latency. (Process 10 years of data).
2.  **Serving (Online):** Low throughput, low latency. (Score 1 customer in 10ms).

### 1.2 The Feature Store Architecture

*   **Offline Store (e.g., Snowflake/S3):** Stores years of history. Used for Training.
*   **Online Store (e.g., Redis/DynamoDB):** Stores only the *latest* value. Used for Serving.
*   **Registry:** The "Source of Truth" for feature definitions.

---

## 2. Mathematical Framework

### 2.1 Point-in-Time Correctness (Time Travel)

*   **Scenario:**
    *   Jan 1: Customer Credit Score = 700.
    *   Jan 15: Customer has an accident.
    *   Feb 1: Customer Credit Score = 600.
*   **The Mistake:** If you train a model today to predict the Jan 15 accident, and you simply join "Current Credit Score", you use 600. This is **Data Leakage** (Future information).
*   **The Fix:** An "AS OF" join.
    *   `SELECT * FROM claims c LEFT JOIN credit s ON c.user_id = s.user_id AND s.timestamp <= c.timestamp`

### 2.2 Consistency

$$ F_{offline}(x) \equiv F_{online}(x) $$

*   The logic to compute the feature must be identical.
*   Feast ensures this by decoupling the *definition* from the *storage*.

---

## 3. Theoretical Properties

### 3.1 Entity-Centric Modeling

*   Features are attached to **Entities** (Driver, Vehicle, Policy).
*   *Example:* `driver_stats:avg_speed` is attached to `driver_id`.

### 3.2 Materialization

*   The process of moving data from Offline to Online.
*   *Frequency:* Real-time (Kafka) or Batch (Nightly).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Feast Definition (Python)

```python
from feast import Entity, Feature, FeatureView, ValueType
from feast.infra.offline_stores.file_source import FileSource

# 1. Define Entity
driver = Entity(name="driver_id", value_type=ValueType.INT64)

# 2. Define Source
driver_stats_source = FileSource(
    path="data/driver_stats.parquet",
    event_timestamp_column="event_timestamp"
)

# 3. Define Feature View
driver_stats_view = FeatureView(
    name="driver_stats",
    entities=["driver_id"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="conv_rate", dtype=ValueType.FLOAT),
        Feature(name="acc_rate", dtype=ValueType.FLOAT),
    ],
    batch_source=driver_stats_source
)
```

### 4.2 Retrieval

*   **Training (Historical):**
    ```python
    training_df = store.get_historical_features(
        entity_df=claims_df,
        features=["driver_stats:conv_rate"]
    ).to_df()
    ```
*   **Serving (Online):**
    ```python
    features = store.get_online_features(
        features=["driver_stats:conv_rate"],
        entity_rows=[{"driver_id": 123}]
    ).to_dict()
    ```

---

## 5. Evaluation & Validation

### 5.1 Drift Monitoring

*   Compare the distribution of `F_{offline}` (Training) vs `F_{online}` (Production).
*   **PSI (Population Stability Index):** If PSI > 0.2, the feature has drifted.

### 5.2 Latency Testing

*   **Requirement:** `get_online_features` must return in < 10ms.
*   **Optimization:** Use Redis Cluster or DynamoDB DAX.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "TTL" Trap**
    *   If you set TTL=24 hours, and the driver hasn't driven in 2 days, Feast returns `NULL`.
    *   *Fix:* Set TTL appropriate to the business logic (e.g., "Last known speed" might be valid for 1 month).

2.  **Trap: On-Demand Transformations**
    *   Some features need input from the *request* (e.g., `haversine_distance(user_loc, accident_loc)`).
    *   Feast supports "On-Demand Feature Views" for this.

---

## 7. Advanced Topics & Extensions

### 7.1 Streaming Features

*   **Scenario:** "Number of clicks in the last 5 minutes".
*   **Architecture:** Kafka -> Flink (Aggregation) -> Feast Online Store (Push).
*   This bypasses the Offline Store for latency reasons (Lambda Architecture).

### 7.2 Feature Sharing

*   **Problem:** Claims Team builds "Driver Risk". Marketing Team builds "Driver Risk". They are different.
*   **Solution:** Centralized Feature Registry. "Build once, reuse everywhere."

---

## 8. Regulatory & Governance Considerations

### 8.1 Feature Lineage

*   **Question:** "Why was this claim denied?"
*   **Answer:** "Because Model v2 used Feature `risk_score` v3, which was calculated using data from Source Z."
*   Feast provides metadata to trace this lineage.

---

## 9. Practical Example

### 9.1 Worked Example: Real-Time Pricing

**Scenario:**
*   **Goal:** Quote a price for a new customer on the website.
*   **Inputs:**
    *   User Input: Age, Car Model.
    *   **Feature Store:** "Credit Score" (Nightly Batch), "Accident History" (Monthly Batch), "Current Location Risk" (Real-time).
*   **Process:**
    1.  User enters ID.
    2.  Backend calls `store.get_online_features()`.
    3.  Feast merges Batch + Real-time features.
    4.  Model predicts Price.
    5.  Latency: 50ms.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Feast** bridges the gap between Data Engineering and Data Science.
2.  **Point-in-Time Correctness** prevents leakage.
3.  **Online Store** enables real-time AI.

### 10.2 When to Use This Knowledge
*   **Productionizing Models:** Moving from POC to Prod.
*   **Real-Time Systems:** Fraud, Pricing, Recommendations.

### 10.3 Critical Success Factors
1.  **Latency:** If Redis is slow, the whole app is slow.
2.  **Data Freshness:** If the "Online" store is 2 days old, it's useless.

### 10.4 Further Reading
*   **Uber Engineering:** "Michelangelo: Machine Learning Platform".

---

## Appendix

### A. Glossary
*   **Entity:** The primary key (User, Car).
*   **TTL:** Time To Live (How long a feature value is valid).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Point-in-Time** | $t_{feature} \le t_{event}$ | Join Logic |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
