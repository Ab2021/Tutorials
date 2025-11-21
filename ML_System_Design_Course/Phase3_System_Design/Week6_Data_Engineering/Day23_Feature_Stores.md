# Day 23: Feature Stores

> **Phase**: 3 - System Design
> **Week**: 6 - Data Engineering
> **Focus**: Solving Training-Serving Skew
> **Reading Time**: 50 mins

---

## 1. The Problem: Training-Serving Skew

In a naive ML setup:
1.  **Training**: Data Scientist writes a Python script to extract features from a CSV dump.
2.  **Serving**: Backend Engineer re-implements the logic in Java/Go to fetch data from SQL and compute features.

**Result**: Logic mismatch. The definition of "Average Order Value" might differ slightly. The model fails in production.

---

## 2. The Solution: Feature Store

A centralized repository for feature logic and data.
*   **Single Source of Truth**: Define feature logic once.
*   **Offline Store (Batch)**: Cheap storage (S3/BigQuery) for training history.
*   **Online Store (Low Latency)**: Fast storage (Redis/DynamoDB) for real-time inference.

### 2.1 Key Capabilities
1.  **Point-in-Time Correctness (Time Travel)**: When generating training data, the feature store ensures that for a label at time $T$, we only see feature values known *before* $T$. This prevents **Data Leakage**.
2.  **Materialization**: Automatically computing features and pushing them to the Online Store.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Feature Freshness
**Scenario**: A user clicks an item. The "Last 10 Clicks" feature must update immediately.
**Solution**:
*   **Stream-to-Feature-Store**: Use Flink/Kafka to compute features in real-time and write to Redis (Online Store).

### Challenge 2: Entity Resolution
**Scenario**: Features are keyed by `user_id`. But at inference time, you only have `device_id`.
**Solution**:
*   **ID Graph**: A mapping service to link `device_id` -> `user_id` before querying the feature store.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What is "Point-in-Time Correctness" (Time Travel)?**
> **Answer**: It ensures that when we create a training example for an event at time $t$, the features associated with that event reflect the state of the world at time $t$, not $t+1$.
> *   Example: Predicting "Will Churn" on Jan 1st. We must use the "Number of Support Tickets" count as of Jan 1st. If we use the count from today (Feb 1st), we leak future information.

**Q2: Why do we need separate Online and Offline stores?**
> **Answer**:
> *   **Offline (S3/Parquet)**: Optimized for **Throughput** (scanning TBs of data for training). Cheap. High latency.
> *   **Online (Redis/Cassandra)**: Optimized for **Latency** (ms lookups for single entities). Expensive. Low throughput for scans.
> *   The Feature Store syncs them.

**Q3: How does a Feature Store help with collaboration?**
> **Answer**: It prevents "Feature Duplication". Team A creates "User Age". Team B needs "User Age". Instead of re-implementing it, Team B looks up the definition in the Feature Store registry and reuses it.

---

## 5. Further Reading
- [Feast: Open Source Feature Store](https://feast.dev/)
- [Tecton: Enterprise Feature Store Architecture](https://www.tecton.ai/blog/what-is-a-feature-store/)
