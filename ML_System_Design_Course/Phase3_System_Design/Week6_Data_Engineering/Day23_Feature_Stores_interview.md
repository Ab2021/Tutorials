# Day 23: Feature Stores - Interview Questions

> **Topic**: MLOps Infrastructure
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is a Feature Store? (Feast/Tecton).
**Answer:**
*   System to manage, store, and serve features for ML.
*   Solves: Reuse, Consistency, Point-in-time correctness.

### 2. Explain the difference between Offline Store and Online Store.
**Answer:**
*   **Offline**: Cheap, high capacity (S3/BigQuery). Used for Training (Batch).
*   **Online**: Low latency, high availability (Redis/DynamoDB). Used for Serving (Real-time).

### 3. What is Point-in-Time (Time Travel) Join?
**Answer:**
*   Joining training labels with features *as they looked at that specific time*.
*   Prevents **Data Leakage** (using future info).
*   `ASOF JOIN`.

### 4. What is Feature Engineering?
**Answer:**
*   Transforming raw data into formats suitable for ML.
*   One-hot, Scaling, Aggregations, Embeddings.

### 5. Why is "Feature Reuse" important?
**Answer:**
*   Avoids duplicate work. Team A builds "User LTV". Team B can just use it.
*   Standardizes definitions.

### 6. How do you handle Feature Versioning?
**Answer:**
*   Features change (logic changes).
*   Feature Store tracks versions (`v1`, `v2`).
*   Models pin specific versions.

### 7. What is an Entity in a Feature Store?
**Answer:**
*   The primary key (e.g., `user_id`, `driver_id`).
*   Features are associated with entities.

### 8. Explain the concept of "Materialization".
**Answer:**
*   Computing feature values and storing them in the Online Store.
*   Can be scheduled (Batch) or Triggered (Stream).

### 9. How do you handle Streaming Features?
**Answer:**
*   Compute aggregates (e.g., "clicks last 5 mins") on stream (Flink).
*   Push to Online Store immediately.

### 10. What is Feature Drift?
**Answer:**
*   Statistical properties of a feature change over time.
*   Feature Store can monitor this (compare Training vs Serving stats).

### 11. What is the difference between a Feature and a Label?
**Answer:**
*   **Feature**: Input ($X$).
*   **Label**: Target ($Y$).
*   Feature Store manages $X$.

### 12. How do you handle High Cardinality Categorical Features?
**Answer:**
*   **Hashing**: Hash to fixed buckets. Collision risk.
*   **Embeddings**: Learn dense vector.
*   **Target Encoding**: Replace with mean target value.

### 13. What is "Feature Backfill"?
**Answer:**
*   Computing a new feature for all historical data.
*   Expensive but needed to train on history.

### 14. How does a Feature Store help with "Training-Serving Skew"?
**Answer:**
*   Guarantees that the code used to compute features offline is the *exact same* code used online (or results are synced).

### 15. What is a "Feature View"?
**Answer:**
*   Logical group of features (e.g., "User Demographics").
*   Defines source, transformation, and entities.

### 16. How do you test feature logic?
**Answer:**
*   Unit tests on transformation functions.
*   Data expectations on output.

### 17. What is the cost of a Feature Store?
**Answer:**
*   **Infrastructure**: Redis is expensive.
*   **Compute**: Materialization jobs.
*   **Complexity**: Another system to maintain.

### 18. When should you NOT use a Feature Store?
**Answer:**
*   Small team, single model.
*   No online serving (Batch only).
*   Static data.

### 19. How do you handle PII in a Feature Store?
**Answer:**
*   Access Control (RBAC).
*   Encryption.
*   Masking/Tokenization.

### 20. What is the relationship between Data Warehouse and Feature Store?
**Answer:**
*   DW is the *source* for the Offline Store.
*   Feature Store adds the "Serving" layer and "Point-in-time" logic on top of DW.
