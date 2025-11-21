# Day 21: System Design Framework - Interview Questions

> **Topic**: Designing ML Systems
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What are the key components of an ML System Design interview?
**Answer:**
1.  **Requirements**: Functional (What to do) & Non-functional (Latency, Scale).
2.  **Data**: Sources, Labeling, Features.
3.  **Model**: Baseline, Architecture, Loss.
4.  **Training**: Pipeline, Retraining strategy.
5.  **Serving**: Batch vs Online, Caching.
6.  **Monitoring**: Metrics, Drift.

### 2. How do you estimate the capacity (QPS/Storage) for a system?
**Answer:**
*   **QPS**: DAU $\times$ Actions/Day / 86400. Peak = 2-5x Average.
*   **Storage**: Users $\times$ Data/User $\times$ Retention.
*   **Bandwidth**: QPS $\times$ Request Size.

### 3. Explain the difference between Batch Prediction and Online Prediction.
**Answer:**
*   **Batch**: Run model on all users at night. Save results to DB. Fast read. Stale data.
*   **Online**: Run model on-demand when request comes. Fresh data. Latency constraints.

### 4. What is the CAP Theorem? How does it apply to ML?
**Answer:**
*   Consistency, Availability, Partition Tolerance. Pick 2.
*   **ML**: Usually prefer **Availability** (AP). Better to show a slightly stale recommendation than an error page.

### 5. How do you handle the "Cold Start" problem?
**Answer:**
*   **New User**: Use popular items, location-based, or demographic heuristics.
*   **New Item**: Use content-based features (embedding of description/image) instead of interaction history.

### 6. What is a Feature Store? Why is it needed?
**Answer:**
*   Centralized repository for features.
*   Ensures consistency between **Training** (Offline) and **Serving** (Online).
*   Prevents "Training-Serving Skew".

### 7. Explain "Training-Serving Skew".
**Answer:**
*   Model performance drops in production because input data distribution differs from training.
*   **Causes**: Bug in feature code, different data sources, time lag.

### 8. How do you design a system for High Availability?
**Answer:**
*   **Redundancy**: Replicas of services.
*   **Load Balancing**: Distribute traffic.
*   **Failover**: Auto-switch to backup.
*   **Circuit Breakers**: Stop calling failing service.

### 9. What is Latency Budgeting?
**Answer:**
*   Total time allowed for request (e.g., 200ms).
*   Allocate budget to components: DB (10ms), Feature Store (20ms), Model Inference (100ms), Network (50ms).

### 10. How do you handle "Thundering Herd" problem?
**Answer:**
*   Many clients retry simultaneously after a failure, overwhelming the system.
*   **Fix**: Exponential Backoff with **Jitter** (Randomness).

### 11. What is the difference between Scale Up (Vertical) and Scale Out (Horizontal)?
**Answer:**
*   **Up**: Bigger machine (More RAM/CPU). Limit: Hardware cost/max.
*   **Out**: More machines. Limit: Complexity of distributed system. Preferred for cloud.

### 12. How do you choose between SQL and NoSQL for an ML system?
**Answer:**
*   **SQL (Postgres)**: Structured data, ACID transactions, Complex joins (Metadata).
*   **NoSQL (Cassandra/Dynamo)**: High write throughput, Unstructured, Horizontal scaling (User Logs, Features).

### 13. What is Consistent Hashing?
**Answer:**
*   Technique to distribute keys across N servers.
*   Minimizes data movement when a server is added/removed ($1/N$ keys move).
*   Used in Caching/Sharding.

### 14. How do you design for Data Privacy (GDPR)?
**Answer:**
*   **Encryption**: At rest and in transit.
*   **Anonymization**: Remove PII.
*   **Right to be Forgotten**: Ability to delete user data (and potentially retrain model).

### 15. What is a "Shadow Mode" deployment?
**Answer:**
*   Deploy new model alongside old one.
*   Route traffic to both. Return old prediction to user. Log new prediction.
*   Compare performance safely without affecting user.

### 16. What is A/B Testing in System Design?
**Answer:**
*   Split traffic into Control (A) and Treatment (B).
*   Compare business metrics (CTR, Revenue).
*   Must ensure random assignment and statistical significance.

### 17. How do you handle Feedback Loops in ML systems?
**Answer:**
*   Model predicts X -> User clicks X -> Model learns X is good -> Predicts X more.
*   **Fix**: Exploration (Randomness/Bandits), Positional bias correction.

### 18. What is "Online Learning"?
**Answer:**
*   Model updates weights continuously as data arrives.
*   **Pros**: Adapts fast.
*   **Cons**: Unstable. Can be poisoned by bad data. Hard to debug.

### 19. How do you monitor an ML system in production?
**Answer:**
*   **System Metrics**: Latency, CPU, Memory, Error Rate.
*   **Model Metrics**: Prediction distribution, Null rate.
*   **Business Metrics**: CTR, Conversion.

### 20. What is "Gradual Rollout" (Canary Deployment)?
**Answer:**
*   Deploy to 1% of users. Monitor.
*   Increase to 10%, 50%, 100%.
*   Rollback if errors spike.
