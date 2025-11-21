# Day 44: System Design Template - Interview Questions

> **Topic**: Structuring the Interview
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is the standard structure of an ML System Design interview?
**Answer:**
1.  **Requirements** (5 min).
2.  **Data Engineering** (5 min).
3.  **Feature Engineering** (10 min).
4.  **Model Modeling** (10 min).
5.  **Evaluation** (5 min).
6.  **Serving & Scaling** (5 min).
7.  **Monitoring** (5 min).

### 2. What questions should you ask in the "Requirements" phase?
**Answer:**
*   **Goal**: What are we optimizing? (CTR, Revenue, User Happiness).
*   **Constraints**: Latency? Budget? Storage?
*   **Scale**: DAU? QPS?
*   **Platform**: Mobile vs Web?

### 3. How do you handle "Estimation" (Back-of-the-envelope)?
**Answer:**
*   Round numbers.
*   $10^5$ users $\times 100$ actions = $10^7$ events/day.
*   Storage: $10^7 \times 1KB = 10GB/day$.

### 4. What is the "Baseline Model"?
**Answer:**
*   Simple heuristic or linear model.
*   "Recommend most popular items".
*   "Logistic Regression".
*   Sets the floor for performance.

### 5. How do you transition from Training to Serving?
**Answer:**
*   Discuss **Feature Store**.
*   Discuss **Model Registry**.
*   Discuss **Serialization** (ONNX).

### 6. What are the common "Bottlenecks" to look for?
**Answer:**
*   **Data**: Reading from S3 during training.
*   **Compute**: Matrix multiplication.
*   **Network**: Sending large embeddings.
*   **Database**: High write QPS.

### 7. How do you design for "Scalability"?
**Answer:**
*   **Stateless** services (Horizontal scaling).
*   **Caching** (Redis).
*   **Async processing** (Kafka).
*   **Sharding** DBs.

### 8. What is the "Feedback Loop"?
**Answer:**
*   How does user action get back to the model?
*   Logging -> Kafka -> Data Lake -> Training Data.

### 9. How do you handle "Personalization"?
**Answer:**
*   User Embeddings.
*   Real-time session features.

### 10. What is "Online Learning" vs "Batch Retraining"?
**Answer:**
*   **Batch**: Retrain nightly. Safe.
*   **Online**: Update weights on every click. Risky but fast adaptation.

### 11. How do you handle "Bias"?
**Answer:**
*   **Data Bias**: Sampling.
*   **Algorithmic Bias**: Loss function.
*   **Metric**: Fairness metrics.

### 12. What is "Privacy by Design"?
**Answer:**
*   Differential Privacy.
*   Federated Learning.
*   Data minimization.

### 13. How do you handle "Seasonality"?
**Answer:**
*   Time features (Day of week, Month).
*   Retrain frequently.

### 14. What is the "Cold Start" strategy?
**Answer:**
*   Heuristics (Popularity).
*   Content-based filtering.
*   Bandits.

### 15. How do you choose the Loss Function?
**Answer:**
*   **Regression**: MSE, MAE, Huber.
*   **Classification**: Cross-Entropy, Focal Loss (Imbalanced).
*   **Ranking**: Pairwise (BPR), Listwise (NDCG).

### 16. What is "Hyperparameter Tuning" in production?
**Answer:**
*   AutoML.
*   Population Based Training (PBT).

### 17. How do you handle "Debuggability"?
**Answer:**
*   Log inputs/outputs.
*   Model explainability (SHAP).
*   Trace IDs.

### 18. What is "Cost Optimization"?
**Answer:**
*   Spot instances.
*   Quantization.
*   Caching.

### 19. How do you handle "Multi-Modal" inputs?
**Answer:**
*   Separate encoders (CNN for image, BERT for text).
*   Concatenate embeddings.
*   Fusion layer.

### 20. What is the "North Star Metric"?
**Answer:**
*   The single metric that best captures value.
*   e.g., "Time Spent" (YouTube), "Sessions with Booking" (Airbnb).
