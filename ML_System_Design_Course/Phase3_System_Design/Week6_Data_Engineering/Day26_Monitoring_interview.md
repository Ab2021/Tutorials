# Day 26: Monitoring & Drift - Interview Questions

> **Topic**: Reliability
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Why do ML models degrade over time?
**Answer:**
*   **Data Drift**: Input distribution changes ($P(X)$).
*   **Concept Drift**: Relationship between input and target changes ($P(Y|X)$).
*   **Upstream Changes**: Broken pipeline, schema change.

### 2. What is Data Drift (Covariate Shift)?
**Answer:**
*   Change in distribution of input features.
*   Example: Training on summer images, predicting on winter images.

### 3. What is Concept Drift?
**Answer:**
*   Change in the mapping from X to Y.
*   Example: "Spam" definition changes over time. Same email text, different label.

### 4. How do you detect Drift?
**Answer:**
*   **Statistical Tests**: KS Test (Continuous), Chi-Square (Categorical).
*   **Distance Metrics**: PSI (Population Stability Index), KL Divergence, Wasserstein Distance.

### 5. What is Population Stability Index (PSI)?
**Answer:**
*   Measure of how much a distribution has shifted.
*   PSI < 0.1: No change. PSI > 0.2: Significant drift.
*   Sum of $(Actual\% - Expected\%) \times \ln(Actual\% / Expected\%)$.

### 6. What is Prediction Drift?
**Answer:**
*   Monitoring the distribution of the model's *outputs*.
*   If model usually predicts 5% fraud, and suddenly predicts 20%, something is wrong.

### 7. How do you handle Delayed Labels (Ground Truth)?
**Answer:**
*   In credit scoring, default happens months later.
*   **Proxy Metrics**: Monitor prediction distribution (drift) immediately.
*   **Short-term outcomes**: Monitor "First payment default" as proxy for "Loan default".

### 8. What is "Training-Serving Skew"?
**Answer:**
*   Performance difference between training and serving.
*   **Schema Skew**: Int vs Float.
*   **Feature Logic Skew**: Different libraries.
*   **Distribution Skew**: Sampling bias.

### 9. How do you monitor Feature Quality?
**Answer:**
*   **Null Rate**: Sudden spike in NaNs.
*   **Range**: Values outside [0, 1].
*   **Cardinality**: New categories appearing.

### 10. What is Z-score based anomaly detection?
**Answer:**
*   Calculate Mean and Std of a metric (e.g., QPS) over history.
*   Alert if current value > Mean + 3*Std.

### 11. Explain "Feedback Loop" in monitoring.
**Answer:**
*   Model decisions affect future data.
*   RecSys: User only clicks what is shown. Data becomes biased towards current model.
*   Monitor **Exploration** metrics.

### 12. What tools are used for ML Monitoring?
**Answer:**
*   **Evidently AI**, **Arize**, **Fiddler**, **Prometheus/Grafana**.

### 13. What do you do when Drift is detected?
**Answer:**
*   **Analyze**: Is it real drift or data quality issue?
*   **Retrain**: If real, retrain on recent data.
*   **Fallback**: Switch to rule-based system or older stable model.

### 14. What is "Label Shift"?
**Answer:**
*   Change in distribution of $Y$ (Prior probability).
*   Example: During COVID, fraud rates spiked globally.

### 15. How do you monitor Embedding Drift?
**Answer:**
*   Embeddings are high-dimensional. Hard to histogram.
*   **Reduce Dimensions**: PCA/UMAP to 2D, then monitor.
*   **Cosine Similarity**: Measure average distance between recent embeddings and reference set.

### 16. What is "Performance Monitoring"?
**Answer:**
*   Tracking Accuracy, F1, AUC, MAE in production (if labels available).

### 17. What is "System Monitoring" vs "ML Monitoring"?
**Answer:**
*   **System**: CPU, RAM, Latency, 500 Errors. (DevOps).
*   **ML**: Drift, Accuracy, Bias. (MLOps).

### 18. How do you define a Reference Window for drift?
**Answer:**
*   Usually the **Training Set** or a **Validation Set**.
*   Or "Last Week" vs "This Week".

### 19. What is "Bias Drift"?
**Answer:**
*   Model starts discriminating against a protected group (e.g., Age/Gender) due to shifting demographics.
*   Monitor Fairness metrics (Disparate Impact).

### 20. How often should you retrain?
**Answer:**
*   **Periodic**: Nightly/Weekly.
*   **Trigger-based**: When Drift > Threshold or Performance < Threshold.
