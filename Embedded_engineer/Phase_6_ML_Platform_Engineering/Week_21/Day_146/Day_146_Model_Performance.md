# Day 146: Is the Model Crazy? Performance Monitoring
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 21: Observability for ML Systems

---

> **🎯 Focus Area:** Your system latency is fine (20ms). But the Model is predicting "Dog" for every single image. **Model Performance Monitoring** tracks the statistical health of predictions, not just the speed of the server.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Monitor** the Distribution of Predictions (Did we stop predicting "Fraud"?).
2.  **Track** Confidence Scores (Is the model becoming uncertain?).
3.  **Detect** "Silent Failures" (e.g., Input features are all zeros).
4.  **Visualize** Prediction Drift on a Grafana dashboard.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install scikit-learn numpy prometheus-client`.

---

## 📖 Theoretical Foundation

### 1. Functional Monitoring vs Model Monitoring
*   **Functional:** "Is the server returning 200 OK?" (Day 141).
*   **Model:** "Is the server returning *useful* answers?"
    *   **Prediction Drift:** The output `y_pred` distribution shifted.
    *   **Feature Drift:** The input `X` distribution shifted.
    *   **Missing Features:** Are we receiving `NaN`s at inference time?

### 2. The Feedback Loop Problem
In many ML systems (Fraud, Ads), we don't know the **Ground Truth** (Label) for days or weeks. We cannot calculate "Accuracy" in real-time.
**Proxy Metrics:**
*   Prediction Count per Class.
*   Average Confidence Score.
*   Missing Value Rate.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Middleware for Statistical Tracking

We extend our FastAPI service to log histograms of *outputs*.

#### 📁 `src/monitor_middleware.py`
```python
from prometheus_client import Histogram, Counter
import numpy as np

# A. Prediction Distribution
# Buckets for regression (0.0 to 1.0) or classifications
PREDICTION_VALUE = Histogram(
    "model_prediction_value", 
    "Output value of the model", 
    ["model_version"],
    buckets=np.linspace(0, 1, 11) # [0.0, 0.1, ... 1.0]
)

# B. Confidence
CONFIDENCE_SCORE = Histogram(
    "model_confidence_score",
    "Softmax probability of top class",
    ["model_version"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

# C. Input Health
MISSING_FEATURES = Counter(
    "model_missing_features_total",
    "Count of Null inputs",
    ["feature_name"]
)

def monitor_prediction(model_version, inputs, outputs, probabilities=None):
    """
    Called after every inference batch.
    """
    # 1. Track Output Distribution
    # If outputs is binary 0/1, this tracks class imbalance
    for pred in outputs:
        PREDICTION_VALUE.labels(model_version=model_version).observe(pred)
        
    # 2. Track Confidence
    if probabilities is not None:
        # probabilities: shape (B, N_Classes)
        max_probs = np.max(probabilities, axis=1)
        for conf in max_probs:
            CONFIDENCE_SCORE.labels(model_version=model_version).observe(conf)
            
    # 3. Track Input Quality
    # inputs: Pandas DataFrame
    null_counts = inputs.isnull().sum()
    for feature, count in null_counts.items():
        if count > 0:
            MISSING_FEATURES.labels(feature_name=feature).inc(count)
```

### 👨‍💻 Infrastructure: Grafana Visualization

**Panel: Prediction Drift**
*   **Visual:** Heatmap.
*   **Query:** `sum(rate(model_prediction_value_bucket[1h])) by (le)`.
*   **Insight:** If the heatmap changes color (e.g., shifts from 0.1 to 0.9) suddenly, the model behavior has changed.

**Panel: Uncertainty Spike**
*   **Visual:** Time Series.
*   **Query:** `avg_over_time(model_confidence_score_sum[5m]) / avg_over_time(model_confidence_score_count[5m])`.
*   **Insight:** If average confidence drops from 0.9 to 0.6, the model is seeing "Out of Distribution" data.

---

## 🔬 Lab Exercise: "The Null Attack"

### Task
Simulate Broken Upstream Data.
1.  **Baseline:** Run predictions with normal data. Confidence ~0.9.
2.  **Attack:** Send requests where `age` feature is `NaN` (or -1).
3.  **Observation:**
    *   `model_missing_features_total` spikes for `age`.
    *   `model_confidence_score` likely drops (Model is confused).
4.  **Alert:** Set an alert on `rate(model_missing_features_total[5m]) > 10`.

---

## 📖 Advanced Theory: Evidently AI / Arize
For simple stats, Prometheus is fine.
For complex Drift Detection (KS-Test, PSI, KL-Divergence), you need specialized tools like **Evidently AI** or **Arize**.
*   **Prometheus:** Tracks scalars.
*   **Evidently:** Tracks distributions and correlations. Requires logging payloads (Sampling) to a separate store (Postgres/S3) and running batch analysis.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Silent Failure:** ML models fail silently. They don't throw Exceptions. They just produce garbage. Monitoring statistical properties is the only defense.
2.  **Granularity:** Do not log every prediction to Prometheus (Cardinality Explosion). Use Histograms.
3.  **Feedback:** If you CAN get ground truth (e.g., user clicked the recommendation), log it immediately as `model_actual_click`. Calculate accuracy in real-time.

### API Summary
```python
Histogram(...).observe(val)
```

---

**Day 146 Complete** ✅

*Next: Day 147 - Week 21 Review & Project - The Comprehensive Dashboard.*
