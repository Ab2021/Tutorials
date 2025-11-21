# Day 26: Monitoring & Observability

> **Phase**: 3 - System Design
> **Week**: 6 - Data Engineering
> **Focus**: Keeping Models Healthy
> **Reading Time**: 45 mins

---

## 1. Why Models Fail in Production

Software bugs are static (NullPointer). ML bugs are dynamic (Data changes).

### 1.1 Data Drift (Covariate Shift)
*   **Definition**: The distribution of input data $P(X)$ changes, but the relationship $P(Y|X)$ stays the same.
*   **Example**: A model trained on summer photos receives winter photos.
*   **Detection**: Kullback-Leibler (KL) Divergence, Kolmogorov-Smirnov (KS) Test, Population Stability Index (PSI).

### 1.2 Concept Drift
*   **Definition**: The relationship $P(Y|X)$ changes. The "Concept" of what is "Spam" or "Fraud" evolves.
*   **Example**: Fraudsters change tactics. A previously "safe" pattern is now fraud.
*   **Detection**: Requires ground truth labels (which might be delayed). Monitor downstream business metrics (e.g., Chargeback rate).

---

## 2. Monitoring Stack

### 2.1 Infrastructure Monitoring
*   **Metrics**: Latency (p99), Throughput, GPU Utilization, Memory.
*   **Tools**: Prometheus, Grafana.

### 2.2 Model Monitoring
*   **Metrics**: Prediction distribution (Did mean prediction shift?), Feature distribution, Null rate.
*   **Tools**: WhyLabs, Arize AI, Evidently AI.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Delayed Labels
**Scenario**: In credit scoring, you only know if a user defaults after 12 months. You can't calculate Accuracy today.
**Solution**:
*   **Proxy Metrics**: Monitor "Early Payment Default" (missed first payment).
*   **Drift Monitoring**: If input distribution shifts, assume performance *might* degrade and trigger retraining.

### Challenge 2: Alert Fatigue
**Scenario**: You alert if KL Divergence > 0.1. It fires every night at 3 AM because traffic drops.
**Solution**:
*   **Seasonality**: Compare "This Monday" vs "Last Monday", not "Monday" vs "Sunday".
*   **Adaptive Thresholds**: Use anomaly detection (Z-score) on the metric itself.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What is the Population Stability Index (PSI)?**
> **Answer**: A metric to measure how much a variable's distribution has shifted between two samples (Train vs. Serving).
> *   PSI < 0.1: No significant change.
> *   PSI 0.1 - 0.25: Moderate change.
> *   PSI > 0.25: Significant drift. Retrain.

**Q2: How do you distinguish between Data Drift and a Bug?**
> **Answer**:
> *   **Bug**: Sudden, sharp drop in feature values (e.g., all ages become 0 or Null). Usually caused by upstream schema change.
> *   **Drift**: Gradual or structural shift (e.g., users getting younger over months).

**Q3: If you detect drift, what should you do?**
> **Answer**:
> 1.  **Verify**: Is it a bug?
> 2.  **Retrain**: If valid drift, retrain on the new data.
> 3.  **Fallback**: If critical, switch to a rules-based system or an older stable model until retraining completes.

---

## 5. Further Reading
- [A Comprehensive Guide to Data Drift](https://www.evidentlyai.com/blog/machine-learning-monitoring-data-drift)
- [Prometheus for ML Monitoring](https://prometheus.io/)
