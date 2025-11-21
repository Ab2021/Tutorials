# Day 26 (Part 1): Advanced Monitoring & Drift

> **Phase**: 6 - Deep Dive
> **Topic**: Statistical Process Control
> **Focus**: KS-Test, MMD, and Embedding Drift
> **Reading Time**: 60 mins

---

## 1. Drift Algorithms

### 1.1 Kolmogorov-Smirnov (KS) Test
*   **Type**: Numerical Data.
*   **Idea**: Compare Cumulative Distribution Functions (CDF). Max distance between two CDFs.
*   **Pros**: Non-parametric (doesn't assume Gaussian).
*   **Cons**: 1D only.

### 1.2 Chi-Square Test
*   **Type**: Categorical Data.
*   **Idea**: Compare observed frequency vs expected frequency.

### 1.3 Maximum Mean Discrepancy (MMD)
*   **Type**: High-dimensional / Embeddings.
*   **Idea**: Kernel-based method to compare two distributions.
*   **Use Case**: Detecting drift in Image/Text embeddings.

---

## 2. Feedback Loops

### 2.1 Degenerate Feedback
*   **Scenario**: RecSys recommends "Clickbait". User clicks. Model learns "Clickbait is good". Recommends more.
*   **Result**: Drift in target variable distribution.
*   **Fix**: Exploration (Epsilon-Greedy) to gather data on non-clickbait items.

---

## 3. Tricky Interview Questions

### Q1: Univariate vs Multivariate Drift?
> **Answer**:
> *   **Univariate**: Feature A shifted. Feature B shifted. Easy to detect.
> *   **Multivariate**: A and B have same marginals, but correlation changed.
> *   **Detection**: Train a classifier to distinguish Train vs Test data. If Accuracy > 0.5, there is multivariate drift.

### Q2: How to monitor Unsupervised Models (Clustering)?
> **Answer**:
> *   Monitor **Cluster Stability**.
> *   Monitor **Distance to Centroids**. If points are moving further from centroids, the clusters are becoming invalid.

### Q3: What is "Prediction Drift"?
> **Answer**: The distribution of $Y_{pred}$ changes.
> *   Easier to monitor than Feature Drift (1 column vs 100).
> *   Good proxy for Concept Drift.

---

## 4. Practical Edge Case: Seasonality
*   **Problem**: Traffic on Sunday looks different from Monday. False Alarm.
*   **Fix**: Compare "This Sunday" vs "Last Sunday". Or use Time-Series Anomaly Detection (Prophet/Holt-Winters) on the metric.

