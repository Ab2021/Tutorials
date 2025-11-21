# Day 48 (Part 1): Advanced Mock - Ad Click Prediction

> **Phase**: 6 - Deep Dive
> **Topic**: Computational Advertising
> **Focus**: CTR, Calibration, and Online Learning
> **Reading Time**: 60 mins

---

## 1. Online Learning

Ads data is infinite and non-stationary.

### 1.1 FTRL (Follow The Regularized Leader)
*   Optimizer designed for sparse data + L1 regularization.
*   Standard SGD fails to produce true sparsity. FTRL does.
*   **Memory**: Stores gradients for billions of features.

---

## 2. Calibration

Advertisers pay for Clicks, but we predict Probability.

### 2.1 Isotonic Regression
*   Map predicted probability $P$ to calibrated probability $P'$.
*   Ensures monotonicity.
*   **Reliability Diagram**: Plot Predicted vs Observed. Should be $y=x$.

---

## 3. Tricky Interview Questions

### Q1: Delayed Feedback?
> **Answer**:
> *   Click happens 1s after view. Conversion happens 7 days later.
> *   **Negative Sampling**: If we train immediately, conversion is 0 (False Negative).
> *   **Fix**: Wait window (attribution window). Or use "Positive-Unlabeled" learning.

### Q2: Explaining "Why did I see this ad?"
> **Answer**:
> *   **SHAP values**: "Because you visited Site X and are Age Y."
> *   **Privacy**: Don't reveal sensitive targeting.

### Q3: Cold Start Ads?
> **Answer**:
> *   **Explore**: Boost bid for new ads artificially to get impressions.
> *   **Content Features**: Use Image/Text embeddings to predict CTR based on similar ads.

---

## 4. Practical Edge Case: Data Leakage
*   **Trap**: Using "Total Clicks on Ad" as a feature.
*   **Why**: Includes the *current* click.
*   **Fix**: Use "Total Clicks *until yesterday*".

