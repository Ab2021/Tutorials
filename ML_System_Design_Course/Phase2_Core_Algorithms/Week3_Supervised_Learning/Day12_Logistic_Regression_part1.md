# Day 12 (Part 1): Advanced Logistic Regression

> **Phase**: 6 - Deep Dive
> **Topic**: Classification Nuances
> **Focus**: Calibration, Multiclass, and Solvers
> **Reading Time**: 60 mins

---

## 1. Multiclass Strategies

### 1.1 One-vs-Rest (OvR)
*   Train $K$ binary classifiers.
*   **Pros**: Simple, parallelizable.
*   **Cons**: Imbalanced datasets (1 vs K-1). Scale of confidence scores might differ.

### 1.2 Softmax Regression (Multinomial)
*   Train 1 model with $K$ outputs. Normalize with Softmax.
*   **Pros**: Calibrated probabilities across classes. Learns correlations between classes.
*   **Cons**: More complex optimization.

---

## 2. Perfect Separation

### 2.1 The Problem
*   If data is linearly separable, MLE pushes weights to infinity ($\infty$) to drive sigmoid to exactly 0 or 1.
*   **Result**: Overfitting. Numerical instability.
*   **Fix**: Regularization (L2) bounds the weights.

---

## 3. Tricky Interview Questions

### Q1: Is Logistic Regression a Linear Classifier?
> **Answer**: Yes. The decision boundary is $w^T x + b = 0$, which is a hyperplane (linear). The *probability* curve is non-linear (sigmoid), but the *boundary* is linear.

### Q2: Why use Log-Loss instead of MSE for classification?
> **Answer**:
> 1.  **Convexity**: Log-Loss is convex for Logistic Regression. MSE is non-convex (wavy) with sigmoid, leading to local minima.
> 2.  **Gradient**: Log-Loss gradient is $(y - p)x$. Strong gradient even when wrong. MSE gradient includes $p(1-p)$ derivative of sigmoid, which vanishes when saturated (Vanishing Gradient).

### Q3: How do you interpret the weights?
> **Answer**:
> *   $\beta_j$ is the change in **Log-Odds** for a unit change in $x_j$.
> *   $e^{\beta_j}$ is the multiplicative change in **Odds**.
> *   It is *not* linear probability change.

---

## 4. Practical Edge Case: Rare Events
*   **Problem**: With 1% positives, standard Logistic Regression underestimates probability (bias).
*   **Fix**: **King/Zeng Correction** or Prior Correction. Resample training data to 50/50, train model, then adjust the intercept $\beta_0$ to correct for the sampling bias.

