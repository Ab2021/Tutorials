# Gradient Boosting Machines (GBM) (Part 2) - Theoretical Deep Dive

## Overview
A standard GBM is powerful but prone to overfitting. To make it production-ready for insurance, we need **Regularization**. This session covers **Stochastic Gradient Boosting**, **L1/L2 Regularization**, and the critical technique of **Early Stopping**.

---

## 1. Conceptual Foundation

### 1.1 The Overfitting Risk

*   GBMs reduce bias aggressively.
*   If left unchecked, a GBM will memorize the residuals of the training set (Zero Training Error, High Test Error).
*   **Solution:** Constrain the learning process.

### 1.2 Stochastic Gradient Boosting

*   Inspired by Bagging.
*   **Row Subsampling (`subsample`):**
    *   At each iteration, sample a fraction (e.g., 0.7) of the training data *without replacement*.
    *   Fit the next tree on this subset.
    *   *Benefit:* Faster training, lower variance (prevents the model from obsessing over specific outliers).
*   **Column Subsampling (`colsample_bytree`):**
    *   At each iteration, sample a fraction of features.
    *   *Benefit:* Decorrelates trees (similar to Random Forest).

### 1.3 Regularization (L1 & L2)

*   We can add a penalty term to the Loss Function based on the leaf weights ($w$).
    $$ Obj = Loss + \gamma T + \frac{1}{2} \lambda \sum w^2 + \alpha \sum |w| $$
    *   $\gamma$ (Gamma): Minimum loss reduction required to make a split.
    *   $\lambda$ (Lambda): L2 Regularization (Ridge). Smooths the weights.
    *   $\alpha$ (Alpha): L1 Regularization (Lasso). Forces some weights to zero (Feature Selection).

---

## 2. Mathematical Framework

### 2.1 Learning Rate vs. Number of Trees

*   There is an inverse relationship between Learning Rate ($\eta$) and Number of Trees ($M$).
*   Lower $\eta$ (e.g., 0.01) requires higher $M$ (e.g., 1000).
*   **Strategy:**
    1.  Fix $\eta$ to a small value (0.05 or 0.1).
    2.  Tune $M$ using Early Stopping.
    3.  Do not tune $\eta$ and $M$ simultaneously (waste of time).

### 2.2 Early Stopping

*   Instead of guessing $M$, we monitor the Validation Error.
*   **Algorithm:**
    1.  Split data into Train and Validation.
    2.  Train tree 1, 2, 3...
    3.  After each tree, calculate Error on Validation set.
    4.  If Error doesn't improve for `patience` rounds (e.g., 10), STOP.
    5.  Return the model at the best iteration.

---

## 3. Theoretical Properties

### 3.1 Frequency-Severity Modeling

*   **Traditional Approach:** $Pure Premium = Frequency \times Severity$.
*   **GBM Approach:**
    1.  **Frequency Model:** GBM with Poisson Loss. Predicts Claim Count.
    2.  **Severity Model:** GBM with Gamma Loss. Predicts Average Cost per Claim.
    3.  **Combination:** Multiply predictions.
*   *Why separate?* Different drivers. "Age" might increase Frequency but decrease Severity.

### 3.2 The "Shrinkage" Effect

*   The Learning Rate $\eta$ "shrinks" the contribution of each new tree.
*   $F_m(x) = F_{m-1}(x) + \eta h_m(x)$.
*   This prevents the model from trusting any single tree too much. It forces the model to learn *slowly*.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Early Stopping in Python (sklearn)

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit with Early Stopping
gbm = GradientBoostingRegressor(
    n_estimators=1000, # Set high
    learning_rate=0.05,
    validation_fraction=0.1, # Internal validation split
    n_iter_no_change=10, # Patience
    tol=0.0001,
    random_state=42
)

gbm.fit(X_train, y_train)

print(f"Optimal Trees: {gbm.n_estimators_}")
```

### 4.2 Stochastic Boosting (Subsampling)

```python
# Stochastic Gradient Boosting
sgb = GradientBoostingRegressor(
    subsample=0.7, # Use 70% of data for each tree
    max_features='sqrt', # Use sqrt(features) for each split
    learning_rate=0.05,
    n_estimators=500
)

sgb.fit(X_train, y_train)
```

---

## 5. Evaluation & Validation

### 5.1 The Learning Curve

*   Plot Training Error vs. Validation Error over iterations.
*   **Good:** Training Error decreases, Validation Error decreases then flattens.
*   **Overfitting:** Training Error decreases, Validation Error decreases then *increases*. (Early stopping catches this inflection point).

### 5.2 Hyperparameter Tuning Strategy

1.  **Step 1:** Fix `learning_rate=0.1`. Tune `n_estimators` with Early Stopping.
2.  **Step 2:** Tune Tree Parameters (`max_depth`, `min_samples_leaf`).
3.  **Step 3:** Tune Stochastic Parameters (`subsample`, `max_features`).
4.  **Step 4:** Lower `learning_rate` to 0.01 and increase `n_estimators` proportionally.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Subsample < 0.5**
    *   If `subsample` is too low, the trees become too weak and unstable.
    *   *Rule of Thumb:* Keep `subsample` between 0.6 and 0.9.

2.  **Trap: L1 Regularization on Correlated Features**
    *   L1 will pick *one* feature and zero out the others.
    *   This is good for feature selection but can be unstable if the "chosen one" changes between runs.

### 6.2 Implementation Challenges

1.  **Data Leakage in Early Stopping:**
    *   Do *not* use the Test set for Early Stopping.
    *   Use a separate Validation set.
    *   Final evaluation must be on the held-out Test set.

---

## 7. Advanced Topics & Extensions

### 7.1 DART (Dropouts meet Multiple Additive Regression Trees)

*   Inspired by Deep Learning "Dropout".
*   Randomly drop (ignore) some trees during the training process.
*   Prevents the later trees from just fixing the minor errors of the first few trees.
*   *Available in:* XGBoost, LightGBM.

### 7.2 Monotonic Constraints

*   Crucial for Insurance.
*   `monotone_constraints` parameter in XGBoost.
*   Ensures the rate curve doesn't "wiggle" illogically.

---

## 8. Regulatory & Governance Considerations

### 8.1 The "Black Box" Defense

*   **Regulator:** "Your GBM is too complex."
*   **Actuary:** "We used L1 regularization to remove irrelevant variables and Monotonic Constraints to ensure logical behavior. It's a disciplined model, not a wild one."

---

## 9. Practical Example

### 9.1 Worked Example: The "Frequency-Severity" Pipeline

**Scenario:**
*   **Frequency Model:** GBM (Poisson), `max_depth=3`, `subsample=0.8`.
*   **Severity Model:** GBM (Gamma), `max_depth=2`, `subsample=0.7`. (Severity is noisier, so simpler model).
*   **Prediction:** $E[Loss] = E[Freq] \times E[Sev]$.
*   **Result:** Outperforms Tweedie GBM slightly because it allows different features to drive Freq vs. Sev.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Stochastic Boosting** adds randomness to reduce variance.
2.  **Early Stopping** prevents overfitting automatically.
3.  **Regularization** (L1/L2) keeps the weights small.

### 10.2 When to Use This Knowledge
*   **Production Models:** Never deploy a GBM without Early Stopping and Subsampling.

### 10.3 Critical Success Factors
1.  **Patience:** Don't stop too early (`patience=5` is too low, use 10-20).
2.  **Validation Set:** Must be representative.

### 10.4 Further Reading
*   **Friedman (2002):** "Stochastic Gradient Boosting".
*   **XGBoost Documentation:** "Parameters".

---

## Appendix

### A. Glossary
*   **Stochastic:** Random.
*   **Patience:** How many bad rounds to wait before quitting.
*   **Lasso:** L1 Regularization.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Regularized Obj** | $L + \gamma T + \frac{1}{2}\lambda ||w||^2$ | XGBoost Objective |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
