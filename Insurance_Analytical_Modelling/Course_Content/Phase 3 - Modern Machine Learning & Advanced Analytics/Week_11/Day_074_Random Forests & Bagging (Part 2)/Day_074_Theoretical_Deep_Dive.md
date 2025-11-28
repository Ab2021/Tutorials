# Random Forests & Bagging (Part 2) - Theoretical Deep Dive

## Overview
A Random Forest "out of the box" is good, but a **Tuned** Random Forest is excellent. This session covers **Hyperparameter Tuning** strategies and how to open the "Black Box" using **Partial Dependence Plots (PDP)** and **ICE Plots**.

---

## 1. Conceptual Foundation

### 1.1 Hyperparameters vs. Parameters

*   **Parameters:** Learned from data (e.g., Split points in a tree).
*   **Hyperparameters:** Set *before* training (e.g., Number of trees).
*   **Goal:** Find the combination of hyperparameters that minimizes the OOB or CV Error.

### 1.2 The "Black Box" Problem

*   Regulators hate Black Boxes. They want to know: "Does risk increase with Age?"
*   **PDP (Partial Dependence Plot):** Shows the *average* effect of a feature on the prediction, marginalizing over all other features.
*   **ICE (Individual Conditional Expectation):** Shows the effect for *each individual* policyholder. (Reveals heterogeneity).

---

## 2. Mathematical Framework

### 2.1 Key Hyperparameters

1.  **`n_estimators` (Number of Trees):**
    *   *Effect:* More is better (lower variance). No risk of overfitting.
    *   *Trade-off:* Training time.
2.  **`max_features` (Split Subset Size):**
    *   *Effect:* Controls diversity.
    *   *Small:* High diversity, low correlation, but individual trees are weak.
    *   *Large:* Strong trees, but high correlation.
    *   *Default:* $\sqrt{p}$ (Classification), $p/3$ (Regression).
3.  **`min_samples_leaf`:**
    *   *Effect:* Controls the depth/complexity of individual trees.
    *   *Actuarial Tip:* Set this higher (e.g., 50 or 100) to ensure leaf nodes have credible experience.

### 2.2 Partial Dependence (PDP) Math

Let $S$ be the feature of interest (e.g., Age) and $C$ be the complement set (all other features).
The Partial Dependence function $\hat{f}_S(x_S)$ is:
$$ \hat{f}_S(x_S) = \frac{1}{N} \sum_{i=1}^N \hat{f}(x_S, x_{C}^{(i)}) $$
*   *Meaning:* We force every policy in the dataset to have Age = $x_S$, keep their other features ($x_C$) unchanged, predict the premium, and average the results.

---

## 3. Theoretical Properties

### 3.1 Interaction Detection with ICE

*   If all ICE curves are parallel, there is no interaction.
*   If ICE curves cross each other (e.g., for some people, Risk increases with Age; for others, it decreases), there is a strong interaction.
*   *Example:* Age vs. Risk.
    *   Sports Car drivers: Risk drops with Age.
    *   Sedan drivers: Risk is flat.
    *   *Result:* ICE plots will show divergent lines.

### 3.2 Extrapolation (Again)

*   PDPs are only valid within the range of the data.
*   Plotting a PDP for "Age = 100" when the max age in data is 80 is misleading.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Grid Search in Python

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define Parameter Grid
param_grid = {
    'n_estimators': [100, 300],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [10, 50, 100],
    'max_depth': [10, 20, None]
}

# Grid Search with 5-fold CV
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

print(f"Best Params: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_
```

### 4.2 PDP and ICE Plots

```python
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Plot PDP for Feature 0 (e.g., Age)
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(best_rf, X_train, [0], ax=ax, kind='average')
plt.title("PDP: Effect of Age on Claim Cost")
plt.show()

# Plot ICE for Feature 0
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(best_rf, X_train, [0], ax=ax, kind='individual', 
                                        subsample=50, alpha=0.3) # Subsample 50 lines
plt.title("ICE: Individual Effects of Age")
plt.show()
```

---

## 5. Evaluation & Validation

### 5.1 Large Loss Prediction (Case Study)

*   **Problem:** Large losses (Severity) are rare and right-skewed.
*   **RF Advantage:** RF handles skewness well (no need for Log transform, though it helps).
*   **Tuning:**
    *   Increase `min_samples_leaf` to avoid fitting to single outlier claims.
    *   Use `MAE` (Mean Absolute Error) criterion instead of `MSE` to reduce sensitivity to extreme outliers (if supported by library).

### 5.2 Stability of Explanations

*   Run the PDP on different folds.
*   If the shape of the Age curve changes from "U-shape" to "Linear", the model is unstable.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: OOB vs. CV for Tuning**
    *   You *can* tune using OOB score (faster).
    *   But standard `GridSearchCV` uses CV.
    *   *Pro Tip:* Write a custom loop to tune using OOB if data is massive.

2.  **Trap: Correlated Features in PDP**
    *   PDP assumes features are independent.
    *   If Age and Experience are correlated, forcing Age=20 while keeping Experience=30 creates an "Impossible Data Point" (20-year-old with 30 years experience).
    *   *Fix:* Use **ALE (Accumulated Local Effects)** plots (Day 92).

### 6.2 Implementation Challenges

1.  **Computational Cost:**
    *   Grid Search grows exponentially ($4 \times 2 \times 3 \times 3 = 72$ fits).
    *   *Solution:* **RandomizedSearchCV**. Randomly sample 20 combinations. Usually finds a result 95% as good in 10% of the time.

---

## 7. Advanced Topics & Extensions

### 7.1 Feature Interactions

*   **Friedman's H-statistic:** Measures the strength of interactions in the model.
*   If $H > 0$, the PDP of two features is not just the sum of their individual PDPs.

### 7.2 Monotonic Constraints (in RF)

*   Some implementations (like XGBoost) allow monotonic constraints.
*   Standard sklearn RF does not.
*   *Workaround:* Post-hoc smoothing of the predictions (Isotonic Regression).

---

## 8. Regulatory & Governance Considerations

### 8.1 "The Model is too Complex"

*   **Regulator:** "I can't audit 500 trees."
*   **Actuary:** "Here is the PDP. It shows the exact relationship between Age and Price. It matches our actuarial intuition (U-shape). The complexity is just to handle the noise."

---

## 9. Practical Example

### 9.1 Worked Example: The "Wildfire" Model

**Scenario:**
*   Predicting Wildfire Risk Score (0-100).
*   **Features:** Temperature, Humidity, Vegetation, Wind.
*   **Interaction:** High Wind + Low Humidity = Extreme Risk.
*   **GLM:** Missed the interaction (unless manually added).
*   **RF:** Captured it.
*   **PDP:** Showed that Wind Risk is flat *until* Humidity drops below 20%, then it spikes.
*   **Action:** New underwriting rule for "Red Flag Days".

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Grid Search** finds the best hyperparameters.
2.  **PDP** shows the average effect.
3.  **ICE** shows the individual effect.

### 10.2 When to Use This Knowledge
*   **Model Validation:** Proving to the business that the model isn't doing crazy things.
*   **Pricing:** Understanding the shape of risk curves.

### 10.3 Critical Success Factors
1.  **Don't Over-Tune:** The gain from 500 to 1000 trees is minimal.
2.  **Visualize:** Always look at the PDPs.

### 10.4 Further Reading
*   **Molnar:** "Interpretable Machine Learning" (The Bible of PDP/ICE).
*   **Goldstein et al.:** "Peeking Inside the Black Box".

---

## Appendix

### A. Glossary
*   **Grid Search:** Exhaustive search.
*   **Random Search:** Stochastic search.
*   **Marginalize:** Average over.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **PDP** | $\frac{1}{N} \sum \hat{f}(x_S, x_C^{(i)})$ | Interpretation |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
