# Gradient Boosting Machines (GBM) (Part 1) - Theoretical Deep Dive

## Overview
Random Forests build trees in parallel (Bagging). **Gradient Boosting Machines (GBM)** build trees sequentially (Boosting). Each tree corrects the mistakes of the previous one. This session covers the **Boosting Theory**, **Gradient Descent in Function Space**, and the **Tweedie Loss Function** for insurance.

---

## 1. Conceptual Foundation

### 1.1 Bagging vs. Boosting

*   **Bagging (Random Forest):**
    *   Trees are independent.
    *   Goal: Reduce Variance.
    *   Analogy: A democracy of experts voting.
*   **Boosting (GBM):**
    *   Trees are dependent.
    *   Goal: Reduce Bias.
    *   Analogy: A team where each member fixes the errors of the previous member.

### 1.2 The Boosting Mechanism

1.  **Step 1:** Fit a simple model $F_0(x)$ (e.g., the mean).
2.  **Step 2:** Calculate the errors (Residuals): $r_i = y_i - F_0(x_i)$.
3.  **Step 3:** Fit a weak tree $h_1(x)$ to predict the residuals $r_i$.
4.  **Step 4:** Update the model: $F_1(x) = F_0(x) + \eta h_1(x)$.
    *   $\eta$: Learning Rate (Shrinkage).
5.  **Repeat:** Fit $h_2(x)$ to the new residuals.

### 1.3 Gradient Descent in Function Space

*   Why fit residuals?
*   For MSE Loss ($L = \frac{1}{2}(y - F)^2$), the negative gradient is:
    $$ -\frac{\partial L}{\partial F} = y - F = Residual $$
*   So, fitting the residual is equivalent to taking a step in the direction of the negative gradient.
*   This allows us to optimize *any* differentiable loss function (Poisson, Gamma, Tweedie) just by changing the gradient formula.

---

## 2. Mathematical Framework

### 2.1 Loss Functions for Insurance

1.  **Squared Error (Gaussian):**
    *   $L = (y - F)^2$.
    *   Gradient: $y - F$.
    *   *Use:* General regression (not great for insurance).
2.  **Poisson (Frequency):**
    *   $L = F - y \log(F)$ (Negative Log Likelihood).
    *   Gradient: $y - F$ (if using log-link).
    *   *Use:* Claim Counts.
3.  **Tweedie (Pure Premium):**
    *   Handles the "Zero-Inflated" mass.
    *   Parameter $p$:
        *   $p=1$: Poisson.
        *   $p=2$: Gamma.
        *   $1 < p < 2$: Compound Poisson-Gamma (Tweedie).
    *   *Use:* Modeling Loss Cost directly.

### 2.2 Regularization

GBMs are prone to overfitting because they aggressively reduce bias.
1.  **Learning Rate ($\eta$):**
    *   Scale the contribution of each tree (e.g., 0.01 or 0.1).
    *   Lower $\eta$ requires more trees but generalizes better.
2.  **Subsampling:**
    *   Train each tree on a random fraction (e.g., 50%) of data. (Stochastic Gradient Boosting).
3.  **Tree Constraints:** Max Depth, Min Samples.

---

## 3. Theoretical Properties

### 3.1 The "Weak Learner" Assumption

*   In Boosting, the individual trees should be **weak** (shallow).
*   *Random Forest:* Deep trees (low bias, high variance).
*   *GBM:* Shallow trees (high bias, low variance). The boosting process fixes the bias.

### 3.2 Monotonicity

*   GBMs can enforce monotonic constraints.
*   *Example:* We can force the model such that "As Insurance Score increases, Premium must decrease".
*   This is crucial for regulatory approval.

---

## 4. Modeling Artifacts & Implementation

### 4.1 GBM in Python (sklearn)

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance

# Simulate Poisson Data
X = np.random.rand(1000, 5)
mu = np.exp(2 * X[:, 0] - X[:, 1])
y = np.random.poisson(mu)

# Fit GBM with Poisson Loss
gbm = GradientBoostingRegressor(
    loss='poisson', # Specific for counts
    learning_rate=0.1,
    n_estimators=100,
    max_depth=3,
    random_state=42
)

gbm.fit(X, y)

# Predict
y_pred = gbm.predict(X)

# Evaluate
deviance = mean_poisson_deviance(y, y_pred)
print(f"Poisson Deviance: {deviance:.4f}")
```

### 4.2 Tweedie Loss (Pure Premium)

*   `sklearn` supports Tweedie via `loss='tweedie'` (in newer versions) or `HistGradientBoostingRegressor`.
*   Standard `GradientBoostingRegressor` is slower. `HistGradientBoostingRegressor` is the modern, fast implementation (similar to LightGBM).

```python
from sklearn.ensemble import HistGradientBoostingRegressor

# Fit Tweedie GBM (p=1.5)
tweedie_gbm = HistGradientBoostingRegressor(
    loss='tweedie',
    power=1.5, # Compound Poisson-Gamma
    learning_rate=0.1,
    max_iter=100
)

tweedie_gbm.fit(X, y) # y would be claim amount here
```

---

## 5. Evaluation & Validation

### 5.1 Early Stopping

*   **Problem:** How many trees? 100? 1000?
*   **Solution:** Monitor the Validation Error. Stop adding trees when the error stops decreasing.
*   *Implementation:* `n_iter_no_change=10`.

### 5.2 Lift Charts (Double Lift)

*   Compare GBM vs. GLM.
*   Sort policies by GBM prediction / GLM prediction ratio.
*   Plot actual loss ratio in each bucket.
*   *Goal:* Show that GBM identifies risk that GLM misses.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Overfitting with High Learning Rate**
    *   If $\eta=1.0$, the model memorizes the residuals instantly.
    *   *Fix:* Always use $\eta < 0.1$.

2.  **Trap: Outliers in Residuals**
    *   If one claim is \$10M, the residual is massive. The next tree focuses entirely on fixing this one claim.
    *   *Fix:* Use Robust Loss functions (Huber Loss) or clip the target.

### 6.2 Implementation Challenges

1.  **Training Time:**
    *   GBM is sequential. It cannot be parallelized easily (unlike Random Forest).
    *   *Solution:* Use XGBoost or LightGBM (Day 77) which parallelize the *tree construction* (not the boosting).

---

## 7. Advanced Topics & Extensions

### 7.1 Interaction Depth

*   The `max_depth` of the trees controls the order of interactions.
*   `max_depth=1`: Additive model (No interactions).
*   `max_depth=2`: Pairwise interactions ($X_1 \times X_2$).
*   `max_depth=3`: Three-way interactions.
*   *Actuarial Standard:* Depth 2-4 is usually sufficient.

---

## 8. Regulatory & Governance Considerations

### 8.1 Stability

*   GBMs can be unstable if not regularized.
*   **Shapley Values (SHAP):** Essential for explaining GBM predictions to regulators (Day 92).

---

## 9. Practical Example

### 9.1 Worked Example: The "Residual" Doctor

**Scenario:**
*   You have a standard GLM.
*   You fit a GBM to the *residuals* of the GLM ($y - GLM_{pred}$).
*   **Result:** The GBM finds that "Color of Car" (which was missing in GLM) explains the remaining error.
*   **Action:** Add "Color" to the GLM. (Boosting as a feature discovery tool).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Boosting** reduces bias by fixing errors sequentially.
2.  **Learning Rate** is the brake pedal against overfitting.
3.  **Tweedie Loss** is the gold standard for pure premium.

### 10.2 When to Use This Knowledge
*   **Pricing:** When you need maximum accuracy.
*   **Kaggle:** GBMs win almost every tabular competition.

### 10.3 Critical Success Factors
1.  **Tuning:** GBMs require more tuning than Random Forests.
2.  **Early Stopping:** Essential to prevent overfitting.

### 10.4 Further Reading
*   **Friedman (2001):** "Greedy Function Approximation: A Gradient Boosting Machine".
*   **Natekin & Knoll:** "Gradient Boosting Machines, a Tutorial".

---

## Appendix

### A. Glossary
*   **Residual:** Difference between Actual and Predicted.
*   **Shrinkage:** Learning Rate.
*   **Stochastic Boosting:** Subsampling rows/columns.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Boosting Update** | $F_m(x) = F_{m-1}(x) + \eta h_m(x)$ | Model Update |
| **Poisson Loss** | $F - y \log(F)$ | Count Data |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
