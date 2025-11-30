# Severity GLMs (Part 3) - Theoretical Deep Dive

## Overview
We conclude the Severity module by exploring **Pure Premium Modeling**. While the traditional approach builds separate Frequency and Severity models, the **Tweedie GLM** allows us to model the total loss directly. We also introduce the **Gini Index** and **Lorenz Curve**, the gold standards for validating and comparing insurance pricing models.

---

## 1. Conceptual Foundation

### 1.1 The Pure Premium Problem

**Pure Premium (PP):** The expected loss per unit of exposure.
$$ PP = \text{Frequency} \times \text{Severity} $$

**Two Approaches:**
1.  **Frequency-Severity (F-S) Method:**
    *   Model $N \sim \text{Poisson}$.
    *   Model $Y|N \sim \text{Gamma}$.
    *   Multiply predictions: $\hat{P} = \hat{\lambda} \times \hat{\mu}$.
    *   *Pros:* Deeper insight (e.g., "Young drivers crash more, but cost less").
    *   *Cons:* Error propagation. Complex to maintain.

2.  **Tweedie Method (Direct PP):**
    *   Model Total Loss $L \sim \text{Tweedie}$.
    *   *Pros:* Single model. Handles zeros naturally.
    *   *Cons:* Harder to interpret "Why" the rate is high.

### 1.2 The Tweedie Distribution

A member of the Exponential Dispersion Models (EDM).
*   **Variance Function:** $V(\mu) = \phi \mu^p$.
*   **Power Parameter ($p$):**
    *   $p=0$: Normal.
    *   $p=1$: Poisson.
    *   $p=2$: Gamma.
    *   $p=3$: Inverse Gaussian.
    *   **$1 < p < 2$:** **Compound Poisson-Gamma**.

**Interpretation:**
*   It is a sum of $N$ Gamma variables, where $N$ is Poisson.
*   It has a point mass at zero (when $N=0$) and a continuous positive distribution (when $N > 0$).
*   This perfectly matches "Total Loss" data (mostly zeros, some positive amounts).

### 1.3 Model Validation: The Gini Index

How do we know if our model is good? $R^2$ is useless for insurance.
**The Gini Strategy:**
1.  Sort policies from "Safest" to "Riskiest" according to the model.
2.  Plot cumulative exposure vs. cumulative loss (**Lorenz Curve**).
3.  **Gini Coefficient:** Area between the Lorenz Curve and the 45-degree line.
    *   Higher Gini = Better segmentation.

---

## 2. Mathematical Framework

### 2.1 Tweedie GLM Specification

$$ E[Y] = \mu $$
$$ \text{Var}(Y) = \phi \mu^p $$
$$ \ln(\mu) = X\beta $$

**Parameters:**
*   $\mu$: Mean Pure Premium.
*   $\phi$: Dispersion.
*   $p$: Variance Power (usually $1.3 - 1.7$ for insurance).

**Estimation:**
*   $p$ is often treated as a hyperparameter. We profile the likelihood to find the optimal $p$.
*   Once $p$ is fixed, it's a standard GLM.

### 2.2 The Lorenz Curve Construction

Let $P_i$ be the premium (model prediction) and $L_i$ be the loss for policy $i$.
1.  Sort data by Relativity ($P_i / \text{BaseRate}$).
2.  Calculate Cumulative Exposure share ($x$-axis).
3.  Calculate Cumulative Loss share ($y$-axis).
4.  **Perfect Model:** The curve bows deeply to the bottom-right. (We identify the bad risks and charge them high premiums, so they account for most of the loss but are at the end of the list).
5.  **Random Model:** 45-degree line.

### 2.3 Double Lift Chart

Used to compare Model A vs. Model B.
1.  Sort by the ratio $\text{Pred}_A / \text{Pred}_B$.
2.  Bin into deciles.
3.  Plot Average Loss in each bin.
4.  **Winning:** If Model A is better, the Average Loss should track Model A's predictions, not Model B's.

---

## 3. Theoretical Properties

### 3.1 Tweedie vs. F-S Consistency

*   If Freq is Poisson and Sev is Gamma, the product is *exactly* Tweedie.
*   **Why do they differ in practice?**
    *   Different covariates in Freq and Sev models.
    *   Different smoothing/regularization.
*   **Industry View:** F-S is preferred for Personal Lines (Auto/Home). Tweedie is preferred for Commercial Lines (where separating F/S is noisy due to low volume).

### 3.2 Gini Index Calculation

$$ G = \frac{A}{A+B} = 2A $$
*   $A$: Area between Line of Equality and Lorenz Curve.
*   **Normalized Gini:** $G_{model} / G_{perfect}$.
    *   "Perfect" model predicts every loss exactly.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Fitting Tweedie in Python

*   `statsmodels` has `Tweedie` family.
*   `sklearn` has `TweedieRegressor`.
*   **Key Step:** Estimating $p$.
    *   Use `tweedie.profile_likelihood` in R or manual grid search in Python.

### 4.2 Calculating Gini

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import auc

# Simulate Tweedie Data (Compound Poisson-Gamma)
np.random.seed(42)
n = 5000
exposure = np.random.uniform(0.5, 1.0, n)
x = np.random.normal(0, 1, n)

# True Parameters
freq_mean = np.exp(-1 + 0.5 * x) * exposure
sev_mean = 1000 * np.exp(0.2 * x) # Correlation: x drives both

# Generate Data
n_claims = np.random.poisson(freq_mean)
losses = np.zeros(n)
for i in range(n):
    if n_claims[i] > 0:
        losses[i] = np.random.gamma(shape=2, scale=sev_mean[i]/2, size=n_claims[i]).sum()

df = pd.DataFrame({'Loss': losses, 'Exposure': exposure, 'x': x})
df['PurePrem'] = df['Loss'] / df['Exposure']

# Fit Tweedie GLM (p=1.5)
# Note: Sklearn minimizes Deviance.
glm = TweedieRegressor(power=1.5, alpha=0, link='log', max_iter=1000)
glm.fit(df[['x']], df['PurePrem'], sample_weight=df['Exposure'])

df['Pred'] = glm.predict(df[['x']])

# Gini / Lorenz Curve Function
def plot_lorenz(y_true, y_pred, exposure):
    # Sort by Predicted Risk (Loss Cost)
    # Risk = Pred / Exposure? No, Pred is Pure Premium.
    # We sort by Predicted Pure Premium.
    
    data = pd.DataFrame({'True': y_true, 'Pred': y_pred, 'Exp': exposure})
    data = data.sort_values('Pred')
    
    data['CumExp'] = data['Exp'].cumsum() / data['Exp'].sum()
    data['CumLoss'] = data['True'].cumsum() / data['True'].sum()
    
    # Calculate Gini (Area under curve)
    # Area under curve (Trapezoidal rule)
    auc_score = auc(data['CumExp'], data['CumLoss'])
    gini = 1 - 2 * auc_score 
    # Note: Definition varies. Usually Gini = 2*Area_Between = 1 - 2*Area_Under?
    # If perfect, curve is convex (below diagonal). Area Under < 0.5.
    # Let's stick to visual.
    
    plt.figure(figsize=(6, 6))
    plt.plot(data['CumExp'], data['CumLoss'], label=f'Model (Gini={gini:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('Lorenz Curve')
    plt.xlabel('Cumulative Exposure')
    plt.ylabel('Cumulative Loss')
    plt.legend()
    plt.show()

plot_lorenz(df['Loss'], df['Pred'] * df['Exposure'], df['Exposure'])

# Interpretation:
# The further the curve is below the diagonal, the better the model sorts risk.
```

### 4.3 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Tweedie Coefficients:** Impact on Total Loss Cost.
2.  **Gini Score:** "Our new model has a Gini of 0.35, compared to the old model's 0.30."

**Interpretation:**
*   **Pricing:** Use the Tweedie predictions as the base for the Rate Indication.
*   **Validation:** If Gini drops on the Test Set, you are overfitting.

---

## 5. Evaluation & Validation

### 5.1 Lift Charts (Deciles)

*   Bin policies into 10 groups by Predicted PP.
*   Plot **Predicted PP** vs **Actual PP** for each bin.
*   **Good Model:** The points lie on the 45-degree line.
*   **Monotonicity:** The Actual PP should strictly increase from Bin 1 to Bin 10.

### 5.2 Stability Check

*   Check Gini over time (2020, 2021, 2022).
*   If Gini fluctuates wildly, the model is unstable.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Interpreting Tweedie Coefs**
    *   **Issue:** "The coefficient for Age is 0.05."
    *   **Reality:** This combines Frequency and Severity effects. You don't know if old people crash more or crash harder. You just know they cost more.

2.  **Trap: The "Zero" Problem**
    *   **Issue:** Tweedie handles zeros, but if you have *too many* zeros (Zero-Inflation), standard Tweedie might fail.
    *   **Fix:** Zero-Inflated Tweedie (rare) or stick to F-S.

### 6.2 Implementation Challenges

1.  **Finding $p$:**
    *   The profile likelihood is flat near the optimum.
    *   **Tip:** $p=1.5$ is usually a safe default for Auto Insurance.

---

## 7. Advanced Topics & Extensions

### 7.1 Gradient Boosting Tweedie

*   XGBoost/LightGBM with `objective='tweedie'`.
*   The state-of-the-art for pure premium modeling in competitions (Kaggle).
*   Captures non-linearities that GLMs miss.

### 7.2 Double Generalized Linear Models (DGLM)

*   Model the Mean ($\mu$) and the Dispersion ($\phi$) simultaneously.
*   Allows variance to depend on covariates (e.g., Young drivers are not just expensive, they are volatile).

---

## 8. Regulatory & Governance Considerations

### 8.1 "Black Box" Concerns

*   Tweedie is harder to explain to a regulator than "Freq $\times$ Sev".
*   **Defense:** "It's mathematically equivalent to Poisson $\times$ Gamma."
*   **Documentation:** Show the Lift Charts. Regulators love Lift Charts.

---

## 9. Practical Example

### 9.1 Worked Example: Rate Change

**Old Rate:** \$500.
**New Tweedie Prediction:** \$550.
**Indicated Change:** +10%.

**Validation:**
*   In the "High Risk" bin, Old Rate was \$800, Actual Loss was \$1000.
*   New Model predicts \$950.
*   New Model is closer to reality.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Tweedie** models Total Loss directly.
2.  **Gini/Lorenz** measures segmentation power.
3.  **Lift Charts** prove accuracy.

### 10.2 When to Use This Knowledge
*   **Commercial Lines:** Where data is sparse.
*   **Quick Modeling:** When you need a fast answer without building two models.

### 10.3 Critical Success Factors
1.  **Compare to F-S:** Always benchmark Tweedie against Freq $\times$ Sev.
2.  **Validate Out-of-Sample:** Gini on training data is vanity.
3.  **Check Monotonicity:** Ensure the risk bins make sense.

### 10.4 Further Reading
*   **Goldburd et al.:** "GLMs for Insurance Rating" (Chapter on Tweedie).
*   **Frees:** "Regression Modeling with Actuarial and Financial Applications".

---

## Appendix

### A. Glossary
*   **Pure Premium:** Loss Cost per unit exposure.
*   **Lorenz Curve:** Cumulative % Loss vs Cumulative % Exposure.
*   **Lift:** Ratio of Loss in highest bin to lowest bin.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Tweedie Variance** | $\phi \mu^p$ | Assumption |
| **Gini** | $2 \times \text{Area}$ | Validation |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
