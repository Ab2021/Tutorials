# Stochastic Reserving (GLM & ODP) - Theoretical Deep Dive

## Overview
We have seen that Chain Ladder is "Distribution-Free" (Mack's Method). However, it can also be framed as a **Generalized Linear Model (GLM)**. Specifically, the **Over-Dispersed Poisson (ODP)** model reproduces the Chain Ladder estimates exactly. This connection unlocks the full power of statistical diagnostics (Residual Plots, AIC/BIC) and allows for **Bootstrapping** to generate full reserve distributions.

---

## 1. Conceptual Foundation

### 1.1 The Chain Ladder as a GLM

**The Insight:**
The Chain Ladder method assumes that the expected incremental loss in cell $(i, j)$ is proportional to the cumulative loss in the previous period $(i, j-1)$.
$$ E[C_{i, j}] = f_{j-1} \cdot C_{i, j-1} $$
This multiplicative structure is identical to a GLM with a **Log Link** function.
$$ \ln(E[Y]) = \text{Row Parameter} + \text{Column Parameter} $$

### 1.2 Over-Dispersed Poisson (ODP)

**Why not standard Poisson?**
*   Standard Poisson assumes Variance = Mean ($\phi = 1$).
*   Insurance claims are highly volatile. Variance $\gg$ Mean.
*   **ODP:** Assumes Variance = $\phi \cdot \text{Mean}$.
    *   $\phi$ (Phi) is the **Dispersion Parameter**.
    *   It scales the variance without changing the mean prediction.

### 1.3 The Bootstrap Connection

*   If we fit an ODP GLM to the triangle, we get fitted values ($\mu_{ij}$) and residuals ($r_{ij}$).
*   **Bootstrapping:** We can resample these residuals to create thousands of "fake" triangles that could have happened.
*   Refitting the GLM to each fake triangle gives us a distribution of reserves.

---

## 2. Mathematical Framework

### 2.1 The ODP GLM Specification

*   **Distribution:** Over-Dispersed Poisson.
*   **Link Function:** Log.
*   **Predictors:** Accident Year ($i$) and Development Year ($j$) as categorical factors.
    $$ \ln(\mu_{ij}) = \alpha_i + \beta_j $$
*   **Constraints:** To make it identifiable, we usually set $\alpha_1 = 0$ or $\sum \beta = 0$.

### 2.2 Pearson Residuals

The "Standardized" residual for ODP is:
$$ r_{ij} = \frac{C_{ij} - \hat{\mu}_{ij}}{\sqrt{\hat{\mu}_{ij}}} $$
*   Note: We divide by $\sqrt{\mu}$, not $\sqrt{\phi \mu}$, because we estimate $\phi$ from the sum of squared residuals later.
*   **Dispersion Estimate:**
    $$ \hat{\phi} = \frac{\sum r_{ij}^2}{N - p} $$
    *   $N$: Number of cells.
    *   $p$: Number of parameters (Rows + Cols - 1).

### 2.3 The Bootstrap Algorithm (England & Verrall)

1.  **Fit GLM:** Get $\hat{\mu}_{ij}$ and $r_{ij}$.
2.  **Resample:** Draw $r^*_{ij}$ from the set of residuals (with replacement).
3.  **Create Pseudo-Data:** $C^*_{ij} = \hat{\mu}_{ij} + r^*_{ij} \sqrt{\hat{\mu}_{ij}}$.
    *   *Correction:* If $C^* < 0$, we might need to adjust (since Poisson must be non-negative), but ODP allows quasi-likelihood.
4.  **Refit:** Run Chain Ladder (or GLM) on $C^*_{ij}$ to get new Reserve $R^*$.
5.  **Repeat:** 10,000 times.

---

## 3. Theoretical Properties

### 3.1 Equivalence Theorem

*   **Theorem:** The Maximum Likelihood Estimates (MLE) of the ODP GLM parameters yield the exact same reserve estimates as the Volume-Weighted Chain Ladder.
*   *Implication:* You don't need to run a GLM to get the *mean*. You run it to get the *variance* and *residuals*.

### 3.2 Process Variance vs. Parameter Variance

*   **Bootstrap Distribution:** Captures **Parameter Variance** (uncertainty in $\alpha, \beta$).
*   **Process Variance:** Must be added separately.
    *   For each bootstrap sample, we simulate the *actual outcome* from the Poisson distribution with mean $R^*$.
    *   Total Variance = Variance of Means + Mean of Variances.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Fitting ODP in Python (statsmodels)

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Data: Long Format (AccidentYear, DevYear, IncrementalLoss)
data = pd.DataFrame({
    'AY': [2020, 2020, 2021, 2021],
    'DY': [1, 2, 1, 2],
    'Loss': [100, 50, 110, 60]
})

# Convert to Categorical
data['AY'] = data['AY'].astype(str)
data['DY'] = data['DY'].astype(str)

# Fit GLM (Poisson Family)
# Note: statsmodels Poisson assumes phi=1. We handle phi manually.
model = smf.glm('Loss ~ AY + DY', data=data, family=sm.families.Poisson()).fit()

print(model.summary())

# Fitted Values (Should match Chain Ladder Incremental)
data['Fitted'] = model.predict()
print(data)

# Calculate Dispersion (Phi)
# Pearson Chi2 / Degrees of Freedom
phi = model.pearson_chi2 / model.df_resid
print(f"Dispersion (Phi): {phi:.3f}")

# Residuals
data['Residual'] = model.resid_pearson
```

### 4.2 The Bootstrap Loop

```python
# Pseudo-code for Bootstrap
n_sims = 1000
reserves = []

residuals = data['Residual'].values
fitted = data['Fitted'].values

for k in range(n_sims):
    # 1. Resample Residuals
    res_star = np.random.choice(residuals, size=len(residuals), replace=True)
    
    # 2. Create Pseudo Data
    # Handle negative values if necessary (Chain Ladder doesn't like negatives)
    pseudo_loss = fitted + res_star * np.sqrt(fitted)
    pseudo_loss = np.maximum(pseudo_loss, 0) 
    
    # 3. Refit Chain Ladder (Fastest way)
    # We use CL because it's equivalent to ODP MLE
    # (Assume we have a function run_chain_ladder(pseudo_loss))
    res_k = run_chain_ladder(pseudo_loss)
    
    # 4. Add Process Variance (Simulate Gamma/Poisson outcome)
    # The reserve is the MEAN. The actual payment is a random draw.
    # Draw from Gamma(mean=res_k, scale=phi)
    final_outcome = np.random.gamma(shape=res_k/phi, scale=phi)
    
    reserves.append(final_outcome)

# Analysis
print(f"Mean Reserve: {np.mean(reserves)}")
print(f"99.5% VaR: {np.percentile(reserves, 99.5)}")
```

---

## 5. Evaluation & Validation

### 5.1 Residual Plots

*   **Plot 1:** Residuals vs. Fitted Values. (Check for Heteroscedasticity).
    *   *ODP Assumption:* Variance $\propto$ Mean. If residuals fan out, this holds.
*   **Plot 2:** Residuals vs. Calendar Year.
    *   *Diagonal Effects:* If you see a trend (e.g., all positive residuals in 2023), you have **Inflation** that the Chain Ladder missed.
    *   *Fix:* Add a Calendar Year parameter to the GLM (Separation Method).

### 5.2 Normality Check

*   Bootstrapping assumes residuals are roughly identically distributed.
*   Check the histogram of residuals. If it's bi-modal or heavily skewed, the bootstrap might be biased.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: "Poisson means Count"**
    *   **Issue:** We are modeling *Loss Amounts*, not counts.
    *   **Reality:** We use ODP as a "Quasi-Likelihood". It doesn't mean the losses are integers. It just means the Variance/Mean relationship looks like Poisson.

2.  **Trap: Negative Incrementals**
    *   **Issue:** Log link fails for negative losses ($ln(-5)$).
    *   **Fix:** Use **MCMC** or specific adjustments (e.g., add a constant to all cells, fit, then subtract).

### 6.2 Implementation Challenges

1.  **Degrees of Freedom:**
    *   In a small triangle (5x5), you have 15 data points and 9 parameters.
    *   $df = 6$. The estimate of $\phi$ will be very unstable.
    *   **Rule:** Don't bootstrap small triangles.

---

## 7. Advanced Topics & Extensions

### 7.1 The "Bootstrap Chain Ladder" Package (R)

*   The industry standard is the `BootChainLadder` function in the `ChainLadder` package in R.
*   Python's `chainladder` package also has a bootstrap module.

### 7.2 Correlated Bootstraps

*   If you have 5 lines of business.
*   Resample the residuals *simultaneously* (using the same index vector) to preserve cross-line correlations.
*   Essential for calculating the **Diversification Benefit**.

---

## 8. Regulatory & Governance Considerations

### 8.1 Solvency II Internal Models

*   Regulators require a "Probability Distribution Forecast" (PDF).
*   ODP Bootstrap is the standard method for generating this PDF.
*   **Validation:** You must demonstrate that the bootstrap distribution covers the actual outcomes (Backtesting).

---

## 9. Practical Example

### 9.1 Worked Example: The "Super-Dispersed" Line

**Scenario:**
*   Medical Malpractice.
*   Chain Ladder Reserve: \$100M.
*   Mack Std Error: \$20M.
*   ODP Bootstrap 99.5%: \$250M.
*   **Why the difference?**
    *   Mack assumes variance is proportional to $C_{ij}$.
    *   ODP Bootstrap might pick up non-linearities or heavy tails in the residuals that Mack misses.
    *   *Result:* The Bootstrap is often more conservative (higher capital) for volatile lines.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Chain Ladder = ODP GLM.**
2.  **Residuals** tell the story of the model's failure.
3.  **Bootstrap** turns a point estimate into a risk profile.

### 10.2 When to Use This Knowledge
*   **Capital Setting:** You cannot calculate VaR without a distribution.
*   **Model Diagnostics:** Use GLM residuals to validate your Chain Ladder assumptions.

### 10.3 Critical Success Factors
1.  **Inspect the Residuals:** They are the "Check Engine Light" of reserving.
2.  **Watch for Outliers:** One bad data point can ruin the bootstrap (since it gets resampled).
3.  **Understand Dispersion:** High $\phi$ means high volatility.

### 10.4 Further Reading
*   **England & Verrall (2002):** "Stochastic Claims Reserving in General Insurance".
*   **Taylor & McGuire:** "GLMs for Loss Reserving".

---

## Appendix

### A. Glossary
*   **Dispersion ($\phi$):** The variance-to-mean ratio.
*   **Quasi-Likelihood:** A method to fit GLMs without specifying the full probability density.
*   **Pearson Residual:** Scaled residual used for bootstrapping.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **ODP Variance** | $\text{Var}(Y) = \phi \cdot E[Y]$ | GLM Assumption |
| **Pearson Resid** | $(Y - \mu) / \sqrt{\mu}$ | Diagnostics |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
