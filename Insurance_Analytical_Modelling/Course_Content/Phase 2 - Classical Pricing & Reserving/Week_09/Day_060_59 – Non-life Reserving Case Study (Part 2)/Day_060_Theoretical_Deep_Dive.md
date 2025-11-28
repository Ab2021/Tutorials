# Non-life Reserving Case Study (Part 2) - Theoretical Deep Dive

## Overview
A single number (Point Estimate) is almost guaranteed to be wrong. Stochastic Reserving provides a **Distribution** of possible outcomes. This session covers the **Bootstrapped ODP (Over-Dispersed Poisson)** model, the calculation of **VaR (Value at Risk)** and **TVaR**, and how to communicate uncertainty to stakeholders.

---

## 1. Conceptual Foundation

### 1.1 Why Stochastic?

*   **Deterministic:** "The reserve is \$100M."
    *   *Problem:* Is it \$100M $\pm$ \$5M or $\pm$ \$50M?
*   **Stochastic:** "The mean reserve is \$100M. The 99.5% VaR is \$140M."
    *   *Benefit:* Allows for capital setting (Solvency II) and risk margin calculation.

### 1.2 The ODP Model

*   **Assumption:** Incremental losses follow an Over-Dispersed Poisson distribution.
    *   $E[C_{i,j}] = m_{i,j}$
    *   $Var[C_{i,j}] = \phi \cdot m_{i,j}$
*   **GLM Equivalent:** This is equivalent to a GLM with a Log Link and Poisson error structure, where the predictors are Accident Year and Development Year factors.

### 1.3 Bootstrapping

*   **Goal:** To generate 10,000 "Alternative Histories".
*   **Process:**
    1.  Fit the ODP model to the actual triangle.
    2.  Calculate **Pearson Residuals** (Actual - Fitted) / Scale.
    3.  **Resample** the residuals with replacement.
    4.  Create a "Pseudo-Triangle" using the resampled residuals.
    5.  Re-fit the model to the Pseudo-Triangle.
    6.  Project future losses.
    7.  Repeat 10,000 times.

---

## 2. Mathematical Framework

### 2.1 Pearson Residuals

$$ r_{i,j} = \frac{C_{i,j} - m_{i,j}}{\sqrt{m_{i,j}}} $$
*   We resample these $r_{i,j}$ to get $r^*_{i,j}$.
*   Then generate pseudo-data: $C^*_{i,j} = m_{i,j} + r^*_{i,j} \sqrt{m_{i,j}}$.
*   *Note:* We must adjust for degrees of freedom (Hat Matrix).

### 2.2 Risk Measures

*   **VaR (Value at Risk):** The $p$-th percentile.
    *   $VaR_{99.5\%} = \text{The value where 99.5\% of simulations are lower.}$
*   **TVaR (Tail VaR):** The average of values *above* the VaR.
    *   $TVaR_{99\%} = E[X | X > VaR_{99\%}]$.
    *   *Why TVaR?* It is a "Coherent Risk Measure" (Sub-additive). VaR is not.

---

## 3. Theoretical Properties

### 3.1 Process Variance vs. Parameter Variance

*   **Process Variance:** Even if we know the true mean, the actual outcome is random (Poisson noise).
*   **Parameter Variance:** We don't know the true mean. Our LDFs are estimates.
*   **Bootstrap:** Captures *both*. The resampling captures Parameter Variance. The simulation of the final outcome (using Gamma or Poisson) captures Process Variance.

### 3.2 The "Mack" vs. "Bootstrap"

*   **Mack's Method:** Analytical formula for Standard Error. (Fast, but assumes log-linear variance).
*   **Bootstrap:** Simulation-based. (Slow, but flexible).
*   *Result:* They usually give similar Standard Errors, but Bootstrap gives the full *shape* of the distribution (Skewness).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Bootstrapping in Python (chainladder)

```python
import pandas as pd
import chainladder as cl

# Load Data (Triangle)
# 'cl.load_sample' loads a sample triangle
triangle = cl.load_sample('genins')

# 1. Fit the Bootstrap ODP Model
# n_sims = 1000
bootstrap = cl.BootstrapODP(n_sims=1000)
bootstrap.fit(triangle)

# 2. Get the Distribution of Ultimate Reserves
# 'ibnr' is the simulated IBNR for each simulation
ibnr_dist = bootstrap.ibnr_.sum('origin').values # Sum across all Accident Years

# 3. Calculate Statistics
mean_ibnr = ibnr_dist.mean()
std_ibnr = ibnr_dist.std()
cv_ibnr = std_ibnr / mean_ibnr

# 4. Calculate VaR and TVaR
var_995 = np.percentile(ibnr_dist, 99.5)
tvar_99 = ibnr_dist[ibnr_dist > np.percentile(ibnr_dist, 99)].mean()

print(f"Mean IBNR: {mean_ibnr:,.0f}")
print(f"Std Dev: {std_ibnr:,.0f} (CV: {cv_ibnr:.1%})")
print(f"VaR 99.5%: {var_995:,.0f}")
print(f"TVaR 99%: {tvar_99:,.0f}")
```

### 4.2 Visualizing the Output

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(ibnr_dist, kde=True, color='blue', bins=50)
plt.axvline(mean_ibnr, color='red', linestyle='--', label='Mean')
plt.axvline(var_995, color='green', linestyle='--', label='VaR 99.5%')
plt.title("Distribution of Total IBNR Reserves")
plt.xlabel("Reserve Amount")
plt.legend()
plt.show()
```

---

## 5. Evaluation & Validation

### 5.1 Convergence Check

*   Did 1,000 simulations stabilize the tail?
*   **Test:** Run 10,000. If VaR changes significantly, you need more sims.
*   **Rule of Thumb:** For mean/std, 1,000 is fine. For VaR 99.5%, you need 10,000+.

### 5.2 Residual Plots

*   Plot Pearson Residuals vs. Development Year.
*   **Expectation:** Random scatter around 0.
*   **Bad Sign:** A "Fan" shape (Heteroscedasticity) or a Trend (Model misspecification).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Negative Incrementals**
    *   **Issue:** ODP assumes incrementals are positive (Poisson).
    *   **Reality:** Salvage/Subrogation causes negatives.
    *   **Fix:** Use the **Mack Bootstrap** (which works on LDFs directly) or handle negatives as "Paid < 0" (Log-Normal trick).

2.  **Trap: Correlation between Lines**
    *   **Issue:** Bootstrapping Auto and WC independently.
    *   **Reality:** Inflation affects both.
    *   **Fix:** Use **Synchronous Bootstrapping** (Resample the *same* residual index for both lines in each simulation) to preserve correlation.

### 6.2 Implementation Challenges

1.  **The "Zero" Problem:**
    *   If a cell is 0, the residual is 0.
    *   If the triangle is sparse, the bootstrap fails.
    *   **Solution:** Aggregate data (Quarterly -> Annual) or use a different model.

---

## 7. Advanced Topics & Extensions

### 7.1 One-Year View vs. Ultimate View

*   **Ultimate View:** Uncertainty until the claim closes (Run-off).
*   **One-Year View (Solvency II):** Uncertainty over the next 12 months.
*   **CDR (Claims Development Result):** The change in best estimate over 1 year.
*   *Calculation:* Requires "Re-Reserving" inside the simulation loop (Actuary-in-the-Box).

### 7.2 Bayesian MCMC

*   Alternative to Bootstrap.
*   Allows expert judgment (Priors) on the LDFs.
*   *Tool:* PyMC or Stan.

---

## 8. Regulatory & Governance Considerations

### 8.1 Solvency II / SST

*   Requires the "Probability Distribution Forecast" (PDF) of the reserves.
*   **Risk Margin:** Cost of Capital $\times$ Sum of Discounted SCRs.
*   The Bootstrap model provides the inputs for this calculation.

---

## 9. Practical Example

### 9.1 Worked Example: The "Fan Chart"

**Scenario:**
*   CFO wants to know: "How bad could it get?"
*   **Actuary:** Produces a Fan Chart.
*   **Center:** Best Estimate.
*   **Inner Band (50%):** Likely range.
*   **Outer Band (90%):** Possible range.
*   **Result:** CFO sees that the reserve is volatile. Decides to hold \$10M extra capital (Management Margin).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **ODP Bootstrap** is the industry standard for variability.
2.  **VaR/TVaR** quantify the tail risk.
3.  **Process + Parameter Variance** = Total Prediction Error.

### 10.2 When to Use This Knowledge
*   **Capital Modeling:** Feeding the Internal Model.
*   **Risk Margins:** Calculating the "Market Value Margin".

### 10.3 Critical Success Factors
1.  **Check Residuals:** If the model doesn't fit, the bootstrap is garbage.
2.  **Simulations:** Run enough to stabilize the tail.
3.  **Correlation:** Don't ignore dependencies between lines of business.

### 10.4 Further Reading
*   **England & Verrall:** "Stochastic Claims Reserving in General Insurance".
*   **Shapland:** "Using the Bootstrap Method for Reserving".

---

## Appendix

### A. Glossary
*   **ODP:** Over-Dispersed Poisson.
*   **VaR:** Value at Risk.
*   **TVaR:** Tail Value at Risk.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Pearson Residual** | $(C-m)/\sqrt{m}$ | Bootstrap Input |
| **TVaR** | $E[X \mid X > VaR]$ | Risk Measure |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
