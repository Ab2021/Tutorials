# Capital & Solvency Intro Case Study (Part 3) - Theoretical Deep Dive

## Overview
The Standard Formula is a "One Size Fits All" approach. For large insurers, it often fails to capture specific risks (e.g., Cyber, Pandemic). This session covers **Internal Models**, the use of **Copulas** for dependency modeling, and **Monte Carlo Simulation** for capital aggregation.

---

## 1. Conceptual Foundation

### 1.1 Why Build an Internal Model?

*   **Risk Profile:** If you write "Satellite Insurance", the Standard Formula has no module for it.
*   **Capital Efficiency:** Standard Formula assumes correlations are fixed (e.g., 0.25). Real correlations might be lower (Diversification).
*   **Risk Management:** It forces the business to understand its own risks, rather than just ticking boxes.

### 1.2 The Components

1.  **Marginal Distributions:** Modeling each risk separately (e.g., Earthquake Severity ~ Pareto).
2.  **Dependency Structure:** How do risks move together? (Copulas).
3.  **Aggregation:** Combining them to get the Total Loss Distribution.
4.  **Risk Measure:** Calculating VaR 99.5% or TVaR 99.0%.

### 1.3 Copulas

*   **Problem:** Linear correlation (Pearson) only works for Normal distributions. Insurance risks are skewed (Fat Tails).
*   **Solution:** A Copula links marginal distributions to a joint distribution.
*   **Sklar's Theorem:** Any joint distribution can be written as a function of its marginals and a copula.
*   **Types:**
    *   **Gaussian Copula:** Tail independence. (Bad for financial crashes).
    *   **t-Copula:** Tail dependence. (Good for modeling "When it rains, it pours").
    *   **Clayton/Gumbel:** Asymmetric dependence.

---

## 2. Mathematical Framework

### 2.1 Monte Carlo Aggregation

$$ L_{total} = \sum_{i=1}^{N} L_i $$
*   We cannot sum the VaRs: $VaR(A+B) \neq VaR(A) + VaR(B)$. (Unless perfectly correlated).
*   **Algorithm:**
    1.  Generate correlated random numbers $U_1, U_2, ..., U_n$ using a Copula.
    2.  Transform $U_i$ to Loss $L_i$ using the Inverse CDF: $L_i = F_i^{-1}(U_i)$.
    3.  Sum the losses: $L_{total} = \sum L_i$.
    4.  Repeat 100,000 times.
    5.  Take the 99.5th percentile.

### 2.2 Capital Allocation

*   Once we have Total Capital (\$1B), how much belongs to the "Auto" line vs. "Property"?
*   **Euler Allocation:**
    $$ K_i = \frac{\partial VaR}{\partial L_i} \times L_i $$
    *   *Property:* The sum of allocated capital equals the total capital.
    *   *Meaning:* Capital is allocated based on the *marginal contribution* to the tail risk.

---

## 3. Theoretical Properties

### 3.1 Tail Dependence

*   **Definition:** $\lim_{u \to 1} P(X > u | Y > u)$.
*   **Gaussian Copula:** Limit is 0. (Independence in the tail).
*   **t-Copula:** Limit > 0. (Crash together).
*   **Reality:** In 2008, Credit and Equity crashed together. In 2020, Mortality and Assets crashed together. Tail dependence is real.

### 3.2 Model Stability

*   **Monte Carlo Error:** The VaR estimate fluctuates with the seed.
*   **Convergence:** $Error \propto \frac{1}{\sqrt{N}}$. To halve the error, you need $4x$ simulations.
*   **Variance Reduction:** Techniques like "Latin Hypercube Sampling" or "Importance Sampling" speed up convergence.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Internal Model in Python

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, t

# 1. Define Marginals
# Risk A: Normal (e.g., Expense Risk)
# Risk B: LogNormal (e.g., Catastrophe Risk)
mu_a, std_a = 100, 10
mu_b, sigma_b = 4, 1.0 # LogNormal parameters

# 2. Define Dependency (Gaussian Copula)
rho = 0.5
cov_matrix = [[1, rho], [rho, 1]]
n_sims = 100000

# 3. Generate Correlated Uniforms
# Step A: Generate Multivariate Normal
z = np.random.multivariate_normal([0, 0], cov_matrix, n_sims)
# Step B: Transform to Uniform (CDF)
u = norm.cdf(z)

# 4. Transform to Marginals (Inverse CDF)
loss_a = norm.ppf(u[:, 0], loc=mu_a, scale=std_a)
loss_b = lognorm.ppf(u[:, 1], s=sigma_b, scale=np.exp(mu_b))

# 5. Aggregate
total_loss = loss_a + loss_b

# 6. Calculate SCR (VaR 99.5%)
mean_loss = np.mean(total_loss)
var_995 = np.percentile(total_loss, 99.5)
scr = var_995 - mean_loss

print(f"Mean Loss: {mean_loss:,.0f}")
print(f"VaR 99.5%: {var_995:,.0f}")
print(f"SCR: {scr:,.0f}")

# Visualization
plt.hist(total_loss, bins=100, density=True, alpha=0.6, color='b')
plt.axvline(var_995, color='r', linestyle='--', label='VaR 99.5%')
plt.title("Total Loss Distribution (Internal Model)")
plt.legend()
plt.show()
```

### 4.2 Euler Allocation Script

```python
# Euler Allocation (Kernel Estimation approach)
# Allocate Capital to Risk A and Risk B
# Condition: Total Loss is close to VaR
epsilon = 0.01 * var_995
tail_events = (total_loss > var_995 - epsilon) & (total_loss < var_995 + epsilon)

# Conditional Expectation
alloc_a = np.mean(loss_a[tail_events])
alloc_b = np.mean(loss_b[tail_events])

print(f"Allocated Capital A: {alloc_a:.0f}")
print(f"Allocated Capital B: {alloc_b:.0f}")
print(f"Sum: {alloc_a + alloc_b:.0f} (Check vs VaR: {var_995:.0f})")
```

---

## 5. Evaluation & Validation

### 5.1 Backtesting

*   Compare the 1-in-200 year model prediction to actual history.
*   *Problem:* We don't have 200 years of data.
*   **Solution:** Backtest the "1-in-10" year components (e.g., Attritional Claims).

### 5.2 Sensitivity Testing

*   **Parameter Uncertainty:** What if correlation is 0.7 instead of 0.5?
*   **Copula Choice:** What if we use a t-Copula?
*   **Impact:** Often, the Copula choice drives the capital more than the marginals.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Diversification Trap"**
    *   **Issue:** Relying on diversification to lower capital.
    *   **Reality:** In a crisis, correlations go to 1. Diversification disappears when you need it most.
    *   **Fix:** Stress test with high correlations.

2.  **Trap: Granularity**
    *   **Issue:** Modeling "All Property" as one line.
    *   **Reality:** Residential and Commercial Property behave differently.
    *   **Fix:** Split the lines, then aggregate.

### 6.2 Implementation Challenges

1.  **Runtime:**
    *   Running 1 million sims with complex re-valution logic (nested stochastic) takes days.
    *   **Solution:** Proxy Models (Polynomials) or LSMC (Least Squares Monte Carlo).

---

## 7. Advanced Topics & Extensions

### 7.1 Nested Stochastic Modeling

*   To calculate SCR at Time 1, we need the Value of Liabilities at Time 1.
*   But Liabilities at Time 1 depend on future cash flows (Time 2...50).
*   **Result:** Simulation inside a Simulation. (Computationally impossible).
*   **Technique:** Replicating Portfolios.

### 7.2 Climate Change Modeling

*   How to incorporate "Physical Risk" (more hurricanes) into the Internal Model?
*   **Method:** Adjust the parameters of the Catastrophe LogNormal distribution based on climate science (IPCC scenarios).

---

## 8. Regulatory & Governance Considerations

### 8.1 Model Change Policy

*   You cannot just change the model to lower capital.
*   **Major Change:** Requires Regulator Approval (6 month process).
*   **Minor Change:** Internal governance only.

---

## 9. Practical Example

### 9.1 Worked Example: The "t-Copula" Effect

**Scenario:**
*   Portfolio: Equities and Corporate Bonds.
*   **Gaussian Copula:** SCR = \$100M. (Assumes they don't crash together often).
*   **t-Copula (df=4):** SCR = \$130M. (Assumes "Fat Tails").
*   **Decision:** The CRO chooses the t-Copula to be prudent, even though it costs more capital.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Monte Carlo** allows for flexible aggregation.
2.  **Copulas** define the dependency structure.
3.  **Euler Allocation** splits the bill fairly.

### 10.2 When to Use This Knowledge
*   **Capital Optimization:** Buying Reinsurance to lower the SCR.
*   **Strategic Asset Allocation:** Deciding how much Equity to hold.

### 10.3 Critical Success Factors
1.  **Random Number Generation:** Use a high-quality generator (Mersenne Twister).
2.  **Validation:** Challenge every assumption.
3.  **Documentation:** If it's not written down, it doesn't exist (for the regulator).

### 10.4 Further Reading
*   **McNeil, Frey, Embrechts:** "Quantitative Risk Management".
*   **Wang:** "Aggregation of Correlated Risk Portfolios".

---

## Appendix

### A. Glossary
*   **Marginal:** The distribution of a single risk.
*   **Joint:** The distribution of all risks together.
*   **Seed:** The starting point for the random number generator.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Euler Allocation** | $E[L_i \mid L = VaR]$ | Capital Allocation |
| **Gaussian Copula** | $\Phi_\Sigma(\Phi^{-1}(u_1), ...)$ | Dependency |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
