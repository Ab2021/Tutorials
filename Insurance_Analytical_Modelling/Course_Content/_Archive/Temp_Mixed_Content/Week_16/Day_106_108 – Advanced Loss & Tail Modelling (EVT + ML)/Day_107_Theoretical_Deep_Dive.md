# Advanced Loss & Tail Modelling (Part 2) - Peaks Over Threshold (POT) - Theoretical Deep Dive

## Overview
Yesterday, we threw away 99% of our data by only keeping the "Annual Maximum". Today, we use **Peaks Over Threshold (POT)** to keep *every* large loss. This is the industry standard for modern capital modeling.

---

## 1. Conceptual Foundation

### 1.1 Block Maxima vs. POT

*   **Block Maxima:** "What is the worst storm of the year?" (Ignores the 2nd worst storm, which might still be huge).
*   **POT:** "What are all the storms that caused > \$10M in damage?" (Captures all extreme events).
*   **Efficiency:** POT uses data much more efficiently, leading to lower variance in VaR estimates.

### 1.2 Pickands-Balkema-De Haan Theorem

*   This is the "Central Limit Theorem" for thresholds.
*   It states that for a sufficiently high threshold $u$, the distribution of excesses ($X - u$) converges to the **Generalized Pareto Distribution (GPD)**.

---

## 2. Mathematical Framework

### 2.1 The Generalized Pareto Distribution (GPD)

$$ H(y) = 1 - \left( 1 + \frac{\xi y}{\sigma} \right)^{-1/\xi} $$

*   $y = x - u$: The excess amount over the threshold.
*   $\sigma$: Scale parameter.
*   $\xi$: Shape parameter (Tail Index).
    *   $\xi > 0$: Heavy Tail (Pareto-like). Most insurance risks.
    *   $\xi = 0$: Exponential Tail.
    *   $\xi < 0$: Finite Tail (Beta-like).

### 2.2 Tail Value at Risk (TVaR)

*   Also known as Expected Shortfall (ES).
*   **Formula (for GPD):**
    $$ \text{TVaR}_p = \text{VaR}_p + \frac{\sigma + \xi(\text{VaR}_p - u)}{1 - \xi} $$
*   *Insight:* TVaR depends heavily on $\xi$. If $\xi \ge 1$, the mean is infinite, and insurance is impossible.

---

## 3. Theoretical Properties

### 3.1 Threshold Selection

*   **The Goldilocks Problem:**
    *   **Too Low:** Bias. The data isn't "extreme" enough, so GPD doesn't fit.
    *   **Too High:** Variance. Too few data points to estimate parameters.
*   **Mean Residual Life (MRL) Plot:**
    *   Plot Mean Excess ($E[X-u|X>u]$) vs. Threshold $u$.
    *   Look for the region where the plot is **linear**.

### 3.2 Declustering

*   **Assumption:** GPD assumes I.I.D. data.
*   **Reality:** A hurricane causes 50 claims in 2 days. They are dependent.
*   **Fix:** "Runs Method". Define a cluster (e.g., events within 3 days). Keep only the maximum of the cluster.

---

## 4. Modeling Artifacts & Implementation

### 4.1 SciPy Implementation

```python
import numpy as np
from scipy.stats import genpareto
import matplotlib.pyplot as plt

# 1. Data (All losses)
losses = np.random.pareto(a=2, size=1000) * 1000

# 2. Select Threshold (e.g., 95th percentile)
u = np.percentile(losses, 95)
excesses = losses[losses > u] - u

# 3. Fit GPD
# shape (c) = xi, scale = sigma
c, loc, scale = genpareto.fit(excesses, floc=0)

# 4. Calculate 1-in-200 Year VaR
# Probability relative to the threshold
# If we have 1000 points and 50 exceedances, P(Exceed) = 0.05
# We want 1-in-200 (0.5% probability total)
# Conditional prob needed = 0.005 / 0.05 = 0.1
p_cond = 0.1
var_excess = genpareto.ppf(1 - p_cond, c, loc=0, scale=scale)
var_total = u + var_excess

print(f"Threshold: {u:.2f}")
print(f"VaR (1-in-200): {var_total:.2f}")
```

### 4.2 PyExtremes

*   A dedicated Python library for EVT.
*   Automates MRL plots and Declustering.

---

## 5. Evaluation & Validation

### 5.1 Parameter Stability Plot

*   Fit GPD across a range of thresholds.
*   Plot $\xi$ vs. $u$.
*   $\xi$ should be constant (stable) in the region where the GPD is valid.

### 5.2 Return Level Plot

*   Similar to Block Maxima, but adapted for POT.
*   Checks if the model extrapolates reasonably to the 1000-year return period.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The Infinite Mean**
    *   If $\xi > 1$, the theoretical mean is infinite.
    *   *Consequence:* You cannot price a "Unlimited Reinstatement" treaty. The price is infinity.
    *   *Reality Check:* Is the tail really that heavy, or is it an outlier?

2.  **Trap: Seasonality**
    *   Threshold exceedances might only happen in Summer (Hurricanes).
    *   *Fix:* Non-stationary Poisson Process for the *rate* of exceedances.

### 6.2 Implementation Challenges

1.  **Reporting:**
    *   Management understands "Normal Distribution". They don't understand "Xi = 0.8".
    *   *Viz:* Show the "Tail Plot" (Log-Log scale) to demonstrate how GPD captures the risk better than Normal.

---

## 7. Advanced Topics & Extensions

### 7.1 Composite Models (Splicing)

*   **Body:** Lognormal (for losses < \$1M).
*   **Tail:** GPD (for losses > \$1M).
*   **Splicing:** Ensure the PDF is continuous at the splicing point.

### 7.2 Bayesian GPD

*   Use PyMC3 to fit GPD.
*   Allows for priors on $\xi$ (e.g., "Tail index is likely between 0.1 and 0.5").
*   Useful for small datasets.

---

## 8. Regulatory & Governance Considerations

### 8.1 Capital Requirements

*   **Solvency II:** Requires 99.5% VaR over 1 year.
*   **Swiss Solvency Test (SST):** Requires 99% TVaR (Expected Shortfall).
*   GPD is the standard tool for calculating these metrics for Catastrophe Risk.

---

## 9. Practical Example

### 9.1 Worked Example: Cyber Liability

**Scenario:**
*   Cyber claims are rare but huge (Ransomware).
*   **Data:** 500 claims over 5 years.
*   **Analysis:**
    *   MRL Plot shows linearity above \$5M. Set $u = 5M$.
    *   Fit GPD: $\xi = 0.7$ (Very heavy tail).
*   **Capital Calculation:**
    *   99.5% VaR = \$150M.
    *   If we assumed Lognormal, VaR would be \$50M.
    *   *Conclusion:* Lognormal massively understates Cyber Risk.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **POT** uses more data than Block Maxima.
2.  **GPD** models the excess over a threshold.
3.  **Threshold Selection** is an art and a science.

### 10.2 When to Use This Knowledge
*   **Reinsurance Pricing:** Pricing layers like "\$10M xs \$10M".
*   **Capital Modeling:** Determining the "Worst Case Scenario".

### 10.3 Critical Success Factors
1.  **Declustering:** Don't count the same storm twice.
2.  **Stability:** Check if your VaR changes wildly if you move the threshold slightly.

### 10.4 Further Reading
*   **McNeil, Frey, Embrechts:** "Quantitative Risk Management".

---

## Appendix

### A. Glossary
*   **Exceedance:** A data point greater than the threshold.
*   **MRL:** Mean Residual Life.
*   **Splicing:** Joining two distributions.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **GPD Survival** | $S(y) = (1 + \xi y / \sigma)^{-1/\xi}$ | Probability |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
