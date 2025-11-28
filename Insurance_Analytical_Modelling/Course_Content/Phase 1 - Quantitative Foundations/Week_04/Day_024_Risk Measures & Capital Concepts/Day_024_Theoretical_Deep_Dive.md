# Risk Measures & Capital Concepts - Theoretical Deep Dive

## Overview
This session covers the quantitative frameworks used to measure risk and determine the capital required to support it. We move beyond simple variance to advanced tail metrics like Value-at-Risk (VaR) and Tail Value-at-Risk (TVaR/CTE). We also explore the concept of Coherent Risk Measures, Economic Capital modeling, and regulatory frameworks like Solvency II and RBC.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Risk Measure:** A function $\rho(X)$ that maps a random loss variable $X$ to a single number (capital amount) representing the riskiness of $X$.

**Economic Capital (EC):** The amount of capital a firm *needs* to hold to ensure solvency over a specified time horizon with a specified confidence level (e.g., 99.5% over 1 year).

**Regulatory Capital:** The amount of capital a firm is *required* to hold by law (e.g., Solvency II SCR, NAIC RBC).

**Key Terminology:**
*   **Solvency:** The ability to meet long-term financial obligations.
*   **Tail Risk:** The risk of extreme events (far right of the loss distribution).
*   **Coherence:** A set of mathematical properties that a "good" risk measure should satisfy.
*   **Diversification Benefit:** The reduction in total capital when combining imperfectly correlated risks.

### 1.2 Historical Context & Evolution

**Origin:**
*   **Variance:** Early finance (Markowitz) used Standard Deviation as the risk measure.
*   **VaR (1990s):** JP Morgan popularized Value-at-Risk (RiskMetrics) for banking trading books.

**Evolution:**
*   **Incoherence of VaR:** Artzner et al. (1999) proved VaR is not "coherent" (it discourages diversification in some cases).
*   **TVaR/ES:** Actuaries and regulators moved towards Tail Value-at-Risk (Expected Shortfall) to capture the shape of the tail.
*   **Solvency II (2016):** EU adopted a 1-in-200 year VaR standard for insurance capital.

**Current State:**
*   **Internal Models:** Large insurers build stochastic models (100k simulations) to calculate EC, rather than using standard formulas.
*   **SST (Swiss Solvency Test):** Uses TVaR, considered more robust than Solvency II's VaR.

### 1.3 Why This Matters

**Business Impact:**
*   **Capital Allocation:** Which business line is "expensive" in terms of capital? (e.g., Catastrophe insurance requires more capital per dollar of premium than Auto insurance).
*   **RoEC (Return on Economic Capital):** The primary metric for pricing and performance measurement.

**Regulatory Relevance:**
*   **Intervention:** If Capital < Required Capital, regulators take control of the company.
*   **ORSA:** Own Risk and Solvency Assessment (insurers must self-assess their capital needs).

---

## 2. Mathematical Framework

### 2.1 Value-at-Risk (VaR)

**Definition:** The quantile of the loss distribution.
$$ \text{VaR}_p(X) = \inf \{ x \in \mathbb{R} : P(X \le x) \ge p \} $$
*   *Interpretation:* "We are $p\%$ confident that the loss will not exceed $\text{VaR}_p$."
*   *Example:* $\text{VaR}_{99.5\%} = \$100M$ means there is a 0.5% chance of losing more than $100M.

**Pros:** Simple, widely understood.
**Cons:** Ignores the magnitude of the loss *beyond* the threshold. (Is the bad scenario \$101M or \$10B?). Not subadditive.

### 2.2 Tail Value-at-Risk (TVaR) / CTE

**Definition:** The expected loss *given* that the loss exceeds the VaR.
$$ \text{TVaR}_p(X) = E[X | X > \text{VaR}_p(X)] $$
*   *Also known as:* Conditional Tail Expectation (CTE), Expected Shortfall (ES).

**Pros:** Coherent (subadditive), captures tail severity.
**Cons:** Harder to estimate (requires more data in the tail).

### 2.3 Coherent Risk Measures (Artzner Axioms)

A risk measure $\rho$ is coherent if it satisfies:
1.  **Monotonicity:** If $X \le Y$ always, then $\rho(X) \le \rho(Y)$.
2.  **Subadditivity:** $\rho(X + Y) \le \rho(X) + \rho(Y)$. (Diversification works).
3.  **Positive Homogeneity:** $\rho(c X) = c \rho(X)$ for $c > 0$.
4.  **Translation Invariance:** $\rho(X + c) = \rho(X) + c$. (Adding cash reduces risk 1-for-1).

*   **VaR Fails Subadditivity:** Combining two heavy-tailed risks can result in a VaR *higher* than the sum of individual VaRs.
*   **TVaR Satisfies All 4.**

### 2.4 Aggregation & Copulas

How to combine risks ($X, Y$) to find $\rho(X+Y)$?
*   **Correlation Matrix:** Assumes linear dependence (Gaussian). Underestimates tail dependence.
*   **Copulas:** Functions that join marginal distributions into a joint distribution.
    $$ F_{XY}(x, y) = C(F_X(x), F_Y(y)) $$
    *   **t-Copula:** Has tail dependence (if X crashes, Y is likely to crash too). Crucial for modeling financial crises.

---

## 3. Theoretical Properties

### 3.1 Normal Distribution Special Case

If $X \sim N(\mu, \sigma^2)$:
*   $\text{VaR}_p = \mu + \sigma \Phi^{-1}(p)$.
*   $\text{TVaR}_p = \mu + \sigma \frac{\phi(\Phi^{-1}(p))}{1-p}$.
*   *Note:* For Normal distributions, VaR *is* subadditive. The incoherence only arises with heavy tails.

### 3.2 Spectral Risk Measures

A generalization where the risk measure is a weighted average of quantiles.
$$ \rho(X) = \int_0^1 \text{VaR}_u(X) \phi(u) du $$
*   Allows the user to assign "Risk Aversion" weights to different parts of the tail.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

*   **Loss Distributions:** Fitted curves for each risk (Market, Credit, Insurance).
*   **Correlation Matrix:** Or Copula parameters describing dependencies.

### 4.2 Preprocessing Steps

**Step 1: Simulation (Monte Carlo)**
*   Simulate 100,000 years of losses for each risk type.
*   Use Copulas to induce correlation in the random draws.

**Step 2: Aggregation**
*   Sum the losses for each simulation year to get Total Loss.

### 4.3 Model Specification (Python Example)

Calculating VaR and TVaR for a portfolio of two risks using a Gaussian Copula.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simulation Parameters
n_sims = 100000
np.random.seed(42)

# Risk 1: Auto Insurance (Lognormal)
mu1, sigma1 = 10, 0.5

# Risk 2: Catastrophe (Pareto)
# Pareto(alpha, scale)
alpha2, scale2 = 2.5, 50000

# Dependence: Gaussian Copula with rho = 0.5
rho = 0.5
cov_matrix = [[1, rho], [rho, 1]]
# Generate correlated normal variables
Z = np.random.multivariate_normal([0, 0], cov_matrix, n_sims)
# Transform to Uniform [0,1]
U = norm.cdf(Z)

# Transform Uniforms to Marginals (Inverse CDF method)
Loss1 = np.exp(mu1 + sigma1 * norm.ppf(U[:, 0])) # Lognormal
# Pareto Inverse CDF: scale * ((1-u)^(-1/alpha) - 1)
Loss2 = scale2 * ((1 - U[:, 1])**(-1/alpha2) - 1)

TotalLoss = Loss1 + Loss2

# Calculate Risk Measures (99.5%)
p = 0.995

# 1. VaR
var_995 = np.percentile(TotalLoss, p * 100)

# 2. TVaR (Average of losses > VaR)
tvar_995 = TotalLoss[TotalLoss > var_995].mean()

print(f"VaR (99.5%): ${var_995:,.0f}")
print(f"TVaR (99.5%): ${tvar_995:,.0f}")

# Check Subadditivity (Diversification)
var1 = np.percentile(Loss1, p * 100)
var2 = np.percentile(Loss2, p * 100)
sum_var = var1 + var2

print(f"\nVaR(Risk1): ${var1:,.0f}")
print(f"VaR(Risk2): ${var2:,.0f}")
print(f"Sum of VaRs: ${sum_var:,.0f}")
print(f"Diversification Benefit: ${sum_var - var_995:,.0f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.hist(TotalLoss, bins=100, range=(0, var_995*1.5), alpha=0.7, label='Total Loss')
plt.axvline(var_995, color='r', linestyle='--', label='VaR 99.5%')
plt.axvline(tvar_995, color='k', linestyle='-', label='TVaR 99.5%')
plt.title('Total Loss Distribution with Risk Measures')
plt.legend()
plt.show()
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Required Capital:** Usually set to VaR(99.5%) - Mean Loss.
2.  **Tail Ratio:** TVaR / VaR. Indicates how heavy the tail is.

**Interpretation:**
*   **High Diversification:** Means the risks are uncorrelated or the copula allows for independence.
*   **TVaR >> VaR:** Indicates a "fat tail" (e.g., Catastrophe risk). The bad scenario is *really* bad.

---

## 5. Evaluation & Validation

### 5.1 Backtesting

*   Compare calculated VaR to actual realized losses over time.
*   **Kupiec Test:** If we use 99% VaR, we expect 1 breach every 100 days. If we see 5 breaches in 100 days, the model is wrong.

### 5.2 Stress Testing

*   "What if correlation goes to 1.0?"
*   "What if the Pareto alpha drops from 2.5 to 1.5?"
*   Essential for understanding model sensitivity.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Relying on VaR for Tail Risk**
    *   **Issue:** Two portfolios can have the same VaR but vastly different extreme losses.
    *   **Fix:** Always look at TVaR or stress tests alongside VaR.

2.  **Trap: Gaussian Copula in Crisis**
    *   **Issue:** Gaussian copula assumes tail independence. In a crisis, "correlations go to 1."
    *   **Result:** Underestimation of joint extreme events (2008 Financial Crisis).

### 6.2 Implementation Challenges

1.  **Data Scarcity:** Estimating the 99.9th percentile requires extrapolating far beyond historical data.
2.  **Aggregation Levels:** Summing VaRs from business units is conservative (ignores diversification) but simple. Using a full correlation matrix is accurate but complex.

---

## 7. Advanced Topics & Extensions

### 7.1 Allocation of Capital (Euler)

How to split the Total Capital back to individual business units?
*   **Euler Allocation:** $\text{Capital}_i = E[X_i | S = \text{VaR}(S)]$.
*   **Property:** The allocated capitals sum exactly to the total capital.
*   **Meaning:** It measures the marginal contribution of risk $i$ to the total tail risk.

### 7.2 Distortion Risk Measures

*   Apply a distortion function $g(S(x))$ to the survival probabilities.
*   Example: Dual Power Transform.
*   Used in pricing to generate risk-loaded premiums.

---

## 8. Regulatory & Governance Considerations

### 8.1 Solvency II (EU)

*   **Standard Formula:** Factor-based approach with correlation matrices.
*   **Internal Model:** Full stochastic simulation (requires regulatory approval).
*   **Metric:** 99.5% VaR over 1 year.

### 8.2 NAIC RBC (US)

*   **Formula:** $RBC = \sqrt{R_1^2 + R_2^2 + \dots}$.
*   **Covariance Adjustment:** The square root formula assumes independence between certain risk buckets (Asset vs. Underwriting).

---

## 9. Practical Example

### 9.1 Worked Example: Euler Allocation

**Scenario:**
*   Portfolio $S = X + Y$.
*   We simulate 10 scenarios.
*   Target: TVaR at 80% level (Average of worst 2 scenarios).

**Simulations:**
| Scen | X | Y | S | Rank |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 10 | 10 | 20 | |
| ... | ... | ... | ... | |
| 9 | 100 | 50 | 150 | 2 |
| 10 | 20 | 200 | 220 | 1 |

**Calculation:**
1.  **Total TVaR:** Average of S in worst 2 cases (Scen 9, 10).
    $$ \text{TVaR}(S) = (150 + 220) / 2 = 185 $$
2.  **Allocation to X:** Average of X in those *same* scenarios.
    $$ \text{Alloc}(X) = (100 + 20) / 2 = 60 $$
3.  **Allocation to Y:** Average of Y in those *same* scenarios.
    $$ \text{Alloc}(Y) = (50 + 200) / 2 = 125 $$
4.  **Check:** $60 + 125 = 185$.

**Interpretation:**
*   Even though X had a bad loss in Scen 9, Y was the main driver of the *very* worst case (Scen 10).
*   Y gets more capital allocated.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **VaR** is the threshold; **TVaR** is the average beyond the threshold.
2.  **Coherence** (Subadditivity) is vital for aggregation.
3.  **Capital** protects against insolvency.

### 10.2 When to Use This Knowledge
*   **Capital Modeling:** Determining solvency needs.
*   **Performance Measurement:** RAROC (Risk-Adjusted Return on Capital).
*   **Reinsurance:** Assessing the value of tail protection.

### 10.3 Critical Success Factors
1.  **Know the Tail:** The choice of distribution (Normal vs. Pareto) changes VaR drastically.
2.  **Respect Correlations:** Independence is a dangerous assumption.
3.  **Validate:** Backtest against history.

### 10.4 Further Reading
*   **McNeil, Frey, Embrechts:** "Quantitative Risk Management".
*   **Artzner et al.:** "Coherent Measures of Risk".

---

## Appendix

### A. Glossary
*   **VaR:** Value at Risk.
*   **CTE:** Conditional Tail Expectation (same as TVaR).
*   **SCR:** Solvency Capital Requirement.
*   **Copula:** Function linking marginals to joint distribution.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **VaR** | $F^{-1}(p)$ | Quantile |
| **TVaR** | $E[X \| X > \text{VaR}]$ | Tail Avg |
| **Subadditivity** | $\rho(X+Y) \le \rho(X) + \rho(Y)$ | Coherence |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,000+*
