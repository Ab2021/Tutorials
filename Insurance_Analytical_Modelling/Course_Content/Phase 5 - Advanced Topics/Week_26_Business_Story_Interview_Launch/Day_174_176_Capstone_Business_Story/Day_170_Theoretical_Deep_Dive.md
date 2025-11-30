# SOA/CAS-aligned Problem Sets (Part 3) - Advanced Topics & Simulations - Theoretical Deep Dive

## Overview
"Closed-form solutions are for textbooks. Simulations are for the real world."
The modern actuarial exam (like CAS Exam 9 or SOA Exam PA) is moving away from formulas and towards **Stochastic Modeling**.
This day focuses on **Monte Carlo Simulation**: The universal hammer for solving complex actuarial problems where the math is too hard to do by hand.

---

## 1. Conceptual Foundation

### 1.1 The Aggregate Loss Model

$$ S = \sum_{i=1}^{N} X_i $$
*   $N$: Frequency (Random Variable, e.g., Poisson).
*   $X$: Severity (Random Variable, e.g., Lognormal).
*   **Challenge:** The distribution of $S$ is a "Convolution". It has no simple formula.
*   **Solution:** Simulate 10,000 years.

### 1.2 Ruin Theory

*   **Question:** "What is the probability that the insurer goes bankrupt in the next 10 years?"
*   **Process:** Surplus Process $U(t) = U(0) + P \times t - S(t)$.
*   **Simulation:** Track the bank account daily. If it hits 0, Game Over.

---

## 2. Mathematical Framework

### 2.1 Inverse Transform Sampling

How do we generate random numbers from *any* distribution?
1.  Generate $u \sim \text{Uniform}(0, 1)$.
2.  Calculate $x = F^{-1}(u)$, where $F^{-1}$ is the inverse CDF (Quantile Function).
*   **Python:** `scipy.stats.lognorm.ppf(u, s)`.

### 2.2 Reinsurance Optimization

*   **Goal:** Minimize $\text{Var}(Net Loss)$ subject to $\text{Cost} < \text{Budget}$.
*   **Method:**
    1.  Simulate Gross Losses.
    2.  Apply Reinsurance Structure (e.g., 5M xs 5M).
    3.  Calculate Net Losses.
    4.  Calculate Metric (VaR, TVaR).

---

## 3. Theoretical Properties

### 3.1 Convergence Rate

*   **Law of Large Numbers:** As $n \to \infty$, the sample mean converges to the true mean.
*   **Standard Error:** Error decreases at a rate of $1/\sqrt{n}$.
    *   To cut the error in half, you need 4x the simulations.

### 3.2 Variance Reduction Techniques

*   **Antithetic Variates:** If you simulate $u$, also simulate $1-u$. They are negatively correlated, reducing the variance of the average.
*   **Importance Sampling:** Simulate more samples from the "tail" (where the disasters happen) and weight them down.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Problem 1: Aggregate Loss Simulation (Frequency-Severity)

**Problem:** Frequency $\sim \text{Poisson}(100)$. Severity $\sim \text{Lognormal}(\mu=10, \sigma=2)$. Calculate the 99.5% VaR of Aggregate Loss.

```python
import numpy as np
import matplotlib.pyplot as plt

n_sims = 10000
aggregate_losses = []

for _ in range(n_sims):
    # 1. Simulate Frequency
    n_claims = np.random.poisson(100)
    
    # 2. Simulate Severity
    if n_claims > 0:
        severities = np.random.lognormal(mean=10, sigma=2, size=n_claims)
        total_loss = np.sum(severities)
    else:
        total_loss = 0
        
    aggregate_losses.append(total_loss)

# 3. Calculate VaR
var_995 = np.percentile(aggregate_losses, 99.5)
print(f"99.5% VaR: ${var_995:,.0f}")

# Visualization
plt.hist(aggregate_losses, bins=50, density=True, alpha=0.7)
plt.axvline(var_995, color='r', linestyle='--', label='99.5% VaR')
plt.title("Aggregate Loss Distribution")
plt.legend()
```

### 4.2 Problem 2: Ruin Probability Simulation

**Problem:** Initial Surplus = \$1M. Premium = \$1.2M/year. Claims $\sim \text{Compound Poisson}$. Time = 10 years.

```python
initial_surplus = 1000000
premium_rate = 1200000
years = 10
n_sims = 1000

ruin_count = 0

for _ in range(n_sims):
    surplus = initial_surplus
    t = 0
    while t < years:
        # Time to next claim (Exponential)
        dt = np.random.exponential(1/100) # Lambda=100 claims/year
        t += dt
        
        if t > years: break
        
        # Collect Premium
        surplus += premium_rate * dt
        
        # Pay Claim
        claim = np.random.lognormal(9, 1) # Mean ~ $13k
        surplus -= claim
        
        # Check Ruin
        if surplus < 0:
            ruin_count += 1
            break

prob_ruin = ruin_count / n_sims
print(f"Probability of Ruin: {prob_ruin:.1%}")
```

---

## 5. Evaluation & Validation

### 5.1 Panjer Recursion (The Check)

*   **Method:** A numerical algorithm to calculate the *exact* distribution of $S$ (for discrete distributions).
*   **Validation:** Compare your Monte Carlo result to the Panjer result. They should match.

### 5.2 Seed Management

*   **Requirement:** Reproducibility.
*   **Code:** `np.random.seed(42)`.
*   **Audit:** If the auditor runs your code, they must get the exact same \$ value.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 The "Memory" Problem

*   **Issue:** Storing 10 Million individual claim amounts in RAM.
*   **Fix:** Don't store the severities. Just sum them up immediately (`total_loss += claim`) and discard.

### 6.2 Correlation

*   **Naive:** Simulating Line A and Line B independently.
*   **Reality:** A hurricane hits both Property and Auto.
*   **Fix:** Use a Copula (e.g., Gaussian Copula) to generate correlated random numbers.
    *   `mvnorm = np.random.multivariate_normal(...)`
    *   `u = norm.cdf(mvnorm)`

---

## 7. Advanced Topics & Extensions

### 7.1 Dynamic Hedging (Variable Annuities)

*   **Problem:** You sold a "Put Option" embedded in an insurance policy.
*   **Simulation:**
    *   Simulate stock paths (Geometric Brownian Motion).
    *   At each step, calculate Delta.
    *   Rebalance the hedge portfolio.
    *   Calculate "Hedge Error".

### 7.2 Catastrophe Modeling

*   **Structure:** Event Table (Hurricane Andrew, Earthquake 1906).
*   **Simulation:**
    *   Draw an Event ID from the table (Poisson).
    *   Apply the Event Footprint to your Exposure.
    *   Apply Deductibles/Limits.

---

## 8. Regulatory & Governance Considerations

### 8.1 Solvency II Internal Models

*   **Requirement:** If you use a simulation model for Capital Requirements, it must pass the "Use Test".
*   **Meaning:** You must use the model for actual business decisions, not just for the regulator.

---

## 9. Practical Example

### 9.1 The "Reinsurance Tower" Optimization

**Scenario:** You have \$10M budget.
**Options:**
A. Buy \$5M xs \$5M layer.
B. Buy \$10M xs \$10M layer.
C. Buy 50% Quota Share.
**Task:** Which option minimizes the probability of losing > \$20M?
**Python:**
1.  Simulate 10,000 Gross Loss years.
2.  Apply Structure A, B, C to each year.
3.  Count how many Net Loss years > \$20M.
4.  Pick the winner.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Monte Carlo** solves the unsolvable.
2.  **Convergence** requires $N$ to be large.
3.  **Correlation** is critical for multi-line modeling.

### 10.2 When to Use This Knowledge
*   **Capital Modeling:** Determining Economic Capital.
*   **Reinsurance:** Structuring complex treaties.

### 10.3 Critical Success Factors
1.  **Speed:** Use `numpy` vectorization, not loops.
2.  **Sanity Checks:** Does the average simulated loss match the theoretical mean?

### 10.4 Further Reading
*   **Klugman, Panjer, Willmot:** "Loss Models" (Simulation Chapter).

---

## Appendix

### A. Glossary
*   **VaR:** Value at Risk (Percentile).
*   **TVaR:** Tail Value at Risk (Average of losses > VaR).
*   **Copula:** A function that links marginal distributions to a joint distribution.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Standard Error** | $\sigma / \sqrt{n}$ | Simulation Accuracy |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
