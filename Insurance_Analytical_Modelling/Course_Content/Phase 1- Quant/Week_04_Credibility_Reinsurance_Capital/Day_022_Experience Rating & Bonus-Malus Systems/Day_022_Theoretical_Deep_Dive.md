# Experience Rating & Bonus-Malus Systems - Theoretical Deep Dive

## Overview
This session explores Experience Rating, the mechanism for adjusting premiums based on individual loss history, with a specific focus on Bonus-Malus Systems (BMS) used in automobile insurance. We model BMS using Markov Chains, analyze transition matrices, calculate stationary distributions, and discuss the "Hunger for Bonus" phenomenon where policyholders self-insure small losses to protect their discount.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Experience Rating:** A method of modifying a premium for a specific risk based on the past loss experience of that risk.
*   **Prospective:** Adjusts *future* premiums (e.g., NCD in auto).
*   **Retrospective:** Adjusts *past* premiums (e.g., Retro-rated Workers' Comp).

**Bonus-Malus System (BMS):** A specific type of experience rating common in personal lines (Auto/Home).
*   **Bonus:** Premium discount for claim-free years.
*   **Malus:** Premium surcharge for years with claims.
*   **Structure:** A set of classes (levels) with associated premium relativities. Movement between classes is determined by claim count.

**Key Terminology:**
*   **No-Claim Discount (NCD):** The percentage reduction from the base rate.
*   **Transition Rules:** The logic dictating movement (e.g., "Down 1 step for 0 claims, Up 2 steps for 1 claim").
*   **Hunger for Bonus:** The economic incentive for an insured to pay a small claim out-of-pocket to avoid a premium increase.

### 1.2 Historical Context & Evolution

**Origin:**
*   **Europe:** BMS has been highly regulated and standardized in countries like France and Belgium.
*   **USA:** "Safe Driver Plans" emerged in the mid-20th century.

**Evolution:**
*   **Rigid Systems:** Early systems had strict rules (e.g., "3 years clean = 10% off").
*   **Markov Modeling:** Actuaries began modeling the long-term stability of these systems using stochastic processes.
*   **Deregulation:** In many markets (e.g., EU), insurers now design their own bespoke BMS to compete for good drivers.

**Current State:**
*   **Claim Forgiveness:** Modern features that allow one "free" accident without losing the bonus (complicates the Markov model).
*   **Telematics Integration:** Blending traditional NCD (based on claims) with UBI (based on driving behavior).

### 1.3 Why This Matters

**Business Impact:**
*   **Retention:** High NCD levels act as "Golden Handcuffs." A customer with a 60% discount is less likely to switch insurers if the new insurer doesn't match it.
*   **Profitability:** Malus surcharges compensate for the higher risk of drivers who crash.
*   **Behavior Modification:** Incentivizes safer driving (and under-reporting of small claims).

**Regulatory Relevance:**
*   **Portability:** In many jurisdictions, insurers must honor the NCD earned with a previous insurer.
*   **Fairness:** Regulators monitor if the "Malus" is too punitive (e.g., doubling the premium for one fender bender).

---

## 2. Mathematical Framework

### 2.1 Markov Chain Representation

A BMS can be modeled as a discrete-time Markov Chain.
*   **States ($S$):** The NCD levels (e.g., Level 0 to Level 20).
*   **Time ($t$):** Policy years.
*   **Transition Probability ($p_{ij}$):** Probability of moving from Class $i$ to Class $j$ in one year.

**Assumptions:**
1.  **Markov Property:** Next year's class depends *only* on this year's class and this year's claims (not the entire history).
2.  **Stationarity:** Transition rules don't change over time.
3.  **Poisson Claims:** The number of claims $K$ follows a Poisson distribution with mean $\lambda$.

### 2.2 Transition Matrix ($P$)

Let $p_k$ be the probability of having $k$ claims in a year (e.g., $e^{-\lambda} \lambda^k / k!$).
The matrix entry $P_{ij}$ is the sum of probabilities of all claim counts that cause a move from $i$ to $j$.

*Example Rule:*
*   0 Claims: Move down 1 level (or stay at 0).
*   1 Claim: Move up 2 levels.
*   2+ Claims: Move up 4 levels.

$$ P = \begin{bmatrix} p_0 & 0 & p_1 & 0 & p_{2+} & \dots \\ p_0 & 0 & p_1 & 0 & p_{2+} & \dots \\ 0 & p_0 & 0 & p_1 & 0 & \dots \\ \vdots & \vdots & \vdots & \vdots & \vdots & \ddots \end{bmatrix} $$

### 2.3 Stationary Distribution ($\pi$)

The long-run proportion of policyholders in each class.
$$ \pi = \pi P $$
*   $\pi$ is the left eigenvector of $P$ corresponding to eigenvalue 1.
*   **Interpretation:** In the steady state, $\pi_i$ is the % of the portfolio in Class $i$.

**Average Premium Level:**
$$ \bar{b} = \sum_{i} \pi_i \times b_i $$
*   $b_i$: The premium relativity for Class $i$ (e.g., 0.60 for 40% NCD).
*   **Financial Balance:** If $\bar{b} < 1$, the system gives back more in bonuses than it collects in maluses. The base rate must be loaded to compensate.

### 2.4 Optimal Retention (Hunger for Bonus)

When should a policyholder pay a claim $C$ out of pocket?
*   **Cost of Claiming:** $C$.
*   **Cost of Not Claiming:** Present Value of future premium increases.
    $$ \Delta P = \sum_{t=1}^{\infty} v^t (P_{claim, t} - P_{no\_claim, t}) $$
*   **Decision Rule:** File claim if $C > \Delta P$.

**Optimal Retention Limit ($L$):** The threshold where $C = \Delta P$.
*   This acts as an *implied deductible*.
*   The insurer sees a "censored" claim frequency $\lambda^* = \lambda \times P(X > L)$.

---

## 3. Theoretical Properties

### 3.1 Toughness of the System

*   **Elasticity:** How quickly does the premium respond to a claim?
*   **Recovery Time:** How many claim-free years does it take to return to the original level after an accident?

### 3.2 Loimaranta Efficiency

A measure of how well the BMS estimates the true risk parameter $\lambda$.
$$ E = \frac{\text{Cov}(\lambda, b(\lambda))}{\text{Var}(b(\lambda))} $$
*   Ideally, the premium paid $b(\lambda)$ should be proportional to the true risk $\lambda$.

### 3.3 Asymptotic Stability

*   Does the system settle down?
*   Most BMS are ergodic (have a unique stationary distribution).
*   **Problem:** If the "Bonus" class is too easy to reach, everyone ends up there (mass at the bottom). The system loses its discriminatory power.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

*   **Transition Rules:** The exact logic (e.g., table mapping Current Class + Claim Count -> Next Class).
*   **Relativities:** The discount/surcharge % for each class.
*   **Claim Frequency:** Expected $\lambda$ for the portfolio.

### 4.2 Preprocessing Steps

**Step 1: Define State Space**
*   Map NCD levels to integers (0 to N).

**Step 2: Calculate Claim Probabilities**
*   Compute $P(K=0), P(K=1), P(K=2+)$ using Poisson($\lambda$).

### 4.3 Model Specification (Python Example)

Simulating a BMS Markov Chain and calculating the stationary distribution.

```python
import numpy as np
import pandas as pd

# System Definition
# Levels: 0 (Best, 50% discount) to 5 (Worst, 100% surcharge)
# Relativities: [0.5, 0.6, 0.7, 0.8, 1.0, 2.0]
# Rules:
# - 0 Claims: Move down 1 level (min 0)
# - 1 Claim: Move up 2 levels (max 5)
# - 2+ Claims: Move to Level 5

levels = np.arange(6)
relativities = np.array([0.5, 0.6, 0.7, 0.8, 1.0, 2.0])
lambda_freq = 0.15 # Average driver frequency

# Poisson Probabilities
p0 = np.exp(-lambda_freq)
p1 = np.exp(-lambda_freq) * lambda_freq
p2_plus = 1 - p0 - p1

print(f"Prob(0): {p0:.4f}, Prob(1): {p1:.4f}, Prob(2+): {p2_plus:.4f}")

# Construct Transition Matrix P (6x6)
P = np.zeros((6, 6))

for i in levels:
    # 0 Claims: Down 1
    next_0 = max(0, i - 1)
    P[i, next_0] += p0
    
    # 1 Claim: Up 2
    next_1 = min(5, i + 2)
    P[i, next_1] += p1
    
    # 2+ Claims: To 5
    next_2 = 5
    P[i, next_2] += p2_plus

print("\nTransition Matrix P:")
print(np.round(P, 2))

# Calculate Stationary Distribution (pi)
# Solve pi * P = pi  =>  pi * (P - I) = 0
# Add constraint sum(pi) = 1
A = P.T - np.eye(6)
A[-1] = 1 # Replace last equation with sum constraint
b = np.zeros(6)
b[-1] = 1

pi = np.linalg.solve(A, b)

print("\nStationary Distribution (Long-Run % in each class):")
for i, prob in enumerate(pi):
    print(f"Level {i} (Rel {relativities[i]}): {prob:.2%}")

# Average Premium Relativity
avg_relativity = np.sum(pi * relativities)
print(f"\nAverage Portfolio Relativity: {avg_relativity:.4f}")

# Interpretation:
# If the Base Rate is $1000, the average collected premium is $1000 * avg_relativity.
# If avg_relativity < 1.0, we have an "Off-Balance".
# We must divide the Base Rate by avg_relativity to collect the target revenue.
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Stationary Distribution:** Where will the book of business settle in 10 years?
2.  **Off-Balance Factor:** How much to inflate the base rate to pay for the discounts.

**Interpretation:**
*   **Top-Heavy:** If 90% of drivers are in the best class, the system isn't distinguishing risk well.
*   **Volatility:** If the stationary distribution is spread out, the system is actively sorting drivers.

---

## 5. Evaluation & Validation

### 5.1 Simulation Testing

*   Run a Monte Carlo simulation of 10,000 drivers for 20 years.
*   Check if the theoretical stationary distribution matches the simulation.
*   Check the "Churn Rate" (how many customers leave due to malus).

### 5.2 Fairness Metrics

*   **Bayesian Consistency:** Does a driver with true frequency $\lambda=0.10$ pay less than a driver with $\lambda=0.20$ in the long run?
*   **Speed of Convergence:** How many years does it take for a bad driver to reach the high-premium class?

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Ignoring the Off-Balance**
    *   **Issue:** Setting the Base Rate assuming the average relativity is 1.0.
    *   **Reality:** In mature BMS, the average relativity is often 0.60 or lower (everyone has a bonus).
    *   **Result:** Massive underpricing.

2.  **Trap: Hunger for Bonus Impact**
    *   **Issue:** Assuming observed frequency = true frequency.
    *   **Reality:** Observed frequency is *lower* because people hide small claims.
    *   **Fix:** Adjust the Poisson $\lambda$ for the implied deductible effect.

### 6.2 Implementation Challenges

1.  **New Business:**
    *   Where do new customers start? Level 0 (Best)? Level 5 (Worst)? Or a neutral "Entry Level"?
    *   This affects the transition dynamics significantly.

2.  **Lapses:**
    *   Bad drivers often lapse (switch insurers) to try to reset their NCD.
    *   Insurers need data sharing (CUE in UK, CLUE in US) to prevent this gaming.

---

## 7. Advanced Topics & Extensions

### 7.1 Transition Matrices with Lapses

Add a "Lapse" state to the Markov Chain.
*   $P(\text{Lapse} | \text{Malus})$ is high.
*   $P(\text{Lapse} | \text{Bonus})$ is low.
*   This models the "Retention" benefit of the BMS.

### 7.2 Optimal Bonus Scales

Using Dynamic Programming to find the optimal $L$ (retention limit) for the policyholder.
*   Value Function $V(i)$ = Expected future cost given current state $i$.
*   Policyholder minimizes $V(i)$.

---

## 8. Regulatory & Governance Considerations

### 8.1 Transparency

*   Policyholders must clearly understand the rules. "If I claim, my premium goes up by X%."
*   Complex "Black Box" BMS are often discouraged.

### 8.2 Discrimination

*   Is the Malus punitive? Some jurisdictions limit the max surcharge (e.g., cannot exceed 200% of base).

---

## 9. Practical Example

### 9.1 Worked Example: Hunger for Bonus

**Scenario:**
*   Current Class: Level 0 (50% discount, Prem = $500).
*   Next Class if Claim: Level 2 (20% discount, Prem = $800).
*   Next Class if No Claim: Level 0 (50% discount, Prem = $500).
*   Time Horizon: 1 Year (simplified).
*   Discount Rate: 0%.

**Decision:**
*   Accident occurs with damage $C = $200.
*   **Cost to Claim:** Deductible ($0) + Premium Increase ($800 - $500 = $300). Total = $300.
*   **Cost to Pay:** $200.
*   **Result:** Pay the $200 out of pocket. Do not report.

**Implication for Actuary:**
*   The insurer never sees this $200 loss.
*   The "Observed Frequency" is lower than reality.
*   The "Observed Severity" is higher (only big claims are reported).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **BMS** rewards claim-free years.
2.  **Markov Chains** model the system dynamics.
3.  **Stationary Distribution** shows the long-term cost.
4.  **Hunger for Bonus** causes under-reporting of small claims.

### 10.2 When to Use This Knowledge
*   **Product Design:** Creating a new NCD scale.
*   **Pricing:** Calculating the Off-Balance factor.
*   **Reserving:** Adjusting frequency trends for NCD-induced reporting changes.

### 10.3 Critical Success Factors
1.  **Account for Off-Balance:** Crucial for revenue adequacy.
2.  **Monitor Churn:** Ensure the Malus doesn't drive away too many customers.
3.  **Simulate:** Don't rely solely on matrix algebra; simulate the real-world lapse behavior.

### 10.4 Further Reading
*   **Lemaire:** "Bonus-Malus Systems in Automobile Insurance" (The classic text).
*   **ASTIN Bulletin:** Numerous papers on optimal BMS design.

---

## Appendix

### A. Glossary
*   **NCD:** No Claim Discount.
*   **Malus:** Premium surcharge.
*   **Stationary Distribution:** Long-run stable state probabilities.
*   **Off-Balance:** The ratio of Collected Premium to Base Premium.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Stationary Dist** | $\pi P = \pi$ | Long-run analysis |
| **Avg Relativity** | $\sum \pi_i b_i$ | Off-balance calc |
| **Poisson Prob** | $e^{-\lambda} \lambda^k / k!$ | Transition calc |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,000+*
