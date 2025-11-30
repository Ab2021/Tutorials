# SOA/CAS-aligned Problem Sets (Part 2) - Life & Annuities - Theoretical Deep Dive

## Overview
"Life insurance math is just compound interest with a probability of death."
While P&C actuaries worry about "Severity", Life actuaries worry about "Mortality" and "Interest Rates".
This day focuses on **Life Contingencies**: The mathematics of when you die and how much it costs.
We will solve classic **Exam FAM-L** problems using Python, replacing the "Commutation Functions" ($D_x, N_x$) of the past with direct vector calculation.

---

## 1. Conceptual Foundation

### 1.1 The Life Table

*   **$l_x$:** Number of lives surviving to age $x$.
*   **$d_x$:** Number of deaths between $x$ and $x+1$.
*   **$q_x$:** Probability of dying between $x$ and $x+1$ ($d_x / l_x$).
*   **$p_x$:** Probability of surviving ($1 - q_x$).

### 1.2 Present Value of Future Benefits (PVFB)

*   **Insurance ($A_x$):** Pays \$1 if you die.
    *   $PV = \sum v^{t+1} \times {}_{t}p_x \times q_{x+t}$
*   **Annuity ($a_x$):** Pays \$1 if you live.
    *   $PV = \sum v^t \times {}_{t}p_x$

---

## 2. Mathematical Framework

### 2.1 Gompertz Law of Mortality

A parametric model for human mortality.
$$ \mu_x = B c^x $$
*   **Interpretation:** Mortality increases exponentially with age.
*   **Python:** `mu = B * c**age`.

### 2.2 The Equivalence Principle

To calculate the Net Premium ($P$):
$$ \text{PV(Premiums)} = \text{PV(Benefits)} $$
$$ P \times \ddot{a}_x = 1 \times A_x $$
$$ P = \frac{A_x}{\ddot{a}_x} $$

---

## 3. Theoretical Properties

### 3.1 Reserves (The "Sinking Fund")

*   **Concept:** In early years, Premium > Cost of Insurance. The excess builds a "Reserve".
*   **Formula:** ${}_tV_x = \text{PV(Future Benefits)} - \text{PV(Future Premiums)}$.
*   **Recursive (Fackler):** $({}_tV_x + P)(1+i) = q_{x+t} + p_{x+t} \times {}_{t+1}V_x$.

### 3.2 Select vs. Ultimate Mortality

*   **Select:** Mortality depends on Age AND Duration (Time since underwriting).
    *   A 40-year-old who just passed a medical exam is healthier than a 40-year-old who bought the policy 10 years ago.
*   **Ultimate:** Mortality depends only on Age (Selection effect wears off).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Constructing a Life Table in Python

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Define Mortality Law (Gompertz)
B = 0.0003
c = 1.07
ages = np.arange(0, 121)

# Force of Mortality
mu = B * c**ages
# Approx q_x = 1 - exp(-mu)
qx = 1 - np.exp(-mu)
qx[-1] = 1.0 # Cap at age 120

# 2. Build Table
lx = np.zeros_like(ages, dtype=float)
dx = np.zeros_like(ages, dtype=float)
lx[0] = 100000

for i in range(len(ages)-1):
    dx[i] = lx[i] * qx[i]
    lx[i+1] = lx[i] - dx[i]

df = pd.DataFrame({'Age': ages, 'qx': qx, 'lx': lx, 'dx': dx})
print(df.head())
print(df.tail())
```

### 4.2 Pricing Whole Life Insurance ($A_x$)

**Problem:** Calculate the Net Single Premium for a Whole Life policy for a 30-year-old. Interest rate $i=5\%$.

```python
age_issue = 30
interest_rate = 0.05
v = 1 / (1 + interest_rate)

# Subset table from age 30 onwards
sub_df = df[df['Age'] >= age_issue].copy()
sub_df['t'] = sub_df['Age'] - age_issue

# Probability of dying in year t (deferred t years)
# t|q_x = (lx[x+t] - lx[x+t+1]) / lx[x]
sub_df['deferred_death_prob'] = sub_df['dx'] / df.loc[df['Age']==age_issue, 'lx'].values[0]

# Present Value Factor (Paid at END of year)
sub_df['discount'] = v ** (sub_df['t'] + 1)

# Ax Calculation
Ax = np.sum(sub_df['deferred_death_prob'] * sub_df['discount'])
print(f"Net Single Premium (Ax) per $1 benefit: {Ax:.4f}")
```

### 4.3 Pricing a Life Annuity ($\ddot{a}_x$)

**Problem:** Calculate the PV of a \$1 annuity due (paid at start of year) for a 30-year-old.

```python
# Probability of surviving to start of year t
# t_p_x = lx[x+t] / lx[x]
sub_df['survival_prob'] = sub_df['lx'] / df.loc[df['Age']==age_issue, 'lx'].values[0]

# Present Value Factor (Paid at START of year)
sub_df['discount_annuity'] = v ** sub_df['t']

# ax Calculation
ax_due = np.sum(sub_df['survival_prob'] * sub_df['discount_annuity'])
print(f"Annuity Factor (ax_due): {ax_due:.4f}")

# Net Level Premium
P = Ax / ax_due
print(f"Annual Premium per $1 benefit: {P:.4f}")
```

---

## 5. Evaluation & Validation

### 5.1 The Woolhouse Approximation

*   **Context:** Annuities paid monthly ($m=12$).
*   **Formula:** $\ddot{a}_x^{(m)} \approx \ddot{a}_x - \frac{m-1}{2m}$.
*   **Python Check:** Calculate the monthly annuity exactly (vector of 1200 months) and compare to Woolhouse.

### 5.2 Reserve Testing

*   **Check:** ${}_{120-x}V_x$ must equal 1.0 (Endowment at age 120).
*   **Logic:** If you live to the end of the table, the policy pays out. The reserve must equal the benefit.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Timing of Cash Flows

*   **Insurance:** Usually paid at *end* of year of death.
*   **Annuity:** Usually paid at *start* of year (Annuity Due) or *end* (Annuity Immediate).
*   **Python:** Be careful with `v ** t` vs `v ** (t+1)`.

### 6.2 Interest Rate Volatility

*   **Exam:** $i$ is constant (5%).
*   **Reality:** $i$ is a Yield Curve.
*   **Fix:** Use a vector of discount factors `v_t` derived from the Spot Curve.

---

## 7. Advanced Topics & Extensions

### 7.1 Universal Life (UL)

*   **Structure:** Flexible premium. Account Value mechanics.
*   **Python:** Requires a month-by-month simulation loop (Cost of Insurance charges, Interest Crediting, Expense loads).

### 7.2 Variable Annuities (VA) with Guarantees (GMxB)

*   **Complexity:** The benefit depends on the S&P 500.
*   **Method:** Stochastic Simulation (Monte Carlo).
    *   Simulate 10,000 stock market paths.
    *   Calculate payout for each.
    *   Take the average PV.

---

## 8. Regulatory & Governance Considerations

### 8.1 Standard Valuation Law

*   **Requirement:** You must use the CSO 2017 Mortality Table for statutory reserves.
*   **Python:** Load the official table from the SOA website (XML/Excel format).

---

## 9. Practical Example

### 9.1 The "Pension Buyout"

**Scenario:** A company wants to offload its pension plan to an insurer.
**Data:** List of 5,000 retirees (Age, Gender, Benefit Amount).
**Task:** Calculate the lump sum price.
**Python:**
1.  Load Mortality Table (IAM 2012).
2.  Vectorized calculation of $\ddot{a}_x$ for each retiree.
3.  Sum(Benefit $\times \ddot{a}_x$).
4.  Add profit margin.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Life Table** drives everything.
2.  **Equivalence Principle** balances Inflows and Outflows.
3.  **Reserves** smooth the cost over time.

### 10.2 When to Use This Knowledge
*   **Life/Health/Pension:** Daily work.
*   **P&C:** Workers Comp (Lifetime Pension claims).

### 10.3 Critical Success Factors
1.  **Precision:** Life insurance contracts last 50 years. Small errors compound.
2.  **Scenarios:** Always test sensitivity to Mortality Improvement.

### 10.4 Further Reading
*   **Dickson, Hardy, Waters:** "Actuarial Mathematics for Life Contingent Risks".
*   **Python Package:** `lifelib` (Open source actuarial library).

---

## Appendix

### A. Glossary
*   **CSO:** Commissioners Standard Ordinary (Mortality Table).
*   **NSP:** Net Single Premium.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Premium** | $P = A_x / \ddot{a}_x$ | Pricing |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
