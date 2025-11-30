# Life Pricing & Reserving Case Study (Part 1) - Theoretical Deep Dive

## Overview
Life Insurance is fundamentally different from Non-Life. The claim is certain (everyone dies); the uncertainty is *when*. This session covers the building blocks of Life Actuarial Science: **Mortality Tables**, **Commutation Functions**, and the calculation of **Net vs. Gross Premiums**.

---

## 1. Conceptual Foundation

### 1.1 The Life Table

*   **$l_x$:** Number of people alive at age $x$.
*   **$d_x$:** Number of deaths between age $x$ and $x+1$. ($d_x = l_x - l_{x+1}$).
*   **$q_x$:** Probability of dying between $x$ and $x+1$. ($q_x = d_x / l_x$).
*   **$p_x$:** Probability of surviving. ($p_x = 1 - q_x$).

### 1.2 Select vs. Ultimate

*   **Select Period:** When you first buy a policy, you are healthy (underwriting). Mortality is low.
*   **Ultimate Period:** After 10-15 years, the "selection effect" wears off. You are just average.
*   **Notation:** $q_{[x]+t}$ is the mortality of a person aged $x+t$ who bought the policy at age $x$.

### 1.3 Products

1.  **Term Life:** Pays if you die within $n$ years. (Pure protection).
2.  **Whole Life:** Pays whenever you die. (Protection + Savings).
3.  **Endowment:** Pays if you die within $n$ years OR if you survive to year $n$.

---

## 2. Mathematical Framework

### 2.1 Present Value of Future Benefits (PVFB)

For a Whole Life policy of \$1 on a person aged $x$:
$$ A_x = \sum_{t=0}^{\infty} v^{t+1} \cdot {}_{t}p_x \cdot q_{x+t} $$
*   $v$: Discount factor ($1/(1+i)$).
*   ${}_{t}p_x$: Probability of surviving $t$ years.
*   $q_{x+t}$: Probability of dying in year $t+1$.

### 2.2 Present Value of Future Premiums (PVFP)

For a Whole Life Annuity Due (Premiums paid at start of year):
$$ \ddot{a}_x = \sum_{t=0}^{\infty} v^t \cdot {}_{t}p_x $$

### 2.3 The Equivalence Principle

**Net Premium ($P$)** is set such that:
$$ PV(\text{Premiums}) = PV(\text{Benefits}) $$
$$ P \cdot \ddot{a}_x = A_x \implies P = \frac{A_x}{\ddot{a}_x} $$

### 2.4 Commutation Functions (The "Old School" Way)

*   $D_x = v^x l_x$
*   $N_x = \sum_{t=0}^{\infty} D_{x+t}$
*   $M_x = \sum_{t=0}^{\infty} v^{x+t+1} d_{x+t}$
*   **Formula:** $P_x = M_x / N_x$.
*   *Why?* Before computers, this made calculation possible. Today, we simulate.

---

## 3. Theoretical Properties

### 3.1 Reserves (Prospective vs. Retrospective)

*   **Prospective Reserve:** $PV(\text{Future Benefits}) - PV(\text{Future Premiums})$.
*   **Retrospective Reserve:** $FV(\text{Past Premiums}) - FV(\text{Past Claims})$.
*   **Theorem:** They are equal (if assumptions match experience).

### 3.2 Gross Premium Loading

$$ GP = \frac{NP + E_{fixed} + E_{claim}}{1 - E_{variable} - \text{Profit}} $$
*   **Expenses:**
    *   Acquisition (High in year 1).
    *   Maintenance (Level).
    *   Claim Settlement.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Mortality Table Lookup (Python)

```python
import pandas as pd
import numpy as np

# Mock Mortality Table (CSO 2001 Male Non-Smoker)
# Age 0 to 120
ages = np.arange(0, 121)
# Gompertz Law for simplicity: q_x = B * c^x
B = 0.00005
c = 1.10
qx = np.minimum(1.0, B * c**ages)
lx = np.zeros(121)
lx[0] = 100000

for i in range(120):
    lx[i+1] = lx[i] * (1 - qx[i])

mortality_table = pd.DataFrame({'Age': ages, 'qx': qx, 'lx': lx})

print(mortality_table.iloc[30:35]) # View Age 30-34
```

### 4.2 Pricing a Term Policy

```python
def price_term_insurance(age, term, interest_rate, sum_assured):
    v = 1 / (1 + interest_rate)
    
    # 1. Calculate PVFB (Benefits)
    pvfb = 0
    for t in range(term):
        # Prob of dying in year t+1
        q_x_t = mortality_table.loc[age+t, 'qx']
        # Prob of surviving to t
        p_x_t = mortality_table.loc[age+t, 'lx'] / mortality_table.loc[age, 'lx']
        
        # Actuarial Present Value
        # Note: p_x_t * q_x_t is roughly prob of dying exactly in year t+1
        # More precise: t_p_x * q_{x+t}
        prob_death = (mortality_table.loc[age+t, 'lx'] - mortality_table.loc[age+t+1, 'lx']) / mortality_table.loc[age, 'lx']
        
        pvfb += sum_assured * v**(t+1) * prob_death
        
    # 2. Calculate PVFP (Annuity)
    pvfp_factor = 0
    for t in range(term):
        prob_survival = mortality_table.loc[age+t, 'lx'] / mortality_table.loc[age, 'lx']
        pvfp_factor += v**t * prob_survival
        
    net_premium = pvfb / pvfp_factor
    return net_premium

premium_30yr_term = price_term_insurance(30, 20, 0.04, 100000)
print(f"\nNet Annual Premium for 20-Year Term ($100k): ${premium_30yr_term:.2f}")
```

---

## 5. Evaluation & Validation

### 5.1 Profit Testing

*   We don't just trust the Net Premium. We build a cash flow model.
*   **Year 1:** Premium - Expense - Death Benefit - Reserve Increase.
*   **IRR:** Calculate the Internal Rate of Return on the capital invested (Acquisition Strain).
*   **Goal:** IRR > Hurdle Rate (e.g., 12%).

### 5.2 Sensitivity Analysis

*   What if interest rates drop to 2%?
*   What if mortality improves by 1% per year? (Longevity Risk).
*   What if lapse rates double? (Lapse Risk).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Lapse-Supported" Product**
    *   **Issue:** Some products (Term to 100) are priced assuming people will quit.
    *   **Risk:** If people *don't* quit (because they are sick), the insurer loses money.
    *   **Rule:** Never rely solely on lapses for profitability.

2.  **Trap: Negative Reserves**
    *   **Issue:** In early years, PV(Premiums) > PV(Benefits).
    *   **Result:** Reserve < 0.
    *   **Regulation:** Most regulators floor the reserve at 0 (or Cash Surrender Value). You cannot book a negative liability (asset).

### 6.2 Implementation Challenges

1.  **Joint Life:**
    *   Pricing for a couple (First-to-Die or Second-to-Die).
    *   **Math:** $p_{xy} = p_x \cdot p_y$ (assuming independence).
    *   *Reality:* Broken Heart Syndrome (dependence).

---

## 7. Advanced Topics & Extensions

### 7.1 Universal Life (UL)

*   Decouples the protection and savings.
*   **Account Value:** $AV_t = (AV_{t-1} + P - \text{Expense} - \text{COI}) \times (1+i)$.
*   **COI:** Cost of Insurance (Mortality Charge).
*   *Flexibility:* Policyholder can skip premiums if AV is sufficient.

### 7.2 Variable Annuities (VA)

*   Investment risk is passed to the policyholder.
*   **Guarantees:** GMDB (Death Benefit), GMWB (Withdrawal Benefit).
*   *Hedging:* Insurers use derivatives to hedge these guarantees.

---

## 8. Regulatory & Governance Considerations

### 8.1 Standard Valuation Law (SVL)

*   Prescribes the Mortality Table (CSO 2017) and Max Interest Rate for statutory reserves.
*   **Principle-Based Reserving (PBR):** The new US standard (VM-20). Allows companies to use their own experience if credible.

---

## 9. Practical Example

### 9.1 Worked Example: The "Select" Effect

**Scenario:**
*   Age 40.
*   **Aggregate Table:** $q_{40} = 0.002$.
*   **Select Table:** $q_{[40]} = 0.001$.
*   **Impact:** If you price using the Aggregate table, you overcharge healthy applicants. A competitor using Select tables will steal all the good risks (Adverse Selection).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Equivalence Principle:** PV(In) = PV(Out).
2.  **Reserves** smooth the cost over time.
3.  **Mortality** is the key driver.

### 10.2 When to Use This Knowledge
*   **Product Development:** Designing a new Term product.
*   **Valuation:** Calculating year-end reserves.

### 10.3 Critical Success Factors
1.  **Interest Rate Assumption:** In long-term business, 1% difference in rate = 20% difference in price.
2.  **Expense Analysis:** Know your actual cost per policy.
3.  **Lapse Assumptions:** Crucial for profitability.

### 10.4 Further Reading
*   **Dickson, Hardy, Waters:** "Actuarial Mathematics for Life Contingent Risks".
*   **Bowers et al.:** "Actuarial Mathematics".

---

## Appendix

### A. Glossary
*   **Annuity Due:** Payments at start of period.
*   **Endowment:** Pays on survival.
*   **Surrender Value:** Cash you get if you quit.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Net Premium** | $A_x / \ddot{a}_x$ | Pricing |
| **Reserve** | $A_{x+t} - P \ddot{a}_{x+t}$ | Valuation |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
